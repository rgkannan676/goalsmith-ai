"""The agent loop: a hand-built LangGraph graph with human approval.

    START -> agent --(safe tool calls)--> tools -> agent -> ... -> END
                   \\--(risky tool calls)--> approval -> tools
                   \\--(final answer or budget hit)--> END

- `agent` calls the model, unless the step budget is used up.
- `approval` pauses the run (`interrupt()`) until a human approves, edits or
  rejects each risky call. The paused state lives in the checkpointer, so the run
  can be resumed later, even by a different process.
- `tools` runs every tool call that has not been answered yet, through the registry.

Observability: each `run()` / `resume()` is an "agent" span. Model calls (traced by
the model's callback, see observability/tracing.py) and tool calls nest inside it.
All spans of one run share the session id = run id, so Langfuse groups a run that
was paused and resumed. Tokens and cost add up in the run state.
"""

from collections.abc import Callable, Coroutine, Mapping
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from goalsmith.config import ModelPrice
from goalsmith.core.serde import checkpoint_serializer
from goalsmith.core.state import RunResult, RunState
from goalsmith.harness.budget import Budget, BudgetUsage
from goalsmith.harness.permissions import (
    ApprovalRequest,
    Decision,
    Permissions,
    apply_decisions,
    approval_requests,
    interrupt_payload,
    missing_decisions,
    parse_decisions,
    pending_requests,
)
from goalsmith.harness.tools import ToolContext, ToolRegistry
from goalsmith.observability.logs import bind_run, get_logger
from goalsmith.observability.spans import (
    current_trace_id,
    event,
    mark_warning,
    observation,
    set_metadata,
    set_output,
    trace_attributes,
)
from goalsmith.observability.usage import usage_from_message
from goalsmith.prompts import Prompt, load_prompt

log = get_logger(__name__)

OnUpdate = Callable[[str, dict[str, Any]], None]
"""Called after each node with (node name, the partial update it returned)."""


class RunStateError(RuntimeError):
    """The run is not in a state that allows this (e.g. resuming a run that is not paused)."""


class AgentLoop:
    def __init__(
        self,
        model: BaseChatModel,
        registry: ToolRegistry,
        context: ToolContext,
        *,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        system_prompt: Prompt | None = None,
        prices: Mapping[str, ModelPrice] | None = None,
    ) -> None:
        # bind_tools tells the model which tools exist and what arguments they take.
        self._model = model.bind_tools(registry.schemas())
        self._registry = registry
        self._context = context
        self._budget = budget or Budget()
        self._permissions = permissions or Permissions()
        # A versioned prompt (prompts/agent_system.md); its version goes on every trace.
        self._prompt = system_prompt or load_prompt("agent_system")
        self._prices = prices or {}
        # A checkpointer saves the state after every node. Pausing for approval needs
        # one; the in-memory saver lasts only as long as this process.
        self.graph = self._build_graph(checkpointer or InMemorySaver(serde=checkpoint_serializer()))

    def _build_graph(
        self, checkpointer: BaseCheckpointSaver[Any]
    ) -> CompiledStateGraph[RunState, None, RunState, RunState]:
        graph = StateGraph(RunState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("approval", self._approval_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._route_after_agent, ["approval", "tools", END])
        graph.add_edge("approval", "tools")
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=checkpointer)

    # --- nodes: each takes the state and returns a partial update --------------

    async def _agent_node(self, state: RunState) -> dict[str, Any]:
        usage = state["budget"]
        if self._budget.exhausted_by(usage):
            return {"status": "budget_exceeded", "answer": self._partial_answer(state)}

        response = await self._model.ainvoke([SystemMessage(self._prompt.text), *state["messages"]])
        tokens = usage_from_message(response, self._prices)
        update: dict[str, Any] = {"messages": [response], "budget": usage.add_step(tokens)}
        if not response.tool_calls:
            update |= {"status": "completed", "answer": response.text}
        return update

    def _approval_node(self, state: RunState) -> dict[str, Any]:
        message = _last_ai_message(state["messages"])
        requests = approval_requests(message.tool_calls, self._registry, self._permissions)
        # The run stops here and its state is checkpointed. When resumed with
        # Command(resume=decisions), this node runs again from the top and
        # `interrupt()` returns the decisions. Nothing with side effects happens
        # before this line, so re-running the node is safe.
        decisions = parse_decisions(interrupt(interrupt_payload(requests)))
        edited, rejections = apply_decisions(message, decisions)
        return {"messages": [edited, *rejections]}

    async def _tools_node(self, state: RunState) -> dict[str, Any]:
        message = _last_ai_message(state["messages"])
        answered = _answered_call_ids(state["messages"])
        calls = [call for call in message.tool_calls if call.get("id") not in answered]
        results = await self._registry.execute_all(calls, self._context)
        return {"messages": results}

    # --- edge ------------------------------------------------------------------

    def _route_after_agent(self, state: RunState) -> str:
        # Still "running" after the agent node means the model asked for tools.
        if state["status"] != "running":
            return END
        message = _last_ai_message(state["messages"])
        if approval_requests(message.tool_calls, self._registry, self._permissions):
            return "approval"
        return "tools"

    # --- running ---------------------------------------------------------------

    async def run(self, goal: str, on_update: OnUpdate | None = None) -> RunResult:
        """Start a new run. Returns when it finishes or pauses for approval."""
        config = self._config()
        if (await self.graph.aget_state(config)).values:
            raise RunStateError(f"Run {self._context.run_id!r} already exists. Use resume.")
        initial: RunState = {
            "goal": goal,
            "messages": [HumanMessage(goal)],
            "budget": BudgetUsage(),
            "status": "running",
            "answer": None,
        }
        return await self._traced("agent.run", goal, self._stream(initial, config, on_update))

    async def resume(
        self, decisions: Mapping[str, Decision], on_update: OnUpdate | None = None
    ) -> RunResult:
        """Continue a paused run with a decision for every pending tool call."""
        pending = await self.pending()
        if not pending:
            raise RunStateError(f"Run {self._context.run_id!r} is not waiting for approval.")
        missing = missing_decisions(pending, decisions)
        if missing:
            raise ValueError(f"No decision for tool call(s): {', '.join(missing)}")
        resume_value = {call_id: d.model_dump(mode="json") for call_id, d in decisions.items()}
        return await self._traced(
            "agent.resume", resume_value, self._continue(resume_value, on_update)
        )

    async def _continue(
        self, resume_value: dict[str, Any], on_update: OnUpdate | None
    ) -> RunResult:
        event("approval.decided", input=resume_value)
        for call_id, decision in resume_value.items():
            log.info("approval.decided", call_id=call_id, action=decision["action"])
        return await self._stream(Command(resume=resume_value), self._config(), on_update)

    async def messages(self) -> list[AnyMessage]:
        """The run's full message history (goal, model replies, tool results)."""
        snapshot = await self.graph.aget_state(self._config())
        return list(snapshot.values.get("messages", []))

    async def pending(self) -> list[ApprovalRequest]:
        """The approval requests this run is paused on (empty if it is not paused)."""
        snapshot = await self.graph.aget_state(self._config())
        return [r for i in snapshot.interrupts for r in pending_requests(i.value)]

    async def _stream(
        self,
        graph_input: RunState | Command[Any],
        config: RunnableConfig,
        on_update: OnUpdate | None,
    ) -> RunResult:
        async for chunk in self.graph.astream(graph_input, config, stream_mode="updates"):
            # Chunks are {node_name: partial_update}. LangGraph also reports a pause
            # as {"__interrupt__": ...}; that is not a node, so it is skipped.
            if on_update is None:
                continue
            for node, update in cast(dict[str, Any], chunk).items():
                if not node.startswith("__") and isinstance(update, dict):
                    on_update(node, update)
        return await self._result(config)

    async def _result(self, config: RunnableConfig) -> RunResult:
        # The checkpointer holds the final (or paused) state, so read it from there.
        snapshot = await self.graph.aget_state(config)
        state = cast(RunState, snapshot.values)
        pending = [r for i in snapshot.interrupts for r in pending_requests(i.value)]
        return RunResult(
            run_id=self._context.run_id,
            status="awaiting_approval" if pending else state["status"],
            answer=state["answer"] or "",
            steps=state["budget"].steps,
            pending=pending,
            usage=state["budget"].tokens,
            trace_id=current_trace_id(),
        )

    async def _traced(
        self, name: str, input: Any, work: Coroutine[Any, Any, RunResult]
    ) -> RunResult:
        """Run `work` inside an "agent" span, and record how it ended.

        This is outside the graph nodes, so none of it repeats when a run resumes.
        """
        run_id = self._context.run_id
        with (
            bind_run(run_id),
            trace_attributes(name=name, session_id=run_id, metadata=self._prompt.trace_metadata()),
            observation(name, "agent", input=input, metadata={"prompt": self._prompt.ref}) as span,
        ):
            result = await work
            set_output(span, result.answer or [r.model_dump() for r in result.pending])
            fields = {
                "status": result.status,
                "steps": result.steps,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cost_usd": round(result.usage.cost_usd, 6),
            }
            set_metadata(span, **fields)
            if result.status == "awaiting_approval":
                event("approval.requested", input=[r.model_dump() for r in result.pending])
                for request in result.pending:
                    log.info("approval.requested", tool=request.tool, call_id=request.call_id)
            elif result.status == "budget_exceeded":
                mark_warning(span, "Stopped at the step budget")
            paused = result.status == "awaiting_approval"
            log.info("run.paused" if paused else "run.finished", **fields)
            return result

    def _config(self) -> RunnableConfig:
        # thread_id is the checkpointer's key for this run's saved state.
        # A step is up to three graph hops (agent, approval, tools). This LangGraph
        # limit is a backstop; our own budget should always stop the run first.
        # metadata reaches every model call's callbacks, so model spans show the prompt.
        return {
            "configurable": {"thread_id": self._context.run_id},
            "recursion_limit": 3 * self._budget.max_steps + 5,
            "metadata": self._prompt.trace_metadata(),
        }

    def _partial_answer(self, state: RunState) -> str:
        notice = f"Stopped: reached the step budget ({self._budget.max_steps} model calls)."
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage) and message.text.strip():
                return f"{notice}\n\nPartial result:\n\n{message.text}"
        return f"{notice} No partial answer was produced."


def _last_ai_message(messages: list[AnyMessage]) -> AIMessage:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    raise RuntimeError("expected a model message to act on")


def _answered_call_ids(messages: list[AnyMessage]) -> set[str]:
    """Ids of the last model message's tool calls that already have a ToolMessage
    (rejections added by the approval node)."""
    answered: set[str] = set()
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            break
        if isinstance(message, ToolMessage):
            answered.add(message.tool_call_id)
    return answered
