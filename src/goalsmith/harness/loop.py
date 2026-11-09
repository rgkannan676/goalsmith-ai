"""The agent loop: a hand-built LangGraph graph.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                   \\--(final answer or budget hit)--> END

- `agent` calls the model, unless the step budget is used up.
- `tools` runs every tool call from the model's last message, through the registry.
- The conditional edge after `agent` decides whether to loop or stop.
"""

from collections.abc import Callable
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from goalsmith.core.state import RunResult, RunState
from goalsmith.harness.budget import Budget, BudgetUsage
from goalsmith.harness.tools import ToolContext, ToolRegistry

# Moves to versioned templates in prompts/ in M5, when prompt versions are traced.
DEFAULT_SYSTEM_PROMPT = """\
You are Goalsmith, an agent that completes the user's goal using the tools provided.
- Think briefly about what is needed, then act. Use tools instead of guessing facts.
- Files you write go into the run workspace. Use relative paths.
- When you rely on a web result, cite its URL.
- When you rely on a document passage, cite its source id in square brackets, e.g. [handbook.md#3].
- When the goal is done, reply with the final answer and no tool calls."""

OnUpdate = Callable[[str, dict[str, Any]], None]
"""Called after each node with (node name, the partial update it returned)."""


class AgentLoop:
    def __init__(
        self,
        model: BaseChatModel,
        registry: ToolRegistry,
        context: ToolContext,
        *,
        budget: Budget | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        # bind_tools tells the model which tools exist and what arguments they take.
        self._model = model.bind_tools(registry.schemas())
        self._registry = registry
        self._context = context
        self._budget = budget or Budget()
        self._system_prompt = system_prompt
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph[RunState, None, RunState, RunState]:
        graph = StateGraph(RunState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._route_after_agent, ["tools", END])
        graph.add_edge("tools", "agent")
        return graph.compile()

    # --- nodes: each takes the state and returns a partial update --------------

    async def _agent_node(self, state: RunState) -> dict[str, Any]:
        usage = state["budget"]
        if self._budget.exhausted_by(usage):
            return {"status": "budget_exceeded", "answer": self._partial_answer(state)}

        response = await self._model.ainvoke(
            [SystemMessage(self._system_prompt), *state["messages"]]
        )
        update: dict[str, Any] = {"messages": [response], "budget": usage.add_step()}
        if not response.tool_calls:
            update |= {"status": "completed", "answer": response.text}
        return update

    async def _tools_node(self, state: RunState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            raise RuntimeError("tools node reached without a model message to act on")
        results = await self._registry.execute_all(last.tool_calls, self._context)
        return {"messages": results}

    # --- edge ------------------------------------------------------------------

    @staticmethod
    def _route_after_agent(state: RunState) -> str:
        # Still "running" after the agent node means the model asked for tools.
        return "tools" if state["status"] == "running" else END

    # --- running ---------------------------------------------------------------

    async def run(self, goal: str, on_update: OnUpdate | None = None) -> RunResult:
        initial: RunState = {
            "goal": goal,
            "messages": [HumanMessage(goal)],
            "budget": BudgetUsage(),
            "status": "running",
            "answer": None,
        }
        # Each step is two graph hops (agent + tools). This LangGraph limit is a
        # backstop; our own budget should always stop the run first.
        config: RunnableConfig = {"recursion_limit": 2 * self._budget.max_steps + 5}

        final = initial
        async for mode, chunk in self.graph.astream(
            initial, config, stream_mode=["updates", "values"]
        ):
            # With several stream modes, LangGraph yields (mode, chunk) pairs.
            # "updates": {node_name: partial_update}. "values": the full state.
            if mode == "updates" and on_update is not None:
                updates = cast(dict[str, dict[str, Any]], chunk)
                for node, update in updates.items():
                    on_update(node, update)
            elif mode == "values":
                final = cast(RunState, chunk)

        return RunResult(
            run_id=self._context.run_id,
            status=final["status"],
            answer=final["answer"] or "",
            steps=final["budget"].steps,
        )

    def _partial_answer(self, state: RunState) -> str:
        notice = f"Stopped: reached the step budget ({self._budget.max_steps} model calls)."
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage) and message.text.strip():
                return f"{notice}\n\nPartial result:\n\n{message.text}"
        return f"{notice} No partial answer was produced."
