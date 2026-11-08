# ADR 0001: Record architecture decisions

- Status: accepted
- Date: 2026-11-08

## Context
Goalsmith is a learning and portfolio project. The reasoning behind choices matters as much as the code, both for future changes and for explaining the design in interviews.

## Options
1. Keep decisions in commit messages and the README.
2. Keep one short Architecture Decision Record (ADR) per real decision in `docs/adr/`.

## Decision
Option 2. Each ADR is one page: context, options, decision, consequences. Files are numbered `NNNN-short-title.md`.

## Consequences
- Every milestone that makes a real decision adds an ADR.
- A superseded ADR is kept and marked "superseded by NNNN".
