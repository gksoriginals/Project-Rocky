---
name: code-refactoring
description: Use when refactoring existing code safely, especially when preserving behavior, reducing duplication, or removing dead compatibility paths.
---

# Code Refactoring

Use this skill when changing existing code should be incremental, test-backed, and behavior-preserving.

## Core guidance

- Prefer small, reversible changes.
- Add or update characterization tests before changing behavior in uncertain areas.
- Preserve public APIs, persisted formats, and user-visible behavior unless a breaking change is explicitly requested.
- Remove dead compatibility code only after confirming it is no longer needed.
- Run the smallest relevant tests after each increment.

## Good fits

- Dead-code removal
- Backward-compatibility cleanup
- Module or function simplification
- Testable extraction of tangled logic
- Safe branching during a staged migration

## Practical checks

- Identify the hotspot before editing.
- Keep the blast radius narrow.
- Use adapters or seams when the old and new code must coexist temporarily.
- Treat performance-sensitive paths as behavior-sensitive.

