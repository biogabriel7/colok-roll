# Changes Summary: Architectural Refactor

## Date: 2025-10-07

### Overview
Restructured the `colokroll` package to remove the phase-based architecture, introduce dedicated packages for configuration, IO, preprocessing, analysis, and visualization, and provide compatibility shims for existing consumers.

---

## Highlights

- Added `colokroll/config/` with renamed dataclasses (`ImageIOConfig`, `ProjectionConfig`, etc.) and runtime configuration utilities.
- Migrated image loading/MIP creation to a new `colokroll/io/` package.
- Removed the phase-based API; `colokroll/__init__` now exposes the consolidated toolkit surface.
- Visualization, preprocessing, and analysis modules live in dedicated packages.

