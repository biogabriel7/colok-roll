"""Interactive visualization helpers (placeholder for future expansion)."""

import warnings


def not_implemented(*args, **kwargs):  # pragma: no cover - placeholder
    warnings.warn(
        "Interactive UI components are not yet implemented.",
        UserWarning,
        stacklevel=2,
    )


__all__ = ["not_implemented"]


