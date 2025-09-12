"""Compatibility shim for `advanced_statistics`.

This module redirects imports like `import advanced_statistics.technical_indicators`
to the real package located at
`realtime_anomaly_project.advanced_statistics` while also providing
TYPE_CHECKING-friendly names so editors/linters can resolve symbols.

This is intentionally minimal and only exists to avoid changing many
tests that import `advanced_statistics` at top-level.
"""
from __future__ import annotations

from importlib import import_module
import sys
from typing import TYPE_CHECKING

# Runtime behaviour: import the real package and place it in sys.modules
# under this top-level name so runtime imports work unchanged.
_real = import_module("realtime_anomaly_project.advanced_statistics")
sys.modules[__name__] = _real

# For editors and static analyzers, re-export the common subpackages
# via TYPE_CHECKING imports so `advanced_statistics.technical_indicators`
# can be resolved by tools that examine the source.
if TYPE_CHECKING:
    # these imports are only for type checkers / linters
    from realtime_anomaly_project.advanced_statistics import (
        control_chart, frequency_complexity, model_residuals,
        rolling_stats, seasonality_autocorr, technical_indicators,
    )

__all__ = [
    "control_chart",
    "frequency_complexity",
    "model_residuals",
    "rolling_stats",
    "seasonality_autocorr",
    "technical_indicators",
]
