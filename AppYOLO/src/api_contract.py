"""Small, dependency-free helpers shared by the HTTP boundary and tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v"})


def validate_upload_filename(
    filename: Optional[str],
    allowed_suffixes: Iterable[str],
    kind: str,
) -> str:
    """Validate an upload name without allowing a client-controlled path."""

    safe_name = Path(filename or "").name
    suffix = Path(safe_name).suffix.lower()
    if not safe_name or suffix not in {str(item).lower() for item in allowed_suffixes}:
        supported = ", ".join(sorted(str(item) for item in allowed_suffixes))
        raise ValueError(f"Unsupported {kind} format. Expected one of: {supported}")
    return safe_name


def validate_upload_size(size_bytes: int, max_bytes: int, kind: str) -> None:
    """Reject an upload as soon as the streamed byte count crosses the limit."""

    if size_bytes > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(f"{kind.capitalize()} upload exceeds the {limit_mb:.0f} MB limit")


def _finite_temperature(value: Any, *, maximum: float = 200.0) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < -50.0 or parsed > maximum:
        return None
    return round(parsed, 2)


def _finite_rgb_estimate(value: Any) -> Optional[float]:
    """RGB estimates are not sensor inputs and may represent a hot flame."""

    return _finite_temperature(value, maximum=5000.0)


def build_temperature_observation(
    *,
    sensor_temperature_celsius: Any,
    rgb_temperature_celsius: Any,
    host_temperature_celsius: Any = None,
    host_temperature_source: str = "unavailable",
    ambient_temperature_celsius: float = 25.0,
) -> dict[str, Any]:
    """Return one explicit temperature contract for decisions and telemetry.

    A calibrated thermal sensor is authoritative for the scene. An RGB-derived
    value is retained as a useful estimate, but is deliberately marked as
    uncalibrated. Host/CPU temperature is diagnostic metadata and never becomes
    the scene temperature used by the safety decision engine.
    """

    sensor_value = _finite_temperature(sensor_temperature_celsius)
    rgb_value = _finite_rgb_estimate(rgb_temperature_celsius)
    host_value = _finite_temperature(host_temperature_celsius)
    ambient_value = _finite_temperature(ambient_temperature_celsius) or 25.0

    if sensor_value is not None:
        scene_value = sensor_value
        scene_source = "thermal_sensor"
        calibrated = True
    elif rgb_value is not None:
        scene_value = rgb_value
        scene_source = "rgb_estimate"
        calibrated = False
    else:
        scene_value = ambient_value
        scene_source = "ambient_fallback"
        calibrated = False

    return {
        "scene_temperature_celsius": scene_value,
        "scene_temperature_source": scene_source,
        "scene_temperature_calibrated": calibrated,
        "rgb_estimate_temperature_celsius": rgb_value,
        "host_temperature_celsius": host_value,
        "host_temperature_source": host_temperature_source or "unavailable",
    }


def format_sse(payload: dict[str, Any]) -> str:
    """Serialize one Server-Sent Events message with actual line separators."""

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def parse_optional_temperature(value: Any) -> Optional[float]:
    """Parse a form/query temperature while preserving an omitted value."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parsed = _finite_temperature(value)
    if parsed is None:
        raise ValueError("Temperature must be a finite number between -50 and 200 °C")
    return parsed
