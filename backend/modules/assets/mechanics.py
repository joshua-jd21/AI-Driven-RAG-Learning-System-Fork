"""Code-generation helpers for physics asset mobjects.

Each public function returns a multi-line Python code string (un-indented)
suitable for insertion into a Manim GeneratedScene.construct() body.
The caller must add 8-space indentation via `_indent_block()`.
"""
from __future__ import annotations

import math
from typing import Any

ASSET_IDS: dict[str, str] = {
    "block": "A rectangular crate/box",
    "hockey_puck": "A flat hockey puck (ellipse profile)",
    "car": "A simple side-view car",
    "inclined_plane": "A triangular inclined ramp",
    "ground": "A horizontal ground/surface",
    "wall": "A vertical wall/barrier",
    "arrow_force": "A labeled force vector arrow",
    "velocity_indicator": "A velocity arrow indicator",
    "acceleration_trail": "Motion trail dots showing acceleration",
}

# These are the named colors and palette values already used by the mechanics
# templates.  Keeping this contract here lets planning and code generation
# validate the same values without importing Manim into the planner.
SUPPORTED_COLOR_VALUES = (
    "red", "blue", "green", "yellow", "orange", "pink", "white", "black",
    "gray", "grey", "#f7c948", "#41d4a8", "#ff7a59", "#4fc3f7",
    "#e0e6f0", "#c8d3e6", "#909090", "#8b6914", "#c8d8f0", "#a0b8e0",
    "#6b8cba",
)

ASSET_PARAM_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "block": {
        "label": {"type": "string"},
        "color": {"type": "string", "enum": list(SUPPORTED_COLOR_VALUES)},
        "width": {"type": "number", "minimum": 0.1},
        "height": {"type": "number", "minimum": 0.1},
    },
    "hockey_puck": {
        "label": {"type": "string"},
    },
    "car": {
        "label": {"type": "string"},
        "color": {"type": "string", "enum": list(SUPPORTED_COLOR_VALUES)},
    },
    "inclined_plane": {
        "angle": {"type": "number", "minimum": 0.0, "maximum": 89.0},
        "width": {"type": "number", "minimum": 0.1},
    },
    "ground": {
        "texture": {"type": "string", "enum": ["grass", "ice", "rough", "ground"]},
        "extent": {"type": "number", "minimum": 0.1},
    },
    "wall": {
        "side": {"type": "string", "enum": ["left", "right"]},
    },
    "arrow_force": {
        "label": {"type": "string"},
        "direction": {"type": "string", "enum": ["LEFT", "RIGHT", "UP", "DOWN", "left", "right", "up", "down"]},
        "color": {"type": "string", "enum": list(SUPPORTED_COLOR_VALUES)},
        "length": {"type": "number", "minimum": 0.1},
    },
    "velocity_indicator": {
        "color": {"type": "string", "enum": list(SUPPORTED_COLOR_VALUES)},
        "magnitude": {"type": "number", "minimum": 0.1},
    },
    "acceleration_trail": {},
}


def validate_params(asset_id: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Validate planner parameters before an asset reaches code generation."""
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(
            f"Invalid params for asset '{asset_id}': expected an object, got {type(params).__name__}"
        )
    schema = ASSET_PARAM_SCHEMAS.get(asset_id)
    if schema is None:
        raise ValueError(f"Unknown asset_id '{asset_id}'")

    for name, value in params.items():
        rule = schema.get(name)
        if rule is None:
            raise ValueError(
                f"Invalid parameter '{name}' for asset '{asset_id}': unsupported parameter"
            )
        expected_type = rule["type"]
        if expected_type == "number":
            if isinstance(value, bool):
                valid_number = False
            else:
                try:
                    valid_number = math.isfinite(float(value))
                except (TypeError, ValueError):
                    valid_number = False
            minimum = rule.get("minimum")
            maximum = rule.get("maximum")
            if valid_number and minimum is not None:
                valid_number = float(value) >= minimum
            if valid_number and maximum is not None:
                valid_number = float(value) <= maximum
            if not valid_number:
                bounds = ""
                if minimum is not None:
                    bounds += f", minimum {minimum}"
                if maximum is not None:
                    bounds += f", maximum {maximum}"
                raise ValueError(
                    f"Invalid parameter '{name}' for asset '{asset_id}': {value!r}; "
                    f"expected a finite numeric value{bounds}"
                )
        elif expected_type == "string" and not isinstance(value, str):
            raise ValueError(
                f"Invalid parameter '{name}' for asset '{asset_id}': {value!r}; "
                "expected a string"
            )
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(
                f"Invalid parameter '{name}' for asset '{asset_id}': {value!r}; "
                f"expected one of {rule['enum']}"
            )
    return dict(params)


def get_code(asset_id: str, instance_id: str, params: dict[str, Any]) -> str:
    """Return un-indented Python code that creates the Manim VGroup `instance_id`."""
    builders: dict[str, Any] = {
        "block": _block,
        "hockey_puck": _hockey_puck,
        "car": _car,
        "inclined_plane": _inclined_plane,
        "ground": _ground,
        "wall": _wall,
        "arrow_force": _arrow_force,
        "velocity_indicator": _velocity_indicator,
        "acceleration_trail": _acceleration_trail,
    }
    fn = builders.get(asset_id)
    if fn is None:
        raise ValueError(f"Unknown asset_id '{asset_id}'. Valid: {sorted(builders)}")
    return fn(instance_id, validate_params(asset_id, params))


def get_position_hint(asset_id: str) -> str:
    hints = {
        "block": "center-left, on top of ground",
        "hockey_puck": "center or left, on top of ground/ice",
        "car": "left side, on top of ground",
        "inclined_plane": "center-right, bottom-left corner at origin",
        "ground": "bottom of scene (DOWN*2.0)",
        "wall": "right or left edge",
        "arrow_force": "next_to target object, aligned with direction",
        "velocity_indicator": "above target object",
        "acceleration_trail": "behind target object",
    }
    return hints.get(asset_id, "center of scene")


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _numeric_param(asset_id: str, params: dict, name: str, default: float) -> float:
    """Validate numeric asset parameters before they enter code generation."""
    value = params.get(name, default)
    if isinstance(value, bool):
        raise ValueError(
            f"Invalid parameter '{name}' for asset '{asset_id}': {value!r}; "
            "expected a finite numeric value"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid parameter '{name}' for asset '{asset_id}': {value!r}; "
            "expected a finite numeric value"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Invalid parameter '{name}' for asset '{asset_id}': {value!r}; "
            "expected a finite numeric value"
        )
    return parsed


def _block(v: str, p: dict) -> str:
    label = p.get("label", "")
    color = p.get("color", "#f7c948")
    w = _numeric_param("block", p, "width", 1.4)
    h = _numeric_param("block", p, "height", 0.9)
    lines = [
        f'{v}_rect = Rectangle(width={w}, height={h}, color="{color}", fill_opacity=0.88, stroke_width=2)',
    ]
    if label:
        lines += [
            f'{v}_label = Text("{label}", font_size=18, color=WHITE)',
            f'{v} = VGroup({v}_rect)',
            f'# center label on block',
            f'{v}_label.move_to({v}_rect.get_center())',
            f'{v}.add({v}_label)',
        ]
    else:
        lines += [f'{v} = VGroup({v}_rect)']
    return "\n".join(lines)


def _hockey_puck(v: str, p: dict) -> str:
    label = p.get("label", "")
    lines = [
        f'{v} = VGroup(',
        f'    Ellipse(width=0.9, height=0.35, color="#c8d8f0", fill_opacity=0.95, stroke_width=2),',
        f'    Ellipse(width=0.72, height=0.24, color="#a0b8e0", fill_opacity=0.55, stroke_width=0),',
        f')',
    ]
    if label:
        lines += [
            f'{v}_text = Text("{label}", font_size=14, color=WHITE).move_to({v}.get_center())',
            f'{v}.add({v}_text)',
        ]
    return "\n".join(lines)


def _car(v: str, p: dict) -> str:
    color = p.get("color", "#41d4a8")
    label = p.get("label", "")
    lines = [
        f'{v}_body = Rectangle(width=2.0, height=0.55, color="{color}", fill_opacity=0.88, stroke_width=2)',
        f'{v}_roof = Rectangle(width=1.1, height=0.42, color="{color}", fill_opacity=0.65, stroke_width=1)',
        f'{v}_roof.next_to({v}_body, UP, buff=0)',
        f'{v}_roof.shift(RIGHT*0.1)',
        f'{v}_wL = Circle(radius=0.22, color=WHITE, fill_opacity=1.0, stroke_width=2)',
        f'{v}_wR = Circle(radius=0.22, color=WHITE, fill_opacity=1.0, stroke_width=2)',
        f'{v}_wL.move_to({v}_body.get_center() + LEFT*0.6 + DOWN*0.38)',
        f'{v}_wR.move_to({v}_body.get_center() + RIGHT*0.6 + DOWN*0.38)',
        f'{v} = VGroup({v}_body, {v}_roof, {v}_wL, {v}_wR)',
    ]
    if label:
        lines += [
            f'{v}_lbl = Text("{label}", font_size=16, color=WHITE).move_to({v}_body.get_center())',
            f'{v}.add({v}_lbl)',
        ]
    return "\n".join(lines)


def _inclined_plane(v: str, p: dict) -> str:
    angle_deg = _numeric_param("inclined_plane", p, "angle", 30)
    width = _numeric_param("inclined_plane", p, "width", 4.0)
    height = width * math.tan(math.radians(angle_deg))
    return "\n".join([
        f'{v} = Polygon(',
        f'    ORIGIN, RIGHT*{width:.3f}, RIGHT*{width:.3f} + UP*{height:.3f},',
        f'    color="#6b8cba", fill_opacity=0.45, stroke_width=2',
        f')',
    ])


def _ground(v: str, p: dict) -> str:
    extent = _numeric_param("ground", p, "extent", 7.0)
    texture = p.get("texture", "ground")
    color_map = {"grass": "#7ecfa0", "ice": "#a8d8ea", "rough": "#c09060", "ground": "#909090"}
    color = color_map.get(texture, "#909090")
    half = extent / 2.0
    return "\n".join([
        f'{v}_line = Line(LEFT*{half:.2f}, RIGHT*{half:.2f}, color="{color}", stroke_width=5)',
        f'{v}_fill = Rectangle(width={extent:.2f}, height=0.28, color="{color}", fill_opacity=0.28, stroke_width=0)',
        f'{v}_fill.next_to({v}_line, DOWN, buff=0)',
        f'{v} = VGroup({v}_line, {v}_fill)',
    ])


def _wall(v: str, p: dict) -> str:
    side = p.get("side", "right")
    shift = "RIGHT*3.2" if side == "right" else "LEFT*3.2"
    return "\n".join([
        f'{v} = Rectangle(width=0.3, height=3.5, color="#8b6914", fill_opacity=0.88, stroke_width=2)',
        f'{v}.shift({shift})',
    ])


def _arrow_force(v: str, p: dict) -> str:
    label = p.get("label", "F")
    direction = p.get("direction", "RIGHT").upper()
    color = p.get("color", "#ff7a59")
    length = _numeric_param("arrow_force", p, "length", 1.6)
    dir_vec = {"RIGHT": f"RIGHT*{length:.2f}", "LEFT": f"LEFT*{length:.2f}",
               "UP": f"UP*{length:.2f}", "DOWN": f"DOWN*{length:.2f}"}
    label_side = {"RIGHT": "UP", "LEFT": "UP", "UP": "RIGHT", "DOWN": "RIGHT"}
    end = dir_vec.get(direction, f"RIGHT*{length:.2f}")
    lside = label_side.get(direction, "UP")
    lines = [f'{v}_arrow = Arrow(ORIGIN, {end}, color="{color}", stroke_width=5, buff=0)']
    if label:
        lines += [
            f'{v}_label = Text("{label}", font_size=22, color="{color}", weight=BOLD)',
            f'{v}_label.next_to({v}_arrow, {lside}, buff=0.06)',
            f'{v} = VGroup({v}_arrow, {v}_label)',
        ]
    else:
        lines += [f'{v} = VGroup({v}_arrow)']
    return "\n".join(lines)


def _velocity_indicator(v: str, p: dict) -> str:
    mag = _numeric_param("velocity_indicator", p, "magnitude", 1.2)
    color = p.get("color", "#4fc3f7")
    return "\n".join([
        f'{v} = VGroup(',
        f'    Arrow(ORIGIN, RIGHT*{mag:.2f}, color="{color}", stroke_width=4, buff=0),',
        f'    Text("v", font_size=20, color="{color}", slant=ITALIC).shift(RIGHT*{mag/2:.2f}+UP*0.3),',
        f')',
    ])


def _acceleration_trail(v: str, p: dict) -> str:
    return "\n".join([
        f'{v} = VGroup(*[',
        f'    Dot(radius=0.05+i*0.04, color="#4fc3f7", fill_opacity=max(0.15, 0.8-i*0.22))',
        f'    .shift(LEFT*(0.45+i*0.52))',
        f'    for i in range(5)',
        f'])',
    ])
