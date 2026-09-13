"""MuMu offline-AI battle automation reference implementation."""

__version__ = "3.0.1"

# Delivered capabilities, not model identities or claims of battle acceptance.
# Keep chronological order; all release displays derive from this catalog.
IMPLEMENTED_STAGES = (
    ("M3-D", "离线评估"),
    ("M3-E", "受控实测"),
    ("P1", "预测推演 · 实验版"),
)


def current_stage_label() -> str:
    stage, name = IMPLEMENTED_STAGES[-1]
    return f"{stage} · {name}"


def release_label() -> str:
    return f"Royal Lab {__version__} · {current_stage_label()}"


def implemented_stages_label() -> str:
    return " · ".join(f"{stage} {name}" for stage, name in IMPLEMENTED_STAGES) + "（已接入）"
