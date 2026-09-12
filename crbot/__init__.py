"""MuMu offline-AI battle automation reference implementation."""

__version__ = "2.5.2"

# Delivered capabilities, not model identities or claims of battle acceptance.
# Keep chronological order; all release displays derive from this catalog.
IMPLEMENTED_STAGES = (
    ("M3-D", "离线评估"),
    ("M3-E", "受控实测"),
)


def release_label() -> str:
    stage, name = IMPLEMENTED_STAGES[-1]
    return f"Royal Lab {__version__} · {stage} · {name}"


def implemented_stages_label() -> str:
    return " · ".join(f"{stage} {name}" for stage, name in IMPLEMENTED_STAGES) + "（已接入）"
