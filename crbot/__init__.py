"""MuMu offline-AI battle automation reference implementation."""

__version__ = "3.15.7"

# Delivered capabilities, not model identities or claims of battle acceptance.
# Keep chronological order; all release displays derive from this catalog.
IMPLEMENTED_STAGES = (
    ("M3-D", "离线评估"),
    ("M3-E", "受控实测"),
    ("P1", "预测推演 · 实验版"),
    ("SL0", "学习数据契约"),
    ("SL1", "自主候选调度"),
    ("SL2", "学习动作评分"),
    ("SL3", "自动实战验收"),
    ("SL4", "自动部署回退"),
    ("G1", "方格战场推演"),
    ("V1", "战场单位识别"),
    ("R1", "画面停滞自动恢复"),
    ("MR0", "只读内存诊断"),
    ("MR1", "本地手动内存采集"),
    ("V2", "YOLO 切片训练接入"),
    ("UI1", "浅色轻量工作台"),
    ("V3", "真实战场标注训练"),
    ("UI2", "YOLO 识别画面"),
    ("V4", "YOLO 持续训练"),
    ("G2", "卡牌属性约束"),
    ("UI3", "卡牌与切片库"),
    ("P2", "GPU 批量战斗推演 · 实验版"),
)


def current_stage_label() -> str:
    stage, name = IMPLEMENTED_STAGES[-1]
    return f"{stage} · {name}"


def release_label() -> str:
    return f"Royal Lab {__version__} · {current_stage_label()}"


def implemented_stages_label() -> str:
    return " · ".join(f"{stage} {name}" for stage, name in IMPLEMENTED_STAGES) + "（已接入）"
