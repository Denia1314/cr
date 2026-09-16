# 战场单位识别

V1 接入真实战场单位检测。加载顺序为本地训练冠军优先；没有可加载冠军时，使用已安装的公开预训练基线。模型文件缺失、摘要不匹配、依赖缺失或推理失败会显示原因，并退回原来的等级标记/血条位置检测。

## 安装与日常启动

双击 `setup_battlefield.bat` 安装并对本地历史截图进行只读验证。普通 BAT 和 EXE 启动入口的 `tools/ensure_gpu.py` 也会准备识别依赖与权重。首次联网下载约 31 MB 权重；ONNX Runtime 首次安装还需要额外下载。完整文件已经存在且 SHA-256 正确时不重复下载。战斗循环不联网安装。

GPU 环境固定使用 `onnxruntime-gpu==1.23.2`，匹配本项目 PyTorch CUDA 12.8/cuDNN 9；其他环境使用相同版本 CPU 包。`CRBOT_COMPUTE_DEVICE=cpu` 保留 CPU 路径。GPU 与 CPU wheel 不混装。仅在启动阶段预热；首次 GPU 内核编译可能较慢。

直接运行 Python 的用户先执行：

```powershell
.venv\Scripts\python.exe tools/ensure_battlefield.py
.venv\Scripts\python.exe -m tools.verify_battlefield
```

配置项位于 `training`，省略时使用以下默认值：

| 配置 | 默认 | 含义 |
|---|---:|---|
| `pretrained_enabled` | `true` | 无本地冠军时允许公开预训练模型 |
| `pretrained_confidence` | `0.55` | 单位检测最低置信度 |
| `pretrained_side_confidence` | `0.85` | 旧配置兼容保留；3.12.5 起不参与敌我判定 |

公开基线的兵种阈值独立于本地 YOLO 冠军的 `runtime_confidence`。3.12.5 起，两种模型的敌我判定均以头顶红／蓝等级标志为准：要求数字形状和相应背景色，将标志与单位框一对一关联；位置不合适、敌我标志冲突或没有可见标志时不提交确定兵种观测。红色衣服、红色地面或分类器的高分不能替代等级标志。等级标记回退也使用同一检测规则，无法识别兵种时仍显示未知单位。

独立记录 `side_evidence=level_badge`、`level_badge_bbox` 及 `level`。目前只对可明确辨识的双 1 字形记录等级 11；其他或遮挡的数字保留未知，红色背景本身不代表 11 级。一个标志不会重复赋给两个单位。此处等级是观测字段，暂不改写推演的 `assumed_level`。

## 模型与身份含义

来源为 [Pbatch/ClashRoyaleBuildABot](https://github.com/Pbatch/ClashRoyaleBuildABot/tree/f89ecbf309e698cb9fccdfb64a6ff2a930e6541e)，固定提交 `f89ecbf309e698cb9fccdfb64a6ff2a930e6541e`。

| 文件 | 作用 | SHA-256 |
|---|---|---|
| `units_M_480x352.onnx` | YOLOv10m，97 类单位/对象 | `4cf422370650b1d18647cffc05e02bf75fdc2fbdb1f8ef983f4b8d6f8ea908b3` |
| `side.onnx` | 旧版敌我分类权重，继续校验/同步，3.12.5 起不执行 | `686bc75eae2ddc872b556534aeac8b4f5c6e891df76eef12e9fc91d406ab2583` |

权重的导出时间为 2024-08-31。适配器严格恢复裁剪和 letterbox 坐标，使用模型内置类别索引，并映射到本项目卡牌目录，例如 `hungry_dragon → baby_dragon`、`minipekka → mini_pekka`、`archer → archers`。

**97 个原始类别不等于支持当前全部卡牌。** 雪球特效、石头人分裂体、凤凰蛋等没有可靠本地实体统计映射的类别会被忽略，不套用母体属性。进化、冠军技能、混合兵种及新卡牌仍可能漏识别。一个目标框表示一个实体；识别到骷髅不能证明对方刚打了骷髅兵卡，召唤物也不能作为已确认出牌事件扣费。

血量、等级和进化状态与兵种身份分开。识别到兵种不保证可读出血量；此类字段继续保留未知，不能凭空补成满血。

公主塔读取标定血条；敌方血条要求粉红颜色，避免倒塔后露出的红色道路被当成剩余血量。连续至少两张新帧、跨越至少 0.2 秒出现标准场地的石木废墟特征，且读不到有效血条，才确认摧毁。确认后同局锁定为零，下一局显式重置；单纯读不到血条、平坦地面或短暂特效不足以判死。废墟检测是保守视觉规则，不是训练后的塔检测模型；未覆盖皮肤、遮挡和坐标不匹配仍可能保持未知。推演和逐帧视图均接入该状态，未知塔血量不再画满血条。

模型保存在 `models/battlefield/pretrained/pbatch/`，不写入本地训练 `champion`。模型状态窗口和启动日志标明“公开预训练，待本地准确率验收”。已有本地冠军优先，公开基线不自动晋级、不伪造 precision/recall/mAP。

## 验证和同步

`tools.verify_battlefield` 调用与机器人相同的 `LearnedBattlefieldDetector.detect`，读取历史截图，生成带框预览和 `reports/battlefield-recognition/<时间>/results.json`。报告区分加载成功、实际推理次数、实际 CUDA 算子执行、检测数量与完整对局验收。检测数量不是准确率；目前仍需要独立标注集才能测量 precision/recall。

运行中的 `.training-sync/runtime-status.json` 新增 `battlefield_recognition`：包括版本、加载状态、错误、推理次数、GPU 次数、检测数量和最近耗时。旧进程需要重启机器人才能加载新代码和模型。

既有私有训练同步额外交换这两个固定摘要的 ONNX 文件、来源清单及许可证说明，路径为数据仓库的 `models/battlefield_pretrained/<版本>/`。接收时重新核验摘要，不改变训练冠军或质量结论。代码仓库不包含模型、原始截图或本地报告；另一台电脑须更新代码并独立准备运行环境。

上游仓库/side 模型 MIT 声明和检测模型内置 AGPL-3.0 声明见 [第三方声明](../THIRD_PARTY_NOTICES.md)。
