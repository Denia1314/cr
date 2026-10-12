# 姿态补样与独立验证（3.18.0 / V6）

V6 已实现人工核对的真实姿态补样和独立验证工具。阶段名称表示工具能力；训练完成、局部试用质量通过、敌我确认和真实对局验收分别记录。

模型类别中已有超骑，也可能把登场、跃击姿态认成冰人或骑士。增加训练样本后，仍须检查真实画面；等级标志不可见时，正确兵种分类也不等于确定敌我。原运行置信度 0.7 和 V5 未确认目标机制保留。

## 数据准备

先在本地报告目录保存人工逐张查看的标注 JSON 数组。每条含 `path`（相对工作目录）、原图 `sha256`、实际对局 `group`、`split`（train 或 holdout）、`card`（模型原类别名，如 mega-knight）、归一化 `bbox`（x1,y1,x2,y2）和 `annotation_source=manual_visual_review`。出牌日志用于寻找图片，不能自动成为正确标注。

`crbot.pose_dataset` 要求图像校验和一致、框有效、类别兼容，按整场对局隔离训练与留出集，并禁止重复源图或相同内容跨集。原始图片不改写，输出必须是新目录。补样使用已经核对的身体裁剪，按原画面尺度略作变化，贴入已有标注的训练场景，避免遮盖原单位框，保留全部背景标签。验证背景不会用于合成。

```powershell
python -m crbot.pose_dataset --base training/exports/katacr-pose-general --root . --reviews reports/mega-knight-pose/manual-boxes.json --output training/exports/mega-knight-pose-new --copies 20
python -m crbot.slice_dataset train --source training/exports/mega-knight-pose-new --output models/battlefield/mega-knight-pose-new --base-model models/battlefield/katacr-real-units-v3/candidate-5/weights/best.pt --epochs 30 --batch 16 --device auto
```

路径以本机实际已有数据和权重为准。训练输出放在现有自动选择系列之外，避免只因训练完成而启用。训练沿用原继续训练规则和数据指纹。增强数量、原始训练图片数量和独立验证数量在 manifest 分开保存；一张图生成 20 份增强，仍只是一张原始图。

## 固定验证与试用门槛

另建 `validation.json`，只包含留出对局的人工标注；无目标的负例用空 `bbox`。正例和负例都校验对局、内容哈希和源图，拒绝与补样训练图重合及重复计数。正例不用于生成补样。

```powershell
python -m crbot.pose_validation --baseline models/battlefield/katacr-real-units-v3/candidate-5/weights/best.pt --candidate models/battlefield/mega-knight-pose-new/candidate/weights/best.pt --root . --reviews reports/mega-knight-pose/validation.json --dataset training/exports/mega-knight-pose-new --output reports/mega-knight-pose/comparison-new
```

本机留出画面使用相同数据、0.7 置信度、0.5 IoU、战场顶部 80% 裁剪；按置信度一对一匹配，重复框算误报。报告保留逐图预测框、TP/FP/FN、数据身份与权重 SHA256，同时对原有标注验证集计算全兵种 mAP50 和 precision。原通用验证集已经是战场裁剪图，按原尺寸验证。

通用验证集中没有目标标签的完整标注画面，还作为目标类别的易混淆负例检查：该类的每个高分输出都计入误报和目标 precision 的分母。因为没有真实目标的类别可能不参与总体 mAP 平均，只看全兵种 mAP 会漏掉这类误报。`local_totals` 单独保存本机画面成绩，`general_target_fp` 保存额外误报；两者不能混淆。

预热后单帧耗时包含预处理和后处理，不能当作实际游戏刷新帧率。

预设局部试用门槛：至少三张目标正例、两张负例、两个独立对局；目标 precision 至少 0.9、recall 至少 2/3 且高于原模型；全兵种 mAP50 和 precision 各不能下降超过 0.02；推理中位耗时不超过 100ms。非有限指标或对照数据不一致直接失败。

`gate.json` 的通过范围只为 `limited_pose_trial`，始终保留 `battle_acceptance=false`。本工具不修改运行配置、冠军注册表或自动部署状态。小规模验证不能证明所有姿态、所有兵种准确度或胜率，需要更多独立人工标注和受控实战。

## 本机训练记录

2026-10-12 已核对八张训练源图（四场对局），另三张超骑图和两张无超骑图来自两场独立留出对局。原通用验证集有 194 张图、559 个有效标注；其中没有超骑，所以通用训练的 best.pt 选择不能单独证明超骑改善。

首轮 30 轮微调用 160 张姿态合成增强，保存权重完成真实对照验证，但没有通过局部试用门槛；训练产物及逐图证据保留在本地。后续实验仍使用固定留出集，不能把迭代复测当成全新独立验收。

第二轮 20 轮采用 640 张增强、冻结前段网络和较低学习率。末轮权重在本机留出集检出 2/3 个超骑且两张空场没有误报，但总体 mAP50 从 0.3092 降到 0.2816；扩大到 194 张无超骑标签的通用验证画面，固定批次复测产生 14 个超骑误报，目标 precision 为 2/(2+14)=0.125。视觉复核确认存在把皮卡判成超骑的高置信度错误。新旧权重平均也未通过原门槛。所有候选保持隔离，没有更换当前机器人权重。

原始画面、人工标注、增强数据、报告和权重不进入代码 Git。通过验收的正式模型按已有私有训练同步机制发布；外部切片实验没有自动晋级为该流程的正式冠军。
