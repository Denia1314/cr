# YOLO 切片接入（V2 / 3.13.0）

3.14.4 诊断补充：权重加载不等于可用于游戏。最近一次对局抽取的三张截图，在 conf=0.25 下得到 0、1、0 个框，唯一框置信度约 0.359，均无法通过运行门槛 0.7。合成切片的验证成绩不能外推到真实战场。运行状态新增 raw_boxes 与 accepted_units，区分网络输出和通过映射/徽标筛选的单位；启动日志明确显示切片候选未实战验收。下一步应使用真实战场背景与多单位场景训练，配合独立真实截图标注验证，而非单纯降低运行置信度。

本机已按用户要求直接启用 `training.slice_model_path=models/battlefield/katacr/candidate/weights/best.pt`，优先加载完成 50 轮训练的切片候选模型。此为显式使用候选，不改变冠军注册表及未验收标记。停止并重新启动机器人任务后生效；仅更新配置不会替换已运行实例中的模型。权重仍保存在本地，另一台机器使用此配置前需要自行准备对应权重和候选 manifest。回退时删除此配置项，即恢复既有冠军/公开基线选择流程。

来源：https://github.com/wty-yy/Clash-Royale-Detection-Dataset
固定版本：`31b4151fedb1b914e99c3c122c16dca61cb2b905`，上游 MIT 许可随素材保存。

双击 `import_yolo_slices.bat` 下载并导入。素材位于 `training/external/katacr-slices`，属于本地训练资产，不进入代码 Git 仓库。导入保留原类别、文件名、从属类别、尺寸及 SHA256，背景和文字作为辅助素材保存。

双击 `train_yolo_slices.bat` 可导出训练集并启动全量 50 轮训练（GPU 优先），耗时取决于硬件。训练完成只保存候选，不自动启用。也可在 Royal Lab 的 Python 环境运行：

```powershell
python -m crbot.slice_dataset import
python -m crbot.slice_dataset export --source training/external/katacr-slices --output training/exports/katacr-bootstrap
python -m crbot.slice_dataset train --source training/exports/katacr-bootstrap --output models/battlefield/katacr --epochs 50 --device auto
```

训练依赖见 `requirements-training.txt`。沿用项目 YOLO11n 基础模型，可用 `--base-model` 指定本地模型。Windows 使用 workers=0；auto 通过实际 CUDA 运算选择 GPU，否则 CPU。训练输出候选模型及来源清单，不修改冠军注册表。

导出为标准 YOLO 五列检测标签，类别是原始单位类别。每张合成图放一张经过透明蒙版裁剪的切片；按切片内容哈希先分 train/val，避免同一切片跨集合。生成的是引导训练数据：简单随机背景有明显领域差异，合成验证指标不代表真实战场精度。完整切片资产可用于后续更丰富的场景合成。

试验模型可在 config.json 的 training 中显式设置 `slice_model_path` 为 `models/battlefield/katacr/candidate/weights/best.pt`（以实际训练输出为准）。这会优先使用试验模型；不配置则保留现有识别路径。试验加载失败会返回原画面威胁估计，并在 status.error 报告原因。

运行适配器先核对模型类别与清单，再把连字符名称映射至已有单位目录；无法可靠映射的子单位和进化单位暂不赋予卡牌统计。敌我关系仍由画面等级徽标确认，不由切片文件名或位置推断。`external_slice_candidate` 明确区别于正式冠军，`deployment_confirmed`、`local_quality_validated`、`battle_acceptance` 均保持 false。

V2 表示导入、合成导出、YOLO 训练和试验运行接口已经实现，不表示完成模型训练质量验收或实战验收。正式采用前仍需独立真实对局标注、离线质量评估和受控实测。

本次本机验证：完整导入 4,654 张图片（4,525 张类别切片、129 张辅助素材），导出 train=3,582 / val=943。RTX 5070 Ti 完成 16 张训练图和 16 张验证图的一轮 YOLO11n 流程测试，并成功加载保存权重执行推理。此流程测试没有足够训练量，不应启用为正式识别模型。

新增 4 项切片测试和原有 9 项公开战场识别测试通过。全项目回归运行时 658 项中有 1 项失败：`test_two_card_defense_reduces_tower_loss_with_recomputed_followup_positions`，塔伤均为 253；在本次修改前的 HEAD 原始代码副本上复测同样失败。此已有推演问题未纳入切片接入改动。
