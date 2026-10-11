# YOLO 切片接入（V2 / 3.13.0）

用户后续明确要求先直接启用：当前运行配置优先加载 `models/battlefield/katacr-real-units-v3/candidate/weights/best.pt`（模型标识 `katacr-2b0171bc060e`），`slice_fallback_enabled=true`。候选结果通过原有置信度、类别映射和等级徽标筛选后进入敌我单位观测及战场威胁判断；没有有效单位时使用公开模型回退。此为显式试用，历史失败复测结论及未验收标记保留。关闭旧窗口并重新打开后，启动新任务生效。

## V3 / 3.15.0 真实战场训练

新增 `export-real`：读取固定上游 `images/part2` 的 Labelme 矩形框，将敌我数字及攻击/冻结状态后缀转换为原单位类别。保留进化单位类别，未知标签计入 ignored_annotations，不使用上游非标准多列 txt。校验图像尺寸、框坐标及来源版本。按完整对局分 train/val，再均匀抽取时间帧；相同图片的哈希不可跨集合。

```powershell
python -m crbot.slice_dataset export-real --output training/exports/katacr-real-v1 --per-battle 8
python -m crbot.slice_dataset train --source training/exports/katacr-real-v1 --output models/battlefield/katacr-real-v1 --base-model models/battlefield/katacr/candidate/weights/best.pt --epochs 50 --device auto
```

也可使用 `train_yolo_real.bat`。本次选出 41 场对局中 212 张训练图和 64 张验证图，原切片模型用于初始化。来源是旧版上游游戏画面；当前游戏效果另用本机截图复测。本机截图没有用于训练，也没有自动当成正确标注。

中途本机复测显示第一版偏向塔和血条，另建单位专用版本：`export-real --units-only --per-battle 32` 只保留现有机器人目录可映射的 85 类单位，跳过没有单位框的帧，得到 train=649 / val=194。该次训练使用 YOLO11n 基础权重、batch=16、20 轮；第一版保留用于对照。13,101 个被忽略的标注主要是非任务类别（塔、血条、文字等），没有作为单位标签使用。

单位 v2 本机复测有两个有效框，但视觉复核将加农炮战车误认为大炮，不能仅按有框数量启用。采样审计发现 cannon-cart 仅 5 个训练框，giant 仅 9 个。新增 `--focus-units cannon-cart giant`，按同一对局分组补充稀有类别出现的帧（每组最多 64 个补充帧），保留验证集隔离。单位 v3 为 train=755 / val=194，cannon-cart=36、giant=84 个训练框，以 v2 权重再微调 20 轮。本机截图仍不进入训练。

BAT 默认生成和训练单位 v3；有 v2 权重时继续微调，没有时从 YOLO11n 基础权重训练。重复训练生成新的 candidate 目录，不自动更改运行配置。不要将训练次数或目录版本当成验收结果。

本轮结果（2026-10-11）：旧切片模型、真实标注 v1、单位 v2、补充采样单位 v3 均在同一组 12 张本机截图、conf=0.7 下复测，回退关闭。旧模型 0 个框；v1 虽每帧有塔/血条框，但没有有效单位；v2 两个框均将战车误认成大炮；v3 三个有效框中，一个 X 连弩视觉复核正确，另两个仍把战车认成大炮。该候选拒绝启用，所有权重保留。另行验证恢复配置可加载公开 YOLOv10 基线、使用 CUDA，并在本机真实截图返回带敌我证据的火枪手。新增/相关 27 项测试通过。未测量完整真实截图精确率、未做新的完整对局验收。

复测可用 `--reference-audit <旧audit.json>` 锁定帧名与 SHA256，避免目录新增截图影响对比样本。原图、覆盖图、audit.json、local-review.json 和训练权重均留在忽略目录，代码 Git 不包含这些资产；新增 `*.pt` 忽略规则保护基础权重下载。

`real_battle_bootstrap` 模型推理时截取整张游戏截图顶部 80% 的竞技场，再把检测框映射回完整截图坐标；旧切片模型保留原坐标方式。等级徽标、置信度、单位目录映射等筛选仍生效。

默认 `slice_fallback_enabled=true`：候选模型没有通过筛选的单位时，尝试调用既有公开战场基线；回退结果保留 `public_pretrained` 来源。状态显示 primary_units、accepted_units、fallback_loaded、fallback_used 和 fallback_error，不能将回退结果当成候选成绩。公开基线缺失/失败时保留原威胁估计。独立复测工具关闭回退，以单独衡量候选。

模型运行版本使用 `katacr-<权重SHA256前12位>`；source_revision 单独显示上游版本。不同训练权重不会再共用一个运行版本标识。

`tools/audit_slice_frames.py --model <best.pt> --frames <samples目录> --output <本地报告目录>` 可保存均匀抽取的 12 张真实画面检测覆盖图及 audit.json。有框/有效单位数不等于精确率；未人工标注的截图不用于声称准确率或实战验收。

3.14.4 诊断补充：权重加载不等于可用于游戏。最近一次对局抽取的三张截图，在 conf=0.25 下得到 0、1、0 个框，唯一框置信度约 0.359，均无法通过运行门槛 0.7。合成切片的验证成绩不能外推到真实战场。运行状态新增 raw_boxes 与 accepted_units，区分网络输出和通过映射/徽标筛选的单位；启动日志明确显示切片候选未实战验收。下一步应使用真实战场背景与多单位场景训练，配合独立真实截图标注验证，而非单纯降低运行置信度。

本机曾按用户要求直接启用 `training.slice_model_path=models/battlefield/katacr/candidate/weights/best.pt`。3.15.0 完成真实标注训练及视觉复核后，候选仍有高置信度误分类，因此删除此配置项，恢复既有冠军/公开基线选择流程（本机验证实际加载公开基线）。停止并重新启动机器人任务后生效；仅更新配置不会替换已运行实例中的模型。所有候选权重仍保存在本地，未改变冠军注册表，也没有将失败候选标记为已验收。

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
