from __future__ import annotations

import argparse
import os
import json
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .adb import DeviceError, MumuDevice
from .action_review import audit_action_review, export_action_review
from .annotate import run_annotation
from .calibrate import run_calibration
from .card_sync import bootstrap_community_catalog, sync_card_catalog
from .config import load_config, resolve_project_path
from .controlled_experiment import audit_controlled_experiment
from .engine import BotEngine
from .demonstration import DemonstrationRecorder
from .cards import CardCatalog
from .learning import (
    ModelRegistry,
    audit_learning_data,
    auto_label_with_champion,
    export_yolo_dataset,
    run_learning_cycle,
    train_detector,
)
from .imitation import audit_demonstrations, train_imitation_policy
from .replay import audit_replay, backfill_replay_history
from .replay_learning import (
    audit_replay_learning,
    evaluate_replay_snapshot,
    train_replay_policy,
)
from .vision import WorkflowRecognizer
from .training_sync import DEFAULT_REPOSITORY, ReplaySync, SyncWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crbot",
        description="皇室战争 MuMu 离线人机训练控制器",
    )
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="检查 MuMu、ADB、游戏与标定")
    subcommands.add_parser("calibrate", help="打开画面标定工具")
    subcommands.add_parser("annotate", help="打开对局数据人工标注工具")
    subcommands.add_parser("demonstrate", help="监控手动离线对局并学习触摸操作")
    cards_sync = subcommands.add_parser("cards-sync", help="从官方 API 或 JSON 同步卡牌清单")
    cards_sync.add_argument("--input", help="已导出的官方 cards JSON；不填写则请求官方 API")
    cards_sync.add_argument(
        "--token-env",
        default="CLASH_ROYALE_API_TOKEN",
        help="保存官方 API token 的环境变量名",
    )
    cards_sync.add_argument(
        "--community-bootstrap",
        action="store_true",
        help="无官方 token 时，从 RoyaleAPI 静态数据建立通用目录并下载卡图",
    )
    learn = subcommands.add_parser("learn", help="审计、导出或训练战场识别模型")
    learn.add_argument(
        "action",
        choices=("audit", "export", "train", "auto-label", "cycle"),
        help="audit=检查，export=导出，train=训练，auto-label=伪标注，cycle=安全学习闭环",
    )
    learn.add_argument("--output", help="export 的输出目录")
    learn.add_argument("--limit", type=int, default=0, help="auto-label 最多检查多少帧")
    imitate = subcommands.add_parser("imitate", help="审计或训练手动示范模仿策略")
    imitate.add_argument("action", choices=("audit", "train"))
    replay = subcommands.add_parser("replay", help="审计、回填或影子训练自动战斗经验")
    replay.add_argument(
        "action",
        choices=("audit", "backfill", "evaluate", "experiment-audit", "review-export", "review-audit", "train"),
    )
    replay.add_argument("--review-file", help="动作确认人工复核 JSONL 文件")
    replay.add_argument("--review-limit", type=int, default=200, help="复核导出的完整尝试数")
    sync = subcommands.add_parser("sync", help="双机回放数据共享")
    sync.add_argument("action", choices=("setup", "now", "status"), nargs="?", default="now")
    sync.add_argument("--repository", default=DEFAULT_REPOSITORY)
    sync.add_argument("--trainer", action="store_true", help="只在第一台训练机初始化时使用")
    capture = subcommands.add_parser("capture", help="保存一张当前截图")
    capture.add_argument("--output", help="输出 PNG 路径")
    run = subcommands.add_parser("run", help="运行自动训练")
    run.add_argument("--dry-run", action="store_true", help="只识别与记录，不点击")
    run.add_argument("--max-battles", type=int, default=0, help="完成多少局后停止，0 表示不限")
    return parser


def doctor(device: MumuDevice, config: dict, config_path: Path) -> int:
    serial = device.connect()
    info = device.info()
    package = config["game"]["package"]
    installed = device.package_installed(package)
    image = device.screenshot()
    recognizer = WorkflowRecognizer(config, config_path)
    calibrated = recognizer.calibrated_names()
    print(f"连接模式 : {device.connection_description}")
    print(f"MuMu CLI : {device.cli_path or '未使用'}")
    print(f"ADB      : {device.adb_path}")
    print(f"设备     : vmindex={device.vm_index}, serial={serial}")
    print(f"Android  : {info.get('android_version')}, started={info.get('is_android_started')}")
    print(f"截图     : {image.width}x{image.height}")
    print(f"游戏包   : {package} -> {'已安装' if installed else '未安装'}")
    print(f"已标定   : {', '.join(calibrated) if calibrated else '无'}")
    if config.get("automation", {}).get("single_marker_mode", False):
        required = {"offline_ai_marker"}
    else:
        required = {"offline_ai_marker", "start_battle", "battle_marker"}
    missing = sorted(required - set(calibrated))
    if missing:
        print(f"必须补充 : {', '.join(missing)}")
    return 0 if installed and not missing else 2


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config, config_path = load_config(arguments.config)
        if arguments.command == "sync":
            service = ReplaySync(config_path.parent)
            if arguments.action == "setup":
                result = service.setup(arguments.repository, arguments.trainer)
            elif arguments.action == "status":
                result = service.status()
            else:
                if not service.config.get("enabled"):
                    raise ValueError("请先运行 sync setup 完成本机配置")
                result = service.sync()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if arguments.command == "cards-sync":
            catalog_value = config.get("dataset", {}).get("card_catalog", "data/cards.json")
            catalog_path = resolve_project_path(config_path, str(catalog_value))
            if arguments.community_bootstrap:
                count, icons = bootstrap_community_catalog(catalog_path)
                print(f"已建立 {count} 张卡牌的通用目录，下载 {icons} 张卡图：{catalog_path}")
                return 0
            input_path = (
                Path(arguments.input).expanduser().resolve()
                if arguments.input
                else None
            )
            token = None if input_path is not None else os.environ.get(arguments.token_env)
            count = sync_card_catalog(
                catalog_path,
                token=token,
                input_path=input_path,
            )
            print(f"已同步 {count} 张卡牌基础资料：{catalog_path}")
            return 0
        if arguments.command == "annotate":
            catalog_value = config.get("dataset", {}).get("card_catalog", "data/cards.json")
            run_annotation(
                config_path.parent,
                resolve_project_path(config_path, str(catalog_value)),
            )
            return 0
        if arguments.command == "learn":
            catalog_value = config.get("dataset", {}).get("card_catalog", "data/cards.json")
            catalog = CardCatalog.load(
                resolve_project_path(config_path, str(catalog_value))
            )
            training_config = dict(config.get("training", {}))
            registry = ModelRegistry(config_path.parent)
            champion = registry.champion()
            champion_version = str(champion.get("version", "")) if champion else ""
            if arguments.action == "audit":
                result = audit_learning_data(
                    config_path.parent,
                    catalog,
                    training_config,
                    champion_version=champion_version,
                ).to_dict()
            elif arguments.action == "export":
                output = (
                    Path(arguments.output).expanduser().resolve()
                    if arguments.output
                    else config_path.parent
                    / "training"
                    / "exports"
                    / datetime.now().strftime("%Y%m%d_%H%M%S")
                )
                result = export_yolo_dataset(
                    config_path.parent,
                    catalog,
                    training_config,
                    output,
                    champion_version=champion_version,
                )
            elif arguments.action == "train":
                result = train_detector(config_path.parent, catalog, training_config)
            elif arguments.action == "auto-label":
                result = auto_label_with_champion(
                    config_path.parent,
                    catalog,
                    training_config,
                    limit=max(0, int(arguments.limit)),
                )
            else:
                result = run_learning_cycle(
                    config_path.parent,
                    catalog,
                    training_config,
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if arguments.command == "imitate":
            catalog_value = config.get("dataset", {}).get("card_catalog", "data/cards.json")
            catalog = CardCatalog.load(
                resolve_project_path(config_path, str(catalog_value))
            )
            demonstration_config = dict(config.get("demonstration", {}))
            if arguments.action == "audit":
                result = audit_demonstrations(
                    config_path.parent, catalog, demonstration_config
                ).to_dict()
            else:
                result = train_imitation_policy(
                    config_path.parent, catalog, demonstration_config
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if arguments.command == "replay":
            replay_config = dict(config.get("replay", {}))
            result: dict[str, object] = {}
            review_file = (
                Path(arguments.review_file).expanduser().resolve()
                if arguments.review_file
                else config_path.parent / "reports" / "action_confirmation_review_v2.jsonl"
            )
            if arguments.action == "experiment-audit":
                result["experiment_audit"] = audit_controlled_experiment(config_path.parent)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
            if arguments.action == "review-export":
                result["review_export"] = export_action_review(
                    config_path.parent,
                    review_file,
                    limit=arguments.review_limit,
                    seed=int(replay_config.get("training_seed", 20260908)),
                )
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
            if arguments.action == "review-audit":
                result["review_audit"] = audit_action_review(
                    review_file,
                    seed=int(replay_config.get("training_seed", 20260908)),
                )
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
            if arguments.action == "backfill":
                result["backfill"] = backfill_replay_history(
                    config_path.parent,
                    replay_config,
                )
            result["audit"] = audit_replay(
                config_path.parent,
                replay_config,
            ).to_dict()
            catalog_value = config.get("dataset", {}).get("card_catalog", "data/cards.json")
            catalog = CardCatalog.load(
                resolve_project_path(config_path, str(catalog_value))
            )
            result["learning_audit"] = audit_replay_learning(
                config_path.parent,
                catalog,
                replay_config,
            ).to_dict()
            if arguments.action == "evaluate":
                result["evaluation"] = evaluate_replay_snapshot(
                    config_path.parent,
                    catalog,
                    replay_config,
                )
            if arguments.action == "train":
                result["training"] = train_replay_policy(
                    config_path.parent,
                    catalog,
                    replay_config,
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        device = MumuDevice(config)
        if arguments.command == "doctor":
            return doctor(device, config, config_path)

        device.connect()
        if arguments.command == "calibrate":
            run_calibration(device, config, config_path)
            return 0
        if arguments.command == "demonstrate":
            DemonstrationRecorder(device, config, config_path).run()
            return 0
        if arguments.command == "capture":
            image = device.screenshot()
            output = (
                Path(arguments.output).expanduser().resolve()
                if arguments.output
                else config_path.parent / "captures" / f"capture_{datetime.now():%Y%m%d_%H%M%S}.png"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output, format="PNG")
            print(output)
            return 0
        if arguments.command == "run":
            engine = BotEngine(
                device,
                config,
                config_path,
                dry_run=arguments.dry_run,
                max_battles=arguments.max_battles,
            )
            sync_worker = SyncWorker(config_path.parent).start()
            try:
                engine.run()
            finally:
                sync_worker.close()
            return 0
    except KeyboardInterrupt:
        print("\n用户已停止。")
        return 130
    except (DeviceError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0
