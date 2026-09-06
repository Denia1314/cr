"""Private GitHub replay exchange. Game processes never wait for the network."""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.request import getproxies


DEFAULT_REPOSITORY = "Denia1314/cr-training-data"
SCHEMA = 1
MAX_RECORD_BYTES = 10 * 1024 * 1024
MAX_MODEL_BYTES = 50 * 1024 * 1024
MAX_CHECKOUT_BYTES = 512 * 1024 * 1024


class SyncError(ValueError):
    pass


class SyncTransportError(SyncError):
    pass


class SyncBusyError(SyncError):
    pass


def encode(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def complete_rows(path: Path) -> list[dict[str, Any]]:
    """Ignore only the incomplete final line of a file being appended to."""
    if not path.is_file():
        return []
    result = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                break
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise SyncError(f"记录格式错误：{path.name}")
                result.append(row)
    return result


@contextlib.contextmanager
def exclusive(path: Path):
    """OS lock is released on process exit, including crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SyncBusyError("同步或训练正在进行，请稍后重试") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def gh_path() -> str:
    found = shutil.which("gh")
    if found:
        return found
    for candidate in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "CodexTools/github-cli/bin/gh.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "GitHub CLI/gh.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    raise SyncError("请先安装 GitHub CLI，并运行 gh auth login 登录 GitHub")


def command(args: list[str], cwd: Path | None = None) -> str:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    try:
        result = subprocess.run(args, cwd=cwd, env=env, capture_output=True,
                                encoding="utf-8", errors="replace", timeout=45,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SyncTransportError("同步工具不可用或网络超时；本地数据保留，下次重试") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1200:]
        raise SyncTransportError(detail or "同步命令失败")
    return result.stdout.strip()


def settings(root: Path) -> dict[str, Any]:
    return read_json(root / ".training-sync/device.json", {})


def training_allowed(root: Path) -> bool:
    value = settings(root)
    return not value.get("enabled") or bool(value.get("trainer"))


def replay_source_prefix(run: Path) -> str:
    origin = read_json(run / ".sync-origin.json") or read_json(run / ".shared-replay.json", {})
    if all(re.fullmatch(r"[0-9a-f]{32}", str(origin.get(key, ""))) for key in ("device_id", "run_id")):
        return f"{origin['device_id']}:{origin['run_id']}:"
    return ""


class ReplaySync:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.state = self.root / ".training-sync"
        self.checkout = self.state / "repository"
        self.config = settings(self.root)

    def git(self, *args: str) -> str:
        options = ["git", "-c", "user.name=Royal Lab Sync", "-c", "user.email=sync@localhost",
                   "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false",
                   "-c", f"core.hooksPath={self.state / 'no-hooks'}"]
        if self.config.get("repository"):
            helper = '!"' + gh_path().replace("\\", "/") + '" auth git-credential'
            options += ["-c", "credential.helper=", "-c", f"credential.helper={helper}"]
            proxy = getproxies().get("https")
            if proxy:
                options += ["-c", f"http.proxy={proxy}"]
        return command(options + list(args), self.checkout)

    def _commit(self) -> None:
        self.git("add", "--all")
        if self.git("diff", "--cached", "--name-only"):
            self.git("commit", "-m", "Sync completed replay data")

    def _pull(self) -> None:
        self.git("fetch", "origin", "main")
        try:
            self.git("rebase", "FETCH_HEAD")
        except SyncError:
            if (self.checkout / ".git/rebase-merge").exists() or (self.checkout / ".git/rebase-apply").exists():
                self.git("rebase", "--abort")
            raise

    def _push(self) -> None:
        for attempt in range(3):
            try:
                self.git("push", "origin", "HEAD:main")
                return
            except SyncError:
                if attempt == 2:
                    raise
                self._pull()

    def protocol(self) -> dict[str, Any]:
        value = read_json(self.checkout / "protocol.json", {})
        if value.get("schema") != SCHEMA or not re.fullmatch(r"[0-9a-f]{32}", str(value.get("trainer_device", ""))):
            raise SyncError("数据仓库未初始化或协议版本不兼容，请先在训练机完成 setup --trainer")
        actual_trainer = value["trainer_device"] == self.config.get("device_id")
        if actual_trainer != bool(self.config.get("trainer")):
            raise SyncError("训练机身份与数据仓库不一致，请重新检查本机设置")
        return value

    def setup(self, repository: str, trainer: bool = False) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise SyncError("仓库名称必须是 owner/name")
        private = command([gh_path(), "api", f"repos/{repository}", "--jq", ".private"])
        if private != "true":
            raise SyncError("训练数据仓库必须为私有仓库")
        with exclusive(self.state / "sync.lock"):
            if self.config and self.config.get("repository") != repository:
                raise SyncError("当前目录已连接其他数据仓库，请使用独立项目目录")
            self.config = dict(self.config) or {"device_id": uuid.uuid4().hex}
            trainer = trainer or bool(self.config.get("trainer"))
            self.config.update(repository=repository, trainer=trainer, enabled=False)
            # Keep the identity if initial network setup fails midway.
            if not (self.state / "device.json").exists():
                atomic_write(self.state / "device.json", encode(self.config))
            self.checkout.mkdir(parents=True, exist_ok=True)
            if not (self.checkout / ".git").exists():
                self.git("init", "-b", "main")
                self.git("remote", "add", "origin", f"https://github.com/{repository}.git")
            self.git("fetch", "origin", "main")
            # A failed setup can leave a valid checkout; never replace its commits.
            try:
                self.git("rev-parse", "--verify", "HEAD")
            except SyncError:
                self.git("checkout", "-B", "main", "FETCH_HEAD")
            else:
                self._commit()
                self._pull()
            protocol = read_json(self.checkout / "protocol.json")
            if protocol is None and trainer:
                atomic_write(self.checkout / "protocol.json", encode({
                    "schema": SCHEMA, "trainer_device": self.config["device_id"],
                    "description": "Immutable completed replay episodes; no screenshots or credentials.",
                }))
            self.protocol()
            self._commit()
            self._push()
            self.config["enabled"] = True
            atomic_write(self.state / "device.json", encode(self.config))
        return self.status()

    def status(self) -> dict[str, Any]:
        return {"enabled": bool(self.config.get("enabled")),
                "repository": self.config.get("repository", DEFAULT_REPOSITORY),
                "device_id": self.config.get("device_id"),
                "role": "trainer" if self.config.get("trainer") else "collector",
                "last_sync": read_json(self.state / "status.json", {})}

    def _origin(self, run: Path) -> dict[str, Any]:
        path = run / ".sync-origin.json"
        value = read_json(path)
        if value is None:
            value = {"device_id": self.config["device_id"], "run_id": uuid.uuid4().hex}
            atomic_write(path, encode(value))
        for field in ("device_id", "run_id"):
            if not re.fullmatch(r"[0-9a-f]{32}", str(value.get(field, ""))):
                raise SyncError("对局来源编号无效")
        return value

    def export_records(self) -> int:
        count = 0
        for run in sorted((self.root / "runs").glob("*")):
            if not run.is_dir() or (run / ".shared-replay.json").exists():
                continue
            episodes = complete_rows(run / "replay_episodes.jsonl")
            if not episodes:
                continue
            origin = self._origin(run)
            if origin["device_id"] != self.config["device_id"]:
                continue
            transitions = complete_rows(run / "replay_transitions.jsonl")
            seen_battles: set[int] = set()
            for episode in episodes:
                battle = int(episode["battle_index"])
                if battle < 1 or battle in seen_battles:
                    raise SyncError(f"对局编号重复或无效：{run.name}")
                seen_battles.add(battle)
                rows = [dict(row) for row in transitions if row.get("battle_index") == battle]
                if len(rows) != int(episode["action_count"]) or (rows and not rows[-1].get("done")):
                    continue  # Still being written, or interrupted before completion.
                if len({row["transition_id"] for row in rows}) != len(rows):
                    raise SyncError(f"动作编号重复：{run.name} / {battle}")
                prefix = f"{origin['device_id']}:{origin['run_id']}:"
                for row in rows:
                    row["transition_id"] = prefix + str(row["transition_id"])
                    row.pop("frame", None)
                    row.pop("next_frame", None)
                episode = dict(episode, episode_id=prefix + str(episode["episode_id"]))
                episode.pop("result_frame", None)
                payload = {"schema": SCHEMA, **origin, "run_name": run.name,
                           "episode": episode, "transitions": rows,
                           "manifest": read_json(run / "replay_manifest.json", {})}
                data = encode(payload)
                if len(data) > MAX_RECORD_BYTES:
                    raise SyncError("单局记录超过 10 MiB，同步已暂停，请检查异常数据")
                target = self.checkout / "records" / origin["device_id"] / origin["run_id"] / f"b{battle:08d}.json"
                if target.exists():
                    if target.read_bytes() != data:
                        raise SyncError(f"已发布对局被修改：{run.name} / {battle}；未覆盖远端记录")
                else:
                    atomic_write(target, data)
                    count += 1
        return count

    def import_records(self) -> int:
        count = 0
        local_origins = set()
        for path in (self.root / "runs").glob("*/.sync-origin.json"):
            origin = read_json(path, {})
            local_origins.add((origin.get("device_id"), origin.get("run_id")))
        for path in sorted((self.checkout / "records").glob("*/*/*.json")):
            if any(item.is_symlink() for item in (path, path.parent, path.parent.parent)) or path.stat().st_size > MAX_RECORD_BYTES:
                raise SyncError("远端记录类型或大小不符合协议")
            payload = read_json(path)
            device, run_id = payload.get("device_id"), payload.get("run_id")
            if (device, run_id) in local_origins:
                continue
            if payload.get("schema") != SCHEMA or any(
                not re.fullmatch(r"[0-9a-f]{32}", str(value)) for value in (device, run_id)
            ) or path.parent.name != run_id or path.parent.parent.name != device:
                raise SyncError("远端对局协议或来源编号无效")
            episode, rows = payload["episode"], payload["transitions"]
            battle = int(episode["battle_index"])
            prefix = f"{device}:{run_id}:"
            if (battle < 1 or path.name != f"b{battle:08d}.json"
                    or len(rows) != int(episode["action_count"])
                    or not str(episode["episode_id"]).startswith(prefix)
                    or (rows and not rows[-1].get("done"))
                    or any(row.get("battle_index") != battle or not str(row.get("transition_id", "")).startswith(prefix) for row in rows)
                    or len({row["transition_id"] for row in rows}) != len(rows)):
                raise SyncError("远端对局不完整或动作编号无效")
            name = f"shared_{device}_{run_id}_b{battle:08d}"
            target = self.root / "runs" / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if target.exists():
                if read_json(target / ".shared-replay.json", {}).get("sha256") != digest:
                    raise SyncError("共享对局内容发生变化，未覆盖本地记录")
                continue
            staging = self.state / "imports" / uuid.uuid4().hex
            staging.mkdir(parents=True)
            try:
                atomic_write(staging / "replay_transitions.jsonl", b"".join(encode(row) for row in rows))
                atomic_write(staging / "replay_episodes.jsonl", encode(episode))
                atomic_write(staging / "replay_manifest.json", encode(payload.get("manifest", {})))
                atomic_write(staging / ".shared-replay.json", encode({"sha256": digest, "device_id": device, "run_id": run_id}))
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, target)  # A trainer sees a whole episode, never half a file.
            finally:
                if staging.exists():
                    for child in staging.iterdir():
                        child.unlink()
                    staging.rmdir()
            count += 1
        return count

    def compatibility(self, replay_config: dict[str, Any] | None = None) -> str:
        config = read_json(self.root / "config.json", {})
        replay = dict(config.get("replay", {}) if replay_config is None else replay_config)
        replay.pop("allow_bot_training", None)
        digest = hashlib.sha256(encode(replay))
        catalog = self.root / str(config.get("dataset", {}).get("card_catalog", "data/cards.json"))
        digest.update(encode(read_json(catalog, {})))
        for name in ("replay_learning.py", "replay.py", "replay_evaluation.py", "imitation.py", "cards.py", "battle_perception.py", "policy.py", "tactics.py", "action_feedback.py", "learned_perception.py", "action_confirmation.py"):
            path = self.root / "crbot" / name
            # Git may check the same source out with CRLF on one PC and LF on another.
            digest.update(path.read_text(encoding="utf-8-sig").encode("utf-8") if path.is_file() else b"missing")
        return digest.hexdigest()

    def publish_model(self) -> int:
        if not self.config.get("trainer"):
            return 0
        registry = read_json(self.root / "models/replay_policy/registry.json", {})
        candidates = [item for item in registry.get("candidates", [])
                      if item.get("quality_passed") is True and item.get("status") in {"champion", "shadow_pass"}]
        if not candidates:
            return 0
        compatibility = self.compatibility()
        candidates = [item for item in candidates if item.get("sync_compatibility") == compatibility]
        if not candidates:
            return 0
        champion = registry.get("champion")
        candidate = champion if champion in candidates else max(candidates, key=lambda item: float(item.get("created_at_unix", 0)))
        # Compatibility was stamped when training, not inferred for an old model.
        model_root = self.root / "models/replay_policy"
        source = (model_root / str(candidate["model_path"])).resolve()
        if not source.is_relative_to(model_root.resolve()) or source.suffix != ".npz" or source.stat().st_size > MAX_MODEL_BYTES:
            raise SyncError("模型路径或大小无效")
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        pointer = {"schema": SCHEMA, "trainer_device": self.config["device_id"],
                   "sha256": digest, "compatibility": compatibility, "candidate": candidate}
        pointer_path = self.checkout / "models/replay_policy.json"
        if read_json(pointer_path) == pointer:
            return 0
        atomic_write(self.checkout / "models" / (digest + ".npz"), data)
        atomic_write(pointer_path, encode(pointer))
        return 1

    def import_model(self) -> int:
        if self.config.get("trainer"):
            return 0
        pointer = read_json(self.checkout / "models/replay_policy.json")
        if not pointer:
            return 0
        if pointer.get("schema") != SCHEMA or pointer.get("trainer_device") != self.protocol()["trainer_device"]:
            raise SyncError("远端模型发布者或协议无效")
        if pointer.get("compatibility") != self.compatibility():
            return 0
        digest = str(pointer.get("sha256", ""))
        candidate = dict(pointer.get("candidate", {}))
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or candidate.get("quality_passed") is not True or candidate.get("status") not in {"champion", "shadow_pass"} or candidate.get("sync_compatibility") != pointer["compatibility"]:
            raise SyncError("远端模型未通过验证")
        source = self.checkout / "models" / (digest + ".npz")
        if source.is_symlink() or source.stat().st_size > MAX_MODEL_BYTES:
            raise SyncError("远端模型大小或类型无效")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise SyncError("模型校验失败，保留现有模型")
        model_root = self.root / "models/replay_policy"
        registry_path = model_root / "registry.json"
        registry = read_json(registry_path, {"schema_version": 1, "champion": None, "candidates": []})
        publication = hashlib.sha256(encode(pointer)).hexdigest()
        existing = next((item for item in registry.get("candidates", []) if item.get("sync_publication") == publication), None)
        replay_config = read_json(self.root / "config.json", {}).get("replay", {})
        if existing is not None:
            # An explicit local opt-in after downloading must also take effect.
            if existing.get("promoted") is True and replay_config.get("allow_bot_training") is True and registry.get("champion") != existing:
                registry["champion"] = existing
                atomic_write(registry_path, encode(registry))
                return 1
            return 0
        relative = f"candidates/shared_{digest}.npz"
        atomic_write(model_root / relative, data)
        candidate.update(model_path=relative, sync_sha256=digest, sync_publication=publication, sync_trainer=pointer["trainer_device"])
        registry.setdefault("candidates", []).append(candidate)
        if candidate.get("promoted") is True and replay_config.get("allow_bot_training") is True:
            registry["champion"] = candidate
        atomic_write(registry_path, encode(registry))
        return 1

    def sync(self) -> dict[str, Any]:
        if not self.config.get("enabled"):
            return {"enabled": False}
        with exclusive(self.state / "sync.lock"):
            try:
                self._commit()  # Recover completed local files after an interrupted push.
                self._pull()
                self.protocol()
                exported = self.export_records()
                imported = self.import_records()
                published = self.publish_model()
                received = self.import_model()
                total = sum(path.stat().st_size for folder in ("records", "models")
                            for path in (self.checkout / folder).rglob("*") if path.is_file())
                if total > MAX_CHECKOUT_BYTES:
                    raise SyncError("共享数据已超过 512 MiB，请迁移存储后再继续上传")
                self._commit()
                self._push()
                result = {"enabled": True, "time": time.time(), "exported_episodes": exported,
                          "imported_episodes": imported, "published_models": published,
                          "received_models": received, "shared_bytes": total, "error": None}
                atomic_write(self.state / "status.json", encode(result))
                return result
            except (SyncError, OSError, ValueError, KeyError, TypeError) as exc:
                atomic_write(self.state / "status.json", encode({"time": time.time(), "error": str(exc)}))
                if isinstance(exc, SyncError):
                    raise
                raise SyncError(str(exc)) from exc


def prepare_training(root: Path) -> None:
    if not training_allowed(root):
        raise SyncError("本机是采集机；回放训练由指定训练机执行")
    if settings(root).get("enabled"):
        try:
            ReplaySync(root).sync()
        except (SyncTransportError, SyncBusyError) as exc:
            # A network outage can use the last synchronized data, but a role change cannot.
            ReplaySync(root).protocol()
            print(f"[共享] 训练使用本地缓存，稍后补同步：{exc}")


def shared_training(function):
    @functools.wraps(function)
    def wrapped(project_root, *args, **kwargs):
        root = Path(project_root).resolve()
        with exclusive(root / ".training-sync/training.lock"):
            prepare_training(root)
            result = function(project_root, *args, **kwargs)
            if settings(root).get("enabled"):
                try:
                    ReplaySync(root).sync()
                except SyncError as exc:
                    print(f"[共享] 模型保存在本机，稍后补传：{exc}")
            return result
    return wrapped


class SyncWorker:
    def __init__(self, root: Path, notify: Callable[[str], None] = print, interval: float = 60):
        self.root, self.notify, self.interval = root, notify, max(5, interval)
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> "SyncWorker":
        if settings(self.root).get("enabled"):
            self.thread = threading.Thread(target=self._run, name="replay-sync", daemon=True)
            self.thread.start()
        return self

    def _run(self) -> None:
        last_error = ""
        while not self.stop.is_set():
            try:
                result = ReplaySync(self.root).sync()
                if last_error or any(result.get(key) for key in ("exported_episodes", "imported_episodes", "published_models", "received_models")):
                    self.notify(f"[共享] 上传 {result.get('exported_episodes', 0)} 局，接收 {result.get('imported_episodes', 0)} 局，模型接收 {result.get('received_models', 0)} 个")
                last_error = ""
            except (SyncError, OSError, ValueError) as exc:
                error = str(exc)
                if error != last_error:
                    self.notify(f"[共享] 本地采集继续，稍后重试：{error}")
                last_error = error
            self.wake.wait(self.interval)
            self.wake.clear()

    def close(self) -> None:
        self.stop.set()
        self.wake.set()
