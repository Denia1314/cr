"""Monotonic deployment notifications, separate from the latest candidate channel."""
from __future__ import annotations

import hashlib
import re

from .training_sync import (MAX_MODEL_BYTES, SyncError, atomic_write, encode, exclusive, read_json)


def _model_bytes(model_root, entry):
    path = (model_root / str(entry.get("model_path", ""))).resolve()
    if (not path.is_relative_to(model_root.resolve()) or path.suffix != ".npz"
            or not path.is_file() or path.stat().st_size > MAX_MODEL_BYTES):
        raise SyncError("部署权重路径、类型或大小无效")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("model_sha256"):
        raise SyncError("部署权重校验失败")
    return data


def publish_deployment(sync):
    if not sync.config.get("trainer"):
        return 0
    with exclusive(sync.root / ".training-sync/model-registry.lock"):
        registry = read_json(sync.root / "models/replay_policy/registry.json", {})
        dep = registry.get("deployment")
        if not dep or dep.get("state") not in {"stable", "rolled_back"}:
            return 0
        entry = dep.get("entry")
        if dep.get("compatibility") != sync.compatibility():
            return 0
        if entry:
            data = _model_bytes(sync.root / "models/replay_policy", entry)
            atomic_write(sync.checkout / "models" / (entry["model_sha256"] + ".npz"), data)
        pointer = dict(schema=1, trainer_device=sync.config["device_id"], generation=dep["generation"],
            deployment_id=dep["id"], state=dep["state"], compatibility=dep["compatibility"],
            entry=entry, report=dep.get("report", {}), runtime_contract=dep.get("runtime_contract"))
        path = sync.checkout / "models/deployment.json"
        previous = read_json(path, {})
        if previous.get("trainer_device") == pointer["trainer_device"]:
            if previous.get("generation", 0) > pointer["generation"]:
                raise SyncError("远端部署代次更高，拒绝覆盖")
            if previous.get("generation") == pointer["generation"] and previous != pointer:
                raise SyncError("同一部署代次内容冲突")
        if previous == pointer:
            return 0
        atomic_write(path, encode(pointer))
        return 1


def import_deployment(sync):
    if sync.config.get("trainer"):
        return 0
    pointer = read_json(sync.checkout / "models/deployment.json")
    if not pointer:
        return 0
    if (pointer.get("schema") != 1 or pointer.get("trainer_device") != sync.protocol()["trainer_device"]
            or not isinstance(pointer.get("generation"), int) or pointer["generation"] < 1
            or not re.fullmatch(r"[a-f0-9]{64}", str(pointer.get("runtime_contract", "")))
            or pointer.get("state") not in {"stable", "rolled_back"}):
        raise SyncError("部署通知来源或协议无效")
    if pointer.get("compatibility") != sync.compatibility():
        return 0
    entry = pointer.get("entry")
    if entry:
        checksum = entry.get("model_sha256", "")
        if (not re.fullmatch(r"[a-f0-9]{64}", checksum)
                or entry.get("sync_compatibility") != pointer["compatibility"] or entry.get("promoted") is not True):
            raise SyncError("部署通知模型元数据无效")
        data = _model_bytes(sync.checkout / "models", {**entry, "model_path": checksum + ".npz"})
    elif pointer["state"] != "rolled_back":
        raise SyncError("空部署仅允许明确回退通知")
    with exclusive(sync.root / ".training-sync/model-registry.lock"):
        path = sync.root / "models/replay_policy/registry.json"
        registry = read_json(path, {"schema_version": 1, "champion": None, "candidates": []})
        high = registry.get("remote_deployment_watermark", {})
        checksum = hashlib.sha256(encode(pointer)).hexdigest()
        if high.get("trainer") == pointer["trainer_device"]:
            if pointer["generation"] < high.get("generation", 0):
                return 0
            if pointer["generation"] == high.get("generation"):
                if high["sha256"] != checksum:
                    raise SyncError("同代次部署通知被改写")
                return 0
        if entry:
            relative = "candidates/deployed_" + entry["model_sha256"] + ".npz"
            atomic_write(sync.root / "models/replay_policy" / relative, data)
            entry = {**entry, "model_path": relative}
            registry.setdefault("candidates", []).append(entry)
        registry["remote_deployment"] = {**pointer, "entry": entry}
        registry["remote_deployment_watermark"] = dict(trainer=pointer["trainer_device"], generation=pointer["generation"], sha256=checksum)
        atomic_write(path, encode(registry))  # Never change the live champion here.
        return 1


def acknowledge_publication(sync):
    """Only after Git push and remote-head verification succeed."""
    if not sync.config.get("trainer"):
        return
    pointer = read_json(sync.checkout / "models/deployment.json")
    if not pointer:
        return
    with exclusive(sync.root / ".training-sync/model-registry.lock"):
        path = sync.root / "models/replay_policy/registry.json"
        registry = read_json(path, {})
        dep = registry.get("deployment", {})
        if dep.get("id") == pointer.get("deployment_id") and dep.get("generation") == pointer.get("generation"):
            registry["deployment_publication"] = dict(generation=pointer["generation"], deployment_id=pointer["deployment_id"], uploaded=True)
            atomic_write(path, encode(registry))
