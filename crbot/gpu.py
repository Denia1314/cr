"""Optional CUDA acceleration for immutable policy arrays.

CRBOT_COMPUTE_DEVICE=auto (default), cpu, cuda, or cuda:N.
Model serialization and feature extraction remain portable NumPy operations.
"""
from __future__ import annotations

import logging
import os
import threading
import weakref
from collections import OrderedDict
from functools import lru_cache

import numpy as np

_cache = OrderedDict()
_lock = threading.RLock()
_MAX_BYTES = 256 * 1024 * 1024


@lru_cache(maxsize=8)
def resolve_device(requested: str = "auto") -> str:
    if requested == "cpu":
        return "cpu"
    if requested != "auto" and requested != "cuda" and not requested.startswith("cuda:"):
        raise ValueError(f"Unsupported compute device: {requested}")
    try:
        import torch
        device = "cuda:0" if requested in ("auto", "cuda") else requested
        # Check a real kernel, not only driver visibility.
        torch.ones(1, device=device).sum().item()
        logging.getLogger(__name__).warning("Policy compute: %s (%s)", device, torch.cuda.get_device_name(device))
        return device
    except (ImportError, OSError, RuntimeError, AssertionError) as exc:
        if requested != "auto":
            raise RuntimeError("CUDA unavailable; run setup_gpu.bat or select cpu") from exc
        logging.getLogger(__name__).warning("CUDA unavailable; policy compute uses CPU: %s", exc)
        return "cpu"


def vision_device(requested="auto") -> str:
    value = str(requested)
    if value.isdigit():
        value = f"cuda:{value}"
    return resolve_device(value)


def compute_device() -> str:
    return resolve_device(os.environ.get("CRBOT_COMPUTE_DEVICE", "auto"))


class HandDescriptorMatcher:
    """Exact ORB Hamming ratio counts using batched CUDA matrix multiplication."""

    def __init__(self, references, requested=None):
        self.device = resolve_device(str(requested)) if requested is not None else compute_device()
        self.calls = 0
        self.references = references
        self.bank = None
        self.lengths = None

    def counts(self, descriptors, ratio):
        if self.device == "cpu":
            return None
        if not self.references:
            return []
        import torch
        with torch.inference_mode():
            if self.bank is None:
                width = self.references[0].shape[1]
                lengths = [len(r) for r in self.references]
                if min(lengths) < 2 or any(r.dtype != np.uint8 or r.shape[1] != width for r in self.references):
                    raise ValueError("ORB references must be uint8 arrays with at least two rows and equal width")
                bank = np.zeros((len(lengths), max(lengths), width * 8), dtype=np.float16)
                for i, reference in enumerate(self.references):
                    bank[i, :len(reference)] = np.unpackbits(reference, axis=1).astype(np.float16) * 2 - 1
                self.bank = torch.as_tensor(bank, device=self.device)
                self.lengths = torch.as_tensor(lengths, device=self.device)
            signs = np.unpackbits(descriptors, axis=1).astype(np.float16) * 2 - 1
            query = torch.as_tensor(signs, device=self.device)
            counts = []
            columns = torch.arange(self.bank.shape[1], device=self.device)
            # Eight templates per batch bounds the temporary distance matrix.
            for start in range(0, len(self.references), 8):
                bank = self.bank[start:start + 8]
                distances = (query.shape[1] - torch.matmul(query, bank.transpose(1, 2))) * 0.5
                padding = columns[None, :] >= self.lengths[start:start + 8, None]
                distances.masked_fill_(padding[:, None, :], float("inf"))
                nearest = distances.topk(2, dim=-1, largest=False).values.to(torch.float64)
                counts.append((nearest[..., 0] < ratio * nearest[..., 1]).sum(dim=1))
            result = torch.cat(counts).cpu().tolist()
            self.calls += 1
            if self.calls == 1 and not getattr(self,"warming_up",False):
                print(f"[GPU] 手牌匹配已实际使用 {self.device}；模板={len(result)}")
            return result


def _tensor(array, device):
    """Cache only read-only snapshots; callers treat model arrays as immutable."""
    import torch
    key = (id(array), device)
    with _lock:
        entry = _cache.get(key)
        if entry is not None and entry[0]() is array:
            _cache.move_to_end(key)
            return entry[1]
        value = torch.as_tensor(np.ascontiguousarray(array), device=device)
        _cache[key] = (weakref.ref(array), value)
        while len(_cache) > 128 or sum(v[1].numel() * v[1].element_size() for v in _cache.values()) > _MAX_BYTES:
            _cache.popitem(last=False)
        return value


def predict(x, y, sample, neighbors, *, mode="plain", sample_weights=None,
            positive_weight=1.0, negative_weight=1.0, phase_index=None):
    """Return None for CPU, otherwise perform distance, selection and reduction on CUDA."""
    device = compute_device()
    if device == "cpu":
        return None
    import torch
    with torch.inference_mode():
        tx, ty = _tensor(x, device), _tensor(y, device)
        query = torch.as_tensor(sample, device=device)
        weights = _tensor(sample_weights, device) if sample_weights is not None else None
        if phase_index is not None:
            mask = (tx[:, phase_index] >= 0.5) == (query[phase_index] >= 0.5)
            tx, ty = tx[mask], ty[mask]
            if weights is not None:
                weights = weights[mask]
        if not len(ty):
            return 0.0 if mode == "local" else 0.5
        distances = ((tx - query.reshape(1, -1)) ** 2).sum(dim=1)
        count = min(max(1, neighbors), len(ty))
        indices = torch.topk(distances, count, largest=False).indices
        # Match NumPy's existing boundary tie selection exactly.
        boundary = distances[indices[-1]]
        if int((distances == boundary).sum().item()) > 1:
            cpu_indices = np.argpartition(distances.cpu().numpy(), count - 1)[:count]
            indices = torch.as_tensor(cpu_indices, device=device)
        values = ty[indices].to(torch.float64)
        if mode == "plain":
            return float(values.mean().item())
        if mode == "distance":
            w = 1.0 / (distances[indices].sqrt().to(torch.float64) + 0.05)
        else:
            w = torch.ones_like(values)
            if mode == "balanced":
                w = torch.where(values >= 0.5, positive_weight, negative_weight).to(torch.float64)
            if weights is not None:
                w = w * weights[indices]
        denominator = w.sum()
        if denominator.item() <= 0:
            return 0.0 if mode == "local" else 0.5
        return float(((values * w).sum() / denominator).item())


if __name__ == "__main__":
    os.environ["CRBOT_COMPUTE_DEVICE"] = "cuda:0"
    import torch
    x = np.asarray([[0, 0], [1, 1], [3, 3]], dtype=np.float32)
    result = predict(x, np.asarray([0, 1, 0], dtype=np.float32), x[1], 1)
    if result != 1.0:
        raise RuntimeError(f"CUDA policy check failed: {result}")
    print(f"CUDA policy check passed: {torch.cuda.get_device_name(0)}; torch={torch.__version__}")
