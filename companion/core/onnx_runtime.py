"""Shared ONNX Runtime session factory.

Every ONNX-loading module calls `make_session(path, prefer_gpu=True)`
instead of `onnxruntime.InferenceSession(...)` directly. This:

* wires `CUDAExecutionProvider → CPUExecutionProvider` fallback in one
  place (previously each module picked its own, often defaulting to CPU);
* shares a single session-options object so every model uses the same
  CUDA arena strategy (`kSameAsRequested`) — prevents each model from
  reserving its own large CUDA arena on an 8 GB unified-memory device;
* gives Silero VAD the option to force CPU-only (its inference is <1 ms
  already; CUDA context overhead isn't worth it).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)

_HW_CUDA: Optional[bool] = None


def _cpu_only_policy() -> bool:
    return os.environ.get("COMPANION_ONNX_CPU_ONLY", "").lower() in ("1", "true", "yes")


def _hw_cuda_available() -> bool:
    """Pure hardware/runtime check — ignores the CPU-only env override.
    Used by callers that know they need GPU (e.g. Parakeet STT on 8 GB Orin:
    the global policy keeps small ONNX models on CPU, but Parakeet is the
    turn-latency bottleneck so it overrides to GPU)."""
    global _HW_CUDA
    if _HW_CUDA is not None:
        return _HW_CUDA
    try:
        import onnxruntime as ort  # type: ignore
        _HW_CUDA = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        _HW_CUDA = False
    log.info(
        "ONNX Runtime: CUDAExecutionProvider %s",
        "available" if _HW_CUDA else "unavailable",
    )
    return _HW_CUDA


def is_cuda_available() -> bool:
    """Policy check: False if the CPU-only env override is set.

    Honors COMPANION_ONNX_CPU_ONLY as a default-to-CPU policy (set by
    main.py when `runtime.onnx_cuda_enabled: false`). Individual callers
    can still force GPU with `make_session(..., force_gpu=True)`.
    """
    if _cpu_only_policy():
        if _HW_CUDA is None:
            log.info("ONNX Runtime: default policy = CPU (COMPANION_ONNX_CPU_ONLY set)")
        return False
    return _hw_cuda_available()


def make_session(
    model_path: str,
    *,
    prefer_gpu: bool = True,
    intra_op_num_threads: int = 2,
    device_id: int = 0,
    force_gpu: bool = False,
) -> Any:
    """Create an `onnxruntime.InferenceSession` with consistent settings.

    `force_gpu=True` bypasses the global CPU-only policy and uses CUDA
    whenever the hardware supports it. Used by latency-critical models
    (e.g. Parakeet) that should stay on GPU even when other ONNX models
    are kept on CPU to protect the llama.cpp allocation.
    """
    import onnxruntime as ort  # type: ignore

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_num_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cuda_ok = _hw_cuda_available() if force_gpu else is_cuda_available()
    providers: list = []
    if prefer_gpu and cuda_ok:
        providers.append((
            "CUDAExecutionProvider",
            {
                "device_id": device_id,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
            },
        ))
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    active = sess.get_providers()
    log.info("ONNX session %s providers=%s", os.path.basename(model_path), active)
    return sess
