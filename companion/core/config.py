"""Central configuration — dataclass schema over config.yaml.

One source of truth. Every subsystem reads its own section of AppConfig;
no module touches raw yaml. Invalid keys or types fail at startup, not at runtime.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)


# ── Subsystem configs ────────────────────────────────────────────────────────

@dataclass
class AppMetaConfig:
    name: str = "Companion"
    data_dir: str = "data"
    log_dir: str = "logs"
    log_level: str = "INFO"
    singlish: bool = False
    proactive_enabled: bool = False


@dataclass
class HardwareConfig:
    mic_name_hint: str = "ReSpeaker"
    speaker_name_hint: str = ""
    camera_sensor_id: int = 0
    screen_serial_port: str = "/dev/ttyUSB0"
    screen_serial_baud: int = 115200


@dataclass
class LLMConfig:
    model: str = "gemma-4-e2b"
    model_paths: dict[str, str] = field(default_factory=lambda: {
        "gemma-4-e2b": "models/gemma-4-e2b-it-q4_k_m.gguf",
        "gemma-4-e4b": "models/gemma-4-e4b-it-q4_k_m.gguf",
    })
    n_gpu_layers: int = -1
    context_length: int = 2048
    max_tokens: int = 120
    # Optional vision projector GGUF. When set and the file exists, the
    # model is loaded with a multimodal chat handler so the same LLM can
    # also answer questions about images — one model serving both chat
    # and VLM duties (saves the ~3 GB Moondream would otherwise eat).
    mmproj_path: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    system_prompt: str = (
        "You are a warm, curious AI companion. You see the user via camera and hear "
        "them via microphone. Keep replies to 1-3 short sentences. Finish every "
        "sentence. Acknowledge the user's mood subtly, not overtly."
    )


@dataclass
class VLMConfig:
    enabled: bool = True
    model_path: str = "models/moondream2-q4.gguf"
    mmproj_path: str = "models/moondream2-mmproj-f16.gguf"
    scene_watch_hz: float = 1.0
    max_tokens: int = 80
    # Number of layers to offload to GPU. Jetson Orin Nano's 8 GB shared
    # VRAM usually can't fit Gemma 4 + YOLO + HSEmotion + a fully-offloaded
    # Moondream at the same time. 0 = CPU only (safe but slower).
    # Set to -1 in config.yaml only if you have the headroom.
    n_gpu_layers: int = 0


@dataclass
class FunctionGemmaConfig:
    enabled: bool = True
    model_path: str = "models/function-gemma-270m-q4.gguf"
    confidence_threshold: float = 0.55


@dataclass
class MemoryConfig:
    enabled: bool = True
    chroma_dir: str = "data/chroma"
    top_k: int = 3
    max_entries_per_speaker: int = 500


@dataclass
class STTConfig:
    backend: str = "parakeet"                  # parakeet | whisper
    parakeet_model_dir: str = "models/parakeet-tdt-0.6b-v3"
    whisper_model_size: str = "base.en"
    whisper_compute_type: str = "int8"
    streaming: bool = True
    beam_size: int = 5
    language: str = "en"


@dataclass
class TTSConfig:
    engine: str = "kokoro"                     # kokoro | piper
    voice: str = "af_heart"
    speaking_rate: float = 1.1
    lang: str = "en-us"
    output_sample_rate: int = 24000
    piper_model: str = "en_US-hfc_female-medium"
    loud_fallback: bool = True                 # warn loudly if primary engine fails


@dataclass
class VADConfig:
    threshold: float = 0.5
    silence_duration_ms: int = 500
    min_speech_duration_ms: int = 250
    speech_pad_ms: int = 30


@dataclass
class EOUConfig:
    enabled: bool = True
    model_path: str = "models/eou/livekit-eou-v0.4.1-intl.onnx"
    extra_wait_ms: int = 600
    confidence_threshold: float = 0.55


@dataclass
class WakeWordConfig:
    enabled: bool = False
    phrase: str = "hey buddy"
    model_path: str = "models/wake_word/hey_buddy.tflite"
    sensitivity: float = 0.5


@dataclass
class SpeakerIDConfig:
    enabled: bool = True
    model_path: str = "models/speaker_id/titanet-l.onnx"
    speakers_file: str = "models/speakers.json"
    match_threshold: float = 0.65


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    input_device_name: str = "ReSpeaker"
    output_device_name: str = ""
    input_gain: float = 1.0                    # software gain applied to mic input; ReSpeaker UAC1.0 has no ALSA volume


@dataclass
class ReSpeakerConfig:
    vendor_id: int = 0x2886
    product_id: int = 0x0018
    led_brightness: int = 20
    doa_enabled: bool = True
    # Mounting offset in degrees, subtracted from raw DOA so 0° = body forward.
    # Calibrate from the Audio tab in the debug GUI.
    doa_offset_deg: float = 0.0


@dataclass
class VisionConfig:
    enabled: bool = True
    emotion_enabled: bool = True               # false = face detection + tracking only (skip HSEmotion inference)
    sensor_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_method: int = 0
    use_csi: bool = True
    detect_every_n_frames: int = 2
    yolo_pose_model_path: str = "models/vision/yolo26n-pose.onnx"
    face_model_path: str = "models/vision/face_detection_yunet_2023mar.onnx"
    emotion_model_path: str = "models/vision/enet_b0_8_best_afew.onnx"
    face_score_threshold: float = 0.5
    smoothing: float = 0.7
    staleness_fade_seconds: float = 3.0


@dataclass
class DisplayConfig:
    backend: str = "pygame"                    # pygame | esp32_serial
    width: int = 320                           # landscape — firmware uses rotation=1
    height: int = 240
    fullscreen: bool = False
    serial_port: str = "/dev/ttyCH341USB0"
    serial_baud: int = 115200
    auto_dismiss_seconds: float = 4.0


@dataclass
class EngagementConfig:
    """Face-gated engagement rules for continuous-mode listening.

    A VAD speech-end only triggers a turn when:
      * a face is (or was recently) visible, AND
      * the voiced speech lasted at least `min_speech_ms`, AND
      * the DOA angle (if reliable) lines up with the face direction.
    """
    require_face: bool = True
    face_lookback_ms: int = 2000
    doa_face_concordance_deg: float = 20.0
    min_speech_ms: int = 400


@dataclass
class ConversationConfig:
    mode: str = "continuous"                   # continuous | ptt | wake_word
    max_history: int = 6
    verbosity: str = "normal"                  # brief | normal | detailed
    log_conversations: bool = True
    log_directory: str = "logs"
    allow_interruption: bool = True
    idle_timeout_s: float = 12.0               # soft-idle after this much silence
    engagement: EngagementConfig = field(default_factory=EngagementConfig)


@dataclass
class MotorConfig:
    """Head-tracking motor module (two ST3215 servos in differential bevel gear).

    Most fields are set by the calibration wizard (companion/ui/calibration_window.py).
    Kinematics convention: gear_ratio = crown_teeth / pinion_teeth (e.g. 40/20 = 2.0).
    """
    enabled: bool = False                          # master toggle; main.py skips init if false
    sim_only: bool = True                          # force SimulatedBus even when enabled
    port: str = "/dev/ttyUSB0"
    baudrate: int = 1000000
    sync_write: bool = True                        # GroupSyncWrite for coordinated L/R motion

    left_servo_id: int = 1
    right_servo_id: int = 2

    # Calibration outputs
    left_zero_tick: int = 2048
    right_zero_tick: int = 2048
    left_direction: int = 1                        # ±1, per-motor sign
    right_direction: int = -1
    gear_ratio_nominal: float = 2.0                # CAD: crown_teeth/pinion_teeth (40/20)
    gear_ratio_measured: float = 2.0               # overwritten by empirical measurement
    backlash_deg: float = 1.0
    invert_pan: bool = False                       # axis-level sign flips (separate from per-motor)
    invert_tilt: bool = False

    # Soft limits in head frame, degrees
    pan_limits_deg: list[float] = field(default_factory=lambda: [-90.0, 90.0])
    tilt_limits_deg: list[float] = field(default_factory=lambda: [-30.0, 30.0])

    # Motion + safety
    max_speed_ticks_per_s: int = 2000
    max_acceleration: int = 50
    torque_limit: int = 800                        # 0-1000 (ST3215 register)
    max_temperature_c: float = 65.0
    home_on_startup: bool = False
    poll_hz: float = 30.0

    # Stall detection — if a motor fails to reach its goal for stall_timeout_s,
    # the controller stops pushing and holds current position (prevents forcing
    # through mechanical stops / obstructions / a pinched cable)
    stall_detect: bool = True
    stall_position_error_ticks: int = 30     # ~2.6° of motor shaft, ~1.3° of head
    stall_timeout_s: float = 1.5


@dataclass
class RuntimeConfig:
    """Cross-cutting runtime knobs (GPU, thread pools, tick rates)."""
    vram_headroom_mb: int = 1500             # readiness refuses loads that cross this
    onnx_cuda_enabled: bool = True           # force-disable to debug with CPU-only
    turn_workers: int = 2                    # ThreadPoolExecutor size for turn work
    behavior_tick_hz: float = 20.0
    health_tick_hz: float = 1.0
    telemetry_ring_size: int = 100


@dataclass
class GUIConfig:
    window_width: int = 1280
    window_height: int = 820
    font_size: int = 13
    theme: str = "catppuccin_mocha"
    show_system_monitor: bool = True
    show_doa_visualization: bool = True
    doa_update_interval_ms: int = 100
    system_monitor_interval_ms: int = 2000


# ── Top-level ────────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    app: AppMetaConfig = field(default_factory=AppMetaConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    function_gemma: FunctionGemmaConfig = field(default_factory=FunctionGemmaConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    eou: EOUConfig = field(default_factory=EOUConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    speaker_id: SpeakerIDConfig = field(default_factory=SpeakerIDConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    respeaker: ReSpeakerConfig = field(default_factory=ReSpeakerConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    project_root: str = ""

    def llm_model_path(self) -> str:
        rel = self.llm.model_paths.get(self.llm.model)
        if not rel:
            raise ValueError(f"Unknown LLM model key: {self.llm.model}")
        return self._abspath(rel)

    def abspath(self, rel_or_abs: str) -> str:
        return self._abspath(rel_or_abs)

    def _abspath(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.project_root, p)


# ── Loader ───────────────────────────────────────────────────────────────────

def _resolve_type(raw_type: Any):
    """Given a field type (may be a forward-reference string under
    `from __future__ import annotations`), try to resolve it to a class."""
    if isinstance(raw_type, str):
        return globals().get(raw_type)
    return raw_type


def _coerce(cls, data: Any):
    """Recursively build a dataclass from a dict, leaving unknown keys with a warning."""
    if data is None:
        return cls()
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        log.warning(f"Expected dict for {cls.__name__}, got {type(data).__name__}; using defaults")
        return cls()

    kwargs: dict[str, Any] = {}
    field_map = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in field_map:
            log.warning(f"Unknown config key: {cls.__name__}.{key} (ignored)")
            continue
        f = field_map[key]
        resolved = _resolve_type(f.type)
        if resolved is not None and is_dataclass(resolved):
            kwargs[key] = _coerce(resolved, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge `overlay` into `base`. Scalars from overlay win.

    Used to stack `config.yaml` <- `config.local.yaml` <- env overrides
    without losing nested defaults.
    """
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _env_overrides(prefix: str = "COMPANION_") -> dict:
    """Translate `COMPANION_<SECTION>_<KEY>=value` env vars into a nested dict.

    Example: `COMPANION_LLM_MAX_TOKENS=200` becomes `{"llm": {"max_tokens": "200"}}`.
    String values are coerced by the dataclass field type at `_coerce` time.
    """
    out: dict = {}
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        parts = k[len(prefix):].lower().split("_")
        if len(parts) < 2:
            continue
        section, *rest = parts
        key = "_".join(rest)
        out.setdefault(section, {})[key] = _auto_cast(v)
    return out


def _auto_cast(s: str):
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load layered config: `config.yaml` < `config.local.yaml` < env vars.

    `config.local.yaml` (gitignored, optional) lets each device override
    the checked-in defaults without touching `config.yaml` — useful when
    dev vs production Jetsons have different camera indices or model
    paths. Environment variables override everything and are handy for
    one-off runs.
    """
    if path is None:
        here = Path(__file__).resolve().parents[2]
        path = str(here / "config.yaml")
    project_root = str(Path(path).resolve().parent)

    merged: dict = {}
    if os.path.exists(path):
        with open(path, "r") as fh:
            merged = yaml.safe_load(fh) or {}
        log.info("Config loaded from %s", path)
    else:
        log.warning("Config not found at %s; using defaults", path)

    local_path = os.path.join(project_root, "config.local.yaml")
    if os.path.exists(local_path):
        with open(local_path, "r") as fh:
            local = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, local)
        log.info("Layered config.local.yaml applied")

    env_overlay = _env_overrides()
    if env_overlay:
        merged = _deep_merge(merged, env_overlay)
        log.info("Env overrides applied: keys=%s", list(env_overlay.keys()))

    cfg = _coerce(AppConfig, merged)
    cfg.project_root = project_root
    return cfg
