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


@dataclass
class ReSpeakerConfig:
    vendor_id: int = 0x2886
    product_id: int = 0x0018
    led_brightness: int = 20
    doa_enabled: bool = True


@dataclass
class VisionConfig:
    enabled: bool = True
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
class ConversationConfig:
    mode: str = "ptt"                          # ptt | continuous | wake_word
    max_history: int = 6
    verbosity: str = "normal"                  # brief | normal | detailed
    log_conversations: bool = True
    log_directory: str = "logs"
    allow_interruption: bool = True


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


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load config.yaml into AppConfig. Missing file → defaults."""
    if path is None:
        here = Path(__file__).resolve().parents[2]
        path = str(here / "config.yaml")
    project_root = str(Path(path).resolve().parent)

    if not os.path.exists(path):
        log.warning(f"Config not found at {path}; using defaults")
        cfg = AppConfig()
        cfg.project_root = project_root
        return cfg

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = _coerce(AppConfig, raw)
    cfg.project_root = project_root
    log.info(f"Config loaded from {path}")
    return cfg
