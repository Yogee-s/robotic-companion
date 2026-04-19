"""Boot smoke test — import main, verify everything constructs.

This doesn't run the main loop (it would try to open the mic and
camera) but imports every module and constructs the objects that don't
require hardware. Catches structural regressions that unit tests miss.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)


def test_all_core_modules_import():
    import companion.behavior.engine          # noqa: F401
    import companion.behavior.tracking        # noqa: F401
    import companion.conversation.coordinator  # noqa: F401
    import companion.conversation.manager     # noqa: F401
    import companion.conversation.states      # noqa: F401
    import companion.conversation.turn        # noqa: F401
    import companion.core.config              # noqa: F401
    import companion.core.event_bus           # noqa: F401
    import companion.core.events              # noqa: F401
    import companion.core.errors              # noqa: F401
    import companion.core.gpu_arbiter         # noqa: F401
    import companion.core.health              # noqa: F401
    import companion.core.onnx_runtime        # noqa: F401
    import companion.core.readiness           # noqa: F401
    import companion.core.telemetry           # noqa: F401
    import main                               # noqa: F401


def test_default_config_loads_without_error():
    from companion.core.config import load_config

    cfg = load_config(os.path.join(_ROOT, "config.yaml"))
    assert cfg.conversation.mode in ("continuous", "ptt", "wake_word")
    assert cfg.runtime.behavior_tick_hz >= 1.0
    assert cfg.conversation.engagement.min_speech_ms >= 200


def test_main_has_expected_entrypoint():
    import main

    assert callable(getattr(main, "main", None))


def test_readiness_report_classifies_missing_files():
    """Even if models are absent, the probe should produce a structured
    report rather than crash."""
    from companion.core.config import load_config
    from companion.core.readiness import check_all

    cfg = load_config(os.path.join(_ROOT, "config.yaml"))
    rep = check_all(cfg)
    # Report must have at least one item and a well-defined boolean `ok`.
    assert rep.items
    assert isinstance(rep.ok, bool)
