"""Config loader — layering + env overrides + coercion."""

import os
import tempfile

import yaml

from companion.core.config import _deep_merge, _env_overrides, load_config


def test_deep_merge_prefers_overlay_for_scalars():
    base = {"a": 1, "b": {"x": 10, "y": 20}}
    overlay = {"b": {"x": 99}, "c": "new"}
    merged = _deep_merge(base, overlay)
    assert merged == {"a": 1, "b": {"x": 99, "y": 20}, "c": "new"}


def test_env_overrides_build_nested_dict(monkeypatch):
    monkeypatch.setenv("COMPANION_LLM_MAX_TOKENS", "200")
    monkeypatch.setenv("COMPANION_RUNTIME_BEHAVIOR_TICK_HZ", "25.0")
    out = _env_overrides()
    assert out["llm"]["max_tokens"] == 200
    assert out["runtime"]["behavior_tick_hz"] == 25.0


def test_load_config_applies_local_overrides():
    with tempfile.TemporaryDirectory() as d:
        yaml.safe_dump({"llm": {"max_tokens": 100}}, open(os.path.join(d, "config.yaml"), "w"))
        yaml.safe_dump(
            {"llm": {"max_tokens": 300}},
            open(os.path.join(d, "config.local.yaml"), "w"),
        )
        cfg = load_config(os.path.join(d, "config.yaml"))
        assert cfg.llm.max_tokens == 300
