"""TelemetryRecorder — JSONL persistence + ring-buffer stats."""

import json
import os
import tempfile

from companion.core.telemetry import TelemetryRecorder, TurnTrace


def test_record_writes_one_jsonl_line_per_turn():
    with tempfile.TemporaryDirectory() as d:
        rec = TelemetryRecorder(log_dir=d)
        tr = TurnTrace(turn_id="abc")
        tr.mark("vad_end")
        rec.record(tr)

        jsonl_files = [f for f in os.listdir(d) if f.startswith("traces_")]
        assert len(jsonl_files) == 1
        path = os.path.join(d, jsonl_files[0])
        with open(path) as fh:
            lines = fh.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["turn_id"] == "abc"
        assert parsed["t_vad_end"] is not None


def test_latency_ms_between_phases():
    with tempfile.TemporaryDirectory() as d:
        rec = TelemetryRecorder(log_dir=d)
        tr = TurnTrace(turn_id="x")
        tr.t_vad_end = 100.0
        tr.t_first_audio = 100.5
        rec.record(tr)
        ms = rec.latency_ms("vad_end", "first_audio")
        assert ms == [500.0]
        assert rec.percentile(ms, 95.0) == 500.0
