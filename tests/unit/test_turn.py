"""Turn lifecycle — cancellation, stamping, trace serialization."""

from companion.conversation.turn import Turn


def test_turn_id_is_unique():
    ids = {Turn().turn_id for _ in range(200)}
    assert len(ids) == 200


def test_turn_cancel_is_idempotent():
    t = Turn()
    assert not t.is_cancelled
    t.cancel("one")
    t.cancel("two")  # second call must not reset the reason
    assert t.is_cancelled
    assert t.trace.interrupt_reason == "one"


def test_turn_mark_sets_timestamp_once():
    t = Turn()
    t.mark("vad_end")
    first = t.trace.t_vad_end
    assert first is not None
    # Re-marking is a no-op; we want the earliest stamp kept.
    t.mark("vad_end")
    assert t.trace.t_vad_end == first


def test_trace_as_dict_round_trips():
    t = Turn()
    t.mark("vad_end")
    d = t.trace.as_dict()
    assert d["turn_id"] == t.turn_id
    assert d["t_vad_end"] is not None
    assert d["cancelled"] is False
