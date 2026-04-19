"""GPU arbiter — realtime preemption + mutual exclusion."""

import threading
import time

from companion.core.gpu_arbiter import GPUArbiter


def test_realtime_is_exclusive():
    arb = GPUArbiter()
    with arb.realtime():
        acquired_nested = []

        def other():
            with arb.realtime():
                acquired_nested.append(True)

        t = threading.Thread(target=other)
        t.start()
        time.sleep(0.05)
        # Other thread must still be waiting.
        assert not acquired_nested
    t.join(timeout=1.0)
    assert acquired_nested == [True]


def test_background_guard_flags_pending_realtime():
    arb = GPUArbiter()
    with arb.background() as guard:
        assert not guard.should_yield()
        # Signal a pending realtime caller by directly setting the event
        # (a real realtime() __enter__ does this).
        arb._realtime_waiting.set()
        assert guard.should_yield()


def test_affect_tag_parsing():
    """Extract-affect-tag in manager.py handles trailing + middle cases."""
    from companion.conversation.manager import _extract_affect_tag

    clean, tag = _extract_affect_tag("Great, glad you liked it! [affect: happy]")
    assert tag == "happy"
    assert "affect" not in clean

    clean2, tag2 = _extract_affect_tag("Just a reply. [affect: curious]  ")
    assert tag2 == "curious"
    assert clean2 == "Just a reply."

    clean3, tag3 = _extract_affect_tag("No tag here at all.")
    assert tag3 is None
    assert clean3 == "No tag here at all."
