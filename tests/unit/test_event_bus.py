"""EventBus — subscribe / publish / raising-handler isolation."""

import time
from dataclasses import dataclass

from companion.core.event_bus import EventBus


@dataclass(frozen=True)
class Ping:
    n: int


def test_sync_publish_delivers_to_all_subscribers():
    bus = EventBus()
    calls_a: list[int] = []
    calls_b: list[int] = []
    bus.subscribe(Ping, lambda e: calls_a.append(e.n))
    bus.subscribe(Ping, lambda e: calls_b.append(e.n))
    bus.publish(Ping(n=1))
    bus.publish(Ping(n=2))
    assert calls_a == [1, 2]
    assert calls_b == [1, 2]


def test_raising_subscriber_does_not_break_others():
    bus = EventBus()
    received: list[int] = []

    def boom(_ev):
        raise RuntimeError("subscriber bug")

    bus.subscribe(Ping, boom)
    bus.subscribe(Ping, lambda e: received.append(e.n))
    bus.publish(Ping(n=7))
    assert received == [7]


def test_async_publish_eventually_delivers():
    bus = EventBus(async_workers=1)
    bus.start()
    try:
        got: list[int] = []
        bus.subscribe(Ping, lambda e: got.append(e.n))
        bus.publish_async(Ping(n=42))
        # Give the worker a moment to drain.
        for _ in range(20):
            if got:
                break
            time.sleep(0.05)
        assert got == [42]
    finally:
        bus.stop()


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    calls: list[int] = []
    handler = lambda e: calls.append(e.n)  # noqa: E731
    bus.subscribe(Ping, handler)
    bus.publish(Ping(n=1))
    bus.unsubscribe(Ping, handler)
    bus.publish(Ping(n=2))
    assert calls == [1]
