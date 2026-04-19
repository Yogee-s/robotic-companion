"""HealthMonitor — publishes HealthDegraded on starved mic, clears on recovery."""

import time

from companion.core.event_bus import EventBus
from companion.core.events import HealthDegraded, HealthRecovered
from companion.core.health import HealthMonitor


class _FakeAudioIn:
    def __init__(self) -> None:
        self.is_starved = False


def test_starved_mic_raises_health_event():
    bus = EventBus()
    mic = _FakeAudioIn()
    got_down: list[HealthDegraded] = []
    got_up: list[HealthRecovered] = []
    bus.subscribe(HealthDegraded, got_down.append)
    bus.subscribe(HealthRecovered, got_up.append)

    mon = HealthMonitor(event_bus=bus, tick_hz=20.0, audio_input=mic)
    mon.start()
    try:
        mic.is_starved = True
        # Wait up to ~0.5 s for the watchdog tick at 20 Hz to fire.
        for _ in range(20):
            if got_down:
                break
            time.sleep(0.05)
        assert got_down, "expected HealthDegraded after mic starvation"
        assert got_down[0].subsystem == "audio_in"

        mic.is_starved = False
        for _ in range(20):
            if got_up:
                break
            time.sleep(0.05)
        assert got_up, "expected HealthRecovered once mic is healthy again"
        assert got_up[0].subsystem == "audio_in"
    finally:
        mon.stop()
