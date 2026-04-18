"""Head tracking motor module — two ST3215 servos in a differential bevel gear.

Public surface:
    HeadController: high-level pan/tilt API with telemetry polling.
    ServoBus / ST3215Bus / SimulatedBus: hardware vs in-memory bus.
    kinematics: pure forward/inverse differential-bevel math.
"""

from companion.motor.bus import (
    ServoBus,
    ST3215Bus,
    SimulatedBus,
    ServoTelemetry,
)
from companion.motor.controller import HeadController
from companion.motor.kinematics import head_pose_to_ticks, ticks_to_head_pose

__all__ = [
    "HeadController",
    "ServoBus",
    "ST3215Bus",
    "SimulatedBus",
    "ServoTelemetry",
    "head_pose_to_ticks",
    "ticks_to_head_pose",
]
