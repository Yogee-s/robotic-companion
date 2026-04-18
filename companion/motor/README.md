# Head Motor Subsystem

This document describes the head motor module — hardware, software
architecture, configuration, bring-up procedure, CLI reference, and the
issues encountered during bring-up (kept here because several are
non-obvious and save future debugging time).

---

## 1. Mechanism

The robot's head is driven by a **two-motor differential bevel-gear** pan/tilt:

- Two **Feetech ST3215** smart serial servos, mirror-mounted on the left and right of the head
- Each motor drives a **20-tooth bevel pinion**
- Both pinions mesh with a single **40-tooth crown bevel gear** carried by the head
- **Gear reduction ratio** = 40 / 20 = **2.0** (one turn of the crown = two turns of the motor)

### Differential kinematics

With both motor shafts in the same mathematical frame:

| Head DOF | Motor rotation |
|---|---|
| **Tilt** (pitch, head looks up/down) | Both motors rotate the **same** direction |
| **Pan** (yaw, head looks left/right) | Motors rotate **opposite** directions |

Forward and inverse kinematics (see `kinematics.py`):

```
pitch = (θ_L + θ_R) / (2 · gear_ratio)
pan   = (θ_L − θ_R) / (2 · gear_ratio)

θ_L = gear_ratio · (pitch + pan)
θ_R = gear_ratio · (pitch − pan)
```

Sign conventions (`left_direction`, `right_direction`, `invert_pan`, `invert_tilt`) are resolved per-robot during calibration because motors are mirror-mounted and the pan/tilt physical axes depend on gear orientation.

---

## 2. Hardware

| Component | Details |
|---|---|
| Servos | Feetech ST3215 (model number 2307) |
| Encoder | Magnetic absolute, 4096 ticks / revolution (single-turn mode) |
| Communication | Asynchronous half-duplex serial, Feetech protocol (Dynamixel-derived) |
| Adapter | Waveshare bus-servo USB adapter (WCH CH343 chip, enumerates as `/dev/ttyACM0`) |
| Baud rate | 1 000 000 (default) |
| Supply voltage | 7.4–12 V external (USB does NOT power the servos) |

### Key voltage note

The adapter's USB connection only provides power for its logic chip — the ST3215 servos require their **own 7.4–12 V supply** fed directly to the bus power pins. A low or disconnected supply manifests as the servo LED staying red (undervoltage alarm) and all register writes returning `0x01` in the status byte.

---

## 3. Software architecture

```
motor/
  ├── bus.py            — ServoBus protocol + ST3215Bus (hardware) + SimulatedBus
  ├── controller.py     — HeadController (high-level pan/tilt API, polling, telemetry)
  ├── kinematics.py     — Pure-function differential bevel kinematics
  ├── calibration.py    — Calibration data classes + save_to_config_yaml
  ├── cli.py            — Command-line entry points (scan, calibrate, test, etc.)
  └── head_motor_quickstart.ipynb — Guided bring-up notebook
```

### Layering

```
   CLI / Notebook / Wizard
            │
            ▼
   HeadController  ◄───── kinematics (pure functions)
            │
            ▼
     ServoBus (ST3215Bus or SimulatedBus)
            │
            ▼
      scservo_sdk  →  PortHandler → pyserial → /dev/ttyACM0
```

- `ServoBus` is a `typing.Protocol` — `HeadController` works with either a live `ST3215Bus` or a `SimulatedBus` identically. Swapping is controlled by the `motor.sim_only` config flag.
- All per-servo I/O is encapsulated in `bus.py`. Higher layers never touch encoder ticks directly; they work in head-frame degrees.
- `HeadController` runs a background poll thread (`poll_hz`, default 30 Hz) that reads telemetry (position, speed, load, voltage, temperature) and publishes a `HeadState` snapshot to registered callbacks.

### Thermal and stall protection

- `HeadController` checks per-servo temperature on each poll and disables torque above `max_temperature_c` (default 65 °C).
- Stall detection (`stall_detect`, default `true`) watches position error. If either motor is stuck more than `stall_position_error_ticks` (default 30) from its commanded goal for longer than `stall_timeout_s` (default 1.5 s), the controller writes a new goal equal to the current actual position so the servo stops pushing but still holds against gravity. Relies on the firmware torque-limit as the ultimate safety backstop.

---

## 4. Configuration (`config.yaml` → `motor:`)

```yaml
motor:
  enabled: false
  sim_only: true                    # force SimulatedBus even when enabled
  port: /dev/ttyACM0
  baudrate: 1000000
  sync_write: true                  # GroupSyncWrite for coordinated L/R moves

  left_servo_id: 1
  right_servo_id: 2

  # Calibration outputs
  left_zero_tick: 2048              # encoder tick at "forward + level"
  right_zero_tick: 2048
  left_direction: 1                 # ±1, per-motor sign
  right_direction: -1
  gear_ratio_nominal: 2.0           # CAD value
  gear_ratio_measured: 2.0          # empirically measured
  backlash_deg: 1.0
  invert_pan: false                 # axis-level sign
  invert_tilt: false

  pan_limits_deg: [-90.0, 90.0]
  tilt_limits_deg: [-30.0, 30.0]

  # Motion + safety
  max_speed_ticks_per_s: 2000
  max_acceleration: 50
  torque_limit: 800                 # 0-1000 (firmware register)
  max_temperature_c: 65.0
  home_on_startup: false
  poll_hz: 30.0

  # Stall detection
  stall_detect: true
  stall_position_error_ticks: 30
  stall_timeout_s: 1.5
```

---

## 5. Bring-up procedure

**Do this once per robot, in order.** Full walk-through is in
[`head_motor_quickstart.ipynb`](head_motor_quickstart.ipynb).

| Step | Command | Purpose |
|---|---|---|
| 0 | `source companion_env/bin/activate` | Activate the project venv (scservo_sdk is installed there) |
| 1 | `python -m companion.motor.cli test --sim …` | Verify software path works in simulator, no hardware needed |
| 2 | `ls /dev/ttyACM* /dev/ttyUSB*` | Confirm adapter shows up |
| 3 | `python -m companion.motor.cli scan` | Find servos on the bus, confirm supply voltage |
| 4 | `python -m companion.motor.cli assign-id --from 1 --to 2` | One-time: new servos ship at ID 1; disconnect one, reassign the other to ID 2, reconnect. IDs persist in EEPROM. |
| 5 | `python -m companion.motor.cli set-single-turn` | Force servos into bounded single-turn position mode (0–4095 ticks). Power-cycle after. |
| 5b | `python -m companion.motor.cli reset-safety-limits` | Restore sane voltage / temperature / torque thresholds in case they were clobbered. Power-cycle after. |
| 6 | `python -m companion.motor.cli recenter` | Re-zero each servo's internal encoder so the current mechanical position becomes tick 2048. Power-cycle after. |
| 7 | `python -m companion.motor.cli dump-registers` | Confirm everything stuck (`MIN=0, MAX=4095, MODE=0`, `PRESENT_POSITION` in 0–4095 range, `PRESENT_VOLTAGE` in spec). |
| 8 | `python -m companion.motor.cli calibrate` | Run the 9-step GUI wizard. Outputs are written back to `config.yaml`. |
| 9 | `python -m companion.motor.cli test --pan 15 --tilt 0` | Sanity-check a small commanded move. |

### Calibration wizard steps (step 8 detail)

| Wizard step | What it does | Outputs |
|---|---|---|
| 1. Connect & scan | Opens port, pings servos | — |
| 2. Direction test | Jog each motor ±5° independently | `left_direction`, `right_direction` |
| 3. Rough zero | Torque off; hand-hold head at "forward + level", capture encoder ticks | `left_zero_tick`, `right_zero_tick` (rough) |
| 4. Combined sanity | Command pan/tilt presets, toggle axis inversions if needed | `invert_pan`, `invert_tilt` |
| 5. Limit discovery | Step-jog to each of 4 mechanical limits; mark each | `pan_min_raw_*`, `pan_max_raw_*`, `tilt_min_raw_*`, `tilt_max_raw_*` |
| 6. Refine zeros | Pan zero = midpoint of limits; tilt zero from phone level | Refined `left_zero_tick`, `right_zero_tick`; symmetric `pan_limits_deg`; inward-margin `tilt_limits_deg` |
| 7. Gear ratio | Command +500 ticks both motors, measure actual head pitch with phone level | `gear_ratio_measured` |
| 8. Verify | Interactive sliders + 3D preview; right-drag on preview to move head | Visual confirmation |
| 9. Save | Writes all outputs back to `config.yaml` | Persisted calibration |

---

## 6. CLI reference (`python -m companion.motor.cli …`)

| Command | Purpose |
|---|---|
| `scan` | Scan bus for servos; report voltage and temperature |
| `diag` | Brute-force sweep of baud rates × IDs when scan finds nothing |
| `raw --id <N>` | Send a raw ping packet via pyserial; diagnoses adapter vs servo issues |
| `assign-id --from <A> --to <B>` | Change a servo's ID (persists in EEPROM) |
| `set-single-turn` | Write `MIN_ANGLE_LIMIT=0`, `MAX_ANGLE_LIMIT=4095`, `MODE=0` on every servo found |
| `recenter` | Feetech "calibrate midpoint" trick (write 128 to TORQUE_ENABLE); head's current position becomes the new tick 2048 |
| `reset-safety-limits` | Restore default voltage / temperature / torque thresholds in EEPROM |
| `factory-reset [--full]` | Send Feetech `FACTORY_RESET` instruction; keeps ID by default |
| `dump-registers` | Read back relevant EEPROM + RAM registers for every servo found |
| `jog-test --id <N> --delta <ticks> --n <N>` | Do N small relative moves on one servo and print before/after positions — diagnostic for "servo not honoring goals" |
| `calibrate [--sim]` | Launch the 9-step GUI wizard |
| `sim` | Standalone simulator (sliders + 3D preview, no hardware) |
| `test --pan <deg> --tilt <deg>` | One-shot commanded move with readback |

---

## 7. Lessons learned & common pitfalls

These took a long time to debug; they're documented here so future work doesn't have to rediscover them.

### 7.1 `scservo_sdk` endianness

The official Feetech Python SDK requires **`protocol_end = 0`** (little-endian) for STS/SMS-series servos including the ST3215.

```python
#                        ↓ STS/SMS = 0, SCS = 1
self._packet = PacketHandler(0)
```

Passing `1` results in every 2-byte register read/write being byte-swapped. The effect is subtle because the read/write are symmetrically wrong — readback matches what was "written" — but the servo internally sees transposed values. Symptoms included:

- `MAX_ANGLE_LIMIT = 4095` (0x0FFF) written → servo internally stored 0xFF0F = 65 295 → effectively no upper limit → servo behaved in multi-turn / wheel mode despite the "bounded" setting
- Small goal writes (e.g. `goal = 57`) became `goal = 14592` internally, causing massive unwanted rotations
- `PRESENT_POSITION` readings in the 40 000–65 000 range were just byte-swapped versions of normal 0–4095 values

Reference: see the official Feetech SDK example's comment — `# SCServo bit end (STS/SMS=0, SCS=1)`.

### 7.2 Two servos with the same factory ID

New ST3215 servos ship with **ID = 1**. Connecting both to the bus at once produces a bus collision (both reply to any ping-ID-1 packet simultaneously), observed as 4-byte garbage responses. The fix is to **physically disconnect one servo**, reassign the other to ID 2, then reconnect both.

### 7.3 EEPROM writes require unlocking

Register 55 (`LOCK`) defaults to 1 (locked) on power-up. EEPROM writes (including `ID` at register 5, angle limits, mode, offset, safety limits) must be preceded by `LOCK = 0` and finished with `LOCK = 1`, or the write only updates a RAM shadow and reverts on power cycle. This was the cause of ID reassignments not persisting across power cycles until `set_id` was fixed to do the unlock/write/relock dance.

### 7.4 EEPROM commit timing

ST3215 EEPROM writes take ~10 ms to commit internally. Back-to-back writes without a pause cause the next packet to be received mid-commit and return error code `-6` (corrupt reply). All EEPROM-write helpers in `bus.py` retry on `-6` and insert a 30 ms sleep between writes.

### 7.5 Clamping `GOAL_POSITION` to 4095 is wrong in multi-turn mode

`ST3215Bus.write_goal` clamps to `[0, 4095]`. If the servo is erroneously in multi-turn mode (e.g. factory-default `MIN_ANGLE_LIMIT = MAX_ANGLE_LIMIT = 0`), writing a clamped goal causes the servo to interpret it as an **absolute multi-turn position** and unwind through multiple rotations to reach it. This was the root cause of "every small jog makes the motor spin many times" symptoms. Fixed by `set-single-turn` (forces `MIN/MAX` to `0/4095` and `MODE = 0`).

### 7.6 `GOAL_SPEED = 0` means "unlimited speed"

On Feetech servos, `GOAL_SPEED` register (46) at value 0 does **not** mean "do not move" — it means "use maximum speed, no cap." After a power cycle, this is the default. `HeadController.connect()` writes the configured speed via `_apply_motion_params`, but any code path that uses `ST3215Bus` directly (e.g. the diagnostic `jog-test`) must explicitly set `GOAL_SPEED` or motions will occur at uncontrolled full speed.

### 7.7 Voltage-limit EEPROM corruption

Observed a batch of servos with `MAX_INPUT_VOLTAGE = 80` (= 8.0 V) in EEPROM. On a normal 12 V supply this latched a persistent `0x01` overvoltage alarm in the status byte of every reply. The alarm didn't block I/O but did cause unexpected fail-safe motion when torque was enabled. `reset-safety-limits` restores the factory defaults (MAX 14 V, MIN 4 V, temperature 70 °C, torque 1000).

### 7.8 Multi-turn encoder accumulation

The ST3215 encoder is single-turn absolute (0–4095), but the `PRESENT_POSITION` register at address 56 is two bytes that can accumulate multi-turn counts while in certain modes. Once the counter is above 4095, writing any single-turn-range goal triggers a multi-rotation "unwind" back to the correct single-turn slot. The `recenter` command uses the Feetech trick of writing `128` to `TORQUE_ENABLE` — this instructs the firmware to treat the current physical position as tick 2048, resetting the multi-turn accumulator and aligning the OFFSET register accordingly.

### 7.9 Stall detection vs. calibration

`HeadController._check_stall` holds the current position if a motor hasn't reached its goal within `stall_timeout_s` (default 1.5 s). During calibration — which intentionally drives the head toward limits and tests long sweeps — this causes "motion stops halfway, press again" UX. The calibration wizard now sets `stall_detect = False` on its private config copy for the duration of the wizard; the original value is still on disk and reloaded when normal operation resumes.

### 7.10 `GOAL_TIME` synchronizes two-motor moves

For pure pan (motors moving equal but opposite amounts), small differences in mechanical load cause one motor to complete slightly before the other, producing a transient tilt during an otherwise pure-pan move. The verify-page slider uses **time-mode** motion: registers 44–45 (`GOAL_TIME`) are set to 150 ms, so both motors complete any commanded move in the same time window regardless of distance. Eliminates cross-axis jitter during drag.

---

## 8. Quick test without hardware

Every CLI path supports `--sim` (and the calibration wizard has a simulator checkbox on its first page). The simulator bus implements the `ServoBus` protocol with first-order dynamics, so `HeadController`, the wizard, and the 3D preview all work identically with no servos connected. Useful for:

- Rehearsing the wizard flow before touching hardware
- Integration-testing higher-level behaviour (expressions, gaze tracking)
- CI / unit tests

```bash
python -m companion.motor.cli calibrate --sim
python -m companion.motor.cli test --sim --pan 25 --tilt -10
python -m companion.motor.cli sim                 # standalone slider UI
```

---

## 9. Dependencies

- `feetech-servo-sdk` — PyPI package, imports as `scservo_sdk`
- `pyserial` — transitive via `feetech-servo-sdk`
- `PyQt5` — GUI wizard, 3D preview
- `pyqtgraph` (optional, with `pyqtgraph.opengl`) — 3D head model; falls back to 2D top/side indicators if unavailable
- `PyYAML` — config file round-trip (via `companion.core.config`)

---

## 10. File reference

| File | Lines (approx) | Responsibility |
|---|---|---|
| `bus.py` | ~400 | `ServoBus` protocol; `ST3215Bus` (hardware); `SimulatedBus` (in-memory dynamics) |
| `controller.py` | ~310 | `HeadController` — torque, pose commands, polling, stall detection, thermal watchdog |
| `kinematics.py` | ~100 | Pure-function differential bevel kinematics + tick/degree conversions |
| `calibration.py` | — | `CalibrationResult` dataclass; `save_to_config_yaml`; human-readable summary |
| `cli.py` | ~500 | All command-line entry points described in §6 |
| `head_motor_quickstart.ipynb` | — | Notebook walking through sections 1–7 of the bring-up |
