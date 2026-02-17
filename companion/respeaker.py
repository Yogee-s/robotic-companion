"""
ReSpeaker USB Mic Array v3.1 Interface.

Provides Direction of Arrival (DOA) reading, LED control, and VAD status
via USB HID communication with the XMOS XVF-3800 chip.
"""

import logging
import struct
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import usb.core
    import usb.util
    HAS_USB = True
except ImportError:
    HAS_USB = False
    logger.warning("pyusb not installed. ReSpeaker features disabled.")


# XMOS parameters: name -> (resource_id, offset, type)
# From Seeed's official tuning.py for the ReSpeaker USB Mic Array
XMOS_PARAMETERS = {
    "DOAANGLE":       (21, 0, "int"),    # DOA angle 0-359
    "SPEECHDETECTED": (19, 22, "int"),   # Speech detection status
    "VOICEACTIVITY":  (19, 32, "int"),   # VAD voice activity
}

TIMEOUT = 100000


class ReSpeakerArray:
    """
    Interface to ReSpeaker USB Mic Array v3.1.

    Reads DOA (Direction of Arrival) angle and controls LEDs via USB HID.
    Uses the same USB protocol as Seeed's official tuning.py.
    Gracefully degrades if hardware is not connected.
    """

    def __init__(self, config: dict):
        self._vendor_id = config.get("vendor_id", 0x2886)
        self._product_id = config.get("product_id", 0x0018)
        self._led_brightness = config.get("led_brightness", 20)
        self._doa_enabled = config.get("doa_enabled", True)

        self._device = None
        self._lock = threading.Lock()
        self._connected = False
        self._doa_angle = 0
        self._vad_active = False
        self._running = False
        self._thread = None

        self._connect()

    def _connect(self):
        """Attempt to connect to ReSpeaker USB device."""
        if not HAS_USB:
            logger.warning("pyusb not available. ReSpeaker will use simulated data.")
            return

        try:
            self._device = usb.core.find(
                idVendor=self._vendor_id,
                idProduct=self._product_id,
            )
            if self._device is None:
                logger.warning(
                    f"ReSpeaker device (VID={self._vendor_id:#06x}, "
                    f"PID={self._product_id:#06x}) not found. Using simulated data."
                )
                return

            # Do NOT detach kernel driver — it breaks audio capture.
            # Just make sure we can do control transfers.
            self._connected = True
            logger.info("ReSpeaker connected")
        except usb.core.USBError as e:
            logger.warning(f"ReSpeaker USB error: {e}. Using simulated data.")
        except Exception as e:
            logger.warning(f"ReSpeaker connection failed: {e}. Using simulated data.")

    def _read_parameter(self, name: str) -> int:
        """
        Read a parameter from the XMOS chip via USB vendor control transfer.

        Protocol (from Seeed's tuning.py):
          - bRequest = 0
          - wValue   = 0x80 | offset | (0x40 if int type)
          - wIndex   = resource_id
          - read 8 bytes, unpack as two 32-bit ints
        """
        if not self._connected or self._device is None:
            return 0

        param = XMOS_PARAMETERS.get(name)
        if param is None:
            return 0

        resource_id, offset, ptype = param

        try:
            cmd = 0x80 | offset
            if ptype == "int":
                cmd |= 0x40

            result = self._device.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,              # bRequest
                cmd,            # wValue
                resource_id,    # wIndex
                8,              # wLength (read 8 bytes)
                TIMEOUT,
            )

            if result is not None and len(result) >= 8:
                response = struct.unpack(b"ii", bytes(result))
                if ptype == "int":
                    return response[0]
                else:
                    return response[0] * (2.0 ** response[1])
            elif result is not None and len(result) >= 4:
                return struct.unpack(b"i", bytes(result[:4]))[0]
        except usb.core.USBError as e:
            logger.debug(f"USB read error for {name}: {e}")
        except Exception as e:
            logger.debug(f"Read parameter error for {name}: {e}")
        return 0

    def _write_parameter(self, name: str, value):
        """
        Write a parameter to the XMOS chip via USB vendor control transfer.

        Protocol (from Seeed's tuning.py):
          - bRequest = 0
          - wValue   = 0
          - wIndex   = resource_id
          - payload  = struct.pack('iii', offset, int_value, 1)  for int
        """
        if not self._connected or self._device is None:
            return

        param = XMOS_PARAMETERS.get(name)
        if param is None:
            return

        resource_id, offset, ptype = param

        try:
            if ptype == "int":
                payload = struct.pack(b"iii", offset, int(value), 1)
            else:
                payload = struct.pack(b"ifi", offset, float(value), 0)

            self._device.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,              # bRequest
                0,              # wValue
                resource_id,    # wIndex
                payload,        # data
                TIMEOUT,
            )
        except usb.core.USBError as e:
            logger.debug(f"USB write error for {name}: {e}")

    def get_doa(self) -> int:
        """
        Get Direction of Arrival angle in degrees (0-359).
        0° = front, 90° = right, 180° = back, 270° = left.
        Returns simulated value if hardware is not connected.
        """
        if self._connected:
            with self._lock:
                angle = self._read_parameter("DOAANGLE")
                self._doa_angle = angle % 360
        return self._doa_angle

    def get_vad_status(self) -> bool:
        """Get on-chip Voice Activity Detection status."""
        if self._connected:
            with self._lock:
                val = self._read_parameter("SPEECHDETECTED")
                self._vad_active = val > 0
        return self._vad_active

    def set_led_color(self, pixel: int, r: int, g: int, b: int):
        """
        Set LED color for a specific pixel on the ReSpeaker array.

        Args:
            pixel: LED index (0-11 for v3.1)
            r, g, b: Color values (0-255)
        """
        if not self._connected or self._device is None:
            return

        try:
            self._device.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, 0, 0x1C + pixel,
                struct.pack("BBB", r, g, b),
                TIMEOUT,
            )
        except usb.core.USBError:
            pass

    def set_all_leds(self, r: int, g: int, b: int):
        """Set all LEDs to the same color."""
        for i in range(12):
            self.set_led_color(i, r, g, b)

    def show_doa_on_leds(self, angle: int):
        """
        Light up the LED closest to the DOA angle.

        The ReSpeaker v3.1 has 12 LEDs spaced 30° apart.
        """
        if not self._connected:
            return

        # Determine which LED is closest to the angle
        led_index = round(angle / 30.0) % 12

        # Set all LEDs dim, highlight the active one
        for i in range(12):
            if i == led_index:
                self.set_led_color(i, 0, 0, self._led_brightness)  # Blue highlight
            else:
                self.set_led_color(i, 0, 0, 2)  # Very dim blue

    def start_doa_polling(self, callback=None, interval_ms: int = 100):
        """Start a background thread that continuously polls DOA."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(callback, interval_ms / 1000.0),
            daemon=True,
        )
        self._thread.start()

    def _poll_loop(self, callback, interval: float):
        """Background DOA polling loop."""
        while self._running:
            angle = self.get_doa()
            vad = self.get_vad_status()

            if self._connected:
                self.show_doa_on_leds(angle)

            if callback:
                try:
                    callback(angle, vad)
                except Exception as e:
                    logger.error(f"DOA callback error: {e}")

            time.sleep(interval)

    def stop(self):
        """Stop polling and release USB device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._connected and self._device:
            try:
                self.set_all_leds(0, 0, 0)  # Turn off LEDs
                usb.util.dispose_resources(self._device)
            except Exception:
                pass

        self._connected = False
        logger.info("ReSpeaker stopped.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __del__(self):
        self.stop()
