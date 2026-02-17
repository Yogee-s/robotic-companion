#!/usr/bin/env python3
"""
ReSpeaker DOA Visual Compass — shows sound direction in real-time.

A simple tkinter GUI that displays a compass with an arrow pointing
toward the detected sound source direction.
"""

import sys
import os
import struct
import math
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import usb.core
    import usb.util
except ImportError:
    print("❌ pyusb not installed. Run: pip install pyusb")
    sys.exit(1)

try:
    import tkinter as tk
except ImportError:
    print("❌ tkinter not available. Run: sudo apt install python3-tk")
    sys.exit(1)


class DOACompass:
    """Tkinter compass widget showing Direction of Arrival."""

    def __init__(self, root):
        self.root = root
        self.root.title("🎤 ReSpeaker DOA Compass")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        self.size = 400
        self.center = self.size // 2
        self.radius = 160

        # Current state
        self.angle = 0
        self.vad_active = False
        self.running = True

        # Canvas
        self.canvas = tk.Canvas(
            root, width=self.size, height=self.size,
            bg="#1a1a2e", highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=(20, 5))

        # Info label
        self.info_var = tk.StringVar(value="DOA: 0°  |  Silent")
        self.info_label = tk.Label(
            root, textvariable=self.info_var,
            font=("Monospace", 16, "bold"),
            fg="#e0e0e0", bg="#1a1a2e"
        )
        self.info_label.pack(pady=(5, 5))

        # Status label
        self.status_var = tk.StringVar(value="Connecting to ReSpeaker...")
        self.status_label = tk.Label(
            root, textvariable=self.status_var,
            font=("Monospace", 10),
            fg="#888888", bg="#1a1a2e"
        )
        self.status_label.pack(pady=(0, 15))

        # Draw static compass elements
        self._draw_compass_base()

        # Dynamic elements (will be updated)
        self.arrow_line = None
        self.arrow_dot = None
        self.center_dot = None

        # Connect to ReSpeaker
        self.device = None
        self._connect()

        # Start polling
        self._poll()

        # Handle close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _connect(self):
        """Find and connect to ReSpeaker."""
        try:
            self.device = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if self.device is None:
                # Try alternate PID
                self.device = usb.core.find(idVendor=0x2886)
            if self.device:
                self.status_var.set(f"✓ Connected (PID={self.device.idProduct:#06x})")
            else:
                self.status_var.set("⚠ ReSpeaker not found — using simulated data")
        except Exception as e:
            self.status_var.set(f"⚠ USB error: {e}")

    def _draw_compass_base(self):
        """Draw the static compass background."""
        cx, cy = self.center, self.center
        r = self.radius

        # Outer ring
        self.canvas.create_oval(
            cx - r - 10, cy - r - 10, cx + r + 10, cy + r + 10,
            outline="#333355", width=2
        )

        # Inner rings
        for frac in [0.33, 0.66, 1.0]:
            rr = int(r * frac)
            self.canvas.create_oval(
                cx - rr, cy - rr, cx + rr, cy + rr,
                outline="#222244", width=1
            )

        # Direction labels and tick marks
        directions = [
            (0, "N", "#ff6b6b"), (45, "NE", "#888"),
            (90, "E", "#4ecdc4"), (135, "SE", "#888"),
            (180, "S", "#ffe66d"), (225, "SW", "#888"),
            (270, "W", "#a8e6cf"), (315, "NW", "#888"),
        ]

        for deg, label, color in directions:
            # Tick mark
            rad = math.radians(90 - deg)  # Convert to math angle (0=East, CCW)
            x1 = cx + (r + 5) * math.cos(rad)
            y1 = cy - (r + 5) * math.sin(rad)
            x2 = cx + (r + 15) * math.cos(rad)
            y2 = cy - (r + 15) * math.sin(rad)
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

            # Label
            lx = cx + (r + 30) * math.cos(rad)
            ly = cy - (r + 30) * math.sin(rad)
            font_size = 14 if len(label) == 1 else 10
            self.canvas.create_text(
                lx, ly, text=label, fill=color,
                font=("Monospace", font_size, "bold")
            )

        # 30° tick marks (for 12 mic positions)
        for deg in range(0, 360, 30):
            rad = math.radians(90 - deg)
            x1 = cx + (r - 5) * math.cos(rad)
            y1 = cy - (r - 5) * math.sin(rad)
            x2 = cx + (r + 3) * math.cos(rad)
            y2 = cy - (r + 3) * math.sin(rad)
            self.canvas.create_line(x1, y1, x2, y2, fill="#444466", width=1)

    def _read_doa(self):
        """Read DOA angle from ReSpeaker."""
        if self.device is None:
            return 0, False

        angle = 0
        vad = False

        try:
            # Read DOA: tuning.py protocol
            result = self.device.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, 0x00C0, 21, 8, 5000,
            )
            if len(bytes(result)) >= 8:
                angle = struct.unpack("<ii", bytes(result)[:8])[0] % 360
        except Exception:
            pass

        try:
            # Read SPEECHDETECTED
            result = self.device.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, 0x80 | 22 | 0x40, 19, 8, 5000,
            )
            if len(bytes(result)) >= 8:
                vad = struct.unpack("<ii", bytes(result)[:8])[0] > 0
        except Exception:
            pass

        return angle, vad

    def _update_arrow(self, angle, vad_active):
        """Redraw the DOA arrow."""
        cx, cy = self.center, self.center

        # Remove old arrow
        if self.arrow_line:
            self.canvas.delete(self.arrow_line)
        if self.arrow_dot:
            self.canvas.delete(self.arrow_dot)
        if self.center_dot:
            self.canvas.delete(self.center_dot)

        # Arrow color depends on VAD
        color = "#ff4444" if vad_active else "#4488ff"
        glow = "#ff6666" if vad_active else "#6699ff"

        # Calculate arrow endpoint
        rad = math.radians(90 - angle)
        arrow_len = self.radius * 0.85
        ex = cx + arrow_len * math.cos(rad)
        ey = cy - arrow_len * math.sin(rad)

        # Draw arrow line
        self.arrow_line = self.canvas.create_line(
            cx, cy, ex, ey,
            fill=color, width=4, arrow=tk.LAST, arrowshape=(16, 20, 8)
        )

        # Draw dot at arrow tip
        dot_r = 8
        self.arrow_dot = self.canvas.create_oval(
            ex - dot_r, ey - dot_r, ex + dot_r, ey + dot_r,
            fill=glow, outline=color, width=2
        )

        # Center dot
        cr = 6
        self.center_dot = self.canvas.create_oval(
            cx - cr, cy - cr, cx + cr, cy + cr,
            fill="#555577", outline="#7777aa", width=2
        )

        # Update info label
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        dir_idx = round(angle / 45) % 8
        dir_name = directions[dir_idx]
        vad_str = "🟢 Speaking" if vad_active else "⚪ Silent"
        self.info_var.set(f"DOA: {angle:3d}° {dir_name}  |  {vad_str}")

    def _poll(self):
        """Poll DOA and update display."""
        if not self.running:
            return

        angle, vad = self._read_doa()
        self.angle = angle
        self.vad_active = vad
        self._update_arrow(angle, vad)

        # Schedule next poll (100ms = 10Hz)
        self.root.after(100, self._poll)

    def _on_close(self):
        """Handle window close."""
        self.running = False
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DOACompass(root)
    root.mainloop()


if __name__ == "__main__":
    main()
