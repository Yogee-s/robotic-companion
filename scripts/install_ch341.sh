#!/bin/bash
set -e

echo "Installing CH341 kernel driver..."
sudo cp /tmp/ch341ser_linux/driver/ch341.ko /lib/modules/$(uname -r)/kernel/drivers/usb/serial/
sudo depmod -a
sudo modprobe ch341

echo "Driver installed and loaded! Checking if device appeared..."
ls -la /dev/ttyUSB* || echo "Device still not showing up. Try unplugging and replugging the screen."
echo "Done!"
