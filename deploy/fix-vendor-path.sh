#!/bin/bash
# lidar_server.py line 52 hardcodes the vendor driver path into /home/jpark:
#
#   VENDOR_PATH = "/home/jpark/matrixlidar/python/raspberry"
#   sys.path.insert(0, VENDOR_PATH)
#
# ProtectHome=yes makes /home unreadable to the service account, so the
# import at line 385 fails and the unit restart-loops on
# ModuleNotFoundError: No module named 'DFRobot_matrixLidar'.
#
# This copies the vendor driver into /opt/washlet/vendor and makes the
# path an environment override instead of a literal, so the next move does
# not need a code edit.
#
#   sudo deploy/fix-vendor-path.sh
set -euo pipefail
(( EUID == 0 )) || { echo "must run as root" >&2; exit 1; }

SRC=/home/jpark/matrixlidar/python/raspberry
DST=/opt/washlet/vendor

[[ -f "$SRC/DFRobot_matrixLidar.py" ]] || { echo "no driver at $SRC" >&2; exit 1; }

echo "== vendor driver -> $DST =="
install -d -o root -g root -m 0755 "$DST"
cp -a "$SRC/." "$DST/"
chown -R root:root "$DST"
chmod -R go-w "$DST"

echo "== make VENDOR_PATH overridable =="
if grep -q '^VENDOR_PATH = os.environ.get' /opt/washlet/lidar_server.py; then
  echo "  already patched"
else
  cp -a /opt/washlet/lidar_server.py /opt/washlet/lidar_server.py.orig
  sed -i 's|^VENDOR_PATH = .*|VENDOR_PATH = os.environ.get("LIDAR_VENDOR_PATH", "/home/jpark/matrixlidar/python/raspberry")|' \
      /opt/washlet/lidar_server.py
  grep -n '^VENDOR_PATH' /opt/washlet/lidar_server.py
fi

echo "== unit env =="
mkdir -p /etc/systemd/system/lidar-server.service.d
cat > /etc/systemd/system/lidar-server.service.d/vendor.conf <<'CONF'
[Service]
Environment=LIDAR_VENDOR_PATH=/opt/washlet/vendor
CONF

systemctl daemon-reload
systemctl restart lidar-server
sleep 3
systemctl restart washlet
sleep 3

echo "== result =="
systemctl is-active lidar-server washlet
journalctl -u lidar-server -n 3 --no-pager
