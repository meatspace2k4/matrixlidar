#!/bin/bash
# Move the two daemons off jpark and onto nologin system accounts.
#
#   sudo deploy/migrate-to-service-accounts.sh --dry-run   # print, change nothing
#   sudo deploy/migrate-to-service-accounts.sh             # do it
#
# WHY
#
# Both daemons run as jpark today, and jpark is the only member of `sudo`.
# lidar_server.py parses newline JSON from any client that connects to
# :8628, so a bug there is reachable from anything on the LAN, and landing
# on jpark means landing one `sudo` away from root. The daemons need a
# serial port, an IR receiver, and a socket. None of that needs a human
# account and none of it needs sudo.
#
# WHAT MOVES
#
#   /home/jpark/washlet/{matrixlidar,media,venv}  ->  /opt/washlet/
#   /home/jpark/lidar_server.py                   ->  /opt/washlet/
#   /home/jpark/lidar_sessions.jsonl              ->  /var/lib/lidar-server/sessions.jsonl
#
# /home/jpark is 0700, so nothing owned by another account can traverse it.
# That is correct for a home directory and it is why the tree has to leave
# rather than have its permissions loosened.
#
# The venv has 13 scripts with `#!/home/jpark/washlet/venv/bin/python3`
# baked in. pyvenv.cfg points home at /usr/bin, so the tree itself moves
# fine; only those shebangs need rewriting.
#
# jpark keeps ownership of nothing here. To edit the code afterwards, edit
# it in /opt/washlet (root-owned) or deploy from the repo.
set -euo pipefail

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

SRC=/home/jpark
DST=/opt/washlet
run() {
  if (( DRY )); then printf '  would: %s\n' "$*"; else printf '  + %s\n' "$*"; "$@"; fi
}

(( EUID == 0 )) || { echo "must run as root" >&2; exit 1; }
[[ -d "$SRC/washlet/venv" ]] || { echo "no venv at $SRC/washlet/venv" >&2; exit 1; }

echo "== accounts =="
for u in lidar washlet; do
  if id -u "$u" >/dev/null 2>&1; then
    echo "  $u exists"
  else
    # --home-dir /nonexistent so a stray os.path.expanduser("~/...") lands
    # somewhere obviously wrong instead of somewhere plausible. The units
    # set WASHLET_MEDIA/WASHLET_ATVREMOTE explicitly for the same reason.
    run useradd --system --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin "$u"
  fi
done
run usermod -aG dialout lidar     # /dev/serial0 is root:dialout
run usermod -aG video washlet     # /dev/lirc0  is root:video

echo "== code and media -> $DST =="
run install -d -o root -g root -m 0755 "$DST"
for d in matrixlidar media venv; do
  run cp -a "$SRC/washlet/$d" "$DST/"
done
run cp -a "$SRC/lidar_server.py" "$DST/"

echo "== venv shebangs =="
# Only the bin/ scripts carry the old absolute path (13 of them). The sed
# delimiter is | and not #, because the pattern itself starts `^#!` and a
# # delimiter terminates it early: "unknown option to `s'".
if (( DRY )); then
  echo "  would rewrite $(grep -rl '^#!/home/jpark/washlet/venv' "$SRC/washlet/venv/bin/" 2>/dev/null | wc -l) shebangs under $DST/venv/bin/"
else
  grep -rl '^#!/home/jpark/washlet/venv' "$DST/venv/bin/" 2>/dev/null \
    | xargs -r sed -i "s|^#!/home/jpark/washlet/venv|#!$DST/venv|"
fi

echo "== ownership: readable by the services, writable by neither =="
run chown -R root:root "$DST"
run chmod -R go-w "$DST"

echo "== session log -> /var/lib/lidar-server =="
run install -d -o lidar -g lidar -m 0750 /var/lib/lidar-server
if [[ -f "$SRC/lidar_sessions.jsonl" ]]; then
  run cp -a "$SRC/lidar_sessions.jsonl" /var/lib/lidar-server/sessions.jsonl
  run chown lidar:lidar /var/lib/lidar-server/sessions.jsonl
fi

echo "== units =="
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for u in lidar-server washlet; do
  run install -o root -g root -m 0644 "$HERE/$u.service" "/etc/systemd/system/$u.service"
done
run systemctl daemon-reload
run systemctl restart lidar-server
run systemctl restart washlet

cat <<'VERIFY'

== verify ==
  systemctl show lidar-server washlet -p User -p MainPID
  ps -o user,cmd -p "$(systemctl show -P MainPID washlet)"      # expect washlet, not jpark
  systemd-analyze security lidar-server washlet                 # expect well under 9.0
  journalctl -u washlet -n 20 --no-pager                        # expect the ir: line
  python3 -c 'import socket;socket.create_connection(("127.0.0.1",8628),3)' && echo stream ok

Watch one real sit end to end before deleting anything under /home/jpark.
The old tree is left in place on purpose; this script copies, never moves.
VERIFY
