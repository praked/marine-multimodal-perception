#!/usr/bin/env bash
# Runs ON the GPU node (pipe via `bash -s` with the R2 creds exported into
# the remote environment by run_night.sh). Installs a static rclone under
# /scratch0/$USER/bin (no root, no home quota) and writes the R2 remote
# config `$S/rclone.conf` (0600). Idempotent. Scratch is wiped at booking
# end, so this reruns on every fresh booking.
set -euo pipefail
S=${S:-/scratch0/$USER}
mkdir -p "$S/bin" "$S/tmp"
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}" "${R2_ACCESS_KEY_ID:?}" "${R2_SECRET_ACCESS_KEY:?}"

if [[ ! -x "$S/bin/rclone" ]]; then
  cd "$S/tmp"
  curl -fsSL -o rclone.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip
  python3 -c "import zipfile; zipfile.ZipFile('rclone.zip').extractall('.')"
  cp rclone-*-linux-amd64/rclone "$S/bin/rclone" && chmod +x "$S/bin/rclone"
  rm -rf rclone.zip rclone-*-linux-amd64
fi
umask 077
cat > "$S/rclone.conf" <<CONF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
acl = private
no_check_bucket = true
CONF
chmod 600 "$S/rclone.conf"
"$S/bin/rclone" --config "$S/rclone.conf" lsd r2:asvproject-clips --max-depth 1 >/dev/null \
  && echo "rclone $("$S/bin/rclone" version | head -1 | awk '{print $2}') ok; r2:asvproject-clips reachable" \
  || { echo "rclone cannot list r2:asvproject-clips" >&2; exit 1; }
