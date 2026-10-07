#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <artifact-bucket> <aws-region>" >&2
  exit 2
fi

artifact_bucket="$1"
aws_region="$2"
repo_dir="/opt/income-api/repo"
venv_dir="/opt/income-api/venv"

install -d -o ubuntu -g ubuntu /opt/income-api/models
python3 -m venv "$venv_dir"
"$venv_dir/bin/pip" install --disable-pip-version-check -r "$repo_dir/deploy/requirements-serve.txt"

install -m 0644 "$repo_dir/deploy/income-api.service" /etc/systemd/system/income-api.service
printf 'ARTIFACT_BUCKET=%s\nAWS_REGION=%s\nMODEL_PATH=/opt/income-api/models/model.joblib\n' \
  "$artifact_bucket" "$aws_region" > /etc/income-api.env
chmod 0644 /etc/income-api.env
chown -R ubuntu:ubuntu /opt/income-api

systemctl daemon-reload
systemctl enable income-api
systemctl restart income-api

for attempt in $(seq 1 12); do
  if curl -fsS http://localhost:8080/healthz; then
    echo
    echo "Income API health check passed."
    exit 0
  fi
  echo "Waiting for Income API (attempt ${attempt}/12)..."
  sleep 5
done

journalctl -u income-api -n 80 --no-pager
exit 1
