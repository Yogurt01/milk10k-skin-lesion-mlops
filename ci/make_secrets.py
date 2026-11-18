"""Create the Jenkins secret files in ci/secrets/ (`make ci-secrets`, CI-D12). Standard library only.

- jenkins_admin_password: random (like `make env` does for Grafana), created once, never changed afterwards.
- grafana_admin_password: copied from GRAFANA_ADMIN_PASSWORD in the repository's .env (the deploy stage needs the same
  value as the running stack; DCK-D8), rewritten only when it differs.

Files are mode 600 in a mode 700 folder (gitignored). Values are never printed.
"""

from __future__ import annotations

import os
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / "ci" / "secrets"


def write_private(path: Path, value: str) -> None:
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(value)
    os.replace(tmp, path)


def grafana_password(env_file: Path) -> str:
    if not env_file.is_file():
        sys.exit(f"ci-secrets: {env_file} missing (run make env first)")
    match = re.search(r"(?m)^GRAFANA_ADMIN_PASSWORD=(.*)$", env_file.read_text())
    value = match.group(1).strip() if match else ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    if not value:
        sys.exit(f"ci-secrets: GRAFANA_ADMIN_PASSWORD is empty in {env_file}")
    return value


def main() -> int:
    SECRETS.mkdir(mode=0o700, exist_ok=True)
    SECRETS.chmod(0o700)
    admin = SECRETS / "jenkins_admin_password"
    if admin.is_file() and admin.stat().st_size > 0:
        print(f"ci-secrets: {admin.relative_to(ROOT)} exists: not changed")
    else:
        write_private(admin, secrets.token_urlsafe(24))
        print(f"ci-secrets: created {admin.relative_to(ROOT)} (random; it is only in that file)")
    grafana = SECRETS / "grafana_admin_password"
    value = grafana_password(ROOT / ".env")
    if grafana.is_file() and grafana.read_text() == value:
        print(f"ci-secrets: {grafana.relative_to(ROOT)} matches .env: not changed")
    else:
        write_private(grafana, value)
        print(f"ci-secrets: {grafana.relative_to(ROOT)} copied from .env (value not shown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
