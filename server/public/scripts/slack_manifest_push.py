"""Push ``slack/manifest.dev.yaml`` to the Slack app through the manifest API.

Needs an *app configuration token* (api.slack.com/apps → "Your App
Configuration Tokens"), not the bot token, in ``SLACK_CONFIG_TOKEN``, and the
app id in ``SLACK_APP_ID``. Adding an event whose scope the installed token
already holds (``message.channels`` needs ``channels:history``) takes effect
without a reinstall; a scope change still needs one.

    uv run python scripts/slack_manifest_push.py [--check]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from slack_sdk import WebClient

MANIFEST = Path(__file__).resolve().parents[3] / "slack" / "manifest.dev.yaml"


def main(check_only: bool) -> int:
    """Validate the manifest, then update the app unless ``--check`` was given."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token, app_id = os.getenv("SLACK_CONFIG_TOKEN"), os.getenv("SLACK_APP_ID")
    if not token or not app_id:
        print("set SLACK_CONFIG_TOKEN and SLACK_APP_ID in .env", file=sys.stderr)
        return 2
    manifest = json.dumps(yaml.safe_load(MANIFEST.read_text()))
    client = WebClient(token=token)
    verdict = client.api_call("apps.manifest.validate", json={"manifest": manifest})
    print("validate:", verdict.get("ok"), verdict.get("errors", ""))
    if check_only or not verdict.get("ok"):
        return 0 if verdict.get("ok") else 1
    result = client.api_call(
        "apps.manifest.update", json={"app_id": app_id, "manifest": manifest}
    )
    print("update:", result.get("ok"), result.get("error", ""))
    if result.get("permissions_updated"):
        print("scopes changed: reinstall the app from OAuth & Permissions")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv[1:]))
