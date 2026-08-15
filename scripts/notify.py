"""Send a push notification when the digest has new Stage 3 papers.

Kept as two separate pieces so the delivery channel is easy to swap later:
- build_notification() turns a digest.json into a (title, body) pair, or None
  if there's nothing worth notifying about.
- send_ntfy() is the only part that knows about ntfy.sh. Swapping to email
  (or anything else) later means adding a new send_* function and changing
  the one call site in main() — build_notification() doesn't need to change.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

MAX_TITLES_SHOWN = 5


def build_notification(digest: dict, site_url: str) -> tuple[str, str] | None:
    """Build (title, body) for the notification, or None if nothing to send.

    Only skips when arXiv had nothing to fetch at all (total_papers == 0 —
    a weekend/holiday/failed fetch). A "0 relevant papers today" digest still
    notifies, so a lack of Stage 3 papers doesn't look like a silent failure.
    """
    stats = digest["metadata"]["stats"]
    if stats["total_papers"] == 0:
        return None

    passed = stats["stage3_passed"]
    titles = [p["title"] for p in digest["papers"] if p.get("max_stage") == 3]
    shown = titles[:MAX_TITLES_SHOWN]

    if passed == 0:
        lines = ["No papers passed the relevance filter today."]
    else:
        lines = [f"- {t}" for t in shown]
        if len(titles) > len(shown):
            lines.append(f"...and {len(titles) - len(shown)} more")
    if site_url:
        lines.append(f"\n{site_url}")

    title = f"ArXiv Digest: {passed} new paper{'s' if passed != 1 else ''}"
    return title, "\n".join(lines)


def send_ntfy(topic: str, title: str, body: str, click_url: str) -> None:
    """Publish a notification to an ntfy.sh topic."""
    headers = {"Title": title.encode("ascii", errors="replace").decode()}
    if click_url:
        headers["Click"] = click_url

    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}",
        data=body.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()


def main() -> None:
    digest_path = Path(sys.argv[1] if len(sys.argv) > 1 else "frontend/public/digest.json")
    site_url = os.environ.get("SITE_URL", "")
    topic = os.environ.get("NTFY_TOPIC", "")

    if not topic:
        print("NTFY_TOPIC not set, skipping notification")
        return

    digest = json.loads(digest_path.read_text())
    notification = build_notification(digest, site_url)
    if notification is None:
        print("No papers fetched today, skipping notification")
        return

    title, body = notification
    send_ntfy(topic, title, body, site_url)
    print(f"Sent notification: {title}")


if __name__ == "__main__":
    main()
