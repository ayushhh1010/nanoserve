"""Publish the exported model to the Hugging Face Hub.

Two steps, deliberately separate:

    python scripts/export_model.py --github YOUR-GITHUB-USERNAME
    python scripts/publish_hf.py --repo YOUR-HF-USERNAME/nanoserve-27m --confirm

The export writes a directory and touches nothing remote. This script uploads
that directory and nothing else. Splitting them means you can open the export
and read exactly what is about to become public *before* any of it does --
which is the whole reason publishing is not a flag on the exporter.

Authentication is yours to do, once, before running this:

    hf auth login

(`huggingface-cli` was removed in huggingface_hub 1.0; `hf` replaces it.)

That stores a token in your own keychain. This script never asks for, reads, or
handles a token; it only uses a login you already performed. Never paste a
token into a script, a prompt, or a chat window -- a write token can modify or
delete every repository on your account, and pasted secrets end up in shell
history and logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Files that must exist before anything is uploaded. A half-exported
#: directory published to a public URL is worse than no directory: people
#: download it, it fails to load, and the fix requires a second public push.
REQUIRED = ["model.safetensors", "config.json", "tokenizer.json", "README.md"]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=ROOT / "export/nanoserve-27m",
                    help="directory written by scripts/export_model.py")
    ap.add_argument("--repo", required=True,
                    help="target repo id, e.g. ayush/nanoserve-27m")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private; you can flip it public later "
                         "in the web UI, but you cannot un-publish what was public")
    ap.add_argument("--confirm", action="store_true",
                    help="actually upload; without it this is a dry run")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"no export at {args.src}\n"
              f"run: python scripts/export_model.py --github YOUR-GITHUB-USERNAME",
              file=sys.stderr)
        return 1

    files = sorted(p for p in args.src.rglob("*") if p.is_file())
    missing = [f for f in REQUIRED if not (args.src / f).exists()]
    if missing:
        print(f"export at {args.src} is incomplete, missing: {', '.join(missing)}",
              file=sys.stderr)
        return 1

    total = sum(f.stat().st_size for f in files)
    print(f"about to publish {args.src}  ->  https://huggingface.co/{args.repo}")
    print(f"  visibility: {'private' if args.private else 'PUBLIC'}")
    print(f"  {len(files)} files, {human(total)}:")
    for f in files:
        print(f"    {f.relative_to(args.src)}  ({human(f.stat().st_size)})")

    if not args.confirm:
        print("\ndry run. Read the file list above, open README.md and check the "
              "model card says what you want it to say, then re-run with --confirm.")
        return 0

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        print("pip install huggingface-hub", file=sys.stderr)
        return 1

    api = HfApi()
    try:
        who = api.whoami()
    except Exception:  # noqa: BLE001 - any auth failure means the same thing
        print("not logged in. Run:  hf auth login\n"
              "Create a token with *write* access at "
              "https://huggingface.co/settings/tokens", file=sys.stderr)
        return 1

    print(f"\nauthenticated as {who.get('name', '?')}")
    owner = args.repo.split("/")[0]
    if owner != who.get("name") and owner not in {
        o.get("name") for o in who.get("orgs", [])
    }:
        print(f"warning: you are '{who.get('name')}' but the repo id starts with "
              f"'{owner}'. Creating this will fail unless you belong to that org.",
              file=sys.stderr)

    api.create_repo(args.repo, repo_type="model", private=args.private,
                    exist_ok=True)
    try:
        api.upload_folder(
            repo_id=args.repo, folder_path=str(args.src), repo_type="model",
            commit_message="Nanoserve 27M — trained from scratch on TinyStories",
        )
    except HfHubHTTPError as exc:
        print(f"upload failed: {exc}", file=sys.stderr)
        return 1

    print(f"\npublished: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
