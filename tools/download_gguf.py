"""Helper to download a GGUF model and update .env with the local path."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download

from api.logging_setup import get_logger

DEFAULT_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
DEFAULT_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEFAULT_ENV_KEY = "AUTO_ANALYST_LOCAL_MODEL_PATH"

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = BASE_DIR / "data" / "models"
DEFAULT_ENV_FILE = BASE_DIR / ".env"


def _write_env_value(env_path: Path, key: str, value: str) -> None:
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines: list[str] = []

    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download(repo: str, filename: str, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a GGUF model and update .env with its path."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--env-key", default=DEFAULT_ENV_KEY)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    logger = get_logger(__name__)
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir)
    env_file = Path(args.env_file)

    logger.info(
        "gguf_download_start",
        extra={
            "repo": args.repo,
            "model_filename": args.filename,
            "out_dir": str(out_dir),
        },
    )

    model_path = _download(args.repo, args.filename, out_dir)
    _write_env_value(env_file, args.env_key, str(model_path))

    logger.info(
        "gguf_download_complete",
        extra={"model_path": str(model_path), "env_file": str(env_file)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
