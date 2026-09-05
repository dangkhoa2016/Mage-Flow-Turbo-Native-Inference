from __future__ import annotations

import argparse
import datetime
import re
import subprocess

EXPECTED_NAME = "Đăng Khoa"
EXPECTED_EMAIL = "i.am@dangkhoa.dev"
_FORBIDDEN_SUBJECT = re.compile(
    r"(?i)\b(phase[- ]?[a-z0-9]+|corrective|forensic|opencode|codex|handoff|reconstruction|temporary|debug)\b"
)
_MODEL_EXT = (".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".bin")


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def verify_history(repo: str, branch: str) -> tuple[list[str], int]:
    errors: list[str] = []
    log = _git(
        repo,
        "log",
        "--reverse",
        "--format=%H%x09%an%x09%ae%x09%cn%x09%ce%x09%aI%x09%cI%x09%P%x09%s",
        branch,
    )
    prev_epoch: int | None = None
    count = 0

    for line in log.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            errors.append(f"malformed log line: {line}")
            continue

        sha, an, ae, cn, ce, ai, ci, parents, subject = parts[:9]
        count += 1

        if an != EXPECTED_NAME or ae != EXPECTED_EMAIL:
            errors.append(f"{sha}: author identity {an} <{ae}>")
        if cn != EXPECTED_NAME or ce != EXPECTED_EMAIL:
            errors.append(f"{sha}: committer identity {cn} <{ce}>")
        if ai != ci:
            errors.append(f"{sha}: author date {ai} != committer date {ci}")

        epoch = int(datetime.datetime.fromisoformat(ai).timestamp())
        if prev_epoch is not None and epoch < prev_epoch:
            errors.append(f"{sha}: non-monotonic date")
        prev_epoch = epoch

        if _FORBIDDEN_SUBJECT.search(subject):
            errors.append(f"{sha}: forbidden subject: {subject}")

        if parents.strip():
            parent = parents.split()[0]
            diff = _git(repo, "diff", "--name-only", parent, sha)
            if not diff.strip():
                errors.append(f"{sha}: empty commit")

        tree = _git(repo, "ls-tree", "-r", "--name-only", sha)
        for fname in tree.splitlines():
            if fname.endswith(_MODEL_EXT):
                errors.append(f"{sha}: model weight file in tree: {fname}")

    if count == 0:
        errors.append("history is empty")

    return errors, count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", nargs="?", default=".")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()

    errors, count = verify_history(args.repo, args.branch)
    if errors:
        print("PUBLIC_HISTORY_INVARIANTS=FAIL")
        for error in errors:
            print("  - " + error)
        return 1

    print(f"PUBLIC_HISTORY_INVARIANTS=PASS COMMITS={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
