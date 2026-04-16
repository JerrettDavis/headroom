#!/usr/bin/env python3
"""Generate changelog from conventional commits."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).parent.parent

COMMIT_PATTERN = re.compile(
    r"^(feat|fix|ci|chore|perf|refactor|docs|style|test)(\(.+\))?(!)?:\s*(.+)$"
)
BREAKING_CHANGE_PATTERN = re.compile(r"^BREAKING CHANGE:\s*(.+)$", re.MULTILINE)
# Pattern to match each commit entry: subject, optional body, |, hash
# %s%n%b|%H format: "subject\nbody|hash" or "subject|hash" if no body
COMMIT_ENTRY_PATTERN = re.compile(r"^(.+?)(?:\n(.+))?\|(\w+)$", re.MULTILINE)

TYPE_LABELS: dict[str, str] = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "ci": "CI/CD",
    "chore": "Chores",
    "perf": "Performance",
    "refactor": "Refactors",
    "docs": "Documentation",
    "style": "Styles",
    "test": "Tests",
}


class ParsedCommit(NamedTuple):
    type: str
    scope: str | None
    breaking: bool
    message: str
    hash: str


def parse_commits(log_output: str) -> list[ParsedCommit]:
    """Parse git log output into structured commits."""
    commits: list[ParsedCommit] = []

    for match in COMMIT_ENTRY_PATTERN.finditer(log_output):
        subject = match.group(1)
        body = match.group(2) or ""
        commit_hash = match.group(3)

        is_breaking = bool(BREAKING_CHANGE_PATTERN.search(body))
        commit_match = COMMIT_PATTERN.match(subject)
        if commit_match:
            commit_type = commit_match.group(1)
            scope = commit_match.group(2)
            if scope:
                scope = scope[1:-1]  # Remove parentheses
            is_breaking = is_breaking or bool(commit_match.group(3))  # ! in subject
            message = commit_match.group(4)
            commits.append(
                ParsedCommit(
                    type=commit_type,
                    scope=scope,
                    breaking=is_breaking,
                    message=message,
                    hash=commit_hash,
                )
            )
    return commits


def generate_changelog(version: str, commits: list[ParsedCommit]) -> str:
    """Generate markdown changelog from parsed commits."""
    today = date.today().isoformat()
    lines = [f"## [{version}] - {today}", ""]

    # Collect breaking changes
    breaking_commits = [c for c in commits if c.breaking]
    if breaking_commits:
        lines.append("### Breaking Changes")
        for commit in breaking_commits:
            if commit.scope:
                lines.append(f"- **{commit.scope}**: {commit.message} ({commit.hash})")
            else:
                lines.append(f"- {commit.message} ({commit.hash})")
        lines.append("")

    # Group by type
    by_type: dict[str, list[ParsedCommit]] = {}
    for commit in commits:
        by_type.setdefault(commit.type, []).append(commit)

    for commit_type, label in TYPE_LABELS.items():
        type_commits = by_type.get(commit_type, [])
        if not type_commits:
            continue
        lines.append(f"### {label}")
        for commit in type_commits:
            if commit.scope:
                lines.append(f"- **{commit.scope}**: {commit.message} ({commit.hash})")
            else:
                lines.append(f"- {commit.message} ({commit.hash})")
        lines.append("")

    return "\n".join(lines) + "\n"


def run_git_log(since: str | None, cwd: Path) -> str:
    """Run git log command and return output."""
    cmd = ["git", "log", "--pretty=format:%s%n%b|%H"]
    if since:
        cmd.append(f"{since}..HEAD")
    else:
        cmd.append("HEAD")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate changelog from conventional commits")
    parser.add_argument("--version", required=True, help="Version number (e.g., 0.6.0)")
    parser.add_argument("--since", help="Starting tag (exclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout instead of writing")
    args = parser.parse_args()

    log_output = run_git_log(args.since, ROOT)
    commits = parse_commits(log_output)
    changelog = generate_changelog(args.version, commits)

    if args.dry_run:
        print(changelog)
    else:
        output_path = ROOT / ".changelog.md"
        output_path.write_text(changelog, encoding="utf-8")
        print(f"Changelog written to {output_path}")


if __name__ == "__main__":
    main()
