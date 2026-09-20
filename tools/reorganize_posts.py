#!/usr/bin/env python3
"""Safely reorganize flat Jekyll posts into the approved source tree."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


class MigrationError(RuntimeError):
    """Raised when a migration cannot be safely planned or verified."""


MULTICAMPUS_TOPICS = {
    "1-1-1.GIT": "git",
    "1-1-2.PYTHON": "python",
    "1-1-3.PANDAS": "pandas",
    "1-1-4.ALGORITHM": "algorithm",
    "1-1-5.WEB": "web",
    "1-1-6.SQL": "sql",
    "1-1-7.DATA_ANALYSIS": "data-analysis",
    "1-1-8.MACHINE_LEARNING": "machine-learning",
    "1-1-9.DATA_ENGINEERING": "data-engineering",
    "1-1-10.SPECIAL": "special",
    "1-1-11.PROJECT": "projects",
}

UPSTAGE_TOPICS = {
    "1-2-1.AI_LITERACY": "ai-literacy",
    "1-2-2.DATA_STRUCTURE_ALGORITHM": "data-structure-algorithm",
    "1-2-3.DEV_ENV_GIT_DOCKER": "dev-env-git-docker",
    "1-2-4.NETWORK_CLOUD": "network-cloud",
    "1-2-5.PROMPT_ENGINEERING_RAG": "prompt-engineering-rag",
    "1-2-6.AGENTIC_WORKFLOW": "agentic-workflow",
    "1-2-7.AI_SERVICE_PLANNING": "ai-service-planning",
    "1-2-8.AGENT_ARCHITECTURE": "agent-architecture",
    "1-2-9.LLMOPS": "llmops",
    "1-2-10.PROJECTS": "projects",
    "1-2-11.RESOURCES": "resources",
}

PROJECTS = {
    "2-1. 4.5HZ": "4-5hz",
    "2-2. History_Docent": "history-docent",
    "2-3. Visually_Impaired_Service": "visually-impaired-service",
    "2-4. Financial-Agent": "financial-agent",
    "2-5. SeSAC-Note": "sesac-note",
    "2-6. LocalLens": "locallens",
    "2-7. Lumi_agent": "lumi-agent",
    "2-8. Insurance_PF": "insurance-pf",
}

STUDY_TOPICS = {
    "3-1.PYTHON": "python",
    "3-2.RAG": "rag",
    "3-3.AI_AGENT": "ai-agent",
    "3-4.AGENTIC_WORKFLOW": "agentic-workflow",
    "3-5.KNOWLEDGE_GRAPH": "knowledge-graph",
    "3-7.AI_ENGINEERING": "ai-engineering",
}

POST_URL_PATTERN = re.compile(r"{%\s*post_url\s+([^\s%]+)\s*%}")


def front_matter_categories(path: Path) -> list[str]:
    """Return categories from a post's YAML front matter without rewriting it."""
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise MigrationError(f"non-UTF-8 post: {path}") from error

    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise MigrationError(f"front matter missing: {path}")

    try:
        closing_index = lines.index("---", 1)
    except ValueError as error:
        raise MigrationError(f"front matter not closed: {path}") from error

    front_matter = lines[1:closing_index]
    for index, line in enumerate(front_matter):
        match = re.fullmatch(r"categories:\s*(.*)", line)
        if not match:
            continue

        value = match.group(1).strip()
        if value.startswith("[") and value.endswith("]"):
            return [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
        if value:
            return [value.strip("'\"")]

        categories: list[str] = []
        for item_line in front_matter[index + 1 :]:
            item_match = re.fullmatch(r"\s*-\s*(.+?)\s*", item_line)
            if item_match:
                categories.append(item_match.group(1).strip("'\""))
                continue
            if item_line and not item_line[0].isspace():
                break
        return categories

    raise MigrationError(f"categories missing: {path}")


def destination_for(categories: list[str]) -> Path:
    """Map existing front-matter categories to the approved source directory."""
    if not categories:
        raise MigrationError("unmapped category: empty")

    if categories[0] == "1.TIL" and len(categories) >= 3:
        if categories[1] == "1-1.MULTICAMPUS_DATA_ENGINEERING":
            slug = MULTICAMPUS_TOPICS.get(categories[2])
            if slug:
                return Path("til") / "multicampus" / slug
        if categories[1] == "1-2.UPSTAGE_AI_AGENT":
            slug = UPSTAGE_TOPICS.get(categories[2])
            if slug:
                return Path("til") / "upstage" / slug

    if categories[0] == "2.PROJECT" and len(categories) >= 2:
        slug = PROJECTS.get(categories[1])
        if slug:
            return Path("projects") / slug

    if categories[0] == "3.STUDY":
        if len(categories) == 1:
            return Path("studies") / "uncategorized"
        slug = STUDY_TOPICS.get(categories[1])
        if slug:
            return Path("studies") / slug

    raise MigrationError(f"unmapped category: {' / '.join(categories)}")


def flat_post_files(posts_dir: Path) -> list[Path]:
    return sorted(path for path in posts_dir.glob("*.md") if path.is_file())


def build_plan(post_files: list[Path]) -> dict[Path, Path]:
    plan: dict[Path, Path] = {}
    destination_keys: dict[str, Path] = {}
    for source in post_files:
        destination = destination_for(front_matter_categories(source)) / source.name
        collision_key = destination.as_posix().casefold()
        if collision_key in destination_keys:
            raise MigrationError(
                f"destination collision: {destination_keys[collision_key]} and {source} -> {destination}"
            )
        destination_keys[collision_key] = source
        plan[source] = destination
    return plan


def plan_migration(repo_root: Path) -> dict[Path, Path]:
    posts_dir = repo_root / "_posts"
    if not posts_dir.is_dir():
        raise MigrationError(f"posts directory missing: {posts_dir}")
    return build_plan(flat_post_files(posts_dir))


def ensure_flat_posts_dir(posts_dir: Path) -> None:
    direct_files = set(flat_post_files(posts_dir))
    nested_files = {path for path in posts_dir.rglob("*.md") if path.is_file()} - direct_files
    if nested_files:
        raise MigrationError("_posts is already reorganized or partially migrated")


def backup_posts(repo_root: Path, backup_dir: Path) -> int:
    posts_dir = repo_root / "_posts"
    ensure_flat_posts_dir(posts_dir)
    if backup_dir.exists():
        raise MigrationError(f"backup directory already exists: {backup_dir}")

    backup_dir.mkdir(parents=True)
    for source in flat_post_files(posts_dir):
        shutil.copy2(source, backup_dir / source.name)
    return len(flat_post_files(backup_dir))


def apply_migration(repo_root: Path, backup_dir: Path, plan: dict[Path, Path]) -> int:
    posts_dir = repo_root / "_posts"
    ensure_flat_posts_dir(posts_dir)
    current_files = set(flat_post_files(posts_dir))
    if set(plan) != current_files:
        raise MigrationError("migration plan no longer matches the flat _posts tree")

    backup_posts(repo_root, backup_dir)
    for source, relative_destination in plan.items():
        destination = posts_dir / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    return len(plan)


def post_url_lookup(plan: dict[Path, Path]) -> dict[str, str]:
    return {
        source.stem: relative_destination.with_suffix("").as_posix()
        for source, relative_destination in plan.items()
    }


def rewrite_post_url_text(text: str, lookup: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        target = match.group(1).replace("\\", "/")
        normalized_target = target[:-3] if target.endswith(".md") else target
        destination = lookup.get(normalized_target)
        if not destination or destination == normalized_target:
            return match.group(0)
        return "{% post_url " + destination + " %}"

    return POST_URL_PATTERN.sub(replace, text)


def rewrite_post_urls(repo_root: Path, plan: dict[Path, Path]) -> int:
    lookup = post_url_lookup(plan)
    rewritten = 0
    for post in sorted((repo_root / "_posts").rglob("*.md")):
        raw_text = post.read_bytes().decode("utf-8")
        updated_text = rewrite_post_url_text(raw_text, lookup)
        if updated_text != raw_text:
            post.write_bytes(updated_text.encode("utf-8"))
            rewritten += 1
    return rewritten


def verify_migration(repo_root: Path, backup_dir: Path, *, allow_new_posts: bool = False) -> list[str]:
    errors: list[str] = []
    posts_dir = repo_root / "_posts"
    backup_files = flat_post_files(backup_dir)
    if not backup_files:
        return [f"backup has no Markdown posts: {backup_dir}"]

    try:
        expected_plan = build_plan(backup_files)
    except MigrationError as error:
        return [str(error)]

    destination_files = [path for path in posts_dir.rglob("*.md") if path.is_file()]
    if not allow_new_posts and len(backup_files) != len(destination_files):
        errors.append(f"post count mismatch: backup={len(backup_files)} destination={len(destination_files)}")

    flat_remaining = flat_post_files(posts_dir)
    if flat_remaining:
        errors.append(f"flat posts remaining: {len(flat_remaining)}")

    lookup = post_url_lookup(expected_plan)
    for backup_file, relative_destination in expected_plan.items():
        destination = posts_dir / relative_destination
        if not destination.exists():
            errors.append(f"missing destination: {relative_destination}")
            continue
        try:
            expected_text = rewrite_post_url_text(backup_file.read_bytes().decode("utf-8"), lookup)
            destination_text = destination.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 destination: {relative_destination}")
            continue
        if expected_text != destination_text:
            errors.append(f"content mismatch: {relative_destination}")
        if front_matter_categories(backup_file) != front_matter_categories(destination):
            errors.append(f"category mismatch: {relative_destination}")

    expected_destinations = {posts_dir / relative for relative in expected_plan.values()}
    for destination in destination_files:
        if not allow_new_posts and destination not in expected_destinations:
            errors.append(f"unexpected destination: {destination.relative_to(posts_dir)}")
    return errors


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Jekyll repository root")
    parser.add_argument("--backup-dir", default="backups/posts-flat-2026-09-20")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--apply", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--allow-new-posts",
        action="store_true",
        help="during verification, allow posts created after the migration backup",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    backup_dir = resolve_path(repo_root, args.backup_dir)
    try:
        if args.dry_run:
            plan = plan_migration(repo_root)
            for source, destination in plan.items():
                print(f"{source.name} -> {destination.as_posix()}")
            print(f"source_posts={len(plan)} unmapped=0 collisions=0")
            return 0
        if args.apply:
            plan = plan_migration(repo_root)
            backed_up = apply_migration(repo_root, backup_dir, plan)
            rewritten = rewrite_post_urls(repo_root, plan)
            print(f"backup_posts={backed_up} moved_posts={len(plan)} link_targets_rewritten={rewritten}")
            return 0

        errors = verify_migration(repo_root, backup_dir, allow_new_posts=args.allow_new_posts)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        destination_count = len([path for path in (repo_root / "_posts").rglob("*.md") if path.is_file()])
        backup_count = len(flat_post_files(backup_dir))
        print(
            f"backup_posts={backup_count} destination_posts={destination_count} "
            f"new_posts={destination_count - backup_count} "
            "content_mismatches=0 category_mismatches=0 flat_posts_remaining=0"
        )
        return 0
    except MigrationError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
