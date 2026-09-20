from pathlib import Path

import pytest

from tools.reorganize_posts import (
    MigrationError,
    build_plan,
    destination_for,
    plan_migration,
    rewrite_post_urls,
    verify_migration,
)


def write_post(path: Path, categories: list[str], body: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    categories_yaml = "\n".join(f"- {category}" for category in categories)
    path.write_text(
        f"---\ntitle: Test post\ncategories:\n{categories_yaml}\n---\n{body}",
        encoding="utf-8",
    )


def test_destination_for_upstage_ignores_note_type_folder() -> None:
    destination = destination_for(
        [
            "1.TIL",
            "1-2.UPSTAGE_AI_AGENT",
            "1-2-5.PROMPT_ENGINEERING_RAG",
            "COURSE_NOTE",
        ]
    )

    assert destination == Path("til/upstage/prompt-engineering-rag")


def test_destination_for_unknown_category_raises_actionable_error() -> None:
    with pytest.raises(MigrationError, match="unmapped category"):
        destination_for(["9.UNKNOWN"])


def test_destination_for_insurance_project() -> None:
    destination = destination_for(["2.PROJECT", "2-8. Insurance_PF"])

    assert destination == Path("projects/insurance-pf")


def test_plan_migration_rejects_case_insensitive_destination_collision(tmp_path: Path) -> None:
    first = tmp_path / "first" / "2026-01-01-Same.md"
    second = tmp_path / "second" / "2026-01-01-same.md"
    write_post(first, ["3.STUDY", "3-2.RAG"])
    write_post(second, ["3.STUDY", "3-2.RAG"])

    with pytest.raises(MigrationError, match="destination collision"):
        build_plan([first, second])


def test_rewrite_and_verify_preserve_backup_content_after_move(tmp_path: Path) -> None:
    posts = tmp_path / "_posts"
    source = posts / "2026-01-01-source.md"
    target = posts / "2026-01-02-target.md"
    write_post(source, ["3.STUDY", "3-2.RAG"], "{% post_url 2026-01-02-target %}\n")
    write_post(target, ["1.TIL", "1-1.MULTICAMPUS_DATA_ENGINEERING", "1-1-2.PYTHON"])

    plan = plan_migration(tmp_path)
    backup_dir = tmp_path / "backups" / "posts-flat-2026-09-20"

    from tools.reorganize_posts import apply_migration

    apply_migration(tmp_path, backup_dir, plan)
    rewrite_post_urls(tmp_path, plan)

    assert (posts / "studies" / "rag" / source.name).exists()
    rendered_source = (posts / "studies" / "rag" / source.name).read_text(encoding="utf-8")
    assert "{% post_url til/multicampus/python/2026-01-02-target %}" in rendered_source
    assert verify_migration(tmp_path, backup_dir) == []


def test_verify_can_allow_posts_created_after_migration(tmp_path: Path) -> None:
    posts = tmp_path / "_posts"
    source = posts / "2026-01-01-source.md"
    write_post(source, ["3.STUDY", "3-2.RAG"])

    plan = plan_migration(tmp_path)
    backup_dir = tmp_path / "backups" / "posts-flat-2026-09-20"

    from tools.reorganize_posts import apply_migration

    apply_migration(tmp_path, backup_dir, plan)
    write_post(posts / "studies" / "rag" / "2026-01-02-new.md", ["3.STUDY", "3-2.RAG"])

    strict_errors = verify_migration(tmp_path, backup_dir)
    assert any("post count mismatch" in error for error in strict_errors)
    assert any("unexpected destination" in error for error in strict_errors)
    assert verify_migration(tmp_path, backup_dir, allow_new_posts=True) == []
