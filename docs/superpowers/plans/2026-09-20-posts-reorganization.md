# Posts Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the 295 flat Markdown Jekyll posts into the approved shallow content tree with a retained, non-published flat backup and unchanged public URLs.

**Architecture:** A single migration utility derives each destination from a fixed category-path map, validates all mappings before mutation, copies the flat source to a backup, moves posts, and rewrites Liquid post references. A verification mode compares backup and destination hashes plus front matter metadata. Jekyll continues to render only `_posts`.

**Tech Stack:** Jekyll 4.4.1, Chirpy 7.5.0, Python standard library, Ruby/Bundler, GitHub Pages workflow.

**Spec:** `docs/superpowers/specs/2026-09-20-posts-reorganization-design.md`

## Global Constraints

- Preserve `backups/posts-flat-2026-09-20/` until explicit user authorization removes it.
- Exclude `backups` from Jekyll output.
- Retain all 295 Markdown filenames, front matter values, media paths, tags, categories, dates, titles, and `/posts/:title/` URLs; leave `_posts/.placeholder` in place.
- Use only the approved folder names and never create a fourth source-folder level for note type.
- Do not mutate files until dry-run validation has zero unmapped posts and zero destination collisions.

## Review Focus

- An unmapped category must stop the migration before any post move; add a test fixture with an unknown category.
- A duplicate destination filename must stop the migration before any post move; add two fixture posts with the same filename.
- Backup content must equal destination content after only the expected Liquid link-target rewrites; test normalized content comparisons.
- Liquid links to a moved target must include its `_posts`-relative destination path; test a rewritten `post_url` tag.
- Jekyll must not publish `backups`; test the configured exclude list before site build.

---

### Task 1: Create the validated migration utility

**Files:**
- Create: `tools/reorganize_posts.py`
- Create: `tools/tests/test_reorganize_posts.py`

**Interfaces:**
- Consumes: a repository root and `_posts` Markdown files.
- Produces: `plan_migration(repo_root) -> dict[Path, Path]`, `backup_posts(repo_root, backup_dir) -> None`, `verify_migration(repo_root, backup_dir) -> list[str]`.

- [x] **Step 1: Write failing mapping tests**

```python
def test_upstage_note_type_does_not_create_a_fourth_folder():
    assert destination_for(["1.TIL", "1-2.UPSTAGE_AI_AGENT", "1-2-5.PROMPT_ENGINEERING_RAG", "COURSE_NOTE"]) == Path("til/upstage/prompt-engineering-rag")

def test_unknown_category_stops_planning():
    with pytest.raises(MigrationError, match="unmapped category"):
        destination_for(["9.UNKNOWN"])
```

- [x] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tools/tests/test_reorganize_posts.py -q`

Expected: FAIL because `reorganize_posts` does not yet exist.

- [x] **Step 3: Implement mapping, dry-run, backup, move, link rewrite, and verify modes**

Implement command modes exactly as:

```text
python tools/reorganize_posts.py --repo-root . --dry-run
python tools/reorganize_posts.py --repo-root . --backup-dir backups/posts-flat-2026-09-20 --apply
python tools/reorganize_posts.py --repo-root . --backup-dir backups/posts-flat-2026-09-20 --verify
```

The `--dry-run` mode prints source count, mapped destination count, and all mappings. `--apply` refuses to run unless the destination tree is initially flat and the backup does not already contain a partial copy. `--verify` exits nonzero when a count, normalized-content, front matter category, or flat-post check fails.

- [x] **Step 4: Run focused tests**

Run: `python -m pytest tools/tests/test_reorganize_posts.py -q`

Expected: PASS.

### Task 2: Configure non-published recovery storage and execute the migration

**Files:**
- Modify: `_config.yml`
- Create: `backups/posts-flat-2026-09-20/` with 295 original Markdown files
- Modify: `_posts/**/*.md`

**Interfaces:**
- Consumes: the Task 1 migration utility.
- Produces: populated backup and the approved `_posts` tree.

- [x] **Step 1: Add the backup directory to Jekyll excludes**

Add this exact list item under `exclude` in `_config.yml`:

```yaml
  - backups
```

- [x] **Step 2: Validate the migration plan before mutation**

Run: `python tools/reorganize_posts.py --repo-root . --dry-run`

Expected: `source_posts=295`, `unmapped=0`, `collisions=0`.

- [x] **Step 3: Apply the migration and create backup**

Run: `python tools/reorganize_posts.py --repo-root . --backup-dir backups/posts-flat-2026-09-20 --apply`

Expected: `backup_posts=295`, `moved_posts=295`, `link_targets_rewritten=<reported count>`.

- [x] **Step 4: Verify post identity and metadata preservation**

Run: `python tools/reorganize_posts.py --repo-root . --backup-dir backups/posts-flat-2026-09-20 --verify`

Expected: `content_mismatches=0`, `category_mismatches=0`, `flat_posts_remaining=0`.

### Task 3: Validate rendered site and document operation for people and AI

**Files:**
- Modify: `update_categories.py`
- Create: `AGENTS.md`
- Create: `backups/README.md`

**Interfaces:**
- Consumes: the reorganized post tree and backup location.
- Produces: recursive category maintenance, documented navigation conventions, and explicit recovery instructions.

- [x] **Step 1: Update category-maintenance file discovery**

Replace the non-recursive glob with:

```python
post_files = glob.glob('_posts/**/*.md', recursive=True)
```

- [x] **Step 2: Document the content tree and recovery operation**

`AGENTS.md` must state that folders support source navigation, front matter is authoritative metadata, note types do not create folders, and `backups` must not be changed during ordinary content edits. `backups/README.md` must state the original snapshot date, excluded build status, and the exact recovery workflow from the design spec.

- [x] **Step 3: Build the production site**

Run: `bundle exec jekyll build`

Expected: exit code 0 and `_site` contains no `backups` directory.

- [x] **Step 4: Validate generated HTML recursively**

On Windows, run the bundled Ruby API so every nested HTML file is scanned:

```powershell
bundle exec ruby -e "require 'html-proofer'; HTMLProofer.check_directory('_site', { disable_external: true }).run"
```

Expected: exit code 0.

- [x] **Step 5: Inspect final change scope**

Run: `git status --short` and `git diff --check`

Expected: only expected migration files and no whitespace errors.
