# Preserved Flat Post Snapshot

`posts-flat-2026-09-20/` contains the exact pre-reorganization bytes for the 295 Markdown posts that were originally flat in `_posts`.

- The snapshot is excluded from Jekyll output through `_config.yml`.
- `_posts/.placeholder` remained in the live `_posts` root and is not part of the Markdown post snapshot.
- Do not edit or remove this directory during normal authoring.

## Verification

Run this from the repository root:

```powershell
python tools/reorganize_posts.py --repo-root . --backup-dir backups/posts-flat-2026-09-20 --verify
```

The command checks that every snapshot post has a destination post with unchanged front matter and no content change other than the expected `{% post_url %}` path rewrite.

## Recovery

The snapshot is retained for manual recovery only. A restore replaces the reorganized Markdown tree with the snapshot, so treat it as a deliberate destructive operation.

1. Stop after a failed verification. Do not edit the snapshot.
2. Copy any post created after the snapshot to a location outside this repository.
3. In a separate recovery copy of this repository, make `_posts` empty except for `.placeholder`.
4. Copy all 295 `*.md` files from `backups/posts-flat-2026-09-20/` into that `_posts` root without renaming them.
5. Restore the pre-migration versions of `_config.yml`, `update_categories.py`, and post-link content from the Git revision immediately before this migration, then run `bundle exec jekyll build`.

Git history is the preferred second rollback path. Keep this snapshot until its removal is explicitly authorized.
