# Content Navigation Rules

This repository is a Jekyll/Chirpy blog. The published site reads posts from `_posts`.

## Source Tree

- `til/` groups learning notes by source and topic.
- `projects/` groups project case studies by project name.
- `studies/` groups independent research notes by topic.
- Source-folder paths are for people and AI agents to navigate. A path must not exceed three directories below `_posts`.
- Keep `course-note`, `practice`, `tech-note`, and `reference-note` as front matter metadata; do not make them source folders.

## Editing Rules

- Preserve post filenames, front matter categories, tags, dates, titles, asset paths, and the global `/posts/:title/` permalink unless a task explicitly changes them.
- Use lowercase ASCII kebab-case for any new source folder.
- Use `{% post_url relative/path/to/date-slug %}` for internal post links. The path is relative to `_posts` and excludes `.md`.
- Do not introduce Obsidian `[[wiki links]]` into published posts.

## Recovery Boundary

- `backups/posts-flat-2026-09-20/` is the preserved pre-reorganization Markdown snapshot.
- Do not edit, move, or delete the backup during routine content work.
- `backups` and this `AGENTS.md` are excluded from Jekyll output.
- Run `python tools/reorganize_posts.py --repo-root . --backup-dir backups/posts-flat-2026-09-20 --verify` before and after broad post-structure work.
