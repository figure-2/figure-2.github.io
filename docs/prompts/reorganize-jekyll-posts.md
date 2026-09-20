# Jekyll Post Reorganization Prompt

Use this prompt when an AI agent performs or reviews this migration.

```text
You are migrating a Jekyll/Chirpy blog's posts into a shallow folder structure.

Objective:
Make the repository easy for both humans and AI agents to navigate while preserving the rendered blog behavior. Post content is consumed on the published site, not in an Obsidian-specific workflow.

Non-negotiable safety rules:
1. Do not delete any original post. First copy every current `_posts/*.md` file to `backups/posts-flat-2026-09-20/` and verify that both locations contain the same 295 filenames. `_posts/.placeholder` is not a post and remains in place.
2. Add `backups` to Jekyll's `exclude` list before a production build. Never let the backup be published.
3. Keep every filename, front matter value, date, title, tag, category, asset reference, and global permalink unchanged.
4. Do not introduce Obsidian wiki links such as `[[note]]`. Keep Jekyll Liquid `{% post_url %}` links and ordinary Markdown links.
5. Move a post only through the explicit mapping table. Abort before changing files if any category path is unmapped or if a destination filename already exists.
6. Update each `{% post_url %}` target to its destination path relative to `_posts`, so the link does not rely on Jekyll's deprecated flat-path lookup.
7. Never alter categories to match folders. Folders are for source navigation; front matter is the blog metadata source of truth.

Target tree:
_posts/til/multicampus/<topic>/
_posts/til/upstage/<topic>/
_posts/projects/<project>/
_posts/studies/<topic>/

Folder rules:
- Use lowercase ASCII kebab-case names.
- Do not exceed three directories below `_posts`.
- Do not create folders for course-note, practice, tech-note, or reference-note. Those are metadata and must remain co-located by subject.
- Use `studies/uncategorized/` only for a post whose Study category has no second-level topic.

Required verification, in this order:
1. Mapping dry run: every source post has one destination and there are no collisions.
2. Backup hash comparison.
3. Reorganized post count and front matter category comparison.
4. `bundle exec jekyll build`.
5. Validate every generated HTML file. On Windows, use the recursive Ruby API rather than the CLI directory argument, which can scan zero nested files:

   ```powershell
   bundle exec ruby -e "require 'html-proofer'; HTMLProofer.check_directory('_site', { disable_external: true }).run"
   ```
6. Git status and diff review.

Report:
- Source, backup, and destination counts.
- Mapping exceptions, if any.
- Exact verification commands and their results.
- The recovery command or procedure.
Do not claim a build or link check passed unless it was actually run.
```
