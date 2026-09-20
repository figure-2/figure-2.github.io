# Posts Reorganization Design

## Goal

Reorganize the 295 Markdown Jekyll posts into a shallow, topic-oriented `_posts` tree so people and AI agents can locate related sources without changing published post URLs or the existing category taxonomy. The tracked `_posts/.placeholder` file remains in place and is not a post.

## Decisions

- `_posts` remains the sole published post source.
- Before any file move, create `backups/posts-flat-2026-09-20/` as a byte-for-byte copy of the current 295 flat Markdown posts and exclude `backups` from Jekyll output.
- The folder tree is an authoring and code-navigation index. Front matter `categories` and `tags` remain the blog's logical metadata and are not normalized in this migration.
- Folder names use lowercase ASCII kebab-case. The maximum folder depth beneath `_posts` is three.
- `course-note`, `practice`, `tech-note`, and `reference-note` remain metadata, not folders. Material about one topic stays together.
- Filenames, dates, front matter, media paths, and the global permalink `/posts/:title/` remain unchanged.
- `{% post_url %}` references are updated to include each destination path relative to `_posts`; this avoids Jekyll's deprecated flat-path fallback after moving posts.

## Target Tree

```text
_posts/
├─ til/
│  ├─ multicampus/{git,python,pandas,algorithm,web,sql,data-analysis,machine-learning,data-engineering,projects,special}/
│  └─ upstage/{ai-literacy,data-structure-algorithm,dev-env-git-docker,network-cloud,prompt-engineering-rag,agentic-workflow,ai-service-planning,agent-architecture,llmops,projects,resources}/
├─ projects/{4-5hz,history-docent,visually-impaired-service,financial-agent,sesac-note,locallens,lumi-agent}/
└─ studies/{python,rag,ai-agent,agentic-workflow,knowledge-graph,ai-engineering,uncategorized}/
```

## Mapping Rules

| Existing category prefix | Destination |
| --- | --- |
| `1.TIL / 1-1.MULTICAMPUS_DATA_ENGINEERING / <topic>` | `_posts/til/multicampus/<topic-slug>/` |
| `1.TIL / 1-2.UPSTAGE_AI_AGENT / <topic> / <note-type>` | `_posts/til/upstage/<topic-slug>/` |
| `2.PROJECT / <project>` | `_posts/projects/<project-slug>/` |
| `3.STUDY / <topic>` | `_posts/studies/<topic-slug>/` |
| `3.STUDY` with no second category | `_posts/studies/uncategorized/` |

The migration utility owns the explicit category-to-slug map. It must fail before moving any files when a category path has no mapping or a destination filename collides.

## Recovery

1. Stop after any verification failure; do not delete the backup.
2. Restore only by copying the backed-up files back into an empty `_posts` tree, then restore the previous Git revision of configuration and links.
3. Git history remains a second rollback mechanism. The backup is retained until the user explicitly authorizes removal.

## Verification

- The backup contains the exact pre-migration bytes for 295 Markdown posts. Each reorganized post matches its backup after applying only the expected `{% post_url %}` target rewrites.
- Every post is mapped exactly once; no flat Markdown post remains in `_posts`.
- Front matter category values are unchanged for every post.
- Every `{% post_url %}` resolves during `bundle exec jekyll build`.
- `bundle exec htmlproofer _site --disable-external` passes.
- Git diff contains only the expected moves, backup, configuration, migration tooling, documentation, and post-link adjustments.

## Out of Scope

- Changing the public information architecture, category labels, tags, post titles, dates, or permalinks.
- Introducing Obsidian wiki links, an external vault, a content synchronization pipeline, or a new Jekyll collection.
- Removing the preserved flat-post backup.
