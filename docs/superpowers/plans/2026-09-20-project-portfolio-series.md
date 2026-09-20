# Project Portfolio Series Implementation Plan

**Goal:** Publish a seven-post `HistoryDocent` series and a five-post `Insurance_PF` series, supported by a reusable writing prompt and verified against the projects' evidence.

**Architecture:** Each project starts with a standalone recruiter-facing landing post. Later posts expose progressively deeper evidence for technical interviewers. Source documents remain outside the blog repository; only public-safe summaries, aggregate metrics, and decision rationale are published.

**Spec:** `docs/superpowers/specs/2026-09-20-project-portfolio-series-design.md`

## Execution units

| id | depends_on | scope | acceptance_tests | risk_level | rollback_plan |
| --- | --- | --- | --- | --- | --- |
| `PORTFOLIO-SERIES-001` | none | Record evidence hierarchy, series boundaries, titles, categories, dates, and claim limits | Spec exists; 7 HistoryDocent and 5 Insurance_PF topics are distinct | Low | Revert documentation files only |
| `PORTFOLIO-SERIES-002` | `001` | Write reusable portfolio-post prompt | Prompt includes inputs, evidence hierarchy, human voice, claim boundaries, output contract, and audit checklist | Low | Remove prompt file |
| `PORTFOLIO-SERIES-003` | `001`, `002` | Write HistoryDocent posts | Seven posts exist; current metrics match final reports; Notion is identified as historical context; internal links resolve | Medium | Remove the seven new posts; existing parser post remains unchanged |
| `PORTFOLIO-SERIES-004` | `001`, `002` | Write Insurance_PF posts | Five posts exist; qualitative and quantitative results are separated; planned work is not written as completed | Medium | Remove the five new posts |
| `PORTFOLIO-SERIES-005` | `003`, `004` | Extend source-tree mapping for Insurance_PF | Mapping unit test passes; migration verify recognizes the new category | Low | Revert mapping and its unit test |
| `PORTFOLIO-SERIES-006` | `003`, `004`, `005` | Run content, build, link, and worktree review | Front matter audit, post URL check, whitespace check, unit test, Jekyll build, HTML proof, and git diff review pass | Medium | Fix only the failing post or mapping; remove generated `_site` through normal ignored build workflow if needed |

## Task 1: Write the content-control documents

**Files:**

- Create `docs/superpowers/specs/2026-09-20-project-portfolio-series-design.md`
- Create `docs/superpowers/plans/2026-09-20-project-portfolio-series.md`
- Create `docs/prompts/write-project-portfolio-series.md`

- [x] Record the approved 7+5 structure.
- [x] Define current-repository evidence as higher priority than historical Notion material.
- [x] Define claim boundaries for both projects.
- [x] Define the Korean technical-blog voice and AI-writing audit.

## Task 2: Write the HistoryDocent series

**Files:**

- Create seven Markdown posts under `_posts/projects/history-docent/`.
- Leave `2025-10-02-pj-parer-comparison.md` unchanged.

- [x] Write a standalone landing post with role, architecture, decisions, and boundaries.
- [x] Explain scope reduction from voice-tour product to evidence-first RAG backend.
- [x] Explain normalization, provenance, place catalog, and parent-child corpus.
- [x] Explain C0-C6 chunking results and failure-driven reopening conditions.
- [x] Explain dense, hybrid, reranker, and voice rewrite decisions.
- [x] Explain why advanced RAG and active routing candidates were rejected or kept in shadow.
- [x] Explain citation generation, local voice demo evidence, and the final retrospective.

## Task 3: Write the Insurance_PF series

**Files:**

- Create `_posts/projects/insurance-pf/`.
- Create five Markdown posts in the new folder.

- [x] Write a standalone landing post with role, pipeline, results, and boundaries.
- [x] Explain 11-insurer preprocessing, 6,402 chunks, breadcrumbs, tables, and residual quality issues.
- [x] Explain tokenizer selection, Sparse/Dense/Hybrid comparisons, and evaluation-size limits.
- [x] Explain QLoRA, Base/Fine-Tuned/RAG comparisons, and why answer style is not factual accuracy.
- [x] Explain negative clauses, synonym mismatch, calculation failures, and the revised evaluation plan.

## Task 4: Verify content and rendering

- [x] Run `python -m pytest tools/tests/test_reorganize_posts.py -q`.
- [x] Run `python tools/reorganize_posts.py --repo-root . --backup-dir backups/posts-flat-2026-09-20 --verify --allow-new-posts`.
- [x] Run a front matter and post-link audit for all twelve posts.
- [x] Run AI-writing pattern scans and manually review every flagged span.
- [x] Run `bundle exec jekyll build`.
- [x] Run recursive HTML proof with external checks disabled.
- [x] Run `git diff --check` and inspect `git status --short` and `git diff --stat`.

## Completion criteria

- Twelve posts and one reusable prompt are present.
- No existing project post is overwritten.
- All internal series links resolve.
- Every material metric has a local evidence source.
- The build and internal HTML checks pass.
- Remaining uncertainty is reported instead of filled with invented facts.

## Verification record

- Mapping tests: 6 passed.
- Migration integrity: 295 backed-up posts preserved; 12 new posts accepted; no content, category, or flat-post errors.
- Front matter and post links: 12 posts audited; no errors.
- AI-writing pattern scan: no configured pattern matched; headings and claim boundaries were also reviewed manually.
- Jekyll build: completed. The repository still reports pre-existing tag permalink conflicts caused by case variants in older posts; the new posts use the existing lowercase tag variants and add no new variant pair.
- HTML proof: 1,029 files and 4,151 internal links checked successfully with external checks disabled.
