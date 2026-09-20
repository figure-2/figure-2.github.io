# SDD ledger — plan: docs/superpowers/plans/2026-09-20-posts-reorganization.md

Pre-flight: Task 1 produces `tools/reorganize_posts.py`; Task 2 consumes its dry-run, apply, and verify commands; Task 3 consumes the reorganized post tree. No interface conflict found.

Ruling: Execute in the user-designated `master` checkout rather than a worktree — the user explicitly requested the target repository be updated while retaining an original backup; cost if wrong: concurrent local changes may require a manual merge.

Ruling: Store the progress ledger under `docs/superpowers/ledgers/` — the prescribed git-ignored workspace script is unusable on this Windows checkout because its CRLF shell file fails in WSL; cost if wrong: the ledger is tracked documentation rather than disposable scratch state.

Task 1: Ruling: Verify destination content after deterministic `post_url` rewriting instead of raw SHA-256 equality — raw equality contradicts the required link-target migration; cost if wrong: the verifier could permit an unintended non-link edit only if it exactly mimics the expected normalized content.

Task 1: Ruling: Build the case-insensitive collision fixture from two distinct temporary directories — Windows cannot create two names that differ only by case in one source directory; cost if wrong: this unit test validates the planner's generic collision guard rather than a directly creatable flat-tree state.

Task 1: complete (tests: `python -m pytest tools/tests/test_reorganize_posts.py -q` → 4 passed)

Task 2: Ruling: Treat `_posts/.placeholder` as a preserved non-post file — Git tracks 296 `_posts` files, but only 295 have a Markdown extension; cost if wrong: a future non-Markdown post format would require extending the migration allowlist.

Task 3: Ruling: Exclude the generated `_site` directory from Jekyll source input — building to a temporary destination otherwise copies the tracked prior site output and collides with generated tag pages; cost if wrong: any intentional hand-authored source file under `_site` is no longer published.

Task 2: complete (dry-run: `source_posts=295 unmapped=0 collisions=0`; apply: `backup_posts=295 moved_posts=295 link_targets_rewritten=77`; verify: `content_mismatches=0 category_mismatches=0 flat_posts_remaining=0`)

Task 3: Ruling: Do not normalize tag capitalization or spelling as part of a source-tree migration — 31 existing tag slug collisions (for example, `git`/`Git` and `API`/`api`) cause Jekyll archive destination warnings but are independent of post relocation; cost if wrong: unreviewed tag merging could silently change published tag taxonomy.

Task 3: complete (focused tests: 4 passed; Jekyll temporary-destination build: exit 0; HTML-Proofer: 4,051 internal links and 295 hash-link files checked, exit 0; `git diff --check`: no whitespace errors)
