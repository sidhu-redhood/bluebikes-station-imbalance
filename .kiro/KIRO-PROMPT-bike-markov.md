Paste this into Kiro chat (spec mode) from the repo root.

---

We're adding the Bluebikes station Markov model for our FAI project to this repo.

Read, in order:
1. `.kiro/steering/*` (tech stack, conventions, architecture). Follow them.
2. `.kiro/specs/capability-map.md`, the "Bluebikes station imbalance" section at the end.
3. `.kiro/specs/SPEC-bike-markov.md`. This is the source of truth.
4. `tasks/todo-bike-markov.md`. Work through tasks B1 to B6 in order.

A reference implementation of every file already exists (listed under each task). For each task: review the reference against the spec, fix anything that disagrees with the spec or the steering conventions, run the task's verification, then stop and show me the diff and results before committing. One task per commit, stage only the task's files, never push.

Known gaps in the reference, fix them in the matching task:
- B1: `pipeline/pyproject.toml` doesn't declare numpy and pandas.
- B3: `web/lib/bikes.test.ts` doesn't exist yet.
- B4: no "Bikes" link in `TopBar`; the components were never type-checked against `node_modules`, so run `npm run check` and fix what it finds.
- B5: not started.
- B6: README section not written; `demo/bikes/dist/` not in `.gitignore`.

Task B2's `--month` run needs network. If you can't reach the Bluebikes URLs, say so and leave that box for me.

Ask before changing the model math, the `bikes.json` contract or anything outside the files each task lists.
