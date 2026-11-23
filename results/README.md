# Training results

Every finished training run writes a short summary here as `<run timestamp>.md`: its settings, a table of every generation (point margin, rounds, matches, teacher agreement, whether the gate kept it), the gate replay, the final check on new games and which generation shipped. The same file is in `runs/<timestamp>/summary.md`, next to `log.txt`, the trainer's full log.

The `runs/` folders hold the nets and are not tracked by git; these summaries are. To share a run, commit and push its summary:

```bash
git add results
git commit -m "Results of run <timestamp>"
git push
```
