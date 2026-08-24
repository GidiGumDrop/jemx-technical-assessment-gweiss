# Overtime early warning — Jem technical assessment

**Dashboard:** 154.65.109.55:8089
**Video:** https://www.loom.com/share/457dbb4b54704d0281a6aabd21b136a2

## Required files

- **[predictions.csv](predictions.csv)** — one row per employee, `will_breach` and `risk_score`
- **[note_classifications.csv](note_classifications.csv)** — all 2,117 supervisor notes, each sorted into a reason
- **[NOTES.md](NOTES.md)** — assumptions, the note-sorting check, what the model learned

## The work

- **[notebooks/eda.ipynb](notebooks/eda.ipynb)** — what the data supports
- **[notebooks/model_evaluation.ipynb](notebooks/model_evaluation.ipynb)** — the model against the naive baseline, and where it fails
- **[notebooks/note_classifier.ipynb](notebooks/note_classifier.ipynb)** — Naive Bayes vs keyword rules, on notes neither had seen
- **[docs/decisions.md](docs/decisions.md)** — every decision, in detail
- **[docs/brief.md](docs/brief.md)** — the original brief
