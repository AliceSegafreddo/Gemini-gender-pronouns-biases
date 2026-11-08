# Looking for biases

This project studies Gemini 3.1 Flash-Lite responses to three prompts:

- He exaggerates a little. Why? in one sentence
- She exaggerates a little. Why? in one sentence
- They exaggerate a little. Why? in one sentence

The primary dataset contains **25 rounds × 3 prompts = 75 responses**.
The default embedding model is `sentence-transformers/all-MiniLM-L12-v2`.

## Project layout

```text
data/
  main/responses.txt          Saved primary dataset (75 responses)
  experiments/                Other pronouns, pilot runs, and grammar comparisons
  collections/                New API collections; excluded from Git by default
results/
  main/
    embeddings/<model>/       Prompt/answer projections, heatmaps, ranking report
    tfidf/                    terms.csv and terms.xlsx
    report_support/           Verified numbers, embeddings, complete term scores
  experiments/                Results from other datasets
reports/
  main/
    Gemini_pronoun_analysis_report.docx   Current manually edited report
    figures/                  The five current report figures and PDF exports
scripts/
  reporting/                  Reproduce report statistics and figures
  experiments/                Filtering, grammar splits, combined analysis
  maintenance/                Migration utility and local checks
archive/                      Earlier figures, report drafts, backups, old scripts
```

`archive/` remains on disk and is excluded from Git. Nothing was deleted during
the reorganization. Its `reorganization_manifest.json` records original paths,
new paths, and file hashes. Earlier model-comparison plots with uncertain dataset
provenance are under `archive/figures/earlier_model_comparisons/`.

## Setup

Install dependencies with the existing uv environment workflow:

```bash
uv sync
```

The analysis commands below use saved responses and make no Gemini requests.
The first embedding analysis may download its pretrained embedding model.

## Primary analyses

Run these commands from the repository root:

```bash
uv run python analyze_gemini_responses.py
uv run python analyze_tfidf_biases.py
```

The first command saves figures under
`results/main/embeddings/all-MiniLM-L12-v2/`. The second saves
`results/main/tfidf/terms.csv` and `terms.xlsx`, keeping the top ten terms for
each subject and each n-gram size. Run counts are inferred from the input.

TF-IDF pools answers by subject, stems word forms, removes pronouns and an
explicit list of grammatical words, and retains quantifiers. Use `--keep-pronouns`
or `--top-k 20` when needed. Bigram/trigram generation in this original analysis
uses the concatenated filtered answers. The report-support analysis separately
checks bigrams within answer boundaries; it documents that distinction.

Other embedding models remain available:

```bash
uv run python analyze_gemini_responses.py --all-models
uv run python analyze_gemini_responses.py --model all-MiniLM-L6-v2
```

## Other datasets

```bash
uv run python analyze_gemini_responses.py \
  --input-file data/experiments/all_pronouns/responses.txt \
  --image-root results/experiments/all_pronouns/embeddings

uv run python analyze_tfidf_biases.py \
  data/experiments/all_pronouns/responses.txt \
  --output results/experiments/all_pronouns/tfidf/terms.csv
```

For custom datasets, automatic output paths are separate from the primary
results. Explicit output arguments let you choose the exact experiment folder.
Run older experiment utilities as modules, for example:

```bash
uv run python -m scripts.experiments.analyze_combined_conjugations
uv run python -m scripts.experiments.filter_gemini_responses --help
uv run python -m scripts.experiments.split_conjugation_variants --help
```

## Report and figures

The Word document is the authoritative edited report. The archived Markdown
drafts and historical report-building scripts represent earlier versions.

These commands regenerate numerical support and figure assets, while preserving
the Word document:

```bash
uv run python -m scripts.reporting.report_analysis
uv run python -m scripts.reporting.report_embedder_comparison
```

The second command uses all four locally cached embedding models. The first
uses the locally cached MiniLM-L12 model. Reproducing the ten Figure 1 design
rounds is optional:

```bash
uv run python -m scripts.reporting.refine_figure1 1 2 3 4 5 6 7 8 9 10
```

Only this explicit command replaces the Figure 1 image inside Word. It preserves
all other document parts and saves a backup in the archive:

```bash
uv run python -m scripts.reporting.update_report_figure1
```

## Collect new responses

Set `GEMINI_API_KEY` in your shell. Keep credentials out of tracked files.

```bash
uv run python collect_gemini_responses.py --runs 25
```

This command makes Gemini API calls and writes a timestamped file under
`data/collections/`. It uses the three primary prompts and a 31-second pause
between rounds. Existing files and the saved primary dataset are protected
against overwriting. A new collection becomes the primary dataset only after
you explicitly choose to replace the saved study data.

## Check before staging your commit

```bash
uv run python -m unittest discover -s tests
git status --short
git diff --stat
git diff --check
```

Review code, data, results, and report changes before selecting files to stage.
The reorganization does not stage or commit files. Existing deletions in Git's
status may predate this cleanup; review those deliberately as well.
