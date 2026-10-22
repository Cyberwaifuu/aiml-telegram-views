# Predicting Above-Median Post Views in Telegram Channels of Kazakhstan Using Machine Learning

Code for the article. The scripts build post features from a corpus of Telegram posts by
government and blogger channels, train classifiers and write the tables and figures used in the
article. Report texts and figure labels are in Russian.

Main target: `high_views = 1` if a post's views are strictly above the median of its
channel × month × collection run group. Posts younger than 14 days at collection and groups with
fewer than 20 posts are excluded. Training data: posts up to 2026-06-30; test data: from 2026-07-01.

## Setup

Python 3.14.

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The corpus is not included. The scripts expect these files in `data/`:

- `corpus.parquet`: posts with columns `doc_id`, `source`, `channel`, `actor_type`, `date`,
  `local_day`, `text`, `language`, `keyword_pass`, `matched_keywords_ru`, `matched_keywords_kz`,
  `p_on_topic`, `views`, `forwards`, `replies_count`. Only rows with
  `source = "telegram_post"` are used; `build_dataset.py` expects 71,351 of them.
- `collection_times.parquet`: `doc_id`, `channel`, `collected_at_est`, `run_label` (collection
  run), `age_days` (post age at collection, days).
- `channels.csv`: `channel`, `actor_type`, `subtype`, `subscribers`, `verified_at`, `added`,
  `first_collected`. Read by `v3.py`.
- `channel_name_extras.txt`: channel names that are removed from post texts before the term
  tables, one per line, in addition to names taken from the channel handles. Read by
  `diagnostics.py` and `make_summary.py`.

## Run order

Run from the repository root, in this order:

```
python src/build_dataset.py
python src/train_models.py
python src/diagnostics.py
python src/robustness.py
python src/density.py
python src/make_summary.py
python src/v1_target.py
python src/compare_versions.py
python src/verify.py
python src/v3.py
```

| Script | Writes |
|---|---|
| `build_dataset.py` | `data/posts_features.parquet`, `results/dataset_stats.json` |
| `train_models.py` | `results/`: `metrics.csv`, `cv_results.csv`, `best_config.json`, `metrics_by_group.csv`, `permutation_importance.csv`, `confusion_matrix.png`, `roc_curve.png`, `permutation_importance.png` |
| `diagnostics.py` | `results/`: `high_views_by_time.csv`, `high_views_weekday_hour.csv`, `high_views_by_age.csv`, `top_terms.csv`, `term_boilerplate.csv`, `keyword_comparison.csv`, `channel_sign_test.csv`, `monthly_medians.csv`, `weekday_hour_heatmap.png` |
| `robustness.py` | `results/`: `robustness_*.csv`, `robustness_stats.json`, `robustness.md` |
| `density.py` | `results/`: `density_*.csv`, `density_stats.json` |
| `make_summary.py` | `results/`: `summary.md`, `views_by_layer.png` |
| `v1_target.py` | `results/v1_target_diagnostics.md` |
| `compare_versions.py` | `results/v1_vs_v2.md` |
| `verify.py` | nothing; recomputes the numbers in `results/summary.md` from the data, prints PASS/FAIL for each and exits with code 1 if any check fails |
| `v3.py` | `results_v3/`: `t1_metrics.csv`, `t1_group_importance.csv`, `t2_metrics.csv`, `t3_metrics.csv`, `v2_vs_v3.md` |

`v3.py` runs after the main pipeline. It adds the extended feature groups (posting context,
the channel's earlier views, the channel itself), reads `results/` and writes only to
`results_v3/`.

## Article figures and tables

```
python src/article_figures.py
python src/article_tables.py
```

- `article_figures.py` writes `article/figures.xlsx` (Figures 1, 3 and 4 as Excel charts) from
  `results/monthly_medians.csv`, `results/permutation_importance.csv` and
  `results/high_views_by_time.csv`.
- `article/figure2_pipeline.drawio` is Figure 2 (open in draw.io).
- `article/figure*.png` are the images used in the article.
- `article_tables.py` writes `article/table1.csv` (Table 1). It reads
  `data/posts_features.parquet`, so it needs the corpus.
- `article/verify_output.txt` is the output of `verify.py`.

## Notes

- Random seed: `SEED = 42` in `src/common.py`, used wherever there is randomness.
- Models use all CPU cores (`n_jobs=-1`). The results were produced with Python 3.14.2 on
  Windows 10.
- `results_v1/` holds the outputs of an earlier version of the pipeline, in which the target
  had no correction for post age at collection. No script here writes it; `compare_versions.py`
  reads it.
- `results/summary.md` and `results/term_boilerplate.csv` are not included because they contain
  channel names and lines from post texts. `make_summary.py` and `diagnostics.py` recreate them.
- Lint settings are in `ruff.toml` (`ruff check src`).
