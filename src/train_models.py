"""Обучает модели прогноза high_views и high_forwards, сохраняет метрики и графики в results/."""

import json
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    COUNT_FEATURES,
    CYCLIC_PERIODS,
    FEATURES_PATH,
    KEYWORD_FLAG_PREFIX,
    MODEL_LABELS,
    N_PERMUTATION_REPEATS,
    N_SVD,
    RESULTS_DIR,
    SEED,
    TARGETS,
    TEST_START,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TRAIN_END,
    VALIDATION_MONTHS,
    check_no_forbidden,
    feature_columns,
    feature_label,
    format_ru,
)
from plotting import (
    INK,
    INK_SECONDARY,
    MUTED,
    SEQUENTIAL,
    SERIES,
    apply_style,
    ru_ticks,
    save_figure,
)

PRIMARY_TARGET, SECOND_TARGET = TARGETS
FEATURE_SETS = ("A", "B")
SVD_PREFIX = "svd_"
TEXT_GROUP = "text"
N_TOP_FEATURES = 15
GROUPINGS = ("actor_type", "language", "keyword_pass")
BASELINE_MODEL = "DummyClassifier"


@dataclass
class Split:
    fit: pd.DataFrame
    evaluate: pd.DataFrame


@dataclass
class Setup:
    feature_set: str
    cv: list[Split]
    final: Split
    columns: list[str]


@dataclass
class Experiment:
    target: str
    feature_set: str
    model: str
    params: dict
    cv_scores: list[float]
    pipeline: Pipeline
    predicted: np.ndarray
    scores: np.ndarray


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def model_catalog() -> dict[str, tuple[BaseEstimator, dict[str, list]]]:
    return {
        "DummyClassifier": (DummyClassifier(strategy="most_frequent"), {}),
        "LogisticRegression": (LogisticRegression(max_iter=2000), {"C": [0.1, 1.0, 10.0]}),
        "kNN": (
            KNeighborsClassifier(n_jobs=-1),
            {"n_neighbors": [25, 75], "weights": ["uniform", "distance"]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=SEED),
            {"max_depth": [6, 10, 14], "min_samples_leaf": [50]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
            {"max_depth": [12, None], "min_samples_leaf": [5, 20]},
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=400,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=-1,
                random_state=SEED,
            ),
            {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
        ),
        "LightGBM": (
            LGBMClassifier(
                n_estimators=400,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=SEED,
                verbose=-1,
            ),
            {"num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
        ),
        "MLP": (
            MLPClassifier(early_stopping=True, max_iter=200, random_state=SEED),
            {"hidden_layer_sizes": [(64,), (128, 64)], "alpha": [1e-4, 1e-3]},
        ),
    }


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data = pd.read_parquet(FEATURES_PATH)
    features = feature_columns(data)
    train = data[data["local_day"] <= TRAIN_END].reset_index(drop=True)
    test = data[data["local_day"] >= TEST_START].reset_index(drop=True)
    check_split(train, test, len(data))
    return train, test, features


def check_split(train: pd.DataFrame, test: pd.DataFrame, n_total: int) -> None:
    if test["local_day"].min() < TEST_START:
        raise ValueError(f"В проверочной части есть даты раньше {TEST_START}")
    if train["local_day"].max() > TRAIN_END:
        raise ValueError(f"В обучающей части есть даты позже {TRAIN_END}")
    if len(train) + len(test) != n_total:
        raise ValueError("Обучающая и проверочная части не покрывают все посты")


def svd_columns() -> list[str]:
    return [f"{SVD_PREFIX}{i}" for i in range(N_SVD)]


def add_text_components(split: Split) -> Split:
    text_model = make_pipeline(
        TfidfVectorizer(
            ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES
        ),
        TruncatedSVD(n_components=N_SVD, random_state=SEED),
    ).fit(split.fit["text"])

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        components = text_model.transform(frame["text"])
        return frame.join(pd.DataFrame(components, columns=svd_columns(), index=frame.index))

    return Split(transform(split.fit), transform(split.evaluate))


def prepare_split(fit: pd.DataFrame, evaluate: pd.DataFrame, feature_set: str) -> Split:
    split = Split(fit, evaluate)
    return add_text_components(split) if feature_set == "B" else split


def cv_splits(train: pd.DataFrame, feature_set: str) -> list[Split]:
    return [
        prepare_split(train[train["month"] < month], train[train["month"] == month], feature_set)
        for month in VALIDATION_MONTHS
    ]


def set_columns(feature_set: str, features: list[str]) -> list[str]:
    columns = features + svd_columns() if feature_set == "B" else list(features)
    check_no_forbidden(columns)
    return columns


def make_setup(
    feature_set: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> Setup:
    log(f"Подготовка набора {feature_set}")
    return Setup(
        feature_set,
        cv_splits(train, feature_set),
        prepare_split(train, test, feature_set),
        set_columns(feature_set, features),
    )


def encode_cyclic(frame: pd.DataFrame) -> np.ndarray:
    periods = np.array([CYCLIC_PERIODS[column] for column in frame.columns])
    angle = 2 * np.pi * frame.to_numpy(dtype=float) / periods
    return np.hstack([np.sin(angle), np.cos(angle)])


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    flags = [c for c in columns if c.startswith(KEYWORD_FLAG_PREFIX)]
    scaled = CONTINUOUS_FEATURES + [c for c in columns if c.startswith(SVD_PREFIX)]
    counts = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("cyclic", FunctionTransformer(encode_cyclic), list(CYCLIC_PERIODS)),
            ("counts", counts, COUNT_FEATURES),
            ("scaled", StandardScaler(), scaled),
            ("binary", "passthrough", BINARY_FEATURES + flags),
        ],
        sparse_threshold=0,
    )


def build_pipeline(model: str, params: dict, columns: list[str]) -> Pipeline:
    estimator = clone(model_catalog()[model][0]).set_params(**params)
    return Pipeline([("prep", make_preprocessor(columns)), ("model", estimator)])


def fit_and_score(
    model: str, params: dict, split: Split, columns: list[str], target: str
) -> tuple[Pipeline, np.ndarray]:
    pipeline = build_pipeline(model, params, columns).fit(split.fit[columns], split.fit[target])
    return pipeline, pipeline.predict_proba(split.evaluate[columns])[:, 1]


def cv_scores(model: str, params: dict, setup: Setup, target: str) -> list[float]:
    scores = []
    for split in setup.cv:
        _, probabilities = fit_and_score(model, params, split, setup.columns, target)
        scores.append(float(roc_auc_score(split.evaluate[target], probabilities)))
    return scores


def tune(model: str, setup: Setup, target: str) -> tuple[dict, list[dict]]:
    trials = [
        {"params": params, "scores": cv_scores(model, params, setup, target)}
        for params in ParameterGrid(model_catalog()[model][1])
    ]
    return max(trials, key=lambda trial: np.mean(trial["scores"])), trials


def run_final(
    model: str, params: dict, scores: list[float], setup: Setup, target: str
) -> Experiment:
    pipeline, probabilities = fit_and_score(model, params, setup.final, setup.columns, target)
    predicted = pipeline.predict(setup.final.evaluate[setup.columns])
    return Experiment(
        target, setup.feature_set, model, params, scores, pipeline, predicted, probabilities
    )


def classification_metrics(y_true: np.ndarray, predicted: np.ndarray, scores: np.ndarray) -> dict:
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc,
    }


def experiment_row(experiment: Experiment, y_test: np.ndarray) -> dict:
    return {
        "target": experiment.target,
        "feature_set": experiment.feature_set,
        "model": experiment.model,
        "params": json.dumps(experiment.params),
        "cv_roc_auc": np.mean(experiment.cv_scores),
        **classification_metrics(y_test, experiment.predicted, experiment.scores),
    }


def trial_rows(target: str, feature_set: str, model: str, trials: list[dict]) -> list[dict]:
    return [
        {
            "target": target,
            "feature_set": feature_set,
            "model": model,
            "params": json.dumps(trial["params"]),
            **{f"fold_{m}": s for m, s in zip(VALIDATION_MONTHS, trial["scores"], strict=True)},
            "cv_roc_auc": np.mean(trial["scores"]),
        }
        for trial in trials
    ]


def run_feature_set(setup: Setup) -> tuple[list[Experiment], list[dict]]:
    experiments, trials_log = [], []
    for model in model_catalog():
        started = time.perf_counter()
        best, trials = tune(model, setup, PRIMARY_TARGET)
        experiments.append(run_final(model, best["params"], best["scores"], setup, PRIMARY_TARGET))
        trials_log += trial_rows(PRIMARY_TARGET, setup.feature_set, model, trials)
        log(
            f"{setup.feature_set} {model}: CV ROC-AUC {np.mean(best['scores']):.4f}, "
            f"{time.perf_counter() - started:.0f} с, {best['params']}"
        )
    return experiments, trials_log


def select_best(experiments: list[Experiment]) -> Experiment:
    candidates = [e for e in experiments if e.model != BASELINE_MODEL]
    return max(candidates, key=lambda e: np.mean(e.cv_scores))


def repeat_for_second_target(best: Experiment, setup: Setup) -> list[Experiment]:
    repeated = []
    for model, params in ((best.model, best.params), (BASELINE_MODEL, {})):
        scores = cv_scores(model, params, setup, SECOND_TARGET)
        repeated.append(run_final(model, params, scores, setup, SECOND_TARGET))
    return repeated


def permuted_copy(frame: pd.DataFrame, group: list[str], rng: np.random.Generator) -> pd.DataFrame:
    shuffled = frame.copy()
    shuffled[group] = frame[group].to_numpy()[rng.permutation(len(frame))]
    return shuffled


def importance_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {c: [c] for c in columns if not c.startswith(SVD_PREFIX)}
    text_block = [c for c in columns if c.startswith(SVD_PREFIX)]
    # Компоненты SVD переставляются вместе: их вклад оценивается как вклад текста целиком.
    if text_block:
        groups[TEXT_GROUP] = text_block
    return groups


def permutation_importance_by_group(
    pipeline: Pipeline, frame: pd.DataFrame, y_true: np.ndarray, columns: list[str]
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    features = frame[columns]

    def score(sample: pd.DataFrame) -> float:
        return roc_auc_score(y_true, pipeline.predict_proba(sample)[:, 1])

    baseline = score(features)
    rows = []
    for name, group in importance_groups(columns).items():
        drops = [
            baseline - score(permuted_copy(features, group, rng))
            for _ in range(N_PERMUTATION_REPEATS)
        ]
        rows.append(
            {"feature": name, "importance_mean": np.mean(drops), "importance_std": np.std(drops)}
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False, ignore_index=True)


def metrics_by_group(best: Experiment, test: pd.DataFrame) -> pd.DataFrame:
    y_true = test[best.target].to_numpy()
    rows = []
    for grouping in GROUPINGS:
        for value, index in test.groupby(grouping).indices.items():
            y_group = y_true[index]
            metrics = classification_metrics(y_group, best.predicted[index], best.scores[index])
            rows.append(
                {
                    "grouping": grouping,
                    "group": value,
                    "n": len(index),
                    "positive_share": y_group.mean(),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def model_title(experiment: Experiment) -> str:
    return f"{MODEL_LABELS[experiment.model]}, набор {experiment.feature_set}"


def plot_confusion_matrix(best: Experiment, y_true: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, best.predicted)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(matrix, cmap=SEQUENTIAL)
    ax.grid(False)
    low, span = matrix.min(), max(matrix.max() - matrix.min(), 1)
    for (row, col), count in np.ndenumerate(matrix):
        share = format_ru(100 * count / matrix.sum(), 1)
        color = "white" if (count - low) / span > 0.5 else INK
        ax.text(col, row, f"{format_ru(count)}\n{share} %", ha="center", va="center", color=color)
    labels = ["0: не выше медианы", "1: выше медианы"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels, rotation=90, va="center")
    ax.set_xlabel("Прогноз модели\nв ячейках: число постов и доля от всей проверочной части")
    ax.set_ylabel("Фактический класс")
    ax.set_title(f"Матрица ошибок, проверочная часть\n{model_title(best)}")
    save_figure(fig, "confusion_matrix.png")


def plot_roc(best: Experiment, y_true: np.ndarray) -> None:
    false_rate, true_rate, _ = roc_curve(y_true, best.scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.plot(false_rate, true_rate, color=SERIES[0])
    auc = format_ru(roc_auc_score(y_true, best.scores), 3)
    ax.text(0.97, 0.05, f"ROC-AUC = {auc}", ha="right", color=INK, transform=ax.transAxes)
    ax.text(
        0.74,
        0.66,
        "случайный прогноз",
        color=INK_SECONDARY,
        rotation=45,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_xlabel("Доля ложноположительных (FPR)")
    ax.set_ylabel("Доля истинноположительных (TPR)")
    ax.set_title(f"ROC-кривая, проверочная часть\n{model_title(best)}")
    ru_ticks(ax.xaxis, 1)
    ru_ticks(ax.yaxis, 1)
    save_figure(fig, "roc_curve.png")


def plot_importance(importance: pd.DataFrame, best: Experiment) -> None:
    top = importance.head(N_TOP_FEATURES).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(
        [feature_label(f) for f in top["feature"]],
        top["importance_mean"],
        xerr=top["importance_std"],
        height=0.6,
        color=SERIES[0],
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2},
    )
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Снижение ROC-AUC при перестановке признака (среднее ± ст. откл.)")
    title = f"Перестановочная важность: {N_TOP_FEATURES} главных признаков"
    ax.set_title(f"{title}\n{model_title(best)}")
    ru_ticks(ax.xaxis, 3)
    save_figure(fig, "permutation_importance.png")


def save_table(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RESULTS_DIR / name, index=False, encoding="utf-8")


def analyse_best(best: Experiment, setup: Setup) -> None:
    test = setup.final.evaluate
    y_true = test[best.target].to_numpy()
    plot_confusion_matrix(best, y_true)
    plot_roc(best, y_true)
    log("Перестановочная важность")
    importance = permutation_importance_by_group(best.pipeline, test, y_true, setup.columns)
    save_table(importance, "permutation_importance.csv")
    plot_importance(importance, best)
    save_table(metrics_by_group(best, test), "metrics_by_group.csv")


def save_best_config(best: Experiment) -> None:
    config = {
        "selection": "максимальный средний ROC-AUC на кросс-валидации по месяцам обучающей части",
        "target": best.target,
        "feature_set": best.feature_set,
        "model": best.model,
        "params": best.params,
        "cv_roc_auc": float(np.mean(best.cv_scores)),
    }
    (RESULTS_DIR / "best_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), "utf-8"
    )


def main() -> None:
    apply_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    train, test, features = load_split()
    log(f"Обучение: {len(train)} постов, проверка: {len(test)} постов")
    experiments, trials_log, setups = [], [], {}
    for feature_set in FEATURE_SETS:
        setups[feature_set] = make_setup(feature_set, train, test, features)
        found, trials = run_feature_set(setups[feature_set])
        experiments += found
        trials_log += trials

    best = select_best(experiments)
    log(f"Лучшая конфигурация: {best.feature_set} {best.model} {best.params}")
    save_best_config(best)
    analyse_best(best, setups[best.feature_set])
    experiments += repeat_for_second_target(best, setups[best.feature_set])

    rows = [experiment_row(e, test[e.target].to_numpy()) for e in experiments]
    save_table(pd.DataFrame(rows), "metrics.csv")
    save_table(pd.DataFrame(trials_log), "cv_results.csv")
    log("Готово")


if __name__ == "__main__":
    main()
