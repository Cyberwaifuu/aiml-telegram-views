"""Эксперимент v3: контекст публикаций, история просмотров канала, задачи T1–T3.

v2 не меняется: данные и результаты v2 только читаются, всё новое пишется в results_v3/.
Параметры моделей не подбираются: LightGBM — параметры, выбранные в v2, логистическая
регрессия — C = 0,1. TF-IDF и SVD обучаются один раз на обучающей части и используются
во всех задачах.
"""

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from common import (
    FEATURES_PATH,
    MODEL_LABELS,
    N_PERMUTATION_REPEATS,
    RESULTS_DIR,
    ROOT,
    SEED,
    TARGET_GROUP,
    TRAIN_END,
    feature_columns,
    format_ru,
)
from report import markdown_table, metric, share
from train_models import (
    SVD_PREFIX,
    Split,
    check_split,
    classification_metrics,
    make_preprocessor,
    model_catalog,
    prepare_split,
    set_columns,
)
from v3_features import (
    CONTEXT_FEATURES,
    HISTORY_FEATURES,
    check_features,
    corpus_timeline,
    subscribers,
    timeline_features,
)

V3_DIR = ROOT / "results_v3"
REPORT_PATH = V3_DIR / "v2_vs_v3.md"
TASK_TARGETS = {"T1": "high_views", "T2": "extreme_views", "T3": "above_train_median"}
NOT_FEATURES = frozenset(
    {"views", "forwards", "replies_count", "age_days", "run_label", "next_post_hours", "doc_id"}
)
LOG_COUNT_FEATURES = ["posts_prev_1h", "posts_prev_3h", "posts_prev_24h", "day_ordinal"]
IMPUTED_FEATURES = ["log_minutes_since_prev", *HISTORY_FEATURES, "log_subscribers"]
FLAG_FEATURES = ["first_of_day"]
CHANNEL = "channel"
MODELS = ("LightGBM", "LogisticRegression")
BASELINE = "DummyClassifier"
EXTREME_SHARE = 0.25
GROUP_LABELS = {
    "G0": "G0: набор B",
    "G1": "G1: + контекст публикаций",
    "G2": "G2: + история просмотров",
    "G3": "G3: + канал",
    "G3+subscribers": "G3 + log подписчиков",
    "channel": "только канал",
}


def read_best() -> dict:
    return json.loads((RESULTS_DIR / "best_config.json").read_text("utf-8"))


def model_params(best: dict) -> dict[str, dict]:
    return {"LightGBM": best["params"], "LogisticRegression": {"C": 0.1}, BASELINE: {}}


def add_targets(data: pd.DataFrame) -> pd.DataFrame:
    rank = data.groupby(list(TARGET_GROUP))["views"].rank(pct=True)
    extreme = pd.Series(np.nan, index=data.index)
    extreme[rank <= EXTREME_SHARE] = 0
    extreme[rank > 1 - EXTREME_SHARE] = 1
    threshold = data.loc[data["local_day"] <= TRAIN_END, "views"].median()
    return data.assign(
        extreme_views=extreme, above_train_median=(data["views"] > threshold).astype("int8")
    )


def load_data() -> tuple[pd.DataFrame, int]:
    timeline = corpus_timeline()
    features = timeline_features(timeline)
    checked = check_features(timeline, features)
    data = pd.read_parquet(FEATURES_PATH).merge(
        features, on="doc_id", how="left", validate="one_to_one"
    )
    return add_targets(data.merge(subscribers(), on=CHANNEL, how="left")), checked


def text_split(data: pd.DataFrame) -> Split:
    train = data[data["local_day"] <= TRAIN_END].reset_index(drop=True)
    test = data[data["local_day"] > TRAIN_END].reset_index(drop=True)
    check_split(train, test, len(data))
    if set(train["doc_id"]) & set(test["doc_id"]):
        raise ValueError("Посты попали и в обучающую, и в проверочную часть")
    return prepare_split(train, test, "B")


def feature_groups(base: list[str]) -> dict[str, list[str]]:
    first = base + CONTEXT_FEATURES
    second = first + HISTORY_FEATURES
    return {"G0": base, "G1": first, "G2": second, "G3": second + [CHANNEL]}


def check_columns(columns: list[str]) -> None:
    leaked = NOT_FEATURES.intersection(columns)
    if leaked:
        raise ValueError(f"Запрещённые столбцы среди признаков: {sorted(leaked)}")


def preprocessor(columns: list[str]) -> ColumnTransformer:
    extra = {*LOG_COUNT_FEATURES, *IMPUTED_FEATURES, *FLAG_FEATURES, CHANNEL}
    base = [c for c in columns if c not in extra]
    transformers = list(make_preprocessor(base).transformers) if base else []
    log_counts = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
    imputed = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    blocks = [
        ("context_counts", log_counts, [c for c in LOG_COUNT_FEATURES if c in columns]),
        ("imputed", imputed, [c for c in IMPUTED_FEATURES if c in columns]),
        ("flags", "passthrough", [c for c in FLAG_FEATURES if c in columns]),
        ("channel", OneHotEncoder(handle_unknown="ignore"), [c for c in [CHANNEL] if c in columns]),
    ]
    return ColumnTransformer(transformers + [b for b in blocks if b[2]], sparse_threshold=0)


def scalers(transformer: object) -> list[StandardScaler]:
    if isinstance(transformer, StandardScaler):
        return [transformer]
    if isinstance(transformer, Pipeline):
        return [s for step in transformer.named_steps.values() for s in scalers(step)]
    return []


def check_fitted_on_train(pipeline: Pipeline, n_train: int) -> int:
    fitted = [s for _, t, _ in pipeline["prep"].transformers_ for s in scalers(t)]
    for scaler in fitted:
        if not np.all(np.asarray(scaler.n_samples_seen_) == n_train):
            raise ValueError("Предобработка обучена не только на обучающей части")
    return len(fitted)


def estimator(model: str, params: dict[str, dict]) -> BaseEstimator:
    return clone(model_catalog()[model][0]).set_params(**params[model])


class Runner:
    def __init__(self, params: dict[str, dict]) -> None:
        self.params = params
        self.scalers_checked = 0
        self.fits = 0

    def fit(self, model: str, columns: list[str], split: Split, target: str) -> Pipeline:
        check_columns(columns)
        steps = [("prep", preprocessor(columns)), ("model", estimator(model, self.params))]
        pipeline = Pipeline(steps).fit(split.fit[columns], split.fit[target].astype(int))
        self.scalers_checked += check_fitted_on_train(pipeline, len(split.fit))
        self.fits += 1
        return pipeline

    def evaluate(
        self, model: str, columns: list[str], split: Split, target: str
    ) -> tuple[dict, Pipeline]:
        pipeline = self.fit(model, columns, split, target)
        frame, y_true = split.evaluate[columns], split.evaluate[target].astype(int).to_numpy()
        scores = pipeline.predict_proba(frame)[:, 1]
        return classification_metrics(y_true, pipeline.predict(frame), scores), pipeline


def task_rows(
    runner: Runner, task: str, configs: dict[str, list[str]], split: Split
) -> tuple[list[dict], dict]:
    target, rows, pipelines = TASK_TARGETS[task], [], {}
    for name, columns in configs.items():
        for model in MODELS:
            scores, pipelines[(name, model)] = runner.evaluate(model, columns, split, target)
            rows.append({"task": task, "group": name, "model": model, **scores})
    baseline, _ = runner.evaluate(BASELINE, next(iter(configs.values())), split, target)
    rows.append({"task": task, "group": "—", "model": BASELINE, **baseline})
    return rows, pipelines


def subset(split: Split, target: str) -> Split:
    return Split(
        split.fit[split.fit[target].notna()], split.evaluate[split.evaluate[target].notna()]
    )


def importance_groups(columns: list[str], base: list[str]) -> dict[str, list[str]]:
    svd = [c for c in columns if c.startswith(SVD_PREFIX)]
    groups = {
        "G0: текст (TF-IDF + SVD)": svd,
        "G0: остальные признаки": [c for c in base if c not in svd],
        "G1: контекст публикаций": [c for c in CONTEXT_FEATURES if c in columns],
        "G2: история просмотров": [c for c in HISTORY_FEATURES if c in columns],
        "G3: канал": [c for c in [CHANNEL] if c in columns],
    }
    return {name: group for name, group in groups.items() if group}


# Столбцы группы переставляются одной перестановкой строк, каждый со своим типом данных.
def permuted_rows(frame: pd.DataFrame, group: list[str], rng: np.random.Generator) -> pd.DataFrame:
    order = rng.permutation(len(frame))
    shuffled = frame.copy()
    for column in group:
        shuffled[column] = frame[column].to_numpy()[order]
    return shuffled


def group_importance(
    pipeline: Pipeline, columns: list[str], base: list[str], split: Split
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    frame, y_true = split.evaluate[columns], split.evaluate[TASK_TARGETS["T1"]].to_numpy()
    baseline = roc_auc_score(y_true, pipeline.predict_proba(frame)[:, 1])
    rows = []
    for name, group in importance_groups(columns, base).items():
        drops = [
            baseline
            - roc_auc_score(y_true, pipeline.predict_proba(permuted_rows(frame, group, rng))[:, 1])
            for _ in range(N_PERMUTATION_REPEATS)
        ]
        rows.append(
            {"group": name, "columns": len(group), "mean": np.mean(drops), "std": np.std(drops)}
        )
    return pd.DataFrame(rows)


def best_group(t1: pd.DataFrame) -> str:
    lightgbm = t1[t1["model"] == "LightGBM"]
    return lightgbm.loc[lightgbm["roc_auc"].idxmax(), "group"]


def class_counts(split: Split, target: str) -> pd.DataFrame:
    parts = {"обучение": split.fit, "проверка": split.evaluate}
    rows = [
        {"часть": name, "n": int(part[target].notna().sum()), "share": part[target].mean()}
        for name, part in parts.items()
    ]
    return pd.DataFrame(rows)


def metrics_table(rows: pd.DataFrame, with_gain: bool = False) -> str:
    frame = pd.DataFrame(
        {
            "признаки": rows["group"].map(lambda g: GROUP_LABELS.get(g, g)),
            "модель": rows["model"].map(MODEL_LABELS),
            "accuracy": rows["accuracy"].map(metric),
            "F1": rows["f1"].map(metric),
            "ROC-AUC": rows["roc_auc"].map(metric),
        }
    )
    if with_gain:
        gain = rows.groupby("model")["roc_auc"].diff()
        frame["прирост ROC-AUC к предыдущей группе"] = gain.map(lambda v: metric(round(v, 3) + 0.0))
    return markdown_table(frame)


def counts_table(counts: pd.DataFrame) -> str:
    return markdown_table(
        counts.assign(n=counts["n"].map(format_ru), share=counts["share"].map(share)).rename(
            columns={"share": "доля класса 1"}
        )
    )


def importance_table(importance: pd.DataFrame) -> str:
    return markdown_table(
        pd.DataFrame(
            {
                "группа признаков": importance["group"],
                "столбцов": importance["columns"],
                "снижение ROC-AUC, среднее": importance["mean"].map(lambda v: format_ru(v, 4)),
                "ст. откл.": importance["std"].map(lambda v: format_ru(v, 4)),
            }
        )
    )


def v2_reference() -> str:
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    rows = metrics[(metrics["target"] == "high_views") & (metrics["feature_set"] == "B")]
    rows = rows.set_index("model").loc[list(MODELS)]
    listed = "; ".join(
        f"{MODEL_LABELS[model]} — {metric(row['roc_auc'])} (параметры `{row['params']}`)"
        for model, row in rows.iterrows()
    )
    return f"Для сравнения, v2 (results/metrics.csv), набор B, ROC-AUC на проверке: {listed}."


def missing_text(split: Split, column: str) -> str:
    train, test = split.fit[column].isna().mean(), split.evaluate[column].isna().mean()
    return f"{column} — {share(train)} в обучающей и {share(test)} в проверочной части"


def facts(data: pd.DataFrame, split: Split, checked: int, runner: Runner) -> str:
    missing = "; ".join(
        missing_text(split, c) for c in ["log_minutes_since_prev", *HISTORY_FEATURES]
    )
    no_subscribers = sorted(data.loc[data["log_subscribers"].isna(), CHANNEL].unique())
    return "\n".join(
        [
            f"- Посты анализа v2: {format_ru(len(data))}; обучение — "
            f"{format_ru(len(split.fit))}, проверка — {format_ru(len(split.evaluate))} "
            f"(то же разбиение по времени, seed {SEED}).",
            "- Модели без подбора параметров: LightGBM с параметрами v2, логистическая регрессия "
            "с C = 0,1, DummyClassifier (самый частый класс) для каждой задачи.",
            f"- Контекст публикаций — по постам того же канала строго раньше текущего; история "
            f"просмотров — по постам того же канала и прогона сбора, опубликованным не позже чем "
            f"за 24 часа до текущего. Для {checked} случайных постов корпуса оба набора признаков "
            "совпали с прямым расчётом по этим правилам.",
            f"- Среди признаков нет views, forwards, replies_count, age_days, run_label и времени "
            f"до следующего поста: проверено для всех {runner.fits} обученных моделей. "
            f"Масштабирование и импутация обучены только на обучающей части: проверено "
            f"{runner.scalers_checked} масштабировщиков. TF-IDF и SVD обучены на обучающей части "
            "и общие для всех задач.",
            f"- Пропуски: {missing}.",
            f"- Нет числа подписчиков: {', '.join(no_subscribers) or 'нет'}.",
        ]
    )


def limitations(split: Split, t3_counts: pd.DataFrame, threshold: float) -> str:
    train_share, test_share = t3_counts["share"].tolist()
    return "\n".join(
        [
            "- История просмотров использует итоговые просмотры более ранних постов, снятые при "
            "сборе, а не их просмотры на момент публикации текущего поста.",
            "- История отбирается по прогону сбора (run_label); у постов в начале прогона "
            "истории мало, отсюда пропуски.",
            "- Медиана группы, по которой задана цель T1, включает просмотры тех же более ранних "
            "постов, что входят в историю просмотров.",
            "- Лучшая группа признаков выбрана по ROC-AUC на проверочной части, отдельной "
            "валидации нет; параметры не подбирались; одно разбиение по времени, один запуск.",
            "- T2: TF-IDF и SVD обучены на всей обучающей части, включая исключённые средние 50 %.",
            f"- T3: порог — медиана просмотров обучающей части ({format_ru(threshold)}); "
            "просмотры не скорректированы на возраст поста при сборе; доля класса 1 — "
            f"{share(train_share)} в обучающей и {share(test_share)} в проверочной части.",
            "- Число подписчиков — одно измерение на канал из data/channels.csv.",
            "- Контекст публикаций считается по всем постам корпуса, включая посты, "
            "не вошедшие в анализ.",
        ]
    )


def build_report(parts: dict) -> str:
    chosen = parts["chosen"]
    return (
        "\n\n".join(
            [
                "# v2 и v3",
                facts(parts["data"], parts["split"], parts["checked"], parts["runner"]),
                "## T1: high_views (цель v2)",
                "Группы признаков накопительные; прирост — разница ROC-AUC с предыдущей группой "
                "той же модели.",
                metrics_table(parts["t1"], with_gain=True),
                v2_reference(),
                "## Перестановочная важность по группам признаков: "
                f"LightGBM, {GROUP_LABELS[chosen]}",
                f"Лучшая группа T1 по ROC-AUC LightGBM на проверке. {N_PERMUTATION_REPEATS} "
                "перестановок на группу, столбцы группы переставляются вместе.",
                importance_table(parts["importance"]),
                "## T2: верхние 25 % против нижних 25 % просмотров внутри группы",
                "Группа — «канал × месяц × прогон сбора» (как в v2); средние 50 % исключены из "
                "обучающей и проверочной части.",
                counts_table(parts["t2_counts"]),
                metrics_table(parts["t2"]),
                "## T3: просмотры выше медианы обучающей части",
                counts_table(parts["t3_counts"]),
                metrics_table(parts["t3"]),
                "## Ограничения",
                limitations(parts["split"], parts["t3_counts"], parts["threshold"]),
            ]
        )
        + "\n"
    )


def save(frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.to_csv(V3_DIR / f"{name}.csv", index=False, encoding="utf-8")


def main() -> None:
    V3_DIR.mkdir(exist_ok=True)
    runner = Runner(model_params(read_best()))
    data, checked = load_data()
    split = text_split(data)
    base = set_columns("B", feature_columns(data))
    groups = feature_groups(base)
    t1_rows, pipelines = task_rows(runner, "T1", groups, split)
    t1 = pd.DataFrame(t1_rows)
    chosen = best_group(t1)
    importance = group_importance(pipelines[(chosen, "LightGBM")], groups[chosen], base, split)
    compared = {"G0": groups["G0"], chosen: groups[chosen]}
    t2_split = subset(split, TASK_TARGETS["T2"])
    t2 = pd.DataFrame(task_rows(runner, "T2", compared, t2_split)[0])
    extra = {"G3+subscribers": groups["G3"] + ["log_subscribers"], "channel": [CHANNEL]}
    t3 = pd.DataFrame(task_rows(runner, "T3", compared | extra, split)[0])
    threshold = data.loc[data["local_day"] <= TRAIN_END, "views"].median()
    parts = {
        "data": data,
        "split": split,
        "checked": checked,
        "runner": runner,
        "chosen": chosen,
        "t1": t1,
        "importance": importance,
        "t2": t2,
        "t2_counts": class_counts(t2_split, TASK_TARGETS["T2"]),
        "t3": t3,
        "t3_counts": class_counts(split, TASK_TARGETS["T3"]),
        "threshold": threshold,
    }
    save({"t1_metrics": t1, "t1_group_importance": importance, "t2_metrics": t2, "t3_metrics": t3})
    REPORT_PATH.write_text(build_report(parts), "utf-8")
    print(f"Записано: {REPORT_PATH}")


if __name__ == "__main__":
    main()
