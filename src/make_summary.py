"""Собирает results/summary.md и описательный график results/views_by_layer.png."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from common import (
    AGE_BUCKETS,
    DATASET_STATS_PATH,
    FEATURES_PATH,
    LAYER_LABELS,
    MIN_AGE_DAYS,
    MODEL_LABELS,
    N_PERMUTATION_REPEATS,
    N_SVD,
    RESULTS_DIR,
    TARGETS,
    TEST_START,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TRAIN_END,
    VALIDATION_MONTHS,
    feature_columns,
    feature_label,
    format_ru,
)
from density import density_sections
from diagnostics import MIN_CELL_POSTS, TERM_MAX_DF, TERM_MIN_DF, TERM_MODEL_C
from plotting import INK, SERIES, apply_style, ru_ticks, save_figure
from report import (
    DEFINITION_LABELS,
    DIAGNOSTIC_LAYERS,
    WEEKDAY_NAMES,
    layered_share_table,
    markdown_table,
    median_text,
    metric,
    p_value,
    period_parts,
    points,
    share,
)
from robustness import robustness_sections
from term_text import BOILERPLATE_POSTS, BOILERPLATE_SHARE, channel_name_extras

SUMMARY_PATH = RESULTS_DIR / "summary.md"
LAYERS = ("government", "blogger")
LANGUAGES = ("ru", "kz", "parallel", "mixed", "other")
LANGUAGE_NAMES = {"ru": "русский", "kz": "казахский"}
N_TOP_FEATURES = 15
N_HISTOGRAM_BINS = 60
TOPIC_LABELS = {1: "экономические (keyword_pass = 1)", 0: "остальные (keyword_pass = 0)"}
GROUPING_LABELS = {"actor_type": "слой", "language": "язык", "keyword_pass": "тематика"}


def layer_name(layer: str) -> str:
    return LAYER_LABELS.get(layer, layer)


def data_facts(data: pd.DataFrame, stats: dict) -> str:
    channels = data.groupby("actor_type")["channel"].nunique()
    parts = period_parts(data)
    runs = data["run_label"].value_counts().sort_index()
    lines = [
        f"- Постов с source = telegram_post на входе: {format_ru(stats['posts_input'])}.",
        f"- Исключено постов моложе {stats['min_age_days']} дней на момент сбора: "
        f"{format_ru(stats['posts_young_excluded'])}.",
        f"- Исключено групп «канал × месяц × прогон сбора» меньше {stats['min_group_size']} "
        f"постов: {stats['groups_excluded']} из {stats['groups_total']}, "
        f"в них {format_ru(stats['posts_excluded'])} постов.",
        f"- Постов в анализе: {format_ru(len(data))}.",
        "- Постов по прогонам сбора (run_label): "
        + ", ".join(f"{run} — {format_ru(n)}" for run, n in runs.items())
        + ".",
        f"- Каналов: {data['channel'].nunique()} (государственные: {channels['government']}, "
        f"блогеры: {channels['blogger']}).",
        f"- Период (local_day, UTC+5): {data['local_day'].min()} — {data['local_day'].max()}.",
    ]
    for name in ("обучение", "проверка"):
        part = parts[name]
        lines.append(
            f"- {name.capitalize()}: {part['local_day'].min()} — {part['local_day'].max()}, "
            f"{format_ru(len(part))} постов."
        )
    return "\n".join(lines)


def layer_language_table(data: pd.DataFrame) -> str:
    table = pd.crosstab(data["actor_type"], data["language"], margins=True, margins_name="всего")
    table = table.reindex(index=[*LAYERS, "всего"], columns=[*LANGUAGES, "всего"], fill_value=0)
    table = table.map(format_ru)
    table.insert(0, "слой", [layer_name(i) for i in table.index])
    return markdown_table(table)


def layer_row(name: str, part: pd.DataFrame) -> dict:
    return {
        "слой": name,
        "каналов": part["channel"].nunique(),
        "постов": format_ru(len(part)),
        "медиана просмотров": median_text(part["views"].median()),
        "медиана пересылок": median_text(part["forwards"].median()),
    }


def layer_table(data: pd.DataFrame) -> str:
    rows = [layer_row(layer_name(layer), data[data["actor_type"] == layer]) for layer in LAYERS]
    rows.append(layer_row("всего", data))
    return markdown_table(pd.DataFrame(rows))


def class_balance_table(data: pd.DataFrame) -> str:
    rows = [
        {
            "цель": target,
            "часть": name,
            "n": format_ru(len(part)),
            "класс 1": format_ru(part[target].sum()),
            "доля класса 1": share(part[target].mean()),
        }
        for target in TARGETS
        for name, part in period_parts(data).items()
    ]
    return markdown_table(pd.DataFrame(rows))


def topic_cell(part: pd.DataFrame) -> tuple[str, str]:
    return format_ru(len(part)), share(part["high_views"].mean())


def topic_row(name: str, part: pd.DataFrame) -> dict:
    economic_n, economic_share = topic_cell(part[part["keyword_pass"] == 1])
    other_n, other_share = topic_cell(part[part["keyword_pass"] == 0])
    return {
        "слой": name,
        "экономические: n": economic_n,
        "экономические: доля high_views = 1": economic_share,
        "остальные: n": other_n,
        "остальные: доля high_views = 1": other_share,
    }


def topic_table(data: pd.DataFrame) -> str:
    rows = [topic_row(layer_name(layer), data[data["actor_type"] == layer]) for layer in LAYERS]
    rows.append(topic_row("всего", data))
    return markdown_table(pd.DataFrame(rows))


def metrics_table(metrics: pd.DataFrame, best: dict) -> str:
    def model_cell(row: pd.Series) -> str:
        is_best = row["model"] == best["model"] and row["feature_set"] == best["feature_set"]
        return MODEL_LABELS[row["model"]] + (" (выбрана)" if is_best else "")

    return markdown_table(
        pd.DataFrame(
            {
                "набор": metrics["feature_set"],
                "модель": metrics.apply(model_cell, axis=1),
                "ROC-AUC, CV": metrics["cv_roc_auc"].map(metric),
                "accuracy": metrics["accuracy"].map(metric),
                "F1": metrics["f1"].map(metric),
                "ROC-AUC": metrics["roc_auc"].map(metric),
                "параметры": metrics["params"].map(lambda p: f"`{p}`"),
            }
        )
    )


def group_label(grouping: str, value: object) -> str:
    if grouping == "keyword_pass":
        return TOPIC_LABELS[int(value)]
    return layer_name(str(value)) if grouping == "actor_type" else str(value)


def group_table(groups: pd.DataFrame) -> str:
    return markdown_table(
        pd.DataFrame(
            {
                "разрез": groups["grouping"].map(GROUPING_LABELS),
                "группа": [
                    group_label(g, v)
                    for g, v in zip(groups["grouping"], groups["group"], strict=True)
                ],
                "n": groups["n"].map(format_ru),
                "доля класса 1": groups["positive_share"].map(share),
                "accuracy": groups["accuracy"].map(metric),
                "F1": groups["f1"].map(metric),
                "ROC-AUC": groups["roc_auc"].map(metric),
            }
        )
    )


def importance_table(importance: pd.DataFrame) -> str:
    top = importance.head(N_TOP_FEATURES)
    return markdown_table(
        pd.DataFrame(
            {
                "№": range(1, len(top) + 1),
                "признак": top["feature"].map(feature_label),
                "колонка": top["feature"].map(lambda f: f"`{f}`"),
                "снижение ROC-AUC, среднее": top["importance_mean"].map(lambda v: format_ru(v, 4)),
                "ст. откл.": top["importance_std"].map(lambda v: format_ru(v, 4)),
            }
        )
    )


def method_facts(best: dict, n_features: int) -> str:
    return "\n".join(
        [
            f"- Обучение: local_day до {TRAIN_END} включительно; проверка: с {TEST_START}.",
            f"- Набор A: {n_features} признаков (метаданные и форма текста). "
            f"Набор B: A + {N_SVD} компонент TruncatedSVD от TF-IDF (слова и биграммы, "
            f"min_df = {TFIDF_MIN_DF}, до {format_ru(TFIDF_MAX_FEATURES)} признаков).",
            "- Масштабирование, TF-IDF и SVD обучаются только на обучающей части "
            "(при кросс-валидации — на обучающей части каждого фолда).",
            "- Подбор параметров: кросс-валидация с расширяющимся окном по месяцам "
            f"(проверочные месяцы {', '.join(VALIDATION_MONTHS)}), критерий — средний ROC-AUC.",
            f"- Выбор лучшей конфигурации: {best['selection']}.",
            "- Порог классификации 0,5; F1 считается для класса 1.",
            "- age_days и run_label используются только для отбора постов и групп, "
            "в признаки моделей не входят.",
        ]
    )


def plot_views_by_layer(data: pd.DataFrame) -> None:
    edges = np.log10([data["views"].min(), data["views"].max()])
    bins = np.logspace(*edges, N_HISTOGRAM_BINS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    label_y = blended_transform_factory(ax.transData, ax.transAxes)
    medians = {layer: data.loc[data["actor_type"] == layer, "views"].median() for layer in LAYERS}
    left_layer = min(medians, key=medians.get)
    for layer, color in zip(LAYERS, SERIES, strict=True):
        views = data.loc[data["actor_type"] == layer, "views"]
        weights = np.full(len(views), 100 / len(views))
        label = f"{layer_name(layer)} (n = {format_ru(len(views))})"
        ax.hist(views, bins=bins, weights=weights, histtype="stepfilled", color=color, alpha=0.1)
        ax.hist(views, bins=bins, weights=weights, histtype="step", color=color, label=label)
        ax.axvline(medians[layer], color=color, linewidth=1, linestyle="--")
        on_left = layer == left_layer
        ax.text(
            medians[layer] / 1.06 if on_left else medians[layer] * 1.06,
            0.97,
            f"медиана {format_ru(medians[layer])}",
            transform=label_y,
            color=INK,
            ha="right" if on_left else "left",
            va="top",
        )
    ax.set_xscale("log")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ru_ticks(ax.xaxis)
    ru_ticks(ax.yaxis)
    ax.set_xlabel("Просмотры поста (логарифмическая шкала)")
    ax.set_ylabel("Доля постов слоя, %")
    ax.set_title("Распределение просмотров постов по слоям")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
    save_figure(fig, "views_by_layer.png")


def read_result(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def weekday_table(time: pd.DataFrame) -> str:
    weekdays = time[time["variable"] == "weekday"]
    return layered_share_table(weekdays, "день недели", list(range(7)), list(WEEKDAY_NAMES))


def hour_table(time: pd.DataFrame) -> str:
    hours = time[time["variable"] == "hour"]
    return layered_share_table(hours, "час", list(range(24)), [f"{h:02d}" for h in range(24)])


def age_table(ages: pd.DataFrame) -> str:
    buckets = list(AGE_BUCKETS)
    return layered_share_table(ages, "возраст при сборе, дней", buckets, buckets)


def term_cells(terms: pd.DataFrame, direction: str, label: str) -> dict[str, list[str]]:
    part = terms[terms["direction"] == direction].sort_values("rank")
    return {
        f"{label}: термин": part["term"].map(lambda term: f"`{term}`").tolist(),
        f"{label}: коэфф.": part["coefficient"].map(lambda v: format_ru(v, 3)).tolist(),
        f"{label}: постов": part["posts"].map(format_ru).tolist(),
    }


def terms_table(terms: pd.DataFrame) -> str:
    positive = term_cells(terms, "positive", "положительные")
    negative = term_cells(terms, "negative", "отрицательные")
    ranks = range(1, len(terms) // 2 + 1)
    return markdown_table(pd.DataFrame({"№": ranks, **positive, **negative}))


def per_language(terms: pd.DataFrame, column: str) -> str:
    values = terms.groupby("language")[column].first()
    return ", ".join(f"{name} — {format_ru(values[code])}" for code, name in LANGUAGE_NAMES.items())


def cleaning_facts() -> str:
    boilerplate = read_result("term_boilerplate.csv")
    extras = ", ".join(f"«{name}»" for name in channel_name_extras())
    return (
        "Перед TF-IDF из текста удалены: ссылки и домены, @упоминания, #хэштеги; строки "
        "с названиями соцсетей; строки, которые без учёта регистра, эмодзи и знаков повторяются "
        f"не менее чем в {format_ru(100 * BOILERPLATE_SHARE)} % постов обучающей части своего "
        f"канала или не менее чем в {BOILERPLATE_POSTS} его постах (подписи, рубрики, подписи "
        f"к фото: {len(boilerplate)} строк в {boilerplate['channel'].nunique()} каналах, список — "
        f"term_boilerplate.csv); латинские названия каналов из их хэндлов и {extras}."
    )


def terms_facts(terms: pd.DataFrame) -> str:
    return (
        f"Обучающая часть (local_day до {TRAIN_END}), цель high_views, отдельная модель "
        f"для постов каждого языка (постов: {per_language(terms, 'train_posts')}; терминов "
        f"в словаре: {per_language(terms, 'vocabulary')}). {cleaning_facts()} TF-IDF: слова "
        f"и биграммы, min_df = {TERM_MIN_DF}, max_df = {format_ru(TERM_MAX_DF, 1)}, "
        f"до {format_ru(TFIDF_MAX_FEATURES)} признаков, без SVD; LogisticRegression, "
        f"C = {format_ru(TERM_MODEL_C, 1)}. Коэффициент больше 0 — термин связан "
        "с high_views = 1. «Постов» — число постов обучающей части этого языка с термином. "
        "Полный список: top_terms.csv."
    )


def terms_sections(terms: pd.DataFrame) -> list[str]:
    sections = []
    for code, name in LANGUAGE_NAMES.items():
        sections += [f"#### {name.capitalize()}", terms_table(terms[terms["language"] == code])]
    return sections


def comparison_table(comparison: pd.DataFrame) -> str:
    intervals = zip(comparison["ci_low"], comparison["ci_high"], strict=True)
    return markdown_table(
        pd.DataFrame(
            {
                "определение": comparison["definition"].map(DEFINITION_LABELS),
                "слой": comparison["layer"].map(DIAGNOSTIC_LAYERS),
                "порог p_on_topic": comparison["threshold"].map(metric),
                "n (1)": comparison["n1"].map(format_ru),
                "доля (1)": comparison["share1"].map(share),
                "n (0)": comparison["n0"].map(format_ru),
                "доля (0)": comparison["share0"].map(share),
                "разница, п. п.": comparison["diff"].map(points),
                "95 % ДИ, п. п.": [f"[{points(low)}; {points(high)}]" for low, high in intervals],
                "z": comparison["z"].map(lambda v: format_ru(v, 2)),
                "p": comparison["p_value"].map(p_value),
            }
        )
    )


def sign_test_table(signs: pd.DataFrame) -> str:
    return markdown_table(
        pd.DataFrame(
            {
                "слой": signs["layer"].map(DIAGNOSTIC_LAYERS),
                "каналов": signs["channels"],
                "разница < 0": signs["negative"],
                "разница > 0": signs["positive"],
                "медиана разницы, п. п.": signs["median_diff"].map(points),
                "p": signs["p_value"].map(p_value),
            }
        )
    )


def monthly_table(medians: pd.DataFrame) -> str:
    order = medians["layer"].map(list(DIAGNOSTIC_LAYERS).index)
    ordered = medians.assign(order=order).sort_values(["order", "month"])
    return markdown_table(
        pd.DataFrame(
            {
                "слой": ordered["layer"].map(DIAGNOSTIC_LAYERS),
                "месяц": ordered["month"],
                "n": ordered["n"].map(format_ru),
                "медиана просмотров": ordered["median_views"].map(median_text),
                "медиана пересылок": ordered["median_forwards"].map(median_text),
                "медиана возраста при сборе, дней": ordered["median_age_days"].map(format_ru),
            }
        )
    )


def diagnostics_sections() -> list[str]:
    time = read_result("high_views_by_time.csv")
    terms = read_result("top_terms.csv")
    return [
        "## Диагностика",
        "Цель high_views — в определении из раздела «Баланс классов»; все посты в анализе.",
        "### а) Доля high_views = 1 по дню недели",
        weekday_table(time),
        "### а) Доля high_views = 1 по часу публикации (UTC+5)",
        hour_table(time),
        "Тепловая карта «день недели × час»: weekday_hour_heatmap.png (клетки, где меньше "
        f"{MIN_CELL_POSTS} постов, заштрихованы); значения клеток: high_views_weekday_hour.csv.",
        "### б) Доля high_views = 1 по возрасту поста на момент сбора",
        f"Возраст — age_days, интервал включает левую границу. Посты моложе {MIN_AGE_DAYS} дней "
        "исключены из анализа.",
        age_table(read_result("high_views_by_age.csv")),
        "### в) Термины логистической регрессии на TF-IDF",
        terms_facts(terms),
        *terms_sections(terms),
        "### г) Экономические и остальные посты: 95 % ДИ и проверка устойчивости",
        "Группа 1 — keyword_pass = 1 или верхняя треть по p_on_topic (p_on_topic не ниже "
        "квантиля 2/3 по всем постам анализа или по постам своего слоя). z-тест для двух долей, "
        "95 % ДИ разницы — интервал Вальда; посты считаются независимыми.",
        comparison_table(read_result("keyword_comparison.csv")),
        "Знаковый тест по каналам, keyword_pass: разница долей high_views = 1 "
        "(keyword_pass = 1 минус 0) внутри каждого канала; двусторонний биномиальный тест, "
        "вероятность 0,5.",
        sign_test_table(read_result("channel_sign_test.csv")),
        "### д) Медианы просмотров и пересылок по месяцам и слоям",
        "Просмотры и пересылки — как сняты при сборе, без поправки на возраст поста.",
        monthly_table(read_result("monthly_medians.csv")),
        *density_sections("###"),
    ]


def quartile_shares(version: str) -> str:
    quartiles = read_result("robustness_quartiles.csv")
    shares = quartiles.loc[quartiles["version"] == version, "share"]
    return " / ".join(share(value) for value in shares)


def overall_late_gap(version: str) -> str:
    gaps = read_result("robustness_late_gap.csv")
    return points(gaps.loc[(gaps["version"] == version) & (gaps["month"] == "все"), "gap"].iloc[0])


def keyword_gap(definition: str, layer: str) -> str:
    rows = read_result("keyword_comparison.csv")
    return points(
        rows.loc[(rows["definition"] == definition) & (rows["layer"] == layer), "diff"].iloc[0]
    )


def spread_text(variable: str) -> str:
    spread = read_result("density_spread.csv")
    rows = spread[spread["variable"] == variable].set_index("quartile")["spread"]
    within = rows.drop(0)
    return (
        f"{points(within.min())}–{points(within.max())} п. п. "
        f"(без разбиения {points(rows.loc[0])} п. п.)"
    )


def next_post_extremes() -> tuple[str, str]:
    rows = read_result("density_quartiles.csv")
    overall = rows[(rows["measure"] == "next_post_hours") & (rows["layer"] == "all")]
    return share(overall["share"].iloc[0]), share(overall["share"].iloc[-1])


def sign_text(layer: str) -> str:
    row = read_result("channel_sign_test.csv").set_index("layer").loc[layer]
    return f"{int(row['negative'])} из {int(row['channels'])}, p = {p_value(row['p_value'])}"


def top_b_models(best: dict) -> str:
    metrics = read_result("metrics.csv")
    rows = metrics[
        (metrics["target"] == TARGETS[0]) & (metrics["feature_set"] == best["feature_set"])
    ]
    top = rows.nlargest(3, "cv_roc_auc")
    listed = "; ".join(
        f"{MODEL_LABELS[model]} — {metric(cv)} и {metric(test)}"
        for model, cv, test in zip(top["model"], top["cv_roc_auc"], top["roc_auc"], strict=True)
    )
    return f"{listed} (средний ROC-AUC на CV и ROC-AUC на проверке, набор {best['feature_set']})"


def limitations_section(data: pd.DataFrame, best: dict) -> str:
    fast, slow = next_post_extremes()
    test = period_parts(data)["проверка"]
    return "\n".join(
        [
            "- Просмотры и пересылки сняты один раз, в одном из четырёх прогонов сбора, и не "
            "обновлялись; age_days — оценка по логам сбора (provenance/collection_time_report.md).",
            "- Цель v2 не снимает различий в возрасте постов внутри группы «канал × месяц × "
            "прогон»: доля high_views = 1 по четвертям относительного возраста (от младшей "
            f"к старшей) {quartile_shares('v2')}; «дни 25–31 минус дни 1–24» "
            f"{overall_late_gap('v2')} п. п. При медиане по неделе: {quartile_shares('week')} "
            f"и {overall_late_gap('week')} п. п.",
            f"- Посты моложе {MIN_AGE_DAYS} дней на момент сбора исключены, поэтому анализ "
            f"заканчивается {data['local_day'].max()}; проверочная часть — одно окно "
            f"{test['local_day'].min()} — {test['local_day'].max()}, "
            f"{format_ru(len(test))} постов.",
            "- Доля high_views = 1 связана с тем, как скоро канал публикует следующий пост: "
            f"{fast} в нижнем квартиле времени до следующего поста и {slow} в верхнем. Разница "
            "максимальной и минимальной доли по дням недели внутри квартилей этого времени — "
            f"{spread_text('weekday')}, по часам — {spread_text('hour')}.",
            "- keyword_pass, p_on_topic, language и actor_type получены до этого проекта; их "
            "разметка здесь не проверялась.",
            "- Сравнения долей не учитывают другие различия постов (длина, время публикации, "
            "формат, частота публикаций канала); это связь, а не причинное влияние.",
            "- z-тест и 95 % ДИ считают посты независимыми, хотя посты сгруппированы по каналам. "
            f"Знаковый тест по каналам: государственные — {sign_text('government')}; блогеры — "
            f"{sign_text('blogger')}.",
            "- Разница «государственные против блогеров» зависит от определения экономического "
            f"поста: keyword_pass — {keyword_gap('keyword_pass', 'government')} против "
            f"{keyword_gap('keyword_pass', 'blogger')} п. п.; верхняя треть p_on_topic (общий "
            f"порог) — {keyword_gap('p_on_topic_top_global', 'government')} против "
            f"{keyword_gap('p_on_topic_top_global', 'blogger')} п. п.",
            f"- Три лучшие модели по CV: {top_b_models(best)}.",
            "- Таблицы терминов — коэффициенты логистической регрессии с регуляризацией; после "
            "очистки часть шаблонных строк каналов может остаться.",
            "- Расчёты выполнены на одном компьютере: Windows 10, Python 3.14.2, модели "
            "с n_jobs = −1 (все ядра процессора).",
        ]
    )


def build_summary(data: pd.DataFrame) -> str:
    stats = json.loads(DATASET_STATS_PATH.read_text("utf-8"))
    best = json.loads((RESULTS_DIR / "best_config.json").read_text("utf-8"))
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    groups = pd.read_csv(RESULTS_DIR / "metrics_by_group.csv")
    importance = pd.read_csv(RESULTS_DIR / "permutation_importance.csv")
    primary, second = TARGETS
    n_features = len(feature_columns(data))
    best_label = f"{MODEL_LABELS[best['model']]}, набор {best['feature_set']}"
    sections = [
        "# Сводка расчётов",
        "## Данные",
        data_facts(data, stats),
        "### Посты по слоям и языкам",
        layer_language_table(data),
        "### Каналы, посты и медианы по слоям",
        layer_table(data),
        "### Баланс классов",
        "Класс 1: значение строго выше медианы своей группы «канал × месяц × прогон сбора».",
        class_balance_table(data),
        "### Доля high_views = 1 среди экономических и остальных постов",
        "Все посты в анализе; экономические — keyword_pass = 1.",
        topic_table(data),
        "## Модели",
        method_facts(best, n_features),
        f"### Метрики, цель {primary}, проверочная часть",
        metrics_table(metrics[metrics["target"] == primary], best),
        f"### Лучшая конфигурация, повтор для цели {second}",
        f"Конфигурация: {best_label}, параметры `{json.dumps(best['params'])}`.",
        metrics_table(metrics[metrics["target"] == second], best),
        f"### Метрики по группам, {best_label}, цель {primary}, проверочная часть",
        group_table(groups),
        f"### {N_TOP_FEATURES} главных признаков (перестановочная важность)",
        f"{best_label}, цель {primary}, проверочная часть, "
        f"{N_PERMUTATION_REPEATS} перестановок на признак. "
        "Компоненты SVD переставляются вместе как один признак «текст».",
        importance_table(importance),
        *diagnostics_sections(),
        *robustness_sections("##"),
        "## Ограничения",
        limitations_section(data, best),
        "## Файлы",
        "metrics.csv, cv_results.csv, metrics_by_group.csv, permutation_importance.csv, "
        "best_config.json, dataset_stats.json, high_views_by_time.csv, "
        "high_views_weekday_hour.csv, high_views_by_age.csv, top_terms.csv, "
        "term_boilerplate.csv, keyword_comparison.csv, channel_sign_test.csv, "
        "monthly_medians.csv, density_*.csv, density_stats.json, robustness.md, "
        "robustness_*.csv, robustness_stats.json, v1_target_diagnostics.md, v1_vs_v2.md, "
        "confusion_matrix.png, roc_curve.png, permutation_importance.png, views_by_layer.png, "
        "weekday_hour_heatmap.png.",
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    apply_style()
    data = pd.read_parquet(FEATURES_PATH)
    plot_views_by_layer(data)
    SUMMARY_PATH.write_text(build_summary(data), "utf-8")
    print(f"Записано: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
