from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def find_rank_start(df: pd.DataFrame) -> int:
    for index, column in enumerate(df.columns):
        if str(column) == "0":
            return index
    raise ValueError("CARE results DataFrame does not contain rank column '0'")


def get_accuracy_level(predicted_ecs: Iterable[str], true_ecs: Iterable[str]) -> list[int]:
    predicted = [str(ec) for ec in predicted_ecs if pd.notna(ec)]
    true = [str(ec) for ec in true_ecs if pd.notna(ec)]
    if not predicted:
        predicted = ["0.0.0.0"]

    levels = []
    for true_ec in true:
        true_split = true_ec.split(".")
        counters = []
        for predicted_ec in predicted:
            if predicted_ec.count(".") != 3:
                predicted_ec = "0.0.0.0"
            predicted_split = predicted_ec.split(".")
            counter = 0
            for predicted_part, true_part in zip(predicted_split, true_split):
                if predicted_part == true_part:
                    counter += 1
                else:
                    break
            counters.append(counter)
        levels.append(int(np.max(counters)) if counters else 0)
    return levels


def average_accuracy(levels: list[int], level: int) -> float:
    if not levels:
        return 0.0
    return float(np.mean([1 if value >= level else 0 for value in levels]))


def compute_care_metrics(care_df: pd.DataFrame, k_values: tuple[int, ...] = (1, 20)) -> dict:
    rank_start = find_rank_start(care_df)
    ranked = care_df.copy()
    ranked.iloc[:, rank_start:] = ranked.iloc[:, rank_start:].fillna("0.0.0.0")

    metrics: dict[str, dict[str, float]] = {}
    for k in k_values:
        rows = []
        for _, row in ranked.iterrows():
            true_ecs = str(row["EC number"]).split(";")
            predicted = list(row.iloc[rank_start : rank_start + k])
            predicted = predicted[: len(true_ecs)]
            levels = get_accuracy_level(predicted, true_ecs)
            rows.append(levels)

        metrics[f"k={k}"] = {
            f"level_{level}_accuracy": round(
                float(np.mean([average_accuracy(levels, level) for levels in rows])) * 100.0,
                4,
            )
            for level in (4, 3, 2, 1)
        }
    return metrics


def compute_supplemental_ranking_metrics(
    care_df: pd.DataFrame,
    *,
    hit_ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict:
    rank_start = find_rank_start(care_df)
    rank_cols = care_df.columns[rank_start:]
    reciprocal_ranks = []
    hits = {k: [] for k in hit_ks}

    for row in care_df.itertuples(index=False):
        true_ecs = set(str(row[1]).split(";"))
        ranked_ecs = [str(value) for value in row[rank_start:] if pd.notna(value)]
        first_rank = None
        for rank, ec_number in enumerate(ranked_ecs, start=1):
            if ec_number in true_ecs:
                first_rank = rank
                break
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        for k in hit_ks:
            hits[k].append(any(ec_number in true_ecs for ec_number in ranked_ecs[:k]))

    return {
        "mrr": round(float(np.mean(reciprocal_ranks)), 6) if reciprocal_ranks else 0.0,
        "rank_columns": int(len(rank_cols)),
        **{f"hit@{k}": round(float(np.mean(values)) * 100.0, 4) for k, values in hits.items()},
    }


def write_care_ranked_csv(
    eval_dist_df: pd.DataFrame,
    clean_eval_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    clean_df = pd.read_csv(clean_eval_csv, sep="\t")
    rank_rows = []
    for entry in clean_df["Entry"]:
        if entry not in eval_dist_df.columns:
            raise KeyError(f"Missing evaluation distances for Entry={entry}")
        rank_rows.append(list(eval_dist_df[entry].sort_values(ascending=True).index))

    max_ranks = max((len(row) for row in rank_rows), default=0)
    rank_df = pd.DataFrame(rank_rows, columns=[str(i) for i in range(max_ranks)])
    care_df = pd.concat([clean_df.reset_index(drop=True), rank_df], axis=1)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    care_df.to_csv(output_csv, index=False)
    return care_df
