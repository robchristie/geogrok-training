from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from geogrok.retrieval.baseline import row_normalize


@dataclass(frozen=True)
class PairRetrievalReport:
    protocol: str
    query_splits: tuple[str, ...]
    gallery_splits: tuple[str, ...]
    query_count: int
    gallery_count: int
    pair_rows_used: int
    positive_exact_pairs: int
    positive_weak_pairs: int
    hard_negative_pairs: int
    queries_with_pair_labels: int
    queries_evaluated_exact: int
    queries_evaluated_any: int
    queries_with_hard_negatives: int
    exact_recall_at_1: float
    exact_recall_at_5: float
    exact_recall_at_10: float
    any_recall_at_1: float
    any_recall_at_5: float
    any_recall_at_10: float
    exact_mean_reciprocal_rank: float
    any_mean_reciprocal_rank: float
    hard_negative_at_1_rate: float
    hard_negative_in_top_5_rate: float


def chip_ids_from_pairs(pairs: pd.DataFrame) -> set[str]:
    if pairs.empty:
        return set()
    query_ids = pairs["query_chip_id"].fillna("null").astype(str)
    candidate_ids = pairs["candidate_chip_id"].fillna("null").astype(str)
    return set(query_ids).union(set(candidate_ids))


def evaluate_pair_retrieval(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    query_splits: Sequence[str],
    gallery_splits: Sequence[str],
) -> PairRetrievalReport:
    if len(embeddings) != len(metadata):
        raise ValueError("Embedding matrix and metadata row count must match.")
    if len(embeddings) == 0 or pairs.empty:
        return empty_report(query_splits=query_splits, gallery_splits=gallery_splits)

    frame = metadata.reset_index(drop=True).copy()
    frame["chip_id"] = frame["chip_id"].astype(str)
    frame["split_normalized"] = frame["split"].astype(str).str.lower()
    chip_index = {chip_id: int(index) for index, chip_id in enumerate(frame["chip_id"])}

    query_tokens = {split.lower() for split in query_splits}
    gallery_tokens = {split.lower() for split in gallery_splits}
    query_indices = [
        int(index)
        for index in frame.index
        if str(frame.iloc[index]["split_normalized"]) in query_tokens
    ]
    gallery_indices = [
        int(index)
        for index in frame.index
        if str(frame.iloc[index]["split_normalized"]) in gallery_tokens
    ]
    pair_frame = pairs.copy()
    pair_frame["query_chip_id"] = pair_frame["query_chip_id"].astype(str)
    pair_frame["candidate_chip_id"] = pair_frame["candidate_chip_id"].astype(str)
    pair_frame["query_index"] = pair_frame["query_chip_id"].map(chip_index)
    pair_frame["candidate_index"] = pair_frame["candidate_chip_id"].map(chip_index)
    pair_frame = pair_frame.dropna(subset=["query_index", "candidate_index"]).copy()
    if pair_frame.empty:
        return empty_report(
            query_splits=query_splits,
            gallery_splits=gallery_splits,
            query_count=len(query_indices),
            gallery_count=len(gallery_indices),
        )
    pair_frame["query_index"] = pair_frame["query_index"].astype(int)
    pair_frame["candidate_index"] = pair_frame["candidate_index"].astype(int)
    pair_frame = pair_frame[
        pair_frame["query_index"].isin(query_indices)
        & pair_frame["candidate_index"].isin(gallery_indices)
    ].reset_index(drop=True)
    if pair_frame.empty:
        return empty_report(
            query_splits=query_splits,
            gallery_splits=gallery_splits,
            query_count=len(query_indices),
            gallery_count=len(gallery_indices),
        )

    matrix = row_normalize(np.asarray(embeddings, dtype=np.float32))
    similarity = matrix @ matrix.T

    exact_recalls = {1: 0, 5: 0, 10: 0}
    any_recalls = {1: 0, 5: 0, 10: 0}
    exact_rrs: list[float] = []
    any_rrs: list[float] = []
    hard_negative_at_1 = 0
    hard_negative_top_5 = 0
    queries_with_pair_labels = 0
    queries_evaluated_exact = 0
    queries_evaluated_any = 0
    queries_with_hard_negatives = 0

    for query_index, query_pairs in pair_frame.groupby("query_index", sort=True):
        query_index = int(query_index)
        exact_positive_indices = set(
            query_pairs.loc[
                query_pairs["pair_label"] == "positive_exact",
                "candidate_index",
            ].astype(int)
        )
        weak_positive_indices = set(
            query_pairs.loc[
                query_pairs["pair_label"] == "positive_weak",
                "candidate_index",
            ].astype(int)
        )
        any_positive_indices = exact_positive_indices.union(weak_positive_indices)
        hard_negative_indices = set(
            query_pairs.loc[
                query_pairs["pair_label"] == "negative_hard",
                "candidate_index",
            ].astype(int)
        )
        if not exact_positive_indices and not any_positive_indices and not hard_negative_indices:
            continue

        queries_with_pair_labels += 1
        ranking = sorted(
            [candidate for candidate in gallery_indices if candidate != query_index],
            key=lambda candidate: float(similarity[query_index, candidate]),
            reverse=True,
        )

        if exact_positive_indices:
            queries_evaluated_exact += 1
            first_exact_rank = first_rank(ranking, exact_positive_indices)
            if first_exact_rank is not None:
                exact_rrs.append(1.0 / first_exact_rank)
                for k in exact_recalls:
                    if first_exact_rank <= k:
                        exact_recalls[k] += 1

        if any_positive_indices:
            queries_evaluated_any += 1
            first_any_rank = first_rank(ranking, any_positive_indices)
            if first_any_rank is not None:
                any_rrs.append(1.0 / first_any_rank)
                for k in any_recalls:
                    if first_any_rank <= k:
                        any_recalls[k] += 1

        if hard_negative_indices:
            queries_with_hard_negatives += 1
            if ranking and ranking[0] in hard_negative_indices:
                hard_negative_at_1 += 1
            if any(candidate in hard_negative_indices for candidate in ranking[:5]):
                hard_negative_top_5 += 1

    positive_exact_pairs = int((pair_frame["pair_label"] == "positive_exact").sum())
    positive_weak_pairs = int((pair_frame["pair_label"] == "positive_weak").sum())
    hard_negative_pairs = int((pair_frame["pair_label"] == "negative_hard").sum())
    return PairRetrievalReport(
        protocol="pairs",
        query_splits=tuple(query_splits),
        gallery_splits=tuple(gallery_splits),
        query_count=len(query_indices),
        gallery_count=len(gallery_indices),
        pair_rows_used=int(len(pair_frame)),
        positive_exact_pairs=positive_exact_pairs,
        positive_weak_pairs=positive_weak_pairs,
        hard_negative_pairs=hard_negative_pairs,
        queries_with_pair_labels=queries_with_pair_labels,
        queries_evaluated_exact=queries_evaluated_exact,
        queries_evaluated_any=queries_evaluated_any,
        queries_with_hard_negatives=queries_with_hard_negatives,
        exact_recall_at_1=safe_fraction(exact_recalls[1], queries_evaluated_exact),
        exact_recall_at_5=safe_fraction(exact_recalls[5], queries_evaluated_exact),
        exact_recall_at_10=safe_fraction(exact_recalls[10], queries_evaluated_exact),
        any_recall_at_1=safe_fraction(any_recalls[1], queries_evaluated_any),
        any_recall_at_5=safe_fraction(any_recalls[5], queries_evaluated_any),
        any_recall_at_10=safe_fraction(any_recalls[10], queries_evaluated_any),
        exact_mean_reciprocal_rank=safe_mean(exact_rrs),
        any_mean_reciprocal_rank=safe_mean(any_rrs),
        hard_negative_at_1_rate=safe_fraction(hard_negative_at_1, queries_with_hard_negatives),
        hard_negative_in_top_5_rate=safe_fraction(
            hard_negative_top_5,
            queries_with_hard_negatives,
        ),
    )


def first_rank(ranking: Sequence[int], positives: set[int]) -> int | None:
    for rank, candidate in enumerate(ranking, start=1):
        if candidate in positives:
            return rank
    return None


def safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def empty_report(
    *,
    query_splits: Sequence[str],
    gallery_splits: Sequence[str],
    query_count: int = 0,
    gallery_count: int = 0,
) -> PairRetrievalReport:
    return PairRetrievalReport(
        protocol="pairs",
        query_splits=tuple(query_splits),
        gallery_splits=tuple(gallery_splits),
        query_count=query_count,
        gallery_count=gallery_count,
        pair_rows_used=0,
        positive_exact_pairs=0,
        positive_weak_pairs=0,
        hard_negative_pairs=0,
        queries_with_pair_labels=0,
        queries_evaluated_exact=0,
        queries_evaluated_any=0,
        queries_with_hard_negatives=0,
        exact_recall_at_1=0.0,
        exact_recall_at_5=0.0,
        exact_recall_at_10=0.0,
        any_recall_at_1=0.0,
        any_recall_at_5=0.0,
        any_recall_at_10=0.0,
        exact_mean_reciprocal_rank=0.0,
        any_mean_reciprocal_rank=0.0,
        hard_negative_at_1_rate=0.0,
        hard_negative_in_top_5_rate=0.0,
    )
