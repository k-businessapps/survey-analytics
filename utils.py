from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


RATINGS_ENDPOINT = "/api/sync/ratings/by-date-range"
DISPOSITIONS_ENDPOINT = "/api/sync/dispositions/by-date-range"


@dataclass
class MetricResult:
    score: Optional[float]
    answered: int
    extra: Dict[str, Any]


def combine_date_and_time(value: date, is_end: bool = False) -> str:
    if is_end:
        dt = datetime.combine(value, time(23, 59, 59))
    else:
        dt = datetime.combine(value, time(0, 0, 0))
    return dt.isoformat()


def fetch_all_pages(
    api_key: str,
    base_url: str,
    endpoint: str,
    start_date: str,
    end_date: str,
    page_size: int = 100,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {"X-api-Key": api_key, "Accept": "application/json"}

    all_items: List[Dict[str, Any]] = []
    with requests.Session() as session:
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "page": 1,
            "size": page_size,
        }
        response = session.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected response format from {endpoint}: {type(payload)}")
        if "items" not in payload:
            raise ValueError(f"Missing 'items' in response from {endpoint}")

        total_pages = payload.get("pages", 1)
        if not isinstance(total_pages, int):
            raise ValueError(f"Invalid 'pages' value from {endpoint}: {total_pages}")

        all_items.extend(payload.get("items", []))

        for page in range(2, total_pages + 1):
            params["page"] = page
            response = session.get(url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            page_payload = response.json()
            page_items = page_payload.get("items", [])
            if not isinstance(page_items, list):
                raise ValueError(f"Expected list in 'items' for {endpoint} page {page}")
            all_items.extend(page_items)

    return all_items


def build_chat_link(website_id: Any, session_id: Any, existing_url: Any = None) -> str:
    if isinstance(existing_url, str) and existing_url.strip():
        return existing_url.strip()
    if pd.isna(website_id) or pd.isna(session_id):
        return ""
    return f"https://app.crisp.chat/website/{website_id}/inbox/{session_id}/"


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")



def dedupe_ratings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.copy()
    work["updated_at_dt"] = _parse_datetime(work.get("updated_at"))
    work["created_at_dt"] = _parse_datetime(work.get("created_at"))
    work["_sort_dt"] = work["updated_at_dt"].fillna(work["created_at_dt"])
    work = work.sort_values(["session_id", "_sort_dt", "id"], ascending=[True, True, True])
    work = work.drop_duplicates(subset=["session_id"], keep="last")
    return work.drop(columns=["_sort_dt"], errors="ignore")


def prepare_ratings_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        columns = [
            "chat_url",
            "created_at",
            "csat",
            "email",
            "fcr",
            "id",
            "last_operator_name",
            "name",
            "nps",
            "operator_list",
            "remarks",
            "session_id",
            "updated_at",
            "website_id",
        ]
        df = pd.DataFrame(columns=columns)

    df = dedupe_ratings(df)
    df["created_at_dt"] = _parse_datetime(df.get("created_at"))
    df["updated_at_dt"] = _parse_datetime(df.get("updated_at"))
    df["nps"] = pd.to_numeric(df.get("nps"), errors="coerce")
    df["csat"] = pd.to_numeric(df.get("csat"), errors="coerce")
    df["fcr_normalized"] = df.get("fcr", pd.Series(index=df.index, dtype="object")).fillna("").astype(str).str.strip().str.lower()
    df["chat_link"] = [
        build_chat_link(w, s, c)
        for w, s, c in zip(df.get("website_id"), df.get("session_id"), df.get("chat_url"))
    ]
    return df


def prepare_dispositions_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        columns = [
            "category",
            "chat_url",
            "created_at",
            "dff",
            "email",
            "final_notes",
            "flc_type",
            "id",
            "last_operator_name",
            "name",
            "operator_list",
            "session_id",
            "status",
            "sub_type",
            "transcript",
            "type",
            "updated_at",
            "website_id",
        ]
        df = pd.DataFrame(columns=columns)

    df["created_at_dt"] = _parse_datetime(df.get("created_at"))
    df["updated_at_dt"] = _parse_datetime(df.get("updated_at"))
    df["chat_link"] = [
        build_chat_link(w, s, c)
        for w, s, c in zip(df.get("website_id"), df.get("session_id"), df.get("chat_url"))
    ]
    return df


def build_merged_df(dispositions_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    disp = dispositions_df.copy().rename(
        columns={
            "id": "disposition_id",
            "chat_url": "disposition_chat_url",
            "created_at": "disposition_created_at",
            "updated_at": "disposition_updated_at",
            "created_at_dt": "disposition_created_at_dt",
            "updated_at_dt": "disposition_updated_at_dt",
            "last_operator_name": "disposition_last_operator_name",
            "operator_list": "disposition_operator_list",
            "name": "disposition_name",
            "email": "disposition_email",
            "website_id": "disposition_website_id",
            "chat_link": "disposition_chat_link",
        }
    )

    ratings = ratings_df.copy().rename(
        columns={
            "id": "rating_id",
            "chat_url": "rating_chat_url",
            "created_at": "rating_created_at",
            "updated_at": "rating_updated_at",
            "created_at_dt": "rating_created_at_dt",
            "updated_at_dt": "rating_updated_at_dt",
            "last_operator_name": "rating_last_operator_name",
            "operator_list": "rating_operator_list",
            "name": "rating_name",
            "email": "rating_email",
            "website_id": "rating_website_id",
            "chat_link": "rating_chat_link",
        }
    )

    merged = disp.merge(ratings, on="session_id", how="left")
    merged["chat_link"] = merged["disposition_chat_link"].fillna(merged["rating_chat_link"])
    merged["chat_link"] = merged["chat_link"].fillna(
        merged.apply(
            lambda row: build_chat_link(
                row.get("disposition_website_id") or row.get("rating_website_id"),
                row.get("session_id"),
                None,
            ),
            axis=1,
        )
    )
    return merged


def filter_date_range(
    df: pd.DataFrame,
    column: str,
    start: Optional[date],
    end: Optional[date],
) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    work = df.copy()
    series = work[column]
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = _parse_datetime(series)
    mask = pd.Series(True, index=work.index)
    if start is not None:
        start_dt = pd.Timestamp(start).tz_localize("UTC")
        mask &= series >= start_dt
    if end is not None:
        end_dt = pd.Timestamp(end).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= series <= end_dt
    return work[mask].copy()



def filter_multiselect(df: pd.DataFrame, column: str, selected: Iterable[str]) -> pd.DataFrame:
    selected_list = [value for value in selected if value not in (None, "", "All")]
    if df.empty or column not in df.columns or not selected_list:
        return df
    return df[df[column].isin(selected_list)].copy()


COMPARISON_OPERATORS = {
    "Any": None,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "=": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
}


def apply_numeric_filter(df: pd.DataFrame, column: str, operator_label: str, value: Optional[float]) -> pd.DataFrame:
    if df.empty or column not in df.columns or operator_label == "Any" or value is None:
        return df
    series = pd.to_numeric(df[column], errors="coerce")
    comparator = COMPARISON_OPERATORS[operator_label]
    mask = comparator(series, value)
    return df[mask.fillna(False)].copy()



def normalize_text_options(series: pd.Series) -> List[str]:
    if series is None or series.empty:
        return []
    values = (
        series.dropna()
        .astype(str)
        .map(str.strip)
    )
    values = values[values != ""]
    return sorted(values.unique().tolist())



def calculate_nps(df: pd.DataFrame, column: str = "nps") -> MetricResult:
    if df.empty or column not in df.columns:
        return MetricResult(score=None, answered=0, extra={"promoters": 0, "neutrals": 0, "detractors": 0})

    valid = pd.to_numeric(df[column], errors="coerce").dropna()
    if valid.empty:
        return MetricResult(score=None, answered=0, extra={"promoters": 0, "neutrals": 0, "detractors": 0})

    promoters = int((valid == 5).sum())
    neutrals = int((valid == 4).sum())
    detractors = int((valid <= 3).sum())
    score = ((promoters - detractors) / len(valid)) * 100
    return MetricResult(
        score=round(score, 1),
        answered=int(len(valid)),
        extra={"promoters": promoters, "neutrals": neutrals, "detractors": detractors},
    )



def calculate_csat(df: pd.DataFrame, column: str = "csat") -> MetricResult:
    if df.empty or column not in df.columns:
        return MetricResult(score=None, answered=0, extra={"satisfied": 0, "neutral": 0, "dissatisfied": 0})

    valid = pd.to_numeric(df[column], errors="coerce").dropna()
    if valid.empty:
        return MetricResult(score=None, answered=0, extra={"satisfied": 0, "neutral": 0, "dissatisfied": 0})

    satisfied = int(valid.isin([4, 5]).sum())
    neutral = int((valid == 3).sum())
    dissatisfied = int(valid.isin([0, 1, 2]).sum())
    score = (satisfied / len(valid)) * 100
    average = float(valid.mean()) if len(valid) else None
    return MetricResult(
        score=round(score, 1),
        answered=int(len(valid)),
        extra={
            "satisfied": satisfied,
            "neutral": neutral,
            "dissatisfied": dissatisfied,
            "average": round(average, 2) if average is not None else None,
        },
    )



def calculate_fcr(df: pd.DataFrame, column: str = "fcr_normalized") -> MetricResult:
    if df.empty or column not in df.columns:
        return MetricResult(score=None, answered=0, extra={"yes": 0, "no": 0, "partial": 0})

    valid = df[column].fillna("").astype(str).str.strip().str.lower()
    valid = valid[valid != ""]
    if valid.empty:
        return MetricResult(score=None, answered=0, extra={"yes": 0, "no": 0, "partial": 0})

    yes_count = int((valid == "yes").sum())
    no_count = int((valid == "no").sum())
    partial_count = int((valid == "partial").sum())
    score = (yes_count / len(valid)) * 100
    return MetricResult(
        score=round(score, 1),
        answered=int(len(valid)),
        extra={"yes": yes_count, "no": no_count, "partial": partial_count},
    )



def build_distribution_df(metric_name: str, values: Dict[str, Any], answered: int) -> pd.DataFrame:
    data = []
    for label, count in values.items():
        if label == "average":
            continue
        pct = round((count / answered) * 100, 1) if answered else 0.0
        data.append({"metric": metric_name, "label": label.replace("_", " ").title(), "count": count, "percentage": pct})
    return pd.DataFrame(data)



def build_csat_score_distribution(df: pd.DataFrame, column: str = "csat") -> pd.DataFrame:
    valid = pd.to_numeric(df.get(column), errors="coerce").dropna()
    if valid.empty:
        return pd.DataFrame(columns=["score", "count"])
    counts = valid.value_counts().sort_index()
    out = counts.rename_axis("score").reset_index(name="count")
    out["score"] = out["score"].astype(int)
    return out



def paginate_dataframe(df: pd.DataFrame, page: int, page_size: int) -> Tuple[pd.DataFrame, int]:
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx].copy(), total_pages



def metric_tone(metric: str, score: Optional[float]) -> str:
    if score is None:
        return "neutral"
    if metric == "nps":
        if score < 0:
            return "poor"
        if score < 50:
            return "good"
        return "excellent"
    if metric == "csat":
        if score < 70:
            return "poor"
        if score < 85:
            return "good"
        return "excellent"
    if metric == "fcr":
        if score < 60:
            return "poor"
        if score < 80:
            return "good"
        return "excellent"
    return "neutral"
