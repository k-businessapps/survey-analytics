
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from utils import (
    DISPOSITIONS_ENDPOINT,
    RATINGS_ENDPOINT,
    apply_numeric_filter,
    build_merged_df,
    combine_date_and_time,
    fetch_all_pages,
    filter_date_range,
    filter_multiselect,
    normalize_text_options,
    paginate_dataframe,
    prepare_dispositions_df,
    prepare_ratings_df,
)


st.set_page_config(
    page_title="KrispCall Survey Analytics",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "assets" / "logo.png"

BRAND = {
    "primary": "#B14CF5",
    "primary_dark": "#7E22CE",
    "secondary": "#E879F9",
    "ink": "#241138",
    "muted": "#6B5A7B",
    "border": "rgba(177, 76, 245, 0.18)",
    "bg": "#FBF7FF",
    "card": "#FFFFFF",
    "good": "#7E22CE",
    "excellent": "#B14CF5",
    "poor": "#EF4444",
    "neutral": "#A78BFA",
}


@dataclass
class MetricResult:
    score: Optional[float]
    answered: int
    extra: Dict[str, Any]


@st.cache_data(show_spinner=False)
def fetch_support_data(
    api_key: str,
    base_url: str,
    start_iso: str,
    end_iso: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings_rows = fetch_all_pages(api_key, base_url, RATINGS_ENDPOINT, start_iso, end_iso)
    dispositions_rows = fetch_all_pages(api_key, base_url, DISPOSITIONS_ENDPOINT, start_iso, end_iso)
    ratings_df = prepare_ratings_df(ratings_rows)
    dispositions_df = prepare_dispositions_df(dispositions_rows)
    merged_df = build_merged_df(dispositions_df, ratings_df)
    return ratings_df, dispositions_df, merged_df


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, #fff 0%, {BRAND['bg']} 100%);
            }}
            .block-container {{
                padding-top: 3.2rem;
                padding-bottom: 2rem;
                max-width: 1500px;
            }}
            .brand-shell {{
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 0.6rem 0 0.8rem 0;
            }}
            .brand-title {{
                font-size: 1.65rem;
                font-weight: 800;
                color: {BRAND['ink']};
                line-height: 1.15;
                margin: 0;
            }}
            .brand-subtitle {{
                color: {BRAND['muted']};
                font-size: 0.96rem;
                margin-top: 0.2rem;
            }}
            .metric-card {{
                background: {BRAND['card']};
                border: 1px solid {BRAND['border']};
                border-left: 6px solid {BRAND['neutral']};
                border-radius: 18px;
                padding: 1rem 1rem 0.95rem 1rem;
                box-shadow: 0 10px 35px rgba(55, 25, 78, 0.06);
                min-height: 150px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}
            .metric-card.poor {{ border-left-color: {BRAND['poor']}; }}
            .metric-card.good {{ border-left-color: {BRAND['good']}; }}
            .metric-card.excellent {{ border-left-color: {BRAND['excellent']}; }}
            .metric-label {{
                color: {BRAND['muted']};
                text-transform: uppercase;
                letter-spacing: 0.06em;
                font-size: 0.78rem;
                font-weight: 700;
            }}
            .metric-value {{
                color: {BRAND['ink']};
                font-size: 2rem;
                font-weight: 800;
                margin-top: 0.2rem;
                line-height: 1.05;
            }}
            .metric-sub {{
                color: {BRAND['muted']};
                font-size: 0.92rem;
                margin-top: 0.45rem;
            }}
            .note-card {{
                background: rgba(177, 76, 245, 0.06);
                border: 1px solid {BRAND['border']};
                border-radius: 16px;
                padding: 0.9rem 1rem;
                color: {BRAND['ink']};
            }}
            div[data-testid="stSidebar"] > div:first-child {{
                background: linear-gradient(180deg, #fff 0%, #f8f0ff 100%);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    logo_col, text_col = st.columns([1, 7], vertical_alignment="center")
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=110)
    with text_col:
        st.markdown(
            """
            <div class="brand-shell">
                <div>
                    <div class="brand-title">KrispCall Survey Analytics</div>
                    <div class="brand-subtitle">Support survey performance across NPS, CSAT, FCR, ratings, and dispositions.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def check_auth() -> bool:
    if st.session_state.get("authenticated"):
        return True

    auth_secrets = st.secrets.get("auth", {})
    correct_username = auth_secrets.get("username", "")
    correct_password = auth_secrets.get("password", "")

    shell_left, shell_mid, shell_right = st.columns([1.2, 1.8, 1.2])
    with shell_mid:
        st.markdown("### Sign in")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", width="stretch")

        if submitted:
            if username == correct_username and password == correct_password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
    return False


def get_api_config() -> Tuple[str, str]:
    api_secrets = st.secrets.get("api", {})
    api_key = api_secrets.get("key", "")
    base_url = api_secrets.get("base_url", "https://api.supportapps.krispcall.biz")
    return api_key, base_url


def metric_tone(metric_name: str, value: Optional[float]) -> str:
    if value is None:
        return "neutral"

    if metric_name in {"nps", "csat"}:
        if value < 0:
            return "poor"
        if value < 40:
            return "good"
        return "excellent"

    if metric_name == "fcr":
        if value < 60:
            return "poor"
        if value < 80:
            return "good"
        return "excellent"

    return "neutral"


def render_metric_card(title: str, value: Optional[float], suffix: str, subtitle: str, tone: str) -> None:
    if value is None:
        display_value = "N/A"
        suffix = ""
    else:
        display_value = f"{value:.1f}" if isinstance(value, float) else str(value)
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div>
                <div class="metric-label">{title}</div>
                <div class="metric-value">{display_value}{suffix}</div>
            </div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calculate_nps(df: pd.DataFrame) -> MetricResult:
    series = pd.to_numeric(df.get("nps"), errors="coerce").dropna()
    total = int(len(series))
    if total == 0:
        return MetricResult(score=None, answered=0, extra={})

    promoters = int((series == 5).sum())
    neutral = int((series == 4).sum())
    detractors = int((series <= 3).sum())
    score = ((promoters - detractors) / total) * 100

    return MetricResult(
        score=round(float(score), 1),
        answered=total,
        extra={"Promoters": promoters, "Neutral": neutral, "Detractors": detractors},
    )


def calculate_csat(df: pd.DataFrame) -> MetricResult:
    series = pd.to_numeric(df.get("csat"), errors="coerce").dropna()
    total = int(len(series))
    if total == 0:
        return MetricResult(score=None, answered=0, extra={})

    promoters = int((series == 5).sum())
    neutral = int((series == 4).sum())
    detractors = int((series <= 3).sum())
    score = ((promoters - detractors) / total) * 100

    return MetricResult(
        score=round(float(score), 1),
        answered=total,
        extra={"Promoters": promoters, "Neutral": neutral, "Detractors": detractors},
    )


def calculate_fcr(df: pd.DataFrame) -> MetricResult:
    if "fcr_normalized" not in df.columns:
        return MetricResult(score=None, answered=0, extra={})

    series = df["fcr_normalized"].fillna("").astype(str).str.strip().str.lower()
    series = series[series != ""]
    total = int(len(series))
    if total == 0:
        return MetricResult(score=None, answered=0, extra={})

    yes = int((series == "yes").sum())
    no = int((series == "no").sum())
    other = int(total - yes - no)
    score = (yes / total) * 100

    return MetricResult(
        score=round(float(score), 1),
        answered=total,
        extra={"Yes": yes, "No": no, "Other": other},
    )


def add_percentage_labels(df: pd.DataFrame, count_col: str = "count") -> pd.DataFrame:
    out = df.copy()
    total = pd.to_numeric(out.get(count_col), errors="coerce").fillna(0).sum()
    if total > 0:
        out["percentage"] = (pd.to_numeric(out[count_col], errors="coerce").fillna(0) / total * 100).round(1)
        out["percentage_label"] = out["percentage"].map(lambda x: f"{x:.1f}%")
    else:
        out["percentage"] = 0.0
        out["percentage_label"] = ""
    return out


def build_distribution_df(counts: Dict[str, int], answered: int) -> pd.DataFrame:
    rows = []
    for label, count in counts.items():
        pct = round((count / answered * 100), 1) if answered else 0
        rows.append({"label": label, "count": count, "percentage": pct})
    return pd.DataFrame(rows)


def build_metric_bar_chart(df: pd.DataFrame, title: str, color_mode: str = "tri") -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"label": [], "count": [], "percentage_label": []})).mark_bar()

    df = add_percentage_labels(df)

    if color_mode == "tri":
        color_scale = alt.Scale(
            domain=df["label"].tolist(),
            range=[BRAND["excellent"], BRAND["neutral"], BRAND["poor"]][: len(df)],
        )
        color = alt.Color("label:N", scale=color_scale, legend=None)
    else:
        color = alt.value(BRAND["primary"])

    base = alt.Chart(df)
    bars = (
        base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("label:N", sort=None, title=None),
            y=alt.Y("count:Q", title="Count"),
            color=color,
            tooltip=["label", "count", "percentage"],
        )
    )
    labels = (
        base.mark_text(dy=-10, fontSize=12, fontWeight="bold", color=BRAND["ink"])
        .encode(
            x=alt.X("label:N", sort=None),
            y=alt.Y("count:Q"),
            text=alt.Text("percentage_label:N"),
        )
    )
    return (bars + labels).properties(height=280, title=title)


def normalize_operator_first_name(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    s = s.replace("", pd.NA)
    return s.str.split().str[0]


def get_response_session_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    work = df.copy()
    work["nps_filled"] = pd.to_numeric(work.get("nps"), errors="coerce").notna()
    work["csat_filled"] = pd.to_numeric(work.get("csat"), errors="coerce").notna()
    if "fcr_normalized" in work.columns:
        work["fcr_filled"] = work["fcr_normalized"].fillna("").astype(str).str.strip().ne("")
    else:
        work["fcr_filled"] = False
    work["any_response"] = work["nps_filled"] | work["csat_filled"] | work["fcr_filled"]
    work = work[work["any_response"]].copy()
    if work.empty:
        return 0
    return int(work["session_id"].astype(str).nunique())


def calculate_breakdown_metrics_from_ratings(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(
            columns=[
                group_col,
                "nps_score",
                "csat_score",
                "fcr_score",
                "response_sessions",
                "answered_nps",
                "answered_csat",
                "answered_fcr",
            ]
        )

    work = df.copy()
    work = work[work[group_col].notna()].copy()
    work[group_col] = work[group_col].astype(str).str.strip()
    work = work[work[group_col] != ""].copy()
    if work.empty:
        return pd.DataFrame()

    work["nps_filled"] = pd.to_numeric(work.get("nps"), errors="coerce").notna()
    work["csat_filled"] = pd.to_numeric(work.get("csat"), errors="coerce").notna()
    work["fcr_filled"] = work["fcr_normalized"].fillna("").astype(str).str.strip().ne("")
    work["any_response"] = work["nps_filled"] | work["csat_filled"] | work["fcr_filled"]

    rows = []
    for group_value, grp in work.groupby(group_col, dropna=True):
        answered_nps = int(grp.loc[grp["nps_filled"], "session_id"].astype(str).nunique())
        answered_csat = int(grp.loc[grp["csat_filled"], "session_id"].astype(str).nunique())
        answered_fcr = int(grp.loc[grp["fcr_filled"], "session_id"].astype(str).nunique())
        response_sessions = int(grp.loc[grp["any_response"], "session_id"].astype(str).nunique())

        nps_result = calculate_nps(grp)
        csat_result = calculate_csat(grp)
        fcr_result = calculate_fcr(grp)

        rows.append(
            {
                group_col: group_value,
                "nps_score": nps_result.score,
                "csat_score": csat_result.score,
                "fcr_score": fcr_result.score,
                "response_sessions": response_sessions,
                "answered_nps": answered_nps,
                "answered_csat": answered_csat,
                "answered_fcr": answered_fcr,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["response_sessions", group_col], ascending=[False, True]).reset_index(drop=True)


def calculate_breakdown_metrics_from_merged(merged_df: pd.DataFrame, group_col: str, category_value: str) -> pd.DataFrame:
    if merged_df.empty or group_col not in merged_df.columns:
        return pd.DataFrame()

    work = merged_df.copy()
    if "category" in work.columns:
        work = work[work["category"].fillna("").astype(str) == str(category_value)].copy()
    if work.empty:
        return pd.DataFrame()

    keep_cols = ["session_id", group_col, "nps", "csat", "fcr_normalized"]
    keep_cols = [col for col in keep_cols if col in work.columns]
    work = work[keep_cols].copy()
    work[group_col] = work[group_col].fillna("").astype(str).str.strip()
    work = work[work[group_col] != ""].copy()
    work = work.drop_duplicates(subset=["session_id", group_col], keep="first").copy()

    work["nps_filled"] = pd.to_numeric(work.get("nps"), errors="coerce").notna()
    work["csat_filled"] = pd.to_numeric(work.get("csat"), errors="coerce").notna()
    work["fcr_filled"] = work["fcr_normalized"].fillna("").astype(str).str.strip().ne("")
    work["any_response"] = work["nps_filled"] | work["csat_filled"] | work["fcr_filled"]

    rows = []
    for group_value, grp in work.groupby(group_col, dropna=True):
        answered_nps = int(grp.loc[grp["nps_filled"], "session_id"].astype(str).nunique())
        answered_csat = int(grp.loc[grp["csat_filled"], "session_id"].astype(str).nunique())
        answered_fcr = int(grp.loc[grp["fcr_filled"], "session_id"].astype(str).nunique())
        response_sessions = int(grp.loc[grp["any_response"], "session_id"].astype(str).nunique())

        nps_result = calculate_nps(grp)
        csat_result = calculate_csat(grp)
        fcr_result = calculate_fcr(grp)

        rows.append(
            {
                group_col: group_value,
                "nps_score": nps_result.score,
                "csat_score": csat_result.score,
                "fcr_score": fcr_result.score,
                "response_sessions": response_sessions,
                "answered_nps": answered_nps,
                "answered_csat": answered_csat,
                "answered_fcr": answered_fcr,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["response_sessions", group_col], ascending=[False, True]).reset_index(drop=True)


def build_breakdown_metric_chart(df: pd.DataFrame, group_col: str, metric_col: str, title: str, color: str) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({group_col: [], metric_col: [], "label": []})).mark_bar()

    plot_df = df.copy()
    plot_df["label"] = plot_df[metric_col].apply(lambda x: "" if pd.isna(x) else f"{x:.1f}%")

    base = alt.Chart(plot_df)
    bars = (
        base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=color)
        .encode(
            x=alt.X(f"{group_col}:N", sort="-y", title=None),
            y=alt.Y(f"{metric_col}:Q", title="Percentage", scale=alt.Scale(domain=[-100, 100])),
            tooltip=[group_col, metric_col, "response_sessions", "answered_nps", "answered_csat", "answered_fcr"],
        )
    )
    labels = (
        base.mark_text(dy=-10, fontSize=12, fontWeight="bold", color=BRAND["ink"])
        .encode(
            x=alt.X(f"{group_col}:N", sort="-y"),
            y=alt.Y(f"{metric_col}:Q"),
            text=alt.Text("label:N"),
        )
    )
    return (bars + labels).properties(height=360, title=title)


def build_operator_metric_chart(df: pd.DataFrame, metric_col: str, title: str, color: str) -> alt.Chart:
    return build_breakdown_metric_chart(df, "operator_first_name", metric_col, title, color)


def get_default_date_bounds(df: pd.DataFrame, column: str) -> Tuple[date, date]:
    if df.empty or column not in df.columns or df[column].dropna().empty:
        today = date.today()
        return today, today
    series = pd.to_datetime(df[column], utc=True, errors="coerce").dropna()
    return series.min().date(), series.max().date()


def compute_overview_dataset(
    ratings_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    start_date: Optional[date],
    end_date: Optional[date],
    operators: list[str],
    categories: list[str],
    types: list[str],
    sub_types: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    ratings_filtered = filter_date_range(ratings_df, "created_at_dt", start_date, end_date)
    ratings_filtered = filter_multiselect(ratings_filtered, "last_operator_name", operators)

    disposition_filter_selected = any([categories, types, sub_types])
    if not disposition_filter_selected:
        return ratings_filtered, merged_df.iloc[0:0].copy(), False

    merged_filtered = merged_df.copy()
    merged_filtered = filter_date_range(merged_filtered, "rating_created_at_dt", start_date, end_date)
    merged_filtered = filter_multiselect(merged_filtered, "rating_last_operator_name", operators)
    merged_filtered = filter_multiselect(merged_filtered, "category", categories)
    merged_filtered = filter_multiselect(merged_filtered, "type", types)
    merged_filtered = filter_multiselect(merged_filtered, "sub_type", sub_types)

    matched_session_ids = merged_filtered["session_id"].dropna().astype(str).drop_duplicates().tolist()
    rating_scope = ratings_filtered[ratings_filtered["session_id"].astype(str).isin(matched_session_ids)].copy()
    return rating_scope, merged_filtered, True


def get_default_timeline_granularity(start_date: date, end_date: date) -> str:
    total_days = (end_date - start_date).days + 1
    if total_days < 14:
        return "Daily"
    if total_days < 60:
        return "Weekly"
    return "Monthly"


def build_timeline_df(ratings_df: pd.DataFrame, start_date: date, end_date: date, granularity: str) -> pd.DataFrame:
    df = filter_date_range(ratings_df, "created_at_dt", start_date, end_date).copy()
    if df.empty:
        return pd.DataFrame(columns=["period_start", "period_label", "metric", "score"])

    ts = pd.to_datetime(df["created_at_dt"], utc=True, errors="coerce").dt.tz_localize(None)
    df["_ts"] = ts
    df = df[df["_ts"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=["period_start", "period_label", "metric", "score"])

    if granularity == "Daily":
        df["period_start"] = df["_ts"].dt.normalize()
        full_periods = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
        period_labels = {pd.Timestamp(d): pd.Timestamp(d).strftime("%b %d") for d in full_periods}

    elif granularity == "Weekly":
        df["_day"] = df["_ts"].dt.normalize()
        df["period_start"] = df["_day"] - pd.to_timedelta(df["_day"].dt.weekday, unit="D")

        aligned_start = pd.Timestamp(start_date).normalize()
        aligned_start = aligned_start - pd.to_timedelta(aligned_start.weekday(), unit="D")

        aligned_end = pd.Timestamp(end_date).normalize()
        aligned_end = aligned_end - pd.to_timedelta(aligned_end.weekday(), unit="D")

        full_periods = pd.date_range(aligned_start, aligned_end, freq="7D")
        period_labels = {}
        for start in full_periods:
            end = start + pd.Timedelta(days=6)
            period_labels[pd.Timestamp(start)] = f"{start.strftime('%b %d')} - {end.strftime('%b %d')}"

    else:
        df["period_start"] = df["_ts"].dt.to_period("M").dt.to_timestamp()
        full_periods = pd.date_range(pd.Timestamp(start_date).replace(day=1), pd.Timestamp(end_date), freq="MS")
        period_labels = {pd.Timestamp(d): pd.Timestamp(d).strftime("%b %Y") for d in full_periods}

    rows = []
    for period_start, grp in df.groupby("period_start", dropna=True):
        nps_result = calculate_nps(grp)
        csat_result = calculate_csat(grp)
        fcr_result = calculate_fcr(grp)

        label = period_labels.get(pd.Timestamp(period_start), pd.Timestamp(period_start).strftime("%b %d"))

        rows.append({"period_start": period_start, "period_label": label, "metric": "NPS", "score": nps_result.score})
        rows.append({"period_start": period_start, "period_label": label, "metric": "CSAT", "score": csat_result.score})
        rows.append({"period_start": period_start, "period_label": label, "metric": "FCR", "score": fcr_result.score})

    scored_df = pd.DataFrame(rows)

    template_rows = []
    metric_names = ["NPS", "CSAT", "FCR"]
    for period_start in full_periods:
        label = period_labels.get(pd.Timestamp(period_start), pd.Timestamp(period_start).strftime("%b %d"))
        for metric_name in metric_names:
            template_rows.append(
                {
                    "period_start": pd.Timestamp(period_start),
                    "period_label": label,
                    "metric": metric_name,
                }
            )

    template = pd.DataFrame(template_rows)
    merged = template.merge(scored_df, on=["period_start", "period_label", "metric"], how="left")
    return merged

def build_timeline_chart(df: pd.DataFrame, selected_metrics: list[str], granularity: str) -> alt.Chart:
    plot_df = df[df["metric"].isin(selected_metrics)].copy()
    if plot_df.empty:
        return alt.Chart(pd.DataFrame({"period_label": [], "score": [], "metric": []})).mark_line()

    color_scale = alt.Scale(
        domain=["NPS", "CSAT", "FCR"],
        range=["#2563EB", "#F59E0B", "#10B981"],
    )

    wide_df = (
        plot_df.pivot(index=["period_start", "period_label"], columns="metric", values="score")
        .reset_index()
        .sort_values("period_start")
    )

    wide_df["y0"] = -100
    wide_df["y1"] = 100

    x_enc = alt.X(
        "period_label:N",
        sort=alt.SortField(field="period_start", order="ascending"),
        title=granularity,
        axis=alt.Axis(labelAngle=-35),
    )

    hover = alt.selection_point(
        fields=["period_label"],
        on="mouseover",
        empty=False,
        clear="mouseout",
    )

    base = alt.Chart(plot_df).encode(
        x=x_enc,
        y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[-100, 100])),
        color=alt.Color("metric:N", scale=color_scale, legend=alt.Legend(title="Metric")),
    )

    line = base.mark_line(strokeWidth=2.5)

    points = base.mark_point(size=80).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )

    tooltip_fields = [alt.Tooltip("period_label:N", title=granularity)]
    if "NPS" in selected_metrics:
        tooltip_fields.append(alt.Tooltip("NPS:Q", title="NPS", format=".1f"))
    if "CSAT" in selected_metrics:
        tooltip_fields.append(alt.Tooltip("CSAT:Q", title="CSAT", format=".1f"))
    if "FCR" in selected_metrics:
        tooltip_fields.append(alt.Tooltip("FCR:Q", title="FCR", format=".1f"))

    hover_band = (
        alt.Chart(wide_df)
        .mark_bar(opacity=0)
        .encode(
            x=x_enc,
            y=alt.Y("y0:Q", scale=alt.Scale(domain=[-100, 100]), title="Score"),
            y2="y1:Q",
            tooltip=tooltip_fields,
        )
        .add_params(hover)
    )

    rule = (
        alt.Chart(wide_df)
        .mark_rule(color="#94A3B8")
        .encode(x=x_enc)
        .transform_filter(hover)
    )

    return alt.layer(hover_band, line, points, rule).properties(height=360, title="Timeline")

def render_overview_tab(ratings_df: pd.DataFrame, merged_df: pd.DataFrame, fetched_start: date, fetched_end: date) -> None:
    st.subheader("Overview")
    st.caption("Metrics use ratings by default. When disposition filters are applied, the rating scope is narrowed through the merged table without double counting sessions.")

    with st.container(border=True):
        f1, f2, f3, f4, f5, f6 = st.columns([0.9, 0.9, 1.4, 1.2, 1.2, 1.2])
        with f1:
            start_date = st.date_input("FROM", value=fetched_start, min_value=fetched_start, max_value=fetched_end, key="ov_start")
        with f2:
            end_date = st.date_input("TO", value=fetched_end, min_value=fetched_start, max_value=fetched_end, key="ov_end")
        with f3:
            operators = st.multiselect("Last Operator", options=normalize_text_options(ratings_df["last_operator_name"]), key="ov_operators")
        with f4:
            categories = st.multiselect(
                "Disposition category",
                options=normalize_text_options(merged_df["category"] if "category" in merged_df else pd.Series(dtype="object")),
                key="ov_categories",
            )
        with f5:
            types = st.multiselect(
                "Disposition type",
                options=normalize_text_options(merged_df["type"] if "type" in merged_df else pd.Series(dtype="object")),
                key="ov_types",
            )
        with f6:
            sub_types = st.multiselect(
                "Disposition sub_type",
                options=normalize_text_options(merged_df["sub_type"] if "sub_type" in merged_df else pd.Series(dtype="object")),
                key="ov_sub_types",
            )

    metric_df, merged_filtered, using_dispositions = compute_overview_dataset(
        ratings_df, merged_df, start_date, end_date, operators, categories, types, sub_types
    )

    nps_result = calculate_nps(metric_df)
    csat_result = calculate_csat(metric_df)
    fcr_result = calculate_fcr(metric_df)
    response_sessions = get_response_session_count(metric_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("NPS score", nps_result.score, "%", f"Answered responses: {nps_result.answered}", metric_tone("nps", nps_result.score))
    with c2:
        render_metric_card("CSAT score", csat_result.score, "%", f"Answered responses: {csat_result.answered}", metric_tone("csat", csat_result.score))
    with c3:
        render_metric_card("FCR resolution", fcr_result.score, "%", f"Answered responses: {fcr_result.answered}", metric_tone("fcr", fcr_result.score))

    note = "Disposition filters are active. Metrics are being scoped through the merged table." if using_dispositions else "No disposition filters are active. Metrics are based on the ratings table only."
    st.markdown(f'<div class="note-card">{note}</div>', unsafe_allow_html=True)

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        nps_dist = build_distribution_df(nps_result.extra, nps_result.answered)
        st.altair_chart(build_metric_bar_chart(nps_dist, "NPS distribution"), width="stretch")
    with chart_col2:
        csat_dist = build_distribution_df(csat_result.extra, csat_result.answered)
        st.altair_chart(build_metric_bar_chart(csat_dist, "CSAT distribution"), width="stretch")
    with chart_col3:
        fcr_dist = build_distribution_df(fcr_result.extra, fcr_result.answered)
        st.altair_chart(build_metric_bar_chart(fcr_dist, "FCR distribution"), width="stretch")

    stats_left, stats_mid, stats_right = st.columns(3)
    stats_left.metric("Unique response sessions", f"{response_sessions:,}")
    stats_mid.metric("Distinct sessions in scope", f"{metric_df['session_id'].nunique():,}" if not metric_df.empty else "0")
    stats_right.metric("Merged rows matched", f"{len(merged_filtered):,}" if using_dispositions else "0")

    st.markdown("### Timeline")
    st.caption("This timeline is based on the overall fetched ratings data and does not use the Overview filters.")
    default_granularity = get_default_timeline_granularity(fetched_start, fetched_end)

    tl1, tl2 = st.columns([1.2, 2.4])
    with tl1:
        granularity = st.radio(
            "Group by",
            options=["Daily", "Weekly", "Monthly"],
            index=["Daily", "Weekly", "Monthly"].index(default_granularity),
            horizontal=True,
            key="timeline_granularity",
        )
    with tl2:
        selected_metrics = st.multiselect(
            "Show metrics",
            options=["NPS", "CSAT", "FCR"],
            default=["NPS", "CSAT", "FCR"],
            key="timeline_metrics",
        )

    timeline_df = build_timeline_df(ratings_df, fetched_start, fetched_end, granularity)
    if selected_metrics:
        st.altair_chart(build_timeline_chart(timeline_df, selected_metrics, granularity), width="stretch")
    else:
        st.info("Select at least one metric to display the timeline.")


def render_operator_tab(ratings_df: pd.DataFrame, fetched_start: date, fetched_end: date) -> None:
    st.subheader("By operator")
    st.caption("NPS, CSAT, and FCR percentages grouped by operator first name using last operator. Response counts use unique sessions with at least one of NPS, CSAT, or FCR filled.")

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            start_date = st.date_input("FROM", value=fetched_start, min_value=fetched_start, max_value=fetched_end, key="op_start")
        with c2:
            end_date = st.date_input("TO", value=fetched_end, min_value=fetched_start, max_value=fetched_end, key="op_end")
        with c3:
            operators = st.multiselect("Last Operator", options=normalize_text_options(ratings_df["last_operator_name"]), key="op_operators")

    filtered = filter_date_range(ratings_df, "created_at_dt", start_date, end_date)
    filtered = filter_multiselect(filtered, "last_operator_name", operators)
    filtered = filtered.copy()
    filtered["operator_first_name"] = normalize_operator_first_name(filtered["last_operator_name"])

    operator_df = calculate_breakdown_metrics_from_ratings(filtered, "operator_first_name")

    summary1, summary2, summary3 = st.columns(3)
    summary1.metric("Operators", f"{len(operator_df):,}")
    summary2.metric("Unique response sessions", f"{get_response_session_count(filtered):,}")
    summary3.metric("Distinct sessions in scope", f"{filtered['session_id'].nunique():,}" if not filtered.empty else "0")

    st.dataframe(
        operator_df,
        width="stretch",
        hide_index=True,
        column_config={
            "operator_first_name": st.column_config.TextColumn("Operator"),
            "nps_score": st.column_config.NumberColumn("NPS %", format="%.1f"),
            "csat_score": st.column_config.NumberColumn("CSAT %", format="%.1f"),
            "fcr_score": st.column_config.NumberColumn("FCR %", format="%.1f"),
            "response_sessions": st.column_config.NumberColumn("Response Sessions", format="%d"),
            "answered_nps": st.column_config.NumberColumn("Answered NPS", format="%d"),
            "answered_csat": st.column_config.NumberColumn("Answered CSAT", format="%d"),
            "answered_fcr": st.column_config.NumberColumn("Answered FCR", format="%d"),
        },
    )

    st.altair_chart(build_operator_metric_chart(operator_df, "nps_score", "NPS by operator", BRAND["primary_dark"]), width="stretch")
    st.altair_chart(build_operator_metric_chart(operator_df, "csat_score", "CSAT by operator", BRAND["primary"]), width="stretch")
    st.altair_chart(build_operator_metric_chart(operator_df, "fcr_score", "FCR by operator", BRAND["secondary"]), width="stretch")


def render_disposition_tab(merged_df: pd.DataFrame, fetched_start: date, fetched_end: date) -> None:
    st.subheader("By disposition")
    st.caption("Select one disposition category at a time, then review Type and Sub-Type breakdowns underneath it.")

    categories = normalize_text_options(merged_df["category"] if "category" in merged_df else pd.Series(dtype="object"))
    if not categories:
        st.info("No disposition categories are available in the current data.")
        return

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            start_date = st.date_input("FROM", value=fetched_start, min_value=fetched_start, max_value=fetched_end, key="disp_start")
        with c2:
            end_date = st.date_input("TO", value=fetched_end, min_value=fetched_start, max_value=fetched_end, key="disp_end")
        with c3:
            selected_category = st.selectbox("Category", options=categories, key="disp_category")

    filtered = filter_date_range(merged_df, "rating_created_at_dt", start_date, end_date)
    filtered = filtered[filtered["category"].fillna("").astype(str) == str(selected_category)].copy()

    if filtered.empty:
        st.info("No merged rows matched the selected category and date range.")
        return

    type_df = calculate_breakdown_metrics_from_merged(filtered, "type", selected_category)
    subtype_df = calculate_breakdown_metrics_from_merged(filtered, "sub_type", selected_category)

    st.markdown(f"#### Category: {selected_category}")
    st.metric("Distinct sessions in scope", f"{filtered['session_id'].nunique():,}")

    st.markdown("##### Type breakdown")
    st.dataframe(
        type_df,
        width="stretch",
        hide_index=True,
        column_config={
            "type": st.column_config.TextColumn("Type"),
            "nps_score": st.column_config.NumberColumn("NPS %", format="%.1f"),
            "csat_score": st.column_config.NumberColumn("CSAT %", format="%.1f"),
            "fcr_score": st.column_config.NumberColumn("FCR %", format="%.1f"),
            "response_sessions": st.column_config.NumberColumn("Response Sessions", format="%d"),
            "answered_nps": st.column_config.NumberColumn("Answered NPS", format="%d"),
            "answered_csat": st.column_config.NumberColumn("Answered CSAT", format="%d"),
            "answered_fcr": st.column_config.NumberColumn("Answered FCR", format="%d"),
        },
    )
    st.altair_chart(build_breakdown_metric_chart(type_df, "type", "nps_score", "NPS by type", BRAND["primary_dark"]), width="stretch")
    st.altair_chart(build_breakdown_metric_chart(type_df, "type", "csat_score", "CSAT by type", BRAND["primary"]), width="stretch")
    st.altair_chart(build_breakdown_metric_chart(type_df, "type", "fcr_score", "FCR by type", BRAND["secondary"]), width="stretch")

    st.markdown("##### Sub-Type breakdown")
    st.dataframe(
        subtype_df,
        width="stretch",
        hide_index=True,
        column_config={
            "sub_type": st.column_config.TextColumn("Sub-Type"),
            "nps_score": st.column_config.NumberColumn("NPS %", format="%.1f"),
            "csat_score": st.column_config.NumberColumn("CSAT %", format="%.1f"),
            "fcr_score": st.column_config.NumberColumn("FCR %", format="%.1f"),
            "response_sessions": st.column_config.NumberColumn("Response Sessions", format="%d"),
            "answered_nps": st.column_config.NumberColumn("Answered NPS", format="%d"),
            "answered_csat": st.column_config.NumberColumn("Answered CSAT", format="%d"),
            "answered_fcr": st.column_config.NumberColumn("Answered FCR", format="%d"),
        },
    )
    st.altair_chart(build_breakdown_metric_chart(subtype_df, "sub_type", "nps_score", "NPS by sub-type", BRAND["primary_dark"]), width="stretch")
    st.altair_chart(build_breakdown_metric_chart(subtype_df, "sub_type", "csat_score", "CSAT by sub-type", BRAND["primary"]), width="stretch")
    st.altair_chart(build_breakdown_metric_chart(subtype_df, "sub_type", "fcr_score", "FCR by sub-type", BRAND["secondary"]), width="stretch")


def render_numeric_filter_controls(prefix: str, label: str, col_left, col_right):
    with col_left:
        operator = st.selectbox(f"{label} operator", options=["Any", "<", "<=", ">", ">=", "=", "!="], key=f"{prefix}_{label}_op")
    with col_right:
        value = st.number_input(f"{label} value", min_value=0.0, max_value=5.0, step=1.0, value=5.0, key=f"{prefix}_{label}_value")
    return operator, value


def render_ratings_tab(ratings_df: pd.DataFrame) -> None:
    st.subheader("Ratings raw data")
    st.caption("Use date, last operator, score, and FCR filters to inspect the raw ratings table.")

    min_date, max_date = get_default_date_bounds(ratings_df, "created_at_dt")

    with st.container(border=True):
        r1, r2, r3, r4 = st.columns([1, 1, 1.2, 1.2])
        with r1:
            start_date = st.date_input("Date from", value=min_date, min_value=min_date, max_value=max_date, key="ratings_start")
        with r2:
            end_date = st.date_input("Date to", value=max_date, min_value=min_date, max_value=max_date, key="ratings_end")
        with r3:
            operators = st.multiselect("Last Operator", options=normalize_text_options(ratings_df["last_operator_name"]), key="ratings_operators")
        with r4:
            fcr_values = st.multiselect("FCR", options=normalize_text_options(ratings_df["fcr_normalized"]), key="ratings_fcr")

        nps_col1, nps_col2, csat_col1, csat_col2 = st.columns(4)
        nps_operator, nps_value = render_numeric_filter_controls("ratings", "NPS", nps_col1, nps_col2)
        csat_operator, csat_value = render_numeric_filter_controls("ratings", "CSAT", csat_col1, csat_col2)

    filtered = filter_date_range(ratings_df, "created_at_dt", start_date, end_date)
    filtered = filter_multiselect(filtered, "last_operator_name", operators)
    filtered = filter_multiselect(filtered, "fcr_normalized", fcr_values)
    filtered = apply_numeric_filter(filtered, "nps", nps_operator, nps_value)
    filtered = apply_numeric_filter(filtered, "csat", csat_operator, csat_value)
    filtered = filtered.sort_values("created_at_dt", ascending=False)

    summary_left, summary_mid, summary_right = st.columns(3)
    summary_left.metric("Rows", f"{len(filtered):,}")
    summary_mid.metric("Distinct sessions", f"{filtered['session_id'].nunique():,}" if not filtered.empty else "0")
    summary_right.metric("Last Operators", f"{filtered['last_operator_name'].nunique():,}" if not filtered.empty else "0")

    page_size = st.selectbox("Rows per page", options=[25, 50, 100, 200], index=1, key="ratings_page_size")
    page = st.number_input("Page", min_value=1, value=1, step=1, key="ratings_page")
    page_df, total_pages = paginate_dataframe(filtered, int(page), int(page_size))
    st.caption(f"Page {min(int(page), total_pages)} of {total_pages}")

    display_cols = [
        "created_at",
        "last_operator_name",
        "name",
        "email",
        "nps",
        "csat",
        "fcr",
        "remarks",
        "session_id",
        "website_id",
        "chat_link",
    ]
    display_df = page_df[display_cols].copy() if not page_df.empty else pd.DataFrame(columns=display_cols)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn("Created at", format="YYYY-MM-DD HH:mm:ss"),
            "last_operator_name": st.column_config.TextColumn("Last Operator"),
            "chat_link": st.column_config.LinkColumn("Link", display_text="Link"),
            "remarks": st.column_config.TextColumn("Remarks", width="large"),
        },
    )


def render_merged_tab(merged_df: pd.DataFrame) -> None:
    st.subheader("Merged raw data")
    st.caption("This tab keeps ratings filters and adds disposition filters. To avoid ambiguity, both rating and disposition created_at filters are available.")

    rating_min, rating_max = get_default_date_bounds(merged_df, "rating_created_at_dt")
    disp_min, disp_max = get_default_date_bounds(merged_df, "disposition_created_at_dt")

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns([1, 1, 1.2, 1.2])
        with m1:
            rating_start = st.date_input("Rating date from", value=rating_min, min_value=rating_min, max_value=rating_max, key="merged_rating_start")
        with m2:
            rating_end = st.date_input("Rating date to", value=rating_max, min_value=rating_min, max_value=rating_max, key="merged_rating_end")
        with m3:
            disp_start = st.date_input("Disposition date from", value=disp_min, min_value=disp_min, max_value=disp_max, key="merged_disp_start")
        with m4:
            disp_end = st.date_input("Disposition date to", value=disp_max, min_value=disp_min, max_value=disp_max, key="merged_disp_end")

        m5, m6, m7, m8 = st.columns([1.2, 1.2, 1.2, 1.2])
        with m5:
            operators = st.multiselect("Last Operator", options=normalize_text_options(merged_df["rating_last_operator_name"]), key="merged_rating_operators")
        with m6:
            categories = st.multiselect("Category", options=normalize_text_options(merged_df["category"]), key="merged_categories")
        with m7:
            types = st.multiselect("Type", options=normalize_text_options(merged_df["type"]), key="merged_types")
        with m8:
            sub_types = st.multiselect("Sub type", options=normalize_text_options(merged_df["sub_type"]), key="merged_sub_types")

        mf1, mf2, mf3 = st.columns([1.2, 1.2, 1.2])
        with mf1:
            fcr_values = st.multiselect("FCR", options=normalize_text_options(merged_df["fcr_normalized"]), key="merged_fcr")
        with mf2:
            nps_col1, nps_col2 = st.columns(2)
            nps_operator, nps_value = render_numeric_filter_controls("merged", "NPS", nps_col1, nps_col2)
        with mf3:
            csat_col1, csat_col2 = st.columns(2)
            csat_operator, csat_value = render_numeric_filter_controls("merged", "CSAT", csat_col1, csat_col2)

    filtered = filter_date_range(merged_df, "rating_created_at_dt", rating_start, rating_end)
    filtered = filter_date_range(filtered, "disposition_created_at_dt", disp_start, disp_end)
    filtered = filter_multiselect(filtered, "rating_last_operator_name", operators)
    filtered = filter_multiselect(filtered, "category", categories)
    filtered = filter_multiselect(filtered, "type", types)
    filtered = filter_multiselect(filtered, "sub_type", sub_types)
    filtered = filter_multiselect(filtered, "fcr_normalized", fcr_values)
    filtered = apply_numeric_filter(filtered, "nps", nps_operator, nps_value)
    filtered = apply_numeric_filter(filtered, "csat", csat_operator, csat_value)
    filtered = filtered.sort_values(["rating_created_at_dt", "disposition_created_at_dt"], ascending=False)

    merged_left, merged_mid, merged_right = st.columns(3)
    merged_left.metric("Rows", f"{len(filtered):,}")
    merged_mid.metric("Distinct sessions", f"{filtered['session_id'].nunique():,}" if not filtered.empty else "0")
    merged_right.metric("Categories", f"{filtered['category'].nunique():,}" if not filtered.empty else "0")

    page_size = st.selectbox("Rows per page", options=[25, 50, 100, 200], index=1, key="merged_page_size")
    page = st.number_input("Page", min_value=1, value=1, step=1, key="merged_page")
    page_df, total_pages = paginate_dataframe(filtered, int(page), int(page_size))
    st.caption(f"Page {min(int(page), total_pages)} of {total_pages}")

    columns = [
        "rating_created_at",
        "disposition_created_at",
        "rating_last_operator_name",
        "category",
        "type",
        "sub_type",
        "nps",
        "csat",
        "fcr",
        "dff",
        "remarks",
        "session_id",
        "chat_link",
    ]
    existing_cols = [col for col in columns if col in page_df.columns]
    display_df = page_df[existing_cols].copy() if not page_df.empty else pd.DataFrame(columns=existing_cols)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "rating_created_at": st.column_config.DatetimeColumn("Rating created_at", format="YYYY-MM-DD HH:mm:ss"),
            "disposition_created_at": st.column_config.DatetimeColumn("Disposition created_at", format="YYYY-MM-DD HH:mm:ss"),
            "rating_last_operator_name": st.column_config.TextColumn("Last Operator"),
            "chat_link": st.column_config.LinkColumn("Link", display_text="Link"),
            "dff": st.column_config.TextColumn("Disposition detail", width="large"),
            "remarks": st.column_config.TextColumn("Remarks", width="large"),
        },
    )


def render_sidebar_controls() -> Tuple[Optional[date], Optional[date], bool]:
    st.sidebar.header("Data pull")
    st.sidebar.caption("Pull fresh data from the KrispCall support APIs, then slice it inside the app.")
    today = date.today()
    default_start = today.replace(day=1)

    start_date = st.sidebar.date_input("Start date", value=default_start, key="fetch_start")
    end_date = st.sidebar.date_input("End date", value=today, key="fetch_end")
    calculate = st.sidebar.button("Calculate", width="stretch", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Scoring logic**")
    st.sidebar.markdown("- **NPS**: 5 = promoter, 4 = neutral, 0-3 = detractor")
    st.sidebar.markdown("- **CSAT**: 5 = promoter, 4 = neutral, 0-3 = detractor")
    st.sidebar.markdown("- **FCR**: yes / total filled")
    st.sidebar.markdown("- **Response total**: unique sessions with at least one of NPS, CSAT, or FCR filled")
    return start_date, end_date, calculate


def main() -> None:
    inject_css()

    if not check_auth():
        st.stop()

    render_brand_header()
    api_key, base_url = get_api_config()
    if not api_key:
        st.error("API key is missing. Add it to .streamlit/secrets.toml under [api].")
        st.stop()

    start_date, end_date, calculate = render_sidebar_controls()
    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        st.stop()

    if calculate:
        try:
            with st.spinner("Pulling ratings and dispositions from the APIs..."):
                ratings_df, dispositions_df, merged_df = fetch_support_data(
                    api_key,
                    base_url,
                    combine_date_and_time(start_date, is_end=False),
                    combine_date_and_time(end_date, is_end=True),
                )
            st.session_state["ratings_df"] = ratings_df
            st.session_state["dispositions_df"] = dispositions_df
            st.session_state["merged_df"] = merged_df
            st.session_state["fetched_start"] = start_date
            st.session_state["fetched_end"] = end_date
        except Exception as exc:
            response = getattr(exc, "response", None)
            message = str(exc)
            if response is not None:
                status = getattr(response, "status_code", "Unknown")
                body = (getattr(response, "text", "") or "").strip()
                if len(body) > 600:
                    body = body[:600] + "..."
                message = f"API request failed with status {status}. {body}" if body else f"API request failed with status {status}."
            st.error(message)
            if "ratings_df" not in st.session_state:
                st.info("Fix the API issue, then click Calculate again.")
                st.stop()

    if "ratings_df" not in st.session_state:
        st.info("Choose a date range in the sidebar and click Calculate. Data will not be fetched until you click the button.")
        st.stop()

    ratings_df = st.session_state["ratings_df"]
    merged_df = st.session_state["merged_df"]
    fetched_start = st.session_state["fetched_start"]
    fetched_end = st.session_state["fetched_end"]

    overview_tab, operator_tab, disposition_tab, ratings_tab, merged_tab = st.tabs(
        ["Overview", "By operator", "By disposition", "Ratings", "Merged table"]
    )
    with overview_tab:
        render_overview_tab(ratings_df, merged_df, fetched_start, fetched_end)
    with operator_tab:
        render_operator_tab(ratings_df, fetched_start, fetched_end)
    with disposition_tab:
        render_disposition_tab(merged_df, fetched_start, fetched_end)
    with ratings_tab:
        render_ratings_tab(ratings_df)
    with merged_tab:
        render_merged_tab(merged_df)


if __name__ == "__main__":
    main()
