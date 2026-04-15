from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from utils import (
    DISPOSITIONS_ENDPOINT,
    RATINGS_ENDPOINT,
    apply_numeric_filter,
    build_csat_score_distribution,
    build_distribution_df,
    build_merged_df,
    calculate_csat,
    calculate_fcr,
    calculate_nps,
    combine_date_and_time,
    fetch_all_pages,
    filter_date_range,
    filter_multiselect,
    metric_tone,
    normalize_text_options,
    paginate_dataframe,
    prepare_dispositions_df,
    prepare_ratings_df,
)


st.set_page_config(
    page_title="KrispCall Support Analytics",
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


@st.cache_data(show_spinner=False)
def fetch_support_data(api_key: str, base_url: str, start_iso: str, end_iso: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }}
            .brand-shell {{
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 0.3rem 0 1rem 0;
            }}
            .brand-title {{
                font-size: 2rem;
                font-weight: 800;
                color: {BRAND['ink']};
                line-height: 1;
                margin: 0;
            }}
            .brand-subtitle {{
                color: {BRAND['muted']};
                font-size: 0.95rem;
                margin-top: 0.25rem;
            }}
            .metric-card {{
                background: {BRAND['card']};
                border: 1px solid {BRAND['border']};
                border-left: 6px solid {BRAND['neutral']};
                border-radius: 18px;
                padding: 1rem 1rem 0.9rem 1rem;
                box-shadow: 0 10px 35px rgba(55, 25, 78, 0.06);
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
            }}
            .metric-sub {{
                color: {BRAND['muted']};
                font-size: 0.9rem;
                margin-top: 0.2rem;
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
    logo_col, text_col = st.columns([1, 6], vertical_alignment="center")
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=120)
    with text_col:
        st.markdown(
            f"""
            <div class="brand-shell">
                <div>
                    <div class="brand-title">KrispCall Support Analytics</div>
                    <div class="brand-subtitle">Ratings, dispositions, NPS, CSAT, and FCR in one polished Streamlit dashboard.</div>
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
            submitted = st.form_submit_button("Login", use_container_width=True)

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


def render_metric_card(title: str, value: Optional[float], suffix: str, subtitle: str, tone: str) -> None:
    if value is None:
        display_value = "N/A"
        suffix = ""
    else:
        display_value = f"{value:.1f}" if isinstance(value, float) else str(value)
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{display_value}{suffix}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_metric_chart(df: pd.DataFrame, title: str) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"label": [], "count": []})).mark_bar()

    color_scale = alt.Scale(
        domain=df["label"].tolist(),
        range=[BRAND["excellent"], BRAND["neutral"], BRAND["poor"]][: len(df)],
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("label:N", sort=None, title=None),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("label:N", scale=color_scale, legend=None),
            tooltip=["label", "count", "percentage"],
        )
        .properties(height=280, title=title)
    )


def build_csat_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"score": [], "count": []})).mark_bar()
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=BRAND["primary"])
        .encode(
            x=alt.X("score:O", title="CSAT score"),
            y=alt.Y("count:Q", title="Responses"),
            tooltip=["score", "count"],
        )
        .properties(height=280, title="CSAT raw score distribution")
    )


def filter_ratings_core(
    ratings_df: pd.DataFrame,
    start_date: Optional[date],
    end_date: Optional[date],
    operators: list[str],
) -> pd.DataFrame:
    df = filter_date_range(ratings_df, "created_at_dt", start_date, end_date)
    df = filter_multiselect(df, "last_operator_name", operators)
    return df


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
    ratings_filtered = filter_ratings_core(ratings_df, start_date, end_date, operators)

    disposition_filter_selected = any([categories, types, sub_types])
    if not disposition_filter_selected:
        return ratings_filtered, merged_df.iloc[0:0].copy(), False

    merged_filtered = merged_df.copy()
    merged_filtered = filter_date_range(merged_filtered, "rating_created_at_dt", start_date, end_date)
    merged_filtered = filter_multiselect(merged_filtered, "rating_last_operator_name", operators)
    merged_filtered = filter_multiselect(merged_filtered, "category", categories)
    merged_filtered = filter_multiselect(merged_filtered, "type", types)
    merged_filtered = filter_multiselect(merged_filtered, "sub_type", sub_types)

    session_ids = merged_filtered["session_id"].dropna().unique().tolist()
    rating_scope = ratings_filtered[ratings_filtered["session_id"].isin(session_ids)].copy()
    return rating_scope, merged_filtered, True


def get_default_date_bounds(df: pd.DataFrame, column: str) -> Tuple[date, date]:
    if df.empty or column not in df.columns or df[column].dropna().empty:
        today = date.today()
        return today, today
    series = pd.to_datetime(df[column], utc=True, errors="coerce").dropna()
    return series.min().date(), series.max().date()


def render_overview_tab(ratings_df: pd.DataFrame, merged_df: pd.DataFrame, fetched_start: date, fetched_end: date) -> None:
    st.subheader("Overview")
    st.caption("Metrics use ratings only by default. When any disposition filter is selected, the scope is narrowed via the merged table without double counting ratings.")

    with st.container(border=True):
        f1, f2, f3, f4, f5, f6 = st.columns([1, 1, 1.4, 1.2, 1.2, 1.2])
        with f1:
            start_date = st.date_input("Ratings created_at from", value=fetched_start, min_value=fetched_start, max_value=fetched_end, key="ov_start")
        with f2:
            end_date = st.date_input("Ratings created_at to", value=fetched_end, min_value=fetched_start, max_value=fetched_end, key="ov_end")
        with f3:
            operators = st.multiselect(
                "Operator",
                options=normalize_text_options(ratings_df["last_operator_name"]),
                key="ov_operators",
            )
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
        ratings_df,
        merged_df,
        start_date,
        end_date,
        operators,
        categories,
        types,
        sub_types,
    )

    nps_result = calculate_nps(metric_df)
    csat_result = calculate_csat(metric_df)
    fcr_result = calculate_fcr(metric_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "NPS score",
            nps_result.score,
            "%",
            f"Answered: {nps_result.answered}. 5 = promoter, 4 = neutral, 0-3 = detractor.",
            metric_tone("nps", nps_result.score),
        )
    with c2:
        render_metric_card(
            "CSAT score",
            csat_result.score,
            "%",
            f"Answered: {csat_result.answered}. Top-2-box logic: 4-5 satisfied.",
            metric_tone("csat", csat_result.score),
        )
    with c3:
        render_metric_card(
            "FCR resolution",
            fcr_result.score,
            "%",
            f"Answered: {fcr_result.answered}. Formula: yes / total filled * 100.",
            metric_tone("fcr", fcr_result.score),
        )

    note = "Disposition filters are active. Metrics are being scoped through the merged table." if using_dispositions else "No disposition filters are active. Metrics are based on the ratings table only."
    st.markdown(f'<div class="note-card">{note}</div>', unsafe_allow_html=True)

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        nps_dist = build_distribution_df("NPS", nps_result.extra, nps_result.answered)
        st.altair_chart(build_metric_chart(nps_dist, "NPS distribution"), use_container_width=True)
    with chart_col2:
        csat_dist = build_csat_score_distribution(metric_df)
        st.altair_chart(build_csat_distribution_chart(csat_dist), use_container_width=True)
        avg = csat_result.extra.get("average") if csat_result.extra else None
        st.caption(f"Average CSAT: {avg if avg is not None else 'N/A'} / 5")
    with chart_col3:
        fcr_dist = build_distribution_df("FCR", fcr_result.extra, fcr_result.answered)
        st.altair_chart(build_metric_chart(fcr_dist, "FCR distribution"), use_container_width=True)

    stats_left, stats_mid, stats_right = st.columns(3)
    stats_left.metric("Ratings rows in scope", f"{len(metric_df):,}")
    stats_mid.metric("Distinct sessions in scope", f"{metric_df['session_id'].nunique():,}" if not metric_df.empty else "0")
    stats_right.metric("Merged rows matched", f"{len(merged_filtered):,}" if using_dispositions else "0")


def render_numeric_filter_controls(prefix: str, label: str, col_left, col_right):
    with col_left:
        operator = st.selectbox(f"{label} operator", options=["Any", "<", "<=", ">", ">=", "=", "!="], key=f"{prefix}_{label}_op")
    with col_right:
        value = st.number_input(f"{label} value", min_value=0.0, max_value=5.0, step=1.0, value=5.0, key=f"{prefix}_{label}_value")
    return operator, value


def render_ratings_tab(ratings_df: pd.DataFrame) -> None:
    st.subheader("Ratings raw data")
    st.caption("Validated columns from the export: created_at, last_operator_name, nps, csat, fcr, session_id, website_id, and chat_url.")

    min_date, max_date = get_default_date_bounds(ratings_df, "created_at_dt")

    filter_box = st.container(border=True)
    with filter_box:
        r1, r2, r3, r4 = st.columns([1, 1, 1.2, 1.2])
        with r1:
            start_date = st.date_input("Date from", value=min_date, min_value=min_date, max_value=max_date, key="ratings_start")
        with r2:
            end_date = st.date_input("Date to", value=max_date, min_value=min_date, max_value=max_date, key="ratings_end")
        with r3:
            operators = st.multiselect("Operator", options=normalize_text_options(ratings_df["last_operator_name"]), key="ratings_operators")
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
    summary_right.metric("Operators", f"{filtered['last_operator_name'].nunique():,}" if not filtered.empty else "0")

    page_size = st.selectbox("Rows per page", options=[25, 50, 100, 200], index=1, key="ratings_page_size")
    page = st.number_input("Page", min_value=1, value=1, step=1, key="ratings_page")
    page_df, total_pages = paginate_dataframe(filtered, int(page), int(page_size))
    st.caption(f"Page {min(int(page), total_pages)} of {total_pages}")

    display_df = page_df[[
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
    ]].copy() if not page_df.empty else pd.DataFrame(columns=[
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
    ])

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn("Created at", format="YYYY-MM-DD HH:mm:ss"),
            "last_operator_name": st.column_config.TextColumn("Operator"),
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
            operators = st.multiselect("Rating operator", options=normalize_text_options(merged_df["rating_last_operator_name"]), key="merged_rating_operators")
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
        use_container_width=True,
        hide_index=True,
        column_config={
            "rating_created_at": st.column_config.DatetimeColumn("Rating created_at", format="YYYY-MM-DD HH:mm:ss"),
            "disposition_created_at": st.column_config.DatetimeColumn("Disposition created_at", format="YYYY-MM-DD HH:mm:ss"),
            "rating_last_operator_name": st.column_config.TextColumn("Operator"),
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
    calculate = st.sidebar.button("Calculate", use_container_width=True, type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Scoring logic**")
    st.sidebar.markdown("- **NPS**: 5 = promoter, 4 = neutral, 0-3 = detractor")
    st.sidebar.markdown("- **CSAT**: 4-5 = satisfied. Score = satisfied / total answered")
    st.sidebar.markdown("- **FCR**: yes / total filled")
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

    if calculate or "ratings_df" not in st.session_state:
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

    if "ratings_df" not in st.session_state:
        st.info("Choose a date range in the sidebar and click Calculate.")
        st.stop()

    ratings_df = st.session_state["ratings_df"]
    merged_df = st.session_state["merged_df"]
    fetched_start = st.session_state["fetched_start"]
    fetched_end = st.session_state["fetched_end"]

    with st.expander("Current data snapshot", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Ratings rows pulled", f"{len(ratings_df):,}")
        col_b.metric("Disposition rows pulled", f"{len(st.session_state['dispositions_df']):,}")
        col_c.metric("Merged rows", f"{len(merged_df):,}")
        st.caption(f"Fetched range: {fetched_start.isoformat()} to {fetched_end.isoformat()}")

    overview_tab, ratings_tab, merged_tab = st.tabs(["Overview", "Ratings", "Merged table"])
    with overview_tab:
        render_overview_tab(ratings_df, merged_df, fetched_start, fetched_end)
    with ratings_tab:
        render_ratings_tab(ratings_df)
    with merged_tab:
        render_merged_tab(merged_df)


if __name__ == "__main__":
    main()
