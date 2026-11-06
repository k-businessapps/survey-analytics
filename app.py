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
    st.sidebar.markdown("- **Up to May 31, 2026**: old 0-6 scoring logic from the current app")
    st.sidebar.markdown("- **From June 1, 2026**: NPS 4-5 promoter, 3 neutral, 1-2 detractor")
    st.sidebar.markdown("- **From June 1, 2026**: CSAT positive score = 4-5 / total CSAT submissions")
    st.sidebar.markdown("- **Mixed date range**: new formula is used and a warning is shown")
    st.sidebar.markdown("- **FCR**: yes / total filled")
    st.sidebar.markdown("- **Response total**: unique sessions with at least one of NPS, CSAT, or FCR filled")
    st.sidebar.markdown("- **Operator filters**: grouped by first name, so Lisa Palmer and Lisa both appear under Lisa")
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
            ratings_df, merged_df = add_operator_first_name_columns(ratings_df, merged_df)
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
    ratings_df, merged_df = add_operator_first_name_columns(ratings_df, merged_df)
    st.session_state["ratings_df"] = ratings_df
    st.session_state["merged_df"] = merged_df
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
