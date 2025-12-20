import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm

from pathlib import Path

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "big-mac-source-data-v2.csv"
HERO_IMG  = APP_DIR / "vis/Big-Mac-2.png"
LFC_LOGO = APP_DIR / "vis" / "LFC_Bull_Circle_Blue.png"
INTRO_IMG  = APP_DIR / "vis/1.jpg"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, na_values=['#N/A'])
    df = df.dropna(subset=['local_price'])

    # New Economist file: GDP_dollar instead of GDP_local
    cols = df.columns
    if "GDP_local" in cols:
        df["GDP_local"] = pd.to_numeric(df["GDP_local"], errors="coerce")
    elif "GDP_dollar" in cols:
        df["GDP_local"] = pd.to_numeric(df["GDP_dollar"], errors="coerce")
    else:
        # no GDP column available -> create empty so rest of code still runs
        df["GDP_local"] = np.nan

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "name"])


# dollar price
def calc_dollar_price(df):
    df['dollar_price'] = df['local_price'] / df['dollar_ex']
    return df

# raw Big Mac index relative to base currencies
def calc_raw_index(df, base_currencies):
    df = df[~df["dollar_price"].isna()].copy()

    for currency in base_currencies:
        # one base price per date (prevents merge row explosion)
        base_prices = (
            df.loc[df["currency_code"] == currency, ["date", "dollar_price"]]
              .groupby("date", as_index=False)["dollar_price"]
              .mean()
              .rename(columns={"dollar_price": "base_price"})
        )

        df = df.merge(base_prices, on="date", how="left")
        df[currency] = (df["dollar_price"] / df["base_price"]) - 1
        df.drop(columns=["base_price"], inplace=True)

    df[base_currencies] = df[base_currencies].round(5)
    return df




# adjusted index by regressing GDP_local on dollar_price and adjusting residuals
def calc_adjusted_index(df, regression_countries):
    df_gdp = df[(df['GDP_local'] > 0) & (df['iso_a3'].isin(regression_countries))].copy()
    adjusted_data = []
    for date, group in df_gdp.groupby('date'):
        X = np.log(group['GDP_local'])
        X = sm.add_constant(X)
        y = np.log(group['dollar_price'])
        model = sm.OLS(y, X).fit()
        pred = model.predict(X)
        group['adjusted'] = np.exp(y - pred) - 1  # residual expressed as % difference
        adjusted_data.append(group[['date', 'iso_a3', 'adjusted']])
    adjusted_df = pd.concat(adjusted_data)
    df = df.merge(adjusted_df, on=['date', 'iso_a3'], how='left')
    df['adjusted'] = df['adjusted'].fillna(0)
    return df

# countries and base currencies 
big_mac_countries = ['ARG', 'AUS', 'BRA', 'GBR', 'CAN', 'CHL', 'CHN', 'CZE', 'DNK',
                     'EGY', 'HKG', 'HUN', 'IDN', 'ISR', 'JPN', 'MYS', 'MEX', 'NZL',
                     'NOR', 'PER', 'PHL', 'POL', 'RUS', 'SAU', 'SGP', 'ZAF', 'KOR',
                     'SWE', 'CHE', 'TWN', 'THA', 'TUR', 'ARE', 'USA', 'COL', 'CRI',
                     'PAK', 'LKA', 'UKR', 'URY', 'IND', 'VNM', 'GTM', 'HND', 'VEN',
                     'NIC', 'AZE', 'BHR', 'JOR', 'KWT', 'LBN', 'MDA', 'OMN',
                     'QAT', 'ROU', 'EUZ']

regression_countries = ['ARG', 'AUS', 'BRA', 'GBR', 'CAN', 'CHL', 'CHN', 'CZE', 'DNK',
                         'EGY', 'EUZ', 'HKG', 'HUN', 'IDN', 'ISR', 'JPN', 'MYS', 'MEX',
                         'NZL', 'NOR', 'PER', 'PHL', 'POL', 'RUS', 'SAU', 'SGP', 'ZAF',
                         'KOR', 'SWE', 'CHE', 'TWN', 'THA', 'TUR', 'USA', 'COL', 'PAK',
                         'IND', 'AUT', 'BEL', 'NLD', 'FIN', 'FRA', 'DEU', 'IRL', 'ITA',
                         'PRT', 'ESP', 'GRC', 'EST']

base_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY']

import base64
import streamlit as st

def main():
    st.set_page_config(page_title="Big Mac Index Dashboard", layout="wide")

    st.markdown(
        """
        <style>
        /* Calm page scrolling */
        html, body {
            overscroll-behavior: contain;
            touch-action: pan-y;  /* vertical scroll only */
        }

        /* Let the map capture pinch + pan */
        .map-container {
            touch-action: pinch-zoom pan-x pan-y;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )




    # Disable browser‑level pinch zoom on mobile
    st.markdown(
        """
        <meta name="viewport"
              content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
        """,
        unsafe_allow_html=True,
    )


    logo_b64 = base64.b64encode(LFC_LOGO.read_bytes()).decode()

    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:18px;
            padding: 10px 0 4px 0;
        ">
        <img src="data:image/png;base64,{logo_b64}" style="width:90px; height:auto;" />
        <div style="
            font-size:56px;
            font-weight:444;
            line-height:1;
            margin-top:-6px;   /* <-- moves text UP (tune this) */
        ">
            LUISS Finance Club
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.write("")
    st.markdown("" \
    "This is an interavtive dashboard from the LUISS Finance Club to explore The Economist's Big Mac Index." \
    "The Big Mac Index is a lighthearted guide to whether currencies are at their 'correct' level. " \
    "It is based on the theory of purchasing-power parity (PPP), the notion that in the long run exchange rates should move towards the rate that would equalize the prices of an identical basket of goods and services (in this case, a Big Mac) in any two countries. ")
    st.write("")

    st.write("")  # spacing [web:625]
    #st.image(str(INTRO_IMG), width="stretch")
    st.write("")  # spacing [web:625]


    # --- Intro paragraph ---
    st.markdown(
        "Explore currency valuations worldwide using Big Mac prices! "
        "Select a release date and base currency to see how different currencies compare. "
        "Dive into raw and GDP-adjusted indices, track biggest movers, and visualize data on a world map."
    )

    st.write("")  # spacing [web:625]

    # --- Big Mac hero image ---
    st.image(str(HERO_IMG), width="stretch")
    st.caption("Graphics by LUISS Finance Club")

    st.write("")  # spacing [web:625]

    # --- Second paragraph ---
    st.markdown(
        "Developed by the LUISS Finance Club, this dashboard replicates The Economist's Big Mac Index calculations. "
        "It features both raw and GDP-adjusted indices, allowing users to explore currency valuations in depth. "
        "Select different release dates and base currencies to see how valuations change over time and across currencies."
    )

    st.write("")  # spacing [web:625]


    #st.write("Hero image exists?", HERO_IMG.exists(), "Path:", str(HERO_IMG))

    st.title("Interactive Big Mac Index Dashboard")

    st.markdown("""
    This dashboard replicates The Economist's Big Mac Index calculations with:
    - Raw index (currency valuation relative to base currencies)
    - Adjusted index (accounting for GDP differences)
    """)

    df = load_data()
    df = calc_dollar_price(df)
    df = calc_raw_index(df, base_currencies)
    #st.write("Rows after raw index:", len(df), "unique countries:", df["iso_a3"].nunique())
    #df = calc_adjusted_index(df, regression_countries)

    st.sidebar.image(str(INTRO_IMG), width="stretch")

    # --- Release selector (Month Year, no day) ---
    all_dates = sorted(df["date"].dropna().unique())
    all_dates = [pd.Timestamp(d).to_pydatetime() for d in all_dates]
    selected_date = st.sidebar.select_slider(
        "Select release date",
        options=all_dates,
        value=all_dates[-1],
        format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"),  # e.g., "May 2003"
    )
    selected_date = pd.to_datetime(selected_date)

    # Use Economist's official GDP-adjusted index for the selected base
    base_currency = st.sidebar.selectbox(
        "Select Base Currency",
        options=base_currencies,
        index=base_currencies.index("USD"),
        )    


    # Economist's official adjusted index for the chosen base
    adjusted_col = f"{base_currency}_adjusted"
    if adjusted_col in df.columns:
        df["adjusted"] = df[adjusted_col]
    else:
        df["adjusted"] = 0.0

    # date
    # mapping of year strings to full dates (taking unique years)
    years = sorted(df['date'].dt.year.unique())

    # dict: year string -> list of full dates in that year
    year_to_dates = {year: df[df['date'].dt.year == year]['date'].unique() for year in years}

    # year in sidebar
    #selected_year = st.sidebar.selectbox("Select Year", options=years, index=len(years)-1)
    # from selected year, select a specific date (e.g. latest date in that year)
    #selected_date = year_to_dates[selected_year][-1]  # last date in that year

    # --- Date slider across all releases ---
    #all_dates = sorted(df["date"].dropna().unique())
    #min_date = pd.to_datetime(all_dates[0]).to_pydatetime()
    #max_date = pd.to_datetime(all_dates[-1]).to_pydatetime()

    #selected_date = st.sidebar.slider(
        #"Select release date",
        #min_value=min_date,
        #max_value=max_date,
        #value=max_date,
        #format="YYYY-MM-DD",
    #)

    # Snap slider value to the closest available release date in the dataset
    #selected_date = pd.to_datetime(selected_date)
    #selected_date = df.loc[df["date"] <= selected_date, "date"].max()

    # Filter dataframe for the selected date
    df_date = df[df['date'] == selected_date].copy()
    df_date = df_date.sort_values(by=base_currency)

    # --- Country snapshot (for selected date & base) ---
    # --- Country snapshot (persists across year changes) ---
    available_countries = df_date["name"].sort_values().unique()

    # Initialize session_state the first time
    if "selected_country" not in st.session_state:
        st.session_state.selected_country = available_countries[0]

    # If previously selected country is not in this date's list, fall back to first
    if st.session_state.selected_country not in available_countries:
        st.session_state.selected_country = available_countries[0]

    country = st.selectbox(
        "Country snapshot",
        options=available_countries,
        index=list(available_countries).index(st.session_state.selected_country),
    )

    # Update state with any new choice
    st.session_state.selected_country = country

    country_row = df_date[df_date["name"] == country].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Local Big Mac price",
            value=f"{country_row['local_price']:.2f} {country_row['currency_code']}",
            help="Menu price in local currency on the selected release date.",
            border=True,
        )

    with col2:
        st.metric(
            label="Dollar price",
            value=f"${country_row['dollar_price']:.2f}",
            help="Local price converted using the market exchange rate.",
            border=True,
        )

    with col3:
        st.metric(
            label=f"Raw vs {base_currency}",
            value=f"{country_row[base_currency]:+.1%}",
            help=f"Big Mac misvaluation relative to {base_currency} (classic index).",
            border=True,
        )

    with col4:
        st.metric(
            label="GDP‑adjusted",
            value=f"{country_row['adjusted']:+.1%}",
            help="Misvaluation after controlling for income (GDP per capita).",
            border=True,
        )

    st.caption(
        "Explained: The local Big Mac price is converted to dollars using the market exchange rate to get the dollar price. "
        "The raw index compares this dollar price to the price implied by the selected base currency. "
        "The GDP-adjusted index accounts for the tendency of richer countries to have higher prices, "
        "estimating a 'fair value' using the relationship between price levels and GDP per capita."
    )

    st.write("")  # small spacer before 'Biggest movers'


    # --- Momentum: biggest movers vs previous release ---
    prev_date = df.loc[df["date"] < selected_date, "date"].max()

    if pd.isna(prev_date):
        st.info("No previous release date available to compute momentum.")
    else:
        df_prev = df[df["date"] == prev_date][["iso_a3", base_currency, "adjusted"]].copy()

        movers = (
            df_date[["name", "iso_a3", base_currency, "adjusted"]]
            .merge(df_prev, on="iso_a3", suffixes=("", "_prev"), how="inner")
        )

        movers["raw_change"] = movers[base_currency] - movers[f"{base_currency}_prev"]
        movers["adj_change"] = movers["adjusted"] - movers["adjusted_prev"]

        prev_label = pd.Timestamp(prev_date).strftime("%b %Y")  # e.g., "Jul 2024"
        st.subheader(f"Biggest movers since {prev_label}")
        st.caption(
        "Shows the countries whose misvaluation changed the most since the previous release. "
        "‘Raw’ compares Big Mac dollar prices vs the selected base currency; ‘GDP-adjusted’ controls for income effects."
    )


        colA, colB = st.columns(2)

        # Top raw movers
        top_raw = movers.reindex(movers["raw_change"].abs().sort_values(ascending=False).index).head(5)
        with colA:
            st.caption(f"Raw vs {base_currency}")
            RAW_ORANGE = "#ff914d"
            RAW_BLUE   = "#4284ce"

            for _, r in top_raw.iterrows():
                value = r[base_currency]
                delta = r["raw_change"]

                color = RAW_ORANGE if delta < 0 else RAW_BLUE
                sign  = "+" if delta >= 0 else ""

                st.markdown(
                    f"""
                    <div style="
                        border-radius: 10px;
                        padding: 14px 18px;
                        margin-bottom: 8px;
                        background-color: #111827;
                    ">
                    <div style="font-size:14px; opacity:0.8;">{r['name']}</div>
                    <div style="font-size:28px; font-weight:600; margin-top:4px;">
                        {value:+.2%}
                    </div>
                    <div style="
                        display:inline-block;
                        margin-top:8px;
                        padding:2px 10px;
                        border-radius:999px;
                        font-size:13px;
                        color:{color};
                        background-color:rgba(255,255,255,0.06);
                    ">
                        {sign}{delta:.2%}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        # Top adjusted movers
        top_adj = movers.reindex(
        movers["adj_change"].abs().sort_values(ascending=False).index
    ).head(5)

    with colB:
        st.caption("GDP-adjusted")
        RAW_ORANGE = "#ff914d"
        RAW_BLUE = "#4284ce"

        for _, r in top_adj.iterrows():
            value = r["adjusted"]
            delta = r["adj_change"]
            color = RAW_ORANGE if delta < 0 else RAW_BLUE
            sign = "+" if delta >= 0 else ""

            st.markdown(
                f"""
                <div style="
                    border-radius: 10px;
                    padding: 14px 18px;
                    margin-bottom: 8px;
                    background-color: #111827;
                ">
                <div style="font-size:14px; opacity:0.8;">{r['name']}</div>
                <div style="font-size:28px; font-weight:600; margin-top:4px;">
                    {value:+.2%}
                </div>
                <div style="
                    display:inline-block;
                    margin-top:8px;
                    padding:2px 10px;
                    border-radius:999px;
                    font-size:13px;
                    color:{color};
                    background-color:rgba(255,255,255,0.06);
                ">
                    {sign}{delta:.2%}
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    # plot raw index
    st.subheader(f"Raw Big Mac Index vs {base_currency} on {selected_date.date()}")
    st.caption(
        "This is the classic Big Mac Index: positive bars mean the currency looks overvalued vs the base; "
        "negative bars mean undervalued. Values are computed from Big Mac prices converted to dollars."
    )

    df_date['overvalued'] = df_date[base_currency] > 0

    fig1 = px.bar(df_date, y='name', x=base_currency, color='overvalued',
                  labels={'name': 'Country', base_currency: 'Index Value'},
                  color_discrete_map={True: "#4284ce", False: "#ff914d"},
                  orientation='h')
    fig1.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title='Index (over/undervaluation)', showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # plot adjusted index
    st.subheader(f"Adjusted Big Mac Index (GDP adjusted) on {selected_date.date()}")
    st.caption(
        "The adjusted index accounts for the tendency of richer countries to have higher prices. "
        "It estimates a ‘fair value’ using the relationship between price levels and GDP per capita."
    )

    df_date['adjusted_overvalued'] = df_date['adjusted'] > 0

    fig2 = px.bar(df_date, y='name', x='adjusted', color='adjusted_overvalued',
                  labels={'name': 'Country', 'adjusted': 'Adjusted Index Value'},
                  color_discrete_map={True: "#4284ce", False: "#ff914d"},  # True=orange, False=blue
                  orientation='h')
    fig2.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title='Adjusted Index', showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- MAP SECTION ----------------
    st.subheader(f"Map view: Raw Big Mac Index vs {base_currency}")
    st.caption("Countries with Big Mac data are colored; all others are shown in the default land color.")

    df_date_map = df_date.copy()

    EURO_MEMBERS = [
        "AUT", "BEL", "DEU", "ESP", "FIN", "FRA", "GRC", "IRL", "ITA",
        "NLD", "PRT", "EST"
    ]

    # Replicate euro area (EUZ) value over member countries if present
    eu_row = df_date_map[df_date_map["iso_a3"] == "EUZ"]
    if not eu_row.empty:
        eu_val = eu_row.iloc[0]
        replicas = []
        for code in EURO_MEMBERS:
            r = eu_val.copy()
            r["iso_a3"] = code
            replicas.append(r)
        replicas = pd.DataFrame(replicas)
        df_date_map = pd.concat(
            [df_date_map[df_date_map["iso_a3"] != "EUZ"], replicas],
            ignore_index=True,
        )

    map_df = df_date_map.dropna(subset=[base_currency])[
        ["iso_a3", "name", base_currency, "adjusted"]
    ]

    blue_orange = ["#ff914d", "#ffad76", "#7db8fb", "#4284ce"]

    fig_map = px.choropleth(
        map_df,
        locations="iso_a3",
        locationmode="ISO-3",
        color=base_currency,
        hover_name="name",
        hover_data={
            "iso_a3": False,
            base_currency: ":.1%",
            "adjusted": ":.1%",
        },
        labels={
            base_currency: f"Raw misval. vs {base_currency}",
            "adjusted": "GDP‑adjusted misval.",
        },
        color_continuous_scale=blue_orange,
        color_continuous_midpoint=0,
    )

    fig_map.update_layout(
        coloraxis_colorbar=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            yanchor="top",
            lenmode="fraction",
            len=0.7,
            thickness=15,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig_map.update_geos(
        scope="world",
        showcoastlines=True,
        showcountries=True,
        showland=True,
        landcolor="#2A3246",
    )

    # Wrap map in touch‑tuned container for mobile
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st.plotly_chart(
        fig_map,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "doubleClick": "reset",
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")  # small spacer

    # --- Stats for nerds ---
    st.write("")  # small spacer

    with st.expander("Stats for nerds"):
        # ---------- Cross-section stats for this release ----------
        st.markdown("### Cross‑section snapshot")

        # Misvaluation distribution for this release
        misvals = df_date[[ "name", "iso_a3", base_currency, "adjusted" ]].copy()
        misvals["abs_raw"] = misvals[base_currency].abs()

        # Rank of the selected country by raw undervaluation/overvaluation
        misvals["rank_raw"] = misvals[base_currency].rank(method="min", ascending=True)
        misvals["rank_abs"] = misvals["abs_raw"].rank(method="min", ascending=False)

        this = misvals[misvals["name"] == country].iloc[0]

        median_raw = misvals[base_currency].median()
        p10 = misvals[base_currency].quantile(0.10)
        p90 = misvals[base_currency].quantile(0.90)

        # z‑score of current misvaluation vs cross‑section
        mean_raw = misvals[base_currency].mean()
        std_raw = misvals[base_currency].std(ddof=0)
        if std_raw > 0:
            z_score = (this[base_currency] - mean_raw) / std_raw
        else:
            z_score = np.nan

        num_over = (misvals[base_currency] > 0).sum()
        num_under = (misvals[base_currency] < 0).sum()
        total_c = len(misvals)

        col_cs1, col_cs2 = st.columns(2)

        with col_cs1:
            st.markdown("**Where this country sits today**")
            st.write(f"Countries this release: {total_c}")
            st.write(
                f"Raw misvaluation vs {base_currency}: "
                f"{this[base_currency]:+.2%} "
                f"(rank {int(this['rank_abs'])} by magnitude)"
            )
            st.write(
                f"Cross‑section median misvaluation: {median_raw:+.2%} "
                f"(10th pct: {p10:+.2%}, 90th pct: {p90:+.2%})"
            )
            if not np.isnan(z_score):
                st.write(f"Z‑score of misvaluation: {z_score:+.2f}σ")

        with col_cs2:
            st.markdown("**Market wide picture**")
            st.write(f"Overvalued vs {base_currency}: {num_over} of {total_c}")
            st.write(f"Undervalued vs {base_currency}: {num_under} of {total_c}")
            st.write(
                "Rule‑of‑thumb: |misvaluation| ≳ 2σ is a ‘big’ deviation "
                "from burger‑based parity."
            )

        st.markdown("---")

        # ---------- Snapshot & raw vs adjusted gap ----------
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Selected snapshot**")
            st.write(f"Release date: {selected_date.date()}")
            st.write(f"Base currency: {base_currency}")
            st.write(f"Country: {country_row['name']} ({country_row['iso_a3']})")
            st.write(
                f"Local Big Mac price: {country_row['local_price']:.2f} "
                f"{country_row['currency_code']}"
            )
            st.write(f"Dollar price: ${country_row['dollar_price']:.2f}")

            # Implied PPP rate vs base currency (relative to USD)
            st.markdown("**Raw vs GDP‑adjusted**")
            raw_val = country_row[base_currency]
            adj_val = country_row["adjusted"]
            gap = raw_val - adj_val
            st.write(f"Raw misvaluation: {raw_val:+.2%}")
            st.write(f"GDP‑adjusted misvaluation: {adj_val:+.2%}")
            st.write(
                f"Gap (raw − adjusted): {gap:+.2%} "
                "(portion explained by income differences vs abnormal pricing)."
            )

            # Raw data toggle for THIS release
            st.markdown("**Raw data for this release**")
            if st.checkbox("Show raw data for selected date", key="raw_date_checkbox"):
                st.dataframe(
                    df_date[
                        [
                            "name",
                            "iso_a3",
                            "currency_code",
                            "local_price",
                            "dollar_ex",
                            "dollar_price",
                        ]
                        + base_currencies
                        + ["adjusted"]
                    ].sort_values("name"),
                    use_container_width=True,
                )

        # ---------- Time travel for this country ----------
        with col_b:
            st.markdown("**Time travel for this country**")

            country_history = df[df["name"] == country].sort_values("date")

            fig_hist = px.line(
                country_history,
                x="date",
                y=[base_currency, "adjusted"],
                labels={
                    "value": "Misvaluation",
                    "date": "Release date",
                    "variable": "Index type",
                },
                color_discrete_map={
                    base_currency: "#4284ce",
                    "adjusted": "#ff914d",
                },
            )
            fig_hist.update_layout(
                legend_title_text="",
                yaxis_title=f"vs {base_currency}",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            if st.checkbox("Show full time series table", key="raw_time_checkbox"):
                st.dataframe(
                    country_history[
                        ["date", base_currency, "adjusted", "dollar_price", "local_price"]
                    ].sort_values("date"),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()