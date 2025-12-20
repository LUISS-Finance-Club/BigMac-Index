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
    df['GDP_local'] = pd.to_numeric(df['GDP_local'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['date', 'name'])

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
    st.markdown("orem ipsum dolor sit amet, consectetur adipiscing elit.orem ipsum dolor sit amet, consectetur adipiscing elit." \
    "orem ipsum dolor sit amet, consectetur adipiscing elit.orem ipsum dolor sit amet, consectetur adipiscing elit." \
    "orem ipsum dolor sit amet, consectetur adipiscing elit.orem ipsum dolor sit amet, consectetur adipiscing elit." \
    "orem ipsum dolor sit amet, consectetur adipiscing elit.orem ipsum dolor sit amet, consectetur adipiscing elit." \
    "orem ipsum dolor sit amet, consectetur adipiscing elit.orem ipsum dolor sit amet, consectetur adipiscing elit.")
    st.write("")

    st.write("")  # spacing [web:625]
    #st.image(str(INTRO_IMG), width="stretch")
    st.write("")  # spacing [web:625]


    # --- Intro paragraph ---
    st.markdown(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    )

    st.write("")  # spacing [web:625]

    # --- Big Mac hero image ---
    st.image(str(HERO_IMG), width="stretch")
    st.caption("Graphics by LUISS Finance Club")

    st.write("")  # spacing [web:625]

    # --- Second paragraph ---
    st.markdown(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
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
    df = calc_adjusted_index(df, regression_countries)

    # date
    # mapping of year strings to full dates (taking unique years)
    years = sorted(df['date'].dt.year.unique())

    # dict: year string -> list of full dates in that year
    year_to_dates = {year: df[df['date'].dt.year == year]['date'].unique() for year in years}

    # year in sidebar
    #selected_year = st.sidebar.selectbox("Select Year", options=years, index=len(years)-1)
    # from selected year, select a specific date (e.g. latest date in that year)
    #selected_date = year_to_dates[selected_year][-1]  # last date in that year

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

    # base currency selector with default USD
    base_currency = st.sidebar.selectbox("Select Base Currency", options=base_currencies, index=base_currencies.index('USD'))

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

        st.subheader(f"Biggest movers since {prev_date.date()}")
        st.caption(
        "Shows the countries whose misvaluation changed the most since the previous release. "
        "‘Raw’ compares Big Mac dollar prices vs the selected base currency; ‘GDP-adjusted’ controls for income effects."
    )


        colA, colB = st.columns(2)

        # Top raw movers
        top_raw = movers.reindex(movers["raw_change"].abs().sort_values(ascending=False).index).head(5)
        with colA:
            st.caption(f"Raw vs {base_currency}")
            for _, r in top_raw.iterrows():
                st.metric(
                    label=r["name"],
                    value=f"{r[base_currency]:+.2%}",
                    delta=f"{r['raw_change']:+.2%}",
                    border=True,
                )

        # Top adjusted movers
        top_adj = movers.reindex(movers["adj_change"].abs().sort_values(ascending=False).index).head(5)
        with colB:
            st.caption("GDP-adjusted")
            for _, r in top_adj.iterrows():
                st.metric(
                    label=r["name"],
                    value=f"{r['adjusted']:+.2%}",
                    delta=f"{r['adj_change']:+.2%}",
                    border=True,
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

    # GDP vs dollar price scatter with regression
    st.subheader("GDP vs Dollar Price with Linear Regression (Log-Log Scale)")

    st.caption(
        "Each point is a country. The fitted line shows the typical relationship between income (GDP per capita) "
        "and Big Mac dollar prices; deviations from the line drive the GDP-adjusted misvaluation."
    )

    gdp_df = df[(df['date'] == selected_date) & (df['GDP_local'] > 0) & (df['iso_a3'].isin(regression_countries))]

    fig3, ax = plt.subplots(figsize=(8,5))
    sns.regplot(x=np.log(gdp_df['GDP_local']), y=np.log(gdp_df['dollar_price']), ax=ax)
    ax.set_xlabel("Log(GDP per capita)")
    ax.set_ylabel("Log(Big Mac Dollar Price)")
    st.pyplot(fig3)

    # raw data option
    if st.checkbox("Show raw data for selected date"):
        st.write(df_date[['name', 'iso_a3', 'currency_code', 'local_price', 'dollar_ex', 'dollar_price'] + base_currencies + ['adjusted']])

    st.subheader(f"Map view: Raw Big Mac Index vs {base_currency}")
    st.caption("Countries with Big Mac data are colored; all others are shown in the default land color.")

    map_df = df_date[["iso_a3", "name", base_currency, "adjusted"]].copy()
    map_df = df_date.dropna(subset=[base_currency])[["iso_a3", "name", base_currency, "adjusted"]]

    blue_orange = ["#ff914d", "#ffad76", "#7db8fb", "#4284ce"]

    fig_map = px.choropleth(
        map_df,
        locations="iso_a3",
        locationmode="ISO-3",
        color=base_currency,
        hover_name="name",  # big country name at top
        hover_data={
            "iso_a3": False,       # hide ISO code from tooltip
            base_currency: ":.1%", # show raw misvaluation as percent
            "adjusted": ":.1%",    # GDP‑adjusted misvaluation as percent
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
            orientation="h",   # horizontal
            x=0.5,             # center of the plot (0 = left, 1 = right)
            xanchor="center",
            y=-0.15,           # a bit below the map; tweak if needed
            yanchor="top",
            lenmode="fraction",
            len=0.7,           # 70% of width
            thickness=15,      # height of the bar
        ),
    )





    # Make sure the whole world is visible + style “no data” countries
    fig_map.update_geos(
        scope="world",
        showcoastlines=True,
        showcountries=True,
        showland=True,
        landcolor="#2A3246",   # <- “no data” countries
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    st.plotly_chart(fig_map, use_container_width=True)



if __name__ == "__main__":
    main()
