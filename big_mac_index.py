import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm

# Load data from local downloads folder
DATA_PATH = '~/Desktop/big-mac-source-data-v2.csv'  # Change if needed

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, na_values=['#N/A'])
    df = df.dropna(subset=['local_price'])
    df['GDP_local'] = pd.to_numeric(df['GDP_local'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['date', 'name'])

# Calculate dollar price
def calc_dollar_price(df):
    df['dollar_price'] = df['local_price'] / df['dollar_ex']
    return df

# Calculate raw Big Mac index relative to base currencies
def calc_raw_index(df, big_mac_countries, base_currencies):
    df = df[df['iso_a3'].isin(big_mac_countries)].copy()
    df = df[~df['dollar_price'].isna()]

    for currency in base_currencies:
        base_prices = df[df['currency_code'] == currency][['date', 'dollar_price']].rename(columns={'dollar_price': 'base_price'})
        df = df.merge(base_prices, on='date', how='left', suffixes=('', '_base'))
        df[currency] = (df['dollar_price'] / df['base_price']) - 1
        df.drop(columns=['base_price'], inplace=True)
    df[base_currencies] = df[base_currencies].round(5)
    return df

# Calculate adjusted index by regressing GDP_local on dollar_price and adjusting residuals
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

# List of countries and base currencies (same as R)
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

def main():
    from pathlib import Path
    import streamlit as st

    APP_DIR = Path(__file__).parent
    HERO_IMG = APP_DIR / "Desktop/Big-Mac-2.png"

    st.set_page_config(page_title="Big Mac Index Dashboard", layout="wide")

    st.image(str(HERO_IMG), use_container_width=True)

    st.title("Interactive Big Mac Index Dashboard")

    st.markdown("""
    This dashboard replicates The Economist's Big Mac Index calculations with:
    - Raw index (currency valuation relative to base currencies)
    - Adjusted index (accounting for GDP differences)
    """)

    df = load_data()
    df = calc_dollar_price(df)
    df = calc_raw_index(df, big_mac_countries, base_currencies)
    df = calc_adjusted_index(df, regression_countries)

    # Select date
    # Create a mapping of year strings to full dates (taking unique years)
    years = sorted(df['date'].dt.year.unique())

    # Create a dict: year string -> list of full dates in that year
    year_to_dates = {year: df[df['date'].dt.year == year]['date'].unique() for year in years}

    # Select year in sidebar
    selected_year = st.sidebar.selectbox("Select Year", options=years, index=len(years)-1)

    # From selected year, select a specific date (e.g. latest date in that year)
    selected_date = year_to_dates[selected_year][-1]  # last date in that year


    # Base currency selector with default USD
    base_currency = st.sidebar.selectbox("Select Base Currency", options=base_currencies, index=base_currencies.index('USD'))

    # Filter dataframe for the selected date
    df_date = df[df['date'] == selected_date].copy()
    df_date = df_date.sort_values(by=base_currency)

    # Plot raw index
    st.subheader(f"Raw Big Mac Index vs {base_currency} on {selected_date.date()}")
    df_date['overvalued'] = df_date[base_currency] > 0

    fig1 = px.bar(df_date, y='name', x=base_currency, color='overvalued',
                  labels={'name': 'Country', base_currency: 'Index Value'},
                  color_discrete_map={True: 'red', False: 'green'},
                  orientation='h')
    fig1.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title='Index (over/undervaluation)', showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # Plot adjusted index
    st.subheader(f"Adjusted Big Mac Index (GDP adjusted) on {selected_date.date()}")
    df_date['adjusted_overvalued'] = df_date['adjusted'] > 0

    fig2 = px.bar(df_date, y='name', x='adjusted', color='adjusted_overvalued',
                  labels={'name': 'Country', 'adjusted': 'Adjusted Index Value'},
                  color_discrete_map={True: 'red', False: 'green'},
                  orientation='h')
    fig2.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title='Adjusted Index', showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # GDP vs Dollar Price Scatter with regression
    st.subheader("GDP vs Dollar Price with Linear Regression (Log-Log Scale)")

    gdp_df = df[(df['date'] == selected_date) & (df['GDP_local'] > 0) & (df['iso_a3'].isin(regression_countries))]

    fig3, ax = plt.subplots(figsize=(8,5))
    sns.regplot(x=np.log(gdp_df['GDP_local']), y=np.log(gdp_df['dollar_price']), ax=ax)
    ax.set_xlabel("Log(GDP per capita)")
    ax.set_ylabel("Log(Big Mac Dollar Price)")
    st.pyplot(fig3)

    # Show raw data option
    if st.checkbox("Show raw data for selected date"):
        st.write(df_date[['name', 'iso_a3', 'currency_code', 'local_price', 'dollar_ex', 'dollar_price'] + base_currencies + ['adjusted']])

if __name__ == "__main__":
    main()
