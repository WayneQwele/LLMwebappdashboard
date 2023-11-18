

import pandas as pd
import requests
import json
import streamlit as st
import plotly.graph_objects as go


@st.cache_data
def main_candle(symbol: str = 'BTC-USDT') -> pd.DataFrame:
    """ 
    This function pulls the daily candle data from Kucoin for a given symbol.
    It accepts a symbol parameter, which is a string in the format of BASE-QUOTE,
    and returns a pandas dataframe of the daily candle data.
    """
    url = f'https://api.kucoin.com/api/v1/market/candles?type=1day&symbol={symbol}'
    response = requests.get(url)  # call the API
    if response.status_code == 200:  # if result is GOOD
        data = json.loads(response.text)  # load the json response from API
        # create dataframe out of json response
        data_pd = pd.DataFrame(data['data'], columns=[
                               'unix', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        data_pd['date'] = pd.to_datetime(
            data_pd['unix'], unit='s')  # add a human readable date

        # remove the - in the symbol name for the csv file
        pair = symbol.replace("-", "_")
        filename = f'Kucoin_{pair}_day.csv'   # set a filename variable
        data_pd['symbols'] = symbol
        # data_pd.to_csv(filename, index=False)   # write to CSV
        return data_pd


@st.cache_data
def orderbookpull(symbol: str = 'LOOM-BTC') -> pd.DataFrame:
    """ 
    This function pulls the orderbook data from the Kucoin API and returns a pandas dataframe

    """
    url = f'https://api.kucoin.com/api/v1/market/orderbook/level2_20?symbol={symbol}'
    response = requests.get(url)

    if response.status_code == 200:  # if result is GOOD
        orderbook = requests.get(url)
        orderbook = orderbook.json()

        bids = pd.DataFrame(orderbook['data']['bids'])
        asks = pd.DataFrame(orderbook['data']['asks'])

        df = pd.merge(bids, asks, left_index=True, right_index=True)
        df = df.rename({"0_x": "Bid Price", "1_x": "Bid Amount",
                        "0_y": "Ask Price", "1_y": "Ask Amount"}, axis='columns')
        df['symbols'] = symbol
        return df
    else:  # if result is BAD
        print(f"Error: {response.status_code}")
        return None


@st.cache_data
def marketstatspull(symbol: str = 'LOOM-BTC') -> pd.DataFrame:
    """ 
    This function pulls the market data from the Kucoin API and returns a pandas dataframe

    """
    url = f'https://api.kucoin.com/api/v1/market/stats?symbol={symbol}'
    response = requests.get(url)

    if response.status_code == 200:  # if result is GOOD
        marketdata = requests.get(url)
        marketdata = marketdata.json()

        df = pd.DataFrame(marketdata['data'], index=[0])

        df['symbols'] = symbol
        return df
    else:  # if result is BAD
        print(f"Error: {response.status_code}")
        return None

    if __name__ == "__main__":
        pass


#
def clean_dataframe(df):
    # Convert symbol and symbolName to string
    df['symbol'] = df['symbol'].astype(str)
    df['symbolName'] = df['symbolName'].astype(str)

    # Convert the specified columns to float
    float_columns = ['buy', 'sell', 'changeRate', 'changePrice', 'high', 'low',
                     'vol', 'volValue', 'last', 'averagePrice', 'takerFeeRate', 'makerFeeRate']
    df[float_columns] = df[float_columns].apply(
        pd.to_numeric, errors='coerce', downcast='float')

    # Convert takerCoefficient and makerCoefficient to int
    int_columns = ['takerCoefficient', 'makerCoefficient']
    df[int_columns] = df[int_columns].apply(
        lambda x: x.astype(str).str.replace(',', '').astype(int))
    return df


# gets top 20 ticker information
def get_ticker_list():
    tickers = requests.get('https://api.kucoin.com/api/v1/market/allTickers')
    tickers = tickers.json()
    tickers = tickers['data']['ticker']
    ticker_df = pd.DataFrame(tickers)

    # clean up the dataframe
    ticker_df = clean_dataframe(ticker_df)

    filtered_ticker_pairs = ticker_df.sort_values(
        by='volValue', ascending=False).head(20)

    ticker_list = filtered_ticker_pairs['symbol'].tolist()
    return ticker_list
