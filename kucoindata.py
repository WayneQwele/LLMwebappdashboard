
from dataextraction import *
# this file is used to pull data from kucoin api


def get_kucoin_data(ticker_list: list):
    orderbook_data = {}
    candle_data = {}
    market_data = {}
    i = 0
    for key in ticker_list:
        print(key)

        df_market = marketstatspull(key)
        market_data[key] = df_market
        print(df_market.shape)

        df_orderbook = orderbookpull(key)
        orderbook_data[key] = df_orderbook
        print(df_orderbook.shape)

        df_candle = main_candle(key)
        candle_data[key] = df_candle
        print(df_candle.shape)

    return orderbook_data, candle_data, market_data


def get_kucoin_candle_data(ticker_list: list):
    candle_data = {}
    i = 0
    for key in ticker_list:
        print(key)

        df_candle = main_candle(key)
        candle_data[key] = df_candle
        print(df_candle.shape)

    return candle_data
