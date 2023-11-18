

import plotly.graph_objects as go
import warnings  # this is to ignore warnings
import plotly as py
import datetime
import json
import requests
import pandas as pd
from kucoindata import get_kucoin_data
from dataextraction import *
import streamlit.components.v1 as components
import asyncio
from llmsummary import *
import streamlit as st
import numpy as np
import openai

from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
import os
import toml


# Load the secrets file
secrets = toml.load('.secrets/secrets.toml')

# Set the environment variables
for key, value in secrets.items():
    os.environ[key] = value

warnings.filterwarnings('ignore')

###
###

# lets configure a basic sidebar

ticker_list = get_ticker_list()

# lets check if we have an api key
if 'OPENAI_API_KEY' in st.secrets:
    st.success('API key already provided!', icon='✅')
    openai.api_key = st.secrets['OPENAI_API_KEY']
    dict_response = asyncio.run(generate_summary_concurrently(ticker_list))
else:
    print(f'No API key found')

# we pull the data from kucoin
orderbook_data, candle_data, market_data = get_kucoin_data(ticker_list)


def plot_candlestick(ticker_pair):
    st.header(ticker_pair)
    fig = go.Figure(
        data=[go.Candlestick(
            x=candle_data[ticker_pair]['date'],
            open=candle_data[ticker_pair]['open'],
            high=candle_data[ticker_pair]['high'],
            low=candle_data[ticker_pair]['low'],
            close=candle_data[ticker_pair]['close']
        )],
    )
    fig.update_layout(
        yaxis=dict(
            tickformat=":,.2f"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

##########################
# llm summary operations #
#########################


# lets create a title header
st.subheader('Kucoin Data Dashboard')
with st.container():

    st.header('Candle Plot - Daily')
    st.divider()
    # Iterate over the list and create tabs
    # Ticker mapping
    # st.text(f'Fixed width text {type(candle_data)}')

    tab_labels = st.selectbox("Select Ticker Pair", ticker_list)
    # # Iterate over the list and create tabs
    # # Ticker mapping

    ticker_mapping = {item: item for item in ticker_list}

    if tab_labels in ticker_mapping:
        # we pass the ticker pair to the plot candlestick function
        selected_ticker_pair = ticker_mapping[tab_labels]
        # pass the ticker LANGCHAIN prompt summary
        # response = generate_response('3', selected_ticker_pair)
        # st.text(f'Fixed width text {selected_ticker_pair}')
        plot_candlestick(selected_ticker_pair)
        st.markdown(f'AI generated summary:\n  {dict_response[tab_labels]}')
    else:
        st.write("Invalid selection")


# what i want to is query the sticker information


with st.sidebar:
    st.title('🤖💬 OpenAI Chatbot')
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai.api_key = st.secrets['OPENAI_API_KEY']
    else:
        openai.api_key = st.text_input(
            'Enter OpenAI API token:', type='password')
        if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 51):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    # --------------------------------------------------
