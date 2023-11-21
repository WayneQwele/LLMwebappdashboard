


from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import plotly.graph_objects as go
import warnings  # this is to ignore warnings
import plotly as py

import pandas as pd
from kucoindata import get_kucoin_candle_data
from dataextraction import *
import asyncio
from llmsummary import *
import streamlit as st
import numpy as np
import openai

from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

import streamlit as st
import os
import toml
import matplotlib.pyplot as plt

# import chatbot functions
# from chatbotfunctions import *
from chatbotfunctions import load_dictionary_csv
import os
import toml
import streamlit as st
import os
import toml
import openai




def load_secrets():
    try:
        # Load the secrets file
        secrets = toml.load('/.streamlit/secrets.toml') 

        # Set the environment variables
        for key, value in secrets.items():
            os.environ[key] = value

    except Exception as e:
        print(f"Error loading secrets: {str(e)}")

# Load the secrets file
secrets = toml.load('.secrets/secrets.toml')

# Set the environment variables
for key, value in secrets.items():
    os.environ[key] = value

load_secrets()

# Ask the user for OpenAI secret key
openai_api_key = st.text_input('Enter OpenAI API token:', type='password')

# Set the OpenAI API key
if openai_api_key:
    openai.api_key = openai_api_key

warnings.filterwarnings('ignore')

###
###

# lets configure a basic sidebar

ticker_list = get_ticker_list()
dict_response = asyncio.run(generate_summary_concurrently(ticker_list))

# This section loads the api key
# with st.sidebar:
#     st.title('🤖💬 OpenAI Key Check')
#     if 'OPENAI_API_KEY' in st.secrets:
#         st.success('API key already provided!', icon='✅')
#         openai.api_key = st.secrets['OPENAI_API_KEY']
#         dict_response = asyncio.run(generate_summary_concurrently(ticker_list))
#     else:
#         openai.api_key = st.text_input(
#             'Enter OpenAI API token:', type='password')
#         if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 51):
#             st.warning('Please enter your credentials!', icon='⚠️')
#         else:
#             # if key is succesfully entered
#             st.success('Proceed to entering your prompt message!', icon='👉')
#             openai.api_key = st.secrets['OPENAI_API_KEY']
#             dict_response = asyncio.run(
#                 generate_summary_concurrently(ticker_list))


# we pull the candle data from kucoin
candle_data = get_kucoin_candle_data(ticker_list)


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
##########################


# lets create a title header
st.subheader('Kucoin Crypto Data Dashboard')
with st.container():

    st.header('Candle Plot - Daily')
    st.divider()
    # Iterate over the list and create tabs

    tab_labels = st.selectbox("Select Ticker Pair", ticker_list)
    # # Iterate over the list and create tabs
    # # Ticker mapping

    ticker_mapping = {item: item for item in ticker_list}

    if tab_labels in ticker_mapping:
        # we pass the ticker pair to the plot candlestick function
        selected_ticker_pair = ticker_mapping[tab_labels]
        # pass the ticker LANGCHAIN prompt summary
        plot_candlestick(selected_ticker_pair)
        st.markdown(f'AI generated summary:\n  {dict_response[tab_labels]}')
    else:
        st.write("Invalid selection")

with st.container():
    st.header('Chat Bot')
    st.divider()
    # st.text(f'Fixed width text {type(orderbook_data)}')
    # prompt = st.chat_input("Say something")
    # if prompt:
    #     st.write(f"User has sent the following prompt: {prompt}")

    candle_df = load_dictionary_csv(candle_data)
    #st.write(candle_df)


# what i want to is query the sticker information

    # --------------------------------------------------

    #st.write(st.session_state) # this is to check the session state, and track conversation history
    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}]
        #st.write(st.session_state["messages"])

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        #st.write(st.session_state)
        if not openai.api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        llm = ChatOpenAI(
            temperature=0, 
            model="gpt-3.5-turbo-0613", 
            openai_api_key=os.environ['OPENAI_API_KEY'], 
            streaming=True
        )

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            candle_df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            try:
                response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
                #response = st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
