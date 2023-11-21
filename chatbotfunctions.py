import streamlit as st
import pandas as pd
# Load CSV file


def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('See DataFrame'):
        st.write(df)
    return df


def load_dictionary_csv(input_data: dict) -> pd.DataFrame:
    """ 
    This function simply combines the value attributes of a dictionary into a single dataframe.
    The key attribute is added as a column to the dataframe.

    """
    combined_df = pd.DataFrame()
    for key, value in input_data.items():
        if isinstance(value, pd.DataFrame):
            value['high'] = value['high'].astype(float)  # Convert 'high' column to float64
            value['low'] = value['low'].astype(float)  # Convert 'low' column to float64
            value['volume'] = value['volume'].astype(float)  # Convert 'volume' column to float64
            value['turnover'] = value['turnover'].astype(float)  # Convert 'turnover' column to float64
            combined_df = pd.concat([combined_df, value], ignore_index=True)
    with st.expander('See DataFrame'):
        st.write(combined_df)
    return combined_df

# Generate LLM response
