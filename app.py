import tensorflow as tf
import pandas as pd
import joblib
import logging
import yaml
import torch
from ComputeBS_MC import monte_carlo_call
import time
import streamlit as st

# Load configuration from YAML file
with open('config/config_model.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

model_filename = config['files']['model_filename']
scaler_filename = config['files']['scaler_filename']

logging.basicConfig(filename='logs/OptionEvaluation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model and scaler
try:
    loaded_model = tf.keras.models.load_model(model_filename)
    scaler = joblib.load(scaler_filename)
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error("An error occurred while loading the model and scaler: %s", str(e))

# Define the Streamlit app
def main():
    # st.set_theme("dark")


    # Create tabs
    tabs = ["Option Pricing", "Comparision"]
    active_tab = st.sidebar.radio("Select Mode", tabs)

    if active_tab == "Option Pricing":
        optionPricing()
    elif active_tab == "Comparision":
        multiple_predictions()

def optionPricing():
    st.title("Option Pricing App")
    if 'option_value' not in st.session_state:
        st.session_state.option_value = None
    if 'show_table' not in st.session_state:
        st.session_state.show_table = False

    # Input fields
    spot_price = st.number_input("Spot Price", value=50.0, key="spot_price")
    strike_price = st.number_input("Strike Price", value=45.0, key="strike_price")
    maturity = st.number_input("Maturity (in days)", value=90, key="maturity")
    risk_free_interest = st.number_input("Risk-Free Interest Rate", value=0.01, step=0.01, disabled=True, key="risk_free_interest")
    volatility = st.slider("Volatility", min_value=0.1, max_value=1.0, value=0.4, step=0.1, key="volatility")

    # Convert input values to floats
    spot_price = float(spot_price)
    strike_price = float(strike_price)
    maturity = float(maturity)
    risk_free_interest = float(risk_free_interest)
    volatility = float(volatility)

    hide_table_row_index = """
                        <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                        </style>
                        """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    # Predict option value using the model
    if st.button("Predict with Model"):
        try:
            model_option_value = model_custom_predict(spot_price, strike_price, maturity / 365, risk_free_interest, volatility)
            model_option_val = round(model_option_value.item(), 2)
            logging.info("model_option_value " + str(model_option_val))

            st.session_state.option_value = model_option_val
            st.success("The predicted option value with the model is: {:.2f}".format(model_option_val))
            st.session_state.show_table = True

        except Exception as e:
            st.error("An error occurred during model prediction: " + str(e))

    # Display the table if the flag is True
    if st.session_state.show_table:
        input_values = pd.DataFrame({'Spot Price': [spot_price],
                                     'Strike Price': [strike_price],
                                     'Maturity': [maturity],
                                     'Risk-Free Interest Rate': [risk_free_interest],
                                     'Volatility': [volatility],
                                     'Predicted Option Value': [st.session_state.option_value]})

        st.table(input_values.style.format('{:.2f}'))

    # Compare with Monte Carlo simulation
    if st.session_state.option_value is not None:
        if st.button("Compare with Monte Carlo Simulation"):
            try:
                mc_option_value = monte_carlo_call(torch.tensor(spot_price), torch.tensor(strike_price),torch.tensor(risk_free_interest),
                                                   torch.tensor(maturity / 365),
                                                   torch.tensor(volatility))
                mc_option_val = round(mc_option_value.item(), 2) if mc_option_value is not None else None

                diff_percentage = abs(st.session_state.option_value - mc_option_val) / st.session_state.option_value * 100

                comparison_table = pd.DataFrame({'Model Prediction': [st.session_state.option_value],
                                                 'Monte Carlo Simulation': [mc_option_val]})
                                                 # 'Difference (%)': [diff_percentage]})
                st.table(comparison_table.reset_index(drop=True).style.format('{:.2f}'))

            except Exception as e:
                st.error("An error occurred during Monte Carlo simulation: " + str(e))


def multiple_predictions():
    # Read data from CSV file
    records_df = pd.read_csv('random1k.csv')
    st.title("Option Pricing App - Comparison")

    # Add a "Predict" button at the top
    if st.button("Predict Option Values", key="predict_button", help="Click to predict option values"):
        with st.spinner("Predicting option values..."):
            start_time = time.time()

            # Predict option values for all records using apply() with a lambda function
            records_df['Option Value'] = records_df.apply(lambda row: model_custom_predict(row['Spot Price'], row['Strike Price'], row['Maturity'] / 365, row['risk_free_interest'], row['Volatility']), axis=1)

            # Convert option values to appropriate data type
            records_df['Option Value'] = records_df['Option Value'].astype(float)

            end_time = time.time()
            elapsed_time = end_time - start_time
            st.info(f"Calculation completed in {elapsed_time:.2f} seconds.")

    # Display the DataFrame with the option values
    st.dataframe(records_df.style.set_properties(**{'font-weight': 'bold'}))



def model_custom_predict(spot_price, strike_price, maturity, risk_free_interest, volatility):
    inputs_to_model = pd.DataFrame({'Spot Price': spot_price / strike_price,
                                    'Strike Price': strike_price,
                                    'Maturity': maturity,
                                    'risk_free_interest': risk_free_interest,
                                    'Volatility': volatility}, index=[0])  # Specify index as [0]
    input_data_scaled = scaler.transform(inputs_to_model.values)  # Use the 'scaler' variable
    value = loaded_model.predict(input_data_scaled)  # Use the 'loaded_model' variable
    option_value = value * strike_price
    return option_value


if __name__ == "__main__":
    main()
