import tensorflow as tf
import pandas as pd
import joblib
import logging
import yaml
import torch
import numpy as np
from ComputeBS_MC import monte_carlo_call

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
    st.title("Option Pricing App")

    # Input fields
    spot_price = st.number_input("Spot Price", value=50.0)
    strike_price = st.number_input("Strike Price", value=45.0)
    maturity = st.number_input("Maturity (in days)", value=90)
    risk_free_interest = st.number_input("Risk-Free Interest Rate", value=0.01, step=0.01, disabled=True)
    volatility = st.slider("Volatility", min_value=0.1, max_value=1.0, value=0.4, step=0.1)

    # Convert input values to floats
    spot_price = float(spot_price)
    strike_price = float(strike_price)
    maturity = float(maturity)
    risk_free_interest = float(risk_free_interest)
    volatility = float(volatility)

    # Predict option value using the model
    if st.button("Predict with Model"):
        try:
            model_option_value = model_custom_predict(spot_price, strike_price, maturity / 365, risk_free_interest,
                                                      volatility)
            model_option_val = str(round(model_option_value[0][0],2))
            # logging.info("model_option_value " + str(model_option_value[0]))
            logging.info("model_option_value " + model_option_val)

            st.success("The predicted option value with the model is: " + model_option_val)

            # Create a table to display inputs and predicted values
            inputs = pd.DataFrame({'Spot Price': [spot_price],
                                   'Strike Price': [strike_price],
                                   'Maturity': [maturity],
                                   'Risk-Free Interest Rate': [risk_free_interest],
                                   'Volatility': [volatility],
                                   'Predicted Option Value': model_option_val})

            inputs = inputs.applymap(lambda x: round(x, 2))  # Round the values in the DataFrame

            st.table(inputs)

            # Compare with Monte Carlo simulation
            if st.button("Compare with Monte Carlo"):
                try:
                    mc_option_value = monte_carlo_call(torch.tensor(spot_price), torch.tensor(strike_price),
                                                       torch.tensor(maturity / 365), torch.tensor(risk_free_interest),
                                                       torch.tensor(volatility))
                    # mc_option_value = mc_option_value
                    inputs.loc[1] = ['Monte Carlo Simulation', '', '', '', '', mc_option_value]
                    # inputs = inputs.applymap(lambda x: round(x, 2))  # Round the values in the DataFrame
                    st.table(inputs)
                    diff = abs(model_option_value[0] - mc_option_value[0])
                    st.info("Difference between model prediction and Monte Carlo simulation: {:.2f}".format(diff))
                except Exception as e:
                    st.error("An error occurred during Monte Carlo simulation: " + str(e))

        except Exception as e:
            st.error("An error occurred during model prediction: " + str(e))

# Prediction function
def model_custom_predict(spot_price, strike_price, maturity, risk_free_interest, volatility):
    inputs_to_model = pd.DataFrame({'Spot Price': [spot_price / strike_price],
                                    'Strike Price': [strike_price],
                                    'Maturity': [maturity],
                                    'risk_free_interest': [risk_free_interest],
                                    'Volatility': [volatility]
                                    })
    input_data_scaled = scaler.transform(inputs_to_model.values)  # Use the 'scaler' variable
    value = loaded_model.predict(input_data_scaled)  # Use the 'loaded_model' variable
    option_value = value * strike_price
    return option_value

if __name__ == "__main__":
    main()
