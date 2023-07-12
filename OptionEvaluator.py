import tensorflow as tf
import pandas as pd
import joblib
import logging
import yaml

# Load configuration from YAML file
with open('config/config_model.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

model_filename = config['files']['model_filename']
scaler_filename = config['files']['scaler_filename']

logging.basicConfig(filename='logs/OptionEvaluation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model
try:
    loaded_model = tf.keras.models.load_model(model_filename)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("An error occurred while loading the model: %s", str(e))

scaler = joblib.load(scaler_filename)

def model_custom_predict(Spot_Price, Strike_Price, Maturity, risk_free_interest, Volatility):
    try:
        inputs_to_model = pd.DataFrame({'Spot Price': [Spot_Price / Strike_Price],
                                        'Strike Price': [Strike_Price],
                                        'Maturity': [Maturity / 365],
                                        'risk_free_interest': [risk_free_interest],
                                        'Volatility': [Volatility]
                                        })
        input_data_scaled = scaler.transform(inputs_to_model)
        value = loaded_model.predict(input_data_scaled)
        option_value = value * Strike_Price
        return option_value
    except Exception as e:
        logging.error("An error occurred while predicting the option value: %s", str(e))

try:
    result = model_custom_predict(50, 45, 90, 0.01, 0.4)
    print(result)
    logging.info("Option value prediction: %s", result)
except Exception as e:
    logging.error("An error occurred during option value prediction: %s", str(e))
