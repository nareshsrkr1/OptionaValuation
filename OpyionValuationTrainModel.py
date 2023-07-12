import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.optimizers import Adam
import joblib
import yaml
import logging
from json_formatter import JSONFormatter
import numpy as np


def load_dataset(filename):
    df = pd.read_csv(filename)
    df['Maturity'] = df['Maturity'] / 365
    df['Spot Price'] = df['Spot Price'] / df['Strike Price']
    df['Call_Premium'] = df['Call_Premium'] / df['Strike Price']
    X = df.drop('Call_Premium', axis=1)
    Y = df['Call_Premium']
    return X, Y


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def build_model(input_dim, nodes, optimizer, loss, learning_rate, dropout_rate, activation):
    model = Sequential()
    model.add(Dense(nodes, input_dim=input_dim))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_rate))
    model.add(Dense(nodes, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(nodes, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(nodes, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def train_model(model, X_train_scaled, Y_train, batch_size, num_epochs, validation_split):
    logging.info("Training model...")
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Epoch {epoch}/{num_epochs}")
        model.fit(X_train_scaled, Y_train, batch_size=batch_size, epochs=1, validation_split=validation_split, verbose=2)


def evaluate_model(model, X_scaled, Y, data_type):
    y_pred = model.predict(X_scaled)
    accuracy = 100 - (np.mean(np.abs((Y - y_pred) / Y)) * 100)
    logging.info("Accuracy on {} Data: {:.2f}%".format(data_type, accuracy))


def save_model(model, model_filename):
    logging.info("Saving model...")
    model.save(model_filename, save_format='tf')
    logging.info("Model saved successfully.")


def save_scaler(scaler, scaler_filename):
    logging.info("Saving scaler...")
    joblib.dump(scaler, scaler_filename)
    logging.info("Scaler saved successfully.")


def setup_logging(log_level, log_filename, log_format):
    logging.basicConfig(level=log_level, filename=log_filename, filemode='w', format=log_format)


if __name__ == '__main__':
    # Load configuration from YAML file
    with open('config/config_model.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract configuration values
    dataset_filename = config['input']['dataset_filename']
    nodes = config['model']['nodes']
    optimizer = config['model']['optimizer']
    loss = config['model']['loss']
    learning_rate = config['model']['learning_rate']
    dropout_rate = config['model']['dropout_rate']
    activation = config['model']['activation']
    num_epochs = config['model']['num_epochs']
    batch_size = config['model']['batch_size']
    validation_split = config['model']['validation_split']
    log_level = config['logging']['level']
    log_filename = config['logging']['filename']
    log_format = config['logging']['format']
    model_filename = config['files']['model_filename']
    scaler_filename = config['files']['scaler_filename']

    # Setup logging
    setup_logging(log_level, log_filename, log_format)
    json_formatter = JSONFormatter()

    # Create a logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)

    # Set the JSON formatter for the file handler
    file_handler.setFormatter(json_formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    logger.info("Loading dataset...")
    try:
        X, Y = load_dataset(dataset_filename)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        model = build_model(X_train_scaled.shape[1], nodes, optimizer, loss, learning_rate, dropout_rate, activation)

        # Train model
        train_model(model, X_train_scaled, Y_train, batch_size, num_epochs, validation_split)

        y_pred = model.predict(X_test_scaled)
        y_test_reshaped = np.ravel(Y_test)
        y_pred_reshaped = np.ravel(y_pred)

        y_test_reshaped_denormalized = y_test_reshaped * X_test['Strike Price'];
        y_pred_reshaped_denormalized = y_pred_reshaped * X_test['Strike Price'];

        # ERROR PERCENTAGE IN PREDICTION
        error_percent = np.mean(
            np.abs((y_test_reshaped_denormalized - y_pred_reshaped_denormalized) / y_test_reshaped_denormalized)) * 100

        print("Average Error Percentage: {:.2f}%".format(error_percent))
        print("Hence Accuracy is 100-error% so Acuuracy: {:.2f}%".format(100 - error_percent))


        # Evaluate model on test data
        evaluate_model(model, X_test_scaled, Y_test, "Test")

        # Evaluate model on train data
        evaluate_model(model, X_train_scaled, Y_train, "Train")

        # Save model and scaler
        save_model(model, model_filename)
        save_scaler(scaler, scaler_filename)
    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
