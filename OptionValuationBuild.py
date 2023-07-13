import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.optimizers import Adam
from keras.losses import Huber
import joblib
import logging

logging.basicConfig(level=logging.INFO, filename='logs/model_building.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


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


def build_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(Activation('elu'))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(Activation('elu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model


if __name__ == '__main__':
    logging.info("Loading dataset...")
    try:
        X, Y = load_dataset('InputDataSetLatest.csv')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        model = build_model(X_train_scaled.shape[1])

        logging.info("Training model...")
        num_epochs = 100
        batch_size = 64
        history = model.fit(X_train_scaled, Y_train, batch_size=batch_size, epochs=num_epochs,
                            validation_split=0.1, verbose=2)

        # Evaluate model on test data
        test_loss = model.evaluate(X_test_scaled, Y_test)
        test_accuracy = 100 - test_loss * 100
        print("Test Accuracy: {:.2f}%".format(test_accuracy))

        # Evaluate model on train data
        train_loss = model.evaluate(X_train_scaled, Y_train)
        train_accuracy = 100 - train_loss * 100
        print("Train Accuracy: {:.2f}%".format(train_accuracy))

        logging.info("Saving model...")
        model.save('OptionValuationModel_opt', save_format='tf')
        logging.info("Model saved successfully.")

        logging.info("Saving scaler...")
        joblib.dump(scaler, 'scalars_model_opt.save')
        logging.info("Scaler saved successfully.")
    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
