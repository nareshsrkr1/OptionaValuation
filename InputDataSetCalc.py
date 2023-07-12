import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask.diagnostics
import time
from dask import delayed
from dask.diagnostics import ProgressBar
import logging

def simulate_option_price(row, num_simulations=100000):
    S = row['Spot Price']
    K = row['Strike Price']
    r = row['risk_free_interest']
    sigma = row['Volatility']
    T = row['Maturity'] / 365

    z = np.random.standard_normal((num_simulations,))

    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoffs = np.maximum(ST - K, 0)
    option_prices = payoffs * np.exp(-r * T)
    option_value = np.mean(option_prices)
    return option_value

def calculate_option_value(df):
    return df.apply(simulate_option_price, axis=1)

if __name__ == '__main__':
    chunksize = 1000
    data_chunks = pd.read_csv('InputDataSetLatest.csv', chunksize=chunksize)
    start_time = time.time()
    log_file = 'logs/InputDataSetCalc.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    try:
        option_values = []
        for i, data_chunk in enumerate(data_chunks, 1):
            start_chunk_time = time.time()
            data_chunk['Call_Premium'] = calculate_option_value(data_chunk)
            data_chunk['Call_Premium'] = data_chunk['Call_Premium'].round(4)
            option_values.append(data_chunk)
            end_chunk_time = time.time()
            elapsed_time = end_chunk_time - start_chunk_time
            logging.info(f"Chunk {i} processed in file {log_file} by {os.path.basename(__file__)}. Time: {elapsed_time:.2f} seconds")
    except Exception as e:
        logging.error(f"An exception occurred in file {log_file} by {os.path.basename(__file__)}: {str(e)}", exc_info=True)
    else:
        result = pd.concat(option_values)
        result.to_csv('InputDataSetLatest.csv', index=False)
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Total Execution time in file {log_file} by {os.path.basename(__file__)}: {execution_time} seconds")
