import pandas as pd
import numpy as np
import multiprocessing
import time
import logging
import os

def create_input_dataset_raw_file(params):
    sp, k_range, t_range, sigma_range, rf, simulations = params
    result = []
    for k in k_range:
        for t in t_range:
            for volatility in sigma_range:
                result.append({'Spot Price': sp, 'Strike Price': round(k, 2), 'Maturity':t,'risk_free_interest':0.01,
                               'Volatility': round(volatility,2)})
    return result

if __name__ == '__main__':
    S = [50, 2000]
    percentile_factor = 30
    rf = 0.01
    maturity = [90,730]
    sigma = [0.4, 0.9]  # range from 40 to 80
    simulations = 100000

    # Set the number of processes
    num_processes = multiprocessing.cpu_count()

    # Generate the parameter combinations for multiprocessing
    params_list = []
    for sp in range(S[0], S[1], 100):
        k_range = np.arange(sp - int(sp * percentile_factor / 100), sp + int(sp * percentile_factor / 100) + 1,
                            round((sp*0.5/ 100), 2))
        sigma_range = np.arange(sigma[0],sigma[1],0.1)
        params_list.extend([(sp, k_range, range(maturity[0], maturity[1], 30), sigma_range, rf, simulations)])

    start_time = time.time()

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)

    # Initialize logging
    log_file = 'logs/createRawFile.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    try:
        logging.info(f"Execution started by {os.path.basename(__file__)}")
        # Execute the calculations in parallel using the multiprocessing pool
        results = pool.map(create_input_dataset_raw_file, params_list)

        # Concatenate the results from all processes into a single list
        result = [item for sublist in results for item in sublist]

        df = pd.DataFrame(result)
        df.to_csv('InputDataSetLatest.csv', mode='w', index=False)
    except Exception as e:
        logging.error(f"An exception occurred: {str(e)}", exc_info=True)
    finally:
        pool.close()
        pool.join()

        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Total Execution time: {execution_time} seconds")
        logging.info(f"Execution ended by {os.path.basename(__file__)}")
