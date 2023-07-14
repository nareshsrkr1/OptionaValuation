import torch
from torch.autograd import Variable
import math
import logging


def black_scholes_call(S, K, r, T, sigma):
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)

    Nd1 = 0.5 * (1 + torch.erf(d1 / math.sqrt(2)))
    Nd2 = 0.5 * (1 + torch.erf(d2 / math.sqrt(2)))

    call_price = S * Nd1 - K * torch.exp(-r * T) * Nd2

    return call_price


def monte_carlo_call(S, K, r, T, sigma, n_simulations=100000):
    # Generate random numbers
    rand_numbers = torch.randn(n_simulations)

    # Calculate simulated stock prices
    simulated_prices = S * torch.exp((r - 0.5 * sigma ** 2) * T + sigma * torch.sqrt(T) * rand_numbers)

    # Calculate simulated call option payoffs
    call_payoffs = torch.max(simulated_prices - K, torch.tensor(0.0))

    # Calculate option value using Monte Carlo estimation
    call_option_value = torch.exp(-r * T) * torch.mean(call_payoffs)

    return call_option_value


if __name__ == '__main__':
    # Initialize logging
    log_file = 'logs/execution.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    try:
        # print(model_custom_predict(2500, 2450.5, 2, 0.01,0.6))
        # # # 450,494.95000000002045,330,70,126.07
        # 550, 701.25, 90, 0.01, 0.8, 41.5699
        # 550, 701.25, 120, 0.01, 0.4, 10.8082

        S = Variable(torch.tensor(50.0), requires_grad=True)  # Underlying asset price
        K = Variable(torch.tensor(45.0), requires_grad=True)  # Strike price
        r = Variable(torch.tensor(0.01), requires_grad=True)  # Risk-free interest rate
        T = Variable(torch.tensor(90/365), requires_grad=True)  # Time to maturity in years
        sigma = Variable(torch.tensor(0.4), requires_grad=True)  # Volatility
        n_simulations = 100000  # Number of Monte Carlo simulations

        # Calculate the Black-Scholes call option price and derivatives
        call_price = black_scholes_call(S, K, r, T, sigma)
        call_price.backward()

        # Access the gradients
        dS = S.grad
        dK = K.grad
        dr = r.grad
        dT = T.grad
        dsigma = sigma.grad

        # Print the Black-Scholes call option price and derivatives
        logging.info("Black-Scholes Call Price: %s", call_price.item())
        logging.info("dS: %s", dS.item())
        logging.info("dK: %s", dK.item())
        logging.info("dr: %s", dr.item())
        logging.info("dT: %s", dT.item())
        logging.info("dsigma: %s", dsigma.item())

        # Estimate the call option value using Monte Carlo simulation
        call_option_value = monte_carlo_call(S, K, r, T, sigma, n_simulations)

        # Print the Monte Carlo call option value
        logging.info("Monte Carlo Call Option Value: %s", call_option_value.item())

    except Exception as e:
        logging.error("An exception occurred: %s", str(e), exc_info=True)
