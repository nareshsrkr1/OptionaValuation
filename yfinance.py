import yfinance as yf
import pandas as pd
import random
from datetime import timedelta, datetime

symbol = 'AAPL'  # Replace with the desired symbol

# Get the stock data
stock = yf.Ticker(symbol)
stock_data = stock.history(period='1d')
latest_stock_price = stock_data['Close'].iloc[-1]

# Get the available expiration dates
expirations = list(stock.options)  # Convert to a list

# Generate random expiration dates within the range of 90 days to 1 year
num_expirations = 5
random_expirations = set()
today = datetime.today().date()

while len(random_expirations) < num_expirations:
    expiration = random.choice(expirations)
    expiration_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    days_diff = (expiration_date - today).days

    if 90 <= days_diff <= 365 and expiration not in random_expirations:
        random_expirations.add(expiration)

# Get the options data for the random expiration dates
options_data = []
for expiration in random_expirations:
    options_data_temp = stock.option_chain(expiration).calls
    options_data_temp['expirationDate'] = datetime.strptime(expiration, "%Y-%m-%d").date() - today
    options_data.append(options_data_temp.copy())  # Use copy to avoid overwriting

# Combine the options data for all expiration dates
options_calls = pd.concat(options_data)

# Filter and format the options data
options_calls = options_calls[['strike', 'expirationDate', 'impliedVolatility', 'lastPrice']]
options_calls.rename(columns={'impliedVolatility': 'volatility', 'lastPrice': 'callValue'}, inplace=True)

# Add spot price column
options_calls['spotPrice'] = latest_stock_price

# Filter strike prices within 5% of the spot price
spot_price = latest_stock_price
options_calls = options_calls[(options_calls['strike'] >= spot_price * 0.95) & (options_calls['strike'] <= spot_price * 1.05)]

# Display the data
print(f"Latest stock price for {symbol}: {latest_stock_price}")
print(options_calls)
