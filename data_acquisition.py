import yfinance as yf

# Define the ticker symbol
ticker_symbol = "RELIANCE.NS"

# Download historical data
data = yf.download(ticker_symbol, period="1y", interval="1d")

# Print the downloaded data
print(data)
