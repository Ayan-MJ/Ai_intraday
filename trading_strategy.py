import pandas as pd
import yfinance as yf

def generate_sma_signals(data):
    """
    Generate trading signals based on SMA crossover strategy.
    
    Args:
        data (pd.DataFrame): DataFrame containing stock price data
        
    Returns:
        pd.DataFrame: DataFrame with added signal column
    """
    # Calculate the 20-day and 50-day Simple Moving Averages
    data['SMA_short'] = data['Close'].rolling(window=20).mean()
    data['SMA_long'] = data['Close'].rolling(window=50).mean()
    
    # Initialize the Signal column to 0
    data['Signal'] = 0
    
    # Iterate through the DataFrame starting from the 50th day
    for i in range(50, len(data)):
        # Buy signal: SMA_short crosses above SMA_long
        if (data['SMA_short'].iloc[i-1] <= data['SMA_long'].iloc[i-1]) and (data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i]):
            data.loc[data.index[i], 'Signal'] = 1
        
        # Sell signal: SMA_short crosses below SMA_long
        elif (data['SMA_short'].iloc[i-1] >= data['SMA_long'].iloc[i-1]) and (data['SMA_short'].iloc[i] < data['SMA_long'].iloc[i]):
            data.loc[data.index[i], 'Signal'] = -1
    
    return data

if __name__ == "__main__":
    # Define the ticker symbol
    ticker_symbol = "RELIANCE.NS"
    
    # Download historical data
    data = yf.download(ticker_symbol, period="1y", interval="1d")
    
    # Generate trading signals
    signals_df = generate_sma_signals(data)
    
    # Print the DataFrame with signals
    print(signals_df)
    
    # Print only rows with buy or sell signals
    signal_rows = signals_df[signals_df['Signal'] != 0]
    print("\nBuy and Sell Signals:")
    print(signal_rows[['Close', 'SMA_short', 'SMA_long', 'Signal']])
