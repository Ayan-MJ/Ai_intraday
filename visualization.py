import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from trading_strategy import generate_sma_signals

def visualize_sma_strategy(ticker_symbol, period="1y", interval="1d"):
    """
    Visualize the SMA crossover strategy for a given ticker.
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        period (str): Time period to download data for
        interval (str): Data interval
    """
    # Download historical data
    print(f"Downloading data for {ticker_symbol}...")
    data = yf.download(ticker_symbol, period=period, interval=interval)
    
    # Generate trading signals
    signals_df = generate_sma_signals(data)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the closing price
    ax.plot(signals_df.index, signals_df['Close'], label='Close Price', alpha=0.5, linewidth=1.5)
    
    # Plot the SMAs
    ax.plot(signals_df.index, signals_df['SMA_short'], label='20-day SMA', linewidth=1)
    ax.plot(signals_df.index, signals_df['SMA_long'], label='50-day SMA', linewidth=1)
    
    # Plot buy signals
    buy_signals = signals_df[signals_df['Signal'] == 1]
    ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', 
               s=100, label='Buy Signal', alpha=1)
    
    # Plot sell signals
    sell_signals = signals_df[signals_df['Signal'] == -1]
    ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', 
               s=100, label='Sell Signal', alpha=1)
    
    # Format the x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.title(f'SMA Crossover Strategy for {ticker_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{ticker_symbol.replace('.', '_')}_sma_strategy.png")
    print(f"Figure saved as {ticker_symbol.replace('.', '_')}_sma_strategy.png")
    
    # Show the figure
    plt.show()
    
    # Print summary of signals
    buy_signals_count = len(buy_signals)
    sell_signals_count = len(sell_signals)
    print(f"\nStrategy Summary for {ticker_symbol}:")
    print(f"Number of Buy Signals: {buy_signals_count}")
    print(f"Number of Sell Signals: {sell_signals_count}")
    
    if not buy_signals.empty and not sell_signals.empty:
        # Calculate simple returns (not accounting for position sizing or compounding)
        print("\nSignal Details:")
        print(buy_signals[['Close', 'SMA_short', 'SMA_long']])
        print(sell_signals[['Close', 'SMA_short', 'SMA_long']])

if __name__ == "__main__":
    # Define the ticker symbol
    ticker_symbol = "RELIANCE.NS"
    
    # Visualize the strategy
    visualize_sma_strategy(ticker_symbol)
