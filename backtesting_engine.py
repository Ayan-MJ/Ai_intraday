import pandas as pd
import yfinance as yf
import numpy as np

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

def backtest_strategy(data, initial_capital=100000):
    """
    Backtest a trading strategy based on signals in the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Close' prices and 'Signal' column
        initial_capital (float): Initial capital to start with
        
    Returns:
        tuple: (DataFrame with backtesting results, final portfolio value)
    """
    # Create a copy of the data to avoid modifying the original
    backtest_df = data.copy()
    
    # Initialize variables
    position = 0  # 0 = no position, 1 = holding
    entry_price = 0.0
    portfolio_value = float(initial_capital)
    cash = float(initial_capital)
    shares = 0
    
    # Create new columns for tracking with appropriate dtypes
    backtest_df['Position'] = 0
    backtest_df['Trade_Price'] = np.nan
    backtest_df['Shares'] = 0
    backtest_df['Cash'] = float(initial_capital)
    backtest_df['Portfolio_Value'] = float(initial_capital)
    
    # Find the first valid index (after SMA_long is calculated)
    start_idx = backtest_df['SMA_long'].first_valid_index()
    if start_idx is None:
        return backtest_df, portfolio_value
    
    # Iterate through the DataFrame starting from where signals begin
    for i in range(backtest_df.index.get_loc(start_idx), len(backtest_df)):
        current_idx = backtest_df.index[i]
        prev_idx = backtest_df.index[i-1] if i > 0 else current_idx
        
        # Get the current signal and price
        signal = int(backtest_df.loc[current_idx, 'Signal'].item())  # Convert to int
        close_price = float(backtest_df.loc[current_idx, 'Close'].item())  # Convert to float
        
        # Copy previous position and portfolio value
        if i > backtest_df.index.get_loc(start_idx):
            backtest_df.loc[current_idx, 'Position'] = int(backtest_df.loc[prev_idx, 'Position'].item())
            backtest_df.loc[current_idx, 'Cash'] = float(backtest_df.loc[prev_idx, 'Cash'].item())
            backtest_df.loc[current_idx, 'Shares'] = int(backtest_df.loc[prev_idx, 'Shares'].item())
        
        # Buy signal and not holding a position
        if signal == 1 and position == 0:
            position = 1
            entry_price = close_price
            shares = int(cash / close_price)  # Buy as many whole shares as possible
            cash = float(cash - (shares * close_price))
            
            # Update the DataFrame
            backtest_df.loc[current_idx, 'Position'] = position
            backtest_df.loc[current_idx, 'Trade_Price'] = entry_price
            backtest_df.loc[current_idx, 'Shares'] = shares
            backtest_df.loc[current_idx, 'Cash'] = cash
        
        # Sell signal and holding a position
        elif signal == -1 and position == 1:
            position = 0
            exit_price = close_price
            cash = float(cash + (shares * exit_price))
            
            # Update the DataFrame
            backtest_df.loc[current_idx, 'Position'] = position
            backtest_df.loc[current_idx, 'Trade_Price'] = exit_price
            backtest_df.loc[current_idx, 'Shares'] = 0
            backtest_df.loc[current_idx, 'Cash'] = cash
            shares = 0
        
        # Calculate portfolio value (cash + value of holdings)
        portfolio_value = float(cash + (shares * close_price))
        backtest_df.loc[current_idx, 'Portfolio_Value'] = portfolio_value
    
    # Calculate performance metrics
    initial_close = float(backtest_df['Close'].iloc[backtest_df.index.get_loc(start_idx)].item())
    final_close = float(backtest_df['Close'].iloc[-1].item())
    buy_and_hold_return = (final_close / initial_close - 1) * 100
    strategy_return = (portfolio_value / initial_capital - 1) * 100
    
    print(f"\nBacktesting Results:")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${portfolio_value:.2f}")
    print(f"Strategy Return: {strategy_return:.2f}%")
    print(f"Buy and Hold Return: {buy_and_hold_return:.2f}%")
    print(f"Outperformance: {strategy_return - buy_and_hold_return:.2f}%")
    
    # Calculate additional metrics
    trades = backtest_df[backtest_df['Trade_Price'].notna()]
    num_trades = len(trades)
    winning_trades = 0
    losing_trades = 0
    
    if num_trades > 1:
        # Analyze trade pairs (buy and sell)
        buy_trades = trades[trades['Signal'] == 1]
        sell_trades = trades[trades['Signal'] == -1]
        
        # If the last trade is a buy with no corresponding sell, exclude it
        if len(buy_trades) > len(sell_trades):
            buy_trades = buy_trades.iloc[:-1]
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = float(buy_trades.iloc[i]['Trade_Price'].item())
            sell_price = float(sell_trades.iloc[i]['Trade_Price'].item())
            if sell_price > buy_price:
                winning_trades += 1
            else:
                losing_trades += 1
        
        win_rate = (winning_trades / (winning_trades + losing_trades)) * 100 if (winning_trades + losing_trades) > 0 else 0
        
        print(f"Number of Completed Trades: {winning_trades + losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
    
    return backtest_df, portfolio_value

if __name__ == "__main__":
    # Define the ticker symbol
    ticker_symbol = "RELIANCE.NS"
    
    # Download historical data
    print(f"Downloading data for {ticker_symbol}...")
    data = yf.download(ticker_symbol, period="1y", interval="1d")
    
    # Generate trading signals
    signals_df = generate_sma_signals(data)
    
    # Backtest the strategy
    backtest_results, final_value = backtest_strategy(signals_df)
    
    # Print the trade details
    trade_days = backtest_results[backtest_results['Trade_Price'].notna()]
    print("\nTrade Details:")
    print(trade_days[['Close', 'Signal', 'Position', 'Trade_Price', 'Shares', 'Cash', 'Portfolio_Value']])
    
    # Print daily portfolio values for the last 10 days
    print("\nPortfolio Value (Last 10 Days):")
    print(backtest_results[['Close', 'Position', 'Shares', 'Cash', 'Portfolio_Value']].tail(10))
