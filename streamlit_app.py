import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def generate_sma_signals(data, short_window=20, long_window=50):
    """
    Generate trading signals based on SMA crossover strategy.
    
    Args:
        data (pd.DataFrame): DataFrame containing stock price data
        short_window (int): Lookback period for short-term SMA
        long_window (int): Lookback period for long-term SMA
        
    Returns:
        pd.DataFrame: DataFrame with added signal column
    """
    # Calculate the short-term and long-term Simple Moving Averages
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    
    # Initialize the Signal column to 0
    data['Signal'] = 0
    
    # Iterate through the DataFrame starting from the long_window day
    for i in range(long_window, len(data)):
        # Buy signal: SMA_short crosses above SMA_long
        if (data['SMA_short'].iloc[i-1] <= data['SMA_long'].iloc[i-1]) and (data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i]):
            data.loc[data.index[i], 'Signal'] = 1
        
        # Sell signal: SMA_short crosses below SMA_long
        elif (data['SMA_short'].iloc[i-1] >= data['SMA_long'].iloc[i-1]) and (data['SMA_short'].iloc[i] < data['SMA_long'].iloc[i]):
            data.loc[data.index[i], 'Signal'] = -1
    
    return data

def generate_momentum_signals(data, lookback_period=14, buy_threshold=2.0, sell_threshold=2.0):
    """
    Generate trading signals based on price momentum.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Close' prices
        lookback_period (int): Period over which to calculate momentum
        buy_threshold (float): Percentage increase to trigger buy signal
        sell_threshold (float): Percentage decrease to trigger sell signal
        
    Returns:
        pd.DataFrame: DataFrame with 'Signal' column
    """
    df = data.copy()
    
    # Calculate price momentum (percentage change over lookback period)
    df['Momentum'] = df['Close'].pct_change(periods=lookback_period) * 100
    
    # Initialize signal column
    df['Signal'] = 0
    
    # Convert thresholds to float to ensure proper comparison
    buy_threshold = float(buy_threshold)
    sell_threshold = float(sell_threshold)
    
    # Generate buy signals when momentum exceeds buy threshold
    buy_condition = df['Momentum'] > buy_threshold
    df.loc[buy_condition, 'Signal'] = 1
    
    # Generate sell signals when momentum falls below sell threshold
    sell_condition = df['Momentum'] < -sell_threshold
    df.loc[sell_condition, 'Signal'] = -1
    
    # Ensure we don't have consecutive buy or sell signals
    # Only keep the first signal in a series of same signals
    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == df['Signal'].iloc[i-1] and df['Signal'].iloc[i] != 0:
            df.loc[df.index[i], 'Signal'] = 0
    
    return df

def generate_macd_signals(data, short_window=12, long_window=26, signal_window=9):
    """
    Generate trading signals based on MACD (Moving Average Convergence Divergence) crossovers.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Close' prices
        short_window (int): Short-term EMA period (default: 12)
        long_window (int): Long-term EMA period (default: 26)
        signal_window (int): Signal line EMA period (default: 9)
        
    Returns:
        pd.DataFrame: DataFrame with 'Signal', 'MACD', and 'MACD_Signal' columns
    """
    df = data.copy()
    
    # Calculate the short-term and long-term EMAs
    df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    
    # Calculate MACD line and signal line
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Initialize the signal column
    df['Signal'] = 0
    
    # Generate signals based on MACD and signal line crossovers
    for i in range(long_window + signal_window, len(df)):
        # Buy signal: MACD crosses above signal line
        if (df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]) and (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]):
            df.loc[df.index[i], 'Signal'] = 1
        # Sell signal: MACD crosses below signal line
        elif (df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]) and (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]):
            df.loc[df.index[i], 'Signal'] = -1
    
    return df

def calculate_drawdown(portfolio_values):
    """
    Calculate the maximum drawdown from a series of portfolio values.
    
    Args:
        portfolio_values (pd.Series): Series of portfolio values over time
        
    Returns:
        float: Maximum drawdown as a percentage
    """
    # Calculate running maximum
    running_max = portfolio_values.cummax()
    
    # Calculate drawdown in percentage terms
    drawdown = (portfolio_values - running_max) / running_max * 100
    
    # Find the maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown

def backtest_strategy(data, initial_capital=100000, long_window=50, risk_per_trade=1.0, slippage_factor=0.001, 
                     transaction_cost=20, stop_loss_pct=0, take_profit_pct=0):
    """
    Backtest a trading strategy based on signals in the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Close' prices and 'Signal' column
        initial_capital (float): Initial capital to start with
        long_window (int): Long SMA window period, used to determine starting point
        risk_per_trade (float): Percentage of capital to risk per trade
        slippage_factor (float): Price slippage as a decimal (e.g., 0.001 for 0.1%)
        transaction_cost (float): Fixed cost per trade
        stop_loss_pct (float): Stop-loss percentage (0 means no stop-loss)
        take_profit_pct (float): Take-profit percentage (0 means no take-profit)
        
    Returns:
        tuple: (DataFrame with backtesting results, final portfolio value, strategy return, 
                buy and hold return, total trades, winning trades, win rate, max drawdown,
                avg_profit_per_trade, profit_factor, annualized_return)
    """
    # Create a copy of the data to avoid modifying the original
    backtest_df = data.copy()
    
    # Initialize variables
    position = 0  # 0 = no position, 1 = holding
    entry_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0
    portfolio_value = float(initial_capital)
    cash = float(initial_capital)
    shares = 0
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_transaction_costs = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    trade_profits = []  # Track individual trade profits/losses
    
    # Create new columns for tracking with appropriate dtypes
    backtest_df['Position'] = 0
    backtest_df['Trade_Price'] = np.nan
    backtest_df['Shares'] = 0
    backtest_df['Cash'] = float(initial_capital)
    backtest_df['Portfolio_Value'] = float(initial_capital)
    backtest_df['Transaction_Costs'] = 0.0
    backtest_df['Slippage_Impact'] = 0.0
    backtest_df['Stop_Loss_Price'] = np.nan
    backtest_df['Take_Profit_Price'] = np.nan
    backtest_df['Trade_Type'] = ""  # To track if a trade was from a signal or stop-loss/take-profit
    
    # Find the first valid index (after signals can be generated)
    start_idx = backtest_df.index[long_window] if len(backtest_df) > long_window else None
    if start_idx is None:
        return backtest_df, portfolio_value, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    # Store the start date for calculating annualized return
    start_date = backtest_df.index[backtest_df.index.get_loc(start_idx)]
    
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
            backtest_df.loc[current_idx, 'Transaction_Costs'] = float(backtest_df.loc[prev_idx, 'Transaction_Costs'].item())
            backtest_df.loc[current_idx, 'Stop_Loss_Price'] = backtest_df.loc[prev_idx, 'Stop_Loss_Price']
            backtest_df.loc[current_idx, 'Take_Profit_Price'] = backtest_df.loc[prev_idx, 'Take_Profit_Price']
        
        # Buy signal and not holding a position
        if signal == 1 and position == 0:
            position = 1
            
            # Apply slippage to buy price (price is higher when buying)
            buy_price_with_slippage = close_price * (1 + slippage_factor)
            entry_price = buy_price_with_slippage
            
            # Set stop-loss price if stop-loss is enabled
            if stop_loss_pct > 0:
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            else:
                stop_loss_price = 0.0
                
            # Set take-profit price if take-profit is enabled
            if take_profit_pct > 0:
                take_profit_price = entry_price * (1 + take_profit_pct / 100)
            else:
                take_profit_price = float('inf')  # Set to infinity if not used
            
            # Calculate investment amount based on risk percentage
            investment_amount = cash * (risk_per_trade / 100)
            
            # Apply transaction cost
            cash -= transaction_cost
            total_transaction_costs += transaction_cost
            
            # Calculate shares to buy (after accounting for slippage and transaction costs)
            shares = int(investment_amount / buy_price_with_slippage)
            
            if shares > 0:  # Only proceed if we can buy at least one share
                # Calculate actual cost with slippage
                actual_cost = shares * buy_price_with_slippage
                slippage_impact = shares * (buy_price_with_slippage - close_price)
                
                cash = float(cash - actual_cost)
                total_trades += 1
                
                # Update the DataFrame
                backtest_df.loc[current_idx, 'Position'] = position
                backtest_df.loc[current_idx, 'Trade_Price'] = entry_price
                backtest_df.loc[current_idx, 'Shares'] = shares
                backtest_df.loc[current_idx, 'Cash'] = cash
                backtest_df.loc[current_idx, 'Transaction_Costs'] = total_transaction_costs
                backtest_df.loc[current_idx, 'Slippage_Impact'] = slippage_impact
                backtest_df.loc[current_idx, 'Stop_Loss_Price'] = stop_loss_price
                backtest_df.loc[current_idx, 'Take_Profit_Price'] = take_profit_price
                backtest_df.loc[current_idx, 'Trade_Type'] = "Buy Signal"
        
        # Sell signal, stop-loss, or take-profit and holding a position
        elif (signal == -1 or 
              (stop_loss_pct > 0 and close_price <= stop_loss_price) or
              (take_profit_pct > 0 and close_price >= take_profit_price)) and position == 1 and shares > 0:
            position = 0
            
            # Apply slippage to sell price (price is lower when selling)
            sell_price_with_slippage = close_price * (1 - slippage_factor)
            exit_price = sell_price_with_slippage
            
            # Apply transaction cost
            cash -= transaction_cost
            total_transaction_costs += transaction_cost
            
            # Calculate actual proceeds with slippage
            actual_proceeds = shares * sell_price_with_slippage
            slippage_impact = shares * (close_price - sell_price_with_slippage)
            
            cash = float(cash + actual_proceeds)
            
            # Determine if this was a winning trade (accounting for slippage and transaction costs)
            trade_profit = (exit_price - entry_price) * shares - (2 * transaction_cost)  # Both buy and sell costs
            trade_profits.append(trade_profit)
            
            if trade_profit > 0:
                winning_trades += 1
                gross_profit += trade_profit
            else:
                losing_trades += 1
                gross_loss += abs(trade_profit)
            
            # Update the DataFrame
            backtest_df.loc[current_idx, 'Position'] = position
            backtest_df.loc[current_idx, 'Trade_Price'] = exit_price
            backtest_df.loc[current_idx, 'Shares'] = 0
            backtest_df.loc[current_idx, 'Cash'] = cash
            backtest_df.loc[current_idx, 'Transaction_Costs'] = total_transaction_costs
            backtest_df.loc[current_idx, 'Slippage_Impact'] += slippage_impact
            backtest_df.loc[current_idx, 'Stop_Loss_Price'] = np.nan
            backtest_df.loc[current_idx, 'Take_Profit_Price'] = np.nan
            
            # Record if this was a stop-loss, take-profit, or a signal-based sell
            if signal == -1:
                backtest_df.loc[current_idx, 'Trade_Type'] = "Sell Signal"
            elif close_price <= stop_loss_price:
                backtest_df.loc[current_idx, 'Trade_Type'] = "Stop-Loss"
            elif close_price >= take_profit_price:
                backtest_df.loc[current_idx, 'Trade_Type'] = "Take-Profit"
            
            shares = 0
            stop_loss_price = 0.0
            take_profit_price = float('inf')
        
        # Calculate portfolio value (cash + value of holdings)
        portfolio_value = float(cash + (shares * close_price))
        backtest_df.loc[current_idx, 'Portfolio_Value'] = portfolio_value
    
    # Store the end date for calculating annualized return
    end_date = backtest_df.index[-1]
    
    # Calculate performance metrics
    initial_close = float(backtest_df['Close'].iloc[backtest_df.index.get_loc(start_idx)].item())
    final_close = float(backtest_df['Close'].iloc[-1].item())
    buy_and_hold_return = (final_close / initial_close - 1) * 100
    strategy_return = (portfolio_value / initial_capital - 1) * 100
    
    # Calculate win rate
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate maximum drawdown
    max_drawdown = calculate_drawdown(backtest_df['Portfolio_Value'])
    
    # Calculate average profit per trade
    avg_profit_per_trade = sum(trade_profits) / total_trades if total_trades > 0 else 0
    
    # Calculate profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # Calculate annualized return
    total_return = portfolio_value / initial_capital - 1
    days_in_backtest = (end_date - start_date).days
    years_in_backtest = days_in_backtest / 365
    annualized_return = ((1 + total_return) ** (1 / years_in_backtest) - 1) * 100 if years_in_backtest > 0 else 0
    
    return (backtest_df, portfolio_value, strategy_return, buy_and_hold_return, 
            total_trades, winning_trades, win_rate, max_drawdown,
            avg_profit_per_trade, profit_factor, annualized_return)

def download_stock_data(ticker_symbol, time_period):
    import yfinance as yf
    return yf.download(ticker_symbol, period=time_period, interval="1d")

def plot_results(backtest_results, selected_strategy, strategy_params, stock_name, ticker_symbol):
    """
    Plot the results of the backtest using Matplotlib for reliable visualization.
    
    Args:
        backtest_results (pd.DataFrame): DataFrame with backtest results
        selected_strategy (str): Name of the strategy
        strategy_params (dict): Dictionary of strategy parameters
        stock_name (str): Name of the stock
        ticker_symbol (str): Ticker symbol of the stock
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import streamlit as st
    
    if selected_strategy == "MACD":
        # Create figure with two subplots for MACD
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price chart on top subplot
        ax1.plot(backtest_results.index, backtest_results['Close'], label='Close Price', alpha=0.7)
        
        # Highlight buy and sell signals on price chart
        buy_signals = backtest_results[backtest_results['Signal'] == 1]
        sell_signals = backtest_results[backtest_results['Signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
        ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
        
        # Format price chart
        ax1.set_title(f'MACD Strategy on {stock_name} ({ticker_symbol})')
        ax1.set_ylabel('Price (₹)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('₹{x:,.2f}'))
        
        # Plot MACD indicators on bottom subplot
        ax2.plot(backtest_results.index, backtest_results['MACD'], label='MACD', color='blue', alpha=0.8)
        ax2.plot(backtest_results.index, backtest_results['MACD_Signal'], label='Signal Line', color='red', alpha=0.8)
        
        # Plot histogram as bar chart
        positive_hist = backtest_results['MACD_Histogram'].copy()
        negative_hist = backtest_results['MACD_Histogram'].copy()
        positive_hist[positive_hist <= 0] = 0
        negative_hist[negative_hist > 0] = 0
        
        ax2.bar(backtest_results.index, positive_hist, color='green', alpha=0.3, label='Positive Histogram')
        ax2.bar(backtest_results.index, negative_hist, color='red', alpha=0.3, label='Negative Histogram')
        
        # Add horizontal line at zero
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Format MACD chart
        ax2.set_xlabel('Date')
        ax2.set_ylabel('MACD')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    else:
        # Original plotting for other strategies
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(backtest_results.index, backtest_results['Close'], label='Close Price', alpha=0.7)
        
        if selected_strategy == "SMA Crossover":
            sma_short = strategy_params.get('sma_short', 20)
            sma_long = strategy_params.get('sma_long', 50)
            ax.plot(backtest_results.index, backtest_results['SMA_short'], label=f'{sma_short}-day SMA', alpha=0.8)
            ax.plot(backtest_results.index, backtest_results['SMA_long'], label=f'{sma_long}-day SMA', alpha=0.8)
        
        # Highlight buy and sell signals
        buy_signals = backtest_results[backtest_results['Signal'] == 1]
        sell_signals = backtest_results[backtest_results['Signal'] == -1]
        
        ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
        ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
        
        ax.set_title(f'{selected_strategy} Strategy on {stock_name} ({ticker_symbol})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₹)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('₹{x:,.2f}'))
    
    # Display the plot
    st.pyplot(fig)

def generate_signals(data, selected_strategy, params):
    if selected_strategy == "SMA Crossover":
        short = params.get('sma_short', 20)
        long = params.get('sma_long', 50)
        df = data.copy()
        df['SMA_short'] = df['Close'].rolling(window=short).mean()
        df['SMA_long'] = df['Close'].rolling(window=long).mean()
        df['Signal'] = 0
        for i in range(long, len(df)):
            if (df['SMA_short'].iloc[i-1] <= df['SMA_long'].iloc[i-1]) and (df['SMA_short'].iloc[i] > df['SMA_long'].iloc[i]):
                df.loc[df.index[i], 'Signal'] = 1
            elif (df['SMA_short'].iloc[i-1] >= df['SMA_long'].iloc[i-1]) and (df['SMA_short'].iloc[i] < df['SMA_long'].iloc[i]):
                df.loc[df.index[i], 'Signal'] = -1
        return df
    elif selected_strategy == "Momentum":
        lookback = params.get('lookback_period', 14)
        buy_th = params.get('buy_threshold', 2.0)
        sell_th = params.get('sell_threshold', 2.0)
        return generate_momentum_signals(data, lookback, buy_th, sell_th)
    elif selected_strategy == "MACD":
        short = params.get('macd_short', 12)
        long = params.get('macd_long', 26)
        signal = params.get('macd_signal', 9)
        return generate_macd_signals(data, short, long, signal)
    else:
        raise ValueError(f"Unknown strategy: {selected_strategy}")

if __name__ == "__main__":
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    
    # Set page config
    st.set_page_config(
        page_title="Trading Strategy Backtester",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📊 Trading Strategy Backtester")
    
    # Sidebar Inputs
    with st.sidebar:
        st.header("Trading Strategy Settings")
        indian_stocks = {
            "Reliance Industries": "RELIANCE.NS",
            "Tata Consultancy Services": "TCS.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "Infosys": "INFY.NS",
            "State Bank of India": "SBIN.NS",
            "ICICI Bank": "ICICIBANK.NS",
            "Hindustan Unilever": "HINDUNILVR.NS",
            "Bharti Airtel": "BHARTIARTL.NS",
            "ITC Limited": "ITC.NS",
            "Larsen & Toubro": "LT.NS",
            "Axis Bank": "AXISBANK.NS",
            "Kotak Mahindra Bank": "KOTAKBANK.NS",
            "Maruti Suzuki": "MARUTI.NS",
            "Asian Paints": "ASIANPAINT.NS",
            "Bajaj Finance": "BAJFINANCE.NS"
        }
        stock_name = st.selectbox("Select Stock", list(indian_stocks.keys()))
        ticker_symbol = indian_stocks[stock_name]
        time_period = st.selectbox("Select Time Period", ["1y", "2y", "5y", "max"], index=0)
        
        # Allow multiple strategy selection
        strategy_types = st.multiselect("Select Strategies", 
                                      ["SMA Crossover", "Momentum", "MACD"],
                                      default=["SMA Crossover"])
        
        if not strategy_types:
            st.error("Please select at least one strategy.")
            st.stop()
        
        with st.expander("Strategy Explanations"):
            st.markdown("""
            ### SMA Crossover Strategy
            **What it is:** This strategy uses two moving averages - a short-term one (faster) and a long-term one (slower).
            **How it works:**
            - **Buy Signal:** When the short-term average crosses *above* the long-term average
            - **Sell Signal:** When the short-term average crosses *below* the long-term average
            **Why it works:** The idea is that when shorter-term prices move above the longer-term average, it indicates upward momentum.
            **Best for:** Markets showing clear trends rather than sideways/choppy markets.
            
            ### Momentum Strategy
            **What it is:** This strategy buys stocks that have been rising and sells when they start falling.
            **How it works:**
            - **Buy Signal:** When price increases by a certain percentage over a set period
            - **Sell Signal:** When price decreases by a certain percentage over a set period
            **Why it works:** Stocks that have been rising tend to continue rising in the short term (price momentum).
            **Best for:** Strong trending markets where price movements persist for extended periods.
            
            ### MACD Strategy
            **What it is:** Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator.
            **How it works:**
            - **MACD Line:** Difference between fast and slow EMAs (typically 12 and 26 periods)
            - **Signal Line:** EMA of the MACD Line (typically 9 periods)
            - **Buy Signal:** When MACD Line crosses *above* the Signal Line
            - **Sell Signal:** When MACD Line crosses *below* the Signal Line
            **Why it works:** MACD helps identify changes in the strength, direction, momentum, and duration of a trend.
            **Best for:** Trending markets with clear momentum shifts.
            """)
        
        st.subheader("Capital & Risk Settings")
        initial_capital = st.number_input("Initial Capital (₹)", min_value=10000, max_value=10000000, value=100000, step=10000)
        risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        # Risk management inputs
        st.subheader("Risk Management")
        stop_loss_pct = st.number_input("Stop-Loss (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.1,
                                       help="Percentage below entry price to automatically sell and limit losses. Set to 0 to disable.")
        take_profit_pct = st.number_input("Take-Profit (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.1,
                                         help="Percentage above entry price to automatically sell and lock in profits. Set to 0 to disable.")
        
        st.subheader("Trading Costs")
        slippage_factor = st.slider("Slippage (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100
        transaction_cost = st.number_input("Transaction Cost per Trade (₹)", min_value=0, max_value=100, value=20, step=5)
        
        # Strategy-specific parameters
        st.subheader("Strategy Parameters")
        
        # SMA Crossover parameters
        if "SMA Crossover" in strategy_types:
            st.markdown("#### SMA Crossover Parameters")
            sma_short = st.slider("Short SMA Period", min_value=5, max_value=50, value=20, step=1)
            sma_long = st.slider("Long SMA Period", min_value=20, max_value=200, value=50, step=5)
        
        # Momentum parameters
        if "Momentum" in strategy_types:
            st.markdown("#### Momentum Parameters")
            lookback_period = st.slider("Lookback Period (days)", min_value=1, max_value=60, value=14, step=1)
            buy_threshold = st.slider("Buy Threshold (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            sell_threshold = st.slider("Sell Threshold (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        
        # MACD parameters
        if "MACD" in strategy_types:
            st.markdown("#### MACD Parameters")
            macd_short = st.number_input("MACD Fast EMA Period", min_value=5, max_value=30, value=12, step=1)
            macd_long = st.number_input("MACD Slow EMA Period", min_value=10, max_value=50, value=26, step=1)
            macd_signal = st.number_input("MACD Signal Line Period", min_value=3, max_value=20, value=9, step=1)

    # 1. Download data
    with st.spinner('Downloading market data...'):
        st.info(f"Downloading data for {stock_name} ({ticker_symbol})...")
        data = download_stock_data(ticker_symbol, time_period)
        if data.empty:
            st.error(f"No data found for ticker {ticker_symbol}. Please check the ticker symbol and try again.")
            st.stop()
    
    # 2. Generate signals
    if "SMA Crossover" in strategy_types:
        sma_params = {'sma_short': sma_short, 'sma_long': sma_long}
        sma_signals_df = generate_signals(data, "SMA Crossover", sma_params)
    if "Momentum" in strategy_types:
        momentum_params = {'lookback_period': lookback_period, 'buy_threshold': buy_threshold, 'sell_threshold': sell_threshold}
        momentum_signals_df = generate_signals(data, "Momentum", momentum_params)
    if "MACD" in strategy_types:
        macd_params = {'macd_short': macd_short, 'macd_long': macd_long, 'macd_signal': macd_signal}
        macd_signals_df = generate_signals(data, "MACD", macd_params)
    
    # Determine the appropriate window for backtesting start point
    if "SMA Crossover" in strategy_types:
        sma_window = max(sma_params['sma_short'], sma_params['sma_long'])
    if "Momentum" in strategy_types:
        momentum_window = momentum_params['lookback_period']
    if "MACD" in strategy_types:
        macd_window = max(macd_params['macd_long'], macd_params['macd_signal'])
    
    # 3. Run backtest
    if "SMA Crossover" in strategy_types:
        sma_backtest_results, sma_final_value, sma_strategy_return, sma_buy_and_hold_return, sma_total_trades, sma_winning_trades, sma_win_rate, sma_max_drawdown, sma_avg_profit_per_trade, sma_profit_factor, sma_annualized_return = backtest_strategy(
            sma_signals_df, initial_capital, long_window=sma_window, risk_per_trade=risk_per_trade,
            slippage_factor=slippage_factor, transaction_cost=transaction_cost, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct
        )
    if "Momentum" in strategy_types:
        momentum_backtest_results, momentum_final_value, momentum_strategy_return, momentum_buy_and_hold_return, momentum_total_trades, momentum_winning_trades, momentum_win_rate, momentum_max_drawdown, momentum_avg_profit_per_trade, momentum_profit_factor, momentum_annualized_return = backtest_strategy(
            momentum_signals_df, initial_capital, long_window=momentum_window, risk_per_trade=risk_per_trade,
            slippage_factor=slippage_factor, transaction_cost=transaction_cost, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct
        )
    if "MACD" in strategy_types:
        macd_backtest_results, macd_final_value, macd_strategy_return, macd_buy_and_hold_return, macd_total_trades, macd_winning_trades, macd_win_rate, macd_max_drawdown, macd_avg_profit_per_trade, macd_profit_factor, macd_annualized_return = backtest_strategy(
            macd_signals_df, initial_capital, long_window=macd_window, risk_per_trade=risk_per_trade,
            slippage_factor=slippage_factor, transaction_cost=transaction_cost, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct
        )
    
    # 4. Display results and plots
    tab1, tab2, tab3 = st.tabs(["Strategy Signals", "Portfolio Performance", "Trade Details"])
    with tab1:
        st.subheader("Strategy Signals Visualization")
        if "SMA Crossover" in strategy_types:
            plot_results(sma_backtest_results, "SMA Crossover", sma_params, stock_name, ticker_symbol)
        if "Momentum" in strategy_types:
            plot_results(momentum_backtest_results, "Momentum", momentum_params, stock_name, ticker_symbol)
        if "MACD" in strategy_types:
            plot_results(macd_backtest_results, "MACD", macd_params, stock_name, ticker_symbol)
        st.subheader("Performance Summary")
        if "SMA Crossover" in strategy_types:
            sma_summary_text = f"""
            Based on this backtest of the **SMA Crossover** strategy on **{stock_name}** over the selected period:
            - Starting with ₹{initial_capital:,.2f}, the strategy resulted in a final portfolio value of **₹{sma_final_value:,.2f}** 
              ({sma_strategy_return:.2f}% return).
            - The strategy executed **{sma_total_trades}** trades with a win rate of **{sma_win_rate:.2f}%**.
            - The average profit per trade was **₹{sma_avg_profit_per_trade:.2f}**.
            - The maximum drawdown was **{sma_max_drawdown:.2f}%**.
            **Compared to Buy & Hold:** The strategy {' outperformed' if sma_strategy_return > sma_buy_and_hold_return else ' underperformed'} 
            a simple buy-and-hold approach by **{abs(sma_strategy_return - sma_buy_and_hold_return):.2f}%**.
            """
            st.markdown(sma_summary_text)
        if "Momentum" in strategy_types:
            momentum_summary_text = f"""
            Based on this backtest of the **Momentum** strategy on **{stock_name}** over the selected period:
            - Starting with ₹{initial_capital:,.2f}, the strategy resulted in a final portfolio value of **₹{momentum_final_value:,.2f}** 
              ({momentum_strategy_return:.2f}% return).
            - The strategy executed **{momentum_total_trades}** trades with a win rate of **{momentum_win_rate:.2f}%**.
            - The average profit per trade was **₹{momentum_avg_profit_per_trade:.2f}**.
            - The maximum drawdown was **{momentum_max_drawdown:.2f}%**.
            **Compared to Buy & Hold:** The strategy {' outperformed' if momentum_strategy_return > momentum_buy_and_hold_return else ' underperformed'} 
            a simple buy-and-hold approach by **{abs(momentum_strategy_return - momentum_buy_and_hold_return):.2f}%**.
            """
            st.markdown(momentum_summary_text)
        if "MACD" in strategy_types:
            macd_summary_text = f"""
            Based on this backtest of the **MACD** strategy on **{stock_name}** over the selected period:
            - Starting with ₹{initial_capital:,.2f}, the strategy resulted in a final portfolio value of **₹{macd_final_value:,.2f}** 
              ({macd_strategy_return:.2f}% return).
            - The strategy executed **{macd_total_trades}** trades with a win rate of **{macd_win_rate:.2f}%**.
            - The average profit per trade was **₹{macd_avg_profit_per_trade:.2f}**.
            - The maximum drawdown was **{macd_max_drawdown:.2f}%**.
            **Compared to Buy & Hold:** The strategy {' outperformed' if macd_strategy_return > macd_buy_and_hold_return else ' underperformed'} 
            a simple buy-and-hold approach by **{abs(macd_strategy_return - macd_buy_and_hold_return):.2f}%**.
            """
            st.markdown(macd_summary_text)
    
    # (Portfolio and Trade Details tabs remain unchanged)
    
    # 5. Show explanations
    with st.expander("Understanding the Metrics"):
        st.markdown("""
        ## Understanding Trading Performance Metrics
        ### Basic Performance Metrics
        - **Final Portfolio Value**: The ending value of your portfolio after all trades.
          - *Good*: Higher than your initial capital
          - *Bad*: Lower than your initial capital
        - **Strategy Return**: The percentage gain or loss from your trading strategy.
          - *Good*: Positive returns, especially if higher than market benchmarks
          - *Bad*: Negative returns or returns lower than risk-free alternatives
        - **Buy & Hold Return**: What you would have earned by simply buying and holding.
          - *Use this*: As a benchmark to evaluate if your strategy adds value
        - **Total Trades**: The number of completed buy and sell transactions.
          - *Too few*: May indicate missed opportunities
          - *Too many*: May lead to excessive transaction costs
        - **Winning Trades**: The number of trades that resulted in a profit.
          - *Context*: Should be evaluated as a percentage (win rate)
        - **Win Rate**: The percentage of trades that were profitable.
          - *Good*: Above 50%, though some strategies work with lower win rates if profits per trade are high
          - *Bad*: Below 40% typically indicates a problematic strategy
        - **Maximum Drawdown**: The largest percentage drop from peak to trough.
          - *Good*: Less than 20% for most retail strategies
          - *Bad*: Over 30% may indicate excessive risk
        ### Advanced Performance Metrics
        - **Average Profit per Trade**: The average amount of profit (or loss) per trade.
          - *Good*: Positive value, higher is better
          - *Bad*: Negative value indicates losing money on average
        - **Profit Factor**: The ratio of gross profits to gross losses.
          - *Good*: Above 1.5 indicates a robust strategy
          - *Excellent*: Above 2.0 is considered very strong
          - *Bad*: Below 1.0 means you're losing money
        - **Annualized Return**: The strategy's return normalized to a one-year period.
          - *Good*: Above market benchmarks (typically 8-10% for equity markets)
          - *Excellent*: Above 15% consistently
          - *Bad*: Below risk-free rate (typically 3-5%)
        ### Trading Costs and Slippage
        - **Slippage**: The difference between expected and actual execution prices.
          - *Impact*: Higher for less liquid stocks and during volatile markets
        - **Transaction Costs**: Brokerage fees and other trading costs.
          - *Impact*: Can significantly reduce profitability, especially for frequent traders
        ### Interpreting Results
        A good trading strategy should:
        1. Generate positive returns above market benchmarks
        2. Have a reasonable win rate (typically above 50%)
        3. Keep drawdowns manageable
        4. Maintain a profit factor above 1.5
        5. Produce consistent returns when adjusted for time (annualized return)
        Remember that past performance doesn't guarantee future results. Always test strategies across different market conditions.
        """)

    with tab3:
        # Display trade details
        st.write("### Trade Details")
        if "SMA Crossover" in strategy_types:
            sma_trade_days = sma_backtest_results[sma_backtest_results['Trade_Price'].notna()]
        if "Momentum" in strategy_types:
            momentum_trade_days = momentum_backtest_results[momentum_backtest_results['Trade_Price'].notna()]
        if "MACD" in strategy_types:
            macd_trade_days = macd_backtest_results[macd_backtest_results['Trade_Price'].notna()]
        
        if "SMA Crossover" in strategy_types and not sma_trade_days.empty:
            # Create a more user-friendly trade table
            sma_trade_table = pd.DataFrame({
                'Date': sma_trade_days.index,
                'Action': ['Buy' if s == 1 else 'Sell' for s in sma_trade_days['Signal']],
                'Type': sma_trade_days['Trade_Type'],
                'Price': sma_trade_days['Trade_Price'],
                'Shares': sma_trade_days['Shares'],
                'Trade Value': sma_trade_days['Trade_Price'] * sma_trade_days['Shares'],
                'Cash After Trade': sma_trade_days['Cash'],
                'Transaction Cost': transaction_cost,
                'Portfolio Value': sma_trade_days['Portfolio_Value']
            })
            
            st.dataframe(sma_trade_table, use_container_width=True)
            
            # Count stop-loss trades
            sma_stop_loss_trades = len(sma_trade_days[sma_trade_days['Trade_Type'] == "Stop-Loss"])
            if stop_loss_pct > 0:
                st.info(f"Number of trades triggered by stop-loss: {sma_stop_loss_trades}")
            
            # Display total transaction costs and slippage impact
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Transaction Costs", f"₹{sma_backtest_results['Transaction_Costs'].iloc[-1]:.2f}")
            with col2:
                st.metric("Total Slippage Impact", f"₹{sma_backtest_results['Slippage_Impact'].sum():.2f}")
        
        if "Momentum" in strategy_types and not momentum_trade_days.empty:
            # Create a more user-friendly trade table
            momentum_trade_table = pd.DataFrame({
                'Date': momentum_trade_days.index,
                'Action': ['Buy' if s == 1 else 'Sell' for s in momentum_trade_days['Signal']],
                'Type': momentum_trade_days['Trade_Type'],
                'Price': momentum_trade_days['Trade_Price'],
                'Shares': momentum_trade_days['Shares'],
                'Trade Value': momentum_trade_days['Trade_Price'] * momentum_trade_days['Shares'],
                'Cash After Trade': momentum_trade_days['Cash'],
                'Transaction Cost': transaction_cost,
                'Portfolio Value': momentum_trade_days['Portfolio_Value']
            })
            
            st.dataframe(momentum_trade_table, use_container_width=True)
            
            # Count stop-loss trades
            momentum_stop_loss_trades = len(momentum_trade_days[momentum_trade_days['Trade_Type'] == "Stop-Loss"])
            if stop_loss_pct > 0:
                st.info(f"Number of trades triggered by stop-loss: {momentum_stop_loss_trades}")
            
            # Display total transaction costs and slippage impact
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Transaction Costs", f"₹{momentum_backtest_results['Transaction_Costs'].iloc[-1]:.2f}")
            with col2:
                st.metric("Total Slippage Impact", f"₹{momentum_backtest_results['Slippage_Impact'].sum():.2f}")
        
        if "MACD" in strategy_types and not macd_trade_days.empty:
            # Create a more user-friendly trade table
            macd_trade_table = pd.DataFrame({
                'Date': macd_trade_days.index,
                'Action': ['Buy' if s == 1 else 'Sell' for s in macd_trade_days['Signal']],
                'Type': macd_trade_days['Trade_Type'],
                'Price': macd_trade_days['Trade_Price'],
                'Shares': macd_trade_days['Shares'],
                'Trade Value': macd_trade_days['Trade_Price'] * macd_trade_days['Shares'],
                'Cash After Trade': macd_trade_days['Cash'],
                'Transaction Cost': transaction_cost,
                'Portfolio Value': macd_trade_days['Portfolio_Value']
            })
            
            st.dataframe(macd_trade_table, use_container_width=True)
            
            # Count stop-loss trades
            macd_stop_loss_trades = len(macd_trade_days[macd_trade_days['Trade_Type'] == "Stop-Loss"])
            if stop_loss_pct > 0:
                st.info(f"Number of trades triggered by stop-loss: {macd_stop_loss_trades}")
            
            # Display total transaction costs and slippage impact
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Transaction Costs", f"₹{macd_backtest_results['Transaction_Costs'].iloc[-1]:.2f}")
            with col2:
                st.metric("Total Slippage Impact", f"₹{macd_backtest_results['Slippage_Impact'].sum():.2f}")
