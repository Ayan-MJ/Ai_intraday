import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

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

def generate_momentum_signals(data, lookback_period=14, buy_threshold=0.02, sell_threshold=-0.02):
    """
    Generate trading signals based on momentum strategy.
    
    Args:
        data (pd.DataFrame): DataFrame containing stock price data
        lookback_period (int): Period over which to calculate momentum
        buy_threshold (float): Threshold above which to generate buy signals
        sell_threshold (float): Threshold below which to generate sell signals
        
    Returns:
        pd.DataFrame: DataFrame with added signal column
    """
    # Calculate momentum as percentage change over lookback period
    data['Momentum'] = data['Close'].pct_change(periods=lookback_period)
    
    # Initialize the Signal column to 0
    data['Signal'] = 0
    
    # Generate signals based on momentum thresholds
    for i in range(lookback_period + 1, len(data)):
        # Only generate a new signal if we're not already in a position
        current_position = 1 if data.iloc[i-1]['Signal'] == 1 else 0
        
        # Buy signal: Momentum above buy threshold and not already in a position
        if data.iloc[i]['Momentum'] > buy_threshold and current_position == 0:
            data.loc[data.index[i], 'Signal'] = 1
        
        # Sell signal: Momentum below sell threshold and already in a position
        elif data.iloc[i]['Momentum'] < sell_threshold and current_position == 1:
            data.loc[data.index[i], 'Signal'] = -1
    
    return data

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

def backtest_strategy(data, initial_capital=100000, long_window=50, risk_per_trade=1.0, slippage_factor=0.001, transaction_cost=20):
    """
    Backtest a trading strategy based on signals in the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Close' prices and 'Signal' column
        initial_capital (float): Initial capital to start with
        long_window (int): Long SMA window period, used to determine starting point
        risk_per_trade (float): Percentage of capital to risk per trade
        slippage_factor (float): Price slippage as a decimal (e.g., 0.001 for 0.1%)
        transaction_cost (float): Fixed cost per trade
        
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
        
        # Buy signal and not holding a position
        if signal == 1 and position == 0:
            position = 1
            
            # Apply slippage to buy price (price is higher when buying)
            buy_price_with_slippage = close_price * (1 + slippage_factor)
            entry_price = buy_price_with_slippage
            
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
        
        # Sell signal and holding a position
        elif signal == -1 and position == 1 and shares > 0:
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
            shares = 0
        
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

# Set page config
st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon="📈",
    layout="wide"
)

# Define a list of popular Indian stocks
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

# Sidebar for strategy selection and parameters
with st.sidebar:
    st.header("Trading Strategy Settings")
    
    # Stock selection
    stock_name = st.selectbox("Select Stock", list(indian_stocks.keys()))
    ticker_symbol = indian_stocks[stock_name]
    
    # Time period selection
    time_period = st.selectbox("Select Time Period", ["1y", "2y", "5y", "max"], index=0)
    
    # Strategy selection
    strategy_type = st.selectbox("Select Strategy", ["SMA Crossover", "Momentum"])
    
    # Strategy explanation
    with st.expander("Strategy Explanation"):
        if strategy_type == "SMA Crossover":
            st.markdown("""
            ### SMA Crossover Strategy
            
            **What it is:** This strategy uses two moving averages - a short-term one (faster) and a long-term one (slower).
            
            **How it works:**
            - **Buy Signal:** When the short-term average crosses *above* the long-term average
            - **Sell Signal:** When the short-term average crosses *below* the long-term average
            
            **Why it works:** The idea is that when shorter-term prices move above the longer-term average, it indicates upward momentum.
            
            **Best for:** Markets showing clear trends rather than sideways/choppy markets.
            """)
        else:  # Momentum strategy
            st.markdown("""
            ### Momentum Strategy
            
            **What it is:** This strategy buys stocks that have been rising and sells when they start falling.
            
            **How it works:**
            - **Buy Signal:** When price increases by a certain percentage over a set period
            - **Sell Signal:** When price decreases by a certain percentage over a set period
            
            **Why it works:** Stocks that have been rising tend to continue rising in the short term (price momentum).
            
            **Best for:** Strong trending markets where price movements persist for extended periods.
            """)
    
    # Capital and risk settings
    st.subheader("Capital & Risk Settings")
    initial_capital = st.number_input("Initial Capital (₹)", min_value=10000, max_value=10000000, value=100000, step=10000)
    risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    
    # Trading costs settings
    st.subheader("Trading Costs")
    slippage_factor = st.slider("Slippage (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100
    transaction_cost = st.number_input("Transaction Cost per Trade (₹)", min_value=0, max_value=100, value=20, step=5)
    
    # Strategy-specific parameters
    st.subheader("Strategy Parameters")
    
    if strategy_type == "SMA Crossover":
        sma_short = st.slider("Short SMA Period", min_value=5, max_value=50, value=20, step=1)
        sma_long = st.slider("Long SMA Period", min_value=20, max_value=200, value=50, step=5)
    else:  # Momentum strategy
        lookback_period = st.slider("Lookback Period (days)", min_value=1, max_value=60, value=14, step=1)
        buy_threshold = st.slider("Buy Threshold (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        sell_threshold = st.slider("Sell Threshold (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

# Main app
st.title("📊 Trading Strategy Backtester")

# Load data
with st.spinner('Downloading market data...'):
    # Use the selected ticker symbol
    st.info(f"Downloading data for {stock_name} ({ticker_symbol})...")
    data = yf.download(ticker_symbol, period=time_period, interval="1d")
    
    if data.empty:
        st.error(f"No data found for ticker {ticker_symbol}. Please check the ticker symbol and try again.")
        st.stop()
    
    # Generate trading signals based on selected strategy
    if strategy_type == "SMA Crossover":
        signals_df = generate_sma_signals(data, short_window=sma_short, long_window=sma_long)
        strategy_window = sma_long  # For backtesting start point
    else:  # Momentum strategy
        signals_df = generate_momentum_signals(data, lookback_period=lookback_period, 
                                              buy_threshold=buy_threshold, sell_threshold=sell_threshold)
        strategy_window = lookback_period  # For backtesting start point
    
    # Backtest the strategy
    backtest_results, final_value, strategy_return, buy_and_hold_return, total_trades, winning_trades, win_rate, max_drawdown, avg_profit_per_trade, profit_factor, annualized_return = backtest_strategy(
        signals_df, initial_capital, long_window=strategy_window, risk_per_trade=risk_per_trade,
        slippage_factor=slippage_factor, transaction_cost=transaction_cost
    )
    
    # Display metrics in two rows
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Initial Capital", f"₹{initial_capital:,.2f}")
    
    with col2:
        st.metric("Final Portfolio Value", f"₹{final_value:,.2f}")
    
    with col3:
        st.metric("Strategy Return", f"{strategy_return:.2f}%", 
                 delta=f"{strategy_return - buy_and_hold_return:.2f}% vs Buy & Hold")
    
    with col4:
        st.metric("Buy & Hold Return", f"{buy_and_hold_return:.2f}%")
    
    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", f"{total_trades}")
    
    with col2:
        st.metric("Winning Trades", f"{winning_trades}/{total_trades}")
    
    with col3:
        st.metric("Win Rate", f"{win_rate:.2f}%")
    
    with col4:
        st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
    
    # Third row of metrics - Advanced performance metrics
    st.subheader("Advanced Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg. Profit per Trade", f"₹{avg_profit_per_trade:.2f}")
    
    with col2:
        st.metric("Profit Factor", f"{profit_factor:.2f}")
    
    with col3:
        st.metric("Annualized Return", f"{annualized_return:.2f}%")
    
    # Explanation of metrics
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
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Strategy Signals", "Portfolio Performance", "Trade Details"])

    with tab1:
        st.subheader("Strategy Signals Visualization")
        
        # Create a figure for the strategy visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the closing price
        ax.plot(backtest_results.index, backtest_results['Close'], label='Close Price', alpha=0.5)
        
        # Plot the strategy-specific indicators
        if strategy_type == "SMA Crossover":
            ax.plot(backtest_results.index, backtest_results['SMA_short'], label=f'{sma_short}-day SMA', alpha=0.8)
            ax.plot(backtest_results.index, backtest_results['SMA_long'], label=f'{sma_long}-day SMA', alpha=0.8)
        else:  # Momentum strategy
            # For momentum, we can plot the price with a different visualization
            pass
        
        # Highlight buy and sell signals
        buy_signals = backtest_results[backtest_results['Signal'] == 1]
        sell_signals = backtest_results[backtest_results['Signal'] == -1]
        
        # Plot buy signals
        ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
        
        # Plot sell signals
        ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
        
        # Add annotations for buy and sell signals
        for idx, row in buy_signals.iterrows():
            ax.annotate('Buy', 
                       (idx, row['Close']),
                       xytext=(0, 15),
                       textcoords='offset points',
                       fontsize=8,
                       color='green',
                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
        
        for idx, row in sell_signals.iterrows():
            ax.annotate('Sell', 
                       (idx, row['Close']),
                       xytext=(0, -15),
                       textcoords='offset points',
                       fontsize=8,
                       color='red',
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        # Add labels and title
        ax.set_title(f'{strategy_type} Strategy on {stock_name} ({ticker_symbol})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₹)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format the y-axis as currency
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('₹{x:,.2f}'))
        
        # Show the plot
        st.pyplot(fig)
        
        # Add a performance summary
        st.subheader("Performance Summary")
        
        # Create a summary based on the backtesting results
        summary_text = f"""
        Based on this backtest of the **{strategy_type}** strategy on **{stock_name}** over the selected period:
        
        - Starting with ₹{initial_capital:,.2f}, the strategy resulted in a final portfolio value of **₹{final_value:,.2f}** 
          ({strategy_return:.2f}% return).
        - The strategy executed **{total_trades}** trades with a win rate of **{win_rate:.2f}%**.
        - The average profit per trade was **₹{avg_profit_per_trade:.2f}**.
        - The maximum drawdown was **{max_drawdown:.2f}%**.
        
        **Compared to Buy & Hold:** The strategy {' outperformed' if strategy_return > buy_and_hold_return else ' underperformed'} 
        a simple buy-and-hold approach by **{abs(strategy_return - buy_and_hold_return):.2f}%**.
        """
        
        st.markdown(summary_text)

    with tab2:
        # Create a figure for the portfolio performance
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the portfolio value
        ax.plot(backtest_results.index, backtest_results['Portfolio_Value'], label='Portfolio Value', color='blue')
        
        # Calculate buy and hold portfolio value
        start_idx = backtest_results.index[strategy_window] if len(backtest_results) > strategy_window else backtest_results.index[0]
        if start_idx:
            start_price = backtest_results.loc[start_idx, 'Close'].item()
            shares_held = initial_capital / start_price
            buy_hold_values = backtest_results['Close'] * shares_held
            ax.plot(backtest_results.index, buy_hold_values, label='Buy & Hold', color='gray', linestyle='--')
        
        # Plot drawdown
        drawdown = (backtest_results['Portfolio_Value'] - backtest_results['Portfolio_Value'].cummax()) / backtest_results['Portfolio_Value'].cummax() * 100
        ax2 = ax.twinx()
        ax2.fill_between(backtest_results.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(min(drawdown.min() * 1.5, -5), 5)  # Set y-axis limits for drawdown
        
        # Format the x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Add labels and title
        plt.title(f'Portfolio Performance for {stock_name}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (₹)')
        ax.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Add a chart showing daily returns
        st.write("### Daily Returns")
        backtest_results['Daily_Return'] = backtest_results['Portfolio_Value'].pct_change() * 100
        
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.bar(backtest_results.index, backtest_results['Daily_Return'], color='blue', alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Daily Portfolio Returns (%)')
        ax2.set_ylabel('Return (%)')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig2)
    
    with tab3:
        # Display trade details
        st.write("### Trade Details")
        trade_days = backtest_results[backtest_results['Trade_Price'].notna()]
        
        if not trade_days.empty:
            # Create a more user-friendly trade table
            trade_table = pd.DataFrame({
                'Date': trade_days.index,
                'Action': ['Buy' if s == 1 else 'Sell' for s in trade_days['Signal']],
                'Price': trade_days['Trade_Price'],
                'Shares': trade_days['Shares'],
                'Trade Value': trade_days['Trade_Price'] * trade_days['Shares'],
                'Cash After Trade': trade_days['Cash'],
                'Transaction Cost': transaction_cost,
                'Portfolio Value': trade_days['Portfolio_Value']
            })
            
            st.dataframe(trade_table, use_container_width=True)
            
            # Display total transaction costs and slippage impact
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Transaction Costs", f"₹{backtest_results['Transaction_Costs'].iloc[-1]:.2f}")
            with col2:
                st.metric("Total Slippage Impact", f"₹{backtest_results['Slippage_Impact'].sum():.2f}")
            
            # Calculate trade statistics
            buy_trades = trade_days[trade_days['Signal'] == 1]
            sell_trades = trade_days[trade_days['Signal'] == -1]
            
            # If the last trade is a buy with no corresponding sell, exclude it
            if len(buy_trades) > len(sell_trades):
                buy_trades = buy_trades.iloc[:-1]
            
            if not buy_trades.empty and not sell_trades.empty:
                # Calculate trade pairs and profits
                trade_pairs = min(len(buy_trades), len(sell_trades))
                profits = []
                
                for i in range(trade_pairs):
                    buy_price = buy_trades.iloc[i]['Trade_Price'].item()
                    buy_shares = buy_trades.iloc[i]['Shares'].item()
                    sell_price = sell_trades.iloc[i]['Trade_Price'].item()
                    profit = (sell_price - buy_price) * buy_shares - (2 * transaction_cost)  # Include transaction costs
                    profit_pct = ((sell_price / buy_price) - 1) * 100 - ((2 * transaction_cost) / (buy_price * buy_shares) * 100)  # Include transaction costs in percentage
                    profits.append({
                        'Buy Date': buy_trades.index[i],
                        'Buy Price': buy_price,
                        'Sell Date': sell_trades.index[i],
                        'Sell Price': sell_price,
                        'Shares': buy_shares,
                        'Profit/Loss (₹)': profit,
                        'Return (%)': profit_pct
                    })
                
                if profits:
                    profits_df = pd.DataFrame(profits)
                    st.write("### Trade Pair Analysis")
                    st.dataframe(profits_df, use_container_width=True)
                    
                    # Summary statistics
                    winning_trades_analysis = sum(1 for p in profits if p['Profit/Loss (₹)'] > 0)
                    total_trades_analysis = len(profits)
                    win_rate_analysis = (winning_trades_analysis / total_trades_analysis) * 100 if total_trades_analysis > 0 else 0
                    
                    st.write(f"**Win Rate**: {win_rate_analysis:.2f}% ({winning_trades_analysis}/{total_trades_analysis} trades)")
                    
                    if winning_trades_analysis > 0:
                        avg_win = sum(p['Profit/Loss (₹)'] for p in profits if p['Profit/Loss (₹)'] > 0) / winning_trades_analysis
                        st.write(f"**Average Win**: ₹{avg_win:.2f}")
                    
                    if total_trades_analysis - winning_trades_analysis > 0:
                        avg_loss = sum(p['Profit/Loss (₹)'] for p in profits if p['Profit/Loss (₹)'] <= 0) / (total_trades_analysis - winning_trades_analysis)
                        st.write(f"**Average Loss**: ₹{avg_loss:.2f}")
        else:
            st.info("No trades were executed during this period.")

# Add footer
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This app demonstrates different trading strategies and allows you to backtest them on historical stock data.
It helps you understand how different parameters affect trading performance and provides insights into strategy effectiveness.

**Disclaimer**: This is for educational purposes only and should not be considered financial advice.
""")
