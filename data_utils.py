"""
Data utility functions for the Mag7 trading environment.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def download_mag7_data(period="1y", save_to_csv=False):
    """
    Download historical data for Magnificent 7 stocks.
    
    Args:
        period: Time period (e.g., "1y", "2y", "6mo", "1mo")
        save_to_csv: Whether to save data to CSV file
        
    Returns:
        DataFrame with closing prices for all stocks
    """
    mag7_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    print(f"Downloading {period} of data for Magnificent 7 stocks...")
    
    data = {}
    for ticker in mag7_tickers:
        try:
            print(f"  Downloading {ticker}...", end=" ")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            data[ticker] = hist['Close']
            print(f"✓ ({len(hist)} days)")
        except Exception as e:
            print(f"✗ Error: {e}")
            data[ticker] = None
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Remove any stocks that failed to download
    df = df.dropna(axis=1, how='all')
    
    # Forward fill any missing values
    df = df.fillna(method='ffill')
    
    print(f"\nSuccessfully downloaded data for {len(df.columns)} stocks")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total trading days: {len(df)}")
    
    if save_to_csv:
        filename = f"mag7_data_{period}.csv"
        df.to_csv(filename)
        print(f"Data saved to {filename}")
    
    return df


def calculate_returns(prices_df):
    """Calculate daily returns from price data."""
    returns = prices_df.pct_change().dropna()
    return returns


def get_stock_info(ticker):
    """Get detailed information about a stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        print(f"\n{ticker} - {info.get('longName', 'N/A')}")
        print(f"{'='*60}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Industry: {info.get('industry', 'N/A')}")
        print(f"Market Cap: ${info.get('marketCap', 0):,.0f}")
        print(f"Current Price: ${info.get('currentPrice', 0):.2f}")
        print(f"52 Week High: ${info.get('fiftyTwoWeekHigh', 0):.2f}")
        print(f"52 Week Low: ${info.get('fiftyTwoWeekLow', 0):.2f}")
        print(f"Average Volume: {info.get('averageVolume', 0):,.0f}")
        
        return info
    except Exception as e:
        print(f"Error getting info for {ticker}: {e}")
        return None


def analyze_price_data(prices_df):
    """Analyze price data and print statistics."""
    print("\nPrice Data Analysis")
    print("="*80)
    
    # Calculate returns
    returns = calculate_returns(prices_df)
    
    # Statistics for each stock
    stats = []
    for ticker in prices_df.columns:
        stock_returns = returns[ticker]
        stats.append({
            'Ticker': ticker,
            'Start Price': prices_df[ticker].iloc[0],
            'End Price': prices_df[ticker].iloc[-1],
            'Total Return %': ((prices_df[ticker].iloc[-1] / prices_df[ticker].iloc[0]) - 1) * 100,
            'Mean Daily Return %': stock_returns.mean() * 100,
            'Std Dev %': stock_returns.std() * 100,
            'Sharpe Ratio': (stock_returns.mean() / stock_returns.std()) * np.sqrt(252) if stock_returns.std() > 0 else 0
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    
    print(f"\nOverall Statistics:")
    print(f"  Best Performer: {stats_df.loc[stats_df['Total Return %'].idxmax(), 'Ticker']}")
    print(f"  Worst Performer: {stats_df.loc[stats_df['Total Return %'].idxmin(), 'Ticker']}")
    print(f"  Highest Sharpe: {stats_df.loc[stats_df['Sharpe Ratio'].idxmax(), 'Ticker']}")
    
    return stats_df


if __name__ == "__main__":
    # Download data
    df = download_mag7_data(period="1y", save_to_csv=True)
    
    # Analyze data
    analyze_price_data(df)
    
    # Get info for each stock
    print("\n" + "="*80)
    print("Stock Information")
    print("="*80)
    for ticker in df.columns:
        get_stock_info(ticker)
