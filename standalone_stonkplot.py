#!/usr/bin/env python3
"""
✨ stonkplot.py - Elegant Stock Visualization Tool ✨

An elegant, lightweight stock plotting utility with AI-powered trend forecasting.
Perfect for quick market insights with minimal effort.

Usage:
    python stonkplot.py AAPL
    python stonkplot.py TSLA --forecast
    python stonkplot.py NVDA --days 90 --theme dark
    python stonkplot.py MSFT --forecast --days 120 --forecast-days 10 --export

Created with 🖤 by DailyStonks
"""

import argparse
import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Tuple, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from matplotlib.figure import Figure
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Fetching & Processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@lru_cache(maxsize=32)
def fetch_data(ticker: str, days: int = 60) -> Tuple[np.ndarray, List[dt.datetime]]:
    """Fetch and prepare stock data, with caching for performance.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of historical days to retrieve
        
    Returns:
        Tuple of (prices array, dates list)
    """
    try:
        end = dt.datetime.now()
        start = end - dt.timedelta(days=days)
        
        # Fetch with progress=False to avoid console output
        df = yf.download(
            ticker, 
            start=start.strftime('%Y-%m-%d'), 
            end=end.strftime('%Y-%m-%d'), 
            progress=False
        )
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        # Return numpy array for efficient computation and dates for plotting
        return df['Close'].values, df.index.tolist()
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {str(e)}")


def forecast_trend(
    prices: np.ndarray, 
    forecast_days: int = 5,
    confidence: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate AI-powered price forecast with confidence intervals.
    
    Args:
        prices: Historical price data
        forecast_days: Number of days to forecast
        confidence: Confidence interval (0-1)
        
    Returns:
        Tuple of (predicted prices, lower bound, upper bound)
    """
    # Prepare data
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices
    
    # Standardize for better model performance
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Train model with regularization
    model = Ridge(alpha=0.5)
    model.fit(X_scaled, y_scaled)
    
    # Generate future dates for prediction
    future_X = np.arange(len(prices) + forecast_days).reshape(-1, 1)
    future_X_scaled = scaler_X.transform(future_X)
    
    # Predict with rescaling
    predicted_scaled = model.predict(future_X_scaled)
    predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()
    
    # Calculate simple confidence interval
    residuals = y - model.predict(X_scaled) * scaler_y.scale_ + scaler_y.mean_
    std_error = np.std(residuals)
    margin = std_error * 1.96 * (1 - confidence)  # Adjusts width based on confidence
    
    lower_bound = predicted - margin
    upper_bound = predicted + margin
    
    return predicted, lower_bound, upper_bound


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_stonk(
    ticker: str, 
    show_forecast: bool = False,
    days: int = 60,
    forecast_days: int = 5,
    theme: str = 'light',
    export: bool = False
) -> Optional[Figure]:
    """Create beautiful stock visualization with optional forecast.
    
    Args:
        ticker: Stock ticker symbol
        show_forecast: Whether to show AI trend forecast
        days: Number of historical days to display
        forecast_days: Number of days to forecast
        theme: Visual theme ('light', 'dark', or 'cyberpunk')
        export: Whether to save the plot as an image
        
    Returns:
        Matplotlib figure if export=True, otherwise None
    """
    # Fetch and process data
    try:
        prices, dates = fetch_data(ticker, days=days)
        
        # Set up plot styling based on theme
        plt.style.use('seaborn-v0_8-whitegrid' if theme == 'light' else 'dark_background')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Theme-specific colors
        themes = {
            'light': {'price': '#1E88E5', 'forecast': '#FF9800', 'fill': 'rgba(255,152,0,0.2)'},
            'dark': {'price': '#00E5FF', 'forecast': '#FF9100', 'fill': 'rgba(255,145,0,0.25)'},
            'cyberpunk': {'price': '#00FF9F', 'forecast': '#FF00E4', 'fill': 'rgba(255,0,228,0.2)'}
        }
        
        colors = themes.get(theme, themes['light'])
        
        # Special styling for cyberpunk theme
        if theme == 'cyberpunk':
            plt.rcParams['font.family'] = 'monospace'
            fig.patch.set_facecolor('#120458')
            ax.set_facecolor('#120458')
            ax.grid(color='#2F1C6A', linestyle='--', alpha=0.7)
        
        # Plot historical data with enhanced styling
        ax.plot(
            dates, 
            prices, 
            label=f"{ticker}", 
            color=colors['price'], 
            linewidth=2.5,
            marker='o', 
            markersize=0,  # Hidden by default
            markevery=5,  # Show markers periodically
            markerfacecolor='white'
        )
        
        # Create future dates for forecasting
        last_date = dates[-1]
        future_dates = [
            last_date + dt.timedelta(days=i) 
            for i in range(1, forecast_days + 1)
        ]
        all_dates = dates + future_dates
        
        # Add AI forecast if requested
        if show_forecast:
            pred, lower, upper = forecast_trend(
                prices,
                forecast_days=forecast_days,
                confidence=0.9
            )
            
            # Plot the forecast line
            ax.plot(
                all_dates[-len(pred):], 
                pred, 
                linestyle='--', 
                color=colors['forecast'], 
                linewidth=2,
                label=f"AI Forecast"
            )
            
            # Add confidence interval
            ax.fill_between(
                all_dates[-len(pred):],
                lower,
                upper,
                color=colors['fill'],
                label="90% Confidence"
            )
            
            # Add special marker for last actual price
            ax.plot(
                [dates[-1]], 
                [prices[-1]], 
                'o', 
                color=colors['price'], 
                markersize=8,
                markerfacecolor='white'
            )
        
        # Calculate price change
        price_change = prices[-1] - prices[0]
        price_change_pct = (price_change / prices[0]) * 100
        change_color = 'green' if price_change >= 0 else 'red'
        
        # Add annotations
        change_sign = '+' if price_change >= 0 else ''
        title = (
            f"{ticker} | ${prices[-1]:.2f} | {change_sign}{price_change_pct:.2f}% "
            f"{'+ AI Trend' if show_forecast else ''}"
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price (USD)", fontsize=10)
        
        # Add price change annotation
        bbox_props = dict(
            boxstyle="round,pad=0.3", 
            facecolor='white' if theme == 'light' else '#333333',
            alpha=0.8
        )
        
        ax.annotate(
            f"{change_sign}{price_change_pct:.2f}%",
            xy=(dates[-1], prices[-1]),
            xytext=(15, 0),
            textcoords="offset points",
            fontsize=12,
            fontweight='bold',
            color=change_color,
            bbox=bbox_props
        )
        
        # Enhance grid
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates nicely
        fig.autofmt_xdate()
        
        # Add DailyStonks branding in corner
        fig.text(
            0.99, 0.01, 
            "DailyStonks", 
            ha='right', va='bottom', 
            fontsize=8, fontstyle='italic',
            alpha=0.7
        )
        
        # Customize legend
        ax.legend(
            loc='upper left',
            framealpha=0.9,
            facecolor='white' if theme == 'light' else '#333333'
        )
        
        plt.tight_layout()
        
        # Export if requested
        if export:
            output_dir = Path("stonk_charts")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{ticker}_{timestamp}.png"
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {output_path}")
            
            return fig
        else:
            plt.show()
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Parse command line arguments and run the application."""
    parser = argparse.ArgumentParser(
        description="✨ Elegant stock visualization with AI forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "ticker", 
        help="Stock ticker symbol (e.g., AAPL, TSLA)"
    )
    
    parser.add_argument(
        "--forecast", "-f",
        action="store_true", 
        help="Include AI-based price trend forecast"
    )
    
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=60,
        help="Number of historical days to display"
    )
    
    parser.add_argument(
        "--forecast-days", "-fd",
        type=int,
        default=5,
        help="Number of days to forecast ahead"
    )
    
    parser.add_argument(
        "--theme", "-t",
        choices=["light", "dark", "cyberpunk"],
        default="light",
        help="Visual theme for the chart"
    )
    
    parser.add_argument(
        "--export", "-e",
        action="store_true",
        help="Export chart as PNG image"
    )
    
    args = parser.parse_args()
    
    # Run with parsed arguments
    plot_stonk(
        args.ticker.upper(),
        show_forecast=args.forecast,
        days=args.days,
        forecast_days=args.forecast_days,
        theme=args.theme,
        export=args.export
    )


if __name__ == "__main__":
    try:
        print(f"📈 stonkplot.py - Elegant Stock Visualization Tool")
        main()
    except KeyboardInterrupt:
        print("\n✓ Operation cancelled by user")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(