#!/usr/bin/env python3
"""
stonkplot.py - Elegant Stock Visualization Tool

An elegant, lightweight stock plotting utility with AI-powered trend forecasting.
Perfect for quick market insights with minimal effort.

Usage:
    python stonkplot.py AAPL
    python stonkplot.py TSLA --forecast
    python stonkplot.py NVDA --days 90 --theme dark
    python stonkplot.py MSFT --forecast --days 120 --forecast-days 10 --export

Created with by DailyStonks
"""

import sys
import subprocess
import importlib.util
import os
import argparse
import platform
from pathlib import Path
import textwrap


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dependency Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED_PACKAGES = {
    "matplotlib": "matplotlib>=3.5.0",
    "numpy": "numpy>=1.20.0",
    "yfinance": "yfinance>=0.2.12",
    "scikit-learn": "scikit-learn>=1.0.0",
}


def is_package_installed(package_name):
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None


def install_package(package_spec):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec, "--quiet"])
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_dependencies():
    """Check and install dependencies if needed."""
    missing_packages = {}
    
    # Check which packages are missing
    for package_name, package_spec in REQUIRED_PACKAGES.items():
        if not is_package_installed(package_name):
            missing_packages[package_name] = package_spec
    
    if not missing_packages:
        return True
    
    # Print nice message about missing dependencies
    print(f"\n{'' * 30}")
    print(" Missing dependencies detected!")
    print(f"{'' * 30}")
    print("\nThe following packages are required but not installed:")
    for package_name in missing_packages:
        print(f"   {package_name}")
    
    # Ask for confirmation before installing
    user_input = input("\nWould you like to install these dependencies now? [Y/n]: ").strip().lower()
    if user_input not in ["", "y", "yes"]:
        print("\n Dependencies are required to run this script. Exiting...")
        return False
    
    # Install missing packages
    print("\n Installing dependencies...")
    
    for package_name, package_spec in missing_packages.items():
        print(f"   Installing {package_name}...", end="", flush=True)
        if install_package(package_spec):
            print(" ")
        else:
            print(" ")
            print(f"\n Failed to install {package_name}. Please install manually with:")
            print(f"    pip install {package_spec}")
            return False
    
    print("\n All dependencies installed successfully!")
    print(" Initializing stonkplot...")
    print(f"{'' * 30}\n")
    
    # Re-import modules to make them available
    globals()["plt"] = __import__("matplotlib.pyplot").pyplot
    globals()["np"] = __import__("numpy")
    globals()["yf"] = __import__("yfinance")
    globals()["Ridge"] = getattr(__import__("sklearn.linear_model", fromlist=["Ridge"]), "Ridge")
    globals()["StandardScaler"] = getattr(__import__("sklearn.preprocessing", fromlist=["StandardScaler"]), "StandardScaler")
    
    # Import datetime normally since it's a standard library
    global dt
    import datetime as dt
    from functools import lru_cache
    from typing import Tuple, List, Optional, Union
    from matplotlib.figure import Figure
    
    return True


# Only attempt imports after checking dependencies in the main execution
if all(is_package_installed(package) for package in REQUIRED_PACKAGES):
    # Standard library imports
    import datetime as dt
    from functools import lru_cache
    from typing import Tuple, List, Optional, Union
    
    # Third-party imports (these are guaranteed to be available at this point)
    import matplotlib.pyplot as plt
    import numpy as np
    import yfinance as yf
    from matplotlib.figure import Figure
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Fetching & Processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# NOTE: Removed the @lru_cache decorator to fix the "lru_cache is not defined" error
# The caching is now imported at runtime to avoid this issue
def fetch_data(ticker: str, days: int = 60) -> tuple[np.ndarray, List[dt.datetime]]:
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
        
        # Fixed: Added auto_adjust=False to prevent the "unsupported format string" error
        df = yf.download(
            ticker, 
            start=start.strftime('%Y-%m-%d'), 
            end=end.strftime('%Y-%m-%d'), 
            progress=False,
            auto_adjust=False  # This fixes the format string error
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
        print(f"Error: {str(e)}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_header():
    """Print a cool ASCII art header."""
    header = """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃                     stonkplot                       ┃
    ┃           Elegant Stock Visualization Tool          ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    print(header)


def main():
    """Parse command line arguments and run the application."""
    parser = argparse.ArgumentParser(
        description="Elegant stock visualization with AI forecasting",
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


def check_python_version():
    """Check if Python version is at least 3.7."""
    if sys.version_info < (3, 7):
        print("Python 3.7 or higher is required to run this script.")
        print(f"   Current Python version: {platform.python_version()}")
        return False
    return True


if __name__ == "__main__":
    try:
        print_header()
        
        # Check Python version
        if not check_python_version():
            sys.exit(1)
        
        # Check and install dependencies if needed
        if not all(is_package_installed(package) for package in REQUIRED_PACKAGES):
            if not ensure_dependencies():
                sys.exit(1)
        
        # Run the main application
        main()
        
    except KeyboardInterrupt:
        print("\n Operation cancelled by user")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
