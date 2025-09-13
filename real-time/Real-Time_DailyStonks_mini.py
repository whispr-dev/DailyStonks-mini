#!/usr/bin/env python3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RealTime_DailyStonks_mini - the realtime stonkplot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
realtime_stonkplot.py - Elegant Stock Visualization Tool

An elegant, lightweight stock plotting utility with AI-powered trend forecasting.
Perfect for quick market insights with minimal effort.

Usage:
    python stonkplot.py AAPL
    python stonkplot.py TSLA --forecast
    python stonkplot.py --days 90 --theme dark
    python stonkplot.py NVDA --forecast --days 120 --forecast-days 10 --export
    python stonkplot.py --forecast --days 60 --forecast-days 30 --refresh 10

Created by DailyStonks
"""

import sys
import subprocess
import importlib.util
import os
import time
import argparse
import platform
from pathlib import Path
import textwrap
import datetime as dt
import readchar as rc
from typing import Tuple, List, Optional, Union
from functools import lru_cache
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

POPULAR_TICKERS = [
    "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
    "NVDA", "META", "NFLX", "AMD", "INTC"
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dependency Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED_PACKAGES = {
    "matplotlib": "matplotlib>=3.5.0",
    "numpy": "numpy>=1.20.0",  # or optional upgarde to have cude gfx boost on math:
#   "CuPy": "CuPy=12.4",  #  ignore if cuda no go
    "yfinance": "yfinance>=0.2.12",
    "pandas": "pandas>=1.3.0",  # Added pandas as it's needed by yfinance
    "sklearn": "scikit-learn>=1.0.0",  # Fixed: was "sklearn" but should be "scikit-learn"
    "readchar": "readchar>=1.0.0",  # Added readchar for arrow-keys in ticker scroll
}

def is_package_installed(package_name):
    """Check if a package is installed."""
    # Handle sklearn/scikit-learn naming issue
    if package_name == "sklearn":
        return importlib.util.find_spec("sklearn") is not None
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
    print("\nMissing dependencies detected!")
    print("\nThe following packages are required but not installed:")
    for package_name in missing_packages:
        print(f"  - {package_name}")
    
    # Ask for confirmation before installing
    user_input = input("\nWould you like to install these dependencies now? [Y/n]: ").strip().lower()
    if user_input not in ["", "y", "yes"]:
        print("\nDependencies are required to run this script. Exiting...")
        return False
    
    # Install missing packages
    print("\nInstalling dependencies...")
    
    for package_name, package_spec in missing_packages.items():
        print(f"  Installing {package_name}...", end="", flush=True)
        if install_package(package_spec):
            print(" Done")
        else:
            print(" Failed")
            print(f"\nFailed to install {package_name}. Please install manually with:")
            print(f"    pip install {package_spec}")
            return False
    
    print("\nAll dependencies installed successfully!")
    print("Initializing stonkplot...\n")
    
    return True


# Import dependencies
if not all(is_package_installed(package) for package in REQUIRED_PACKAGES):
    if not ensure_dependencies():
        sys.exit(1)

# Now that dependencies are guaranteed to be installed, import them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import readchar as rc
from matplotlib.figure import Figure
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Fetching & Processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
        
        # Direct pandas approach to avoid yfinance formatting issues
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=f"{days}d")
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        # Work with raw numpy arrays to prevent formatting issues
        prices = df['Close'].to_numpy()
        dates = df.index.tolist()
        
        return prices, dates
        
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
    export: bool = False,
    fig: Optional[Figure] = None,
    ax: Optional[plt.Axes] = None,
    refresh_mode: bool = False,
    show_loading_screen: bool = False,
    debug: bool = False
) -> Optional[Figure]:
    try:
        if debug:
            print(f"🐛 Fetching data for {ticker}...")
        
        prices, dates = fetch_data(ticker, days=days)
        
        if debug:
            print(f"🐛 Got {len(prices)} prices, {len(dates)} dates")

        if fig is None or ax is None:
            if debug:
                print("🐛 Creating new figure...")
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            if debug:
                print("🐛 Clearing existing axes...")
            ax.clear()

        # Theme-specific colors
        themes = {
            'light': {'style': 'seaborn-v0_8-whitegrid', 'price': '#1E88E5', 'forecast': '#FF9800', 'fill': (1.0, 0.6, 0.0, 0.2)},
            'dark': {'style': 'dark_background', 'price': '#00E5FF', 'forecast': '#FF9100', 'fill': (1.0, 0.56, 0.0, 0.25)},
            'cyberpunk': {'style': 'dark_background', 'price': '#00FF9F', 'forecast': '#FF00E4', 'fill': (1.0, 0.0, 0.9, 0.2)},
            'regret': {'style': 'dark_background', 'price': '#FF00E4', 'forecast': '#FF9800', 'fill': (1.0, 0.6, 0.0, 0.2)}
        }

        colors = themes.get(theme, themes['light'])
        
        if debug:
            print(f"🐛 Using theme: {theme}, style: {colors['style']}")
            
        try:
            plt.style.use(colors['style'])
        except Exception as e:
            if debug:
                print(f"🐛 Style warning: {e}, using default")
            # Fallback to a basic style if the requested one fails
            plt.style.use('default')

        if show_loading_screen:
            text_color = 'white' if theme != 'light' else 'black'
            for i in range(3):
                ax.clear()
                ax.set_facecolor('#ffffff' if theme == 'light' else '#000000')
                ax.text(
                    0.5, 0.5, f"LOADING{'.' * (i+1)}",
                    fontsize=28,
                    fontweight='bold',
                    color=text_color,
                    ha='center',
                    va='center',
                    transform=ax.transAxes
                )
                if fig.canvas.manager is not None:  # Check if canvas is still valid
                    fig.canvas.draw_idle()  # Instead of fig.canvas.draw()
                    fig.canvas.flush_events()  # Process GUI events
                    plt.pause(0.35)
                else:
                    break
            ax.clear()

        # Theme-specific layout overrides
        if theme in ['cyberpunk', 'regret']:
            fig.patch.set_facecolor('#120458')
            ax.set_facecolor('#120458')
            ax.grid(color='#2F1C6A', linestyle='--', alpha=0.7)

        # Plot price
        ax.plot(
            dates, prices,
            label=f"{ticker}",
            color=colors['price'],
            linewidth=2.5,
            marker='o',
            markersize=0,
            markevery=5,
            markerfacecolor='white'
        )

        # Forecasting
        last_date = dates[-1]
        future_dates = [last_date + dt.timedelta(days=i) for i in range(1, forecast_days + 1)]
        all_dates = dates + future_dates

        if show_forecast:
            pred, lower, upper = forecast_trend(prices, forecast_days, confidence=0.9)
            ax.plot(
                all_dates[-len(pred):],
                pred,
                linestyle='--',
                color=colors['forecast'],
                linewidth=2,
                label="AI Forecast"
            )
            ax.fill_between(
                all_dates[-len(pred):],
                lower,
                upper,
                color=colors['fill'],
                label="90% Confidence"
            )
            ax.plot(
                [dates[-1]], [prices[-1]],
                'o',
                color=colors['price'],
                markersize=8,
                markerfacecolor='white'
            )

        price_change = prices[-1] - prices[0]
        price_change_pct = (price_change / prices[0]) * 100
        change_color = 'green' if price_change >= 0 else 'red'
        change_sign = '+' if price_change >= 0 else ''

        title = f"{ticker} | ${prices[-1]:.2f} | {change_sign}{price_change_pct:.2f}% {'+ AI Trend' if show_forecast else ''}"
        ax.set_title(title, fontsize=14, fontweight='bold', color='white' if theme != 'light' else 'black')
        ax.set_xlabel("Date", fontsize=10, color='white' if theme != 'light' else 'black')
        ax.set_ylabel("Price (USD)", fontsize=10, color='white' if theme != 'light' else 'black')

        ax.tick_params(colors='white' if theme != 'light' else 'black')

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

        fig.autofmt_xdate()

        fig.text(
            0.99, 0.01,
            "DailyStonks",
            ha='right', va='bottom',
            fontsize=8, fontstyle='italic',
            color='white' if theme != 'light' else 'black',
            alpha=0.7
        )

        ax.legend(
            loc='upper left',
            framealpha=0.9,
            facecolor='white' if theme == 'light' else '#333333',
            labelcolor='black' if theme == 'light' else 'white'
        )

        plt.tight_layout()

        if export:
            output_dir = Path("stonk_charts")
            output_dir.mkdir(exist_ok=True)
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{ticker}_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {output_path}")
        elif refresh_mode:
            # More robust canvas handling for refresh mode
            if fig.canvas.manager is not None:
                try:
                    fig.canvas.draw()
                    fig.canvas.flush_events()  # Process any pending GUI events
                    plt.pause(0.1)  # Slightly longer pause for stability
                except Exception as e:
                    print(f"Canvas error (non-fatal): {e}")
                    return fig  # Return the figure even if canvas update fails
            else:
                print("Canvas lost, recreating...")
                return None  # Signal to recreate the figure
        else:
            plt.show()

        return fig

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Fancy header
def print_header():
    """Print a fancy header."""
    header = """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃                     stonkplot                       ┃
    ┃           Elegant Stock Visualization Tool          ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    print(header)

# Ticker scroll menu main logic
def scroll_ticker_menu() -> str:
    """Interactive CLI ticker selector with arrow key support and manual entry."""
    print("\n🔁 Use ↑/↓ arrows to scroll. Press Enter to select.")
    print("Or type a custom ticker and press Enter. (ESC/q to quit)\n")

    index = 0
    custom_input = ""

    while True:
        # Clear line and redraw
        print(f"\r→ {POPULAR_TICKERS[index]}{' '*20}", end='', flush=True)

        try:
            key = rc.readkey()
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

        if key == rc.key.UP:
            index = (index - 1) % len(POPULAR_TICKERS)
        elif key == rc.key.DOWN:
            index = (index + 1) % len(POPULAR_TICKERS)
        elif key == rc.key.ENTER:
            if custom_input:
                return custom_input.upper()
            return POPULAR_TICKERS[index]
        elif key == rc.key.ESC or key.lower() == 'q':
            print("\nExiting.")
            sys.exit(0)
        elif key.isalnum():  # Start manual input
            custom_input += key
            print(f"\r→ {custom_input.upper()}{' '*20}", end='', flush=True)
        elif key == rc.key.BACKSPACE:
            custom_input = custom_input[:-1]
            print(f"\r→ {custom_input.upper()}{' '*20}", end='', flush=True)
        else:
            pass  # ignore other keys


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Elegant stock visualization with AI forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("ticker", nargs="?", default=None, help="Stock ticker symbol (e.g., AAPL, TSLA)")
    parser.add_argument("--forecast", "-f", action="store_true", help="Include AI-based price trend forecast")
    parser.add_argument("--days", "-d", type=int, default=60, help="Number of historical days to display")
    parser.add_argument("--forecast-days", "-fd", type=int, default=5, help="Number of days to forecast ahead")
    parser.add_argument("--theme", "-t", choices=["light", "dark", "cyberpunk", "regret"], default="light", help="Visual theme for the chart")
    parser.add_argument("--export", "-e", action="store_true", help="Export chart as PNG image")
    parser.add_argument("--refresh", "-r", type=int, default=0, help="Enable auto-refresh every X seconds (0 to disable)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    if args.debug:
        print("🐛 DEBUG MODE ENABLED")
        print(f"🐛 Python version: {sys.version}")
        print(f"🐛 Matplotlib backend: {matplotlib.get_backend()}")

    ticker = args.ticker
    if not ticker:
        try:
            ticker = scroll_ticker_menu()
            if args.debug:
                print(f"🐛 Selected ticker: {ticker}")
        except Exception as e:
            print(f"❌ Error in ticker selection: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return

    # Test matplotlib setup
    if args.debug:
        print("🐛 Testing matplotlib setup...")
        try:
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            plt.close(test_fig)
            print("🐛 ✅ Matplotlib test passed")
        except Exception as e:
            print(f"🐛 ❌ Matplotlib test failed: {e}")
            return

    # Initialize matplotlib figure outside the loop for better stability
    print("🔧 Initializing matplotlib...")
    plt.ion()  # Turn on interactive mode
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        if args.debug:
            print(f"🐛 Created figure {fig.number}")
    except Exception as e:
        print(f"❌ Failed to create initial figure: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return
    
    first_frame = True

    def clear_screen():
        if not args.debug:  # Don't clear screen in debug mode
            os.system("cls" if os.name == "nt" else "clear")

    try:
        refresh_count = 0
        while True:
            if args.debug:
                print(f"🐛 Loop iteration {refresh_count}")
            
            clear_screen()
            if not args.debug:  # Don't show header in debug mode to avoid clutter
                print_header()

            # If figure was lost, recreate it
            if fig is None or not plt.fignum_exists(fig.number):
                print("🔧 Recreating lost figure...")
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    if args.debug:
                        print(f"🐛 Recreated figure {fig.number}")
                except Exception as e:
                    print(f"❌ Failed to recreate figure: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                    break

            if args.debug:
                print(f"🐛 Calling plot_stonk with ticker={ticker}")

            try:
                result_fig = plot_stonk(
                    ticker.upper(),
                    show_forecast=args.forecast,
                    days=args.days,
                    forecast_days=args.forecast_days,
                    theme=args.theme,
                    export=args.export,
                    fig=fig,
                    ax=ax,
                    refresh_mode=args.refresh > 0,
                    show_loading_screen=first_frame,
                    debug=args.debug
                )
                
                if args.debug:
                    print(f"🐛 plot_stonk returned: {result_fig is not None}")
                    
            except Exception as e:
                print(f"❌ Error in plot_stonk: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                break

            # Handle figure recreation if needed
            if result_fig is None and args.refresh > 0:
                print("🔧 Figure lost, recreating...")
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    continue
                except Exception as e:
                    print(f"❌ Failed to recreate figure after loss: {e}")
                    break

            first_frame = False
            refresh_count += 1

            if args.refresh <= 0:
                if args.debug:
                    print("🐛 No refresh mode, showing plot and waiting...")
                # For non-refresh mode, keep the plot open
                try:
                    plt.show(block=True)  # This will block until window is closed
                except Exception as e:
                    print(f"❌ Error showing plot: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                break

            print(f"\n🔄 Refreshing in {args.refresh} seconds... (Ctrl+C to stop)")
            
            # More robust sleep with KeyboardInterrupt handling
            try:
                for i in range(args.refresh):
                    if args.debug and i == 0:
                        print(f"🐛 Starting {args.refresh}s sleep...")
                    time.sleep(1)
                    # Check if figure still exists during sleep
                    if not plt.fignum_exists(fig.number):
                        print("🔧 Figure closed by user")
                        return
            except KeyboardInterrupt:
                print("\n⏹️ Refresh stopped by user")
                break

    except Exception as e:
        print(f"❌ Unexpected error in main loop: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        plt.ioff()  # Turn off interactive mode
        try:
            if fig is not None and plt.fignum_exists(fig.number):
                plt.close(fig)
                if args.debug:
                    print("🐛 Closed figure")
        except Exception as e:
            if args.debug:
                print(f"🐛 Error closing figure: {e}")


def check_python_version():
    """Check if Python version is at least 3.7."""
    if sys.version_info < (3, 7):
        print("Python 3.7 or higher is required to run this script.")
        print(f"Current Python version: {platform.python_version()}")
        return False
    return True


# To tie it all off
if __name__ == "__main__":
    try:
        print_header()

        if not check_python_version():
            sys.exit(1)

        main()

# And ending gracefully with err msgs for ends:
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        
# Fin.