# 📈 stonkplot.py - Elegant Stock Visualization Tool

An elegant, lightweight stock plotting utility with AI-powered trend forecasting.
Perfect for quick market insights with minimal effort.

![AAPL Stock Demo](https://i.imgur.com/example.png)

## ✨ Features

- **Beautiful Stock Visualizations** - Clean, informative charts that highlight key data points
- **AI-Powered Forecasting** - Optional trend predictions with confidence intervals
- **Multiple Visual Themes** - Choose from light, dark, or cyberpunk aesthetics
- **Data Caching** - Smart performance optimizations for faster repeat lookups
- **Simple CLI Interface** - Intuitive command-line options for customization
- **Export Capability** - Save high-resolution PNG charts for reports and sharing

## 🚀 Installation

```bash
# 📈 stonkplot.py - Elegant Stock Visualization Tool

An elegant, lightweight stock plotting utility with AI-powered trend forecasting.
Perfect for quick market insights with minimal effort.

![stonkplot demo](https://github.com/dailystonks/stonkplot/assets/example/stonkplot_demo.png)

## ✨ Features

- **Beautiful Stock Visualizations** - Clean, informative charts that highlight key data points
- **AI-Powered Forecasting** - Optional trend predictions with confidence intervals
- **Multiple Visual Themes** - Choose from light, dark, or cyberpunk aesthetics
- **Data Caching** - Smart performance optimizations for faster repeat lookups
- **Simple CLI Interface** - Intuitive command-line options for customization
- **Export Capability** - Save high-resolution PNG charts for reports and sharing

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/dailystonks/stonkplot.git
cd stonkplot

# Install dependencies
pip install -r requirements.txt
```

Or just grab the standalone script:

```bash
# Make sure you have the required packages
pip install matplotlib numpy yfinance scikit-learn
```

## 📦 Dependencies

- Python 3.7+
- matplotlib
- numpy
- yfinance
- scikit-learn

## 🛠️ Usage

### Basic Usage

```bash
# Basic stock chart
python stonkplot.py AAPL

# Add AI-powered forecast
python stonkplot.py TSLA --forecast
```

### Advanced Options

```bash
# Change time range
python stonkplot.py NVDA --days 90

# Change theme
python stonkplot.py MSFT --theme dark

# Customize forecast length
python stonkplot.py GOOG --forecast --forecast-days 10

# Export as an image
python stonkplot.py AMZN --export
```

### All Options

```
usage: stonkplot.py [-h] [--forecast] [--days DAYS] [--forecast-days FORECAST_DAYS]
                  [--theme {light,dark,cyberpunk}] [--export]
                  ticker

✨ Elegant stock visualization with AI forecasting

positional arguments:
  ticker                Stock ticker symbol (e.g., AAPL, TSLA)

options:
  -h, --help            show this help message and exit
  --forecast, -f        Include AI-based price trend forecast (default: False)
  --days, -d DAYS       Number of historical days to display (default: 60)
  --forecast-days, -fd FORECAST_DAYS
                        Number of days to forecast ahead (default: 5)
  --theme, -t {light,dark,cyberpunk}
                        Visual theme for the chart (default: light)
  --export, -e          Export chart as PNG image (default: False)
```

## 🖼️ Examples

### Light Theme (Default)
![Light Theme Example](https://github.com/dailystonks/stonkplot/assets/example/light_theme.png)

### Dark Theme 
![Dark Theme Example](https://github.com/dailystonks/stonkplot/assets/example/dark_theme.png)

### Cyberpunk Theme
![Cyberpunk Theme Example](https://github.com/dailystonks/stonkplot/assets/example/cyberpunk_theme.png)

## 🔮 About the AI Forecast

The forecast uses a Ridge regression model with:

- Data standardization for better accuracy
- Confidence intervals showing prediction uncertainty
- 90% confidence bands by default

The model works best for short-term trend analysis and should not be used as financial advice.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest features
- Submit pull requests

## ⚖️ License