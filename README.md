<div align="center"><img src="https://github.com/whispr-dev/DailyStonks__mini/blob/main/assets/DailyStonks_preview.png?raw=true" alt="DailyStonks Banner"><p><i>The code so clean, you could eat off it. (Please don't.)</i></p></div>DailyStonks Mini: Precision & Polish in a Python PackageDailyStonks Mini is a standalone, lightweight stock plotting utility with an AI-powered trend forecasting feature. While it's a fully functional tool, its primary purpose is a showcase of our commitment to clean code, elegant design, and robust engineering.This project demonstrates:Modular and Maintainable Code: A tidy codebase that's easy to read, understand, and extend.Intelligent Automation: Smart dependency management and data caching that simply works.Polished User Experience: Intuitive command-line interface and beautiful, customizable visual themes.What It DoesJust a few lines of code to generate insightful stock charts, complete with optional AI-driven trend forecasts. It's a testament to the power of a well-engineered tool that provides significant value with minimal complexity.Sample Outputs<br><div align="center"><img src="github.com/whispr-dev/DailyStonks__mini/blob/main/assets/AAPL-90day-cybertheme-ai90-0fc.png" width="400" alt="Apple 90-day light theme chart"><img src="https://https://github.com/whispr-dev/DailyStonks__mini/blob/main/assets/TSLA-240day-darktheme-90fc.png?raw=true" width="400" alt="Tesla 240-day dark theme chart"><img src="ttps://github.com/whispr-dev/DailyStonks__mini/blob/main/assets/AAPL-360day-cybertheme-45fc.png?raw=true" width="400" alt="Apple 360-day cyberpunk theme chart" </div>Getting StartedBecause a truly well-built tool should be easy to use, we've made setup a breeze.InstallationClone the repository and install the dependencies. The script will handle the rest.# Clone the repository
git clone [https://github.com/your-username/dailystonks-mini.git](https://github.com/your-username/dailystonks-mini.git)
cd dailystonks-mini

# Install dependencies (they'll be auto-managed)
pip install -r requirements.txt
UsageThe CLI is designed for clarity.# Basic usage
python stonkplot.py AAPL

# With AI forecasting and a different theme
python stonkplot.py TSLA --forecast --theme dark

# Change the time range and forecast length
python stonkplot.py NVDA --days 90 --forecast --forecast-days 10

# Export to a file
python stonkplot.py MSFT --export
For a full list of options, use --help.LicenseThis project is licensed under a Hybrid License (MIT + CC0). Feel free to use it, learn from it, and even contribute.For more details, see the LICENSE file.<br>Created by the team at DailyStonks.Find more of our work at github.com/dailystonks.
