# API Data Collection and Analysis

## Weather API Project

This project collects data from an online weather API and updates it to a CSV file. The main tasks include:

- Fetching weather data from the API.
- Parsing the received data.
- Storing the parsed data into a CSV file for further analysis.

### Steps to Run the Project

1. **Install Dependencies**: Ensure you have the necessary libraries installed.
2. **Configure API Access**: Set up your API key and endpoint.
3. **Run the Script**: The script is automatically executed every 2 hours using GitHub Actions to collect and store the data.

### Requirements

- Python 3.x
- `requests` library
- `pandas` library

### Example Usage

```python
import requests
import pandas as pd

# Fetch data from the API
response = requests.get('YOUR_API_ENDPOINT')
data = response.json()

# Parse and store data in a CSV file
df = pd.DataFrame(data)
df.to_csv('weather_data.csv', index=False)
```

### GitHub Actions Workflow

Create a `.github/workflows/data_collection.yml` file with the following content to automate the script execution:

```yaml
name: Collect Weather Data

on:
     schedule:
          - cron: '0 */2 * * *' # Runs every 2 hours

jobs:
     collect-data:
          runs-on: ubuntu-latest

          steps:
          - name: Checkout repository
               uses: actions/checkout@v2

          - name: Set up Python
               uses: actions/setup-python@v2
               with:
                    python-version: '3.x'

          - name: Install dependencies
               run: |
                    python -m pip install --upgrade pip
                    pip install requests pandas

          - name: Run script
               run: |
                    python /path/to/your/script.py
```

### Notes

- A public API is used.
- Handle exceptions and errors for robust data collection.
- The script is scheduled to run every 2 hours using GitHub Actions for continuous data collection.
