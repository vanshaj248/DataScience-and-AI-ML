import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime
import csv
import os
import subprocess

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 21.1959,
	"longitude": 72.8302,
	"current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "is_day", "precipitation"],
	"timezone": "auto",
	"forecast_days": 16
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Current values. The order of variables needs to be the same as requested.
current = response.Current()
current_temperature_2m = current.Variables(0).Value()
current_relative_humidity_2m = current.Variables(1).Value()
current_apparent_temperature = current.Variables(2).Value()
current_is_day = current.Variables(3).Value()
current_precipitation = current.Variables(4).Value()

date = datetime.now().date()
time = datetime.now().time()
print(date, time)

# Define the CSV file path
csv_file_path = '/Users/vanshajshah/Documents/Programming/code/DataScience-and-AI-ML/DataScience and AI ML/Data Collection and Storage/Api_Learn/API Data Collection and Analysis/weatherdata.csv'

def update_csv():
	# Check if the CSV file exists
	file_exists = os.path.isfile(csv_file_path)

	# Open the CSV file in append mode
	with open(csv_file_path, mode='a', newline='') as file:
		writer = csv.writer(file)
		
		# Write the header if the file does not exist
		if not file_exists:
			writer.writerow(["date", "time", "current_temperature", "apparent_temperature"])
		
		# Write the data
		writer.writerow([date, time, current_temperature_2m, current_apparent_temperature])

	print("Data written to CSV file.")

def commit_and_push_changes():
	# Git commands to commit and push
	subprocess.run(['git', 'config', 'user.name', 'github-actions[bot]'])
	subprocess.run(['git', 'config', 'user.email', 'github-actions[bot]@users.noreply.github.com'])
	subprocess.run(['git', 'add', csv_file_path])
	subprocess.run(['git', 'commit', '-m', 'Update weather data'])
	subprocess.run(['git', 'push'])

if __name__ == "__main__":
	update_csv()
	commit_and_push_changes()