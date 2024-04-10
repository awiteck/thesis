import requests
import csv
import pandas as pd

API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def getWeatherData(latitude, longitude, timeStart, timeEnd, timezone):
    response = requests.get(
        API_BASE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": timeStart,
            "end_date": timeEnd,
            "hourly": "temperature_2m,relativehumidity_2m,cloudcover,windspeed_10m,precipitation",
            "timezone": timezone,
        },
    )

    if response.status_code == 200:
        data = response.json()
        times = data["hourly"]["time"]
        temperatures = data["hourly"]["temperature_2m"]
        humidity = data["hourly"]["relativehumidity_2m"]
        cloudcover = data["hourly"]["cloudcover"]
        windspeed_10m = data["hourly"]["windspeed_10m"]
        precipitation = data["hourly"]["precipitation"]
        # Create a DataFrame from the lists
        df = pd.DataFrame(
            {
                "Time": times,
                "temperature": temperatures,
                "humidity": humidity,
                "cloudcover": cloudcover,
                "windspeed": windspeed_10m,
                "precipitation": precipitation,
            }
        )
        return df
    else:
        print(f"Error {response.status_code}: {response.text}")

    """
    response = requests.get(API_BASE_URL, params={
        'latitude': latitude,
        'longitude': longitude,
        'start_date': timeStart,
        'end_date': timeEnd,
        'hourly': 'temperature_2m'
    })

    if response.status_code == 200:
        data = response.json()
        times = data['hourly']['time']
        temperatures = data['hourly']['temperature_2m']

        # Open the CSV file for writing
        with open(outputCSVPath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header row
            writer.writerow(["Time", "Temperature (°C)"])

            # Write each row of data
            for time, temperature in zip(times, temperatures):
                writer.writerow([time, temperature])

        print(f"Data written to {outputCSVPath}")

    else:
        print(f"Error {response.status_code}: {response.text}")
    """
