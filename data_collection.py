# data_ingestion.py - Optimized for Speed

import requests
import math
import pandas as pd
from datetime import date, timedelta, datetime
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import concurrent.futures  # For parallel API calls
import time  # For potential rate limiting, though less frequent now


# --- Configuration Class (Copied from your Accuracy_RF_ARIMA_main.py) ---
class WeatherConfig:
    """Stores configuration settings for the weather forecasting model."""

    def __init__(self):
        self.delhi_latitude = 28.6448  # Approximate latitude for Delhi
        self.delhi_longitude = 77.2167  # Approximate longitude for Delhi
        self.radius_km = 500  # Radius for surrounding areas in kilometers
        self.num_regions = 8  # Number of surrounding regions
        self.earth_radius_km = 6371  # Earth's mean radius in kilometers

        # Define bearings for the 8 regions (N, NE, E, SE, S, SW, W, NW)
        self.region_bearings = [0, 45, 90, 135, 180, 225, 270, 315]
        self.region_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        # Weather variables (used for DB schema and processing)
        self.weather_variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "precipitation",
            "cloud_cover",
        ]

        # PostgreSQL Database Configuration
        self.db_name = "Project_Weather_Forecasting"
        self.db_user = "postgres"
        self.db_password = "1234"
        self.db_host = "localhost"
        self.db_port = "5432"


# --- Database Manager Class (Modified to include insert methods) ---
class DatabaseManager:
    """Manages PostgreSQL database connection and operations."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                dbname=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
                host=self.config.db_host,
                port=self.config.db_port,
            )
            self.conn.autocommit = True  # Auto-commit changes
            print("Database connected.")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def create_tables(self):
        """Creates necessary tables if they don't exist."""
        if not self.conn:
            print("Cannot create tables: No database connection.")
            return

        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS delhi_hourly_weather (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP UNIQUE,
                    temperature_2m REAL,
                    relative_humidity_2m REAL,
                    wind_speed_10m REAL,
                    wind_direction_10m REAL,
                    weather_code INTEGER,
                    precipitation REAL,
                    cloud_cover REAL
                );
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS surrounding_hourly_weather (
                    id SERIAL PRIMARY KEY,
                    region_name TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    timestamp TIMESTAMP,
                    temperature_2m REAL,
                    relative_humidity_2m REAL,
                    wind_speed_10m REAL,
                    wind_direction_10m REAL,
                    weather_code INTEGER,
                    precipitation REAL,
                    cloud_cover REAL,
                    UNIQUE (region_name, timestamp)
                );
            """
            )
            print("Tables checked/created.")
        except psycopg2.Error as e:
            print(f"Error creating tables: {e}")
        finally:
            cur.close()

    def insert_delhi_data(self, data: pd.DataFrame):
        """Inserts hourly Delhi weather data into the delhi_hourly_weather table."""
        if data.empty or not self.conn:
            return

        cur = self.conn.cursor()
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Prepare values for insertion, handling potential NaNs by converting to None
        rows = [
            (
                row["timestamp"],
                row["temperature_2m"] if pd.notna(row["temperature_2m"]) else None,
                (
                    row["relative_humidity_2m"]
                    if pd.notna(row["relative_humidity_2m"])
                    else None
                ),
                row["wind_speed_10m"] if pd.notna(row["wind_speed_10m"]) else None,
                (
                    row["wind_direction_10m"]
                    if pd.notna(row["wind_direction_10m"])
                    else None
                ),
                row["weather_code"] if pd.notna(row["weather_code"]) else None,
                row["precipitation"] if pd.notna(row["precipitation"]) else None,
                row["cloud_cover"] if pd.notna(row["cloud_cover"]) else None,
            )
            for idx, row in data.iterrows()
        ]

        try:
            insert_query = sql.SQL(
                """
                INSERT INTO delhi_hourly_weather (timestamp, temperature_2m, relative_humidity_2m,
                                                  wind_speed_10m, wind_direction_10m, weather_code,
                                                  precipitation, cloud_cover)
                VALUES %s
                ON CONFLICT (timestamp) DO NOTHING;
            """
            )
            execute_values(cur, insert_query, rows)
            print(f"Inserted/Skipped {len(rows)} Delhi hourly records.")
        except psycopg2.Error as e:
            print(f"Error inserting Delhi data: {e}")
        finally:
            cur.close()

    def insert_surrounding_data(self, data: pd.DataFrame):
        """Inserts hourly surrounding region weather data into the surrounding_hourly_weather table."""
        if data.empty or not self.conn:
            return

        cur = self.conn.cursor()
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Prepare values for insertion, handling potential NaNs by converting to None
        rows = [
            (
                row["region_name"],
                row["latitude"] if pd.notna(row["latitude"]) else None,
                row["longitude"] if pd.notna(row["longitude"]) else None,
                row["timestamp"],
                row["temperature_2m"] if pd.notna(row["temperature_2m"]) else None,
                (
                    row["relative_humidity_2m"]
                    if pd.notna(row["relative_humidity_2m"])
                    else None
                ),
                row["wind_speed_10m"] if pd.notna(row["wind_speed_10m"]) else None,
                (
                    row["wind_direction_10m"]
                    if pd.notna(row["wind_direction_10m"])
                    else None
                ),
                row["weather_code"] if pd.notna(row["weather_code"]) else None,
                row["precipitation"] if pd.notna(row["precipitation"]) else None,
                row["cloud_cover"] if pd.notna(row["cloud_cover"]) else None,
            )
            for idx, row in data.iterrows()
        ]

        try:
            insert_query = sql.SQL(
                """
                INSERT INTO surrounding_hourly_weather (region_name, latitude, longitude, timestamp,
                                                        temperature_2m, relative_humidity_2m, wind_speed_10m,
                                                        wind_direction_10m, weather_code, precipitation, cloud_cover)
                VALUES %s
                ON CONFLICT (region_name, timestamp) DO NOTHING;
            """
            )
            execute_values(cur, insert_query, rows)
            print(f"Inserted/Skipped {len(rows)} surrounding region hourly records.")
        except psycopg2.Error as e:
            print(f"Error inserting surrounding data: {e}")
        finally:
            cur.close()

    def close_connection(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")


# --- Open-Meteo API Fetcher Class ---
class OpenMeteoAPI:
    """Fetches weather data from Open-Meteo APIs."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
        self.FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
        self.weather_variables = self.config.weather_variables
        self.hourly_params = ",".join(self.weather_variables)

    def _fetch_data(self, url, latitude, longitude, start_date, end_date):
        """Helper to fetch data from a given URL."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": self.hourly_params,
            "timezone": "Asia/Kolkata",  # Delhi's timezone
            "past_days": 0,  # Ensure we only get data for the specified range
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(
                f"Error fetching data from {url} for {latitude},{longitude} ({start_date} to {end_date}): {e}"
            )
            return None

    def fetch_historical_era5(self, latitude, longitude, start_date, end_date):
        """Fetches historical hourly ERA5 data from Open-Meteo archive for a date range."""
        print(
            f"Fetching ERA5 historical data for {latitude},{longitude} from {start_date} to {end_date}"
        )
        return self._fetch_data(
            self.ERA5_ARCHIVE_URL, latitude, longitude, start_date, end_date
        )

    def fetch_current_day_forecast(self, latitude, longitude, current_date):
        """Fetches forecast data for the *current day* (treated as today's actuals)."""
        print(f"Fetching forecast data for {latitude},{longitude} for {current_date}")
        # For forecast API, start_date and end_date are the same for a single day
        # Also ensure "past_days" is not used or is 0 for forecast API
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": current_date.isoformat(),
            "end_date": current_date.isoformat(),
            "hourly": self.hourly_params,
            "timezone": "Asia/Kolkata",
        }
        try:
            response = requests.get(self.FORECAST_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(
                f"Error fetching forecast data for {latitude},{longitude} ({current_date}): {e}"
            )
            return None


# --- Utility Functions ---
def calculate_destination_point(
    start_lat, start_lon, distance_km, bearing_deg, earth_radius_km
):
    """
    Calculates the destination point (latitude, longitude) from a start point,
    distance, and bearing using the Haversine formula.
    """
    start_lat_rad = math.radians(start_lat)
    start_lon_rad = math.radians(start_lon)
    bearing_rad = math.radians(bearing_deg)

    angular_distance = distance_km / earth_radius_km

    dest_lat_rad = math.asin(
        math.sin(start_lat_rad) * math.cos(angular_distance)
        + math.cos(start_lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )

    dest_lon_rad = start_lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(start_lat_rad),
        math.cos(angular_distance) - math.sin(start_lat_rad) * math.sin(dest_lat_rad),
    )

    dest_lat_deg = math.degrees(dest_lat_rad)
    dest_lon_deg = math.degrees(dest_lon_rad)

    return (dest_lat_deg, dest_lon_deg)


def process_api_response_to_dataframe(
    api_data, region_name=None, latitude=None, longitude=None
):
    """
    Converts Open-Meteo API response to a Pandas DataFrame, adding region info if applicable.
    """
    if not api_data or not api_data.get("hourly") or not api_data["hourly"].get("time"):
        return pd.DataFrame()

    df = pd.DataFrame(api_data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop("time", axis=1, inplace=True)

    # Ensure all expected weather variables are present, filling with NaN if missing
    # This handles cases where some variables might not be returned for all days/locations
    global config  # Access the global config for weather_variables
    expected_cols = [col for col in config.weather_variables if col in df.columns]
    missing_cols = [col for col in config.weather_variables if col not in df.columns]
    for col in missing_cols:
        df[col] = pd.NA  # Or None, or np.nan depending on desired behavior

    if region_name and latitude is not None and longitude is not None:
        df["region_name"] = region_name
        df["latitude"] = latitude
        df["longitude"] = longitude

    # Reorder columns to match DB insertion order if needed, but psycopg2 works with tuple order
    return df


# --- Main Ingestion Logic ---
def main():
    global config  # Make config accessible in process_api_response_to_dataframe
    config = WeatherConfig()
    db_manager = DatabaseManager(config)
    api_fetcher = OpenMeteoAPI(config)

    # Calculate surrounding locations once
    surrounding_locations = []
    for i in range(config.num_regions):
        bearing = config.region_bearings[i]
        region_name = config.region_names[i]
        dest_lat, dest_lon = calculate_destination_point(
            config.delhi_latitude,
            config.delhi_longitude,
            config.radius_km,
            bearing,
            config.earth_radius_km,
        )
        surrounding_locations.append(
            {"latitude": dest_lat, "longitude": dest_lon, "name": region_name}
        )

    # Define the date range for data fetching
    # Start scraping from May 1, 2025 (as per user's request)
    start_ingestion_date = date(2025, 5, 27)
    # End scraping at the day before today for historical ERA5 data
    end_historical_date = date.today() - timedelta(days=1)
    # Today's date for current day forecast
    current_day_date = date.today()

    print(f"Starting data ingestion from {start_ingestion_date} to {current_day_date}.")

    all_delhi_data = []
    all_surrounding_data = []

    # --- Step 1: Fetch and store historical ERA5 data (bulk) ---
    if start_ingestion_date <= end_historical_date:
        print(
            f"\n--- Fetching Historical ERA5 Data ({start_ingestion_date} to {end_historical_date}) ---"
        )

        # Fetch Delhi historical data
        api_data_delhi_historical = api_fetcher.fetch_historical_era5(
            config.delhi_latitude,
            config.delhi_longitude,
            start_ingestion_date,
            end_historical_date,
        )
        df_delhi_historical = process_api_response_to_dataframe(
            api_data_delhi_historical
        )
        if not df_delhi_historical.empty:
            all_delhi_data.append(df_delhi_historical)

        # Fetch surrounding historical data in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=5
        ) as executor:  # Use max_workers for concurrency
            future_to_region = {
                executor.submit(
                    api_fetcher.fetch_historical_era5,
                    coords["latitude"],
                    coords["longitude"],
                    start_ingestion_date,
                    end_historical_date,
                ): coords
                for coords in surrounding_locations
            }

            for future in concurrent.futures.as_completed(future_to_region):
                coords = future_to_region[future]
                try:
                    api_data_surrounding_historical = future.result()
                    df_surrounding_historical = process_api_response_to_dataframe(
                        api_data_surrounding_historical,
                        region_name=coords["name"],
                        latitude=coords["latitude"],
                        longitude=coords["longitude"],
                    )
                    if not df_surrounding_historical.empty:
                        all_surrounding_data.append(df_surrounding_historical)
                except Exception as exc:
                    print(f"Region {coords['name']} generated an exception: {exc}")
        print("Finished fetching all historical ERA5 data.")
    else:
        print("No historical ERA5 data to fetch (start date is today or in future).")

    # --- Step 2: Fetch and store current day's forecast data ---
    print(
        f"\n--- Fetching Current Day's Data (Forecast API for {current_day_date}) ---"
    )

    # Fetch Delhi current day's data
    api_data_delhi_current = api_fetcher.fetch_current_day_forecast(
        config.delhi_latitude, config.delhi_longitude, current_day_date
    )
    df_delhi_current = process_api_response_to_dataframe(api_data_delhi_current)
    if not df_delhi_current.empty:
        all_delhi_data.append(df_delhi_current)

    # Fetch surrounding current day's data in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_region = {
            executor.submit(
                api_fetcher.fetch_current_day_forecast,
                coords["latitude"],
                coords["longitude"],
                current_day_date,
            ): coords
            for coords in surrounding_locations
        }

        for future in concurrent.futures.as_completed(future_to_region):
            coords = future_to_region[future]
            try:
                api_data_surrounding_current = future.result()
                # Ensure the global config is available for process_api_response_to_dataframe
                df_surrounding_current = process_api_response_to_dataframe(
                    api_data_surrounding_current,
                    region_name=coords["name"],
                    latitude=coords["latitude"],
                    longitude=coords["longitude"],
                )
                if not df_surrounding_current.empty:
                    all_surrounding_data.append(df_surrounding_current)
            except Exception as exc:
                print(
                    f"Region {coords['name']} (current day) generated an exception: {exc}"
                )
    print("Finished fetching current day's data.")

    # --- Step 3: Insert all collected data into the database (batch inserts) ---
    print("\n--- Inserting Collected Data into Database ---")
    if all_delhi_data:
        final_df_delhi = (
            pd.concat(all_delhi_data)
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )
        db_manager.insert_delhi_data(final_df_delhi)
    else:
        print("No Delhi data collected for insertion.")

    if all_surrounding_data:
        final_df_surrounding = (
            pd.concat(all_surrounding_data)
            .drop_duplicates(subset=["region_name", "timestamp"])
            .reset_index(drop=True)
        )
        db_manager.insert_surrounding_data(final_df_surrounding)
    else:
        print("No surrounding data collected for insertion.")

    print("\nData ingestion complete.")
    db_manager.close_connection()


if __name__ == "__main__":
    main()
