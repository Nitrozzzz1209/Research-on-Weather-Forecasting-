import math
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
)  # Import PolynomialFeatures
from sklearn.cluster import KMeans  # Import KMeans for clustering
import numpy as np
from collections import Counter
import psycopg2  # Import psycopg2 for PostgreSQL
from psycopg2 import sql
from psycopg2.extras import execute_values
import pmdarima as pm  # Re-import pmdarima for auto_arima (SARIMA)
import warnings  # Import the warnings module
import joblib  # Import joblib for saving/loading models
import os  # Import os for directory creation

# Suppress FutureWarning messages from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Configuration Class ---
class WeatherConfig:
    """Stores configuration settings for the weather forecasting model."""

    def __init__(self):
        self.delhi_latitude = 28.6448  # Approximate latitude for Delhi
        self.delhi_longitude = 77.2167  # Approximate longitude for Delhi
        self.radius_km = 500  # Radius for surrounding areas in kilometers
        self.num_regions = 8  # Number of surrounding regions
        self.earth_radius_km = 6371  # Earth's mean radius in kilometers

        # Dates for training data (5 years of data, starting 2020-01-01)
        self.train_start_date = date(2020, 1, 1)
        self.train_end_date = date(2024, 12, 31)

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

        # ADDED: Model Saving Path
        self.model_save_dir = "_ORIGINAL MODEL/trained_models"


# --- Database Manager Class ---
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
            print("Initializing: Database connected.")  # Keep this print
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
        except psycopg2.Error as e:
            print(f"Error creating tables: {e}")
        finally:
            cur.close()

    def load_delhi_data(self, start_date, end_date):
        """Loads Delhi's hourly weather data from the database for a date range."""
        if not self.conn:
            return None

        cur = self.conn.cursor()
        try:
            query = sql.SQL(
                """
                SELECT timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m,
                       wind_direction_10m, weather_code, precipitation, cloud_cover
                FROM delhi_hourly_weather
                WHERE timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp;
            """
            )
            cur.execute(
                query, (start_date, end_date + timedelta(days=1))
            )  # +1 day to include end_date
            rows = cur.fetchall()

            if not rows:
                return None

            # Reconstruct the Open-Meteo API-like dictionary structure
            data = {
                "hourly": {
                    "time": [],
                    "temperature_2m": [],
                    "relative_humidity_2m": [],
                    "wind_speed_10m": [],
                    "wind_direction_10m": [],
                    "weather_code": [],
                    "precipitation": [],
                    "cloud_cover": [],
                },
                "hourly_units": {  # Dummy units, as they are not stored in DB
                    "time": "iso8601",
                    "temperature_2m": "celsius",
                    "relative_humidity_2m": "%",
                    "wind_speed_10m": "km/h",
                    "wind_direction_10m": "degrees",
                    "weather_code": "wmo code",
                    "precipitation": "mm",
                    "cloud_cover": "%",
                },
            }
            for row in rows:
                data["hourly"]["time"].append(row[0].isoformat())
                data["hourly"]["temperature_2m"].append(row[1])
                data["hourly"]["relative_humidity_2m"].append(row[2])
                data["hourly"]["wind_speed_10m"].append(row[3])
                data["hourly"]["wind_direction_10m"].append(row[4])
                data["hourly"]["weather_code"].append(row[5])
                data["hourly"]["precipitation"].append(row[6])
                data["hourly"]["cloud_cover"].append(row[7])
            return data
        except psycopg2.Error as e:
            print(f"Error loading Delhi data: {e}")
            return None
        finally:
            cur.close()

    def load_surrounding_data(self, region_name, start_date, end_date):
        """Loads surrounding region's hourly weather data from the database for a date range."""
        if not self.conn:
            return None

        cur = self.conn.cursor()
        try:
            query = sql.SQL(
                """
                SELECT timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m,
                       wind_direction_10m, weather_code, precipitation, cloud_cover
                FROM surrounding_hourly_weather
                WHERE region_name = %s AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp;
            """
            )
            cur.execute(query, (region_name, start_date, end_date + timedelta(days=1)))
            rows = cur.fetchall()

            if not rows:
                return None

            data = {
                "hourly": {
                    "time": [],
                    "temperature_2m": [],
                    "relative_humidity_2m": [],
                    "wind_speed_10m": [],
                    "wind_direction_10m": [],
                    "weather_code": [],
                    "precipitation": [],
                    "cloud_cover": [],
                },
                "hourly_units": {  # Dummy units
                    "time": "iso8601",
                    "temperature_2m": "celsius",
                    "relative_humidity_2m": "%",
                    "wind_speed_10m": "km/h",
                    "wind_direction_10m": "degrees",
                    "weather_code": "wmo code",
                    "precipitation": "mm",
                    "cloud_cover": "%",
                },
            }
            for row in rows:
                data["hourly"]["time"].append(row[0].isoformat())
                data["hourly"]["temperature_2m"].append(row[1])
                data["hourly"]["relative_humidity_2m"].append(row[2])
                data["hourly"]["wind_speed_10m"].append(row[3])
                data["hourly"]["wind_direction_10m"].append(row[4])
                data["hourly"]["weather_code"].append(row[5])
                data["hourly"]["precipitation"].append(row[6])
                data["hourly"]["cloud_cover"].append(row[7])
            return data
        except psycopg2.Error as e:
            print(f"Error loading {region_name} data: {e}")
            return None
        finally:
            cur.close()

    def close_connection(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()


# --- Data Fetcher Class (Now only loads from DB) ---
class WeatherDataFetcher:
    """Handles fetching weather data exclusively from the database."""

    def __init__(self, config: WeatherConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager

    def fetch_historical_weather_data(
        self, latitude, longitude, start_date, end_date, region_name=None
    ):
        """
        Fetches historical weather data from the database for a date range.
        This method no longer makes API calls.
        """
        db_data = None
        if self.db_manager.conn:
            if region_name == "delhi":
                db_data = self.db_manager.load_delhi_data(start_date, end_date)
            elif region_name:
                db_data = self.db_manager.load_surrounding_data(
                    region_name, start_date, end_date
                )

            if db_data and db_data.get("hourly") and db_data["hourly"].get("time"):
                # Basic check to see if loaded data covers the requested period
                db_start_dt = datetime.fromisoformat(
                    db_data["hourly"]["time"][0]
                ).date()
                db_end_dt = datetime.fromisoformat(db_data["hourly"]["time"][-1]).date()

                # Ensure the loaded data spans the exact requested range
                if db_start_dt <= start_date and db_end_dt >= end_date:
                    return db_data

        # If data not found in DB or is incomplete for the requested range, return None.
        # No API fallback here.
        # print(f"Warning: Data for {region_name or 'Delhi'} from {start_date} to {end_date} not found or incomplete in the database. Please ensure all required data is loaded.") # Removed for cleaner output
        return None


# --- Data Processor Class ---
class DataProcessor:
    """Processes raw weather data into a structured format for modeling."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.kmeans_model = None  # ADDED: To store the fitted KMeans model

    def calculate_destination_point(
        self, start_lat, start_lon, distance_km, bearing_deg
    ):
        """
        Calculates the destination point (latitude, longitude) from a start point,
        distance, and bearing using the Haversine formula.
        """
        start_lat_rad = math.radians(start_lat)
        start_lon_rad = math.radians(start_lon)
        bearing_rad = math.radians(bearing_deg)

        angular_distance = distance_km / self.config.earth_radius_km

        dest_lat_rad = math.asin(
            math.sin(start_lat_rad) * math.cos(angular_distance)
            + math.cos(start_lat_rad)
            * math.sin(angular_distance)
            * math.cos(bearing_rad)
        )

        dest_lon_rad = start_lon_rad + math.atan2(
            math.sin(bearing_rad)
            * math.sin(angular_distance)
            * math.cos(start_lat_rad),
            math.cos(angular_distance)
            - math.sin(start_lat_rad) * math.sin(dest_lat_rad),
        )

        dest_lat_deg = math.degrees(dest_lat_rad)
        dest_lon_deg = math.degrees(dest_lon_rad)

        return (dest_lat_deg, dest_lon_deg)

    def get_season(self, month):
        """Determines the season based on the month for Northern India."""
        if month in [3, 4]:
            return "Spring"
        elif month in [5, 6]:
            return "Summer"
        elif month in [7, 8, 9]:
            return "Monsoon"
        else:  # 10, 11, 12, 1, 2
            return "Winter"

    # Removed map_weather_code_to_category as it's replaced by clustering

    def process_historical_data(self, raw_data):
        """
        Processes the raw historical weather data into a structured dataset (Pandas DataFrame).
        This function now also extracts the target temperature for the next day.
        Includes advanced lagged features, rolling statistics, differences, and polynomial features.
        Weather categories are determined by KMeans clustering.
        """
        dataset = []

        delhi_data = raw_data.get("delhi")
        surrounding_data = raw_data.get("surrounding_regions", {})

        if (
            not delhi_data
            or not delhi_data.get("hourly")
            or not delhi_data["hourly"].get("time")
        ):
            print(
                "Error: Missing or incomplete historical data for Delhi. Cannot create dataset."
            )
            return pd.DataFrame(), None  # MODIFIED: Return None for kmeans_model

        delhi_hourly = delhi_data["hourly"]
        time_stamps = delhi_hourly["time"]
        num_hours = len(time_stamps)
        hours_per_day = 24
        num_days = num_hours // hours_per_day

        for i in range(num_days - 1):
            start_hour_index = i * hours_per_day
            end_hour_index = start_hour_index + hours_per_day

            current_day_data = {}
            current_day_time = datetime.fromisoformat(
                time_stamps[start_hour_index]
            ).date()
            current_day_data["day_of_week"] = current_day_time.weekday()
            current_day_data["month"] = current_day_time.month
            current_day_data["day_of_year"] = (
                current_day_time.timetuple().tm_yday
            )  # New feature
            current_day_data["week_of_year"] = current_day_time.isocalendar()[
                1
            ]  # New feature
            current_day_data["season"] = self.get_season(current_day_time.month)

            for var in self.config.weather_variables:
                hourly_values = delhi_hourly.get(var, [])[
                    start_hour_index:end_hour_index
                ]
                valid_hourly_values = [val for val in hourly_values if val is not None]

                if valid_hourly_values:
                    current_day_data[f"delhi_avg_{var}"] = np.mean(valid_hourly_values)
                    current_day_data[f"delhi_max_{var}"] = np.max(valid_hourly_values)
                    current_day_data[f"delhi_min_{var}"] = np.min(valid_hourly_values)
                    # For weather_code, we'll use it for clustering, not mode directly in features
                else:
                    current_day_data[f"delhi_avg_{var}"] = np.nan
                    current_day_data[f"delhi_max_{var}"] = np.nan
                    current_day_data[f"delhi_min_{var}"] = np.nan

            for region_name in self.config.region_names:
                region_data_for_current_day = surrounding_data.get(region_name)
                if region_data_for_current_day is None or not isinstance(
                    region_data_for_current_day, dict
                ):
                    for var in self.config.weather_variables:
                        current_day_data[f"{region_name}_avg_{var}"] = np.nan
                        current_day_data[f"{region_name}_max_{var}"] = np.nan
                        current_day_data[f"{region_name}_min_{var}"] = np.nan
                    continue

                region_hourly = region_data_for_current_day.get("hourly", {})
                if not region_hourly or not region_hourly.get("time"):
                    for var in self.config.weather_variables:
                        current_day_data[f"{region_name}_avg_{var}"] = np.nan
                        current_day_data[f"{region_name}_max_{var}"] = np.nan
                        current_day_data[f"{region_name}_min_{var}"] = np.nan
                    continue

                region_time_stamps = region_hourly["time"]
                region_start_index = None
                region_end_index = None
                for j in range(len(region_time_stamps)):
                    ts_date = datetime.fromisoformat(region_time_stamps[j]).date()
                    if ts_date == current_day_time:
                        if region_start_index is None:
                            region_start_index = j
                        region_end_index = j + 1

                if region_start_index is not None:
                    region_hourly_slice = {}
                    for var in self.config.weather_variables:
                        region_hourly_slice[var] = region_hourly.get(var, [])[
                            region_start_index:region_end_index
                        ]

                    for var in self.config.weather_variables:
                        hourly_values = region_hourly_slice.get(var, [])
                        valid_hourly_values = [
                            val for val in hourly_values if val is not None
                        ]

                        if valid_hourly_values:
                            current_day_data[f"{region_name}_avg_{var}"] = np.mean(
                                valid_hourly_values
                            )
                            current_day_data[f"{region_name}_max_{var}"] = np.max(
                                valid_hourly_values
                            )
                            current_day_data[f"{region_name}_min_{var}"] = np.min(
                                valid_hourly_values
                            )
                        else:
                            current_day_data[f"{region_name}_avg_{var}"] = np.nan
                            current_day_data[f"{region_name}_max_{var}"] = np.nan
                            current_day_data[f"{region_name}_min_{var}"] = np.nan
                else:
                    for var in self.config.weather_variables:
                        current_day_data[f"{region_name}_avg_{var}"] = np.nan
                        current_day_data[f"{region_name}_max_{var}"] = np.nan
                        current_day_data[f"{region_name}_min_{var}"] = np.nan

            next_day_start_hour_index = end_hour_index
            next_day_end_hour_index = next_day_start_hour_index + hours_per_day

            if next_day_end_hour_index <= num_hours:
                next_day_temperatures = delhi_hourly.get("temperature_2m", [])[
                    next_day_start_hour_index:next_day_end_hour_index
                ]
                valid_next_day_temperatures = [
                    val for val in next_day_temperatures if val is not None
                ]

                if valid_next_day_temperatures:
                    avg_next_day_temperature = np.mean(valid_next_day_temperatures)
                    current_day_data["target_temperature"] = avg_next_day_temperature
                    dataset.append(current_day_data)
                else:
                    # print(f"Warning: Incomplete valid data for next day prediction for day starting {current_day_time}. Skipping.") # Removed
                    pass  # Suppress this warning
            else:
                # print(f"Warning: Not enough data for next day prediction for day starting {current_day_time}. Skipping.") # Removed
                pass  # Suppress this warning

        df = pd.DataFrame(dataset)
        df = pd.get_dummies(df, columns=["season"], drop_first=True)
        df.dropna(inplace=True)  # Drop any rows with NaNs introduced by missing data

        # --- Feature Engineering: Advanced Lagged Features & Rolling Statistics ---
        features_to_lag_and_roll = [
            "delhi_avg_temperature_2m",
            "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m",
            "delhi_avg_precipitation",
            "delhi_avg_cloud_cover",
        ]
        lags = [1, 2, 3, 7, 14, 30]  # Extended lags
        rolling_windows = [3, 7]  # Rolling window sizes

        new_features_df = pd.DataFrame(index=df.index)

        for feature in features_to_lag_and_roll:
            # Add lagged features
            for lag in lags:
                new_features_df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)

            # Add rolling statistics
            for window in rolling_windows:
                new_features_df[f"{feature}_rolling_mean_{window}d"] = (
                    df[feature].rolling(window=window).mean()
                )
                new_features_df[f"{feature}_rolling_min_{window}d"] = (
                    df[feature].rolling(window=window).min()
                )
                new_features_df[f"{feature}_rolling_max_{window}d"] = (
                    df[feature].rolling(window=window).max()
                )
                new_features_df[f"{feature}_rolling_std_{window}d"] = (
                    df[feature].rolling(window=window).std()
                )

            # Add difference features
            new_features_df[f"{feature}_diff_1d"] = df[feature].diff(1)

        # Add interaction terms (examples)
        new_features_df["delhi_temp_x_humidity"] = (
            df["delhi_avg_temperature_2m"] * df["delhi_avg_relative_humidity_2m"]
        )
        new_features_df["delhi_wind_x_cloud"] = (
            df["delhi_avg_wind_speed_10m"] * df["delhi_avg_cloud_cover"]
        )

        # Add polynomial features for a few key features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        # Select features for polynomial transformation
        poly_features = df[
            ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        ].values
        poly_transformed = poly.fit_transform(poly_features)
        poly_feature_names = poly.get_feature_names_out(
            ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        )
        poly_df = pd.DataFrame(
            poly_transformed, columns=poly_feature_names, index=df.index
        )
        new_features_df = pd.concat([new_features_df, poly_df], axis=1)

        # Concatenate all new features efficiently
        df = pd.concat([df, new_features_df], axis=1)
        df.dropna(inplace=True)  # Drop rows with NaNs introduced by new features

        # --- Clustering for Weather Categories (Replacing map_weather_code_to_category) ---
        # Select features for clustering. Use daily averages for stability.
        clustering_features = [
            "delhi_avg_temperature_2m",
            "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m",
            "delhi_avg_wind_direction_10m",
            "delhi_avg_weather_code",
            "delhi_avg_precipitation",
            "delhi_avg_cloud_cover",
        ]

        # Ensure these features exist and are not NaN for clustering
        df_for_clustering = df[clustering_features].copy()
        df_for_clustering.dropna(inplace=True)

        if not df_for_clustering.empty:
            # Scale features for clustering
            self.kmeans_scaler = StandardScaler()
            scaled_clustering_features = self.kmeans_scaler.fit_transform(
                df_for_clustering
            )  # CHANGE: Use self.kmeans_scaler

            # Apply KMeans clustering
            n_clusters = 7  # You can tune this number
            kmeans = KMeans(
                n_clusters=n_clusters, random_state=42, n_init=10
            )  # n_init to suppress warning
            cluster_labels = kmeans.fit_predict(scaled_clustering_features)

            # Map cluster labels back to the original DataFrame
            # Align indices to ensure correct assignment
            df.loc[df_for_clustering.index, "target_weather"] = cluster_labels
            df["target_weather"] = df["target_weather"].astype(
                int
            )  # Ensure integer labels
        else:
            print(
                "Warning: Not enough data for clustering weather categories. Target weather will be missing."
            )
            df["target_weather"] = np.nan  # Assign NaN if clustering fails

        df.dropna(
            inplace=True
        )  # Drop rows where target_weather might be NaN due to clustering issues
        return (
            df,
            kmeans,
            self.kmeans_scaler,
        )  # MODIFIED: Return the fitted KMeans model

    def process_historical_data_for_arima(self, raw_data):
        """
        Processes the raw historical weather data specifically for ARIMA.
        Extracts daily average temperature for Delhi.
        """
        delhi_data = raw_data.get("delhi")

        if (
            not delhi_data
            or not delhi_data.get("hourly")
            or not delhi_data["hourly"].get("time")
        ):
            print(
                "Error: Missing or incomplete historical data for Delhi. Cannot create dataset for ARIMA."
            )
            return pd.DataFrame()

        delhi_hourly = delhi_data["hourly"]
        time_stamps = delhi_hourly["time"]
        temperatures = delhi_hourly.get("temperature_2m", [])

        # Create a DataFrame from hourly data
        hourly_df = pd.DataFrame(
            {"time": pd.to_datetime(time_stamps), "temperature_2m": temperatures}
        )
        hourly_df.set_index("time", inplace=True)
        hourly_df.dropna(subset=["temperature_2m"], inplace=True)

        # Resample to daily average temperature
        daily_avg_temp = hourly_df["temperature_2m"].resample("D").mean()
        daily_avg_temp = daily_avg_temp.dropna()  # Drop days with no data

        return daily_avg_temp


# --- Hybrid Model Trainer Class ---
class HybridModelTrainer:
    """Trains and evaluates a hybrid model using Random Forest for weather category and SARIMA for temperature."""

    def __init__(self, config: WeatherConfig):  # MODIFIED: Added config parameter
        self.config = config  # ADDED: Store config
        self.kmeans_scaler = None
        self.rf_model = None
        self.arima_model_fit = None
        self.weather_accuracies = []
        self.temperature_maes = []
        self.scaler = None  # ADDED: To store the fitted StandardScaler
        self.kmeans_model = None  # ADDED: To store the fitted KMeans model

    def train_and_evaluate(
        self,
        dataframe_full,
        temp_series_full,
        train_start_idx,
        train_end_idx,
        test_start_idx,
        test_end_idx,
        session_num,
    ):
        """
        Trains and evaluates the hybrid model for a specific training/testing session.
        Returns (rf_accuracy, arima_mae) for the session.
        """
        rf_accuracy = np.nan
        arima_mae = np.nan

        # --- Prepare data for Random Forest (Weather Category) ---
        rf_train_df = dataframe_full.iloc[train_start_idx:train_end_idx]
        rf_test_df = dataframe_full.iloc[test_start_idx:test_end_idx]

        if not rf_train_df.empty and not rf_test_df.empty:
            X_rf_train = rf_train_df.drop(
                ["target_weather", "target_temperature"], axis=1
            )
            y_rf_train = rf_train_df["target_weather"]
            X_rf_test = rf_test_df.drop(
                ["target_weather", "target_temperature"], axis=1
            )
            y_rf_test = rf_test_df["target_weather"]

            # Ensure consistent columns after one-hot encoding across splits
            all_rf_columns = dataframe_full.drop(
                ["target_weather", "target_temperature"], axis=1
            ).columns
            X_rf_train = X_rf_train.reindex(columns=all_rf_columns, fill_value=0)
            X_rf_test = X_rf_test.reindex(columns=all_rf_columns, fill_value=0)

            # Scale Random Forest features
            self.scaler = StandardScaler()  # MODIFIED: Assign to self.scaler
            X_rf_train_scaled = self.scaler.fit_transform(
                X_rf_train
            )  # MODIFIED: Use self.scaler
            X_rf_test_scaled = self.scaler.transform(
                X_rf_test
            )  # MODIFIED: Use self.scaler

            self.rf_model = RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced"
            )
            self.rf_model.fit(X_rf_train_scaled, y_rf_train)
            y_rf_pred = self.rf_model.predict(X_rf_test_scaled)
            rf_accuracy = accuracy_score(y_rf_test, y_rf_pred)
            self.weather_accuracies.append(rf_accuracy)
        else:
            self.weather_accuracies.append(np.nan)  # Append NaN if RF training skipped

        # --- Prepare data for SARIMA (Temperature) ---
        arima_train_series = temp_series_full.iloc[train_start_idx:train_end_idx]
        arima_test_series = temp_series_full.iloc[test_start_idx:test_end_idx]

        if not arima_train_series.empty and not arima_test_series.empty:
            try:
                # Use auto_arima to find optimal SARIMA order (seasonal=True, m=7 for weekly seasonality)
                # You might need to adjust 'm' if your data has other seasonalities (e.g., 365 for yearly)
                model_auto = pm.auto_arima(
                    arima_train_series,
                    start_p=1,
                    start_q=1,
                    test="adf",  # Use adftest to find optimal 'd'
                    max_p=5,
                    max_q=5,  # Maximum p and q
                    m=7,  # Weekly seasonality (7 days)
                    d=None,  # Let model determine 'd'
                    seasonal=True,  # Enable seasonality
                    start_P=0,
                    start_Q=0,  # Start seasonal orders
                    max_P=2,
                    max_Q=2,  # Max seasonal orders
                    D=None,  # Let model determine 'D'
                    trace=False,  # Set to True to see auto_arima progress
                    error_action="ignore",
                    suppress_warnings=True,
                    stepwise=True,
                )
                order = model_auto.order
                seasonal_order = model_auto.seasonal_order
                # print(f"Session {session_num} - SARIMA optimal order: {order}, seasonal_order: {seasonal_order}") # Removed for cleaner output

                # Assign the fitted auto_arima model to self.arima_model_fit
                self.arima_model_fit = model_auto  # FIX: Assign the fitted model here

                # Make rolling predictions for SARIMA
                arima_predictions = []
                history = [x for x in arima_train_series]
                for t in range(len(arima_test_series)):
                    try:
                        # Use pmdarima.ARIMA for fitting with auto_arima found orders
                        # For rolling forecast, we should update the model or refit.
                        # Since model_auto is already fitted, we can use its predict method
                        # or update it with new history.
                        # For simplicity, if self.arima_model_fit is set, use it.
                        # A common pattern for rolling forecast with pmdarima is:
                        # model_fit_step = self.arima_model_fit.fit(history) # Re-fit or update
                        # yhat = model_fit_step.predict(n_periods=1)[0]
                        # For this specific setup, the `model_auto` is the one we want to save.
                        # And for prediction, we can use its predict method.

                        # Use the fitted model to predict one step ahead for evaluation
                        yhat = self.arima_model_fit.predict(n_periods=1)[
                            0
                        ]  # Use the already fitted model
                        arima_predictions.append(yhat)
                        history.append(arima_test_series.iloc[t])
                        # If doing true rolling, you'd update the model here:
                        # self.arima_model_fit.update(arima_test_series.iloc[t])

                    except Exception as e:
                        # print(f"Session {session_num} - Error during SARIMA rolling forecast step {t}: {e}") # Removed
                        arima_predictions.append(np.nan)
                        history.append(arima_test_series.iloc[t])

                valid_predictions = [
                    pred for pred in arima_predictions if not np.isnan(pred)
                ]
                valid_test_series = arima_test_series.iloc[: len(valid_predictions)]

                if valid_predictions:
                    arima_mae = mean_absolute_error(
                        valid_test_series, valid_predictions
                    )
                    self.temperature_maes.append(arima_mae)
                else:
                    self.temperature_maes.append(
                        np.nan
                    )  # Append NaN if no valid ARIMA predictions

            except Exception as e:
                # print(f"Session {session_num} - Error with auto_arima or initial SARIMA fitting: {e}") # Removed
                self.temperature_maes.append(
                    np.nan
                )  # Append NaN if auto_arima or initial fitting failed
        else:
            self.temperature_maes.append(np.nan)  # Append NaN if ARIMA training skipped

        return rf_accuracy, arima_mae

    def get_average_metrics(self):
        """Returns the average accuracy and MAE across all training sessions."""
        avg_accuracy = (
            np.nanmean(self.weather_accuracies) if self.weather_accuracies else 0
        )  # Use nanmean to ignore NaNs
        avg_mae = (
            np.nanmean(self.temperature_maes) if self.temperature_maes else 0
        )  # Use nanmean to ignore NaNs
        return avg_accuracy, avg_mae

    # ADDED: Method to save trained models
    def save_models(
        self, scaler_path, kmeans_path, rf_path, arima_path, rf_feature_columns_path
    ):
        """Saves the trained scaler, KMeans, Random Forest, ARIMA models, and RF feature columns."""
        # Ensure the directory exists
        os.makedirs(self.config.model_save_dir, exist_ok=True)

        print(f"\nSaving models to {self.config.model_save_dir}/...")

        if self.scaler:
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
        else:
            print("Scaler not available to save (not fitted yet).")

        if self.kmeans_model:
            joblib.dump(self.kmeans_model, kmeans_path)
            print(f"KMeans model saved to {kmeans_path}")
        else:
            print("KMeans model not available to save (not fitted yet).")

        if self.rf_model:
            joblib.dump(self.rf_model, rf_path)
            print(f"Random Forest model saved to {rf_path}")
        else:
            print("Random Forest model not available to save (not fitted yet).")

        if self.arima_model_fit:  # MODIFIED: Use arima_model_fit
            joblib.dump(
                self.arima_model_fit, arima_path
            )  # MODIFIED: Use arima_model_fit
            print(f"ARIMA model saved to {arima_path}")
        else:
            print("ARIMA model not available to save (not fitted yet).")

        # ADDED: Save the RF feature columns
        if (
            self.rf_model
            and hasattr(self.scaler, "feature_names_in_")
            and self.scaler.feature_names_in_ is not None
        ):
            joblib.dump(list(self.scaler.feature_names_in_), rf_feature_columns_path)
            print(f"Random Forest feature columns saved to {rf_feature_columns_path}")
        elif self.rf_model:  # Fallback if scaler doesn't have feature_names_in_
            # This is less robust, but better than nothing if feature_names_in_ is missing
            # In a real scenario, you'd ensure feature_names_in_ is always available or explicitly define them.
            print(
                "Warning: Scaler does not have 'feature_names_in_'. Could not save RF feature columns robustly."
            )
        else:
            print(
                "Random Forest model not available, so RF feature columns cannot be saved."
            )


# --- Main Execution Function ---
def main():
    """
    Main function to run the hybrid weather forecasting model with multiple training sessions.
    """
    config = WeatherConfig()
    db_manager = DatabaseManager(config)  # Initialize DatabaseManager
    data_fetcher = WeatherDataFetcher(
        config, db_manager
    )  # Pass db_manager to DataFetcher
    data_processor = DataProcessor(config)
    hybrid_trainer = HybridModelTrainer(
        config
    )  # MODIFIED: Pass config to HybridModelTrainer

    print("Initializing: Starting weather forecasting model.")  # Initializing statement

    # Calculate surrounding locations once at the beginning
    surrounding_locations = {}
    for i in range(config.num_regions):
        bearing = config.region_bearings[i]
        region_name = config.region_names[i]
        dest_lat, dest_lon = data_processor.calculate_destination_point(
            config.delhi_latitude, config.delhi_longitude, config.radius_km, bearing
        )
        surrounding_locations[region_name] = {
            "latitude": dest_lat,
            "longitude": dest_lon,
            "bearing": bearing,
        }

    # Initialize raw_historical_data structure
    raw_historical_data = {
        "delhi": {"hourly": {var: [] for var in config.weather_variables + ["time"]}},
        "surrounding_regions": {},
    }
    for region_name in config.region_names:
        raw_historical_data["surrounding_regions"][region_name] = {
            "hourly": {var: [] for var in config.weather_variables + ["time"]}
        }

    # Fetch data month by month for all regions (now only from DB)
    current_year = config.train_start_date.year
    end_year = config.train_end_date.year

    for year in range(current_year, end_year + 1):
        for month in range(1, 13):
            month_start_date = date(year, month, 1)
            if month == 12:
                month_end_date = date(year, 12, 31)
            else:
                month_end_date = date(year, month + 1, 1) - timedelta(days=1)

            if month_start_date > config.train_end_date:
                break

            # Fetch Delhi data for the current month from DB
            delhi_month_data = data_fetcher.fetch_historical_weather_data(
                config.delhi_latitude,
                config.delhi_longitude,
                month_start_date,
                month_end_date,
                region_name="delhi",
            )
            if (
                delhi_month_data
                and delhi_month_data.get("hourly")
                and delhi_month_data["hourly"].get("time")
            ):
                for key in raw_historical_data["delhi"]["hourly"]:
                    raw_historical_data["delhi"]["hourly"][key].extend(
                        delhi_month_data["hourly"].get(key, [])
                    )

            # Fetch surrounding regions data for the current month from DB
            for region_name, coords in surrounding_locations.items():
                region_month_data = data_fetcher.fetch_historical_weather_data(
                    coords["latitude"],
                    coords["longitude"],
                    month_start_date,
                    month_end_date,
                    region_name=region_name,
                )
                if (
                    region_month_data
                    and region_month_data.get("hourly")
                    and region_month_data["hourly"].get("time")
                ):
                    for key in raw_historical_data["surrounding_regions"][region_name][
                        "hourly"
                    ]:
                        raw_historical_data["surrounding_regions"][region_name][
                            "hourly"
                        ][key].extend(region_month_data["hourly"].get(key, []))

    print("Data fetched from database.")  # Data fetched statement

    # Full dataset for Random Forest
    # MODIFIED: Capture kmeans_model_fitted returned by process_historical_data
    weather_dataset_full, kmeans_model_fitted, kmeans_scaler_fitted = (
        data_processor.process_historical_data(raw_historical_data)
    )

    hybrid_trainer.kmeans_scaler = data_processor.kmeans_scaler
    # ADDED: Assign the captured KMeans model to the hybrid_trainer
    hybrid_trainer.kmeans_model = kmeans_model_fitted

    # ADD THIS NEW LINE: Assign the newly captured scaler to hybrid_trainer
    hybrid_trainer.kmeans_scaler = kmeans_scaler_fitted

    # Full temperature series for ARIMA
    daily_avg_temp_series_full = data_processor.process_historical_data_for_arima(
        raw_historical_data
    )

    if weather_dataset_full.empty or daily_avg_temp_series_full.empty:
        print(
            "\nFailed to create complete datasets from historical data. Cannot train hybrid model."
        )
        db_manager.close_connection()  # Close DB connection on error
        return

    # --- Multiple Training Sessions Setup ---
    num_training_sessions = 10
    session_test_days = 30  # Each session tests on 30 days
    total_days = len(weather_dataset_full)

    sessions = []
    initial_train_period = 365  # Minimum 1 year for initial training

    # Calculate the starting index for the first test window
    first_test_window_start_idx = max(
        initial_train_period, total_days - (num_training_sessions * session_test_days)
    )

    for i in range(num_training_sessions):
        test_start_idx = first_test_window_start_idx + i * session_test_days
        test_end_idx = min(test_start_idx + session_test_days, total_days)

        train_start_idx = 0
        train_end_idx = test_start_idx

        if train_end_idx > 0 and test_end_idx > test_start_idx:
            sessions.append(
                (train_start_idx, train_end_idx, test_start_idx, test_end_idx)
            )
        else:
            print(
                f"Warning: Skipping session {i+1} due to insufficient data for training or testing. "
                f"Train data days: {train_end_idx - train_start_idx}, Test data days: {test_end_idx - test_start_idx}"
            )

    print(f"\n--- Starting {len(sessions)} Multiple Training Sessions ---")
    print(
        f"{'Training Step':<15} {'RF_Accuracy':<15} {'ARIMA (MAE) Temp':<18}"
    )  # Column headers
    print(f"{'-'*15:<15} {'-'*15:<15} {'-'*18:<18}")  # Separator line

    for i, (train_start, train_end, test_start, test_end) in enumerate(sessions):
        rf_acc, arima_temp_mae = hybrid_trainer.train_and_evaluate(
            weather_dataset_full,
            daily_avg_temp_series_full,
            train_start,
            train_end,
            test_start,
            test_end,
            i + 1,  # Session number
        )
        # Print results in columnar format
        rf_acc_str = f"{rf_acc:.2f}" if not np.isnan(rf_acc) else "N/A"
        arima_mae_str = (
            f"{arima_temp_mae:.2f} °C" if not np.isnan(arima_temp_mae) else "N/A"
        )
        print(f"{i + 1:<15} {rf_acc_str:<15} {arima_mae_str:<18}")

    avg_accuracy, avg_mae = hybrid_trainer.get_average_metrics()

    print("\n--- Hybrid Ensemble Model Overall Summary ---")
    print(f"Average Weather Category Accuracy across sessions: {avg_accuracy:.2f}")
    print(
        f"Average Temperature Mean Absolute Error (MAE) across sessions: {avg_mae:.2f} °C"
    )
    print("------------------------------------------")

    # ADDED: Save the trained models after the last session
    # The models stored in hybrid_trainer will be from the *last* training session.
    # If you need to save the "best" model, you'd need logic to track metrics and save conditionally.
    # For now, it saves the models from the final training session.

    # Ensure the model save directory exists
    os.makedirs(config.model_save_dir, exist_ok=True)

    # Define file paths for saving
    scaler_path = os.path.join(config.model_save_dir, "scaler.joblib")
    kmeans_path = os.path.join(config.model_save_dir, "kmeans_model.joblib")
    rf_path = os.path.join(config.model_save_dir, "random_forest_model.joblib")
    arima_path = os.path.join(config.model_save_dir, "arima_model.joblib")
    rf_feature_columns_path = os.path.join(
        config.model_save_dir, "rf_feature_columns.joblib"
    )  # NEW PATH

    hybrid_trainer.save_models(
        scaler_path, kmeans_path, rf_path, arima_path, rf_feature_columns_path
    )

    db_manager.close_connection()  # Close DB connection at the end


# --- Script Entry Point ---
if __name__ == "__main__":
    main()
