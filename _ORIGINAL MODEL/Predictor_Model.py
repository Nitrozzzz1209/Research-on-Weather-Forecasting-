import math
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import pmdarima as pm
import warnings
import joblib
import os

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
        return None


# --- Data Processor Class ---
class DataProcessor:
    """Processes raw weather data into a structured format for modeling."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.kmeans_model = None  # ADDED: To store the fitted KMeans model
        self.kmeans_scaler = None # NEW: To store the fitted KMeans scaler

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

    def process_historical_data(self, raw_data, kmeans_model_for_prediction=None, kmeans_scaler_for_prediction=None):
        """
        Processes the raw historical weather data into a structured format (Pandas DataFrame).
        This function now also extracts the target temperature for the next day.
        Includes advanced lagged features, rolling statistics, differences, and polynomial features.
        Weather categories are determined by KMeans clustering.
        If kmeans_model_for_prediction is provided, it uses this pre-fitted model for clustering.
        If kmeans_scaler_for_prediction is provided, it uses this scaler for transforming clustering features.
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
            return pd.DataFrame(), None

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
            )
            current_day_data["week_of_year"] = current_day_time.isocalendar()[1]
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
                    pass
            else:
                pass

        df = pd.DataFrame(dataset)
        df = pd.get_dummies(df, columns=["season"], drop_first=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True) # Ensure a clean, unique integer index for main df

        # --- Feature Engineering: Advanced Lagged Features & Rolling Statistics ---
        features_to_lag_and_roll = [
            "delhi_avg_temperature_2m",
            "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m",
            "delhi_avg_precipitation",
            "delhi_avg_cloud_cover",
        ]
        lags = [1, 2, 3, 7, 14, 30]
        rolling_windows = [3, 7]

        new_features_df = pd.DataFrame(index=df.index)

        for feature in features_to_lag_and_roll:
            for lag in lags:
                new_features_df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)

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

            new_features_df[f"{feature}_diff_1d"] = df[feature].diff(1)

        new_features_df["delhi_temp_x_humidity"] = (
            df["delhi_avg_temperature_2m"] * df["delhi_avg_relative_humidity_2m"]
        )
        new_features_df["delhi_wind_x_cloud"] = (
            df["delhi_avg_wind_speed_10m"] * df["delhi_avg_cloud_cover"]
        )

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_input_df = df[
            ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        ].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # Handle potential NaNs in poly_input_df before fitting/transforming
        poly_input_df.fillna(poly_input_df.mean(), inplace=True) # Fallback for NaNs

        poly_transformed = poly.fit_transform(poly_input_df)
        poly_feature_names = poly.get_feature_names_out(
            ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        )
        
        # Create full poly_df, then drop the original features to avoid duplication
        full_poly_df = pd.DataFrame(
            poly_transformed, columns=poly_feature_names, index=df.index
        )
        original_features_to_drop = ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        poly_df = full_poly_df.drop(columns=original_features_to_drop, errors='ignore') # Drop original features if they exist

        new_features_df = pd.concat([new_features_df, poly_df], axis=1)

        df = pd.concat([df, new_features_df], axis=1)
        # Safeguard: Remove any duplicate columns that might have been introduced
        df = df.loc[:, ~df.columns.duplicated()]
        df.dropna(inplace=True)

        print(f"DEBUG: Columns of df after feature engineering and dropping duplicates: {df.columns.tolist()}")

        # --- Clustering for Weather Categories ---
        # IMPORTANT: This list must match the features the KMeans model was trained on (7 features)
        clustering_features = [
            "delhi_avg_temperature_2m",
            "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m",
            "delhi_avg_wind_direction_10m", # Added missing feature
            "delhi_avg_weather_code",       # Added missing feature
            "delhi_avg_precipitation",
            "delhi_avg_cloud_cover",
        ]

        df_for_clustering = df[clustering_features].copy()
        
        # Store the original index of df_for_clustering before dropping NaNs
        original_indices_for_clustering = df_for_clustering.index

        df_for_clustering.dropna(inplace=True)

        # Get the indices of the rows that *remained* after dropping NaNs
        valid_original_indices = df_for_clustering.index

        # Now, reset the index of df_for_clustering to a simple RangeIndex for internal consistency
        df_for_clustering.reset_index(drop=True, inplace=True)

        if not df_for_clustering.empty:
            if kmeans_model_for_prediction and kmeans_scaler_for_prediction:
                # Print statements for debugging
                print(f"DEBUG: df_for_clustering index unique: {df_for_clustering.index.is_unique}")
                print(f"DEBUG: df_for_clustering columns before reindex: {df_for_clustering.columns.tolist()}")
                print(f"DEBUG: kmeans_scaler_for_prediction.feature_names_in_: {kmeans_scaler_for_prediction.feature_names_in_.tolist()}")
                print(f"DEBUG: kmeans_model_for_prediction.n_features_in_: {kmeans_model_for_prediction.n_features_in_}")

                # Use the potentially newly trained or original consistent models
                # The reindex is now handled by the _train_models_if_needed logic which ensures consistency
                # or by the main process_historical_data when training from scratch.
                # Here, we assume kmeans_scaler_for_prediction.feature_names_in_ is correct.
                df_for_clustering = df_for_clustering.reindex(columns=kmeans_scaler_for_prediction.feature_names_in_, fill_value=0)

                scaled_clustering_features = kmeans_scaler_for_prediction.transform(df_for_clustering)
                cluster_labels = kmeans_model_for_prediction.predict(scaled_clustering_features)
                
                # Assign cluster labels back to the original 'df' using the 'valid_original_indices'
                df.loc[valid_original_indices, "target_weather"] = cluster_labels
                df["target_weather"] = df["target_weather"].astype(int)
            else:
                # This block runs during initial training
                scaler_cluster = StandardScaler()
                scaled_clustering_features = scaler_cluster.fit_transform(df_for_clustering)
                n_clusters = 7
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_clustering_features)

                # Assign cluster labels back to the original 'df' using the 'valid_original_indices'
                df.loc[valid_original_indices, "target_weather"] = cluster_labels
                df["target_weather"] = df["target_weather"].astype(int)
                self.kmeans_model = kmeans
                self.kmeans_scaler = scaler_cluster # Store the fitted scaler for KMeans


        else:
            print(
                "Warning: Not enough data for clustering weather categories. Target weather will be missing."
            )
            df["target_weather"] = np.nan

        df.dropna(
            inplace=True
        )
        return df, self.kmeans_model if not kmeans_model_for_prediction else kmeans_model_for_prediction


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
        daily_avg_temp = daily_avg_temp.dropna()

        return daily_avg_temp


# --- Hybrid Model Trainer Class ---
class HybridModelTrainer:
    """Trains and evaluates a hybrid model using Random Forest for weather category and SARIMA for temperature."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.rf_model = None
        self.arima_model_fit = None
        self.weather_accuracies = []
        self.temperature_maes = []
        self.scaler = None # Main scaler for RF features
        self.kmeans_model = None
        self.kmeans_scaler = None # NEW: Scaler specifically for KMeans features


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
            self.scaler = StandardScaler()
            X_rf_train_scaled = self.scaler.fit_transform(X_rf_train)
            X_rf_test_scaled = self.scaler.transform(X_rf_test)

            self.rf_model = RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced"
            )
            self.rf_model.fit(X_rf_train_scaled, y_rf_train)
            y_rf_pred = self.rf_model.predict(X_rf_test_scaled)
            rf_accuracy = accuracy_score(y_rf_test, y_rf_pred)
            self.weather_accuracies.append(rf_accuracy)
        else:
            self.weather_accuracies.append(np.nan)

        # --- Prepare data for SARIMA (Temperature) ---
        arima_train_series = temp_series_full.iloc[train_start_idx:train_end_idx]
        arima_test_series = temp_series_full.iloc[test_start_idx:test_end_idx]

        if not arima_train_series.empty and not arima_test_series.empty:
            try:
                model_auto = pm.auto_arima(
                    arima_train_series,
                    start_p=1,
                    start_q=1,
                    test="adf",
                    max_p=5,
                    max_q=5,
                    m=7,
                    d=None,
                    seasonal=True,
                    start_P=0,
                    start_Q=0,
                    max_P=2,
                    max_Q=2,
                    D=None,
                    trace=False,
                    error_action="ignore",
                    suppress_warnings=True,
                    stepwise=True,
                )
                self.arima_model_fit = model_auto

                arima_predictions = []
                history = [x for x in arima_train_series]
                for t in range(len(arima_test_series)):
                    try:
                        yhat = self.arima_model_fit.predict(n_periods=1)[0]
                        arima_predictions.append(yhat)
                        history.append(arima_test_series.iloc[t])
                    except Exception as e:
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
                    self.temperature_maes.append(np.nan)

            except Exception as e:
                self.temperature_maes.append(np.nan)
        else:
            self.temperature_maes.append(np.nan)

        return rf_accuracy, arima_mae

    def get_average_metrics(self):
        """Returns the average accuracy and MAE across all training sessions."""
        avg_accuracy = (
            np.nanmean(self.weather_accuracies) if self.weather_accuracies else 0
        )
        avg_mae = (
            np.nanmean(self.temperature_maes) if self.temperature_maes else 0
        )
        return avg_accuracy, avg_mae

    def save_models(
        self, scaler_path, kmeans_path, rf_path, arima_path, rf_feature_columns_path, kmeans_scaler_path
    ):
        """Saves the trained scaler, KMeans, Random Forest, ARIMA models, and RF feature columns."""
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

        if self.kmeans_scaler:
            joblib.dump(self.kmeans_scaler, kmeans_scaler_path)
            print(f"KMeans Scaler saved to {kmeans_scaler_path}")
        else:
            print("KMeans Scaler not available to save (not fitted yet).")

        if self.rf_model:
            joblib.dump(self.rf_model, rf_path)
            print(f"Random Forest model saved to {rf_path}")
        else:
            print("Random Forest model not available to save (not fitted yet).")

        if self.arima_model_fit:
            joblib.dump(
                self.arima_model_fit, arima_path
            )
            print(f"ARIMA model saved to {arima_path}")
        else:
            print("ARIMA model not available to save (not fitted yet).")

        if (
            self.rf_model
            and hasattr(self.scaler, "feature_names_in_")
            and self.scaler.feature_names_in_ is not None
        ):
            joblib.dump(list(self.scaler.feature_names_in_), rf_feature_columns_path)
            print(f"Random Forest feature columns saved to {rf_feature_columns_path}")
        elif self.rf_model:
            print(
                "Warning: Scaler does not have 'feature_names_in_'. Could not save RF feature columns robustly."
            )
        else:
            print(
                "Random Forest model not available, so RF feature columns cannot be saved."
            )

# --- Weather Forecaster Class (NEW) ---
class WeatherForecaster:
    """
    Loads trained models and provides a 7-day weather forecast.
    """

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.scaler = None
        self.kmeans_model = None
        self.kmeans_scaler = None
        self.rf_model = None
        self.arima_model = None
        self.rf_feature_columns = None
        self.poly_features_for_rf = [
            "delhi_avg_temperature_2m",
            "delhi_avg_wind_speed_10m",
        ]
        self.poly_transformer = PolynomialFeatures(
            degree=2, include_bias=False
        )

        self.load_models()

    def load_models(self):
        """Loads the pre-trained models from the specified directory."""
        model_dir = self.config.model_save_dir
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        kmeans_path = os.path.join(model_dir, "kmeans_model.joblib")
        kmeans_scaler_path = os.path.join(model_dir, "cluster_scaler.joblib")
        rf_path = os.path.join(model_dir, "random_forest_model.joblib")
        arima_path = os.path.join(model_dir, "arima_model.joblib")
        rf_feature_columns_path = os.path.join(model_dir, "rf_feature_columns.joblib")

        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded Scaler from {scaler_path}")
            self.kmeans_model = joblib.load(kmeans_path)
            print(f"Loaded KMeans model from {kmeans_path}")
            self.kmeans_scaler = joblib.load(kmeans_scaler_path)
            print(f"Loaded KMeans Scaler from {kmeans_scaler_path}")
            self.rf_model = joblib.load(rf_path)
            print(f"Loaded Random Forest model from {rf_path}")
            self.arima_model = joblib.load(arima_path)
            print(f"Loaded ARIMA model from {arima_path}")
            self.rf_feature_columns = joblib.load(rf_feature_columns_path)
            print(f"Loaded Random Forest feature columns from {rf_feature_columns_path}")

            # Fit the polynomial transformer on dummy data to get feature names
            dummy_data = np.zeros((1, len(self.poly_features_for_rf)))
            self.poly_transformer.fit(dummy_data)

            # Consistency check for KMeans model and scaler features
            if self.kmeans_model and self.kmeans_scaler:
                kmeans_model_features = self.kmeans_model.n_features_in_
                kmeans_scaler_features = len(self.kmeans_scaler.feature_names_in_)
                print(f"DEBUG: Loaded KMeans model expects {kmeans_model_features} features.")
                print(f"DEBUG: Loaded KMeans scaler expects {kmeans_scaler_features} features.")

                if kmeans_model_features != kmeans_scaler_features:
                    print(f"WARNING: Feature count mismatch between loaded KMeans model ({kmeans_model_features}) and scaler ({kmeans_scaler_features}). Setting models to None to force retraining.")
                    self.kmeans_model = None
                    self.kmeans_scaler = None
                    self.scaler = None # Also set main scaler to None to force full retraining
                    self.rf_model = None
                    self.arima_model = None
                    self.rf_feature_columns = None


        except FileNotFoundError as e:
            print(f"Error loading model: {e}. Please ensure models are trained and saved.")
            self.scaler = None
            self.kmeans_model = None
            self.kmeans_scaler = None
            self.rf_model = None
            self.arima_model = None
            self.rf_feature_columns = None
        except Exception as e:
            print(f"An unexpected error occurred while loading models: {e}")
            self.scaler = None
            self.kmeans_model = None
            self.kmeans_scaler = None
            self.rf_model = None
            self.arima_model = None
            self.rf_feature_columns = None

    def map_cluster_to_weather_description(self, cluster_label):
        """
        Maps a KMeans cluster label to a human-readable weather description.
        NOTE: This is a placeholder mapping. For a real-world application,
        you would analyze the centroids of your KMeans clusters and assign
        meaningful weather descriptions based on the typical weather conditions
        (temperature, humidity, precipitation, cloud cover) within each cluster.
        """
        mapping = {
            0: "Clear/Sunny",
            1: "Partly Cloudy",
            2: "Cloudy",
            3: "Light Rain",
            4: "Moderate Rain",
            5: "Heavy Rain",
            6: "Thunderstorms",
            # Add more mappings if you have more clusters
        }
        return mapping.get(cluster_label, f"Weather Type {cluster_label}")

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

    def prepare_features_for_prediction(self, current_date, last_n_days_df, predicted_avg_temp=None):
        """
        Prepares a single row of features for a given prediction day.
        This function simulates the feature engineering pipeline from DataProcessor.
        It uses the last_n_days_df to calculate lagged and rolling features.
        For non-temperature features, it uses the last known values from last_n_days_df.
        """
        # Create a base dictionary for the current day's features
        current_day_data = {}
        current_day_data["day_of_week"] = current_date.weekday()
        current_day_data["month"] = current_date.month
        # Corrected access for day_of_year and week_of_year
        current_day_data["day_of_year"] = current_date.timetuple().tm_yday
        current_day_data["week_of_year"] = current_date.isocalendar()[1]
        current_day_data["season"] = self.get_season(current_date.month)

        # Populate 'delhi_avg_*' features
        # For temperature, use the predicted_avg_temp if provided, otherwise last historical avg temp
        if predicted_avg_temp is not None:
            current_day_data["delhi_avg_temperature_2m"] = predicted_avg_temp
            # For max/min temp, use a heuristic around the predicted average
            current_day_data["delhi_max_temperature_2m"] = predicted_avg_temp + 5 # Assuming a 5 degree spread
            current_day_data["delhi_min_temperature_2m"] = predicted_avg_temp - 5 # Assuming a 5 degree spread
        else:
            # Use last known historical values if no prediction is available (e.g., for day 1 if no ARIMA forecast yet)
            last_delhi_temp = last_n_days_df["delhi_avg_temperature_2m"].iloc[-1] if not last_n_days_df.empty else np.nan
            current_day_data["delhi_avg_temperature_2m"] = last_delhi_temp
            current_day_data["delhi_max_temperature_2m"] = last_n_days_df["delhi_max_temperature_2m"].iloc[-1] if not last_n_days_df.empty else np.nan
            current_day_data["delhi_min_temperature_2m"] = last_n_days_df["delhi_min_temperature_2m"].iloc[-1] if not last_n_days_df.empty else np.nan

        # For other weather variables, use the last known historical average values
        for var in self.config.weather_variables:
            if var == "temperature_2m": # Already handled
                continue
            avg_col = f"delhi_avg_{var}"
            max_col = f"delhi_max_{var}"
            min_col = f"delhi_min_{var}"
            current_day_data[avg_col] = last_n_days_df[avg_col].iloc[-1] if not last_n_days_df.empty else np.nan
            current_day_data[max_col] = last_n_days_df[max_col].iloc[-1] if not last_n_days_df.empty else np.nan
            current_day_data[min_col] = last_n_days_df[min_col].iloc[-1] if not last_n_days_df.empty else np.nan

        # For surrounding regions, use the last known historical average values
        for region_name in self.config.region_names:
            for var in self.config.weather_variables:
                avg_col = f"{region_name}_avg_{var}"
                max_col = f"{region_name}_max_{var}"
                min_col = f"{region_name}_min_{var}"
                current_day_data[avg_col] = last_n_days_df[avg_col].iloc[-1] if not last_n_days_df.empty else np.nan
                current_day_data[max_col] = last_n_days_df[max_col].iloc[-1] if not last_n_days_df.empty else np.nan
                current_day_data[min_col] = last_n_days_df[min_col].iloc[-1] if not last_n_days_df.empty else np.nan

        # Convert to DataFrame for feature engineering
        current_day_df = pd.DataFrame([current_day_data])
        
        # Get the list of expected season dummy columns from the loaded RF feature columns
        expected_season_dummy_cols = [col for col in self.rf_feature_columns if col.startswith('season_')]
        
        # Create one-hot encoded columns for 'season' in current_day_df
        # This will create columns like 'season_Spring', 'season_Summer', etc.
        # Ensure the 'season' column exists before attempting to get dummies
        if 'season' in current_day_df.columns:
            current_day_df_processed_season = pd.get_dummies(current_day_df['season'], prefix='season', drop_first=True).astype(int)
            
            # Reindex these dummy columns to ensure all expected ones are present, filling missing with 0
            current_day_df_processed_season = current_day_df_processed_season.reindex(columns=expected_season_dummy_cols, fill_value=0)
            
            # Drop the original 'season' column from current_day_df
            current_day_df = current_day_df.drop(columns=['season'])
            
            # Concatenate the non-season features with the one-hot encoded season features
            current_day_df = pd.concat([current_day_df, current_day_df_processed_season], axis=1)
        else:
            # If 'season' column is somehow missing, ensure dummy columns are still created and filled with 0
            empty_season_dummies = pd.DataFrame(0, index=current_day_df.index, columns=expected_season_dummy_cols)
            current_day_df = pd.concat([current_day_df, empty_season_dummies], axis=1)


        # Ensure current_day_df has the same base columns as last_n_days_df before concatenation
        # This is crucial for consistent column alignment.
        base_cols_from_last_n_days = [col for col in last_n_days_df.columns if col not in ["target_weather", "target_temperature"]]
        current_day_df = current_day_df.reindex(columns=base_cols_from_last_n_days, fill_value=0)

        # Now, both dataframes should have consistent columns for concatenation
        combined_df = pd.concat([last_n_days_df[base_cols_from_last_n_days], current_day_df], ignore_index=True)

        # Re-apply feature engineering steps similar to process_historical_data
        features_to_lag_and_roll = [
            "delhi_avg_temperature_2m",
            "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m",
            "delhi_avg_precipitation",
            "delhi_avg_cloud_cover",
        ]
        lags = [1, 2, 3, 7, 14, 30]
        rolling_windows = [3, 7]

        new_features_df = pd.DataFrame(index=combined_df.index)

        for feature in features_to_lag_and_roll:
            for lag in lags:
                new_features_df[f"{feature}_lag_{lag}"] = combined_df[feature].shift(lag)
            for window in rolling_windows:
                new_features_df[f"{feature}_rolling_mean_{window}d"] = (
                    combined_df[feature].rolling(window=window).mean()
                )
                new_features_df[f"{feature}_rolling_min_{window}d"] = (
                    combined_df[feature].rolling(window=window).min()
                )
                new_features_df[f"{feature}_rolling_max_{window}d"] = (
                    combined_df[feature].rolling(window=window).max()
                )
                new_features_df[f"{feature}_rolling_std_{window}d"] = (
                    combined_df[feature].rolling(window=window).std()
                )
            new_features_df[f"{feature}_diff_1d"] = combined_df[feature].diff(1)

        new_features_df["delhi_temp_x_humidity"] = (
            combined_df["delhi_avg_temperature_2m"] * combined_df["delhi_avg_relative_humidity_2m"]
        )
        new_features_df["delhi_wind_x_cloud"] = (
            combined_df["delhi_avg_wind_speed_10m"] * combined_df["delhi_avg_cloud_cover"]
        )

        # Apply polynomial features
        poly_input_df = combined_df[self.poly_features_for_rf].copy()
        poly_input_df.fillna(poly_input_df.mean(), inplace=True)

        poly_transformed = self.poly_transformer.transform(poly_input_df)
        poly_feature_names = self.poly_transformer.get_feature_names_out(self.poly_features_for_rf)
        
        # Create full poly_df, then drop the original features to avoid duplication
        full_poly_df = pd.DataFrame(
            poly_transformed, columns=poly_feature_names, index=combined_df.index
        )
        original_features_to_drop = ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        poly_df = full_poly_df.drop(columns=original_features_to_drop, errors='ignore') # Drop original features if they exist

        new_features_df = pd.concat([new_features_df, poly_df], axis=1)

        combined_df = pd.concat([combined_df, new_features_df], axis=1)
        # Safeguard: Remove any duplicate columns that might have been introduced
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        
        features_for_prediction = combined_df.iloc[[-1]].copy()

        if self.rf_feature_columns is not None:
            features_for_prediction = features_for_prediction.reindex(
                columns=self.rf_feature_columns, fill_value=0
            )
        else:
            print("Warning: RF feature columns not loaded. Prediction might fail due to inconsistent features.")
            return pd.DataFrame()

        features_for_prediction = features_for_prediction[self.rf_feature_columns]

        if self.scaler:
            features_for_prediction.fillna(features_for_prediction.mean(), inplace=True)
            scaled_features = self.scaler.transform(features_for_prediction)
            return pd.DataFrame(scaled_features, columns=self.rf_feature_columns)
        else:
            print("Scaler not loaded. Cannot scale features for prediction.")
            return pd.DataFrame()

    def run_forecast_pipeline(self, db_manager: DatabaseManager):
        """
        Loads pre-trained models, trains them for the last 30 days, and predicts
        weather category and temperature for the next 7 days.
        """
        # Initial check for loaded models. If any are missing due to previous errors or file not found,
        # we will attempt to train them.
        if not (self.scaler and self.kmeans_model and self.rf_model and self.arima_model and self.rf_feature_columns and self.kmeans_scaler):
            print("One or more models are missing or inconsistent. Attempting to train models first.")
            # Trigger training
            self._train_models_if_needed(db_manager)
            # After training, reload models to ensure the forecaster has the newly trained ones
            self.load_models()
            # If models are still missing after training attempt, then exit.
            if not (self.scaler and self.kmeans_model and self.rf_model and self.arima_model and self.rf_feature_columns and self.kmeans_scaler):
                print("Models could not be loaded or trained successfully. Cannot proceed with forecast.")
                return pd.DataFrame()


        print("\n--- Starting Future Weather Prediction ---")

        data_fetcher = WeatherDataFetcher(self.config, db_manager)
        data_processor = DataProcessor(self.config)

        # Determine date range for recent training and future prediction
        today = date.today()
        # Fetch 90 days of historical data to ensure enough data for 30-day lags and rolling features
        history_fetch_start_date = today - timedelta(days=90)
        history_fetch_end_date = today - timedelta(days=1) # Up to yesterday
        prediction_start_date = today + timedelta(days=1) # Tomorrow
        
        print(f"Fetching data from {history_fetch_start_date} to {history_fetch_end_date} for recent model update.")

        # Initialize raw_historical_data structure for fetching
        raw_recent_data = {
            "delhi": {"hourly": {var: [] for var in self.config.weather_variables + ["time"]}},
            "surrounding_regions": {},
        }
        for region_name in self.config.region_names:
            raw_recent_data["surrounding_regions"][region_name] = {
                "hourly": {var: [] for var in self.config.weather_variables + ["time"]}
            }

        # Fetch recent Delhi data from DB
        delhi_recent_data = data_fetcher.fetch_historical_weather_data(
            self.config.delhi_latitude,
            self.config.delhi_longitude,
            history_fetch_start_date,
            history_fetch_end_date,
            region_name="delhi",
        )
        if delhi_recent_data and delhi_recent_data.get("hourly") and delhi_recent_data["hourly"].get("time"):
            for key in raw_recent_data["delhi"]["hourly"]:
                raw_recent_data["delhi"]["hourly"][key].extend(delhi_recent_data["hourly"].get(key, []))

        # Fetch recent surrounding regions data from DB
        surrounding_locations = {}
        for i in range(self.config.num_regions):
            bearing = self.config.region_bearings[i]
            region_name = self.config.region_names[i]
            dest_lat, dest_lon = data_processor.calculate_destination_point(
                self.config.delhi_latitude, self.config.delhi_longitude, self.config.radius_km, bearing
            )
            surrounding_locations[region_name] = {
                "latitude": dest_lat,
                "longitude": dest_lon,
                "bearing": bearing,
            }

        for region_name, coords in surrounding_locations.items():
            region_recent_data = data_fetcher.fetch_historical_weather_data(
                coords["latitude"],
                coords["longitude"],
                history_fetch_start_date,
                history_fetch_end_date,
                region_name=region_name,
            )
            if region_recent_data and region_recent_data.get("hourly") and region_recent_data["hourly"].get("time"):
                for key in raw_recent_data["surrounding_regions"][region_name]["hourly"]:
                    raw_recent_data["surrounding_regions"][region_name]["hourly"][key].extend(
                        region_recent_data["hourly"].get(key, [])
                    )

        print("Recent data fetched from database.")
        print(f"DEBUG: raw_recent_data['delhi']['hourly']['time'] length: {len(raw_recent_data['delhi']['hourly']['time'])}")
        print(f"DEBUG: raw_recent_data['delhi']['hourly']['temperature_2m'] length: {len(raw_recent_data['delhi']['hourly']['temperature_2m'])}")
        print(f"DEBUG: History fetch start date: {history_fetch_start_date}, end date: {history_fetch_end_date}")


        # Process recent data for RF and ARIMA, passing the loaded KMeans model and its dedicated scaler
        # Pass self.kmeans_model and self.kmeans_scaler which are now guaranteed to be consistent (or None)
        recent_weather_df, _ = data_processor.process_historical_data(raw_recent_data, kmeans_model_for_prediction=self.kmeans_model, kmeans_scaler_for_prediction=self.kmeans_scaler)
        recent_temp_series = data_processor.process_historical_data_for_arima(raw_recent_data)

        print(f"DEBUG: recent_weather_df shape after processing: {recent_weather_df.shape}")
        print(f"DEBUG: recent_temp_series shape after processing: {recent_temp_series.shape}")

        if recent_weather_df.empty or recent_temp_series.empty:
            print("Not enough recent data to make predictions. Please ensure data for the last 90 days is available and complete.")
            return pd.DataFrame()

        forecast_results = []
        print("\nPredicting for the next 7 days:")

        # ARIMA requires only the time series.
        # We will "update" the ARIMA model with the latest data before predicting.
        arima_model_updated = self.arima_model.fit(recent_temp_series)
        arima_future_temps = arima_model_updated.predict(n_periods=7)

        # Get the last 30 days of processed data for RF feature generation
        last_n_days_df = recent_weather_df.tail(30).copy()

        for i in range(7):
            forecast_date = prediction_start_date + timedelta(days=i)
            
            # --- ARIMA Prediction (Temperature) ---
            predicted_temperature = arima_future_temps[i] if i < len(arima_future_temps) else np.nan

            # Estimate Max and Min Temperature
            max_temp = predicted_temperature + 5 if not np.isnan(predicted_temperature) else np.nan
            min_temp = predicted_temperature - 5 if not np.isnan(predicted_temperature) else np.nan

            # --- Random Forest Prediction (Weather Category) ---
            # Prepare features for the current forecast day
            features_for_rf_prediction = self.prepare_features_for_prediction(
                forecast_date, last_n_days_df, predicted_avg_temp=predicted_temperature
            )

            predicted_weather_category = np.nan
            if not features_for_rf_prediction.empty and self.rf_model:
                try:
                    rf_prediction = self.rf_model.predict(features_for_rf_prediction)[0]
                    predicted_weather_category = int(rf_prediction)
                except Exception as e:
                    print(f"Error during RF prediction for {forecast_date}: {e}")

            weather_condition = self.map_cluster_to_weather_description(predicted_weather_category)

            forecast_results.append(
                {
                    "Date": forecast_date.strftime("%Y-%m-%d"),
                    "Max Temp": f"{max_temp:.2f}" if not np.isnan(max_temp) else "N/A",
                    "Min Temp": f"{min_temp:.2f}" if not np.isnan(min_temp) else "N/A",
                    "Conditions": weather_condition,
                }
            )

            # Update last_n_days_df for the next iteration by appending the current day's predicted features
            # This is crucial for iterative forecasting with lagged features.
            # We need to recreate a "simulated" row for the predicted day.
            
            # Create a new row with the same columns as recent_weather_df
            new_row_for_history = pd.DataFrame(columns=recent_weather_df.columns)
            new_row_for_history.loc[0, "delhi_avg_temperature_2m"] = predicted_temperature
            new_row_for_history.loc[0, "delhi_max_temperature_2m"] = max_temp
            new_row_for_history.loc[0, "delhi_min_temperature_2m"] = min_temp
            new_row_for_history.loc[0, "target_temperature"] = predicted_temperature
            new_row_for_history.loc[0, "target_weather"] = predicted_weather_category

            # Fill other 'delhi_avg_*' and 'surrounding_regions_avg_*' with last known values
            # This is an assumption for future unknown features.
            for col in recent_weather_df.columns:
                if col.startswith("delhi_avg_") and col != "delhi_avg_temperature_2m":
                    new_row_for_history.loc[0, col] = last_n_days_df[col].iloc[-1] if not last_n_days_df.empty else np.nan
                elif any(col.startswith(f"{r}_avg_") for r in self.config.region_names):
                    new_row_for_history.loc[0, col] = last_n_days_df[col].iloc[-1] if not last_n_days_df.empty else np.nan
                elif col == "day_of_week":
                    new_row_for_history.loc[0, col] = forecast_date.weekday()
                elif col == "month":
                    new_row_for_history.loc[0, col] = forecast_date.month
                elif col == "day_of_year":
                    new_row_for_history.loc[0, col] = forecast_date.timetuple().tm_yday
                elif col == "week_of_year":
                    new_row_for_history.loc[0, col] = forecast_date.isocalendar()[1]
                elif "season_" in col:
                    season_name = self.get_season(forecast_date.month)
                    # Check if the specific season dummy column exists in the target columns
                    if f"season_{season_name}" in new_row_for_history.columns:
                        new_row_for_history.loc[0, f"season_{season_name}"] = 1
                elif col not in ["target_weather", "target_temperature"]:
                    # For other non-target columns, if they are not explicitly set, try to copy from last_n_days_df
                    if col in last_n_days_df.columns:
                        new_row_for_history.loc[0, col] = last_n_days_df[col].iloc[-1] if not last_n_days_df.empty else np.nan

            for col in new_row_for_history.columns:
                if col in last_n_days_df.columns:
                    try:
                        new_row_for_history[col] = new_row_for_history[col].astype(last_n_days_df[col].dtype)
                    except Exception:
                        pass

            last_n_days_df = pd.concat([last_n_days_df, new_row_for_history], ignore_index=True)


        # Modified section for printing the forecast results in a visually appealing table
        print("\n--- 7-Day Weather Forecast ---")
        header = f"{'Date':<10} {'Max Temp':>9} {'Min Temp':>9} {'Conditions':<20}"
        separator = "-" * len(header)
        print(header)
        print(separator)

        for result in forecast_results:
            print(
                f"{result['Date']:<10} {result['Max Temp']:>9} {result['Min Temp']:>9} {result['Conditions']:<20}"
            )
        print(separator)
        print("Future weather prediction complete.")
        return pd.DataFrame(forecast_results)

    def _train_models_if_needed(self, db_manager: DatabaseManager):
        """
        Helper function to train models if they are missing or inconsistent.
        This will be called from run_forecast_pipeline if needed.
        """
        print("\n--- Initiating Model Training (Fallback) ---")
        trainer = HybridModelTrainer(self.config)
        data_fetcher = WeatherDataFetcher(self.config, db_manager)
        data_processor = DataProcessor(self.config)

        # Fetch historical data for the full training period
        raw_training_data = {
            "delhi": {"hourly": {var: [] for var in self.config.weather_variables + ["time"]}},
            "surrounding_regions": {},
        }
        for region_name in self.config.region_names:
            raw_training_data["surrounding_regions"][region_name] = {
                "hourly": {var: [] for var in self.config.weather_variables + ["time"]}
            }

        # Fetch full Delhi training data from DB
        delhi_train_data = data_fetcher.fetch_historical_weather_data(
            self.config.delhi_latitude,
            self.config.delhi_longitude,
            self.config.train_start_date,
            self.config.train_end_date,
            region_name="delhi",
        )
        if delhi_train_data and delhi_train_data.get("hourly") and delhi_train_data["hourly"].get("time"):
            for key in raw_training_data["delhi"]["hourly"]:
                raw_training_data["delhi"]["hourly"][key].extend(delhi_train_data["hourly"].get(key, []))

        # Fetch full surrounding regions training data from DB
        surrounding_locations = {}
        for i in range(self.config.num_regions):
            bearing = self.config.region_bearings[i]
            region_name = self.config.region_names[i]
            dest_lat, dest_lon = data_processor.calculate_destination_point(
                self.config.delhi_latitude, self.config.delhi_longitude, self.config.radius_km, bearing
            )
            surrounding_locations[region_name] = {
                "latitude": dest_lat,
                "longitude": dest_lon,
                "bearing": bearing,
            }

        for region_name, coords in surrounding_locations.items():
            region_train_data = data_fetcher.fetch_historical_weather_data(
                coords["latitude"],
                coords["longitude"],
                self.config.train_start_date,
                self.config.train_end_date,
                region_name=region_name,
            )
            if region_train_data and region_train_data.get("hourly") and region_train_data["hourly"].get("time"):
                for key in raw_training_data["surrounding_regions"][region_name]["hourly"]:
                    raw_training_data["surrounding_regions"][region_name]["hourly"][key].extend(
                        region_train_data["hourly"].get(key, [])
                    )
        
        print("Training data fetched from database for fallback training.")
        
        # Process training data
        df_full_processed, trained_kmeans_model = data_processor.process_historical_data(raw_training_data)
        temp_series_full_processed = data_processor.process_historical_data_for_arima(raw_training_data)

        if df_full_processed.empty or temp_series_full_processed.empty:
            print("Not enough training data to train models. Please ensure historical data is available.")
            return

        # Define training/testing split for the fallback training
        # In a real scenario, you might want a more sophisticated cross-validation.
        train_size = int(len(df_full_processed) * 0.8)
        train_start_idx = 0
        train_end_idx = train_size
        test_start_idx = train_size
        test_end_idx = len(df_full_processed)

        print(f"Training models with {train_size} samples.")
        rf_accuracy, arima_mae = trainer.train_and_evaluate(
            df_full_processed,
            temp_series_full_processed,
            train_start_idx,
            train_end_idx,
            test_start_idx,
            test_end_idx,
            session_num=1 # Only one session for fallback training
        )
        print(f"Fallback Training - RF Accuracy: {rf_accuracy:.2f}, ARIMA MAE: {arima_mae:.2f}")

        # Save the newly trained models
        scaler_path = os.path.join(self.config.model_save_dir, "scaler.joblib")
        kmeans_path = os.path.join(self.config.model_save_dir, "kmeans_model.joblib")
        kmeans_scaler_path = os.path.join(self.config.model_save_dir, "cluster_scaler.joblib")
        rf_path = os.path.join(self.config.model_save_dir, "random_forest_model.joblib")
        arima_path = os.path.join(self.config.model_save_dir, "arima_model.joblib")
        rf_feature_columns_path = os.path.join(self.config.model_save_dir, "rf_feature_columns.joblib")
        
        # IMPORTANT: Assign the kmeans_model and kmeans_scaler from data_processor to trainer
        # before saving, as these are the ones fitted during the processing step.
        trainer.kmeans_model = data_processor.kmeans_model
        trainer.kmeans_scaler = data_processor.kmeans_scaler

        trainer.save_models(scaler_path, kmeans_path, rf_path, arima_path, rf_feature_columns_path, kmeans_scaler_path)
        print("Models trained and saved successfully during fallback.")


# --- Main Execution Function ---
def main():
    """
    Main function to run the hybrid weather forecasting model,
    either for full training and saving models, or for future prediction
    using existing models.
    """
    config = WeatherConfig()
    db_manager = DatabaseManager(config)

    # Initialize WeatherForecaster which will load the models
    forecaster = WeatherForecaster(config)

    # Run the forecast pipeline
    forecaster.run_forecast_pipeline(db_manager)

    db_manager.close_connection()


# --- Script Entry Point ---
if __name__ == "__main__":
    main()