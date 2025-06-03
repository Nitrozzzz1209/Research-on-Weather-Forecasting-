import math
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, silhouette_score # Added silhouette_score
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
        self.delhi_latitude = 28.6448
        self.delhi_longitude = 77.2167
        self.radius_km = 500
        self.num_regions = 8
        self.earth_radius_km = 6371

        self.train_start_date = date(2020, 1, 1)
        self.train_end_date = date(2024, 12, 31)

        self.region_bearings = [0, 45, 90, 135, 180, 225, 270, 315]
        self.region_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        self.weather_variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "precipitation",
            "cloud_cover",
        ]

        self.db_name = "Project_Weather_Forecasting"
        self.db_user = "postgres"
        self.db_password = "1234" # Please use a secure password management system in production
        self.db_host = "localhost"
        self.db_port = "5432"

        self.model_save_dir = "MODEL 1/trained_models"
        # Define a range for k in KMeans to test for Silhouette Score
        self.kmeans_k_range = range(3, 11) # Test k from 3 to 10 clusters


# --- Database Manager Class ---
class DatabaseManager:
    """Manages PostgreSQL database connection and operations."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
                host=self.config.db_host,
                port=self.config.db_port,
            )
            self.conn.autocommit = True
            print("Initializing: Database connected.")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def create_tables(self):
        if not self.conn:
            print("Cannot create tables: No database connection.")
            return
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS delhi_hourly_weather (
                    id SERIAL PRIMARY KEY, timestamp TIMESTAMP UNIQUE, temperature_2m REAL,
                    relative_humidity_2m REAL, wind_speed_10m REAL, wind_direction_10m REAL,
                    weather_code INTEGER, precipitation REAL, cloud_cover REAL
                );"""
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS surrounding_hourly_weather (
                    id SERIAL PRIMARY KEY, region_name TEXT NOT NULL, latitude REAL, longitude REAL,
                    timestamp TIMESTAMP, temperature_2m REAL, relative_humidity_2m REAL,
                    wind_speed_10m REAL, wind_direction_10m REAL, weather_code INTEGER,
                    precipitation REAL, cloud_cover REAL, UNIQUE (region_name, timestamp)
                );"""
            )
        except psycopg2.Error as e:
            print(f"Error creating tables: {e}")
        finally:
            cur.close()

    def load_delhi_data(self, start_date, end_date):
        if not self.conn: return None
        cur = self.conn.cursor()
        try:
            query = sql.SQL("""
                SELECT timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m,
                       wind_direction_10m, weather_code, precipitation, cloud_cover
                FROM delhi_hourly_weather WHERE timestamp >= %s AND timestamp <= %s ORDER BY timestamp;""")
            cur.execute(query, (start_date, end_date + timedelta(days=1)))
            rows = cur.fetchall()
            if not rows: return None
            data = { "hourly": { col: [] for col in self.config.weather_variables + ["time"] }, "hourly_units": {} }
            # Populate hourly_units (can be simplified if not strictly needed for processing)
            data["hourly_units"] = {
                "time": "iso8601", "temperature_2m": "celsius", "relative_humidity_2m": "%",
                "wind_speed_10m": "km/h", "wind_direction_10m": "degrees",
                "weather_code": "wmo code", "precipitation": "mm", "cloud_cover": "%"
            }
            column_names = ["time"] + self.config.weather_variables
            for row in rows:
                data["hourly"]["time"].append(row[0].isoformat())
                for i, var_name in enumerate(self.config.weather_variables):
                    data["hourly"][var_name].append(row[i+1])
            return data
        except psycopg2.Error as e:
            print(f"Error loading Delhi data: {e}")
            return None
        finally: cur.close()

    def load_surrounding_data(self, region_name, start_date, end_date):
        if not self.conn: return None
        cur = self.conn.cursor()
        try:
            query = sql.SQL("""
                SELECT timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m,
                       wind_direction_10m, weather_code, precipitation, cloud_cover
                FROM surrounding_hourly_weather WHERE region_name = %s AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp;""")
            cur.execute(query, (region_name, start_date, end_date + timedelta(days=1)))
            rows = cur.fetchall()
            if not rows: return None
            data = { "hourly": { col: [] for col in self.config.weather_variables + ["time"] }, "hourly_units": {} }
            data["hourly_units"] = {
                "time": "iso8601", "temperature_2m": "celsius", "relative_humidity_2m": "%",
                "wind_speed_10m": "km/h", "wind_direction_10m": "degrees",
                "weather_code": "wmo code", "precipitation": "mm", "cloud_cover": "%"
            }
            column_names = ["time"] + self.config.weather_variables
            for row in rows:
                data["hourly"]["time"].append(row[0].isoformat())
                for i, var_name in enumerate(self.config.weather_variables):
                    data["hourly"][var_name].append(row[i+1])
            return data
        except psycopg2.Error as e:
            print(f"Error loading {region_name} data: {e}")
            return None
        finally: cur.close()

    def close_connection(self):
        if self.conn: self.conn.close()


# --- Data Fetcher Class ---
class WeatherDataFetcher:
    def __init__(self, config: WeatherConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager

    def fetch_historical_weather_data(self, latitude, longitude, start_date, end_date, region_name=None):
        db_data = None
        if self.db_manager.conn:
            if region_name == "delhi":
                db_data = self.db_manager.load_delhi_data(start_date, end_date)
            elif region_name:
                db_data = self.db_manager.load_surrounding_data(region_name, start_date, end_date)
            if db_data and db_data.get("hourly") and db_data["hourly"].get("time"):
                db_start_dt = datetime.fromisoformat(db_data["hourly"]["time"][0]).date()
                db_end_dt = datetime.fromisoformat(db_data["hourly"]["time"][-1]).date()
                if db_start_dt <= start_date and db_end_dt >= end_date:
                    return db_data
        return None


# --- Data Processor Class ---
class DataProcessor:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.kmeans_model = None
        self.kmeans_scaler = None # Scaler for KMeans features

    def calculate_destination_point(self, start_lat, start_lon, distance_km, bearing_deg):
        start_lat_rad, start_lon_rad, bearing_rad = map(math.radians, [start_lat, start_lon, bearing_deg])
        angular_distance = distance_km / self.config.earth_radius_km
        dest_lat_rad = math.asin(math.sin(start_lat_rad) * math.cos(angular_distance) +
                                 math.cos(start_lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad))
        dest_lon_rad = start_lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(start_lat_rad),
                                                math.cos(angular_distance) - math.sin(start_lat_rad) * math.sin(dest_lat_rad))
        return map(math.degrees, [dest_lat_rad, dest_lon_rad])

    def get_season(self, month):
        if month in [3, 4]: return "Spring"
        elif month in [5, 6]: return "Summer"
        elif month in [7, 8, 9]: return "Monsoon"
        else: return "Winter"

    # ENHANCEMENT: Function to calculate Dew Point using Magnus formula
    def _calculate_dew_point(self, T, RH):
        """Calculates dew point in Celsius. T in Celsius, RH in %."""
        # Ensure RH is within a valid range (0-100) to avoid log(0) or log(negative)
        RH = np.clip(RH, 1, 100) # Clip RH to be at least 1% to avoid log(0)
        a = 17.27
        b = 237.7
        alpha = ((a * T) / (b + T)) + np.log(RH / 100.0)
        T_dp = (b * alpha) / (a - alpha)
        return T_dp

    def process_historical_data(self, raw_data):
        dataset = []
        delhi_data = raw_data.get("delhi")
        surrounding_data = raw_data.get("surrounding_regions", {})

        if not delhi_data or not delhi_data.get("hourly") or not delhi_data["hourly"].get("time"):
            print("Error: Missing or incomplete historical data for Delhi. Cannot create dataset.")
            return pd.DataFrame(), None, None

        delhi_hourly = delhi_data["hourly"]
        time_stamps = delhi_hourly["time"]
        num_hours = len(time_stamps)
        hours_per_day = 24
        num_days = num_hours // hours_per_day

        for i in range(num_days - 1): # num_days - 1 to ensure there's a next day for target
            start_hour_index = i * hours_per_day
            end_hour_index = start_hour_index + hours_per_day
            current_day_data = {}
            current_day_time = datetime.fromisoformat(time_stamps[start_hour_index]).date()

            current_day_data["day_of_week"] = current_day_time.weekday()
            current_day_data["month"] = current_day_time.month
            current_day_data["day_of_year"] = current_day_time.timetuple().tm_yday
            current_day_data["week_of_year"] = current_day_time.isocalendar()[1]
            current_day_data["season"] = self.get_season(current_day_time.month)

            for var in self.config.weather_variables:
                hourly_values = delhi_hourly.get(var, [])[start_hour_index:end_hour_index]
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
                if region_data_for_current_day and isinstance(region_data_for_current_day, dict):
                    region_hourly = region_data_for_current_day.get("hourly", {})
                    if region_hourly and region_hourly.get("time"):
                        region_time_stamps = region_hourly["time"]
                        region_start_index, region_end_index = None, None
                        for j, ts_str in enumerate(region_time_stamps):
                            ts_date = datetime.fromisoformat(ts_str).date()
                            if ts_date == current_day_time:
                                if region_start_index is None: region_start_index = j
                                region_end_index = j + 1 # To slice up to this index
                        
                        if region_start_index is not None and region_end_index is not None:
                            for var in self.config.weather_variables:
                                hourly_values = region_hourly.get(var, [])[region_start_index:region_end_index]
                                valid_hourly_values = [val for val in hourly_values if val is not None]
                                if valid_hourly_values:
                                    current_day_data[f"{region_name}_avg_{var}"] = np.mean(valid_hourly_values)
                                    current_day_data[f"{region_name}_max_{var}"] = np.max(valid_hourly_values)
                                    current_day_data[f"{region_name}_min_{var}"] = np.min(valid_hourly_values)
                                else: # Fill with NaN if no valid data for the day
                                    for agg in ["avg", "max", "min"]: current_day_data[f"{region_name}_{agg}_{var}"] = np.nan
                        else: # Fill with NaN if no data for the day
                             for var in self.config.weather_variables:
                                for agg in ["avg", "max", "min"]: current_day_data[f"{region_name}_{agg}_{var}"] = np.nan
                    else: # Fill with NaN if region_hourly or time is missing
                        for var in self.config.weather_variables:
                            for agg in ["avg", "max", "min"]: current_day_data[f"{region_name}_{agg}_{var}"] = np.nan
                else: # Fill with NaN if region data is missing
                    for var in self.config.weather_variables:
                        for agg in ["avg", "max", "min"]: current_day_data[f"{region_name}_{agg}_{var}"] = np.nan


            next_day_start_hour_index = end_hour_index
            next_day_end_hour_index = next_day_start_hour_index + hours_per_day
            if next_day_end_hour_index <= num_hours:
                next_day_temperatures = delhi_hourly.get("temperature_2m", [])[next_day_start_hour_index:next_day_end_hour_index]
                valid_next_day_temperatures = [val for val in next_day_temperatures if val is not None]
                if valid_next_day_temperatures:
                    current_day_data["target_temperature"] = np.mean(valid_next_day_temperatures)
                    dataset.append(current_day_data)

        df = pd.DataFrame(dataset)

        # ENHANCEMENT: Add Cyclical Features for month and day_of_week
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        if 'day_of_week' in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        # Optionally drop original month and day_of_week if not needed elsewhere,
        # but RF can handle them. For now, keep them for potential direct interpretation.

        df = pd.get_dummies(df, columns=["season"], drop_first=True) # One-hot encode season
        
        # ENHANCEMENT: Calculate and add Dew Point
        if 'delhi_avg_temperature_2m' in df.columns and 'delhi_avg_relative_humidity_2m' in df.columns:
            df['delhi_dew_point'] = self._calculate_dew_point(
                df['delhi_avg_temperature_2m'],
                df['delhi_avg_relative_humidity_2m']
            )

        df.dropna(inplace=True) # Initial drop after basic aggregations and new features
        df.reset_index(drop=True, inplace=True) # Reset index before lagged features

        features_to_lag_and_roll = [
            "delhi_avg_temperature_2m", "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m", "delhi_avg_precipitation", "delhi_avg_cloud_cover",
            # Consider adding 'delhi_dew_point' here if you want its lags/rolls
        ]
        lags = [1, 2, 3, 7, 14, 30]
        rolling_windows = [3, 7]
        new_features_df = pd.DataFrame(index=df.index)

        for feature in features_to_lag_and_roll:
            if feature in df.columns: # Check if feature exists before creating lags/rolls
                for lag in lags: new_features_df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
                for window in rolling_windows:
                    new_features_df[f"{feature}_rolling_mean_{window}d"] = df[feature].rolling(window=window).mean()
                    new_features_df[f"{feature}_rolling_min_{window}d"] = df[feature].rolling(window=window).min()
                    new_features_df[f"{feature}_rolling_max_{window}d"] = df[feature].rolling(window=window).max()
                    new_features_df[f"{feature}_rolling_std_{window}d"] = df[feature].rolling(window=window).std()
                new_features_df[f"{feature}_diff_1d"] = df[feature].diff(1)
        
        if 'delhi_avg_temperature_2m' in df.columns and 'delhi_avg_relative_humidity_2m' in df.columns:
            new_features_df["delhi_temp_x_humidity"] = df["delhi_avg_temperature_2m"] * df["delhi_avg_relative_humidity_2m"]
        if 'delhi_avg_wind_speed_10m' in df.columns and 'delhi_avg_cloud_cover' in df.columns:
            new_features_df["delhi_wind_x_cloud"] = df["delhi_avg_wind_speed_10m"] * df["delhi_avg_cloud_cover"]

        poly_features_cols = ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        if all(col in df.columns for col in poly_features_cols):
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_input_df = df[poly_features_cols].copy()
            poly_input_df.fillna(poly_input_df.mean(), inplace=True) # Handle potential NaNs
            poly_transformed = poly.fit_transform(poly_input_df)
            poly_feature_names = poly.get_feature_names_out(poly_features_cols)
            
            # Create full poly_df, then drop the original features from poly_df to avoid duplication if poly adds them
            full_poly_df = pd.DataFrame(poly_transformed, columns=poly_feature_names, index=df.index)
            # The original features are still in 'df'. We are adding squared/interaction terms.
            # We need to ensure we don't add original columns if PolynomialFeatures includes them.
            # get_feature_names_out usually gives unique names for interactions and powers.
            # If a name like 'delhi_avg_temperature_2m' (power 1) is generated, we might want to drop it from full_poly_df
            # to avoid exact duplication with df['delhi_avg_temperature_2m'] when concatenating.
            # However, it's safer to let them be and then use df.loc[:,~df.columns.duplicated()] later.
            new_features_df = pd.concat([new_features_df, full_poly_df], axis=1)


        df = pd.concat([df, new_features_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()] # Remove duplicate columns if any
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True) # Final reset index

        clustering_features = [
            "delhi_avg_temperature_2m", "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m", "delhi_avg_wind_direction_10m",
            "delhi_avg_weather_code", "delhi_avg_precipitation", "delhi_avg_cloud_cover",
        ]
        
        df_for_clustering = df[clustering_features].copy()
        df_for_clustering.dropna(inplace=True) # Ensure no NaNs before scaling

        if not df_for_clustering.empty:
            self.kmeans_scaler = StandardScaler() # Use self.kmeans_scaler
            scaled_clustering_features = self.kmeans_scaler.fit_transform(df_for_clustering)

            # ENHANCEMENT: Find optimal k for KMeans using Silhouette Score
            best_k = -1
            best_silhouette_score = -1 # Silhouette score ranges from -1 to 1
            
            print("\nFinding optimal k for KMeans using Silhouette Score...")
            for k_test in self.config.kmeans_k_range:
                if k_test > len(df_for_clustering)-1 : # KMeans k cannot be >= n_samples
                    print(f"Skipping k={k_test}, not enough samples ({len(df_for_clustering)}) for clustering.")
                    continue
                kmeans_test = KMeans(n_clusters=k_test, random_state=42, n_init='auto')
                cluster_labels_test = kmeans_test.fit_predict(scaled_clustering_features)
                
                # Silhouette score requires at least 2 labels and less than n_samples-1 labels.
                if len(np.unique(cluster_labels_test)) > 1 and len(np.unique(cluster_labels_test)) < len(df_for_clustering) :
                    try:
                        score = silhouette_score(scaled_clustering_features, cluster_labels_test)
                        print(f"  k={k_test}, Silhouette Score: {score:.4f}")
                        if score > best_silhouette_score:
                            best_silhouette_score = score
                            best_k = k_test
                    except ValueError as e:
                         print(f"  Could not calculate Silhouette Score for k={k_test}: {e}")
                else:
                    print(f"  Skipping Silhouette Score for k={k_test} due to insufficient distinct clusters ({len(np.unique(cluster_labels_test))}) for samples ({len(df_for_clustering)}).")


            if best_k != -1:
                print(f"Best k found: {best_k} with Silhouette Score: {best_silhouette_score:.4f}")
                n_clusters = best_k
            else:
                print("Could not determine optimal k via Silhouette Score, defaulting to 7.")
                n_clusters = 7 # Fallback if no suitable k is found

            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = self.kmeans_model.fit_predict(scaled_clustering_features)
            df.loc[df_for_clustering.index, "target_weather"] = cluster_labels
            df["target_weather"] = df["target_weather"].astype(int)
        else:
            print("Warning: Not enough data for clustering. Target weather will be missing.")
            df["target_weather"] = np.nan

        df.dropna(inplace=True) # Drop rows where target_weather might be NaN
        return df, self.kmeans_model, self.kmeans_scaler


    def process_historical_data_for_arima(self, raw_data):
        delhi_data = raw_data.get("delhi")
        if not delhi_data or not delhi_data.get("hourly") or not delhi_data["hourly"].get("time"):
            print("Error: Missing or incomplete historical data for Delhi (ARIMA).")
            return pd.DataFrame()

        delhi_hourly = delhi_data["hourly"]
        if not delhi_hourly.get("temperature_2m"): # Check if temp data exists
             print("Error: temperature_2m data missing in delhi_hourly for ARIMA.")
             return pd.DataFrame()

        hourly_df = pd.DataFrame({
            "time": pd.to_datetime(delhi_hourly["time"]),
            "temperature_2m": delhi_hourly["temperature_2m"]
        })
        hourly_df.set_index("time", inplace=True)
        hourly_df.dropna(subset=["temperature_2m"], inplace=True)
        daily_avg_temp = hourly_df["temperature_2m"].resample("D").mean().dropna()
        return daily_avg_temp


# --- Hybrid Model Trainer Class ---
class HybridModelTrainer:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.rf_model = None
        self.arima_model_fit = None
        self.weather_accuracies = []
        self.temperature_maes = []
        self.scaler = None # Main scaler for RF features
        self.kmeans_model = None # Fitted KMeans model
        self.kmeans_scaler = None # Fitted scaler for KMeans features

    def train_and_evaluate(self, dataframe_full, temp_series_full, train_start_idx,
                           train_end_idx, test_start_idx, test_end_idx, session_num):
        rf_accuracy, arima_mae = np.nan, np.nan
        rf_train_df = dataframe_full.iloc[train_start_idx:train_end_idx]
        rf_test_df = dataframe_full.iloc[test_start_idx:test_end_idx]

        if not rf_train_df.empty and not rf_test_df.empty:
            X_rf_train = rf_train_df.drop(["target_weather", "target_temperature"], axis=1, errors='ignore')
            y_rf_train = rf_train_df["target_weather"]
            X_rf_test = rf_test_df.drop(["target_weather", "target_temperature"], axis=1, errors='ignore')
            y_rf_test = rf_test_df["target_weather"]

            all_rf_columns = dataframe_full.drop(["target_weather", "target_temperature"], axis=1, errors='ignore').columns
            X_rf_train = X_rf_train.reindex(columns=all_rf_columns, fill_value=0)
            X_rf_test = X_rf_test.reindex(columns=all_rf_columns, fill_value=0)

            self.scaler = StandardScaler()
            X_rf_train_scaled = self.scaler.fit_transform(X_rf_train)
            X_rf_test_scaled = self.scaler.transform(X_rf_test)
            
            # Store feature names after fitting the scaler
            if hasattr(self.scaler, 'feature_names_in_'):
                self.rf_feature_columns_ = list(self.scaler.feature_names_in_)
            else: # Fallback for older sklearn versions or if not available
                self.rf_feature_columns_ = list(X_rf_train.columns)


            self.rf_model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
            self.rf_model.fit(X_rf_train_scaled, y_rf_train)
            y_rf_pred = self.rf_model.predict(X_rf_test_scaled)
            rf_accuracy = accuracy_score(y_rf_test, y_rf_pred)
            self.weather_accuracies.append(rf_accuracy)
        else: self.weather_accuracies.append(np.nan)

        arima_train_series = temp_series_full.iloc[train_start_idx:train_end_idx]
        arima_test_series = temp_series_full.iloc[test_start_idx:test_end_idx]

        if not arima_train_series.empty and not arima_test_series.empty:
            try:
                model_auto = pm.auto_arima(
                    arima_train_series, start_p=1, start_q=1, test="adf", max_p=5, max_q=5, m=7,
                    d=None, seasonal=True, start_P=0, start_Q=0, max_P=2, max_Q=2, D=None,
                    trace=False, error_action="ignore", suppress_warnings=True, stepwise=True
                )
                self.arima_model_fit = model_auto
                arima_predictions = []
                history = list(arima_train_series)
                for t in range(len(arima_test_series)):
                    try:
                        yhat = self.arima_model_fit.predict(n_periods=1)[0]
                        arima_predictions.append(yhat)
                        history.append(arima_test_series.iloc[t])
                        # self.arima_model_fit.update(arima_test_series.iloc[t]) # Optional: update model with true value
                    except Exception: arima_predictions.append(np.nan)
                
                valid_predictions = [p for p in arima_predictions if not np.isnan(p)]
                valid_test_series = arima_test_series[:len(valid_predictions)]
                if valid_predictions:
                    arima_mae = mean_absolute_error(valid_test_series, valid_predictions)
                    self.temperature_maes.append(arima_mae)
                else: self.temperature_maes.append(np.nan)
            except Exception: self.temperature_maes.append(np.nan)
        else: self.temperature_maes.append(np.nan)
        return rf_accuracy, arima_mae

    def get_average_metrics(self):
        avg_accuracy = np.nanmean(self.weather_accuracies) if self.weather_accuracies else 0
        avg_mae = np.nanmean(self.temperature_maes) if self.temperature_maes else 0
        return avg_accuracy, avg_mae

    def save_models(self):
        """Saves the trained scaler, KMeans, RF, ARIMA models, and RF feature columns."""
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        print(f"\nSaving models to {self.config.model_save_dir}/...")

        joblib.dump(self.scaler, os.path.join(self.config.model_save_dir, "scaler.joblib"))
        print(f"Scaler saved.")
        joblib.dump(self.kmeans_model, os.path.join(self.config.model_save_dir, "kmeans_model.joblib"))
        print(f"KMeans model saved.")
        joblib.dump(self.kmeans_scaler, os.path.join(self.config.model_save_dir, "cluster_scaler.joblib")) # Save KMeans scaler
        print(f"KMeans Scaler saved.")
        joblib.dump(self.rf_model, os.path.join(self.config.model_save_dir, "random_forest_model.joblib"))
        print(f"Random Forest model saved.")
        joblib.dump(self.arima_model_fit, os.path.join(self.config.model_save_dir, "arima_model.joblib"))
        print(f"ARIMA model saved.")
        
        # Save RF feature columns obtained from the scaler
        if hasattr(self, 'rf_feature_columns_') and self.rf_feature_columns_:
             joblib.dump(self.rf_feature_columns_, os.path.join(self.config.model_save_dir, "rf_feature_columns.joblib"))
             print(f"Random Forest feature columns saved.")
        else:
            print("Warning: RF feature columns not available to save.")


# --- Main Execution Function ---
def main():
    config = WeatherConfig()
    db_manager = DatabaseManager(config)
    data_fetcher = WeatherDataFetcher(config, db_manager)
    data_processor = DataProcessor(config) # Initialize DataProcessor
    hybrid_trainer = HybridModelTrainer(config)

    print("Initializing: Starting weather forecasting model.")
    surrounding_locations = {
        name: dict(zip(["latitude", "longitude"], data_processor.calculate_destination_point(
            config.delhi_latitude, config.delhi_longitude, config.radius_km, bearing)))
        for name, bearing in zip(config.region_names, config.region_bearings)
    }

    raw_historical_data = {
        "delhi": {"hourly": {var: [] for var in config.weather_variables + ["time"]}},
        "surrounding_regions": {
            name: {"hourly": {var: [] for var in config.weather_variables + ["time"]}}
            for name in config.region_names
        },
    }

    # Fetch data (simplified loop for brevity, original logic for month-by-month is fine)
    print("Fetching historical data from database...")
    delhi_full_data = data_fetcher.fetch_historical_weather_data(
        config.delhi_latitude, config.delhi_longitude,
        config.train_start_date, config.train_end_date, region_name="delhi"
    )
    if delhi_full_data and delhi_full_data.get("hourly"):
        for key in raw_historical_data["delhi"]["hourly"]:
            raw_historical_data["delhi"]["hourly"][key].extend(delhi_full_data["hourly"].get(key, []))

    for region_name, coords in surrounding_locations.items():
        region_full_data = data_fetcher.fetch_historical_weather_data(
            coords["latitude"], coords["longitude"],
            config.train_start_date, config.train_end_date, region_name=region_name
        )
        if region_full_data and region_full_data.get("hourly"):
            for key in raw_historical_data["surrounding_regions"][region_name]["hourly"]:
                 raw_historical_data["surrounding_regions"][region_name]["hourly"][key].extend(region_full_data["hourly"].get(key, []))
    print("Data fetched from database.")

    # Process data using the enhanced DataProcessor
    weather_dataset_full, kmeans_model_fitted, kmeans_scaler_fitted = data_processor.process_historical_data(raw_historical_data)
    
    # Pass the fitted KMeans model and its scaler to the trainer
    hybrid_trainer.kmeans_model = kmeans_model_fitted
    hybrid_trainer.kmeans_scaler = kmeans_scaler_fitted # This is crucial

    daily_avg_temp_series_full = data_processor.process_historical_data_for_arima(raw_historical_data)

    if weather_dataset_full.empty or daily_avg_temp_series_full.empty:
        print("\nFailed to create datasets. Cannot train.")
        db_manager.close_connection()
        return

    num_training_sessions = 10
    session_test_days = 30
    total_days = len(weather_dataset_full)
    initial_train_period = 365
    first_test_window_start_idx = max(initial_train_period, total_days - (num_training_sessions * session_test_days))
    sessions = []
    for i in range(num_training_sessions):
        test_start_idx = first_test_window_start_idx + i * session_test_days
        test_end_idx = min(test_start_idx + session_test_days, total_days)
        train_start_idx = 0
        train_end_idx = test_start_idx
        if train_end_idx > 0 and test_end_idx > test_start_idx:
            sessions.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))

    print(f"\n--- Starting {len(sessions)} Training Sessions ---")
    print(f"{'Training Step':<15} {'RF_Accuracy':<15} {'ARIMA (MAE) Temp':<18}")
    print(f"{'-'*15:<15} {'-'*15:<15} {'-'*18:<18}")

    for i, (train_start, train_end, test_start, test_end) in enumerate(sessions):
        rf_acc, arima_temp_mae = hybrid_trainer.train_and_evaluate(
            weather_dataset_full, daily_avg_temp_series_full,
            train_start, train_end, test_start, test_end, i + 1
        )
        rf_acc_str = f"{rf_acc:.2f}" if not np.isnan(rf_acc) else "N/A"
        arima_mae_str = f"{arima_temp_mae:.2f} °C" if not np.isnan(arima_temp_mae) else "N/A"
        print(f"{i + 1:<15} {rf_acc_str:<15} {arima_mae_str:<18}")

    avg_accuracy, avg_mae = hybrid_trainer.get_average_metrics()
    print("\n--- Hybrid Ensemble Model Overall Summary ---")
    print(f"Average Weather Category Accuracy: {avg_accuracy:.2f}")
    print(f"Average Temperature MAE: {avg_mae:.2f} °C")
    print("------------------------------------------")

    hybrid_trainer.save_models() # Call save_models on the trainer instance
    db_manager.close_connection()

if __name__ == "__main__":
    main()