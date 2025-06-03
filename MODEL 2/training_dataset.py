import math
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, silhouette_score
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
        self.sarimax_exog_features = [
            'delhi_avg_relative_humidity_2m',
            'delhi_avg_wind_speed_10m',
            'delhi_avg_cloud_cover',
            'delhi_dew_point'
        ]
        self.max_min_temp_features = [
            'target_temperature_avg', 
            'delhi_avg_cloud_cover',
            # Season dummies will be added dynamically
        ]


        self.db_name = "Project_Weather_Forecasting"
        self.db_user = "postgres"
        self.db_password = "1234"
        self.db_host = "localhost"
        self.db_port = "5432"

        self.model_save_dir = "MODEL 2/trained_models"
        self.kmeans_k_range = range(3, 11)


# --- Database Manager Class ---
class DatabaseManager:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.config.db_name, user=self.config.db_user,
                password=self.config.db_password, host=self.config.db_host, port=self.config.db_port)
            self.conn.autocommit = True
            print("Initializing: Database connected.")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}"); self.conn = None

    def create_tables(self):
        if not self.conn: print("Cannot create tables: No database connection."); return
        cur = self.conn.cursor()
        try:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS delhi_hourly_weather (
                    id SERIAL PRIMARY KEY, timestamp TIMESTAMP UNIQUE, temperature_2m REAL,
                    relative_humidity_2m REAL, wind_speed_10m REAL, wind_direction_10m REAL,
                    weather_code INTEGER, precipitation REAL, cloud_cover REAL);""")
            cur.execute(
                """CREATE TABLE IF NOT EXISTS surrounding_hourly_weather (
                    id SERIAL PRIMARY KEY, region_name TEXT NOT NULL, latitude REAL, longitude REAL,
                    timestamp TIMESTAMP, temperature_2m REAL, relative_humidity_2m REAL,
                    wind_speed_10m REAL, wind_direction_10m REAL, weather_code INTEGER,
                    precipitation REAL, cloud_cover REAL, UNIQUE (region_name, timestamp));""")
        except psycopg2.Error as e: print(f"Error creating tables: {e}")
        finally: cur.close()

    def _load_data_from_db(self, query_sql, params): # Helper to reduce repetition
        if not self.conn: return None
        cur = self.conn.cursor()
        try:
            cur.execute(query_sql, params)
            rows = cur.fetchall()
            if not rows: return None
            data = {"hourly": {col: [] for col in self.config.weather_variables + ["time"]}, "hourly_units": {}}
            # Populate hourly_units (can be simplified if not strictly needed for processing)
            data["hourly_units"] = {
                "time": "iso8601", "temperature_2m": "celsius", "relative_humidity_2m": "%",
                "wind_speed_10m": "km/h", "wind_direction_10m": "degrees",
                "weather_code": "wmo code", "precipitation": "mm", "cloud_cover": "%"
            }
            for row in rows:
                data["hourly"]["time"].append(row[0].isoformat())
                for i, var_name in enumerate(self.config.weather_variables):
                    data["hourly"][var_name].append(row[i+1])
            return data
        except psycopg2.Error as e: print(f"Error loading data type: {e}"); return None
        finally: cur.close()

    def load_delhi_data(self, start_date, end_date):
        query = sql.SQL("""SELECT timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m, wind_direction_10m, weather_code, precipitation, cloud_cover
                           FROM delhi_hourly_weather WHERE timestamp >= %s AND timestamp <= %s ORDER BY timestamp;""")
        return self._load_data_from_db(query, (start_date, end_date + timedelta(days=1)))

    def load_surrounding_data(self, region_name, start_date, end_date):
        query = sql.SQL("""SELECT timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m, wind_direction_10m, weather_code, precipitation, cloud_cover
                           FROM surrounding_hourly_weather WHERE region_name = %s AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp;""")
        return self._load_data_from_db(query, (region_name, start_date, end_date + timedelta(days=1)))

    def close_connection(self):
        if self.conn: self.conn.close()

# --- Data Fetcher Class ---
class WeatherDataFetcher:
    def __init__(self, config: WeatherConfig, db_manager: DatabaseManager):
        self.config = config; self.db_manager = db_manager

    def fetch_historical_weather_data(self, latitude, longitude, start_date, end_date, region_name=None):
        db_data = None
        if self.db_manager.conn:
            if region_name == "delhi": db_data = self.db_manager.load_delhi_data(start_date, end_date)
            elif region_name: db_data = self.db_manager.load_surrounding_data(region_name, start_date, end_date)
            if db_data and db_data.get("hourly") and db_data["hourly"].get("time") and len(db_data["hourly"]["time"]) > 0:
                try:
                    db_start_dt = datetime.fromisoformat(db_data["hourly"]["time"][0]).date()
                    db_end_dt = datetime.fromisoformat(db_data["hourly"]["time"][-1]).date()
                    if db_start_dt <= start_date and db_end_dt >= end_date: return db_data
                except (IndexError, ValueError) as e:
                     print(f"Warning: Problem with time data for {region_name or 'Delhi'} ({start_date}-{end_date}): {e}")
        return None

# --- Data Processor Class ---
class DataProcessor:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.kmeans_model = None
        self.kmeans_scaler = None 

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

    def _calculate_dew_point(self, T, RH):
        RH = np.clip(RH, 1, 100) 
        a = 17.27; b = 237.7
        alpha = ((a * T) / (b + T)) + np.log(RH / 100.0)
        T_dp = (b * alpha) / (a - alpha)
        return T_dp
        
    def process_historical_data(self, raw_data):
        dataset = []
        delhi_data = raw_data.get("delhi"); surrounding_data = raw_data.get("surrounding_regions", {})
        if not delhi_data or not delhi_data.get("hourly") or not delhi_data["hourly"].get("time"): return pd.DataFrame(), None, None
        delhi_hourly = delhi_data["hourly"]; time_stamps = delhi_hourly["time"]
        num_hours = len(time_stamps); hours_per_day = 24; num_days = num_hours // hours_per_day
        for i in range(num_days - 1):
            start_hour_index = i * hours_per_day; end_hour_index = start_hour_index + hours_per_day
            current_day_data = {}; current_day_time = datetime.fromisoformat(time_stamps[start_hour_index]).date()
            current_day_data["day_of_week"] = current_day_time.weekday(); current_day_data["month"] = current_day_time.month
            current_day_data["day_of_year"] = current_day_time.timetuple().tm_yday; current_day_data["week_of_year"] = current_day_time.isocalendar()[1]
            current_day_data["season"] = self.get_season(current_day_time.month)
            for var in self.config.weather_variables:
                hourly_values = delhi_hourly.get(var, [])[start_hour_index:end_hour_index]; valid_hourly_values = [val for val in hourly_values if val is not None]
                if valid_hourly_values: current_day_data[f"delhi_avg_{var}"] = np.mean(valid_hourly_values); current_day_data[f"delhi_max_{var}"] = np.max(valid_hourly_values); current_day_data[f"delhi_min_{var}"] = np.min(valid_hourly_values)
                else: current_day_data[f"delhi_avg_{var}"] = np.nan; current_day_data[f"delhi_max_{var}"] = np.nan; current_day_data[f"delhi_min_{var}"] = np.nan
            for region_name in self.config.region_names: 
                region_data_for_current_day = surrounding_data.get(region_name)
                if region_data_for_current_day and isinstance(region_data_for_current_day, dict):
                    region_hourly = region_data_for_current_day.get("hourly", {})
                    if region_hourly and region_hourly.get("time"):
                        region_time_stamps = region_hourly["time"]; region_start_index, region_end_index = None, None
                        for j, ts_str in enumerate(region_time_stamps):
                            if datetime.fromisoformat(ts_str).date() == current_day_time:
                                if region_start_index is None: region_start_index = j
                                region_end_index = j + 1
                        if region_start_index is not None and region_end_index is not None:
                            for var in self.config.weather_variables:
                                hourly_values = region_hourly.get(var, [])[region_start_index:region_end_index]; valid_hourly_values = [val for val in hourly_values if val is not None]
                                if valid_hourly_values: current_day_data[f"{region_name}_avg_{var}"] = np.mean(valid_hourly_values); current_day_data[f"{region_name}_max_{var}"] = np.max(valid_hourly_values); current_day_data[f"{region_name}_min_{var}"] = np.min(valid_hourly_values)
                                else: ForVarInLoopHelper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"], is_current_day_data=True) # Pass flag
                        else: ForVarInLoopHelper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"], is_current_day_data=True)
                    else: ForVarInLoopHelper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"], is_current_day_data=True)
                else: ForVarInLoopHelper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"], is_current_day_data=True)

            next_day_start_hour_index = end_hour_index; next_day_end_hour_index = next_day_start_hour_index + hours_per_day
            if next_day_end_hour_index <= num_hours:
                next_day_hourly_data = {var: delhi_hourly.get(var, [])[next_day_start_hour_index:next_day_end_hour_index] for var in self.config.weather_variables}
                valid_next_day_temps = [t for t in next_day_hourly_data.get("temperature_2m", []) if t is not None]
                if valid_next_day_temps:
                    current_day_data["target_temperature"] = np.mean(valid_next_day_temps)
                    current_day_data["target_max_temperature"] = np.max(valid_next_day_temps)
                    current_day_data["target_min_temperature"] = np.min(valid_next_day_temps)
                    dataset.append(current_day_data)
        df = pd.DataFrame(dataset)
        if df.empty: return pd.DataFrame(), None, None
        if 'month' in df.columns: df['month_sin'] = np.sin(2*np.pi*df['month']/12); df['month_cos'] = np.cos(2*np.pi*df['month']/12)
        if 'day_of_week' in df.columns: df['day_of_week_sin'] = np.sin(2*np.pi*df['day_of_week']/7); df['day_of_week_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
        df = pd.get_dummies(df, columns=["season"], drop_first=True)
        if 'delhi_avg_temperature_2m' in df.columns and 'delhi_avg_relative_humidity_2m' in df.columns:
            df['delhi_dew_point'] = self._calculate_dew_point(df['delhi_avg_temperature_2m'], df['delhi_avg_relative_humidity_2m'])
        df.dropna(inplace=True); df.reset_index(drop=True, inplace=True)
        features_to_lag_and_roll = ["delhi_avg_temperature_2m", "delhi_avg_relative_humidity_2m", "delhi_avg_wind_speed_10m", "delhi_avg_precipitation", "delhi_avg_cloud_cover", "delhi_dew_point"]
        lags = [1,2,3,7,14,30]; rolling_windows = [3,7]; new_features_df = pd.DataFrame(index=df.index)
        for feature in features_to_lag_and_roll: 
            if feature in df.columns: 
                for lag in lags: new_features_df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
                for window in rolling_windows:
                    new_features_df[f"{feature}_rolling_mean_{window}d"] = df[feature].rolling(window=window, min_periods=1).mean()
                    new_features_df[f"{feature}_rolling_min_{window}d"] = df[feature].rolling(window=window, min_periods=1).min()
                    new_features_df[f"{feature}_rolling_max_{window}d"] = df[feature].rolling(window=window, min_periods=1).max()
                    new_features_df[f"{feature}_rolling_std_{window}d"] = df[feature].rolling(window=window, min_periods=1).std()
                new_features_df[f"{feature}_diff_1d"] = df[feature].diff(1)
        
        if 'delhi_avg_temperature_2m' in df.columns and 'delhi_avg_relative_humidity_2m' in df.columns: new_features_df["delhi_temp_x_humidity"] = df["delhi_avg_temperature_2m"] * df["delhi_avg_relative_humidity_2m"]
        if 'delhi_avg_wind_speed_10m' in df.columns and 'delhi_avg_cloud_cover' in df.columns: new_features_df["delhi_wind_x_cloud"] = df["delhi_avg_wind_speed_10m"] * df["delhi_avg_cloud_cover"]
        poly_features_cols = ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        if all(col in df.columns for col in poly_features_cols):
            poly = PolynomialFeatures(degree=2, include_bias=False); poly_input_df = df[poly_features_cols].copy(); poly_input_df.fillna(poly_input_df.mean(), inplace=True); poly_transformed = poly.fit_transform(poly_input_df); poly_feature_names = poly.get_feature_names_out(poly_features_cols); full_poly_df = pd.DataFrame(poly_transformed, columns=poly_feature_names, index=df.index); new_features_df = pd.concat([new_features_df, full_poly_df], axis=1)
        
        df = pd.concat([df, new_features_df], axis=1); df = df.loc[:,~df.columns.duplicated()]; df.dropna(inplace=True); df.reset_index(drop=True, inplace=True)
        clustering_features = ["delhi_avg_temperature_2m", "delhi_avg_relative_humidity_2m", "delhi_avg_wind_speed_10m", "delhi_avg_wind_direction_10m", "delhi_avg_weather_code", "delhi_avg_precipitation", "delhi_avg_cloud_cover"]
        df_for_clustering = df[clustering_features].copy(); df_for_clustering.dropna(inplace=True)
        if not df_for_clustering.empty: 
            self.kmeans_scaler = StandardScaler(); scaled_clustering_features = self.kmeans_scaler.fit_transform(df_for_clustering); best_k, best_silhouette_score = -1, -1
            print("\nFinding optimal k for KMeans using Silhouette Score...")
            for k_test in self.config.kmeans_k_range: 
                if k_test >= len(df_for_clustering): continue
                kmeans_test = KMeans(n_clusters=k_test, random_state=42, n_init='auto'); cluster_labels_test = kmeans_test.fit_predict(scaled_clustering_features)
                if len(np.unique(cluster_labels_test)) > 1 and len(np.unique(cluster_labels_test)) < len(df_for_clustering):
                    try: 
                        score = silhouette_score(scaled_clustering_features, cluster_labels_test)
                        print(f"  k={k_test}, Silhouette Score: {score:.4f}") # Debug print
                        # CORRECTED SYNTAX FOR THE IF BLOCK
                        if score > best_silhouette_score: 
                            best_silhouette_score = score
                            best_k = k_test
                    except ValueError: pass # Error during silhouette score calculation
            n_clusters = best_k if best_k != -1 else 3
            print(f"KMeans k selected: {n_clusters}") # Debug print
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto'); cluster_labels = self.kmeans_model.fit_predict(scaled_clustering_features); df.loc[df_for_clustering.index, "target_weather"] = cluster_labels
            if "target_weather" in df.columns: df["target_weather"] = df["target_weather"].astype(int)
        else: df["target_weather"] = np.nan
        df.dropna(subset=['target_weather', 'target_temperature', 'target_max_temperature', 'target_min_temperature'], inplace=True)
        return df, self.kmeans_model, self.kmeans_scaler

# Helper function (moved outside the class for clarity or can be a static method)
def ForVarInLoopHelper(weather_variables, data_dict, region_name, aggs, is_current_day_data=False):
    for var_ in weather_variables:
        for agg_ in aggs:
            data_dict[f"{region_name}_{agg_}_{var_}"] = np.nan


# --- Hybrid Model Trainer Class ---
class HybridModelTrainer:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.rf_model = None
        self.sarimax_model_fit = None
        self.max_temp_model = None
        self.min_temp_model = None
        self.weather_accuracies = []
        self.temperature_maes = []
        self.max_temp_maes = []
        self.min_temp_maes = []
        self.scaler = None 
        self.kmeans_model = None
        self.kmeans_scaler = None 
        self.rf_feature_columns_ = None

    def train_and_evaluate(self, dataframe_full, train_start_idx, train_end_idx, 
                           test_start_idx, test_end_idx, session_num, total_sessions):
        rf_accuracy, sarimax_mae, max_temp_mae, min_temp_mae = np.nan, np.nan, np.nan, np.nan
        
        train_df = dataframe_full.iloc[train_start_idx:train_end_idx].copy()
        test_df = dataframe_full.iloc[test_start_idx:test_end_idx].copy()

        if not train_df.empty and not test_df.empty:
            rf_drop_cols = ['target_weather', 'target_temperature', 'target_max_temperature', 'target_min_temperature']
            X_rf_train = train_df.drop(columns=rf_drop_cols, errors='ignore')
            y_rf_train = train_df["target_weather"]
            X_rf_test = test_df.drop(columns=rf_drop_cols, errors='ignore')
            y_rf_test = test_df["target_weather"]

            all_rf_columns = dataframe_full.drop(columns=rf_drop_cols, errors='ignore').columns
            X_rf_train = X_rf_train.reindex(columns=all_rf_columns, fill_value=0)
            X_rf_test = X_rf_test.reindex(columns=all_rf_columns, fill_value=0)

            current_scaler = StandardScaler() # Use a temporary scaler for each session's train set
            X_rf_train_scaled = current_scaler.fit_transform(X_rf_train)
            X_rf_test_scaled = current_scaler.transform(X_rf_test) # Transform test with current session's scaler
            
            if session_num == total_sessions: # If it's the last session, store this scaler and features
                self.scaler = current_scaler
                self.rf_feature_columns_ = list(getattr(self.scaler, 'feature_names_in_', X_rf_train.columns))
            
            current_rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
            current_rf_model.fit(X_rf_train_scaled, y_rf_train)
            if session_num == total_sessions: self.rf_model = current_rf_model

            y_rf_pred_weather_test = current_rf_model.predict(X_rf_test_scaled) # Predictions on test set
            rf_accuracy = accuracy_score(y_rf_test, y_rf_pred_weather_test)
            self.weather_accuracies.append(rf_accuracy)
            
            # Get RF predictions on the training set itself for training Max/Min models
            y_rf_pred_weather_train = current_rf_model.predict(X_rf_train_scaled)

        else: 
            self.weather_accuracies.append(np.nan)
            y_rf_pred_weather_train = None # Ensure it's defined

        y_sarimax_train = train_df['target_temperature']
        y_sarimax_test = test_df['target_temperature']
        
        exog_train_df = train_df[self.config.sarimax_exog_features].fillna(0)
        exog_test_df = test_df[self.config.sarimax_exog_features].fillna(0)
        
        # Align index only if y_sarimax_train/test is not empty
        if not y_sarimax_train.empty and not exog_train_df.empty:
            exog_train_df = exog_train_df.loc[y_sarimax_train.index]
        if not y_sarimax_test.empty and not exog_test_df.empty:
            exog_test_df = exog_test_df.loc[y_sarimax_test.index]

        y_pred_sarimax_train_values = None 

        if not y_sarimax_train.empty:
            try:
                exog_train_np = exog_train_df.values if not exog_train_df.empty else None
                
                current_sarimax_model = pm.auto_arima(
                    y_sarimax_train, exogenous=exog_train_np,
                    start_p=1, start_q=1, test="adf", max_p=5, max_q=5, m=7, 
                    d=None, seasonal=True, start_P=0, start_Q=0, max_P=2, max_Q=2, D=None,
                    trace=False, error_action="ignore", suppress_warnings=True, stepwise=True )
                
                if session_num == total_sessions: self.sarimax_model_fit = current_sarimax_model
                
                if exog_train_np is not None:
                    y_pred_sarimax_train_values = current_sarimax_model.predict_in_sample(X=exog_train_np)
                else:
                    y_pred_sarimax_train_values = current_sarimax_model.predict_in_sample()
                y_pred_sarimax_train_values = pd.Series(y_pred_sarimax_train_values, index=y_sarimax_train.index)


                if not y_sarimax_test.empty:
                    exog_test_np = exog_test_df.values if not exog_test_df.empty else None
                    if exog_test_np is not None and len(exog_test_np) == len(y_sarimax_test):
                        y_pred_sarimax_test_values = current_sarimax_model.predict(n_periods=len(y_sarimax_test), exogenous=exog_test_np)
                        sarimax_mae = mean_absolute_error(y_sarimax_test, y_pred_sarimax_test_values)
                        self.temperature_maes.append(sarimax_mae)
                    else: self.temperature_maes.append(np.nan)
                else: self.temperature_maes.append(np.nan)

            except Exception as e: 
                print(f"Error during SARIMAX for session {session_num}: {e}")
                self.temperature_maes.append(np.nan)
        else: self.temperature_maes.append(np.nan)

        # --- 3. Models for Max and Min Temperature ---
        X_max_min_train = pd.DataFrame(index=train_df.index)
        if y_pred_sarimax_train_values is not None:
             X_max_min_train['target_temperature_avg'] = y_pred_sarimax_train_values
        else: 
            X_max_min_train['target_temperature_avg'] = train_df['target_temperature'] # Fallback

        if y_rf_pred_weather_train is not None and len(y_rf_pred_weather_train) == len(train_df):
            X_max_min_train['target_weather'] = y_rf_pred_weather_train # Use RF's view of weather category
        else:
            X_max_min_train['target_weather'] = train_df['target_weather'] # Fallback
        
        for f_col in self.config.max_min_temp_features: # Use defined features
            if f_col != 'target_temperature_avg' and f_col in train_df.columns: # Avoid re-adding avg temp
                 X_max_min_train[f_col] = train_df[f_col]
        
        active_season_cols = [col for col in train_df.columns if col.startswith("season_")]
        for col in active_season_cols: X_max_min_train[col] = train_df[col]
        
        X_max_min_train.fillna(0, inplace=True)

        y_max_train = train_df['target_max_temperature']
        y_min_train = train_df['target_min_temperature']

        current_max_temp_model, current_min_temp_model = None, None
        if not X_max_min_train.empty and not y_max_train.empty and not y_min_train.empty:
            current_max_temp_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10) 
            current_max_temp_model.fit(X_max_min_train, y_max_train)
            
            current_min_temp_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
            current_min_temp_model.fit(X_max_min_train, y_min_train)

            if session_num == total_sessions:
                self.max_temp_model = current_max_temp_model
                self.min_temp_model = current_min_temp_model

            X_max_min_test = pd.DataFrame(index=test_df.index)
            if 'y_pred_sarimax_test_values' in locals() and y_pred_sarimax_test_values is not None and len(y_pred_sarimax_test_values) == len(test_df):
                 X_max_min_test['target_temperature_avg'] = y_pred_sarimax_test_values
            else: X_max_min_test['target_temperature_avg'] = test_df['target_temperature']

            if 'y_rf_pred_weather_test' in locals() and len(y_rf_pred_weather_test) == len(test_df):
                 X_max_min_test['target_weather'] = y_rf_pred_weather_test
            else: X_max_min_test['target_weather'] = test_df['target_weather']
            
            for f_col in self.config.max_min_temp_features:
                if f_col != 'target_temperature_avg' and f_col in test_df.columns: X_max_min_test[f_col] = test_df[f_col]
            for col in active_season_cols: 
                 if col in test_df.columns: X_max_min_test[col] = test_df[col]
            
            X_max_min_test = X_max_min_test.reindex(columns=X_max_min_train.columns, fill_value=0)
            X_max_min_test.fillna(0, inplace=True)

            if not X_max_min_test.empty:
                y_pred_max_temp = current_max_temp_model.predict(X_max_min_test)
                y_pred_min_temp = current_min_temp_model.predict(X_max_min_test)
                max_temp_mae = mean_absolute_error(test_df['target_max_temperature'], y_pred_max_temp)
                min_temp_mae = mean_absolute_error(test_df['target_min_temperature'], y_pred_min_temp)
                self.max_temp_maes.append(max_temp_mae)
                self.min_temp_maes.append(min_temp_mae)
            else: self.max_temp_maes.append(np.nan); self.min_temp_maes.append(np.nan)
        else: self.max_temp_maes.append(np.nan); self.min_temp_maes.append(np.nan)

        return rf_accuracy, sarimax_mae, max_temp_mae, min_temp_mae

    def get_average_metrics(self):
        avg_accuracy = np.nanmean(self.weather_accuracies) if self.weather_accuracies else np.nan
        avg_sarimax_mae = np.nanmean(self.temperature_maes) if self.temperature_maes else np.nan
        avg_max_temp_mae = np.nanmean(self.max_temp_maes) if self.max_temp_maes else np.nan
        avg_min_temp_mae = np.nanmean(self.min_temp_maes) if self.min_temp_maes else np.nan
        return avg_accuracy, avg_sarimax_mae, avg_max_temp_mae, avg_min_temp_mae

    def save_models(self):
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        print(f"\nSaving models to {self.config.model_save_dir}/...")
        if self.scaler: joblib.dump(self.scaler, os.path.join(self.config.model_save_dir, "scaler.joblib"))
        if self.kmeans_model: joblib.dump(self.kmeans_model, os.path.join(self.config.model_save_dir, "kmeans_model.joblib"))
        if self.kmeans_scaler: joblib.dump(self.kmeans_scaler, os.path.join(self.config.model_save_dir, "cluster_scaler.joblib"))
        if self.rf_model: joblib.dump(self.rf_model, os.path.join(self.config.model_save_dir, "random_forest_model.joblib"))
        if self.sarimax_model_fit: joblib.dump(self.sarimax_model_fit, os.path.join(self.config.model_save_dir, "sarimax_model.joblib"))
        if self.max_temp_model: joblib.dump(self.max_temp_model, os.path.join(self.config.model_save_dir, "max_temp_model.joblib"))
        if self.min_temp_model: joblib.dump(self.min_temp_model, os.path.join(self.config.model_save_dir, "min_temp_model.joblib"))
        if hasattr(self, 'rf_feature_columns_') and self.rf_feature_columns_:
             joblib.dump(self.rf_feature_columns_, os.path.join(self.config.model_save_dir, "rf_feature_columns.joblib"))
             print(f"Random Forest feature columns saved: {len(self.rf_feature_columns_)} columns.")
        else: print("Warning: RF feature columns not available to save.")


# --- Main Execution Function ---
sessions = [] 

def main():
    global sessions 
    config = WeatherConfig()
    db_manager = DatabaseManager(config)
    data_fetcher = WeatherDataFetcher(config, db_manager)
    data_processor = DataProcessor(config) 
    hybrid_trainer = HybridModelTrainer(config)

    print("Initializing: Starting weather forecasting model training.")
    raw_historical_data = { "delhi": {"hourly": {var: [] for var in config.weather_variables + ["time"]}}, "surrounding_regions": { name: {"hourly": {var: [] for var in config.weather_variables + ["time"]}} for name in config.region_names}, }
    print("Fetching historical data from database...") 
    delhi_full_data = data_fetcher.fetch_historical_weather_data(config.delhi_latitude, config.delhi_longitude, config.train_start_date, config.train_end_date, region_name="delhi")
    if delhi_full_data and delhi_full_data.get("hourly"):
        for key in raw_historical_data["delhi"]["hourly"]: raw_historical_data["delhi"]["hourly"][key].extend(delhi_full_data["hourly"].get(key, []))
    surrounding_locations = { name: dict(zip(["latitude", "longitude"], data_processor.calculate_destination_point(config.delhi_latitude, config.delhi_longitude, config.radius_km, bearing))) for name, bearing in zip(config.region_names, config.region_bearings)}
    for region_name, coords in surrounding_locations.items():
        region_full_data = data_fetcher.fetch_historical_weather_data(coords["latitude"], coords["longitude"], config.train_start_date, config.train_end_date, region_name=region_name)
        if region_full_data and region_full_data.get("hourly"):
            for key in raw_historical_data["surrounding_regions"][region_name]["hourly"]: raw_historical_data["surrounding_regions"][region_name]["hourly"][key].extend(region_full_data["hourly"].get(key, []))
    print("Data fetched from database.")


    weather_dataset_full, kmeans_model_fitted, kmeans_scaler_fitted = data_processor.process_historical_data(raw_historical_data)
    
    hybrid_trainer.kmeans_model = kmeans_model_fitted
    hybrid_trainer.kmeans_scaler = kmeans_scaler_fitted

    if weather_dataset_full.empty:
        print("\nFailed to create datasets. Cannot train."); db_manager.close_connection(); return

    num_training_sessions = 10 
    session_test_days = 30  
    total_days_in_df = len(weather_dataset_full)
    min_initial_train_days = 365 * 2 
    
    sessions = [] # Clear global sessions before repopulating
    if total_days_in_df < min_initial_train_days + session_test_days:
        print(f"Not enough data ({total_days_in_df} days) for robust rolling window CV. Using single 80/20 split.")
        train_end_idx_main = int(total_days_in_df * 0.8)
        if train_end_idx_main > 0 and train_end_idx_main < total_days_in_df:
             sessions = [(0, train_end_idx_main, train_end_idx_main, total_days_in_df)]
        else: print("Cannot create even a single valid train/test split. Exiting."); return
    else:
        num_possible_sessions = (total_days_in_df - min_initial_train_days) // session_test_days
        actual_num_sessions = min(num_training_sessions, num_possible_sessions)
        if actual_num_sessions < 1: actual_num_sessions = 1

        for i in range(actual_num_sessions):
            current_test_start_idx = total_days_in_df - ((actual_num_sessions - i) * session_test_days)
            current_test_end_idx = min(current_test_start_idx + session_test_days, total_days_in_df)
            current_train_end_idx = current_test_start_idx
            current_train_start_idx = 0 

            if current_train_end_idx >= min_initial_train_days and current_test_start_idx < current_test_end_idx :
                 sessions.append((current_train_start_idx, current_train_end_idx, current_test_start_idx, current_test_end_idx))

    if not sessions:
        print("No valid training sessions could be created. Check data length and session parameters."); return

    print(f"\n--- Starting {len(sessions)} Training Session(s) ---")
    header = f"{'Session':<10} {'RF Acc':<10} {'SARIMAX MAE':<15} {'MaxT MAE':<10} {'MinT MAE':<10}"
    print(header); print("-" * len(header))

    for i, (train_start, train_end, test_start, test_end) in enumerate(sessions):
        print(f"Running Session {i+1}/{len(sessions)}: Train {train_start}-{train_end-1}, Test {test_start}-{test_end-1}")
        rf_acc, sarimax_temp_mae, max_t_mae, min_t_mae = hybrid_trainer.train_and_evaluate(
            weather_dataset_full, 
            train_start, train_end, test_start, test_end, i + 1, len(sessions)
        )
        rf_acc_str = f"{rf_acc:.2f}" if not np.isnan(rf_acc) else "N/A"
        sarimax_mae_str = f"{sarimax_temp_mae:.2f}" if not np.isnan(sarimax_temp_mae) else "N/A"
        max_t_mae_str = f"{max_t_mae:.2f}" if not np.isnan(max_t_mae) else "N/A"
        min_t_mae_str = f"{min_t_mae:.2f}" if not np.isnan(min_t_mae) else "N/A"
        print(f"{i + 1:<10} {rf_acc_str:<10} {sarimax_mae_str:<15} {max_t_mae_str:<10} {min_t_mae_str:<10}")

    avg_accuracy, avg_sarimax_mae, avg_max_temp_mae, avg_min_temp_mae = hybrid_trainer.get_average_metrics()
    print("\n--- Hybrid Ensemble Model Overall Summary ---")
    print(f"Average Weather Category Accuracy: {avg_accuracy:.2f}")
    print(f"Average SARIMAX Temperature MAE: {avg_sarimax_mae:.2f} °C")
    print(f"Average Max Temperature MAE: {avg_max_temp_mae:.2f} °C")
    print(f"Average Min Temperature MAE: {avg_min_temp_mae:.2f} °C")
    print("------------------------------------------")

    hybrid_trainer.save_models() 
    db_manager.close_connection()

if __name__ == "__main__":
    main()