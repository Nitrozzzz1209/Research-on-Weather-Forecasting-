import math
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    silhouette_score,
)
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
from pmdarima.arima import StepwiseContext # For faster auto_arima in loops
import warnings
import joblib
import os

# Suppress FutureWarning messages from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.utils.validation')


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
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "wind_direction_10m", "weather_code", "precipitation", "cloud_cover",
        ]
        self.sarimax_exog_features = [
            'delhi_avg_relative_humidity_2m', 'delhi_avg_wind_speed_10m',
            'delhi_avg_cloud_cover', 'delhi_dew_point'
        ]
        # Define the base features for Max/Min temperature models
        # Season dummies and predicted weather category will be added dynamically
        self.max_min_temp_features_base = [
            'target_temperature_avg', # SARIMAX predicted average temperature
            'delhi_avg_cloud_cover',  # Example, can be from forecasted exog
            'month_sin', 'month_cos', 
            'day_of_week_sin', 'day_of_week_cos'
        ]


        self.db_name = "Project_Weather_Forecasting"
        self.db_user = "postgres"
        self.db_password = "1234"
        self.db_host = "localhost"
        self.db_port = "5432"

        self.model_save_dir = "MODEL 2/trained_models/" 
        self.kmeans_k_range = range(3, 11)


# --- Database Manager Class ---
class DatabaseManager:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.config.db_name, user=self.config.db_user,
                password=self.config.db_password, host=self.config.db_host, port=self.config.db_port)
            self.conn.autocommit = True
            print("Initializing: Database connected (Forecaster).")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}"); self.conn = None

    def _load_data_from_db(self, query_sql, params):
        if not self.conn: return None
        cur = self.conn.cursor()
        try:
            cur.execute(query_sql, params)
            rows = cur.fetchall()
            if not rows: return None
            data = {"hourly": {col: [] for col in self.config.weather_variables + ["time"]}, "hourly_units": {}}
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
        
    def process_historical_data(self, raw_data, kmeans_model_for_prediction=None, kmeans_scaler_for_prediction=None):
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
                                else: _for_var_in_loop_helper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"])
                        else: _for_var_in_loop_helper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"])
                    else: _for_var_in_loop_helper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"])
                else: _for_var_in_loop_helper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"])
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
                    new_features_df[f"{feature}_rolling_mean_{window}d"] = df[feature].rolling(window=window, min_periods=1).mean(); new_features_df[f"{feature}_rolling_min_{window}d"] = df[feature].rolling(window=window, min_periods=1).min(); new_features_df[f"{feature}_rolling_max_{window}d"] = df[feature].rolling(window=window, min_periods=1).max(); new_features_df[f"{feature}_rolling_std_{window}d"] = df[feature].rolling(window=window, min_periods=1).std()
                new_features_df[f"{feature}_diff_1d"] = df[feature].diff(1)
        if 'delhi_avg_temperature_2m' in df.columns and 'delhi_avg_relative_humidity_2m' in df.columns: new_features_df["delhi_temp_x_humidity"] = df["delhi_avg_temperature_2m"] * df["delhi_avg_relative_humidity_2m"]
        if 'delhi_avg_wind_speed_10m' in df.columns and 'delhi_avg_cloud_cover' in df.columns: new_features_df["delhi_wind_x_cloud"] = df["delhi_avg_wind_speed_10m"] * df["delhi_avg_cloud_cover"]
        poly_features_cols = ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        if all(col in df.columns for col in poly_features_cols):
            poly = PolynomialFeatures(degree=2, include_bias=False); poly_input_df = df[poly_features_cols].copy(); poly_input_df.fillna(poly_input_df.mean(), inplace=True); poly_transformed = poly.fit_transform(poly_input_df); poly_feature_names = poly.get_feature_names_out(poly_features_cols); full_poly_df = pd.DataFrame(poly_transformed, columns=poly_feature_names, index=df.index); new_features_df = pd.concat([new_features_df, full_poly_df], axis=1)
        df = pd.concat([df, new_features_df], axis=1); df = df.loc[:,~df.columns.duplicated()]; df.dropna(inplace=True); df.reset_index(drop=True, inplace=True)
        clustering_features = ["delhi_avg_temperature_2m", "delhi_avg_relative_humidity_2m", "delhi_avg_wind_speed_10m", "delhi_avg_wind_direction_10m", "delhi_avg_weather_code", "delhi_avg_precipitation", "delhi_avg_cloud_cover"]
        df_for_clustering = df[clustering_features].copy(); df_for_clustering.dropna(inplace=True)
        final_kmeans_model, final_kmeans_scaler = kmeans_model_for_prediction, kmeans_scaler_for_prediction
        if kmeans_model_for_prediction and kmeans_scaler_for_prediction: 
            if not df_for_clustering.empty:
                df_for_clustering_reindexed = df_for_clustering.reindex(columns=kmeans_scaler_for_prediction.feature_names_in_, fill_value=0); scaled_clustering_features = kmeans_scaler_for_prediction.transform(df_for_clustering_reindexed); cluster_labels = kmeans_model_for_prediction.predict(scaled_clustering_features); df.loc[df_for_clustering.index, "target_weather"] = cluster_labels; df["target_weather"] = df["target_weather"].astype(int)
            else: df["target_weather"] = np.nan
        else: 
            if not df_for_clustering.empty:
                self.kmeans_scaler = StandardScaler(); scaled_clustering_features = self.kmeans_scaler.fit_transform(df_for_clustering); best_k, best_silhouette_score = -1, -1
                print("\nFinding optimal k for KMeans (DataProcessor)...")
                for k_test in self.config.kmeans_k_range: 
                    if k_test >= len(df_for_clustering): continue
                    kmeans_test = KMeans(n_clusters=k_test, random_state=42, n_init='auto'); cluster_labels_test = kmeans_test.fit_predict(scaled_clustering_features)
                    if len(np.unique(cluster_labels_test)) > 1 and len(np.unique(cluster_labels_test)) < len(df_for_clustering):
                        try: 
                            score = silhouette_score(scaled_clustering_features, cluster_labels_test)
                            print(f"  k={k_test}, Silhouette Score: {score:.4f}")
                            if score > best_silhouette_score: 
                                best_silhouette_score = score
                                best_k = k_test
                        except ValueError: pass
                n_clusters = best_k if best_k != -1 else 3; print(f"KMeans k selected: {n_clusters}")
                self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto'); cluster_labels = self.kmeans_model.fit_predict(scaled_clustering_features); df.loc[df_for_clustering.index, "target_weather"] = cluster_labels;
                if "target_weather" in df.columns: df["target_weather"] = df["target_weather"].astype(int)
                final_kmeans_model, final_kmeans_scaler = self.kmeans_model, self.kmeans_scaler
            else: df["target_weather"] = np.nan
        df.dropna(subset=['target_weather', 'target_temperature', 'target_max_temperature', 'target_min_temperature'], inplace=True)
        return df, final_kmeans_model, final_kmeans_scaler

    def process_historical_data_for_arima(self, raw_data):
        delhi_data = raw_data.get("delhi")
        if not delhi_data or not delhi_data.get("hourly") or not delhi_data["hourly"].get("time"): return pd.DataFrame()
        delhi_hourly = delhi_data["hourly"]
        if not delhi_hourly.get("temperature_2m"): return pd.DataFrame()
        if not delhi_hourly["time"] or not delhi_hourly["temperature_2m"]: return pd.DataFrame()
        try:
            hourly_df = pd.DataFrame({"time": pd.to_datetime(delhi_hourly["time"]), "temperature_2m": delhi_hourly["temperature_2m"]})
            hourly_df.set_index("time", inplace=True); hourly_df.dropna(subset=["temperature_2m"], inplace=True)
            daily_avg_temp = hourly_df["temperature_2m"].resample("D").mean().dropna()
            return daily_avg_temp
        except Exception as e: print(f"Error processing data for ARIMA: {e}"); return pd.DataFrame()

def _for_var_in_loop_helper(weather_variables, data_dict, region_name, aggs):
    for var_ in weather_variables:
        for agg_ in aggs: data_dict[f"{region_name}_{agg_}_{var_}"] = np.nan

# --- Hybrid Model Trainer Class (For fallback training) ---
class HybridModelTrainer:
    def __init__(self, config: WeatherConfig):
        self.config = config; self.rf_model = None; self.sarimax_model_fit = None 
        self.max_temp_model = None; self.min_temp_model = None 
        self.scaler = None; self.kmeans_model = None; self.kmeans_scaler = None
        self.rf_feature_columns_ = None

    def train_and_evaluate(self, dataframe_full, train_start_idx, train_end_idx, 
                           test_start_idx, test_end_idx, session_num, total_sessions): 
        train_df = dataframe_full 
        if not train_df.empty:
            rf_drop_cols = ['target_weather', 'target_temperature', 'target_max_temperature', 'target_min_temperature']
            X_rf_train = train_df.drop(columns=rf_drop_cols, errors='ignore')
            y_rf_train = train_df["target_weather"]
            
            self.scaler = StandardScaler()
            X_rf_train_scaled = self.scaler.fit_transform(X_rf_train)
            self.rf_feature_columns_ = list(getattr(self.scaler, 'feature_names_in_', X_rf_train.columns))
            
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
            self.rf_model.fit(X_rf_train_scaled, y_rf_train)
            print(f"Fallback RF model trained on {len(X_rf_train)} samples.")
            y_rf_pred_weather_train = self.rf_model.predict(X_rf_train_scaled) 
        else:
            y_rf_pred_weather_train = None

        y_sarimax_train = train_df['target_temperature']
        exog_train_df = train_df[self.config.sarimax_exog_features].fillna(0)
        if not y_sarimax_train.empty and not exog_train_df.empty:
             exog_train_df = exog_train_df.loc[y_sarimax_train.index] 

        y_pred_sarimax_train_values = None
        if not y_sarimax_train.empty:
            try:
                exog_train_np = exog_train_df.values if not exog_train_df.empty else None
                self.sarimax_model_fit = pm.auto_arima(
                    y_sarimax_train, exogenous=exog_train_np,
                    start_p=1, start_q=1, test="adf", max_p=3, max_q=3, m=7, 
                    d=None, seasonal=True, start_P=0,start_Q=0, max_P=1,max_Q=1, D=None, 
                    trace=False, error_action="ignore", suppress_warnings=True, stepwise=True)
                print(f"Fallback SARIMAX: order {self.sarimax_model_fit.order}, seasonal {self.sarimax_model_fit.seasonal_order}.")
                if exog_train_np is not None:
                    y_pred_sarimax_train_values = self.sarimax_model_fit.predict_in_sample(X=exog_train_np)
                else:
                    y_pred_sarimax_train_values = self.sarimax_model_fit.predict_in_sample()
                y_pred_sarimax_train_values = pd.Series(y_pred_sarimax_train_values, index=y_sarimax_train.index)
            except Exception as e: print(f"Error training fallback SARIMAX: {e}")
        
        X_max_min_train = pd.DataFrame(index=train_df.index)
        X_max_min_train['target_temperature_avg'] = y_pred_sarimax_train_values if y_pred_sarimax_train_values is not None else train_df['target_temperature']
        X_max_min_train['target_weather'] = y_rf_pred_weather_train if y_rf_pred_weather_train is not None else train_df['target_weather']
        
        for f_col in self.config.max_min_temp_features_base: 
            if f_col != 'target_temperature_avg' and f_col in train_df.columns:
                 X_max_min_train[f_col] = train_df[f_col]
        active_season_cols = [col for col in train_df.columns if col.startswith("season_")]
        for col in active_season_cols: X_max_min_train[col] = train_df[col]
        X_max_min_train.fillna(0, inplace=True)

        y_max_train = train_df['target_max_temperature']
        y_min_train = train_df['target_min_temperature']

        if not X_max_min_train.empty and not y_max_train.empty and not y_min_train.empty:
            self.max_temp_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
            self.max_temp_model.fit(X_max_min_train, y_max_train)
            self.min_temp_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
            self.min_temp_model.fit(X_max_min_train, y_min_train)
            print("Fallback Max/Min temperature models trained.")
        
        return np.nan, np.nan, np.nan, np.nan 

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
        print("Models saved from fallback trainer (if any were trained).")


# --- Weather Forecaster Class ---
class WeatherForecaster:
    def __init__(self, config: WeatherConfig):
        self.config = config; self.scaler = None; self.kmeans_model = None
        self.kmeans_scaler = None; self.rf_model = None; self.sarimax_model = None 
        self.max_temp_model = None; self.min_temp_model = None 
        self.rf_feature_columns = None
        self.poly_features_for_rf = ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        self.poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
        dummy_poly_data = pd.DataFrame(np.zeros((1, len(self.poly_features_for_rf))), columns=self.poly_features_for_rf)
        self.poly_transformer.fit(dummy_poly_data)
        self.load_models()

    def load_models(self):
        model_dir = self.config.model_save_dir 
        required_files = ["scaler.joblib", "kmeans_model.joblib", "cluster_scaler.joblib",
                          "random_forest_model.joblib", "sarimax_model.joblib", 
                          "max_temp_model.joblib", "min_temp_model.joblib", 
                          "rf_feature_columns.joblib"]
        all_models_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

        if all_models_exist:
            try:
                self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
                self.kmeans_model = joblib.load(os.path.join(model_dir, "kmeans_model.joblib"))
                self.kmeans_scaler = joblib.load(os.path.join(model_dir, "cluster_scaler.joblib"))
                self.rf_model = joblib.load(os.path.join(model_dir, "random_forest_model.joblib"))
                self.sarimax_model = joblib.load(os.path.join(model_dir, "sarimax_model.joblib")) 
                self.max_temp_model = joblib.load(os.path.join(model_dir, "max_temp_model.joblib"))
                self.min_temp_model = joblib.load(os.path.join(model_dir, "min_temp_model.joblib"))
                self.rf_feature_columns = joblib.load(os.path.join(model_dir, "rf_feature_columns.joblib"))
                print(f"All models loaded successfully from {model_dir}.")
            except Exception as e:
                print(f"Error loading models from {model_dir}: {e}. Fallback training may be required.")
                self.scaler = self.kmeans_model = self.kmeans_scaler = self.rf_model = self.sarimax_model = \
                self.max_temp_model = self.min_temp_model = self.rf_feature_columns = None
        else:
            print(f"One or more model files not found in {model_dir}. Fallback training may be required.")
            self.scaler = self.kmeans_model = self.kmeans_scaler = self.rf_model = self.sarimax_model = \
            self.max_temp_model = self.min_temp_model = self.rf_feature_columns = None

    def map_cluster_to_weather_description(self, cluster_label): 
        mapping = {0: "Type A (e.g. Clear)", 1: "Type B (e.g. Partly Cloudy)", 2: "Type C (e.g. Overcast/Rainy)"} 
        if pd.isna(cluster_label): return "Unknown"
        return mapping.get(int(cluster_label), f"Weather Type {int(cluster_label)}")

    def get_season(self, month): return DataProcessor(self.config).get_season(month)
    def _calculate_dew_point(self, T, RH): return DataProcessor(self.config)._calculate_dew_point(T,RH)

    def prepare_features_for_prediction(self, current_date_obj, history_df_input, 
                                        predicted_avg_temp=None, 
                                        predicted_weather_category=None,
                                        current_day_forecasted_exog=None): # New argument
        if self.rf_feature_columns is None: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 

        history_df = history_df_input.copy()
        current_day_data_dict = {}

        # 1. Base features for current forecast day
        current_day_data_dict["month"] = current_date_obj.month
        current_day_data_dict["day_of_week"] = current_date_obj.weekday()
        current_day_data_dict["day_of_year"] = current_date_obj.timetuple().tm_yday
        current_day_data_dict["week_of_year"] = current_date_obj.isocalendar()[1]
        current_day_data_dict['month_sin'] = np.sin(2 * np.pi * current_day_data_dict['month'] / 12)
        current_day_data_dict['month_cos'] = np.cos(2 * np.pi * current_day_data_dict['month'] / 12)
        current_day_data_dict['day_of_week_sin'] = np.sin(2 * np.pi * current_day_data_dict['day_of_week'] / 7)
        current_day_data_dict['day_of_week_cos'] = np.cos(2 * np.pi * current_day_data_dict['day_of_week'] / 7)

        current_season_name = self.get_season(current_day_data_dict["month"])
        current_day_data_dict['season_Spring'] = 1 if current_season_name == "Spring" else 0
        current_day_data_dict['season_Summer'] = 1 if current_season_name == "Summer" else 0
        current_day_data_dict['season_Winter'] = 1 if current_season_name == "Winter" else 0

        current_day_data_dict["delhi_avg_temperature_2m"] = predicted_avg_temp # Always use SARIMAX predicted avg temp

        # Use forecasted exogenous variables for the current day if available
        # These are also part of self.config.sarimax_exog_features
        if current_day_forecasted_exog is not None:
            for exog_col in self.config.sarimax_exog_features:
                if exog_col in current_day_forecasted_exog: # Check if key exists in the dict
                    current_day_data_dict[exog_col] = current_day_forecasted_exog[exog_col]
                elif not history_df.empty and exog_col in history_df.columns: # Fallback to history if specific exog not in forecast
                    current_day_data_dict[exog_col] = history_df[exog_col].iloc[-1]
                else:
                    current_day_data_dict[exog_col] = np.nan # Final fallback
        else: # If no forecasted exog provided, use history for these
            for exog_col in self.config.sarimax_exog_features:
                current_day_data_dict[exog_col] = history_df[exog_col].iloc[-1] if not history_df.empty and exog_col in history_df.columns else np.nan


        # For other Delhi vars NOT in sarimax_exog_features, carry forward from history
        other_delhi_vars_to_carry = [
            var for var in ["wind_direction_10m", "weather_code", "precipitation"] 
            if f"delhi_avg_{var}" not in self.config.sarimax_exog_features # Ensure we don't overwrite exog
        ]
        for var_suffix in other_delhi_vars_to_carry:
            col = f"delhi_avg_{var_suffix}"
            current_day_data_dict[col] = history_df[col].iloc[-1] if not history_df.empty and col in history_df.columns else np.nan
        
        # Recalculate dew point if humidity/temp were updated by forecasted exog
        temp_val = current_day_data_dict.get("delhi_avg_temperature_2m", np.nan)
        rh_val = current_day_data_dict.get("delhi_avg_relative_humidity_2m", np.nan) 
        current_day_data_dict['delhi_dew_point'] = self._calculate_dew_point(temp_val, rh_val) if not (np.isnan(temp_val) or np.isnan(rh_val)) else np.nan


        # Carry forward surrounding region features (these are not forecasted independently yet)
        for r_name in self.config.region_names:
            for var_suffix in self.config.weather_variables:
                col = f"{r_name}_avg_{var_suffix}"
                current_day_data_dict[col] = history_df[col].iloc[-1] if not history_df.empty and col in history_df.columns else np.nan
        
        current_day_unscaled_df = pd.DataFrame([current_day_data_dict])

        # 2. Combine with history for derived features
        source_cols_for_derived = list(set(
            [col for col in self.config.sarimax_exog_features if col in current_day_unscaled_df.columns] + 
            [col for col in self.poly_features_for_rf if col in current_day_unscaled_df.columns] + 
            ["delhi_avg_temperature_2m", "delhi_avg_precipitation", "delhi_avg_cloud_cover"] 
        ))
        actual_source_cols_in_history = [col for col in source_cols_for_derived if col in history_df.columns]
        actual_source_cols_in_current = [col for col in source_cols_for_derived if col in current_day_unscaled_df.columns]
        common_source_cols = list(set(actual_source_cols_in_history) & set(actual_source_cols_in_current))
        if not common_source_cols and actual_source_cols_in_current: 
             common_source_cols = actual_source_cols_in_current
        elif not common_source_cols: 
            print(f"Warning: No common source columns for derived features. History cols: {history_df.columns.tolist()}, Current cols: {current_day_unscaled_df.columns.tolist()}")
            combined_df_for_derived = current_day_unscaled_df.copy() # Use only current day if no commonality
        else:
            combined_df_for_derived = pd.concat(
                [history_df[common_source_cols].tail(30), # Use relevant part of history
                current_day_unscaled_df[common_source_cols]], ignore_index=True)


        # 3. Calculate derived features (lags, rolls, interactions, poly)
        derived_features_df = pd.DataFrame(index=combined_df_for_derived.index)
        features_to_lag_roll = [col for col in ["delhi_avg_temperature_2m", "delhi_avg_relative_humidity_2m", "delhi_avg_wind_speed_10m", "delhi_avg_precipitation", "delhi_avg_cloud_cover", "delhi_dew_point"] if col in combined_df_for_derived.columns]
        lags = [1,2,3,7,14,30]; rolling_windows = [3,7]
        for feature in features_to_lag_roll: 
            for lag in lags: derived_features_df[f"{feature}_lag_{lag}"] = combined_df_for_derived[feature].shift(lag)
            for window in rolling_windows: 
                derived_features_df[f"{feature}_rolling_mean_{window}d"] = combined_df_for_derived[feature].rolling(window=window, min_periods=1).mean()
                derived_features_df[f"{feature}_rolling_min_{window}d"] = combined_df_for_derived[feature].rolling(window=window, min_periods=1).min()
                derived_features_df[f"{feature}_rolling_max_{window}d"] = combined_df_for_derived[feature].rolling(window=window, min_periods=1).max()
                derived_features_df[f"{feature}_rolling_std_{window}d"] = combined_df_for_derived[feature].rolling(window=window, min_periods=1).std()
            derived_features_df[f"{feature}_diff_1d"] = combined_df_for_derived[feature].diff(1)
        if "delhi_avg_temperature_2m" in combined_df_for_derived.columns and "delhi_avg_relative_humidity_2m" in combined_df_for_derived.columns: derived_features_df["delhi_temp_x_humidity"] = combined_df_for_derived["delhi_avg_temperature_2m"] * combined_df_for_derived["delhi_avg_relative_humidity_2m"]
        if "delhi_avg_wind_speed_10m" in combined_df_for_derived.columns and "delhi_avg_cloud_cover" in combined_df_for_derived.columns: derived_features_df["delhi_wind_x_cloud"] = combined_df_for_derived["delhi_avg_wind_speed_10m"] * combined_df_for_derived["delhi_avg_cloud_cover"]
        if all(col in combined_df_for_derived.columns for col in self.poly_features_for_rf):
            poly_input = combined_df_for_derived[self.poly_features_for_rf].copy(); poly_input.fillna(poly_input.mean(), inplace=True); poly_transformed = self.poly_transformer.transform(poly_input); poly_names = self.poly_transformer.get_feature_names_out(self.poly_features_for_rf); poly_df = pd.DataFrame(poly_transformed, columns=poly_names, index=combined_df_for_derived.index); derived_features_df = pd.concat([derived_features_df, poly_df], axis=1)

        final_derived_features_for_current_day = derived_features_df.iloc[[-1]].reset_index(drop=True)
        
        all_features_for_current_day_unscaled = pd.concat([current_day_unscaled_df.reset_index(drop=True), final_derived_features_for_current_day], axis=1)
        all_features_for_current_day_unscaled = all_features_for_current_day_unscaled.loc[:, ~all_features_for_current_day_unscaled.columns.duplicated()]
        
        features_for_rf_scaling = all_features_for_current_day_unscaled.reindex(columns=self.rf_feature_columns, fill_value=0)
        features_for_rf_scaling.fillna(0, inplace=True)
        
        # Ensure feature order for scaling and prediction matches rf_feature_columns
        # Convert to NumPy array for RF prediction if model was fitted on array (to avoid UserWarning)
        # However, if rf_feature_columns is used, passing DataFrame is fine if model also saw names.
        # For now, assume model can handle DataFrame with correct column names.
        scaled_features_for_rf_df = pd.DataFrame(self.scaler.transform(features_for_rf_scaling[self.rf_feature_columns]), columns=self.rf_feature_columns) if self.scaler else pd.DataFrame()


        features_for_max_min = pd.DataFrame(index=[0])
        features_for_max_min['target_temperature_avg'] = predicted_avg_temp
        features_for_max_min['target_weather'] = predicted_weather_category if predicted_weather_category is not None else 0 
        
        for f_col in self.config.max_min_temp_features_base:
             if f_col not in ['target_temperature_avg', 'target_weather'] and f_col in all_features_for_current_day_unscaled.columns:
                 features_for_max_min[f_col] = all_features_for_current_day_unscaled[f_col].iloc[0]
        
        active_season_cols = [col for col in self.rf_feature_columns if col.startswith("season_")]
        for col in active_season_cols:
            features_for_max_min[col] = all_features_for_current_day_unscaled[col].iloc[0] if col in all_features_for_current_day_unscaled.columns else 0
        
        if self.max_temp_model and hasattr(self.max_temp_model, 'feature_names_in_'):
            max_min_model_expected_cols = self.max_temp_model.feature_names_in_
        else: 
            max_min_model_expected_cols = self.config.max_min_temp_features_base[:] + active_season_cols
            max_min_model_expected_cols = sorted(list(set(max_min_model_expected_cols)))


        features_for_max_min = features_for_max_min.reindex(columns=max_min_model_expected_cols, fill_value=0)
        features_for_max_min.fillna(0, inplace=True)

        unscaled_for_history = all_features_for_current_day_unscaled.reindex(columns=self.rf_feature_columns, fill_value=0)
        unscaled_for_history.fillna(0, inplace=True)

        return scaled_features_for_rf_df, features_for_max_min, unscaled_for_history


    def run_forecast_pipeline(self, db_manager: DatabaseManager):
        if not (self.scaler and self.kmeans_model and self.rf_model and self.sarimax_model and 
                self.max_temp_model and self.min_temp_model and self.rf_feature_columns and self.kmeans_scaler): 
            print("One or more models missing. Attempting fallback training.")
            self._train_models_if_needed(db_manager)
            self.load_models() 
            if not (self.scaler and self.kmeans_model and self.rf_model and self.sarimax_model and 
                    self.max_temp_model and self.min_temp_model and self.rf_feature_columns and self.kmeans_scaler):
                print("Models could not be loaded/trained. Cannot proceed."); return pd.DataFrame()

        print("\n--- Starting 7-Day Weather Prediction ---")
        data_fetcher = WeatherDataFetcher(self.config, db_manager)
        dp_recent_data = DataProcessor(self.config)

        today = date.today()
        history_fetch_end_date = today - timedelta(days=1)
        history_fetch_start_date = history_fetch_end_date - timedelta(days=89) 
        prediction_start_date = today

        print(f"Fetching recent data from {history_fetch_start_date} to {history_fetch_end_date}.")
        raw_recent_data = { "delhi": {"hourly": {var: [] for var in self.config.weather_variables + ["time"]}},
                            "surrounding_regions": {name: {"hourly": {var: [] for var in self.config.weather_variables + ["time"]}}
                                                    for name in self.config.region_names}}
        delhi_hist_data = data_fetcher.fetch_historical_weather_data(self.config.delhi_latitude, self.config.delhi_longitude, history_fetch_start_date, history_fetch_end_date, region_name="delhi")
        if delhi_hist_data and delhi_hist_data.get("hourly"):
            for key in raw_recent_data["delhi"]["hourly"]: raw_recent_data["delhi"]["hourly"][key].extend(delhi_hist_data["hourly"].get(key, []))
        surrounding_locations = {name: dict(zip(["latitude", "longitude"], dp_recent_data.calculate_destination_point(self.config.delhi_latitude, self.config.delhi_longitude, self.config.radius_km, bearing))) for name, bearing in zip(self.config.region_names, self.config.region_bearings)}
        for region_name, coords in surrounding_locations.items():
            region_hist_data = data_fetcher.fetch_historical_weather_data(coords["latitude"], coords["longitude"], history_fetch_start_date, history_fetch_end_date, region_name=region_name)
            if region_hist_data and region_hist_data.get("hourly"):
                 for key in raw_recent_data["surrounding_regions"][region_name]["hourly"]: raw_recent_data["surrounding_regions"][region_name]["hourly"][key].extend(region_hist_data["hourly"].get(key,[]))
        print("Recent data fetched.")


        recent_weather_df_processed, _, _ = dp_recent_data.process_historical_data(
            raw_recent_data, kmeans_model_for_prediction=self.kmeans_model, kmeans_scaler_for_prediction=self.kmeans_scaler)
        recent_temp_series = dp_recent_data.process_historical_data_for_arima(raw_recent_data) 

        if recent_weather_df_processed.empty or recent_temp_series.empty:
            print("Not enough recent data for predictions after processing."); return pd.DataFrame()
        if self.rf_feature_columns is None: print("ERROR: rf_feature_columns not loaded."); return pd.DataFrame()

        iterative_history_df = recent_weather_df_processed.reindex(columns=self.rf_feature_columns, fill_value=0)
        for col_to_add in ['target_temperature', 'target_max_temperature', 'target_min_temperature', 'target_weather']:
            if col_to_add in recent_weather_df_processed.columns:
                 iterative_history_df[col_to_add] = recent_weather_df_processed[col_to_add]
        
        max_lag_days = 30 
        iterative_history_df = iterative_history_df.tail(max_lag_days + 14).copy() 

        forecast_results = []
        
        future_exog_df = pd.DataFrame(columns=self.config.sarimax_exog_features) 
        print("Forecasting exogenous variables for SARIMAX...")
        for exog_feature_name in self.config.sarimax_exog_features:
            if exog_feature_name in recent_weather_df_processed.columns:
                exog_series = recent_weather_df_processed[exog_feature_name].dropna()
                if len(exog_series) >= 14: 
                    try:
                        with StepwiseContext(max_steps=10): 
                            simple_exog_model = pm.auto_arima(
                                exog_series, start_p=1, start_q=0, max_p=2, max_q=1,
                                m=1, seasonal=False, d=None, D=0, trace=False, 
                                error_action='ignore', suppress_warnings=True, stepwise=True)
                        future_exog_df[exog_feature_name] = simple_exog_model.predict(n_periods=7)
                    except Exception as e_exog:
                        print(f"Could not forecast exog var {exog_feature_name}: {e_exog}. Using last value."); future_exog_df[exog_feature_name] = [exog_series.iloc[-1]] * 7 if not exog_series.empty else [0] * 7
                else: print(f"Not enough data for {exog_feature_name} forecast. Using last value."); future_exog_df[exog_feature_name] = [exog_series.iloc[-1]] * 7 if not exog_series.empty else [0] * 7
            else: print(f"Exog feature {exog_feature_name} not in recent data. Using zeros."); future_exog_df[exog_feature_name] = [0] * 7
        print("Exogenous variable forecast complete.")
        future_exog_values_for_sarimax = future_exog_df.values

        try:
            if not recent_temp_series.empty:
                last_actual_exog_unaligned = recent_weather_df_processed[self.config.sarimax_exog_features].fillna(0)
                common_index = recent_temp_series.index.intersection(last_actual_exog_unaligned.index)
                if not common_index.empty:
                    last_actual_exog = last_actual_exog_unaligned.loc[common_index].tail(len(recent_temp_series.tail(30)))
                    temp_series_for_update = recent_temp_series.loc[common_index].tail(len(recent_temp_series.tail(30)))
                    if len(last_actual_exog) == len(temp_series_for_update) and len(temp_series_for_update) > 0 :
                        self.sarimax_model.update(temp_series_for_update, exogenous=last_actual_exog.values) 
                        print(f"SARIMAX model updated with recent data up to {temp_series_for_update.index.max()}.")
                    else: print(f"SARIMAX update skipped: Mismatch/empty data for update. Temp len: {len(temp_series_for_update)}, Exog len: {len(last_actual_exog)}")
                else: print("SARIMAX update skipped: No common index between temp series and exog for update.")
        except Exception as e:
            print(f"Could not update SARIMAX model, using as is: {e}")
        
        avg_temp_forecasts = self.sarimax_model.predict(n_periods=7, exogenous=future_exog_values_for_sarimax)
        
        predicted_weather_category_for_day = None 

        for i in range(7):
            forecast_date_obj = prediction_start_date + timedelta(days=i)
            try: 
                predicted_avg_temperature = avg_temp_forecasts.iloc[i] if isinstance(avg_temp_forecasts, pd.Series) else avg_temp_forecasts[i]
            except IndexError:
                predicted_avg_temperature = np.nan 
            
            current_day_forecasted_exog_dict = future_exog_df.iloc[i].to_dict() if i < len(future_exog_df) else None

            scaled_rf_feats, feats_for_max_min, unscaled_hist_row = self.prepare_features_for_prediction(
                forecast_date_obj, iterative_history_df.copy(), 
                predicted_avg_temperature, 
                predicted_weather_category_for_day, 
                current_day_forecasted_exog=current_day_forecasted_exog_dict 
            )

            if not scaled_rf_feats.empty and self.rf_model:
                try:
                    # Pass NumPy array if model was fitted on array (to avoid UserWarning)
                    # Ensure columns are in correct order before .values
                    scaled_rf_feats_np = scaled_rf_feats[self.rf_feature_columns].values
                    predicted_weather_category_for_day = int(self.rf_model.predict(scaled_rf_feats_np)[0])
                except Exception as e: print(f"Error during RF prediction for {forecast_date_obj}: {e}"); predicted_weather_category_for_day = 0 
            else: predicted_weather_category_for_day = 0 

            weather_condition_desc = self.map_cluster_to_weather_description(predicted_weather_category_for_day)
            
            feats_for_max_min['target_weather'] = predicted_weather_category_for_day
            if current_day_forecasted_exog_dict:
                for exog_col, exog_val in current_day_forecasted_exog_dict.items():
                    if exog_col in feats_for_max_min.columns:
                        feats_for_max_min[exog_col] = exog_val

            predicted_max_temp, predicted_min_temp = predicted_avg_temperature + 3, predicted_avg_temperature - 3 
            if not feats_for_max_min.empty and self.max_temp_model and self.min_temp_model:
                try:
                    # Ensure feature names for Max/Min models are consistent
                    # If Max/Min models were trained with feature names, use them
                    if hasattr(self.max_temp_model, 'feature_names_in_'):
                        expected_max_min_cols = self.max_temp_model.feature_names_in_
                    else: # Fallback: construct based on config and active seasons
                        active_season_cols = [col for col in self.rf_feature_columns if col.startswith("season_")]
                        expected_max_min_cols = self.config.max_min_temp_features_base[:] + active_season_cols + ['target_weather']
                        expected_max_min_cols = sorted(list(set(expected_max_min_cols)))
                    
                    feats_for_max_min_aligned = feats_for_max_min.reindex(columns=expected_max_min_cols, fill_value=0)
                    feats_for_max_min_aligned.fillna(0, inplace=True)

                    predicted_max_temp = self.max_temp_model.predict(feats_for_max_min_aligned)[0]
                    predicted_min_temp = self.min_temp_model.predict(feats_for_max_min_aligned)[0]
                except Exception as e: print(f"Error during Max/Min temp prediction for {forecast_date_obj}: {e}")
            
            forecast_results.append({
                "Date": forecast_date_obj.strftime("%Y-%m-%d"), "Max Temp": f"{predicted_max_temp:.1f}",
                "Min Temp": f"{predicted_min_temp:.1f}", "Avg Temp": f"{predicted_avg_temperature:.1f}",
                "Conditions": weather_condition_desc,})

            if not unscaled_hist_row.empty:
                unscaled_hist_row['target_weather'] = predicted_weather_category_for_day
                unscaled_hist_row['target_temperature'] = predicted_avg_temperature 
                unscaled_hist_row['target_max_temperature'] = predicted_max_temp 
                unscaled_hist_row['target_min_temperature'] = predicted_min_temp 
                if current_day_forecasted_exog_dict: 
                    for exog_col, exog_val in current_day_forecasted_exog_dict.items():
                        if exog_col in unscaled_hist_row.columns:
                            unscaled_hist_row[exog_col] = exog_val

                row_to_append = unscaled_hist_row.reindex(columns=iterative_history_df.columns, fill_value=0)
                iterative_history_df = pd.concat([iterative_history_df, row_to_append], ignore_index=True)
                if len(iterative_history_df) > (max_lag_days + 14) : iterative_history_df = iterative_history_df.iloc[1:]
            else: print(f"Warning: Could not obtain unscaled features for {forecast_date_obj} to update history.")

        _print_forecast_results(forecast_results)
        return pd.DataFrame(forecast_results)

    def _train_models_if_needed(self, db_manager: DatabaseManager): 
        print("\n--- Initiating Fallback Model Training ---")
        dp_fallback = DataProcessor(self.config)
        trainer_fallback = HybridModelTrainer(self.config) 
        fetcher_fallback = WeatherDataFetcher(self.config, db_manager)
        raw_training_data = { "delhi": {"hourly": {var: [] for var in self.config.weather_variables + ["time"]}},
                            "surrounding_regions": {name: {"hourly": {var: [] for var in self.config.weather_variables + ["time"]}}
                                                    for name in self.config.region_names}}
        delhi_train_data = fetcher_fallback.fetch_historical_weather_data(self.config.delhi_latitude, self.config.delhi_longitude, self.config.train_start_date, self.config.train_end_date, region_name="delhi")
        if delhi_train_data and delhi_train_data.get("hourly"):
            for key in raw_training_data["delhi"]["hourly"]: raw_training_data["delhi"]["hourly"][key].extend(delhi_train_data["hourly"].get(key, []))
        surrounding_locations = {name: dict(zip(["latitude", "longitude"], dp_fallback.calculate_destination_point(self.config.delhi_latitude, self.config.delhi_longitude, self.config.radius_km, bearing))) for name, bearing in zip(self.config.region_names, self.config.region_bearings)}
        for region_name, coords in surrounding_locations.items():
            region_train_data = fetcher_fallback.fetch_historical_weather_data(coords["latitude"], coords["longitude"], self.config.train_start_date, self.config.train_end_date, region_name=region_name)
            if region_train_data and region_train_data.get("hourly"):
                 for key in raw_training_data["surrounding_regions"][region_name]["hourly"]: raw_training_data["surrounding_regions"][region_name]["hourly"][key].extend(region_train_data["hourly"].get(key,[]))
        print("Fallback training data fetched.")

        df_full_processed, kmeans_model, kmeans_scaler = dp_fallback.process_historical_data(raw_training_data)
        
        if df_full_processed.empty: print("Not enough data for fallback training."); return
        
        trainer_fallback.kmeans_model = kmeans_model; trainer_fallback.kmeans_scaler = kmeans_scaler
        trainer_fallback.train_and_evaluate(df_full_processed, 0, len(df_full_processed), 0, 0, 1, 1) 
        trainer_fallback.save_models()
        print("Fallback models trained and saved.")

def _print_forecast_results(forecast_results):
    print("\n--- 7-Day Weather Forecast ---")
    header = f"{'Date':<12} {'Max Temp':>9} {'Min Temp':>9} {'Avg Temp':>9} {'Conditions':<25}" 
    separator = "-" * (len(header) + 5)
    print(header); print(separator)
    for result in forecast_results:
        print(f"{result['Date']:<12} {result['Max Temp']:>9} {result['Min Temp']:>9} {result['Avg Temp']:>9} {result['Conditions']:<25}")
    print(separator)

# --- Main Execution Function ---
def main():
    config = WeatherConfig()
    db_manager = DatabaseManager(config)
    forecaster = WeatherForecaster(config)
    forecaster.run_forecast_pipeline(db_manager)
    db_manager.close_connection()

if __name__ == "__main__":
    main()
