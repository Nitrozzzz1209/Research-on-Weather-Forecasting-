import math
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    silhouette_score,
)  # Added silhouette_score
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

        self.train_start_date = date(2020, 1, 1)  # Used for fallback training
        self.train_end_date = date(2024, 12, 31)  # Used for fallback training

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
        self.db_password = "1234"  # Please use a secure password management system
        self.db_host = "localhost"
        self.db_port = "5432"

        self.model_save_dir = "MODEL 1/trained_models"
        self.kmeans_k_range = range(3, 11)  # For Silhouette score in fallback training


# --- Database Manager Class (Identical to training_dataset.py) ---
class DatabaseManager:
    """Manages PostgreSQL database connection and operations."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.conn = None
        self.connect()
        # self.create_tables() # Tables should already exist

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
            print("Initializing: Database connected (Forecaster).")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    # create_tables can be omitted if we assume tables are created by the training script
    # or made very lightweight (IF NOT EXISTS)

    def load_delhi_data(self, start_date, end_date):
        if not self.conn:
            return None
        cur = self.conn.cursor()
        try:
            query = sql.SQL(
                """
                SELECT timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m,
                       wind_direction_10m, weather_code, precipitation, cloud_cover
                FROM delhi_hourly_weather WHERE timestamp >= %s AND timestamp <= %s ORDER BY timestamp;"""
            )
            cur.execute(
                query, (start_date, end_date + timedelta(days=1))
            )  # Ensure end_date is inclusive
            rows = cur.fetchall()
            if not rows:
                return None
            data = {
                "hourly": {col: [] for col in self.config.weather_variables + ["time"]},
                "hourly_units": {},
            }
            data["hourly_units"] = {
                "time": "iso8601",
                "temperature_2m": "celsius",
                "relative_humidity_2m": "%",
                "wind_speed_10m": "km/h",
                "wind_direction_10m": "degrees",
                "weather_code": "wmo code",
                "precipitation": "mm",
                "cloud_cover": "%",
            }
            # Correctly map columns based on self.config.weather_variables
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
        if not self.conn:
            return None
        cur = self.conn.cursor()
        try:
            query = sql.SQL(
                """
                SELECT timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m,
                       wind_direction_10m, weather_code, precipitation, cloud_cover
                FROM surrounding_hourly_weather WHERE region_name = %s AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp;"""
            )
            cur.execute(query, (region_name, start_date, end_date + timedelta(days=1)))
            rows = cur.fetchall()
            if not rows:
                return None
            data = {
                "hourly": {col: [] for col in self.config.weather_variables + ["time"]},
                "hourly_units": {},
            }
            data["hourly_units"] = {
                "time": "iso8601",
                "temperature_2m": "celsius",
                "relative_humidity_2m": "%",
                "wind_speed_10m": "km/h",
                "wind_direction_10m": "degrees",
                "weather_code": "wmo code",
                "precipitation": "mm",
                "cloud_cover": "%",
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
        if self.conn:
            self.conn.close()


# --- Data Fetcher Class (Identical to training_dataset.py) ---
class WeatherDataFetcher:
    def __init__(self, config: WeatherConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager

    def fetch_historical_weather_data(
        self, latitude, longitude, start_date, end_date, region_name=None
    ):
        db_data = None
        if self.db_manager.conn:
            if region_name == "delhi":
                db_data = self.db_manager.load_delhi_data(start_date, end_date)
            elif region_name:
                db_data = self.db_manager.load_surrounding_data(
                    region_name, start_date, end_date
                )

            if (
                db_data
                and db_data.get("hourly")
                and db_data["hourly"].get("time")
                and len(db_data["hourly"]["time"]) > 0
            ):
                try:
                    db_start_dt = datetime.fromisoformat(
                        db_data["hourly"]["time"][0]
                    ).date()
                    db_end_dt = datetime.fromisoformat(
                        db_data["hourly"]["time"][-1]
                    ).date()
                    if (
                        db_start_dt <= start_date and db_end_dt >= end_date
                    ):  # Check if data covers range
                        return db_data
                    else:
                        # print(f"Warning: Data for {region_name or 'Delhi'} from DB ({db_start_dt}-{db_end_dt}) does not fully cover requested range ({start_date}-{end_date}).") # Debug
                        pass

                except IndexError:
                    # print(f"Warning: Empty time array for {region_name or 'Delhi'} from DB for range {start_date}-{end_date}.") # Debug
                    pass
                except ValueError as e:  # Handle cases where timestamp might be invalid
                    # print(f"Warning: Invalid timestamp format for {region_name or 'Delhi'} from DB: {e}") # Debug
                    pass

        # print(f"Data for {region_name or 'Delhi'} not found or incomplete in DB for {start_date} to {end_date}.") # Debug
        return None


# --- Data Processor Class (Aligned with training_dataset.py) ---
class DataProcessor:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.kmeans_model = (
            None
        )
        self.kmeans_scaler = (
            None
        )

    def calculate_destination_point(
        self, start_lat, start_lon, distance_km, bearing_deg
    ):
        start_lat_rad, start_lon_rad, bearing_rad = map(
            math.radians, [start_lat, start_lon, bearing_deg]
        )
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
        return map(math.degrees, [dest_lat_rad, dest_lon_rad])

    def get_season(self, month):
        if month in [3, 4]:
            return "Spring"
        elif month in [5, 6]:
            return "Summer"
        elif month in [7, 8, 9]:
            return "Monsoon"
        else:
            return "Winter"

    def _calculate_dew_point(self, T, RH):
        RH = np.clip(RH, 1, 100)
        a = 17.27
        b = 237.7
        alpha = ((a * T) / (b + T)) + np.log(RH / 100.0)
        T_dp = (b * alpha) / (a - alpha)
        return T_dp

    def process_historical_data(
        self,
        raw_data,
        kmeans_model_for_prediction=None,
        kmeans_scaler_for_prediction=None,
    ):
        dataset = []
        delhi_data = raw_data.get("delhi")
        surrounding_data = raw_data.get("surrounding_regions", {})

        if (
            not delhi_data
            or not delhi_data.get("hourly")
            or not delhi_data["hourly"].get("time")
        ):
            print(
                "Error: Missing or incomplete historical data for Delhi (DataProcessor)."
            )
            return pd.DataFrame(), None, None

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
            current_day_data["day_of_year"] = current_day_time.timetuple().tm_yday
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
                if region_data_for_current_day and isinstance(
                    region_data_for_current_day, dict
                ):
                    region_hourly = region_data_for_current_day.get("hourly", {})
                    if region_hourly and region_hourly.get("time"):
                        region_time_stamps = region_hourly["time"]
                        region_start_index, region_end_index = None, None
                        for j, ts_str in enumerate(region_time_stamps):
                            ts_date = datetime.fromisoformat(ts_str).date()
                            if ts_date == current_day_time:
                                if region_start_index is None:
                                    region_start_index = j
                                region_end_index = j + 1

                        if (
                            region_start_index is not None
                            and region_end_index is not None
                        ):
                            for var in self.config.weather_variables:
                                hourly_values = region_hourly.get(var, [])[
                                    region_start_index:region_end_index
                                ]
                                valid_hourly_values = [
                                    val for val in hourly_values if val is not None
                                ]
                                if valid_hourly_values:
                                    current_day_data[f"{region_name}_avg_{var}"] = (
                                        np.mean(valid_hourly_values)
                                    )
                                    current_day_data[f"{region_name}_max_{var}"] = (
                                        np.max(valid_hourly_values)
                                    )
                                    current_day_data[f"{region_name}_min_{var}"] = (
                                        np.min(valid_hourly_values)
                                    )
                                else:
                                    for agg in ["avg", "max", "min"]:
                                        current_day_data[
                                            f"{region_name}_{agg}_{var}"
                                        ] = np.nan
                        else: 
                            _for_var_in_loop_helper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"])
                    else: 
                        _for_var_in_loop_helper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"])
                else: 
                    _for_var_in_loop_helper(self.config.weather_variables, current_day_data, region_name, ["avg", "max", "min"])


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
                    current_day_data["target_temperature"] = np.mean(
                        valid_next_day_temperatures
                    )
                    dataset.append(current_day_data)

        df = pd.DataFrame(dataset)
        if df.empty:
            print("DataFrame is empty after initial processing. Cannot proceed.")
            return pd.DataFrame(), None, None

        if "month" in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        if "day_of_week" in df.columns:
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df = pd.get_dummies(df, columns=["season"], drop_first=True)

        if (
            "delhi_avg_temperature_2m" in df.columns
            and "delhi_avg_relative_humidity_2m" in df.columns
        ):
            df["delhi_dew_point"] = self._calculate_dew_point(
                df["delhi_avg_temperature_2m"], df["delhi_avg_relative_humidity_2m"]
            )

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

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
            if feature in df.columns:
                for lag in lags:
                    new_features_df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
                for window in rolling_windows:
                    new_features_df[f"{feature}_rolling_mean_{window}d"] = (
                        df[feature].rolling(window=window, min_periods=1).mean()
                    )
                    new_features_df[f"{feature}_rolling_min_{window}d"] = (
                        df[feature].rolling(window=window, min_periods=1).min()
                    )
                    new_features_df[f"{feature}_rolling_max_{window}d"] = (
                        df[feature].rolling(window=window, min_periods=1).max()
                    )
                    new_features_df[f"{feature}_rolling_std_{window}d"] = (
                        df[feature].rolling(window=window, min_periods=1).std()
                    )
                new_features_df[f"{feature}_diff_1d"] = df[feature].diff(1)

        if (
            "delhi_avg_temperature_2m" in df.columns
            and "delhi_avg_relative_humidity_2m" in df.columns
        ):
            new_features_df["delhi_temp_x_humidity"] = (
                df["delhi_avg_temperature_2m"] * df["delhi_avg_relative_humidity_2m"]
            )
        if (
            "delhi_avg_wind_speed_10m" in df.columns
            and "delhi_avg_cloud_cover" in df.columns
        ):
            new_features_df["delhi_wind_x_cloud"] = (
                df["delhi_avg_wind_speed_10m"] * df["delhi_avg_cloud_cover"]
            )

        poly_features_cols = ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        if all(col in df.columns for col in poly_features_cols):
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_input_df = df[poly_features_cols].copy()
            poly_input_df.fillna(poly_input_df.mean(), inplace=True)
            poly_transformed = poly.fit_transform(poly_input_df)
            poly_feature_names = poly.get_feature_names_out(poly_features_cols)
            full_poly_df = pd.DataFrame(
                poly_transformed, columns=poly_feature_names, index=df.index
            )
            new_features_df = pd.concat([new_features_df, full_poly_df], axis=1)

        df = pd.concat([df, new_features_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        clustering_features = [
            "delhi_avg_temperature_2m",
            "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m",
            "delhi_avg_wind_direction_10m",
            "delhi_avg_weather_code",
            "delhi_avg_precipitation",
            "delhi_avg_cloud_cover",
        ]

        missing_clustering_features = [
            f for f in clustering_features if f not in df.columns
        ]
        if missing_clustering_features:
            print(
                f"Warning: Missing features required for clustering: {missing_clustering_features}. Target weather will be NaN."
            )
            df["target_weather"] = np.nan
            return df, self.kmeans_model, self.kmeans_scaler

        df_for_clustering = df[clustering_features].copy()
        df_for_clustering.dropna(inplace=True)

        final_kmeans_model, final_kmeans_scaler = (
            kmeans_model_for_prediction,
            kmeans_scaler_for_prediction,
        )
        if kmeans_model_for_prediction and kmeans_scaler_for_prediction:
            if not df_for_clustering.empty:
                df_for_clustering_reindexed = df_for_clustering.reindex(
                    columns=kmeans_scaler_for_prediction.feature_names_in_, fill_value=0
                )
                scaled_clustering_features = kmeans_scaler_for_prediction.transform(
                    df_for_clustering_reindexed
                )
                cluster_labels = kmeans_model_for_prediction.predict(
                    scaled_clustering_features
                )
                df.loc[df_for_clustering.index, "target_weather"] = cluster_labels
                df["target_weather"] = df["target_weather"].astype(int)
            else:
                df["target_weather"] = np.nan
        else:
            if not df_for_clustering.empty:
                self.kmeans_scaler = StandardScaler()
                scaled_clustering_features = self.kmeans_scaler.fit_transform(
                    df_for_clustering
                )
                best_k, best_silhouette_score = -1, -1
                print("\nFinding optimal k for KMeans (DataProcessor)...")
                for k_test in self.config.kmeans_k_range:
                    if k_test >= len(df_for_clustering): 
                        print(f"Skipping k={k_test}, not enough samples for clustering.")
                        continue
                    kmeans_test = KMeans(
                        n_clusters=k_test, random_state=42, n_init="auto"
                    )
                    cluster_labels_test = kmeans_test.fit_predict(
                        scaled_clustering_features
                    )
                    if len(np.unique(cluster_labels_test)) > 1 and len(
                        np.unique(cluster_labels_test)
                    ) < len(df_for_clustering):
                        try:
                            score = silhouette_score(
                                scaled_clustering_features, cluster_labels_test
                            )
                            print(f"  k={k_test}, Silhouette: {score:.4f}")
                            # CORRECTED SyntaxError:
                            if score > best_silhouette_score:
                                best_silhouette_score = score 
                                best_k = k_test             
                        except ValueError:
                            pass 
                n_clusters = best_k if best_k != -1 else 3
                print(f"KMeans k selected: {n_clusters}")
                self.kmeans_model = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init="auto"
                )
                cluster_labels = self.kmeans_model.fit_predict(
                    scaled_clustering_features
                )
                df.loc[df_for_clustering.index, "target_weather"] = cluster_labels
                df["target_weather"] = df["target_weather"].astype(int)
                final_kmeans_model, final_kmeans_scaler = (
                    self.kmeans_model,
                    self.kmeans_scaler,
                )
            else:
                df["target_weather"] = np.nan
        df.dropna(subset=["target_weather"], inplace=True)
        return df, final_kmeans_model, final_kmeans_scaler

    def process_historical_data_for_arima(self, raw_data):
        delhi_data = raw_data.get("delhi")
        if (
            not delhi_data
            or not delhi_data.get("hourly")
            or not delhi_data["hourly"].get("time")
        ):
            print(
                "Error: Missing or incomplete 'delhi' or 'hourly' data or 'time' in raw_data for ARIMA."
            )
            return pd.DataFrame()
        delhi_hourly = delhi_data["hourly"]
        if not delhi_hourly.get("temperature_2m"):
            print("Error: 'temperature_2m' key missing in delhi_hourly for ARIMA.")
            return pd.DataFrame()
        if not delhi_hourly["time"] or not delhi_hourly["temperature_2m"]:
            print(
                "Error: 'time' or 'temperature_2m' list is empty in delhi_hourly for ARIMA."
            )
            return pd.DataFrame()
        try:
            hourly_df = pd.DataFrame(
                {
                    "time": pd.to_datetime(delhi_hourly["time"]),
                    "temperature_2m": delhi_hourly["temperature_2m"],
                }
            )
            hourly_df.set_index("time", inplace=True)
            hourly_df.dropna(subset=["temperature_2m"], inplace=True)
            daily_avg_temp = hourly_df["temperature_2m"].resample("D").mean().dropna()
            return daily_avg_temp
        except Exception as e:
            print(f"Error processing data for ARIMA: {e}")
            return pd.DataFrame()

def _for_var_in_loop_helper(weather_variables, current_day_data, region_name, aggs):
    for var_ in weather_variables:
        for agg_ in aggs: current_day_data[f"{region_name}_{agg_}_{var_}"] = np.nan


# --- Hybrid Model Trainer Class (For fallback training) ---
class HybridModelTrainer:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.rf_model = None
        self.arima_model_fit = None
        self.scaler = None
        self.kmeans_model = None
        self.kmeans_scaler = None
        self.rf_feature_columns_ = None

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
        rf_train_df = dataframe_full
        if not rf_train_df.empty:
            X_rf_train = rf_train_df.drop(
                ["target_weather", "target_temperature"], axis=1, errors="ignore"
            )
            y_rf_train = rf_train_df["target_weather"]
            self.scaler = StandardScaler()
            X_rf_train_scaled = self.scaler.fit_transform(X_rf_train)
            self.rf_feature_columns_ = list(
                getattr(self.scaler, "feature_names_in_", X_rf_train.columns)
            )
            self.rf_model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            )
            self.rf_model.fit(X_rf_train_scaled, y_rf_train)
            print(f"Fallback RF model trained on {len(X_rf_train)} samples.")
        arima_train_series = temp_series_full
        if not arima_train_series.empty:
            try:
                model_auto = pm.auto_arima(
                    arima_train_series,
                    start_p=1,
                    start_q=1,
                    test="adf",
                    max_p=3,
                    max_q=3,
                    m=7,
                    d=None,
                    seasonal=True,
                    start_P=0,
                    start_Q=0,
                    max_P=1,
                    max_Q=1,
                    D=None,
                    trace=False,
                    error_action="ignore",
                    suppress_warnings=True,
                    stepwise=True,
                )
                self.arima_model_fit = model_auto
                print(
                    f"Fallback ARIMA: order {model_auto.order}, seasonal {model_auto.seasonal_order}."
                )
            except Exception as e:
                print(f"Error training fallback ARIMA: {e}")
        return np.nan, np.nan

    def save_models(self):
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        print(f"\nSaving fallback models to {self.config.model_save_dir}/...")
        if self.scaler:
            joblib.dump(
                self.scaler, os.path.join(self.config.model_save_dir, "scaler.joblib")
            )
        if self.kmeans_model:
            joblib.dump(
                self.kmeans_model,
                os.path.join(self.config.model_save_dir, "kmeans_model.joblib"),
            )
        if self.kmeans_scaler:
            joblib.dump(
                self.kmeans_scaler,
                os.path.join(self.config.model_save_dir, "cluster_scaler.joblib"),
            )
        if self.rf_model:
            joblib.dump(
                self.rf_model,
                os.path.join(self.config.model_save_dir, "random_forest_model.joblib"),
            )
        if self.arima_model_fit:
            joblib.dump(
                self.arima_model_fit,
                os.path.join(self.config.model_save_dir, "arima_model.joblib"),
            )
        if self.rf_feature_columns_:
            joblib.dump(
                self.rf_feature_columns_,
                os.path.join(self.config.model_save_dir, "rf_feature_columns.joblib"),
            )
        print("Fallback models saved.")


# --- Weather Forecaster Class ---
class WeatherForecaster:
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
        self.poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
        dummy_poly_data = pd.DataFrame(
            np.zeros((1, len(self.poly_features_for_rf))),
            columns=self.poly_features_for_rf,
        )
        self.poly_transformer.fit(dummy_poly_data)
        self.load_models()

    def load_models(self):
        model_dir = self.config.model_save_dir
        required_files = [
            "scaler.joblib",
            "kmeans_model.joblib",
            "cluster_scaler.joblib",
            "random_forest_model.joblib",
            "arima_model.joblib",
            "rf_feature_columns.joblib",
        ]
        all_models_exist = all(
            os.path.exists(os.path.join(model_dir, f)) for f in required_files
        )

        if all_models_exist:
            try:
                self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
                self.kmeans_model = joblib.load(
                    os.path.join(model_dir, "kmeans_model.joblib")
                )
                self.kmeans_scaler = joblib.load(
                    os.path.join(model_dir, "cluster_scaler.joblib")
                )
                self.rf_model = joblib.load(
                    os.path.join(model_dir, "random_forest_model.joblib")
                )
                self.arima_model = joblib.load(
                    os.path.join(model_dir, "arima_model.joblib")
                )
                self.rf_feature_columns = joblib.load(
                    os.path.join(model_dir, "rf_feature_columns.joblib")
                )
                print("All models loaded successfully.")
            except Exception as e:
                print(
                    f"Error loading models: {e}. Fallback training may be required."
                )
                self.scaler = self.kmeans_model = self.kmeans_scaler = self.rf_model = (
                    self.arima_model
                ) = self.rf_feature_columns = None
        else:
            print("One or more model files not found. Fallback training may be required.")
            self.scaler = self.kmeans_model = self.kmeans_scaler = self.rf_model = (
                self.arima_model
            ) = self.rf_feature_columns = None

    def map_cluster_to_weather_description(self, cluster_label):
        mapping = {0: "Pleasant/Clear", 1: "Warm/Humid", 2: "Cool/Cloudy"}
        if pd.isna(cluster_label):
            return "Unknown"
        return mapping.get(int(cluster_label), f"Weather Type {int(cluster_label)}")

    def get_season(self, month):
        return DataProcessor(self.config).get_season(month)

    def _calculate_dew_point(self, T, RH):
        return DataProcessor(self.config)._calculate_dew_point(T, RH)

    def prepare_features_for_prediction(
        self, current_date_obj, history_df_input, predicted_avg_temp=None
    ):
        if self.rf_feature_columns is None:
            print("ERROR: rf_feature_columns not loaded in prepare_features_for_prediction.")
            return pd.DataFrame(), pd.DataFrame()

        history_df = history_df_input.copy()
        current_day_data_dict = {}

        current_day_data_dict["month"] = current_date_obj.month
        current_day_data_dict["day_of_week"] = current_date_obj.weekday()
        current_day_data_dict["day_of_year"] = current_date_obj.timetuple().tm_yday
        current_day_data_dict["week_of_year"] = current_date_obj.isocalendar()[1]
        current_day_data_dict["month_sin"] = np.sin(
            2 * np.pi * current_day_data_dict["month"] / 12
        )
        current_day_data_dict["month_cos"] = np.cos(
            2 * np.pi * current_day_data_dict["month"] / 12
        )
        current_day_data_dict["day_of_week_sin"] = np.sin(
            2 * np.pi * current_day_data_dict["day_of_week"] / 7
        )
        current_day_data_dict["day_of_week_cos"] = np.cos(
            2 * np.pi * current_day_data_dict["day_of_week"] / 7
        )

        current_season_name = self.get_season(current_day_data_dict["month"])
        # Corrected: Align with rf_feature_columns.joblib which has Spring, Summer, Winter
        # This implies Monsoon was the dropped category.
        current_day_data_dict['season_Spring'] = 1 if current_season_name == "Spring" else 0
        current_day_data_dict['season_Summer'] = 1 if current_season_name == "Summer" else 0
        current_day_data_dict['season_Winter'] = 1 if current_season_name == "Winter" else 0
        # If current_season_name is "Monsoon", all the above will be 0.

        current_day_data_dict["delhi_avg_temperature_2m"] = (
            predicted_avg_temp
            if predicted_avg_temp is not None
            else (
                history_df["delhi_avg_temperature_2m"].iloc[-1]
                if not history_df.empty and "delhi_avg_temperature_2m" in history_df
                else np.nan
            )
        )

        other_delhi_vars = [
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "precipitation",
            "cloud_cover",
        ]
        for var_suffix in other_delhi_vars:
            col = f"delhi_avg_{var_suffix}"
            current_day_data_dict[col] = (
                history_df[col].iloc[-1]
                if not history_df.empty and col in history_df.columns
                else np.nan
            )

        temp_val = current_day_data_dict.get("delhi_avg_temperature_2m", np.nan)
        rh_val = current_day_data_dict.get("delhi_avg_relative_humidity_2m", np.nan)
        current_day_data_dict["delhi_dew_point"] = (
            self._calculate_dew_point(temp_val, rh_val)
            if not (np.isnan(temp_val) or np.isnan(rh_val))
            else np.nan
        )

        for r_name in self.config.region_names:
            for var_suffix in self.config.weather_variables:
                col = f"{r_name}_avg_{var_suffix}"
                current_day_data_dict[col] = (
                    history_df[col].iloc[-1]
                    if not history_df.empty and col in history_df.columns
                    else np.nan
                )

        current_day_unscaled_df = pd.DataFrame([current_day_data_dict])

        source_cols_for_lags_rolls = [ # Renamed for clarity
            "delhi_avg_temperature_2m",
            "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m",
            "delhi_avg_precipitation",
            "delhi_avg_cloud_cover",
            "delhi_dew_point",
        ]
        
        # Corrected: Dynamically determine which season columns to use based on rf_feature_columns
        expected_season_columns_in_rf = [
            col for col in self.rf_feature_columns if col.startswith('season_')
        ] # Should be ['season_Spring', 'season_Summer', 'season_Winter']
        
        # Columns to select from history_df for creating combined_df (for lags/rolls)
        columns_for_history_slice = (
            source_cols_for_lags_rolls # Base weather variables
            + ["month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos"] # Cyclical time
            + expected_season_columns_in_rf # Actual season dummies from training
        )
        
        # Ensure all columns in columns_for_history_slice actually exist in history_df before slicing
        final_columns_for_history_slice = [col for col in columns_for_history_slice if col in history_df.columns]
        
        temp_history_for_lags = history_df[final_columns_for_history_slice].copy()

        # Ensure current_day_unscaled_df has the same columns as temp_history_for_lags for concatenation
        current_day_unscaled_df_subset = current_day_unscaled_df.reindex(columns=temp_history_for_lags.columns, fill_value=0).copy()
        
        combined_df_for_derived = pd.concat(
            [temp_history_for_lags, current_day_unscaled_df_subset], ignore_index=True
        )


        derived_features_df = pd.DataFrame(index=combined_df_for_derived.index)
        features_to_lag_roll = [
            col
            for col in [
                "delhi_avg_temperature_2m",
                "delhi_avg_relative_humidity_2m",
                "delhi_avg_wind_speed_10m",
                "delhi_avg_precipitation",
                "delhi_avg_cloud_cover",
            ]
            if col in combined_df_for_derived.columns
        ]

        lags = [1, 2, 3, 7, 14, 30]
        rolling_windows = [3, 7]
        for feature in features_to_lag_roll:
            for lag in lags:
                derived_features_df[f"{feature}_lag_{lag}"] = combined_df_for_derived[
                    feature
                ].shift(lag)
            for window in rolling_windows:
                derived_features_df[f"{feature}_rolling_mean_{window}d"] = (
                    combined_df_for_derived[feature]
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                derived_features_df[f"{feature}_rolling_min_{window}d"] = (
                    combined_df_for_derived[feature]
                    .rolling(window=window, min_periods=1)
                    .min()
                )
                derived_features_df[f"{feature}_rolling_max_{window}d"] = (
                    combined_df_for_derived[feature]
                    .rolling(window=window, min_periods=1)
                    .max()
                )
                derived_features_df[f"{feature}_rolling_std_{window}d"] = (
                    combined_df_for_derived[feature]
                    .rolling(window=window, min_periods=1)
                    .std()
                )
            derived_features_df[f"{feature}_diff_1d"] = combined_df_for_derived[
                feature
            ].diff(1)

        if (
            "delhi_avg_temperature_2m" in combined_df_for_derived.columns
            and "delhi_avg_relative_humidity_2m" in combined_df_for_derived.columns
        ):
            derived_features_df["delhi_temp_x_humidity"] = (
                combined_df_for_derived["delhi_avg_temperature_2m"]
                * combined_df_for_derived["delhi_avg_relative_humidity_2m"]
            )
        if (
            "delhi_avg_wind_speed_10m" in combined_df_for_derived.columns
            and "delhi_avg_cloud_cover" in combined_df_for_derived.columns
        ):
            derived_features_df["delhi_wind_x_cloud"] = (
                combined_df_for_derived["delhi_avg_wind_speed_10m"]
                * combined_df_for_derived["delhi_avg_cloud_cover"]
            )

        if all(
            col in combined_df_for_derived.columns for col in self.poly_features_for_rf
        ):
            poly_input = combined_df_for_derived[self.poly_features_for_rf].copy()
            poly_input.fillna(poly_input.mean(), inplace=True)
            poly_transformed = self.poly_transformer.transform(poly_input)
            poly_names = self.poly_transformer.get_feature_names_out(
                self.poly_features_for_rf
            )
            poly_df = pd.DataFrame(
                poly_transformed,
                columns=poly_names,
                index=combined_df_for_derived.index,
            )
            derived_features_df = pd.concat([derived_features_df, poly_df], axis=1)

        final_derived_features_for_current_day = derived_features_df.iloc[
            [-1]
        ].reset_index(drop=True)

        all_features_for_current_day_unscaled = pd.concat(
            [
                current_day_unscaled_df.reset_index(drop=True), # This has the base features including correct season dummies
                final_derived_features_for_current_day,
            ],
            axis=1,
        )
        all_features_for_current_day_unscaled = (
            all_features_for_current_day_unscaled.loc[
                :, ~all_features_for_current_day_unscaled.columns.duplicated()
            ]
        )

        features_for_scaling = all_features_for_current_day_unscaled.reindex(
            columns=self.rf_feature_columns, fill_value=0
        )
        features_for_scaling.fillna(0, inplace=True)

        if self.scaler:
            scaled_features_for_rf = self.scaler.transform(features_for_scaling)
            # Return unscaled features aligned with rf_feature_columns for history update
            unscaled_for_history = all_features_for_current_day_unscaled.reindex(columns=self.rf_feature_columns, fill_value=0)
            unscaled_for_history.fillna(0, inplace=True) # Ensure no NaNs in history row
            return pd.DataFrame(
                scaled_features_for_rf, columns=self.rf_feature_columns
            ), unscaled_for_history
        else:
            print("ERROR: Scaler not loaded in prepare_features_for_prediction.")
            return pd.DataFrame(), pd.DataFrame()

    def run_forecast_pipeline(self, db_manager: DatabaseManager):
        if not (
            self.scaler
            and self.kmeans_model
            and self.rf_model
            and self.arima_model
            and self.rf_feature_columns
            and self.kmeans_scaler
        ):
            print(
                "One or more models missing/inconsistent. Attempting fallback training."
            )
            self._train_models_if_needed(db_manager)
            self.load_models()
            if not (
                self.scaler
                and self.kmeans_model
                and self.rf_model
                and self.arima_model
                and self.rf_feature_columns
                and self.kmeans_scaler
            ):
                print("Models could not be loaded or trained. Cannot proceed.")
                return pd.DataFrame()

        print("\n--- Starting 7-Day Weather Prediction ---")
        data_fetcher = WeatherDataFetcher(self.config, db_manager)
        dp_recent_data = DataProcessor(self.config)

        today = date.today()
        history_fetch_end_date = today - timedelta(days=1)
        history_fetch_start_date = history_fetch_end_date - timedelta(days=89)
        prediction_start_date = today

        print(
            f"Fetching recent data from {history_fetch_start_date} to {history_fetch_end_date}."
        )
        raw_recent_data = {
            "delhi": {
                "hourly": {var: [] for var in self.config.weather_variables + ["time"]}
            },
            "surrounding_regions": {
                name: {
                    "hourly": {
                        var: [] for var in self.config.weather_variables + ["time"]
                    }
                }
                for name in self.config.region_names
            },
        }
        delhi_hist_data = data_fetcher.fetch_historical_weather_data(
            self.config.delhi_latitude,
            self.config.delhi_longitude,
            history_fetch_start_date,
            history_fetch_end_date,
            region_name="delhi",
        )
        if delhi_hist_data and delhi_hist_data.get("hourly"):
            for key in raw_recent_data["delhi"]["hourly"]:
                raw_recent_data["delhi"]["hourly"][key].extend(
                    delhi_hist_data["hourly"].get(key, [])
                )

        surrounding_locations = {
            name: dict(
                zip(
                    ["latitude", "longitude"],
                    dp_recent_data.calculate_destination_point(
                        self.config.delhi_latitude,
                        self.config.delhi_longitude,
                        self.config.radius_km,
                        bearing,
                    ),
                )
            )
            for name, bearing in zip(
                self.config.region_names, self.config.region_bearings
            )
        }
        for region_name, coords in surrounding_locations.items():
            region_hist_data = data_fetcher.fetch_historical_weather_data(
                coords["latitude"],
                coords["longitude"],
                history_fetch_start_date,
                history_fetch_end_date,
                region_name=region_name,
            )
            if region_hist_data and region_hist_data.get("hourly"):
                for key in raw_recent_data["surrounding_regions"][region_name][
                    "hourly"
                ]:
                    raw_recent_data["surrounding_regions"][region_name]["hourly"][
                        key
                    ].extend(region_hist_data["hourly"].get(key, []))
        print("Recent data fetched.")

        recent_weather_df_processed, _, _ = dp_recent_data.process_historical_data(
            raw_recent_data,
            kmeans_model_for_prediction=self.kmeans_model,
            kmeans_scaler_for_prediction=self.kmeans_scaler,
        )
        recent_temp_series = dp_recent_data.process_historical_data_for_arima(
            raw_recent_data
        )

        if recent_weather_df_processed.empty or recent_temp_series.empty:
            print("Not enough recent data for predictions after processing.")
            return pd.DataFrame()

        if self.rf_feature_columns is None:
            print(
                "ERROR: rf_feature_columns not available after loading/fallback. Cannot proceed."
            )
            return pd.DataFrame()

        # iterative_history_df should contain all columns listed in self.rf_feature_columns
        # plus any original columns needed for context if different (e.g. unscaled versions for lags)
        # For simplicity, we start with rf_feature_columns and add target_temperature for context.
        # The prepare_features_for_prediction will handle generating derived features from this history.
        
        # Ensure recent_weather_df_processed has all rf_feature_columns before taking tail for history
        # This is critical. The df from process_historical_data should naturally have these.
        history_columns = self.rf_feature_columns[:] # Start with RF features
        if 'target_temperature' in recent_weather_df_processed.columns and 'target_temperature' not in history_columns:
            history_columns.append('target_temperature')
        # Add other base columns if they are used by prepare_features_for_prediction but not in rf_feature_columns
        # For example, original 'month', 'day_of_week' if they were dropped from rf_feature_columns after creating sin/cos
        
        iterative_history_df = recent_weather_df_processed.reindex(columns=history_columns, fill_value=0)

        max_lag_days = 30 
        iterative_history_df = iterative_history_df.tail(
            max_lag_days + 7 # Keep enough history for lags for the 7 forecast days
        ).copy() 


        forecast_results = []
        try:
            self.arima_model.update(recent_temp_series.tail(30))
            print(
                f"ARIMA model updated with recent data up to {recent_temp_series.index.max()}."
            )
        except Exception as e:
            print(f"Could not update ARIMA model, using as is: {e}")
        arima_future_temps = self.arima_model.predict(n_periods=7)

        for i in range(7):
            forecast_date_obj = prediction_start_date + timedelta(days=i)
            predicted_temperature = arima_future_temps[i]
            max_temp = predicted_temperature + 3
            min_temp = predicted_temperature - 3

            scaled_features_for_rf, unscaled_features_for_history = (
                self.prepare_features_for_prediction(
                    forecast_date_obj, iterative_history_df.copy(), predicted_temperature # Pass a copy of history
                )
            )

            predicted_weather_label = np.nan
            if not scaled_features_for_rf.empty and self.rf_model:
                try:
                    rf_prediction_val = self.rf_model.predict(scaled_features_for_rf)[
                        0
                    ]
                    predicted_weather_label = int(rf_prediction_val)
                except Exception as e:
                    print(f"Error during RF prediction for {forecast_date_obj}: {e}")

            weather_condition_desc = self.map_cluster_to_weather_description(
                predicted_weather_label
            )
            forecast_results.append(
                {
                    "Date": forecast_date_obj.strftime("%Y-%m-%d"),
                    "Max Temp": f"{max_temp:.1f}",
                    "Min Temp": f"{min_temp:.1f}",
                    "Avg Temp": f"{predicted_temperature:.1f}",
                    "Conditions": weather_condition_desc,
                }
            )

            if not unscaled_features_for_history.empty:
                # unscaled_features_for_history should be a single row DataFrame
                # already aligned with rf_feature_columns by prepare_features_for_prediction
                # Add target_temperature to it for consistent history structure if iterative_history_df has it
                if 'target_temperature' in iterative_history_df.columns and 'target_temperature' not in unscaled_features_for_history.columns:
                    unscaled_features_for_history['target_temperature'] = predicted_temperature # Or actual if known for history

                row_to_append = unscaled_features_for_history.reindex(columns=iterative_history_df.columns, fill_value=0)
                iterative_history_df = pd.concat(
                    [iterative_history_df, row_to_append], ignore_index=True
                )
                # Trim history
                if len(iterative_history_df) > (max_lag_days + 14) : 
                     iterative_history_df = iterative_history_df.iloc[1:]
            else:
                print(
                    f"Warning: Could not obtain unscaled features for {forecast_date_obj} to update history."
                )

        _print_forecast_results(forecast_results)
        return pd.DataFrame(forecast_results)

    def _train_models_if_needed(self, db_manager: DatabaseManager):
        print("\n--- Initiating Fallback Model Training ---")
        dp_fallback = DataProcessor(self.config)
        trainer_fallback = HybridModelTrainer(self.config)
        fetcher_fallback = WeatherDataFetcher(self.config, db_manager)
        raw_training_data = {
            "delhi": {
                "hourly": {var: [] for var in self.config.weather_variables + ["time"]}
            },
            "surrounding_regions": {
                name: {
                    "hourly": {
                        var: [] for var in self.config.weather_variables + ["time"]
                    }
                }
                for name in self.config.region_names
            },
        }
        delhi_train_data = fetcher_fallback.fetch_historical_weather_data(
            self.config.delhi_latitude,
            self.config.delhi_longitude,
            self.config.train_start_date,
            self.config.train_end_date,
            region_name="delhi",
        )
        if delhi_train_data and delhi_train_data.get("hourly"):
            for key in raw_training_data["delhi"]["hourly"]:
                raw_training_data["delhi"]["hourly"][key].extend(
                    delhi_train_data["hourly"].get(key, [])
                )

        surrounding_locations = {
            name: dict(
                zip(
                    ["latitude", "longitude"],
                    dp_fallback.calculate_destination_point(
                        self.config.delhi_latitude,
                        self.config.delhi_longitude,
                        self.config.radius_km,
                        bearing,
                    ),
                )
            )
            for name, bearing in zip(
                self.config.region_names, self.config.region_bearings
            )
        }
        for region_name, coords in surrounding_locations.items():
            region_train_data = fetcher_fallback.fetch_historical_weather_data(
                coords["latitude"],
                coords["longitude"],
                self.config.train_start_date,
                self.config.train_end_date,
                region_name=region_name,
            )
            if region_train_data and region_train_data.get("hourly"):
                for key in raw_training_data["surrounding_regions"][region_name][
                    "hourly"
                ]:
                    raw_training_data["surrounding_regions"][region_name]["hourly"][
                        key
                    ].extend(region_train_data["hourly"].get(key, []))
        print("Fallback training data fetched.")

        df_full_processed, kmeans_model, kmeans_scaler = (
            dp_fallback.process_historical_data(raw_training_data)
        )
        temp_series_full_processed = dp_fallback.process_historical_data_for_arima(
            raw_training_data
        )

        if df_full_processed.empty or temp_series_full_processed.empty:
            print("Not enough data for fallback training.")
            return
        trainer_fallback.kmeans_model = kmeans_model
        trainer_fallback.kmeans_scaler = kmeans_scaler
        trainer_fallback.train_and_evaluate(
            df_full_processed,
            temp_series_full_processed,
            0,
            len(df_full_processed),
            0,
            0,
            1,
        )
        trainer_fallback.save_models()
        print("Fallback models trained and saved.")


def _print_forecast_results(forecast_results):
    print("\n--- 7-Day Weather Forecast ---")
    header = f"{'Date':<12} {'Max Temp':>9} {'Min Temp':>9} {'Avg Temp':>9} {'Conditions':<20}"
    separator = "-" * (len(header) + 5)
    print(header)
    print(separator)
    for result in forecast_results:
        print(
            f"{result['Date']:<12} {result['Max Temp']:>9} {result['Min Temp']:>9} {result['Avg Temp']:>9} {result['Conditions']:<20}"
        )
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


# import joblib

# # Make sure you are in the correct directory where 'trained_models' folder is,
# # or provide the full path to the file.
# try:
#     rf_features = joblib.load("trained_models/rf_feature_columns.joblib")
#     print("Contents of rf_feature_columns.joblib:")
#     print(rf_features)
#     print(f"\nTotal number of features listed: {len(rf_features)}")
# except FileNotFoundError:
#     print("Error: 'trained_models/rf_feature_columns.joblib' not found.")
# except Exception as e:
#     print(f"An error occurred while loading the file: {e}")