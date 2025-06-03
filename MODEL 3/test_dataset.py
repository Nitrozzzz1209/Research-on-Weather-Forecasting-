# test_dataset.py - Evaluate trained models on historical 2025 data

import math
import pandas as pd
from datetime import date, timedelta, datetime
import numpy as np
import psycopg2
from psycopg2 import sql

# from psycopg2.extras import execute_values # Not needed for select-only
import pmdarima as pm
import warnings
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    silhouette_score,  # ADDED import
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)  # For pmdarima warnings


# --- Configuration Class (Should be identical to training_dataset.py's WeatherConfig) ---
class WeatherConfig:
    """Stores configuration settings for the weather forecasting model."""

    def __init__(self):
        self.delhi_latitude = 28.6448
        self.delhi_longitude = 77.2167
        self.radius_km = 500
        self.num_regions = 8
        self.earth_radius_km = 6371

        # These dates define the period the models were trained on.
        # For test_dataset.py, we'll define a separate evaluation period.
        self.train_start_date = date(2020, 1, 1)
        self.train_end_date = date(2024, 12, 31)  # Models were trained up to this date

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
        self.advanced_weather_variables = [
            "dew_point_2m",
            "apparent_temperature",
            "pressure_msl",
            "surface_pressure",
            "rain",
            "snowfall",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "shortwave_radiation",
            "direct_normal_irradiance",
            "diffuse_radiation",
            "sunshine_duration",
            "wind_speed_100m",
            "wind_direction_100m",
            "wind_gusts_10m",
            "et0_fao_evapotranspiration",
            "snow_depth",
            "vapour_pressure_deficit",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm",
            "soil_temperature_100_to_255cm",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm",
        ]
        self.advanced_weather_variables = sorted(
            list(set(self.advanced_weather_variables))
        )
        self.all_hourly_variables_db = sorted(
            list(set(self.weather_variables + self.advanced_weather_variables))
        )

        self.db_name = "Project_Weather_Forecasting"
        self.db_user = "postgres"
        self.db_password = "1234"
        self.db_host = "localhost"
        self.db_port = "5432"

        self.model_save_dir = "MODEL 3/trained_models"
        self.results_save_dir = (
            "MODEL 3/evaluation_results"  # Directory to save evaluation CSV
        )
        self.max_lag_days = (
            30  # Max lag used in feature engineering (adjust if different in training)
        )
        self.kmeans_k_range = range(3, 11)  # Added for DataProcessor consistency


# --- DatabaseManager (Copied from training_dataset.py) ---
class DatabaseManager:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.conn = None
        self.connect()

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
            print("Initializing: Database connected (Test).")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def _load_data_from_table(
        self,
        table_name: str,
        columns: list,
        start_date: date,
        end_date: date,
        region_name: str = None,
    ):
        if not self.conn or self.conn.closed:
            print(
                f"Error: Database connection is closed. Cannot load data from {table_name}."
            )
            return None

        cur = self.conn.cursor()
        column_identifiers = [sql.Identifier("timestamp")] + [
            sql.Identifier(col_name) for col_name in columns
        ]
        select_cols_composable = sql.SQL(", ").join(column_identifiers)
        query_template_str = "SELECT {cols} FROM {table} WHERE timestamp >= %s AND timestamp < %s {region_filter_placeholder} ORDER BY timestamp;"
        actual_region_filter_sql = sql.SQL("")
        params = [start_date, end_date + timedelta(days=1)]
        if region_name:
            actual_region_filter_sql = sql.SQL("AND region_name = %s")
            params.append(region_name)
        final_query = None
        try:
            final_query = sql.SQL(query_template_str).format(
                cols=select_cols_composable,
                table=sql.Identifier(table_name),
                region_filter_placeholder=actual_region_filter_sql,
            )
            cur.execute(final_query, tuple(params))
            rows = cur.fetchall()
            if not rows:
                return None
            colnames = [desc[0] for desc in cur.description]
            df = pd.DataFrame(rows, columns=colnames)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except psycopg2.InterfaceError as ie:
            print(
                f"InterfaceError loading data from {table_name} for {region_name or 'Delhi'}: {ie}"
            )
            print("This might indicate the connection was closed unexpectedly.")
            return None
        except psycopg2.Error as e:
            print(
                f"Error loading data from {table_name} for {region_name or 'Delhi'}: {e}"
            )
            if final_query:
                try:
                    print(
                        f"Failed SQL query (approximate): {final_query.as_string(cur)}"
                    )
                except Exception as e_as_string:
                    print(f"Failed to get query as string: {e_as_string}")
            return None
        finally:
            if cur:
                cur.close()

    def load_delhi_basic_data(self, start_date, end_date):
        return self._load_data_from_table(
            "delhi_hourly_weather", self.config.weather_variables, start_date, end_date
        )

    def load_delhi_advanced_data(self, start_date, end_date):
        return self._load_data_from_table(
            "delhi_advanced_hourly_weather",
            self.config.advanced_weather_variables,
            start_date,
            end_date,
        )

    def load_surrounding_basic_data(self, region_name, start_date, end_date):
        return self._load_data_from_table(
            "surrounding_hourly_weather",
            self.config.weather_variables,
            start_date,
            end_date,
            region_name=region_name,
        )

    def load_surrounding_advanced_data(self, region_name, start_date, end_date):
        return self._load_data_from_table(
            "surrounding_advanced_hourly_weather",
            self.config.advanced_weather_variables,
            start_date,
            end_date,
            region_name=region_name,
        )

    def close_connection(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("Database connection closed (Test).")


# --- WeatherDataFetcher (Copied from training_dataset.py) ---
class WeatherDataFetcher:
    def __init__(self, config: WeatherConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager

    def fetch_historical_weather_data(
        self, latitude, longitude, start_date, end_date, region_name=None
    ):
        is_delhi = region_name == "delhi" or region_name is None
        if is_delhi:
            basic_df = self.db_manager.load_delhi_basic_data(start_date, end_date)
            advanced_df = self.db_manager.load_delhi_advanced_data(start_date, end_date)
        else:
            basic_df = self.db_manager.load_surrounding_basic_data(
                region_name, start_date, end_date
            )
            advanced_df = self.db_manager.load_surrounding_advanced_data(
                region_name, start_date, end_date
            )

        if basic_df is None and advanced_df is None:
            return None
        if basic_df is None:
            return advanced_df
        if advanced_df is None:
            return basic_df

        merged_df = pd.merge(basic_df, advanced_df, on="timestamp", how="inner")
        return merged_df if not merged_df.empty else None


# --- DataProcessor (Copied from training_dataset_v3 and adapted for test_dataset.py) ---
class DataProcessor:  # COPIED FULL CLASS FROM training_dataset_v3
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.kmeans_model = (
            None  # For training, this is fitted. For testing, it's loaded.
        )
        self.kmeans_scaler = (
            None  # For training, this is fitted. For testing, it's loaded.
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
        return (b * alpha) / (a - alpha)

    def process_historical_data(
        self,
        raw_data_all_locations: dict,
        loaded_kmeans_model=None,
        loaded_kmeans_scaler=None,
        is_evaluation_mode=False,
    ):
        delhi_hourly_df = raw_data_all_locations.get("delhi")
        if delhi_hourly_df is None or delhi_hourly_df.empty:
            print("Error: Missing Delhi data for processing.")
            return pd.DataFrame(), None, None

        delhi_hourly_df["timestamp"] = pd.to_datetime(delhi_hourly_df["timestamp"])
        delhi_hourly_df = delhi_hourly_df.set_index("timestamp").sort_index()

        agg_funcs = {}
        mean_min_max_vars = (
            self.config.weather_variables + self.config.advanced_weather_variables
        )
        sum_vars_list = [
            "precipitation",
            "rain",
            "snowfall",
            "sunshine_duration",
            "et0_fao_evapotranspiration",
        ]
        first_last_vars_list = [
            "wind_direction_10m",
            "wind_direction_100m",
            "weather_code",
            "snow_depth",
        ]

        for var in mean_min_max_vars:
            if (
                var not in sum_vars_list
                and var not in first_last_vars_list
                and var in delhi_hourly_df.columns
            ):
                agg_funcs[var] = ["mean", "min", "max"]
        for var in sum_vars_list:
            if var in delhi_hourly_df.columns:
                agg_funcs[var] = "sum"
        for var in first_last_vars_list:
            if var in delhi_hourly_df.columns:
                agg_funcs[var] = "first" if "wind_direction" in var else "last"

        if not agg_funcs:
            return pd.DataFrame(), None, None
        try:
            delhi_daily_df = delhi_hourly_df.resample("D").agg(agg_funcs)
        except Exception as e:
            print(f"Error resampling Delhi data: {e}")
            return pd.DataFrame(), None, None
        delhi_daily_df.columns = [
            "delhi_" + "_".join(col).strip("_") for col in delhi_daily_df.columns.values
        ]
        delhi_daily_df.reset_index(inplace=True)

        surrounding_daily_dfs = {}
        for region_name in self.config.region_names:
            region_hourly_df = raw_data_all_locations.get(region_name)
            if region_hourly_df is not None and not region_hourly_df.empty:
                region_hourly_df["timestamp"] = pd.to_datetime(
                    region_hourly_df["timestamp"]
                )
                region_hourly_df = region_hourly_df.set_index("timestamp").sort_index()
                region_agg_funcs = {
                    k: v for k, v in agg_funcs.items() if k in region_hourly_df.columns
                }
                if region_agg_funcs:
                    try:
                        daily_df = region_hourly_df.resample("D").agg(region_agg_funcs)
                        daily_df.columns = [
                            region_name + "_" + "_".join(col).strip("_")
                            for col in daily_df.columns.values
                        ]
                        daily_df.reset_index(inplace=True)
                        surrounding_daily_dfs[region_name] = daily_df
                    except Exception as e:
                        print(f"Error resampling {region_name} data: {e}")

        processed_df = delhi_daily_df.copy()
        for region_name, region_df in surrounding_daily_dfs.items():
            if region_df is not None and not region_df.empty:
                processed_df = pd.merge(
                    processed_df,
                    region_df,
                    on="timestamp",
                    how="left",
                    suffixes=("", f"_{region_name}_dup"),
                )

        processed_df["day_of_week"] = processed_df["timestamp"].dt.weekday
        processed_df["month"] = processed_df["timestamp"].dt.month
        processed_df["day_of_year"] = processed_df["timestamp"].dt.dayofyear
        processed_df["week_of_year"] = (
            processed_df["timestamp"].dt.isocalendar().week.astype(int)
        )
        processed_df["season"] = processed_df["month"].apply(self.get_season)

        target_temp_col_name = "delhi_temperature_2m_mean"
        if target_temp_col_name in processed_df.columns:
            if not is_evaluation_mode:
                processed_df["target_temperature"] = processed_df[
                    target_temp_col_name
                ].shift(-1)
            processed_df["actual_temperature"] = processed_df[target_temp_col_name]
        else:
            if not is_evaluation_mode:
                processed_df["target_temperature"] = np.nan
            processed_df["actual_temperature"] = np.nan

        if "month" in processed_df.columns:
            processed_df["month_sin"] = np.sin(2 * np.pi * processed_df["month"] / 12)
            processed_df["month_cos"] = np.cos(2 * np.pi * processed_df["month"] / 12)
        if "day_of_week" in processed_df.columns:
            processed_df["day_of_week_sin"] = np.sin(
                2 * np.pi * processed_df["day_of_week"] / 7
            )
            processed_df["day_of_week_cos"] = np.cos(
                2 * np.pi * processed_df["day_of_week"] / 7
            )
        processed_df = pd.get_dummies(
            processed_df, columns=["season"], drop_first=True, prefix="season"
        )

        delhi_avg_temp_col = "delhi_temperature_2m_mean"
        delhi_avg_rh_col = "delhi_relative_humidity_2m_mean"
        if (
            delhi_avg_temp_col in processed_df.columns
            and delhi_avg_rh_col in processed_df.columns
        ):
            processed_df["delhi_dew_point_calculated_mean"] = self._calculate_dew_point(
                processed_df[delhi_avg_temp_col], processed_df[delhi_avg_rh_col]
            )

        if not is_evaluation_mode:
            processed_df.dropna(subset=["target_temperature"], inplace=True)

        features_to_lag_and_roll = []
        for col_prefix in ["delhi_"]:
            for var_base in self.config.all_hourly_variables_db:
                for agg_suffix in ["_mean", "_sum", "_first", "_last", "_min", "_max"]:
                    col_name = f"{col_prefix}{var_base}{agg_suffix}"
                    if col_name in processed_df.columns:
                        features_to_lag_and_roll.append(col_name)
        features_to_lag_and_roll = sorted(list(set(features_to_lag_and_roll)))

        lags = [1, 2, 3, 7, 14, self.config.max_lag_days]
        rolling_windows = [3, 7]
        new_features_df_list = []
        for feature_base_name in features_to_lag_and_roll:
            if feature_base_name in processed_df.columns:
                temp_lag_roll_df = pd.DataFrame(index=processed_df.index)
                for lag in lags:
                    temp_lag_roll_df[f"{feature_base_name}_lag_{lag}"] = processed_df[
                        feature_base_name
                    ].shift(lag)
                for window in rolling_windows:
                    temp_lag_roll_df[f"{feature_base_name}_rolling_mean_{window}d"] = (
                        processed_df[feature_base_name]
                        .rolling(window=window, min_periods=1)
                        .mean()
                    )
                    temp_lag_roll_df[f"{feature_base_name}_rolling_std_{window}d"] = (
                        processed_df[feature_base_name]
                        .rolling(window=window, min_periods=1)
                        .std()
                    )
                temp_lag_roll_df[f"{feature_base_name}_diff_1d"] = processed_df[
                    feature_base_name
                ].diff(1)
                new_features_df_list.append(temp_lag_roll_df)

        if new_features_df_list:
            new_features_concat_df = pd.concat(new_features_df_list, axis=1)
            processed_df = pd.concat([processed_df, new_features_concat_df], axis=1)

        temp_interaction_df = pd.DataFrame(index=processed_df.index)
        if (
            delhi_avg_temp_col in processed_df.columns
            and delhi_avg_rh_col in processed_df.columns
        ):
            temp_interaction_df["delhi_temp_x_humidity"] = (
                processed_df[delhi_avg_temp_col] * processed_df[delhi_avg_rh_col]
            )
        delhi_avg_wind_col = "delhi_wind_speed_10m_mean"
        delhi_avg_cloud_col = "delhi_cloud_cover_mean"
        if (
            delhi_avg_wind_col in processed_df.columns
            and delhi_avg_cloud_col in processed_df.columns
        ):
            temp_interaction_df["delhi_wind_x_cloud"] = (
                processed_df[delhi_avg_wind_col] * processed_df[delhi_avg_cloud_col]
            )

        poly_features_cols = [delhi_avg_temp_col, delhi_avg_wind_col]
        if all(col in processed_df.columns for col in poly_features_cols):
            poly = PolynomialFeatures(
                degree=2, include_bias=False, interaction_only=False
            )
            poly_input_df = processed_df[poly_features_cols].copy()
            for col in poly_features_cols:
                if poly_input_df[col].isnull().any():
                    poly_input_df[col] = poly_input_df[col].fillna(
                        poly_input_df[col].mean()
                    )
            if not poly_input_df.isnull().values.any():
                poly_transformed = poly.fit_transform(poly_input_df)
                poly_feature_names = poly.get_feature_names_out(poly_features_cols)
                poly_df = pd.DataFrame(
                    poly_transformed,
                    columns=poly_feature_names,
                    index=processed_df.index,
                )
                cols_to_add_poly = [
                    col for col in poly_df.columns if col not in processed_df.columns
                ]
                temp_interaction_df = pd.concat(
                    [temp_interaction_df, poly_df[cols_to_add_poly]], axis=1
                )

        processed_df = pd.concat([processed_df, temp_interaction_df], axis=1)
        processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]

        if not is_evaluation_mode:
            processed_df.dropna(inplace=True)

        # K-Means Clustering
        clustering_features_keys = [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean",
            "wind_direction_10m_first",
            "weather_code_last",
            "precipitation_sum",
            "cloud_cover_mean",
        ]
        clustering_features = [
            f"delhi_{key}"
            for key in clustering_features_keys
            if f"delhi_{key}" in processed_df.columns
        ]

        actual_clustering_features = [
            f for f in clustering_features if f in processed_df.columns
        ]

        kmeans_min_k = (
            min(self.config.kmeans_k_range)
            if hasattr(self.config, "kmeans_k_range")
            else 3
        )

        if (
            not actual_clustering_features
            or len(processed_df[actual_clustering_features].dropna()) < kmeans_min_k
        ):
            print(
                f"Warning: Not enough data or features for K-Means. Actual weather will be NaN."
            )
            processed_df["actual_weather_cluster"] = np.nan
            if not is_evaluation_mode:
                processed_df["target_weather"] = np.nan
            return (
                processed_df,
                (loaded_kmeans_model or self.kmeans_model),
                (loaded_kmeans_scaler or self.kmeans_scaler),
            )

        df_for_clustering = processed_df[actual_clustering_features].copy()
        df_for_clustering.dropna(inplace=True)

        if not df_for_clustering.empty and len(df_for_clustering) >= kmeans_min_k:
            current_kmeans_model = loaded_kmeans_model
            current_kmeans_scaler = loaded_kmeans_scaler

            if current_kmeans_scaler is None and not is_evaluation_mode:
                current_kmeans_scaler = StandardScaler()
                scaled_clustering_features = current_kmeans_scaler.fit_transform(
                    df_for_clustering
                )
                self.kmeans_scaler = current_kmeans_scaler
            elif current_kmeans_scaler is not None:
                if hasattr(current_kmeans_scaler, "feature_names_in_"):
                    df_for_clustering_reordered = df_for_clustering.reindex(
                        columns=current_kmeans_scaler.feature_names_in_, fill_value=0
                    )
                else:
                    df_for_clustering_reordered = df_for_clustering
                scaled_clustering_features = current_kmeans_scaler.transform(
                    df_for_clustering_reordered
                )
            else:
                print("Error: K-Means scaler is None in an unexpected situation.")
                scaled_clustering_features = df_for_clustering.values

            if current_kmeans_model is None and not is_evaluation_mode:
                best_k, best_silhouette_score = -1, -float("inf")
                k_range = (
                    self.config.kmeans_k_range
                    if hasattr(self.config, "kmeans_k_range")
                    else range(3, 8)
                )
                for k_test in k_range:
                    if k_test >= len(df_for_clustering):
                        continue
                    kmeans_test_model = KMeans(
                        n_clusters=k_test, random_state=42, n_init="auto"
                    )
                    try:
                        cluster_labels_test = kmeans_test_model.fit_predict(
                            scaled_clustering_features
                        )
                        if len(np.unique(cluster_labels_test)) > 1:
                            score = silhouette_score(
                                scaled_clustering_features, cluster_labels_test
                            )
                            if score > best_silhouette_score:
                                best_silhouette_score = score
                                best_k = k_test
                    except ValueError:
                        pass
                n_clusters = best_k if best_k != -1 else 3
                print(f"KMeans k selected (training): {n_clusters}")
                current_kmeans_model = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init="auto"
                )
                cluster_labels = current_kmeans_model.fit_predict(
                    scaled_clustering_features
                )
                self.kmeans_model = current_kmeans_model
            elif current_kmeans_model is not None:
                cluster_labels = current_kmeans_model.predict(
                    scaled_clustering_features
                )
            else:
                print("Error: K-Means model is None in an unexpected situation.")
                cluster_labels = np.full(len(df_for_clustering), np.nan)

            processed_df.loc[df_for_clustering.index, "actual_weather_cluster"] = (
                cluster_labels
            )
            if "actual_weather_cluster" in processed_df.columns:
                processed_df["actual_weather_cluster"] = processed_df[
                    "actual_weather_cluster"
                ].astype("Int64")

            if not is_evaluation_mode:
                processed_df["target_weather"] = processed_df[
                    "actual_weather_cluster"
                ].shift(-1)
                processed_df.dropna(subset=["target_weather"], inplace=True)
        else:
            processed_df["actual_weather_cluster"] = np.nan
            if not is_evaluation_mode:
                processed_df["target_weather"] = np.nan

        if not is_evaluation_mode:
            processed_df.reset_index(drop=True, inplace=True)

        final_kmeans_model = loaded_kmeans_model or self.kmeans_model
        final_kmeans_scaler = loaded_kmeans_scaler or self.kmeans_scaler

        if "timestamp" in processed_df.columns and not is_evaluation_mode:
            processed_df_final = processed_df.drop(columns=["timestamp"])
        else:
            processed_df_final = processed_df

        return processed_df_final, final_kmeans_model, final_kmeans_scaler

    def process_historical_data_for_arima(self, raw_data_all_locations: dict):
        delhi_hourly_df = raw_data_all_locations.get("delhi")
        if (
            delhi_hourly_df is None
            or delhi_hourly_df.empty
            or "temperature_2m" not in delhi_hourly_df.columns
        ):
            return pd.Series(dtype=float)
        if not isinstance(delhi_hourly_df.index, pd.DatetimeIndex):
            if "timestamp" in delhi_hourly_df.columns:
                delhi_hourly_df["timestamp"] = pd.to_datetime(
                    delhi_hourly_df["timestamp"]
                )
                delhi_hourly_df = delhi_hourly_df.set_index("timestamp")
            else:
                return pd.Series(dtype=float)
        delhi_hourly_df = delhi_hourly_df.sort_index()
        return delhi_hourly_df["temperature_2m"].resample("D").mean().dropna()


# --- Model Evaluator Class (New for test_dataset.py) ---
class ModelEvaluator:
    def __init__(self, config: WeatherConfig, data_processor: DataProcessor):
        self.config = config
        self.data_processor = data_processor
        self.rf_model = None
        self.arima_model = None
        self.kmeans_model = None
        self.scaler = None
        self.cluster_scaler = None
        self.rf_feature_columns = None
        self.poly_transformer = None

        os.makedirs(self.config.results_save_dir, exist_ok=True)

    def load_models(self):
        """Loads all trained models and scalers."""
        model_dir = self.config.model_save_dir
        print(f"Loading models from {model_dir}...")
        try:
            self.rf_model = joblib.load(
                os.path.join(model_dir, "random_forest_model.joblib")
            )
            self.arima_model = joblib.load(
                os.path.join(model_dir, "arima_model.joblib")
            )
            self.kmeans_model = joblib.load(
                os.path.join(model_dir, "kmeans_model.joblib")
            )
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
            self.cluster_scaler = joblib.load(
                os.path.join(model_dir, "cluster_scaler.joblib")
            )
            self.rf_feature_columns = joblib.load(
                os.path.join(model_dir, "rf_feature_columns.joblib")
            )
            print("All models and scalers loaded successfully.")

            poly_path = os.path.join(model_dir, "poly_transformer.joblib")
            if os.path.exists(poly_path):
                self.poly_transformer = joblib.load(poly_path)
                print("Polynomial transformer loaded.")
            else:
                print(
                    "Polynomial transformer not found, assuming not used or not saved."
                )

        except FileNotFoundError as e:
            print(
                f"Error: Model file not found: {e}. Ensure training_dataset.py has been run successfully."
            )
            return False
        except Exception as e:
            print(f"An error occurred loading models: {e}")
            return False
        return True

    def map_cluster_to_weather_description(self, cluster_label):
        if pd.isna(cluster_label):
            return "Unknown"
        if self.kmeans_model:
            num_clusters = self.kmeans_model.n_clusters
            # You might want to define more descriptive names if you analyze the clusters from training
            generic_mapping = {i: f"Weather Type {i}" for i in range(num_clusters)}
            return generic_mapping.get(
                int(cluster_label), f"Undefined Cluster {int(cluster_label)}"
            )
        return f"Cluster {int(cluster_label)}"

    def prepare_features_for_single_day_eval(
        self, historical_df_processed: pd.DataFrame, target_date: date
    ):
        """
        Prepares features for a single target_date using data from historical_df_processed.
        The historical_df_processed should contain data up to and including the target_date,
        as lagged features for target_date are derived from previous days.
        """
        if "timestamp" not in historical_df_processed.columns:
            print(
                "Error: 'timestamp' column missing in processed historical data for feature prep."
            )
            return None

        # Get the row for the specific target_date
        target_date_features_row = historical_df_processed[
            historical_df_processed["timestamp"].dt.date == target_date
        ].copy()  # Use .copy()

        if target_date_features_row.empty:
            print(
                f"Warning: No processed data found for target date {target_date} in feature preparation."
            )
            return None

        # Drop columns that are targets or would not be known at prediction time for this day
        cols_to_drop = [
            "timestamp",
            "target_temperature",
            "actual_temperature",
            "actual_weather_cluster",
        ]
        # Also drop any columns that might have been created by `shift(-1)` for next-day targets in DataProcessor
        # This depends on how DataProcessor structures its output for evaluation mode.
        # For now, assume the above list is sufficient.

        features_for_rf = target_date_features_row.drop(
            columns=cols_to_drop, errors="ignore"
        )

        # Reindex to match the columns the RF model was trained on
        # Fill any potentially missing columns with 0 (or a more sophisticated strategy if needed)
        # This handles cases where some features might not be calculable for the first few days (e.g. long lags)
        # if the buffer wasn't large enough or if they were all NaN.
        features_for_rf = features_for_rf.reindex(
            columns=self.rf_feature_columns, fill_value=0
        )

        # Scale features
        try:
            scaled_features = self.scaler.transform(features_for_rf)
        except Exception as e:
            print(f"Error scaling features for {target_date}: {e}")
            print(f"Features before scaling: \n{features_for_rf.head()}")
            print(
                f"Scaler expected features: {self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else 'N/A'}"
            )
            return None

        return pd.DataFrame(
            scaled_features,
            columns=self.rf_feature_columns,
            index=features_for_rf.index,
        )

    def run_historical_evaluation(
        self,
        evaluation_start_date: date,
        evaluation_end_date: date,
        full_processed_df: pd.DataFrame,
        arima_temp_series: pd.Series,
    ):
        if (
            not self.rf_model
            or not self.arima_model
            or not self.scaler
            or not self.kmeans_model
            or not self.cluster_scaler
            or not self.rf_feature_columns
        ):
            print("Models not loaded. Cannot run evaluation.")
            return None

        results = []

        eval_df_for_actuals = full_processed_df[
            (full_processed_df["timestamp"].dt.date >= evaluation_start_date)
            & (full_processed_df["timestamp"].dt.date <= evaluation_end_date)
        ].copy()

        if eval_df_for_actuals.empty:
            print(
                f"No data available for the evaluation period: {evaluation_start_date} to {evaluation_end_date}"
            )
            return None

        print(
            f"\n--- Evaluating on Data from {evaluation_start_date} to {evaluation_end_date} ---"
        )

        for current_eval_date_obj in pd.date_range(
            evaluation_start_date, evaluation_end_date
        ):
            current_eval_date = current_eval_date_obj.date()
            # print(f"Evaluating for: {current_eval_date}") # Can be verbose

            day_features_scaled_df = self.prepare_features_for_single_day_eval(
                full_processed_df, current_eval_date
            )

            if day_features_scaled_df is None or day_features_scaled_df.empty:
                print(f"  Skipping {current_eval_date} due to missing features for RF.")
                results.append(
                    {
                        "Date": current_eval_date,
                        "Actual_Avg_Temp": np.nan,
                        "Predicted_Avg_Temp": np.nan,
                        "Actual_Condition_Label": np.nan,
                        "Predicted_Condition_Label": np.nan,
                        "Actual_Condition_Desc": "N/A",
                        "Predicted_Condition_Desc": "N/A",
                    }
                )
                continue

            predicted_avg_temp = np.nan
            try:
                arima_history = arima_temp_series[
                    arima_temp_series.index < pd.Timestamp(current_eval_date)
                ]
                if not arima_history.empty:
                    # Create a new model instance based on the loaded one's parameters
                    # This simulates retraining/updating daily.
                    temp_arima_model = pm.ARIMA(
                        order=self.arima_model.order,
                        seasonal_order=self.arima_model.seasonal_order,
                        suppress_warnings=True,
                    )
                    temp_arima_model.fit(arima_history)
                    predicted_avg_temp = temp_arima_model.predict(n_periods=1)[0]
                else:
                    print(f"  ARIMA: Not enough history for {current_eval_date}")
            except Exception as e:
                print(f"  Error during ARIMA prediction for {current_eval_date}: {e}")

            predicted_condition_label = np.nan
            try:
                if not day_features_scaled_df.isnull().values.any():
                    predicted_condition_label = self.rf_model.predict(
                        day_features_scaled_df
                    )[0]
                else:
                    print(
                        f"  RF: Scaled features for {current_eval_date} contain NaNs."
                    )
            except Exception as e:
                print(f"  Error during RF prediction for {current_eval_date}: {e}")

            actual_data_row = eval_df_for_actuals[
                eval_df_for_actuals["timestamp"].dt.date == current_eval_date
            ]
            actual_avg_temp = (
                actual_data_row["actual_temperature"].iloc[0]
                if not actual_data_row.empty and "actual_temperature" in actual_data_row
                else np.nan
            )
            actual_condition_label = (
                actual_data_row["actual_weather_cluster"].iloc[0]
                if not actual_data_row.empty
                and "actual_weather_cluster" in actual_data_row
                else np.nan
            )

            results.append(
                {
                    "Date": current_eval_date,
                    "Actual_Avg_Temp": actual_avg_temp,
                    "Predicted_Avg_Temp": predicted_avg_temp,
                    "Actual_Condition_Label": actual_condition_label,
                    "Predicted_Condition_Label": predicted_condition_label,
                    "Actual_Condition_Desc": self.map_cluster_to_weather_description(
                        actual_condition_label
                    ),
                    "Predicted_Condition_Desc": self.map_cluster_to_weather_description(
                        predicted_condition_label
                    ),
                }
            )

        results_df = pd.DataFrame(results)

        results_filename = os.path.join(
            self.config.results_save_dir,
            f"historical_{evaluation_start_date.year}_evaluation_results.csv",
        )
        results_df.to_csv(results_filename, index=False)
        print(f"\nEvaluation results saved to {results_filename}")

        results_df.dropna(
            subset=[
                "Actual_Avg_Temp",
                "Predicted_Avg_Temp",
                "Actual_Condition_Label",
                "Predicted_Condition_Label",
            ],
            inplace=True,
        )

        if not results_df.empty:
            temp_mae = mean_absolute_error(
                results_df["Actual_Avg_Temp"], results_df["Predicted_Avg_Temp"]
            )
            # Ensure labels are integers for classification metrics
            actual_labels_int = results_df["Actual_Condition_Label"].astype(int)
            predicted_labels_int = results_df["Predicted_Condition_Label"].astype(int)
            cond_accuracy = accuracy_score(actual_labels_int, predicted_labels_int)

            print(f"\n--- Evaluation Metrics ({evaluation_start_date.year} Data) ---")
            print(f"Temperature MAE: {temp_mae:.2f} C")
            print(f"Condition Accuracy: {cond_accuracy:.3f}")

            print("\nClassification Report for Conditions:")
            unique_labels_int = sorted(
                list(
                    set(actual_labels_int.unique()) | set(predicted_labels_int.unique())
                )
            )
            print(
                classification_report(
                    actual_labels_int,
                    predicted_labels_int,
                    labels=unique_labels_int,
                    zero_division=0,
                )
            )
            print("\nConfusion Matrix for Conditions:")
            print(
                confusion_matrix(
                    actual_labels_int, predicted_labels_int, labels=unique_labels_int
                )
            )
        else:
            print(
                "Not enough valid data points to calculate metrics after dropping NaNs."
            )

        return results_df


# --- Main Execution Function ---
def main():
    config = WeatherConfig()
    db_manager = DatabaseManager(config)  # db_manager for main's scope
    data_processor = DataProcessor(config)
    evaluator = ModelEvaluator(config, data_processor)
    # data_fetcher needs a db_manager whose connection is open when fetch_historical_weather_data is called
    data_fetcher = WeatherDataFetcher(config, db_manager)

    if not evaluator.load_models():
        if db_manager.conn and not db_manager.conn.closed:
            db_manager.close_connection()
        return

    evaluation_start_date = date(2025, 1, 1)
    evaluation_end_date = date.today() - timedelta(days=1)

    if evaluation_start_date > evaluation_end_date:
        print(
            f"Evaluation period is invalid or in the future. Start: {evaluation_start_date}, End: {evaluation_end_date}"
        )
        if db_manager.conn and not db_manager.conn.closed:
            db_manager.close_connection()
        return

    fetch_start_date = evaluation_start_date - timedelta(
        days=config.max_lag_days + 35
    )  # Increased buffer for safety with rolling features

    print(
        f"Fetching data for evaluation from {fetch_start_date} to {evaluation_end_date}..."
    )
    raw_data_for_eval = {}
    delhi_eval_data = data_fetcher.fetch_historical_weather_data(
        config.delhi_latitude,
        config.delhi_longitude,
        fetch_start_date,
        evaluation_end_date,
        region_name="delhi",
    )
    if delhi_eval_data is None or delhi_eval_data.empty:
        print("Critical: No Delhi data fetched for the evaluation period. Exiting.")
        if db_manager.conn and not db_manager.conn.closed:
            db_manager.close_connection()
        return
    raw_data_for_eval["delhi"] = delhi_eval_data

    surrounding_locations_coords = {
        name: dict(
            zip(
                ["latitude", "longitude"],
                data_processor.calculate_destination_point(
                    config.delhi_latitude,
                    config.delhi_longitude,
                    config.radius_km,
                    bearing,
                ),
            )
        )
        for name, bearing in zip(config.region_names, config.region_bearings)
    }
    for region_name, coords in surrounding_locations_coords.items():
        region_eval_data = data_fetcher.fetch_historical_weather_data(
            coords["latitude"],
            coords["longitude"],
            fetch_start_date,
            evaluation_end_date,
            region_name=region_name,
        )
        if region_eval_data is not None and not region_eval_data.empty:
            raw_data_for_eval[region_name] = region_eval_data

    print("Processing evaluation data...")
    processed_eval_data_full, _, _ = data_processor.process_historical_data(
        raw_data_for_eval,
        loaded_kmeans_model=evaluator.kmeans_model,
        loaded_kmeans_scaler=evaluator.cluster_scaler,
        is_evaluation_mode=True,
    )

    if (
        "timestamp" not in processed_eval_data_full.columns
        or "actual_temperature" not in processed_eval_data_full.columns
    ):
        print(
            "Error: 'timestamp' or 'actual_temperature' missing from processed_eval_data_full."
        )
        if db_manager.conn and not db_manager.conn.closed:
            db_manager.close_connection()
        return

    # Ensure timestamp is datetime for set_index
    processed_eval_data_full["timestamp"] = pd.to_datetime(
        processed_eval_data_full["timestamp"]
    )
    arima_eval_series = (
        processed_eval_data_full.set_index("timestamp")["actual_temperature"]
        .resample("D")
        .mean()
        .dropna()
    )

    if processed_eval_data_full.empty:
        print("Failed to process data for evaluation.")
        if db_manager.conn and not db_manager.conn.closed:
            db_manager.close_connection()
        return

    evaluator.run_historical_evaluation(
        evaluation_start_date,
        evaluation_end_date,
        processed_eval_data_full,
        arima_eval_series,
    )

    if db_manager.conn and not db_manager.conn.closed:
        db_manager.close_connection()
    print("\nHistorical evaluation script finished.")


if __name__ == "__main__":
    main()
