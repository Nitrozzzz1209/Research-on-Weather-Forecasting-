# training_dataset.py - Integrating Advanced Weather Data (Phase 1)

import math
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, silhouette_score
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.cluster import KMeans
import numpy as np

# from collections import Counter # Not explicitly used, can be removed if still unused
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
        # REVERTED: Use a fixed end date for training data
        self.train_end_date = date(2024, 12, 31)

        self.region_bearings = [0, 45, 90, 135, 180, 225, 270, 315]
        self.region_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        # Basic weather variables
        self.weather_variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "precipitation",
            "cloud_cover",
        ]

        # Advanced weather variables (mirroring the list from your data_collection.py)
        # !! User should ensure this list is up-to-date with successfully collected variables !!
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
            # Ensure any other variables like 'cape', 'lifted_index', 'precipitation_probability'
            # are added here if they were successfully collected and are intended for use.
        ]
        self.advanced_weather_variables = sorted(
            list(set(self.advanced_weather_variables))
        )

        # All hourly variables to be fetched (basic + advanced)
        self.all_hourly_variables_db = (
            self.weather_variables + self.advanced_weather_variables
        )
        # Remove duplicates just in case, though there shouldn't be any if lists are distinct
        self.all_hourly_variables_db = sorted(list(set(self.all_hourly_variables_db)))

        self.db_name = "Project_Weather_Forecasting"
        self.db_user = "postgres"
        self.db_password = "1234"
        self.db_host = "localhost"
        self.db_port = "5432"

        self.model_save_dir = "MODEL 3/trained_models"  # UPDATED
        self.kmeans_k_range = range(3, 11)


# --- Database Manager Class ---
class DatabaseManager:
    """Manages PostgreSQL database connection and operations."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.conn = None
        self.connect()
        # create_tables should ideally only be run once or by the data_collection script.
        # For training, we assume tables exist.
        # self.create_tables() # Assuming tables are created by data_collection.py

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
            print("Initializing: Database connected (Training).")
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
        """Helper function to load data from a specified table."""
        if not self.conn:
            return None
        cur = self.conn.cursor()

        # Create a list of SQL Identifiers for all columns to select
        column_identifiers = [sql.Identifier("timestamp")] + [
            sql.Identifier(col_name) for col_name in columns
        ]
        # Join them into a single Composable object representing the comma-separated list
        select_cols_composable = sql.SQL(", ").join(column_identifiers)

        # Base query string template
        query_template_str = "SELECT {cols} FROM {table} WHERE timestamp >= %s AND timestamp < %s {region_filter_placeholder} ORDER BY timestamp;"

        # Prepare region filter SQL object and parameters
        actual_region_filter_sql = sql.SQL("")  # Empty SQL object if no filter
        params = [
            start_date,
            end_date + timedelta(days=1),
        ]  # end_date is exclusive for < comparison

        if region_name:
            actual_region_filter_sql = sql.SQL(
                "AND region_name = %s"
            )  # %s will be handled by cursor.execute
            params.append(region_name)

        final_query = None  # Initialize for potential use in error message
        try:
            # Compose the final query using sql.SQL().format()
            final_query = sql.SQL(query_template_str).format(
                cols=select_cols_composable,
                table=sql.Identifier(table_name),
                region_filter_placeholder=actual_region_filter_sql,
            )

            cur.execute(final_query, tuple(params))
            rows = cur.fetchall()
            if not rows:
                return None

            # Construct DataFrame
            colnames = [
                desc[0] for desc in cur.description
            ]  # Get all column names from cursor description
            df = pd.DataFrame(rows, columns=colnames)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
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
            else:
                print("Query was not composed successfully before error.")
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
        if self.conn:
            self.conn.close()


# --- Data Fetcher Class ---
class WeatherDataFetcher:
    def __init__(self, config: WeatherConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager

    def fetch_historical_weather_data(
        self, latitude, longitude, start_date, end_date, region_name=None
    ):
        """Fetches and merges basic and advanced historical data for a location."""
        is_delhi = (
            region_name == "delhi" or region_name is None
        )  # Assuming None means Delhi for primary location

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
            print(
                f"No data (basic or advanced) found for {region_name or 'Delhi'} between {start_date} and {end_date}"
            )
            return None
        if basic_df is None:
            # If only advanced data is present, ensure it has a 'timestamp' column
            if advanced_df is not None:
                if (
                    "timestamp" not in advanced_df.columns
                    and "time" in advanced_df.columns
                ):  # Should not happen with _load_data_from_table
                    advanced_df["timestamp"] = pd.to_datetime(advanced_df["time"])
                    advanced_df = advanced_df.drop(columns=["time"], errors="ignore")
                print(
                    f"Warning: Missing basic data for {region_name or 'Delhi'}. Using only advanced."
                )
                return advanced_df
            return None  # Should not happen if advanced_df is also None (covered above)
        if advanced_df is None:
            # If only basic data is present
            if basic_df is not None:
                if (
                    "timestamp" not in basic_df.columns and "time" in basic_df.columns
                ):  # Should not happen
                    basic_df["timestamp"] = pd.to_datetime(basic_df["time"])
                    basic_df = basic_df.drop(columns=["time"], errors="ignore")
                print(
                    f"Warning: Missing advanced data for {region_name or 'Delhi'}. Using only basic."
                )
            return basic_df

        # Both DataFrames should have 'timestamp' from _load_data_from_table
        merged_df = pd.merge(basic_df, advanced_df, on="timestamp", how="inner")

        if merged_df.empty:
            print(
                f"Warning: Merged data is empty for {region_name or 'Delhi'} (no common timestamps between basic and advanced)."
            )
            return None

        return merged_df


# --- Data Processor Class ---
class DataProcessor:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.kmeans_model = None
        self.kmeans_scaler = None

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

    def process_historical_data(self, raw_data_all_locations: dict):
        """
        Processes raw historical data (now a dictionary of DataFrames per location)
        to create a daily aggregated feature set.
        raw_data_all_locations: {'delhi': DataFrame, 'N': DataFrame, ...}
        """
        dataset = []

        delhi_hourly_df = raw_data_all_locations.get("delhi")
        if delhi_hourly_df is None or delhi_hourly_df.empty:
            print(
                "Error: Missing or incomplete historical data for Delhi. Cannot create dataset."
            )
            return pd.DataFrame(), None, None

        delhi_hourly_df["timestamp"] = pd.to_datetime(delhi_hourly_df["timestamp"])
        delhi_hourly_df = delhi_hourly_df.set_index("timestamp").sort_index()

        agg_funcs = {}
        mean_min_max_vars = (
            self.config.weather_variables + self.config.advanced_weather_variables
        )
        # Remove variables that should be summed or handled differently
        sum_vars_list = [
            "precipitation",
            "rain",
            "snowfall",
            "sunshine_duration",
            "et0_fao_evapotranspiration",
        ]
        # Variables for 'first' or 'last' aggregation
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
                if "wind_direction" in var:
                    agg_funcs[var] = "first"
                else:  # weather_code, snow_depth
                    agg_funcs[var] = "last"

        if not agg_funcs:
            print("Warning: No columns found in Delhi data for aggregation.")
            return pd.DataFrame(), None, None

        try:
            delhi_daily_df = delhi_hourly_df.resample("D").agg(agg_funcs)
        except Exception as e:
            print(f"Error during resampling Delhi data: {e}")
            return pd.DataFrame(), None, None

        delhi_daily_df.columns = [
            "delhi_" + "_".join(col).strip("_") for col in delhi_daily_df.columns.values
        ]  # Fix for single agg like sum
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
                        print(f"Error during resampling {region_name} data: {e}")

        processed_df = delhi_daily_df.copy()
        for region_name, region_df in surrounding_daily_dfs.items():
            if (
                region_df is not None and not region_df.empty
            ):  # Check if region_df is not None and not empty
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
            processed_df["target_temperature"] = processed_df[
                target_temp_col_name
            ].shift(-1)
        else:
            print(
                f"Warning: Column '{target_temp_col_name}' not found for creating target_temperature."
            )
            processed_df["target_temperature"] = np.nan

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

        processed_df.dropna(subset=["target_temperature"], inplace=True)

        # Updated list of features for lag/roll to include new aggregated advanced vars if they exist
        # This is a more dynamic way to select them
        features_to_lag_and_roll = []
        for col_prefix in ["delhi_"]:  # Could expand to other regions if needed
            for (
                var_base
            ) in self.config.all_hourly_variables_db:  # Use the combined list
                # Check for mean, sum, first, last aggregated versions
                mean_col = f"{col_prefix}{var_base}_mean"
                sum_col = f"{col_prefix}{var_base}_sum"
                first_col = f"{col_prefix}{var_base}_first"
                last_col = f"{col_prefix}{var_base}_last"
                if mean_col in processed_df.columns:
                    features_to_lag_and_roll.append(mean_col)
                if sum_col in processed_df.columns:
                    features_to_lag_and_roll.append(sum_col)
                if first_col in processed_df.columns:
                    features_to_lag_and_roll.append(first_col)
                if last_col in processed_df.columns:
                    features_to_lag_and_roll.append(last_col)

        features_to_lag_and_roll = sorted(
            list(set(features_to_lag_and_roll))
        )  # Unique sorted list

        lags = [1, 2, 3, 7, 14, 30]
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
                    temp_lag_roll_df[f"{feature_base_name}_rolling_min_{window}d"] = (
                        processed_df[feature_base_name]
                        .rolling(window=window, min_periods=1)
                        .min()
                    )
                    temp_lag_roll_df[f"{feature_base_name}_rolling_max_{window}d"] = (
                        processed_df[feature_base_name]
                        .rolling(window=window, min_periods=1)
                        .max()
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

        delhi_avg_wind_col = (
            "delhi_wind_speed_10m_mean"  # Assuming this is the aggregated name
        )
        delhi_avg_cloud_col = (
            "delhi_cloud_cover_mean"  # Assuming this is the aggregated name
        )
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

        processed_df.dropna(inplace=True)  # Drop NaNs from lags/rolls/interactions
        processed_df.reset_index(drop=True, inplace=True)

        # K-Means Clustering
        clustering_features = [
            "delhi_temperature_2m_mean",
            "delhi_relative_humidity_2m_mean",
            "delhi_wind_speed_10m_mean",
            (
                "delhi_wind_direction_10m_first"
                if "delhi_wind_direction_10m_first" in processed_df.columns
                else "delhi_wind_direction_10m_mean"
            ),  # Be flexible
            (
                "delhi_weather_code_last"
                if "delhi_weather_code_last" in processed_df.columns
                else "delhi_weather_code_mean"
            ),
            "delhi_precipitation_sum",
            "delhi_cloud_cover_mean",
        ]
        # Add new advanced features to clustering_features if desired, e.g.:
        # if 'delhi_surface_pressure_mean' in processed_df.columns: clustering_features.append('delhi_surface_pressure_mean')
        # if 'delhi_dew_point_2m_mean' in processed_df.columns: clustering_features.append('delhi_dew_point_2m_mean')

        actual_clustering_features = [
            f for f in clustering_features if f in processed_df.columns
        ]
        missing_for_clustering = [
            f for f in clustering_features if f not in processed_df.columns
        ]
        if missing_for_clustering:
            print(
                f"Warning: Missing features for K-Means clustering: {missing_for_clustering}."
            )

        if not actual_clustering_features or len(processed_df) < min(
            self.config.kmeans_k_range
        ):
            print(
                f"Warning: Not enough data or features for K-Means. Target weather will be NaN. Samples: {len(processed_df)}, Features: {len(actual_clustering_features)}"
            )
            processed_df["target_weather"] = np.nan
        else:
            df_for_clustering = processed_df[actual_clustering_features].copy()
            df_for_clustering.dropna(inplace=True)

            if not df_for_clustering.empty and len(df_for_clustering) >= min(
                self.config.kmeans_k_range
            ):
                self.kmeans_scaler = StandardScaler()
                scaled_clustering_features = self.kmeans_scaler.fit_transform(
                    df_for_clustering
                )

                best_k, best_silhouette_score = -1, -float("inf")
                print("\nFinding optimal k for KMeans using Silhouette Score...")
                for k_test in self.config.kmeans_k_range:
                    if k_test >= len(df_for_clustering):
                        continue
                    kmeans_test = KMeans(
                        n_clusters=k_test, random_state=42, n_init="auto"
                    )
                    try:
                        cluster_labels_test = kmeans_test.fit_predict(
                            scaled_clustering_features
                        )
                        if len(np.unique(cluster_labels_test)) > 1:
                            score = silhouette_score(
                                scaled_clustering_features, cluster_labels_test
                            )
                            # print(f"  k={k_test}, Silhouette Score: {score:.4f}") # Can be verbose
                            if score > best_silhouette_score:
                                best_silhouette_score = score
                                best_k = k_test
                        # else:
                        # print(f"  k={k_test}, only 1 cluster found.") # Can be verbose
                    except ValueError as e:
                        print(
                            f"  Could not calculate Silhouette Score for k={k_test}: {e}"
                        )

                n_clusters = best_k if best_k != -1 else 3
                print(
                    f"KMeans k selected: {n_clusters} (Silhouette: {best_silhouette_score if best_k != -1 else 'N/A'})"
                )

                self.kmeans_model = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init="auto"
                )
                cluster_labels = self.kmeans_model.fit_predict(
                    scaled_clustering_features
                )
                processed_df.loc[df_for_clustering.index, "target_weather"] = (
                    cluster_labels
                )
                if "target_weather" in processed_df.columns:
                    processed_df["target_weather"] = processed_df[
                        "target_weather"
                    ].astype("Int64")
            else:
                print(
                    "Warning: Not enough data for K-Means clustering after NaN drop. Target weather will be NaN."
                )
                processed_df["target_weather"] = np.nan

        processed_df.dropna(subset=["target_weather"], inplace=True)
        processed_df.reset_index(drop=True, inplace=True)

        if (
            "timestamp" in processed_df.columns
        ):  # Should have been dropped by set_index earlier if resample worked
            processed_df = processed_df.drop(columns=["timestamp"], errors="ignore")

        return processed_df, self.kmeans_model, self.kmeans_scaler

    def process_historical_data_for_arima(self, raw_data_all_locations: dict):
        delhi_hourly_df = raw_data_all_locations.get("delhi")
        if (
            delhi_hourly_df is None
            or delhi_hourly_df.empty
            or "temperature_2m" not in delhi_hourly_df.columns
        ):
            print(
                "Error: Missing or incomplete 'delhi' data or 'temperature_2m' for ARIMA."
            )
            return pd.Series(dtype=float)

        # Ensure timestamp is index if not already
        if not isinstance(delhi_hourly_df.index, pd.DatetimeIndex):
            if "timestamp" in delhi_hourly_df.columns:
                delhi_hourly_df["timestamp"] = pd.to_datetime(
                    delhi_hourly_df["timestamp"]
                )
                delhi_hourly_df = delhi_hourly_df.set_index("timestamp")
            else:  # Should not happen if data fetcher returns df with timestamp column
                print("Error: No 'timestamp' column in Delhi data for ARIMA index.")
                return pd.Series(dtype=float)

        delhi_hourly_df = delhi_hourly_df.sort_index()
        daily_avg_temp = delhi_hourly_df["temperature_2m"].resample("D").mean().dropna()
        return daily_avg_temp


# --- Hybrid Model Trainer Class ---
class HybridModelTrainer:
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.rf_model = None
        self.arima_model_fit = None
        self.weather_accuracies = []
        self.temperature_maes = []
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
        rf_accuracy, arima_mae = np.nan, np.nan

        train_end_idx = min(train_end_idx, len(dataframe_full))
        test_start_idx = min(test_start_idx, len(dataframe_full))
        test_end_idx = min(test_end_idx, len(dataframe_full))

        rf_train_df = dataframe_full.iloc[train_start_idx:train_end_idx]
        rf_test_df = dataframe_full.iloc[test_start_idx:test_end_idx]

        if (
            not rf_train_df.empty
            and "target_weather" in rf_train_df.columns
            and not rf_test_df.empty
            and "target_weather" in rf_test_df.columns
            and not rf_train_df["target_weather"].isnull().all()
            and not rf_test_df["target_weather"].isnull().all()
        ):  # Check for all NaNs in target

            X_rf_train = rf_train_df.drop(
                ["target_weather", "target_temperature"], axis=1, errors="ignore"
            )
            y_rf_train = rf_train_df["target_weather"].astype(
                int
            )  # Ensure target is int for classifier
            X_rf_test = rf_test_df.drop(
                ["target_weather", "target_temperature"], axis=1, errors="ignore"
            )
            y_rf_test = rf_test_df["target_weather"].astype(int)

            # Align columns: Get all possible columns from the full dataset (excluding targets)
            # This helps if some features are all NaN in a slice and get dropped.
            # However, DataProcessor should ideally handle NaNs such that columns are consistent.
            all_possible_features = [
                col
                for col in dataframe_full.columns
                if col not in ["target_weather", "target_temperature"]
            ]
            X_rf_train = X_rf_train.reindex(
                columns=all_possible_features, fill_value=0
            )  # Fill missing with 0, or use mean from training
            X_rf_test = X_rf_test.reindex(columns=all_possible_features, fill_value=0)

            self.rf_feature_columns_ = list(X_rf_train.columns)

            self.scaler = StandardScaler()
            X_rf_train_scaled = self.scaler.fit_transform(X_rf_train)
            X_rf_test_scaled = self.scaler.transform(X_rf_test)

            self.rf_model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            )
            self.rf_model.fit(X_rf_train_scaled, y_rf_train)

            if (
                not X_rf_test_scaled.shape[0] == 0
            ):  # Check if test set is not empty after processing
                y_rf_pred = self.rf_model.predict(X_rf_test_scaled)
                rf_accuracy = accuracy_score(y_rf_test, y_rf_pred)
            self.weather_accuracies.append(
                rf_accuracy
            )  # Append NaN if test set was empty
        else:
            self.weather_accuracies.append(np.nan)
            # print(f"Session {session_num}: RF Train or Test DF is empty or target_weather is all NaN.")

        train_end_idx_temp = min(train_end_idx, len(temp_series_full))
        test_start_idx_temp = min(test_start_idx, len(temp_series_full))
        test_end_idx_temp = min(test_end_idx, len(temp_series_full))

        arima_train_series = temp_series_full.iloc[train_start_idx:train_end_idx_temp]
        arima_test_series = temp_series_full.iloc[test_start_idx_temp:test_end_idx_temp]

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

                if not arima_test_series.empty:
                    arima_predictions = self.arima_model_fit.predict(
                        n_periods=len(arima_test_series)
                    )
                    arima_mae = mean_absolute_error(
                        arima_test_series, arima_predictions
                    )
                self.temperature_maes.append(
                    arima_mae
                )  # Append NaN if test series was empty
            except Exception as e:
                # print(f"Error training/predicting ARIMA for session {session_num}: {e}") # Can be verbose
                self.temperature_maes.append(np.nan)
        else:
            self.temperature_maes.append(np.nan)
            # print(f"Session {session_num}: ARIMA Train Series is empty.")

        return rf_accuracy, arima_mae

    def get_average_metrics(self):
        avg_accuracy = (
            np.nanmean(self.weather_accuracies) if self.weather_accuracies else 0
        )
        avg_mae = np.nanmean(self.temperature_maes) if self.temperature_maes else 0
        return avg_accuracy, avg_mae

    def save_models(self):
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        print(f"\nSaving models to {self.config.model_save_dir}/...")

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
            print(
                f"Random Forest feature columns saved ({len(self.rf_feature_columns_)} features)."
            )
        else:
            print(
                "Warning: RF feature columns not available to save (was likely empty or not set)."
            )
        print("Models saved.")


# --- Main Execution Function ---
def main():
    config = WeatherConfig()
    db_manager = DatabaseManager(config)
    data_fetcher = WeatherDataFetcher(config, db_manager)
    data_processor = DataProcessor(config)
    hybrid_trainer = HybridModelTrainer(config)

    print("Initializing: Starting weather model training process.")

    raw_data_all_locations = {}
    print(
        f"Fetching data for Delhi from {config.train_start_date} to {config.train_end_date}..."
    )
    delhi_merged_df = data_fetcher.fetch_historical_weather_data(
        config.delhi_latitude,
        config.delhi_longitude,
        config.train_start_date,
        config.train_end_date,
        region_name="delhi",
    )
    if delhi_merged_df is not None and not delhi_merged_df.empty:  # Check for non-empty
        raw_data_all_locations["delhi"] = delhi_merged_df
    else:
        print(f"Critical: No data fetched or merged data is empty for Delhi. Exiting.")
        if db_manager.conn:
            db_manager.close_connection()
        return

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
        print(
            f"Fetching data for {region_name} from {config.train_start_date} to {config.train_end_date}..."
        )
        region_merged_df = data_fetcher.fetch_historical_weather_data(
            coords["latitude"],
            coords["longitude"],
            config.train_start_date,
            config.train_end_date,
            region_name=region_name,
        )
        if (
            region_merged_df is not None and not region_merged_df.empty
        ):  # Check for non-empty
            raw_data_all_locations[region_name] = region_merged_df
        else:
            print(
                f"Warning: No data fetched or merged data is empty for region {region_name}."
            )

    print("Data fetching complete. Processing data...")
    weather_dataset_full, kmeans_model_fitted, kmeans_scaler_fitted = (
        data_processor.process_historical_data(raw_data_all_locations)
    )

    if kmeans_model_fitted is None or kmeans_scaler_fitted is None:
        print(
            "Warning: KMeans model or scaler not fitted properly during data processing."
        )
    hybrid_trainer.kmeans_model = kmeans_model_fitted
    hybrid_trainer.kmeans_scaler = kmeans_scaler_fitted

    daily_avg_temp_series_full = data_processor.process_historical_data_for_arima(
        raw_data_all_locations
    )

    if weather_dataset_full.empty or daily_avg_temp_series_full.empty:
        print("\nFailed to create datasets after processing. Cannot train.")
        if db_manager.conn:
            db_manager.close_connection()
        return

    print(
        f"Dataset created with {len(weather_dataset_full)} daily samples and {len(weather_dataset_full.columns)} features."
    )
    print(
        f"ARIMA temperature series has {len(daily_avg_temp_series_full)} daily samples."
    )

    num_training_sessions = 5
    session_test_days = 60
    total_days = len(weather_dataset_full)

    min_train_period = 365
    if total_days < min_train_period + session_test_days:
        print(
            f"Not enough data for training ({total_days} days). Need at least {min_train_period + session_test_days} days."
        )
        if db_manager.conn:
            db_manager.close_connection()
        return

    first_test_window_start_idx = min_train_period

    sessions = []
    current_train_end_idx = first_test_window_start_idx
    while current_train_end_idx + session_test_days <= total_days:
        train_start_idx = 0
        train_end_idx = current_train_end_idx
        test_start_idx = current_train_end_idx
        test_end_idx = current_train_end_idx + session_test_days

        sessions.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))
        current_train_end_idx += session_test_days

    if not sessions:
        print(
            "No valid training/test sessions could be created with the current data and parameters."
        )
        if db_manager.conn:
            db_manager.close_connection()
        return

    print(f"\n--- Starting {len(sessions)} Training Sessions (Walk-Forward) ---")
    print(f"{'Training Step':<15} {'RF_Accuracy':<15} {'ARIMA (MAE) Temp':<18}")
    print(f"{'-'*15:<15} {'-'*15:<15} {'-'*18:<18}")

    for i, (train_start, train_end, test_start, test_end) in enumerate(sessions):
        print(
            f"Session {i+1}: Train {train_start}-{train_end-1}, Test {test_start}-{test_end-1}"
        )
        rf_acc, arima_temp_mae = hybrid_trainer.train_and_evaluate(
            weather_dataset_full,
            daily_avg_temp_series_full,
            train_start,
            train_end,
            test_start,
            test_end,
            i + 1,
        )
        rf_acc_str = f"{rf_acc:.3f}" if not np.isnan(rf_acc) else "N/A"
        arima_mae_str = (
            f"{arima_temp_mae:.2f} C" if not np.isnan(arima_temp_mae) else "N/A"
        )
        print(f"{i + 1:<15} {rf_acc_str:<15} {arima_mae_str:<18}")

    avg_accuracy, avg_mae = hybrid_trainer.get_average_metrics()
    print("\n--- Hybrid Ensemble Model Overall Summary ---")
    print(f"Average Weather Category Accuracy: {avg_accuracy:.3f}")
    print(f"Average Temperature MAE: {avg_mae:.2f} C")
    print("------------------------------------------")

    print("\nTraining final models on all available data...")
    if not weather_dataset_full.empty and not daily_avg_temp_series_full.empty:
        # For final model, train on all data. The scaler and kmeans model from the last session will be used.
        # Or, ideally, refit scaler and kmeans on all data before this final training.
        # For simplicity, we use the state of the trainer after the last walk-forward session.
        # The .rf_model and .arima_model_fit will be overwritten here with models trained on all data.
        hybrid_trainer.train_and_evaluate(
            weather_dataset_full,
            daily_avg_temp_series_full,
            0,
            len(weather_dataset_full),
            0,
            0,  # No actual test set for this final training step for saving
            "Final_Full_Data",
        )
    hybrid_trainer.save_models()

    if db_manager.conn:
        db_manager.close_connection()
    print("Training script finished.")


if __name__ == "__main__":
    main()
