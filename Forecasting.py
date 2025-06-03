import math
import pandas as pd
from datetime import date, timedelta, datetime
import numpy as np
import joblib
import os
import warnings
import psycopg2
from psycopg2 import sql
from sklearn.preprocessing import PolynomialFeatures  # Required for polynomial features
from sklearn.cluster import KMeans  # Required for KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error  # Required for metrics
from sklearn.ensemble import RandomForestClassifier  # Required for RandomForest
from sklearn.preprocessing import StandardScaler  # Required for StandardScaler
import pmdarima as pm  # Required for auto_arima

# Suppress FutureWarning messages from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Configuration Class (Copied for self-containment) ---
class WeatherConfig:
    """Stores configuration settings for the weather forecasting model."""

    def __init__(self):
        self.delhi_latitude = 28.6448  # Approximate latitude for Delhi
        self.delhi_longitude = 77.2167  # Approximate longitude for Delhi
        self.radius_km = 500  # Radius for surrounding areas in kilometers
        self.num_regions = 8  # Number of surrounding regions
        self.earth_radius_km = 6371  # Earth's mean radius in kilometers

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
        self.db_password = "1234"
        self.db_host = "localhost"
        self.db_port = "5432"

        self.model_save_dir = "trained_models"  # Directory where models are saved
        # IMPORTANT: This should match the last date included in your training data
        # from Accuracy_RF_ARIMA_main.py. Adjust if your training data extends further.
        self.train_start_date = date(2020, 1, 1)  # Added missing attribute
        self.train_end_date = date(
            2024, 12, 31
        )  # Example, adjust based on your actual training data end


# --- Database Manager Class (Copied for self-containment) ---
class DatabaseManager:
    """Manages PostgreSQL database connection and operations."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.conn = None
        self.connect()

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
            self.conn.autocommit = True
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

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
                WHERE timestamp >= %s AND timestamp < %s
                ORDER BY timestamp;
            """
            )
            cur.execute(
                query, (start_date, end_date + timedelta(days=1))
            )  # Use < %s for end_date + 1 day to include full end_date
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
                "hourly_units": {},  # Dummy units
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
                WHERE region_name = %s AND timestamp >= %s AND timestamp < %s
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
                "hourly_units": {},  # Dummy units
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


# --- Weather Data Fetcher Class (Only loads from DB) ---
class WeatherDataFetcher:
    """Handles fetching weather data exclusively from the database."""

    def __init__(self, config: WeatherConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager

    def fetch_delhi_weather_data(self, start_date, end_date):
        """Fetches Delhi's historical weather data from the database."""
        return self.db_manager.load_delhi_data(start_date, end_date)

    def fetch_surrounding_weather_data(self, region_name, start_date, end_date):
        """Fetches a surrounding region's historical weather data from the database."""
        return self.db_manager.load_surrounding_data(region_name, start_date, end_date)


# --- Data Processor Class (Adapted for future feature generation) ---
class DataProcessor:
    """Processes raw weather data into features for forecasting."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.poly_transformer = None  # To store fitted PolynomialFeatures
        self.cluster_scaler = None  # To store fitted StandardScaler for clustering
        self.kmeans_model = None  # To store fitted KMeans model
        self.base_columns = (
            None  # To store the list of base columns after initial processing
        )

    def get_season(self, month):
        """Determines the season based on the month for Northern India."""
        if month in [3, 4]:
            return "Spring"
        elif month in [5, 6]:
            return "Summer"
        elif month in [7, 8, 9]:
            return "Monsoon"
        else:
            return "Winter"

    def _create_daily_base_df(self, raw_data):
        """
        Processes raw historical weather data into a structured base dataset (Pandas DataFrame)
        containing daily averages/min/max for Delhi and surrounding regions, and date features.
        Does NOT include lagged, rolling, interaction, or polynomial features.
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
                "Error: Missing or incomplete historical data for Delhi. Cannot create base dataset."
            )
            return pd.DataFrame()

        delhi_hourly = delhi_data["hourly"]
        time_stamps = delhi_hourly["time"]
        num_hours = len(time_stamps)
        hours_per_day = 24
        num_days = num_hours // hours_per_day

        for i in range(
            num_days - 1
        ):  # -1 because we need a next day for target_temperature
            start_hour_index = i * hours_per_day
            end_hour_index = start_hour_index + hours_per_day

            current_day_data = {}
            current_day_time = datetime.fromisoformat(
                time_stamps[start_hour_index]
            ).date()

            # Use Timestamp for index consistency
            current_day_data["date"] = pd.Timestamp(current_day_time).normalize()

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
                    # Only include avg/min/max for relevant variables if they were used in training
                    if var in [
                        "temperature_2m",
                        "wind_speed_10m",
                    ]:  # Example, adjust based on your training
                        current_day_data[f"delhi_max_{var}"] = np.max(
                            valid_hourly_values
                        )
                        current_day_data[f"delhi_min_{var}"] = np.min(
                            valid_hourly_values
                        )
                    if var == "precipitation":
                        current_day_data[f"delhi_sum_{var}"] = np.sum(
                            valid_hourly_values
                        )
                else:
                    current_day_data[f"delhi_avg_{var}"] = np.nan
                    if var in ["temperature_2m", "wind_speed_10m"]:
                        current_day_data[f"delhi_max_{var}"] = np.nan
                        current_day_data[f"delhi_min_{var}"] = np.nan
                    if var == "precipitation":
                        current_day_data[f"delhi_sum_{var}"] = np.nan

            for region_name in self.config.region_names:
                region_data_for_current_day = surrounding_data.get(region_name)

                if region_data_for_current_day is None or not isinstance(
                    region_data_for_current_day, dict
                ):
                    for var in self.config.weather_variables:
                        current_day_data[f"{region_name}_avg_{var}"] = np.nan
                        # Add min/max/sum if they were used for surrounding regions in training
                        if var in ["temperature_2m", "wind_speed_10m"]:
                            current_day_data[f"{region_name}_max_{var}"] = np.nan
                            current_day_data[f"{region_name}_min_{var}"] = np.nan
                        if var == "precipitation":
                            current_day_data[f"{region_name}_sum_{var}"] = np.nan
                    continue

                region_hourly = region_data_for_current_day.get("hourly", {})
                if not region_hourly or not region_hourly.get("time"):
                    for var in self.config.weather_variables:
                        current_day_data[f"{region_name}_avg_{var}"] = np.nan
                        if var in ["temperature_2m", "wind_speed_10m"]:
                            current_day_data[f"{region_name}_max_{var}"] = np.nan
                            current_day_data[f"{region_name}_min_{var}"] = np.nan
                        if var == "precipitation":
                            current_day_data[f"{region_name}_sum_{var}"] = np.nan
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
                            if var in ["temperature_2m", "wind_speed_10m"]:
                                current_day_data[f"{region_name}_max_{var}"] = np.max(
                                    valid_hourly_values
                                )
                                current_day_data[f"{region_name}_min_{var}"] = np.min(
                                    valid_hourly_values
                                )
                            if var == "precipitation":
                                current_day_data[f"{region_name}_sum_{var}"] = np.sum(
                                    valid_hourly_values
                                )
                        else:
                            current_day_data[f"{region_name}_avg_{var}"] = np.nan
                            if var in ["temperature_2m", "wind_speed_10m"]:
                                current_day_data[f"{region_name}_max_{var}"] = np.nan
                                current_day_data[f"{region_name}_min_{var}"] = np.nan
                            if var == "precipitation":
                                current_day_data[f"{region_name}_sum_{var}"] = np.nan
                else:
                    for var in self.config.weather_variables:
                        current_day_data[f"{region_name}_avg_{var}"] = np.nan
                        if var in ["temperature_2m", "wind_speed_10m"]:
                            current_day_data[f"{region_name}_max_{var}"] = np.nan
                            current_day_data[f"{region_name}_min_{var}"] = np.nan
                        if var == "precipitation":
                            current_day_data[f"{region_name}_sum_{var}"] = np.nan

            # Target temperature for the next day
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

        df_base = pd.DataFrame(dataset)
        if df_base.empty:
            return pd.DataFrame()

        df_base = df_base.set_index("date").sort_index()
        df_base = pd.get_dummies(df_base, columns=["season"], drop_first=True)
        # Ensure all columns are numeric where expected, coercing errors
        for col in df_base.columns:
            if col not in [
                "day_of_week",
                "month",
                "day_of_year",
                "week_of_year",
                "target_temperature",
            ] and not col.startswith("season_"):
                df_base[col] = pd.to_numeric(df_base[col], errors="coerce")

        df_base.dropna(
            inplace=True
        )  # Drop any rows with NaNs introduced by missing data

        return df_base

    def _add_engineered_features(self, df_base, is_training_data=True):
        """
        Adds engineered features (lagged, rolling, interaction, polynomial) and target weather
        to a base DataFrame.
        `is_training_data` flag is used to control KMeans fitting.
        """
        df = df_base.copy()  # Work on a copy to avoid modifying original base df

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

        for feature in features_to_lag_and_roll:
            if feature in df.columns:
                # Ensure the feature column is numeric before performing operations
                df[feature] = pd.to_numeric(df[feature], errors="coerce")

                # Add lagged features
                for lag in lags:
                    df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)

                # Add rolling statistics
                for window in rolling_windows:
                    df[f"{feature}_rolling_mean_{window}d"] = (
                        df[feature].rolling(window=window).mean()
                    )
                    df[f"{feature}_rolling_min_{window}d"] = (
                        df[feature].rolling(window=window).min()
                    )
                    df[f"{feature}_rolling_max_{window}d"] = (
                        df[feature].rolling(window=window).max()
                    )
                    df[f"{feature}_rolling_std_{window}d"] = (
                        df[feature].rolling(window=window).std()
                    )

                # Add difference features
                df[f"{feature}_diff_1d"] = df[feature].diff(1)

        # Add interaction terms (examples)
        if (
            "delhi_avg_temperature_2m" in df.columns
            and "delhi_avg_relative_humidity_2m" in df.columns
        ):
            df["delhi_temp_x_humidity"] = pd.to_numeric(
                df["delhi_avg_temperature_2m"], errors="coerce"
            ) * pd.to_numeric(df["delhi_avg_relative_humidity_2m"], errors="coerce")
        if (
            "delhi_avg_wind_speed_10m" in df.columns
            and "delhi_avg_cloud_cover" in df.columns
        ):
            df["delhi_wind_x_cloud"] = pd.to_numeric(
                df["delhi_avg_wind_speed_10m"], errors="coerce"
            ) * pd.to_numeric(df["delhi_avg_cloud_cover"], errors="coerce")

        # Add polynomial features for a few key features
        poly_features_base = ["delhi_avg_temperature_2m", "delhi_avg_wind_speed_10m"]
        if all(f in df.columns for f in poly_features_base):
            poly = PolynomialFeatures(degree=2, include_bias=False)

            # Ensure base features are numeric before polynomial transformation
            poly_input_df = df[poly_features_base].apply(pd.to_numeric, errors="coerce")

            # Fit poly only if training data, otherwise just transform
            if is_training_data:
                # Fit poly on the non-NaN part of the input
                poly.fit(poly_input_df.dropna())
                self.poly_transformer = poly  # Store the fitted transformer
            elif (
                self.poly_transformer is not None
            ):  # Use the stored transformer for prediction
                poly = self.poly_transformer
            else:
                print(
                    "Warning: Polynomial transformer not fitted/loaded. Skipping polynomial features."
                )
                # Return df without polynomial features if transformer is not available for prediction
                return df

            poly_transformed = poly.transform(
                poly_input_df.fillna(0)
            )  # Fillna for transformation
            poly_feature_names = poly.get_feature_names_out(poly_features_base)

            poly_df = pd.DataFrame(
                poly_transformed, columns=poly_feature_names, index=df.index
            )

            # Drop the original base features from poly_df before joining to avoid duplication
            poly_df_to_join = poly_df.drop(columns=poly_features_base, errors="ignore")

            # Use left join to keep all rows from df and add new polynomial features
            df = df.join(poly_df_to_join, how="left")

        df.dropna(inplace=True)  # Drop rows with NaNs introduced by new features

        # --- Clustering for Weather Categories (Replacing map_weather_code_to_category) ---
        clustering_features = [
            "delhi_avg_temperature_2m",
            "delhi_avg_relative_humidity_2m",
            "delhi_avg_wind_speed_10m",
            "delhi_avg_precipitation",
            "delhi_avg_cloud_cover",
        ]

        df_for_clustering = df[clustering_features].copy()
        df_for_clustering.dropna(inplace=True)

        if not df_for_clustering.empty:
            scaler_cluster = StandardScaler()
            if is_training_data:
                scaled_clustering_features = scaler_cluster.fit_transform(
                    df_for_clustering
                )
                self.cluster_scaler = scaler_cluster  # Store scaler
                n_clusters = 7
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_clustering_features)
                self.kmeans_model = kmeans  # Store KMeans model
            elif self.cluster_scaler is not None and self.kmeans_model is not None:
                scaled_clustering_features = self.cluster_scaler.transform(
                    df_for_clustering
                )
                cluster_labels = self.kmeans_model.predict(scaled_clustering_features)
            else:
                print(
                    "Warning: Clustering models not fitted/loaded. Skipping target weather prediction."
                )
                df["target_weather"] = np.nan
                return df

            df.loc[df_for_clustering.index, "target_weather"] = cluster_labels
            df["target_weather"] = df["target_weather"].astype(int)
        else:
            print(
                "Warning: Not enough data for clustering weather categories. Target weather will be missing."
            )
            df["target_weather"] = np.nan

        df.dropna(inplace=True)
        return df

    # Replace the original process_historical_data with a call to the new methods
    def process_historical_data(self, raw_data):
        base_df = self._create_daily_base_df(raw_data)
        if base_df.empty:
            return pd.DataFrame()

        # When processing full historical data for training, fit the transformers
        full_engineered_df = self._add_engineered_features(
            base_df, is_training_data=True
        )
        # Store the base columns for later use in the forecasting loop
        self.base_columns = [
            col for col in base_df.columns if col != "target_temperature"
        ]
        return full_engineered_df

    # Add the missing method
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
            return pd.Series()  # Return an empty Series for ARIMA

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

    # Update generate_features_for_forecast_day to use the new _add_engineered_features
    def generate_features_for_forecast_day(
        self,
        target_date: date,
        history_df_for_features: pd.DataFrame,
        predicted_avg_temp: float,
    ):
        """
        Generates features for a single target forecast day.
        `history_df_for_features` should contain all necessary historical data (including surrounding regions)
        to compute lags and rolling features up to the day before the target_date.
        `predicted_avg_temp` is the ARIMA forecast for the target_date.
        """
        # Create a dictionary to hold all base features for the target date
        forecast_features_dict = {}

        # 1. Add date-based features for the target date
        forecast_features_dict["day_of_week"] = target_date.weekday()
        forecast_features_dict["month"] = target_date.month
        forecast_features_dict["day_of_year"] = target_date.timetuple().tm_yday
        forecast_features_dict["week_of_year"] = target_date.isocalendar()[1]

        season = self.get_season(target_date.month)
        all_seasons = ["Spring", "Summer", "Monsoon", "Winter"]
        for s in all_seasons:
            if (
                s != "Spring"
            ):  # Assuming 'Spring' is dropped by drop_first=True in training
                forecast_features_dict[f"season_{s}"] = 1 if season == s else 0

        # 2. Add predicted average temperature for Delhi (this is a base feature for the forecast day)
        forecast_features_dict["delhi_avg_temperature_2m"] = predicted_avg_temp

        # 3. Estimate other Delhi weather variables (humidity, wind, precipitation, cloud cover)
        # Use average from last 30 historical days for these features
        recent_history_end = history_df_for_features.index.max()
        recent_history_start = recent_history_end - timedelta(days=30)

        recent_delhi_data = history_df_for_features[
            (history_df_for_features.index >= recent_history_start)
            & (history_df_for_features.index <= recent_history_end)
        ]

        # Ensure the slice itself has a unique index
        if not recent_delhi_data.index.is_unique:
            recent_delhi_data = recent_delhi_data.loc[
                ~recent_delhi_data.index.duplicated(keep="first")
            ]

        for var in [
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "cloud_cover",
        ]:
            col_name = f"delhi_avg_{var}"
            if (
                col_name in recent_delhi_data.columns
                and not recent_delhi_data[col_name].empty
            ):
                forecast_features_dict[col_name] = recent_delhi_data[col_name].mean()
            else:
                forecast_features_dict[col_name] = np.nan  # Fallback if no recent data

        # Add min/max/sum for Delhi features if they are part of base features
        for var in ["temperature_2m", "wind_speed_10m"]:
            for stat in ["min", "max"]:
                col_name = f"delhi_{stat}_{var}"
                if (
                    col_name in recent_delhi_data.columns
                    and not recent_delhi_data[col_name].empty
                ):
                    forecast_features_dict[col_name] = recent_delhi_data[
                        col_name
                    ].mean()
                else:
                    forecast_features_dict[col_name] = np.nan

        col_name_prec_sum = "delhi_sum_precipitation"
        if (
            col_name_prec_sum in recent_delhi_data.columns
            and not recent_delhi_data[col_name_prec_sum].empty
        ):
            forecast_features_dict[col_name_prec_sum] = recent_delhi_data[
                col_name_prec_sum
            ].mean()
        else:
            forecast_features_dict[col_name_prec_sum] = np.nan

        # 4. Estimate surrounding region data (average of last 30 historical days)
        for region_name in self.config.region_names:
            for var in self.config.weather_variables:
                col_name = f"{region_name}_avg_{var}"
                if (
                    col_name in recent_delhi_data.columns
                    and not recent_delhi_data[col_name].empty
                ):
                    forecast_features_dict[col_name] = recent_delhi_data[
                        col_name
                    ].mean()
                else:
                    forecast_features_dict[col_name] = np.nan

                # Add min/max/sum for surrounding regions if they are part of base features
                if var in ["temperature_2m", "wind_speed_10m"]:
                    for stat in ["min", "max"]:
                        col_name_reg = f"{region_name}_{stat}_{var}"
                        if (
                            col_name_reg in recent_delhi_data.columns
                            and not recent_delhi_data[col_name_reg].empty
                        ):
                            forecast_features_dict[col_name_reg] = recent_delhi_data[
                                col_name_reg
                            ].mean()
                        else:
                            forecast_features_dict[col_name_reg] = np.nan
                if var == "precipitation":
                    col_name_reg_sum = f"{region_name}_sum_{var}"
                    if (
                        col_name_reg_sum in recent_delhi_data.columns
                        and not recent_delhi_data[col_name_reg_sum].empty
                    ):
                        forecast_features_dict[col_name_reg_sum] = recent_delhi_data[
                            col_name_reg_sum
                        ].mean()
                    else:
                        forecast_features_dict[col_name_reg_sum] = np.nan

        # Create the initial forecast_row DataFrame from the dictionary
        # Convert target_date to pandas.Timestamp for consistency
        forecast_row_base_features = pd.DataFrame(
            [forecast_features_dict], index=[pd.Timestamp(target_date).normalize()]
        )

        # Ensure history_df_for_features has unique columns before concat
        if not history_df_for_features.columns.is_unique:
            history_df_for_features = history_df_for_features.loc[
                :, ~history_df_for_features.columns.duplicated(keep="first")
            ].copy()
            print("DEBUG: history_df_for_features columns made unique.")

        # Reindex forecast_row_base_features to match history_df_for_features columns for perfect alignment
        # This is crucial as history_df_for_features might have more base columns than initially estimated for forecast_row.
        # Use self.base_columns if available, otherwise fall back to history_df_for_features.columns
        target_base_columns = (
            self.base_columns
            if self.base_columns is not None
            else history_df_for_features.columns
        )
        forecast_row_base_features = forecast_row_base_features.reindex(
            columns=target_base_columns, fill_value=0
        )

        # Combine base historical data with the new forecast day's base features
        combined_base_df = pd.concat(
            [history_df_for_features, forecast_row_base_features]
        ).sort_index()

        # Now, add the engineered features to this combined base DataFrame
        # Pass is_training_data=False because we are generating features for prediction, not training
        full_engineered_combined_df = self._add_engineered_features(
            combined_base_df, is_training_data=False
        )

        # Extract the row for the target_date from the fully engineered DataFrame
        final_features_for_target_day = full_engineered_combined_df.loc[
            [pd.Timestamp(target_date).normalize()]
        ].copy()
        final_features_for_target_day.dropna(
            axis=1, how="all", inplace=True
        )  # Drop columns that are all NaN for this row

        return final_features_for_target_day


# --- Forecaster Class ---
class Forecaster:
    """Handles loading models and making future weather predictions."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.scaler = None
        self.kmeans_model = None
        self.rf_model = None
        self.arima_model_fit = None
        self.rf_expected_features = None  # Initialize to None

        self.db_manager = DatabaseManager(config)
        # Store historical max/min temp differences for estimation
        self.temp_max_diff_avg = None
        self.temp_min_diff_avg = None

        self.data_processor = DataProcessor(config)  # Initialize DataProcessor here
        self.data_fetcher = WeatherDataFetcher(config, self.db_manager)

        # Load models *after* data_processor is initialized, as it needs to set its internal transformers
        self.load_models()
        self._calculate_historical_temp_diffs()  # Calculate after models are loaded and data processor is ready

    def load_models(self):
        """Loads the trained models and preprocessing objects."""
        print(f"Loading models from {self.config.model_save_dir}/...")
        try:
            self.scaler = joblib.load(
                os.path.join(self.config.model_save_dir, "scaler.joblib")
            )
            self.kmeans_model = joblib.load(
                os.path.join(self.config.model_save_dir, "kmeans_model.joblib")
            )
            self.rf_model = joblib.load(
                os.path.join(self.config.model_save_dir, "random_forest_model.joblib")
            )
            self.arima_model_fit = joblib.load(
                os.path.join(self.config.model_save_dir, "arima_model.joblib")
            )
            # NEW: Load the RF feature columns
            self.rf_expected_features = joblib.load(
                os.path.join(self.config.model_save_dir, "rf_feature_columns.joblib")
            )

            # Load and set the transformers in DataProcessor
            self.data_processor.poly_transformer = joblib.load(
                os.path.join(self.config.model_save_dir, "poly_transformer.joblib")
            )
            self.data_processor.cluster_scaler = joblib.load(
                os.path.join(self.config.model_save_dir, "cluster_scaler.joblib")
            )
            self.data_processor.kmeans_model = joblib.load(
                os.path.join(self.config.model_save_dir, "kmeans_model.joblib")
            )  # KMeans is loaded twice, but harmless.

            # Load base columns
            self.data_processor.base_columns = joblib.load(
                os.path.join(self.config.model_save_dir, "base_columns.joblib")
            )

            print("Models loaded successfully.")

        except FileNotFoundError as e:
            print(
                f"Error loading models: {e}. Ensure models are trained and saved in '{self.config.model_save_dir}'."
            )
            exit()  # Exit if models cannot be loaded
        except Exception as e:
            print(f"An unexpected error occurred while loading models: {e}")
            exit()

    def _calculate_historical_temp_diffs(self):
        """Calculates average historical differences between max/min and average temperature."""
        # Use the Forecaster's own db_manager
        db_manager = self.db_manager

        # Fetch a reasonable amount of recent historical data to calculate these averages
        # Use a fixed period like the last year from the end of training data
        end_date_for_diffs = self.config.train_end_date
        start_date_for_diffs = self.config.train_end_date - timedelta(
            days=365
        )  # Last year of training data from train_end_date

        delhi_raw_data = db_manager.load_delhi_data(
            start_date_for_diffs, end_date_for_diffs
        )

        if (
            delhi_raw_data
            and delhi_raw_data.get("hourly")
            and delhi_raw_data["hourly"].get("time")
        ):
            hourly_df = pd.DataFrame(delhi_raw_data["hourly"])
            hourly_df["time"] = pd.to_datetime(hourly_df["time"])
            hourly_df = hourly_df.set_index("time")
            hourly_df.dropna(subset=["temperature_2m"], inplace=True)

            daily_avg = hourly_df["temperature_2m"].resample("D").mean()
            daily_max = hourly_df["temperature_2m"].resample("D").max()
            daily_min = hourly_df["temperature_2m"].resample("D").min()

            # Calculate differences
            temp_max_diff = (daily_max - daily_avg).dropna()
            temp_min_diff = (daily_avg - daily_min).dropna()

            if not temp_max_diff.empty:
                self.temp_max_diff_avg = temp_max_diff.mean()
            if not temp_min_diff.empty:
                self.temp_min_diff_avg = temp_min_diff.mean()

        if self.temp_max_diff_avg is None:
            print(
                "Warning: Could not calculate historical max temp difference. Using default 5.0."
            )
            self.temp_max_diff_avg = 5.0  # Default if data is insufficient
        if self.temp_min_diff_avg is None:
            print(
                "Warning: Could not calculate historical min temp difference. Using default 5.0."
            )
            self.temp_min_diff_avg = 5.0  # Default if data is insufficient

    def forecast_future_weather(self, num_days: int = 7):
        """
        Forecasts future weather conditions for a specified number of days,
        relying solely on trained models and historical data.
        """
        forecast_results = []
        today = date.today()

        # Determine the start date for fetching historical data needed for lags and rolling windows
        history_required_days = 30  # Max lag/rolling window size

        # Fetch the latest available historical data from the database
        latest_historical_end_date = today - timedelta(days=1)  # Up to yesterday
        latest_historical_start_date = latest_historical_end_date - timedelta(
            days=history_required_days + 5
        )  # A bit more buffer

        # Fetch raw historical data
        raw_historical_data = {
            "delhi": self.data_fetcher.fetch_delhi_weather_data(
                latest_historical_start_date, latest_historical_end_date
            ),
            "surrounding_regions": {},
        }
        for region_name in self.config.region_names:
            region_data = self.data_fetcher.fetch_surrounding_weather_data(
                region_name, latest_historical_start_date, latest_historical_end_date
            )
            if region_data:
                raw_historical_data["surrounding_regions"][region_name] = region_data

        if raw_historical_data is None:
            print(
                "Error: Not enough recent historical data in the database to generate forecasts."
            )
            return []

        # Process raw historical data into the daily DataFrame format containing only BASE features
        historical_daily_df_base = self.data_processor._create_daily_base_df(
            raw_historical_data
        )

        if historical_daily_df_base.empty:
            print(
                "Error: Base historical daily DataFrame is empty. Cannot generate forecasts."
            )
            return []

        # Ensure initial historical_daily_df_base has a unique and normalized index
        if not historical_daily_df_base.index.is_unique:
            historical_daily_df_base = historical_daily_df_base.loc[
                ~historical_daily_df_base.index.duplicated(keep="first")
            ]
        historical_daily_df_base.index = historical_daily_df_base.index.normalize()
        historical_daily_df_base.sort_index(inplace=True)

        # Remove 'target_temperature' from historical_daily_df_base if it exists,
        # as this DataFrame should only contain features for prediction.
        if "target_temperature" in historical_daily_df_base.columns:
            historical_daily_df_base = historical_daily_df_base.drop(
                columns=["target_temperature"]
            )

        # Store the base columns if not already set (e.g., if running without a full training phase first)
        if self.data_processor.base_columns is None:
            self.data_processor.base_columns = historical_daily_df_base.columns.tolist()

        # Get the latest daily average temperature series for ARIMA from the base data
        arima_history_series = historical_daily_df_base[
            "delhi_avg_temperature_2m"
        ].dropna()

        # Update ARIMA model with the latest historical data before forecasting
        try:
            self.arima_model_fit.update(arima_history_series)
            print("ARIMA model updated with latest historical data.")
        except Exception as e:
            print(
                f"Warning: Could not update ARIMA model with latest data: {e}. Forecasting from last trained state."
            )
            if self.arima_model_fit is None:
                print(
                    "Error: ARIMA model not available for prediction after update failure."
                )
                return []

        # Predict average temperatures for all future days with ARIMA in one go
        try:
            predicted_avg_temps_array = self.arima_model_fit.predict(n_periods=num_days)
            predicted_avg_temps_series = pd.Series(
                predicted_avg_temps_array,
                index=[today + timedelta(days=k) for k in range(num_days)],
            )
        except Exception as e:
            print(f"Error forecasting with ARIMA for {num_days} days: {e}")
            predicted_avg_temps_series = pd.Series(
                [np.nan] * num_days,
                index=[today + timedelta(days=k) for k in range(num_days)],
            )

        # Iterate through each future day for forecasting
        # `current_historical_base_df` will be updated with base features of each forecast day
        current_historical_base_df = historical_daily_df_base.copy()

        for i in range(num_days):
            forecast_date = today + timedelta(days=i)
            print(f"Generating forecast for {forecast_date.isoformat()}...")

            # Get the ARIMA predicted average temperature for this specific forecast date
            predicted_avg_temp = predicted_avg_temps_series.get(forecast_date, np.nan)

            # Pass a slice of current_historical_base_df that *excludes* the current forecast_date
            # This ensures history_df_for_features in generate_features_for_forecast_day is strictly historical base data.
            history_df_for_features_slice = current_historical_base_df[
                current_historical_base_df.index
                < pd.Timestamp(forecast_date).normalize()
            ]

            features_df_for_rf = self.data_processor.generate_features_for_forecast_day(
                forecast_date, history_df_for_features_slice, predicted_avg_temp
            )

            if features_df_for_rf.empty:
                print(
                    f"Could not generate features for {forecast_date}. Skipping RF prediction."
                )
                forecast_results.append(
                    {
                        "Date": forecast_date.isoformat(),
                        "Conditions": "N/A",
                        "Max Temp (°C)": "N/A",
                        "Min Temp (°C)": "N/A",
                        "Avg Temp (°C)": (
                            f"{predicted_avg_temp:.2f}"
                            if pd.notna(predicted_avg_temp)
                            else "N/A"
                        ),
                    }
                )
                continue

            # Reindex features_df_for_rf to match the exact columns from training
            if self.rf_expected_features is not None:
                features_df_for_rf = features_df_for_rf.reindex(
                    columns=self.rf_expected_features, fill_value=0
                )
            else:
                print(
                    "Error: RF expected features not loaded. Cannot ensure consistent prediction."
                )
                forecast_results.append(
                    {
                        "Date": forecast_date.isoformat(),
                        "Conditions": "Error",
                        "Max Temp (°C)": "Error",
                        "Min Temp (°C)": "Error",
                        "Avg Temp (°C)": (
                            f"{predicted_avg_temp:.2f}"
                            if pd.notna(predicted_avg_temp)
                            else "N/A"
                        ),
                    }
                )
                continue

            # Scale the features for RF prediction
            X_forecast_scaled = self.scaler.transform(features_df_for_rf.values)

            # Predict Weather Category
            predicted_weather_category_label = self.rf_model.predict(X_forecast_scaled)[
                0
            ]
            predicted_weather_condition = f"Cluster {predicted_weather_category_label}"

            # Estimate Max/Min Temperature
            predicted_max_temp = (
                predicted_avg_temp + self.temp_max_diff_avg
                if pd.notna(predicted_avg_temp)
                else np.nan
            )
            predicted_min_temp = (
                predicted_avg_temp - self.temp_min_diff_avg
                if pd.notna(predicted_avg_temp)
                else np.nan
            )

            forecast_results.append(
                {
                    "Date": forecast_date.isoformat(),
                    "Conditions": predicted_weather_condition,
                    "Max Temp (°C)": (
                        f"{predicted_max_temp:.2f}"
                        if pd.notna(predicted_max_temp)
                        else "N/A"
                    ),
                    "Min Temp (°C)": (
                        f"{predicted_min_temp:.2f}"
                        if pd.notna(predicted_min_temp)
                        else "N/A"
                    ),
                    "Avg Temp (°C)": (
                        f"{predicted_avg_temp:.2f}"
                        if pd.notna(predicted_avg_temp)
                        else "N/A"
                    ),
                }
            )

            # Update current_historical_base_df by appending the current forecast day's BASE features
            # This ensures the history grows for subsequent iterations with only base features.
            # Ensure the forecast_date is not already in the index before dropping
            forecast_date_ts = pd.Timestamp(forecast_date).normalize()
            if forecast_date_ts in current_historical_base_df.index:
                current_historical_base_df = current_historical_base_df.drop(
                    forecast_date_ts
                )

            # Extract only the base columns from features_df_for_rf for appending
            base_features_to_append = features_df_for_rf[
                self.data_processor.base_columns
            ].copy()

            current_historical_base_df = pd.concat(
                [current_historical_base_df, base_features_to_append]
            )
            current_historical_base_df = current_historical_base_df.loc[
                :, ~current_historical_base_df.columns.duplicated()
            ].copy()
            current_historical_base_df.index = (
                current_historical_base_df.index.normalize()
            )
            current_historical_base_df.sort_index(inplace=True)

        return forecast_results


# --- Utility Functions (for geographical calculations, copied for self-containment) ---
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


# --- Main Execution Function ---
class HybridModelTrainer:
    """Trains and evaluates a hybrid model using Random Forest for weather category and SARIMA for temperature."""

    def __init__(self):
        self.rf_model = None
        self.arima_model_fit = None
        self.weather_accuracies = []
        self.temperature_maes = []

    def train_and_evaluate(
        self,
        dataframe_full,
        temp_series_full,
        train_start_idx,
        train_end_idx,
        test_start_idx,
        test_end_idx,
        session_num,
        data_processor,  # Pass data_processor to save transformers
        config,  # Pass config to save models to correct directory
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
            scaler_rf = StandardScaler()
            X_rf_train_scaled = scaler_rf.fit_transform(X_rf_train)
            X_rf_test_scaled = scaler_rf.transform(X_rf_test)

            self.rf_model = RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced"
            )
            self.rf_model.fit(X_rf_train_scaled, y_rf_train)
            y_rf_pred = self.rf_model.predict(X_rf_test_scaled)
            rf_accuracy = accuracy_score(y_rf_test, y_rf_pred)
            self.weather_accuracies.append(rf_accuracy)

            # Save scaler and RF model
            os.makedirs(config.model_save_dir, exist_ok=True)
            joblib.dump(scaler_rf, os.path.join(config.model_save_dir, "scaler.joblib"))
            joblib.dump(
                self.rf_model,
                os.path.join(config.model_save_dir, "random_forest_model.joblib"),
            )
            joblib.dump(
                all_rf_columns,
                os.path.join(config.model_save_dir, "rf_feature_columns.joblib"),
            )

            # Save DataProcessor's internal transformers (poly_transformer, cluster_scaler, kmeans_model, base_columns)
            if data_processor.poly_transformer is not None:
                joblib.dump(
                    data_processor.poly_transformer,
                    os.path.join(config.model_save_dir, "poly_transformer.joblib"),
                )
            if data_processor.cluster_scaler is not None:
                joblib.dump(
                    data_processor.cluster_scaler,
                    os.path.join(config.model_save_dir, "cluster_scaler.joblib"),
                )
            if data_processor.kmeans_model is not None:
                joblib.dump(
                    data_processor.kmeans_model,
                    os.path.join(config.model_save_dir, "kmeans_model.joblib"),
                )
            if data_processor.base_columns is not None:
                joblib.dump(
                    data_processor.base_columns,
                    os.path.join(config.model_save_dir, "base_columns.joblib"),
                )

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

                # Make rolling predictions for SARIMA
                arima_predictions = []
                history = [x for x in arima_train_series]
                for t in range(len(arima_test_series)):
                    try:
                        # Use pmdarima.ARIMA for fitting with auto_arima found orders
                        model = pm.ARIMA(
                            order=order,
                            seasonal_order=seasonal_order,
                            suppress_warnings=True,
                        )
                        model_fit_step = model.fit(history)
                        yhat = model_fit_step.predict(n_periods=1)[
                            0
                        ]  # Forecast one step ahead
                        arima_predictions.append(yhat)
                        history.append(arima_test_series.iloc[t])
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
                    # Save ARIMA model
                    joblib.dump(
                        model_auto,
                        os.path.join(config.model_save_dir, "arima_model.joblib"),
                    )
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


def main():
    """
    Main function to run the hybrid weather forecasting model with multiple training sessions.
    """
    config = WeatherConfig()
    db_manager = DatabaseManager(config)  # Initialize DatabaseManager
    data_fetcher = WeatherDataFetcher(
        config, db_manager
    )  # Pass db_manager to DataFetcher
    data_processor = DataProcessor(config)  # Initialize DataProcessor
    hybrid_trainer = HybridModelTrainer()

    print("Initializing: Starting weather forecasting model.")  # Initializing statement

    # Calculate surrounding locations once at the beginning
    surrounding_locations = {}
    for i in range(config.num_regions):
        bearing = config.region_bearings[i]
        region_name = config.region_names[i]
        dest_lat, dest_lon = calculate_destination_point(  # Use global utility function
            config.delhi_latitude,
            config.delhi_longitude,
            config.radius_km,
            bearing,
            config.earth_radius_km,
        )
        surrounding_locations[region_name] = {
            "latitude": dest_lat,
            "longitude": dest_lon,
            "bearing": bearing,
        }

    # Initialize raw_historical_data structure
    raw_historical_data = {
        "delhi": {"hourly": {var: [] for var in config.weather_variables + ["time"]}},
        "surrounding_regions": {
            name: {"hourly": {var: [] for var in config.weather_variables + ["time"]}}
            for name in config.region_names
        },
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
            delhi_month_data = data_fetcher.fetch_delhi_weather_data(
                month_start_date,
                month_end_date,
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
                region_month_data = data_fetcher.fetch_surrounding_weather_data(
                    region_name,  # Pass region_name
                    month_start_date,
                    month_end_date,
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

    # Full dataset for Random Forest (will now include all engineered features)
    weather_dataset_full = data_processor.process_historical_data(raw_historical_data)
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
            data_processor,  # Pass data_processor
            config,  # Pass config
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

    db_manager.close_connection()  # Close DB connection at the end


# --- Script Entry Point ---
if __name__ == "__main__":
    main()
