# data_ingestion.py - Optimized for Speed & Monthly Advanced Data Fetching

import requests
import math
import pandas as pd
from datetime import date, timedelta, datetime
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import concurrent.futures  # For parallel API calls
import time  # For rate limiting and delays
import calendar  # For getting days in month


# --- Configuration Class ---
class WeatherConfig:
    """Stores configuration settings for the weather forecasting model."""

    def __init__(self):
        self.delhi_latitude = 28.6448
        self.delhi_longitude = 77.2167
        self.radius_km = 500
        self.num_regions = 8
        self.earth_radius_km = 6371

        self.region_bearings = [0, 45, 90, 135, 180, 225, 270, 315]
        self.region_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        # Basic weather variables (from your original script)
        self.weather_variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "precipitation",
            "cloud_cover",
        ]

        # New advanced weather variables from Open-Meteo
        # Based on user-provided screenshots and explicit removals.
        # !! IMPORTANT !! User MUST STILL verify ALL these names against the official Open-Meteo ERA5 API documentation table
        # if "400 Bad Request" errors persist: https://open-meteo.com/en/docs/era5-api
        self.advanced_weather_variables = [
            "dew_point_2m",
            "apparent_temperature",
            "pressure_msl",
            "surface_pressure", # Already present in previous "safer" list, confirmed by user screenshot
            "rain",             # Already present in previous "safer" list, confirmed by user screenshot
            "snowfall",         # Already present in previous "safer" list, confirmed by user screenshot
            "cloud_cover_low",  # Already present in previous "safer" list, confirmed by user screenshot
            "cloud_cover_mid",  # Already present in previous "safer" list, confirmed by user screenshot
            "cloud_cover_high", # Already present in previous "safer" list, confirmed by user screenshot
            "shortwave_radiation", # Already present in previous "safer" list, confirmed by user screenshot
            "direct_normal_irradiance", # Confirmed by user screenshot (preferred over direct_radiation)
            "diffuse_radiation",    # Already present in previous "safer" list, confirmed by user screenshot
            "sunshine_duration",
            "wind_speed_100m",
            "wind_direction_100m",
            "wind_gusts_10m",
            "et0_fao_evapotranspiration",
            "snow_depth",
            "vapour_pressure_deficit",
            "soil_temperature_0_to_7cm",    # From user screenshot
            "soil_temperature_7_to_28cm",   # From user screenshot
            "soil_temperature_28_to_100cm", # From user screenshot
            "soil_temperature_100_to_255cm",# From user screenshot
            "soil_moisture_0_to_7cm",       # From user screenshot
            "soil_moisture_7_to_28cm",      # From user screenshot
            "soil_moisture_28_to_100cm",    # From user screenshot
            "soil_moisture_100_to_255cm"    # From user screenshot
            # Explicitly removed by user:
            # "cape", "geopotential_height_500hPa", "geopotential_height_700hPa", "geopotential_height_850hPa",
            # "precipitation_probability", "soil_moisture_0_1cm", "soil_temperature_0cm",
            # "temperature_500hPa", "temperature_700hPa", "temperature_850hPa",
            # "winddirection_500hPa", "winddirection_850hPa", "windspeed_500hPa",
            # "windspeed_700hPa", "windspeed_850hPa", "lifted_index"
        ]
        self.advanced_weather_variables = sorted(
            list(set(self.advanced_weather_variables))
        )

        self.db_name = "Project_Weather_Forecasting"
        self.db_user = "postgres"
        self.db_password = "1234"  # Ensure this is secure
        self.db_host = "localhost"
        self.db_port = "5432"


# --- Database Manager Class ---
class DatabaseManager:
    """Manages PostgreSQL database connection and operations."""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.conn = None
        self.connect()
        # Create basic tables (original functionality)
        self.create_tables()
        # Create new advanced tables
        self.create_advanced_weather_tables()

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
            # print("Database connected.") # REMOVED
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def create_tables(self):  # Original method for basic data tables
        if not self.conn:
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
            print(f"Error creating basic weather tables: {e}")
        finally:
            cur.close()

    def create_advanced_weather_tables(self):
        """Creates tables for advanced weather data if they don't exist."""
        if not self.conn:
            return
        cur = self.conn.cursor()

        adv_cols_sql_list = [
            sql.Identifier(col_name) + sql.SQL(" REAL")
            for col_name in self.config.advanced_weather_variables
        ]
        adv_cols_sql = sql.SQL(", ").join(adv_cols_sql_list)

        try:
            delhi_adv_query_str = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS delhi_advanced_hourly_weather (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP UNIQUE,
                    {adv_cols}
                );
            """
            ).format(adv_cols=adv_cols_sql)
            cur.execute(delhi_adv_query_str)

            surr_adv_query_str = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS surrounding_advanced_hourly_weather (
                    id SERIAL PRIMARY KEY,
                    region_name TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    timestamp TIMESTAMP,
                    {adv_cols},
                    UNIQUE (region_name, timestamp)
                );
            """
            ).format(adv_cols=adv_cols_sql)
            cur.execute(surr_adv_query_str)
        except psycopg2.Error as e:
            print(f"Error creating advanced weather tables: {e}")
        finally:
            cur.close()

    def _prepare_rows_for_insert(
        self, df: pd.DataFrame, columns: list, is_surrounding=False
    ):
        rows = []
        base_cols_surrounding = ["region_name", "latitude", "longitude"]

        for _, row_data in df.iterrows():
            row_tuple_list = []
            if is_surrounding:
                for col in base_cols_surrounding:
                    row_tuple_list.append(
                        row_data.get(col) if pd.notna(row_data.get(col)) else None
                    )

            row_tuple_list.append(row_data["timestamp"])

            for col in columns:
                row_tuple_list.append(
                    row_data.get(col) if pd.notna(row_data.get(col)) else None
                )
            rows.append(tuple(row_tuple_list))
        return rows

    def insert_delhi_data(self, data: pd.DataFrame):
        if data.empty or not self.conn:
            return
        cur = self.conn.cursor()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        rows = self._prepare_rows_for_insert(data, self.config.weather_variables)
        try:
            cols_sql = sql.SQL(", ").join(
                map(sql.Identifier, self.config.weather_variables)
            )
            insert_query = sql.SQL(
                "INSERT INTO delhi_hourly_weather (timestamp, {cols}) VALUES %s ON CONFLICT (timestamp) DO NOTHING;"
            ).format(cols=cols_sql)
            execute_values(cur, insert_query, rows, page_size=500)
        except psycopg2.Error as e:
            print(f"Error inserting Delhi basic data: {e}")
        finally:
            cur.close()

    def insert_surrounding_data(self, data: pd.DataFrame):
        if data.empty or not self.conn:
            return
        cur = self.conn.cursor()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        rows = self._prepare_rows_for_insert(
            data, self.config.weather_variables, is_surrounding=True
        )
        try:
            cols_sql = sql.SQL(", ").join(
                map(sql.Identifier, self.config.weather_variables)
            )
            insert_query = sql.SQL(
                "INSERT INTO surrounding_hourly_weather (region_name, latitude, longitude, timestamp, {cols}) VALUES %s ON CONFLICT (region_name, timestamp) DO NOTHING;"
            ).format(cols=cols_sql)
            execute_values(cur, insert_query, rows, page_size=500)
        except psycopg2.Error as e:
            print(f"Error inserting surrounding basic data: {e}")
        finally:
            cur.close()

    def insert_advanced_delhi_data(self, data: pd.DataFrame):
        if data.empty or not self.conn:
            return
        cur = self.conn.cursor()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        rows = self._prepare_rows_for_insert(
            data, self.config.advanced_weather_variables
        )
        try:
            cols_sql = sql.SQL(", ").join(
                map(sql.Identifier, self.config.advanced_weather_variables)
            )
            insert_query = sql.SQL(
                "INSERT INTO delhi_advanced_hourly_weather (timestamp, {cols}) VALUES %s ON CONFLICT (timestamp) DO NOTHING;"
            ).format(cols=cols_sql)
            execute_values(cur, insert_query, rows, page_size=500)
        except psycopg2.Error as e:
            print(f"Error inserting Delhi advanced data: {e}")
        finally:
            cur.close()

    def insert_advanced_surrounding_data(self, data: pd.DataFrame):
        if data.empty or not self.conn:
            return
        cur = self.conn.cursor()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        rows = self._prepare_rows_for_insert(
            data, self.config.advanced_weather_variables, is_surrounding=True
        )
        try:
            cols_sql = sql.SQL(", ").join(
                map(sql.Identifier, self.config.advanced_weather_variables)
            )
            insert_query = sql.SQL(
                "INSERT INTO surrounding_advanced_hourly_weather (region_name, latitude, longitude, timestamp, {cols}) VALUES %s ON CONFLICT (region_name, timestamp) DO NOTHING;"
            ).format(cols=cols_sql)
            execute_values(cur, insert_query, rows, page_size=500)
        except psycopg2.Error as e:
            print(f"Error inserting surrounding advanced data: {e}")
        finally:
            cur.close()

    def close_connection(self):
        if self.conn:
            self.conn.close()


# --- Open-Meteo API Fetcher Class ---
class OpenMeteoAPI:
    def __init__(self):
        self.ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
        self.FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def _fetch_data_generic(
        self, url, latitude, longitude, start_date_str, end_date_str, hourly_params_list
    ):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "hourly": ",".join(hourly_params_list),
            "timezone": "Asia/Kolkata",
        }
        if url == self.ERA5_ARCHIVE_URL:
            params["past_days"] = 0

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=90)
                response.raise_for_status()  # This will raise an HTTPError for bad requests (4XX or 5XX)
                data = response.json()
                return data
            except requests.exceptions.Timeout:
                print(
                    f"Timeout fetching data for {latitude},{longitude} ({start_date_str} to {end_date_str}), attempt {attempt+1}/{max_retries}"
                )
                time.sleep(10 * (attempt + 1))
            except requests.exceptions.HTTPError as e:  # Specifically catch HTTPError to see status code
                print(
                    f"HTTPError fetching data from {url} for {latitude},{longitude} ({start_date_str} to {end_date_str}): {e.response.status_code} {e.response.reason}, attempt {attempt+1}/{max_retries}"
                )
                # If it's a 400 error, it's likely a bad parameter, print the failing URL
                if e.response.status_code == 400:
                    print(f"Failing URL: {e.response.url}")  # Print the exact URL that failed
                time.sleep(10)
            except requests.exceptions.RequestException as e:  # Catch other request exceptions (e.g., connection errors)
                print(
                    f"RequestException fetching data from {url} for {latitude},{longitude} ({start_date_str} to {end_date_str}): {e}, attempt {attempt+1}/{max_retries}"
                )
                time.sleep(10)
            if attempt == max_retries - 1:
                print(
                    f"Failed to fetch data for {latitude},{longitude} after {max_retries} attempts."
                )
        return None

    def fetch_historical_weather(
        self,
        latitude,
        longitude,
        start_date_obj: date,
        end_date_obj: date,
        variable_list: list,
    ):
        return self._fetch_data_generic(
            self.ERA5_ARCHIVE_URL,
            latitude,
            longitude,
            start_date_obj.isoformat(),
            end_date_obj.isoformat(),
            variable_list,
        )

    def fetch_current_day_forecast(
        self, latitude, longitude, current_date_obj: date, variable_list: list
    ):
        return self._fetch_data_generic(
            self.FORECAST_URL,
            latitude,
            longitude,
            current_date_obj.isoformat(),
            current_date_obj.isoformat(),
            variable_list,
        )


# --- Utility Functions ---
def calculate_destination_point(
    start_lat, start_lon, distance_km, bearing_deg, earth_radius_km
):
    start_lat_rad, start_lon_rad, bearing_rad = map(
        math.radians, [start_lat, start_lon, bearing_deg]
    )
    angular_distance = distance_km / earth_radius_km
    dest_lat_rad = math.asin(
        math.sin(start_lat_rad) * math.cos(angular_distance)
        + math.cos(start_lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    dest_lon_rad = start_lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(start_lat_rad),
        math.cos(angular_distance) - math.sin(start_lat_rad) * math.sin(dest_lat_rad),
    )
    return (math.degrees(dest_lat_rad), math.degrees(dest_lon_rad))


def process_api_response_to_dataframe(
    api_data, expected_variables, region_name=None, latitude=None, longitude=None
):
    if (
        not api_data
        or not isinstance(api_data, dict)
        or not api_data.get("hourly")
        or not isinstance(api_data["hourly"], dict)
        or not api_data["hourly"].get("time")
    ):
        print(
            f"Warning: Invalid or empty API data structure for {region_name or 'Delhi_API_Process'}."
        )
        return pd.DataFrame()

    try:
        df = pd.DataFrame(api_data["hourly"])
    except ValueError as e:
        print(
            f"Warning: Could not create DataFrame from hourly data for {region_name or 'Delhi_API_Process'}: {e}"
        )
        return pd.DataFrame()

    if "time" not in df.columns:
        print(
            f"Warning: 'time' column missing in API response hourly data for {region_name or 'Delhi_API_Process'}."
        )
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop("time", axis=1, inplace=True)

    final_cols_ordered = []
    if region_name and latitude is not None and longitude is not None:
        df["region_name"] = region_name
        df["latitude"] = latitude
        df["longitude"] = longitude
        final_cols_ordered.extend(["region_name", "latitude", "longitude"])

    final_cols_ordered.append("timestamp")

    for col in expected_variables:
        if col not in df.columns:
            df[col] = pd.NA
        final_cols_ordered.append(col)

    try:
        final_cols_ordered_unique = []
        for col in final_cols_ordered:
            if col not in final_cols_ordered_unique:
                final_cols_ordered_unique.append(col)
        df = df[final_cols_ordered_unique]
    except KeyError as e:
        print(
            f"Warning: KeyError during column selection for {region_name or 'Delhi_API_Process'}. Missing: {e}. Available: {df.columns.tolist()}"
        )
        present_expected_cols = [
            col for col in final_cols_ordered_unique if col in df.columns
        ]
        df = df[present_expected_cols]
    return df


# --- Ingestion Logic for Advanced Data (Monthly Chunks) ---
def ingest_historical_advanced_data_monthly(
    start_date_overall: date,
    end_date_overall: date,
    db_mngr: DatabaseManager,
    api_fetch: OpenMeteoAPI,
    app_conf: WeatherConfig,
    surrounding_locs_list: list,
):

    current_year = start_date_overall.year
    current_month = start_date_overall.month

    # Handle short periods that don't require monthly looping
    is_short_period = False
    if start_date_overall.year == end_date_overall.year and \
       start_date_overall.month == end_date_overall.month:
        is_short_period = True
    elif (end_date_overall - start_date_overall).days < calendar.monthrange(current_year, current_month)[1] :
         is_short_period = True


    if is_short_period:
        month_name = start_date_overall.strftime("%B")
        print(
            f"extracting data for {month_name} of {start_date_overall.year} (short period: {start_date_overall} to {end_date_overall})"
        )

        all_delhi_adv_data_month = []
        all_surrounding_adv_data_month = []

        api_data_delhi_adv = api_fetch.fetch_historical_weather(
            app_conf.delhi_latitude,
            app_conf.delhi_longitude,
            start_date_overall, 
            end_date_overall,   
            app_conf.advanced_weather_variables,
        )
        df_delhi_adv = process_api_response_to_dataframe(
            api_data_delhi_adv,
            app_conf.advanced_weather_variables,
            region_name="Delhi_Adv",
        )
        if not df_delhi_adv.empty:
            all_delhi_adv_data_month.append(df_delhi_adv)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_region = {
                executor.submit(
                    api_fetch.fetch_historical_weather,
                    coords["latitude"],
                    coords["longitude"],
                    start_date_overall, 
                    end_date_overall,   
                    app_conf.advanced_weather_variables,
                ): coords
                for coords in surrounding_locs_list
            }
            for future in concurrent.futures.as_completed(future_to_region):
                coords = future_to_region[future]
                try:
                    api_data_surr_adv = future.result()
                    df_surr_adv = process_api_response_to_dataframe(
                        api_data_surr_adv,
                        app_conf.advanced_weather_variables,
                        region_name=coords["name"],
                        latitude=coords["latitude"],
                        longitude=coords["longitude"],
                    )
                    if not df_surr_adv.empty:
                        all_surrounding_adv_data_month.append(df_surr_adv)
                except Exception as exc:
                    print(
                        f"Region {coords['name']} (advanced data short period) generated an exception: {exc}"
                    )

        if all_delhi_adv_data_month:
            final_df_delhi_adv = (
                pd.concat(all_delhi_adv_data_month)
                .drop_duplicates(subset=["timestamp"])
                .reset_index(drop=True)
            )
            db_mngr.insert_advanced_delhi_data(final_df_delhi_adv)

        if all_surrounding_adv_data_month:
            final_df_surr_adv = (
                pd.concat(all_surrounding_adv_data_month)
                .drop_duplicates(subset=["region_name", "timestamp"])
                .reset_index(drop=True)
            )
            db_mngr.insert_advanced_surrounding_data(final_df_surr_adv)
        return  # Exit after processing the short period

    # Original monthly loop for longer periods
    while True:
        month_start_date = date(current_year, current_month, 1)
        _, days_in_month = calendar.monthrange(current_year, current_month)
        month_end_date = date(current_year, current_month, days_in_month)

        if month_start_date > end_date_overall:
            break
        actual_fetch_end_date = min(month_end_date, end_date_overall)

        month_name = month_start_date.strftime("%B")
        print(f"extracting data for {month_name} of {current_year}")

        all_delhi_adv_data_month = []
        all_surrounding_adv_data_month = []

        api_data_delhi_adv = api_fetch.fetch_historical_weather(
            app_conf.delhi_latitude,
            app_conf.delhi_longitude,
            month_start_date,
            actual_fetch_end_date,
            app_conf.advanced_weather_variables,
        )
        df_delhi_adv = process_api_response_to_dataframe(
            api_data_delhi_adv,
            app_conf.advanced_weather_variables,
            region_name="Delhi_Adv",
        )
        if not df_delhi_adv.empty:
            all_delhi_adv_data_month.append(df_delhi_adv)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_region = {
                executor.submit(
                    api_fetch.fetch_historical_weather,
                    coords["latitude"],
                    coords["longitude"],
                    month_start_date,
                    actual_fetch_end_date,
                    app_conf.advanced_weather_variables,
                ): coords
                for coords in surrounding_locs_list
            }
            for future in concurrent.futures.as_completed(future_to_region):
                coords = future_to_region[future]
                try:
                    api_data_surr_adv = future.result()
                    df_surr_adv = process_api_response_to_dataframe(
                        api_data_surr_adv,
                        app_conf.advanced_weather_variables,
                        region_name=coords["name"],
                        latitude=coords["latitude"],
                        longitude=coords["longitude"],
                    )
                    if not df_surr_adv.empty:
                        all_surrounding_adv_data_month.append(df_surr_adv)
                except Exception as exc:
                    print(
                        f"Region {coords['name']} (advanced data month: {month_start_date.strftime('%Y-%m')}) generated an exception: {exc}"
                    )

        if all_delhi_adv_data_month:
            final_df_delhi_adv = (
                pd.concat(all_delhi_adv_data_month)
                .drop_duplicates(subset=["timestamp"])
                .reset_index(drop=True)
            )
            db_mngr.insert_advanced_delhi_data(final_df_delhi_adv)

        if all_surrounding_adv_data_month:
            final_df_surr_adv = (
                pd.concat(all_surrounding_adv_data_month)
                .drop_duplicates(subset=["region_name", "timestamp"])
                .reset_index(drop=True)
            )
            db_mngr.insert_advanced_surrounding_data(final_df_surr_adv)

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

        if (
            date(current_year, current_month, 1) > end_date_overall
            and month_end_date >= end_date_overall
        ):
            break

        time.sleep(5)


# --- NEW Function for Original Basic Data Ingestion ---
def ingest_original_basic_data(
    app_conf: WeatherConfig,
    db_mngr: DatabaseManager,
    api_fetch: OpenMeteoAPI,
    surrounding_locs_list: list,
    start_ingestion_date_param: date,
    end_historical_date_param: date,
    current_day_date_param: date,
):

    all_delhi_basic_data = []
    all_surrounding_basic_data = []

    if start_ingestion_date_param <= end_historical_date_param:
        api_data_delhi_hist_basic = api_fetch.fetch_historical_weather(
            app_conf.delhi_latitude,
            app_conf.delhi_longitude,
            start_ingestion_date_param,
            end_historical_date_param,
            app_conf.weather_variables,
        )
        df_delhi_hist_basic = process_api_response_to_dataframe(
            api_data_delhi_hist_basic,
            app_conf.weather_variables,
            region_name="Delhi_Basic_Hist",
        )
        if not df_delhi_hist_basic.empty:
            all_delhi_basic_data.append(df_delhi_hist_basic)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_region = {
                executor.submit(
                    api_fetch.fetch_historical_weather,
                    coords["latitude"],
                    coords["longitude"],
                    start_ingestion_date_param,
                    end_historical_date_param,
                    app_conf.weather_variables,
                ): coords
                for coords in surrounding_locs_list
            }
            for future in concurrent.futures.as_completed(future_to_region):
                coords = future_to_region[future]
                try:
                    api_data_surr_hist_basic = future.result()
                    df_surr_hist_basic = process_api_response_to_dataframe(
                        api_data_surr_hist_basic,
                        app_conf.weather_variables,
                        region_name=coords["name"],
                        latitude=coords["latitude"],
                        longitude=coords["longitude"],
                    )
                    if not df_surr_hist_basic.empty:
                        all_surrounding_basic_data.append(df_surr_hist_basic)
                except Exception as exc:
                    print(
                        f"Region {coords['name']} (basic hist data) generated an exception: {exc}"
                    )

    api_data_delhi_curr_basic = api_fetch.fetch_current_day_forecast(
        app_conf.delhi_latitude,
        app_conf.delhi_longitude,
        current_day_date_param,
        app_conf.weather_variables,
    )
    df_delhi_curr_basic = process_api_response_to_dataframe(
        api_data_delhi_curr_basic,
        app_conf.weather_variables,
        region_name="Delhi_Basic_Curr",
    )
    if not df_delhi_curr_basic.empty:
        all_delhi_basic_data.append(df_delhi_curr_basic)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_region = {
            executor.submit(
                api_fetch.fetch_current_day_forecast,
                coords["latitude"],
                coords["longitude"],
                current_day_date_param,
                app_conf.weather_variables,
            ): coords
            for coords in surrounding_locs_list
        }
        for future in concurrent.futures.as_completed(future_to_region):
            coords = future_to_region[future]
            try:
                api_data_surr_curr_basic = future.result()
                df_surr_curr_basic = process_api_response_to_dataframe(
                    api_data_surr_curr_basic,
                    app_conf.weather_variables,
                    region_name=coords["name"],
                    latitude=coords["latitude"],
                    longitude=coords["longitude"],
                )
                if not df_surr_curr_basic.empty:
                    all_surrounding_basic_data.append(df_surr_curr_basic)
            except Exception as exc:
                print(
                    f"Region {coords['name']} (basic current day data) generated an exception: {exc}"
                )

    if all_delhi_basic_data:
        final_df_delhi_basic = (
            pd.concat(all_delhi_basic_data)
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )
        db_mngr.insert_delhi_data(final_df_delhi_basic)

    if all_surrounding_basic_data:
        final_df_surr_basic = (
            pd.concat(all_surrounding_basic_data)
            .drop_duplicates(subset=["region_name", "timestamp"])
            .reset_index(drop=True)
        )
        db_mngr.insert_surrounding_data(final_df_surr_basic)


# --- Main Execution Logic ---
def main():
    app_config = WeatherConfig()
    db_manager = DatabaseManager(app_config)
    api_fetcher = OpenMeteoAPI()

    surrounding_locations_list = []
    for i in range(app_config.num_regions):
        bearing = app_config.region_bearings[i]
        region_name = app_config.region_names[i]
        dest_lat, dest_lon = calculate_destination_point(
            app_config.delhi_latitude,
            app_config.delhi_longitude,
            app_config.radius_km,
            bearing,
            app_config.earth_radius_km,
        )
        surrounding_locations_list.append(
            {"latitude": dest_lat, "longitude": dest_lon, "name": region_name}
        )

    # Set dates for advanced data ingestion
    # Fetches from Jan 1, 2020, up to yesterday (as today's ERA5 data might be incomplete)
    start_adv_ingestion_date = date(2020, 1, 1)
    # MODIFIED: For full dynamic range up to current date (yesterday for ERA5)
    end_adv_ingestion_date = date.today() - timedelta(days=1)


    if start_adv_ingestion_date <= end_adv_ingestion_date:
        ingest_historical_advanced_data_monthly(
            start_adv_ingestion_date,
            end_adv_ingestion_date,
            db_manager,
            api_fetcher,
            app_config,
            surrounding_locations_list,
        )
    else:
        print(f"Advanced data ingestion range is invalid or in the future: {start_adv_ingestion_date} to {end_adv_ingestion_date}")


    # --- Task 2: Original Basic Data Ingestion (FUNCTION CALL COMMENTED OUT) ---
    # original_start_date_for_basic = date(2025, 5, 27)
    # original_end_historical_for_basic = date.today() - timedelta(days=1)
    # original_current_day_for_basic = date.today()
    #
    # ingest_original_basic_data(
    #     app_config, db_manager, api_fetcher, surrounding_locations_list,
    #     original_start_date_for_basic,
    #     original_end_historical_for_basic,
    #     original_current_day_for_basic
    # )

    db_manager.close_connection()


if __name__ == "__main__":
    main()