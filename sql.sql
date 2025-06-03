-- SELECT *
-- FROM delhi_hourly_weather
-- WHERE timestamp::text LIKE '2025-05-%'
-- order by timestamp;
-------------------------
-- DELETE FROM delhi_hourly_weather
-- WHERE timestamp::text LIKE '2025-05-%';
---------------------------
-- SELECT *
-- FROM surrounding_hourly_weather
-- WHERE timestamp::text LIKE '2025-05-24%'
-- order by timestamp;
---------------------------
-- DELETE FROM surrounding_hourly_weather
-- WHERE timestamp::text LIKE '2025-05-%';
---------------------------
select * from delhi_hourly_weather;
select * from surrounding_hourly_weather;
---------------------------
-- DROP TABLE IF EXISTS  delhi_advanced_hourly_weather;
-- DROP TABLE IF EXISTS  surrounding_advanced_hourly_weather;
---------------------------
select * from delhi_advanced_hourly_weather;

-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "cape" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "geopotential_height_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "geopotential_height_700hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "geopotential_height_850hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "precipitation_probability" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "soil_moisture_0_1cm" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "soil_temperature_0cm" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "temperature_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "temperature_700hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "temperature_850hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "winddirection_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "winddirection_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "winddirection_850hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "windspeed_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "windspeed_700hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM delhi_advanced_hourly_weather WHERE "windspeed_850hPa" IS NOT NULL;
---------------------------
select * from surrounding_advanced_hourly_weather;

-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "cape" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "geopotential_height_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "geopotential_height_700hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "geopotential_height_850hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "precipitation_probability" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "soil_moisture_0_1cm" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "soil_temperature_0cm" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "temperature_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "temperature_700hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "temperature_850hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "winddirection_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "winddirection_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "winddirection_850hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "windspeed_500hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "windspeed_700hPa" IS NOT NULL;
-- SELECT COUNT(*) FROM surrounding_advanced_hourly_weather WHERE "windspeed_850hPa" IS NOT NULL;
---------------------------