-- CREATE DATABASE nyc;

-- Enable PostGIS (includes raster)
CREATE EXTENSION postgis;

-- Enable Topology
CREATE EXTENSION postgis_topology;

-- Enable PostGIS Advanced 3D and other geoprocessing algorithms
CREATE EXTENSION postgis_sfcgal;

-- Fuzzy matching needed for Tiger
CREATE EXTENSION fuzzystrmatch;

-- Rule based standardizer
CREATE EXTENSION address_standardizer;

-- Example rule data set
CREATE EXTENSION address_standardizer_data_us;

-- Enable US Tiger Geocoder
CREATE EXTENSION postgis_tiger_geocoder;


-- postgis_topology (PostGIS Topology) allows topological vector data to be stored in a PostGIS database.
-- CREATE EXTENSION postgis;
-- CREATE EXTENSION postgis_topology;
-- fuzzystrmatch allows functions to determine string similarities and the distance between strings.
-- CREATE EXTENSION fuzzystrmatch;
-- postgis_tiger_geocoder utilizes US Census data for geocoding that deals with US data
-- Tiger extension is for geocoding any address in the US into corresponding latitude and longitude coordinates. The Topologically Integrated Geographic Encoding and Referencing (TIGER)
-- CREATE EXTENSION postgis_tiger_geocoder;
-- address_standardizer is a single-line address parser that takes an input address and normalizes it based on a set of rules.
-- CREATE EXTENSION address_standardizer;

-- Verify the extensions are installed successfully
SELECT name, default_version,installed_version
FROM pg_available_extensions WHERE name LIKE 'postgis%' or name LIKE 'address%';
