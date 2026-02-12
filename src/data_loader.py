"""
Data Loader Module for GreenReArchitect

This module handles connecting to the Satellite API (Google Earth Engine)
and loading raw satellite imagery and metadata.

Unit 2: Data Manipulation & Analysis
"""

import ee
import geemap
import folium
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def authenticate_earth_engine(project_id):
    """
    Authenticate with Google Earth Engine API using project ID.
    """
    try:
        ee.Initialize(project=project_id)
        print("Earth Engine authenticated successfully.")
    except Exception as e:
        print(f"Authentication failed: {e}")

def initialize_ee(project_id):
    """
    Initialize Earth Engine with project ID.

    Parameters:
    - project_id: Google Cloud Project ID

    Returns:
    - Boolean indicating success
    """
    try:
        ee.Initialize(project=project_id)
        return True
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        return False

def fetch_satellite_image(coords, start_date, end_date):
    """
    Fetch satellite image for given coordinates and date range.

    Parameters:
    - coords: [longitude, latitude] coordinates
    - start_date: Start date string (YYYY-MM-DD)
    - end_date: End date string (YYYY-MM-DD)

    Returns:
    - Tuple of (processed_image, region_of_interest)
    """
    # Define region of interest
    roi = ee.Geometry.Point(coords)

    # Load Landsat 8/9 collection
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
        .sort('CLOUD_COVER')

    # Get the least cloudy image
    image = collection.first()

    # Calculate LST from thermal bands
    lst = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)

    return lst, roi

def load_satellite_data(aoi, start_date, end_date):
    """
    Load satellite data: LST (temperature) from Landsat 8/9, RGB from Sentinel-2,
    NDVI (vegetation), NDBI (built-up areas).

    Parameters:
    - aoi: Earth Engine geometry
    - start_date: String in 'YYYY-MM-DD' format
    - end_date: String in 'YYYY-MM-DD' format

    Returns:
    - Dictionary with LST, RGB, NDVI, NDBI images
    """
    # Land Surface Temperature (LST) from Landsat 8/9
    # Using thermal bands to calculate LST
    landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', 20))  # Filter low cloud cover

    def calculate_lst(image):
        # Landsat 8/9 LST calculation using thermal bands
        thermal = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
        return thermal.rename('LST')

    lst_image = landsat_collection.map(calculate_lst).mean()

    # RGB imagery from Sentinel-2
    sentinel_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B4', 'B3', 'B2'])  # RGB bands

    rgb_image = sentinel_collection.median()

    # NDVI from Sentinel-2 (more accurate than Landsat for vegetation)
    ndvi_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B8', 'B4'])  # NIR and Red bands

    def calculate_ndvi(image):
        return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    ndvi_image = ndvi_collection.map(calculate_ndvi).mean()

    # NDBI (Built-up Index) using Sentinel-2
    ndbi_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B11', 'B8'])  # SWIR and NIR bands

    def calculate_ndbi(image):
        return image.normalizedDifference(['B11', 'B8']).rename('NDBI')

    ndbi_image = ndbi_collection.map(calculate_ndbi).mean()

    return {
        'LST': lst_image,
        'RGB': rgb_image,
        'NDVI': ndvi_image,
        'NDBI': ndbi_image
    }

def get_region_stats(image_dict, aoi):
    """
    Extract mean values for LST, NDVI, NDBI in the region.
    """
    stats = {}
    for key, image in image_dict.items():
        stat = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000
        ).getInfo()
        stats[key] = stat.get(key, None)
    return stats

# Example usage
if __name__ == "__main__":
    authenticate_earth_engine()

    # Define AOI (example: San Francisco)
    aoi = ee.Geometry.Point([-122.4194, 37.7749])

    # Load data with past dates
    data = load_satellite_data(aoi, '2024-05-01', '2024-05-31')
    stats = get_region_stats(data, aoi)

    print("Region Statistics:")
    print(f"LST (Temperature): {stats['LST']} K")
    print(f"NDVI (Vegetation): {stats['NDVI']}")
    print(f"NDBI (Built-up): {stats['NDBI']}")
