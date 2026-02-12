import ee
from src.data_loader import authenticate_earth_engine, fetch_satellite_image

PROJECT_ID = "greenify-486918"

# Authenticate Earth Engine with correct project
authenticate_earth_engine(PROJECT_ID)

# Test location (Ahmedabad)
coords = [72.5714, 23.0225]

lst, roi = fetch_satellite_image(
    coords=coords,
    start_date="2024-05-01",
    end_date="2024-05-31"
)

temp = lst.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=roi,
    scale=1000
).getInfo()

print("🌡️ Surface temperature (°C):", temp)
