import ee

# Step 1: Trigger the authentication flow
# A browser window will open. Log in with the account you registered.
ee.Authenticate()

# Step 2: Initialize with your specific Project ID
# Replace 'your-project-id' with the ID you noted during registration
ee.Initialize(project='green-rearchitect-486907')

# Step 3: Test the connection
print(ee.String('Hello from the Earth Engine servers!').getInfo())






# Ahmedabad point
roi = ee.Geometry.Point([72.5714, 23.0225])

# MODIS Land Surface Temperature (Day)
lst_collection = (
    ee.ImageCollection("MODIS/061/MOD11A1")
    .filterDate('2024-05-01', '2024-05-31')  # past data
    .select('LST_Day_1km')
)

# Mean LST
mean_lst = lst_collection.mean()

# Convert to Celsius
lst_celsius = mean_lst.multiply(0.02).subtract(273.15)

# Sample temperature at ROI
sample = lst_celsius.sample(
    region=roi,
    scale=1000
).first()

temp = sample.get('LST_Day_1km').getInfo()
print(f"Surface temperature at Ahmedabad: {temp:.2f} °C")
