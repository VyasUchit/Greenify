# """
# Data Processor Module for GreenReArchitect

# This module handles data cleaning, normalization, and preprocessing
# using Pandas and NumPy.

# Unit 2: Data Manipulation & Analysis
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path

# class DataProcessor:
#     """
#     Class for processing satellite data and heatmaps.
#     """

#     def __init__(self, raw_data_path: str, processed_data_path: str):
#         """
#         Initialize the DataProcessor.

#         Parameters:
#         - raw_data_path: Path to raw data directory
#         - processed_data_path: Path to processed data directory
#         """
#         self.raw_data_path = Path(raw_data_path)
#         self.processed_data_path = Path(processed_data_path)
#         self.processed_data_path.mkdir(parents=True, exist_ok=True)

#     def load_raw_data(self, filename: str) -> pd.DataFrame:
#         """
#         Load raw data from CSV or other formats.

#         Parameters:
#         - filename: Name of the file in raw data directory

#         Returns:
#         - Pandas DataFrame
#         """
#         file_path = self.raw_data_path / filename
#         if file_path.suffix == '.csv':
#             return pd.read_csv(file_path)
#         elif file_path.suffix == '.json':
#             return pd.read_json(file_path)
#         else:
#             raise ValueError(f"Unsupported file format: {file_path.suffix}")

#     def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Clean the data by handling missing values, outliers, etc.

#         Parameters:
#         - df: Input DataFrame

#         Returns:
#         - Cleaned DataFrame
#         """
#         # Remove rows with all NaN values
#         df = df.dropna(how='all')

#         # Fill missing values with mean for numeric columns
#         numeric_columns = df.select_dtypes(include=[np.number]).columns
#         df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

#         # Remove outliers using IQR method
#         for col in numeric_columns:
#             Q1 = df[col].quantile(0.25)
#             Q3 = df[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
#             df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

#         return df

#     def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Normalize the data using Min-Max scaling.

#         Parameters:
#         - df: Input DataFrame

#         Returns:
#         - Normalized DataFrame
#         """
#         numeric_columns = df.select_dtypes(include=[np.number]).columns
#         df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
#         return df

#     def save_processed_data(self, df: pd.DataFrame, filename: str):
#         """
#         Save processed data to the processed data directory.

#         Parameters:
#         - df: DataFrame to save
#         - filename: Output filename
#         """
#         output_path = self.processed_data_path / filename
#         df.to_csv(output_path, index=False)
#         print(f"Processed data saved to {output_path}")

# # Example usage
# if __name__ == "__main__":
#     processor = DataProcessor('data/raw', 'data/processed')

#     # Load raw data
#     raw_df = processor.load_raw_data('satellite_data.csv')

#     # Clean and normalize
#     cleaned_df = processor.clean_data(raw_df)
#     normalized_df = processor.normalize_data(cleaned_df)

#     # Save processed data
#     processor.save_processed_data(normalized_df, 'processed_satellite_data.csv')

"""
Data Processor Module for GreenReArchitect

This module handles data cleaning, normalization, alignment, and ML classification
using Pandas, NumPy, and Scikit-learn.

Unit 2: Data Manipulation & Analysis
Unit 3: Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
import cv2

class HeatIslandClassifier:
    """
    Class for classifying urban heat islands using ML.
    """

    def __init__(self):
        """
        Initialize the classifier.
        """
        self.model = None
        self.kmeans = None

    def prepare_training_data(self, stats_dict):
        """
        Prepare training data from satellite statistics.

        Parameters:
        - stats_dict: Dictionary with LST, NDVI, NDBI values

        Returns:
        - DataFrame with features and labels
        """
        # Mock training data based on typical urban heat island patterns
        # In real scenario, this would be actual labeled data
        data = []
        for i in range(1000):
            lst = np.random.normal(300, 20)  # Kelvin
            ndvi = np.random.normal(0.3, 0.2)  # Vegetation index
            ndbi = np.random.normal(0.1, 0.15)  # Built-up index

            # Classification logic based on thresholds
            if lst > 320 and ndvi < 0.2 and ndbi > 0.2:
                label = 'Critical Heat Zone'
            elif lst > 310 and (ndvi < 0.3 or ndbi > 0.1):
                label = 'At-Risk'
            else:
                label = 'Safe'

            data.append({
                'LST': lst,
                'NDVI': ndvi,
                'NDBI': ndbi,
                'label': label
            })

        df = pd.DataFrame(data)
        return df

    def train_classifier(self, training_data):
        """
        Train Random Forest classifier.

        Parameters:
        - training_data: DataFrame with features and labels
        """
        X = training_data[['LST', 'NDVI', 'NDBI']]
        y = training_data['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Print classification report
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    def classify_region(self, stats_dict):
        """
        Classify a region based on its statistics.

        Parameters:
        - stats_dict: Dictionary with LST, NDVI, NDBI values

        Returns:
        - Classification label
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_classifier first.")

        # Extract only the required features in the correct order
        try:
            input_data = pd.DataFrame([[
                stats_dict.get('LST', 0),
                stats_dict.get('NDVI', 0),
                stats_dict.get('NDBI', 0)
            ]], columns=['LST', 'NDVI', 'NDBI'])
        except Exception as e:
            print(f"Error preparing input data: {e}")
            return "Error"

        # Predict
        try:
            prediction = self.model.predict(input_data)[0]
            return prediction
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown"

    def cluster_hotspots(self, data_points, n_clusters=3):
        """
        Use K-Means to cluster hotspot data.

        Parameters:
        - data_points: Array of data points
        - n_clusters: Number of clusters

        Returns:
        - Cluster labels
        """
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.kmeans.fit_predict(data_points)
        return labels

    def save_model(self, filepath):
        """
        Save the trained model.

        Parameters:
        - filepath: Path to save the model
        """
        if self.model:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model.

        Parameters:
        - filepath: Path to the model file
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

def convert_to_celsius(ee_image):
    """
    Converts MODIS LST (Kelvin with scale factor) to Celsius.
    """
    # MODIS LST scale factor is 0.02. Formula: (DN * 0.02) - 273.15
    celsius_image = ee_image.multiply(0.02).subtract(273.15)
    return celsius_image

def predict_temperature_reduction(green_area_increase):
    """
    Predict temperature reduction based on green area increase.
    Uses literature-based assumptions.

    Parameters:
    - green_area_increase: Percentage increase in green area

    Returns:
    - Predicted temperature reduction in Celsius
    """
    # Based on research: 1% green area increase ≈ 0.1-0.2°C reduction
    reduction = green_area_increase * 0.15
    return reduction

def clean_satellite_data(df):
    """
    Clean satellite data by handling missing values and outliers.

    Parameters:
    - df: DataFrame with satellite data

    Returns:
    - Cleaned DataFrame
    """
    # Remove rows with all NaN values
    df = df.dropna(how='all')

    # Fill missing values with mean for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Remove outliers using IQR method for LST, NDVI, NDBI
    for col in ['LST', 'NDVI', 'NDBI']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df

def normalize_satellite_data(df):
    """
    Normalize satellite data using Min-Max scaling.

    Parameters:
    - df: DataFrame with satellite data

    Returns:
    - Normalized DataFrame
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_normalized = df.copy()
    df_normalized[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
    return df_normalized

def align_satellite_images(lst_image, rgb_image, target_size=(256, 256)):
    """
    Align satellite images to ensure spatial consistency.

    Parameters:
    - lst_image: Land surface temperature image array
    - rgb_image: RGB satellite image array
    - target_size: Target size for alignment

    Returns:
    - Tuple of aligned LST and RGB images
    """
    # Resize LST image to match RGB
    lst_resized = cv2.resize(lst_image, target_size, interpolation=cv2.INTER_LINEAR)

    # Resize RGB image
    rgb_resized = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_LINEAR)

    return lst_resized, rgb_resized

def identify_critical_heat_zones(lst_data, rural_baseline_temp):
    """
    Identify critical heat zones where temperature exceeds rural baseline.

    Parameters:
    - lst_data: Land surface temperature data array
    - rural_baseline_temp: Rural baseline temperature in Celsius

    Returns:
    - Binary mask of critical heat zones
    """
    # Critical zones: temperature > rural baseline + 5°C
    critical_threshold = rural_baseline_temp + 5.0
    heat_zones = (lst_data > critical_threshold).astype(np.uint8)

    return heat_zones

# Example usage
if __name__ == "__main__":
    classifier = HeatIslandClassifier()

    # Prepare and train model
    training_data = classifier.prepare_training_data({})
    classifier.train_classifier(training_data)

    # Example classification
    sample_stats = {'LST': 315, 'NDVI': 0.15, 'NDBI': 0.25}
    classification = classifier.classify_region(sample_stats)
    print(f"Region classification: {classification}")

    # Save model
    classifier.save_model('models/heat_classifier.pkl')
