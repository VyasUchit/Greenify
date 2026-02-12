# GreenReArchitect 🌿

An AI-powered platform for urban green space redesign using satellite imagery and machine learning to combat urban heat islands.

## Overview

GreenReArchitect leverages satellite data from Google Earth Engine, computer vision techniques, and generative AI to identify heat-prone areas in cities and propose sustainable green space redesigns. This project demonstrates end-to-end machine learning engineering from data collection to model deployment.

## Features

- **Satellite Data Integration**: Connect to Google Earth Engine for real-time satellite imagery
- **Heat Island Detection**: CNN-based hotspot identification using computer vision
- **Data Processing**: Robust data cleaning and normalization pipelines
- **Generative AI**: GAN-powered green space redesign suggestions
- **Interactive Web App**: Streamlit-based deployment for easy access
- **Portfolio-Ready**: Well-documented, modular codebase following best practices

## Project Structure

```
GreenReArchitect/
├── data/                      # Dataset storage (Unit 2)
│   ├── raw/                   # Original satellite imagery/metadata
│   └── processed/             # Normalized & cleaned heatmaps
├── models/                    # Saved training checkpoints (Units 3-4)
│   ├── heat_detector.h5       # CNN for hotspot identification
│   └── green_generator.h5     # GAN for green redesigns
├── src/                       # Core logic and modules (Unit 2: OOP)
│   ├── data_loader.py         # Script to connect to Satellite API
│   ├── processor.py           # Data cleaning with Pandas/NumPy
│   └── vision_utils.py        # OpenCV and CNN functions
├── app.py                     # Main Streamlit application (Unit 7)
├── requirements.txt           # List of libraries (TensorFlow, ee, etc.)
└── README.md                  # Project documentation for portfolio
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/GreenReArchitect.git
   cd GreenReArchitect
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Earth Engine:**
   - Create a Google Cloud Project
   - Enable Earth Engine API
   - Authenticate: `earthengine authenticate`
   - Set up service account (optional for production)

5. **Configure environment variables:**
   Create a `.env` file with your API keys:
   ```
   EARTHENGINE_PROJECT=your-project-id
   ```

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the application.

### Data Processing Pipeline

```python
from src.data_loader import authenticate_earth_engine, load_satellite_data
from src.processor import DataProcessor

# Authenticate and load data
authenticate_earth_engine()
collection = load_satellite_data(aoi, '2020-01-01', '2020-12-31')

# Process data
processor = DataProcessor('data/raw', 'data/processed')
raw_df = processor.load_raw_data('satellite_data.csv')
cleaned_df = processor.clean_data(raw_df)
normalized_df = processor.normalize_data(cleaned_df)
processor.save_processed_data(normalized_df, 'processed_data.csv')
```

### Model Training

```python
from src.vision_utils import build_cnn_model, create_data_generator

# Build and train CNN model
model = build_cnn_model()
train_gen, val_gen = create_data_generator('data/processed', batch_size=32)
model.fit(train_gen, validation_data=val_gen, epochs=50)
model.save('models/heat_detector.h5')
```

## Curriculum Units Covered

- **Unit 2**: Data Manipulation & Analysis (Pandas, NumPy, OOP)
- **Unit 3**: Machine Learning (Scikit-learn)
- **Unit 4**: Deep Learning (TensorFlow/Keras)
- **Unit 5**: Computer Vision & Satellite Data (OpenCV, Earth Engine)
- **Unit 6**: Generative AI (GANs with TensorFlow)
- **Unit 7**: Model Deployment (Streamlit)

## Technologies Used

- **Data Science**: NumPy, Pandas, Scikit-learn
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Satellite Data**: Google Earth Engine API
- **Mapping**: Folium, GeoPandas
- **Web App**: Streamlit
- **Visualization**: Matplotlib, Seaborn

## Model Architecture

### Heat Detector CNN
- Input: RGB satellite images (224x224x3)
- Architecture: Conv2D → MaxPool → Conv2D → MaxPool → Dense
- Output: Heat island probability map

### Green Generator GAN
- Generator: U-Net style architecture for image-to-image translation
- Discriminator: PatchGAN for realistic green space generation
- Training: Satellite imagery → Green space redesigns

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Earth Engine for satellite data access
- TensorFlow/Keras for deep learning framework
- Streamlit for easy web app deployment
- Open-source computer vision community

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/GreenReArchitect](https://github.com/yourusername/GreenReArchitect)
