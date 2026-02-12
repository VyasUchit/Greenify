# GreenReArchitect - Technology Stack

## 📊 Project Overview
**GreenReArchitect** is an AI-powered urban heat mitigation platform that detects heat islands and visualizes green redesign solutions using satellite data and machine learning.

---

## 🎨 VISUALIZATION TECHNOLOGIES

### 1. **Streamlit** (App Framework)
- **Version:** >=1.22.0
- **Purpose:** Interactive web dashboard for UI and real-time updates
- **Usage in Project:**
  - Page navigation (Home, Location Analysis, Heat Map, Green Redesign, Impact)
  - Session state management
  - Metric displays and interactive controls

### 2. **Folium** (Map Visualization)
- **Version:** >=0.14.0
- **Purpose:** Interactive maps for geographic data visualization
- **Usage in Project:**
  - Display satellite data on maps
  - Show coordinates of analyzed locations
  - Render heat zones on map layers

### 3. **Geemap** (Earth Engine Mapping)
- **Version:** >=0.20.0
- **Purpose:** Bridge between Google Earth Engine and Python visualization
- **Usage in Project:**
  - Display satellite imagery (LST, NDVI, NDBI)
  - Create interactive map layers with Earth Engine data
  - Visualize land surface temperature with color heatmaps

### 4. **Matplotlib** (Static Plots)
- **Version:** >=3.7.0
- **Purpose:** Statistical and scientific visualization
- **Usage in Project:**
  - Plot historical trends
  - Display CNN/GAN training metrics
  - Generate comparison charts

### 5. **Seaborn** (Statistical Visualization)
- **Version:** >=0.12.0
- **Purpose:** Enhanced statistical graphics
- **Usage in Project:**
  - Heatmap visualizations
  - Distribution plots for temperature analysis
  - Correlation matrices

### 6. **Pillow** (Image Processing)
- **Version:** >=9.0.0
- **Purpose:** Image manipulation and processing
- **Usage in Project:**
  - Handle satellite image formats
  - Image resizing and conversion
  - GAN-generated image output

---

## 🤖 MACHINE LEARNING (ML) TECHNOLOGIES

### 1. **Scikit-learn** (ML Framework)
- **Version:** >=1.2.0
- **Purpose:** General machine learning algorithms
- **Usage in Project:**
  - **RandomForestClassifier:** Classifies urban heat islands into categories
    - "Critical Heat Zone" (LST > 320K, low vegetation)
    - "At-Risk" (LST > 310K)
    - "Safe" (moderate temperatures)
  - **KMeans Clustering:** Groups hotspots for spatial analysis

### 2. **Pandas** (Data Manipulation)
- **Version:** >=2.0.0
- **Purpose:** DataFrames and data processing
- **Usage in Project:**
  - Load and structure satellite statistics
  - Prepare training data for classifiers
  - Data alignment and cleaning

### 3. **NumPy** (Numerical Computing)
- **Version:** >=1.24.0
- **Purpose:** Array operations and mathematical computation
- **Usage in Project:**
  - Satellite data array processing
  - Statistical calculations (NDVI, NDBI, LST)
  - Random data generation for mock training

### 4. **JobLib** (Model Persistence)
- **Version:** >=1.3.0
- **Purpose:** Save and load trained ML models
- **Usage in Project:**
  - Serialize RandomForest classifier
  - Store model checkpoints

---

## 🧠 DEEP LEARNING (DL) TECHNOLOGIES

### 1. **TensorFlow** (DL Framework)
- **Version:** >=2.12.0
- **Purpose:** Deep learning model building and training
- **Usage in Project:**
  - **CNN (Convolutional Neural Network)** for heat detection
  - **GAN (Generative Adversarial Network)** for green space generation

### 2. **Keras** (DL API)
- **Integrated with TensorFlow 2.12+**
- **Purpose:** High-level neural network API
- **Usage in Project:**
  - Model construction (Sequential, Model APIs)
  - Layer definitions (Conv2D, Dense, Dropout, etc.)

### 3. **CNN - HeatDetectorCNN**
```
Architecture:
- Input: 224×224×3 satellite image
- Conv2D (32 filters) + MaxPooling
- Conv2D (64 filters) + MaxPooling
- Conv2D (128 filters) + MaxPooling
- Flatten + Dense(128) + Dropout(0.5)
- Output: Binary classification (hot/not hot)

Purpose: Detect heat patterns in satellite imagery
```

### 4. **GAN - GreenRedesignGAN (U-Net Style)**
```
Generator Architecture:
- Encoder: Multi-scale convolutions with pooling
- Bottleneck: Deep feature extraction
- Decoder: Transposed convolutions (upsampling)
- Skip connections: Feature preservation

Discriminator Architecture:
- PatchGAN style discriminator
- Multi-layer Conv2D with LeakyReLU
- Binary classification (real/fake)

Purpose: Generate realistic green space redesigns
```

---

## 📡 DATA SOURCE

### Google Earth Engine API
- **Version:** >=0.1.350
- **Purpose:** Access satellite data
- **Satellite Data Used:**
  - **LST (Land Surface Temperature):** MODIS thermal data (20-50°C)
  - **NDVI (Normalized Difference Vegetation Index):** Vegetation mapping
  - **NDBI (Normalized Difference Built-up Index):** Urban area detection

---

## 🔧 SUPPORTING LIBRARIES

### 1. **OpenCV (cv2)**
- **Version:** >=4.7.0
- **Purpose:** Computer vision operations
- **Usage:** Image preprocessing for CNN models

### 2. **Python-dotenv**
- **Version:** >=1.0.0
- **Purpose:** Manage API keys securely
- **Usage:** Store Google Earth Engine credentials

### 3. **SciPy**
- **Version:** >=1.10.0
- **Purpose:** Scientific computing (statistical tests)
- **Usage:** Data analysis and filtering

---

## 📋 TECHNOLOGY SUMMARY TABLE

| Layer | Technology | Purpose | Version |
|-------|-----------|---------|---------|
| **Frontend** | Streamlit | Web Dashboard | >=1.22.0 |
| **Visualization** | Folium, Geemap | Map & Spatial Viz | >=0.14.0, >=0.20.0 |
| **Visualization** | Matplotlib, Seaborn | Charts & Plots | >=3.7.0, >=0.12.0 |
| **ML** | Scikit-learn | Classification, Clustering | >=1.2.0 |
| **ML** | Pandas, NumPy | Data Processing | >=2.0.0, >=1.24.0 |
| **DL - CNN** | TensorFlow/Keras | Heat Detection | >=2.12.0 |
| **DL - GAN** | TensorFlow/Keras | Green Redesign | >=2.12.0 |
| **Data Source** | Google Earth Engine | Satellite Imagery | >=0.1.350 |
| **CV** | OpenCV | Image Preprocessing | >=4.7.0 |

---

## 🎯 WORKFLOW FLOW

```
1. DATA ACQUISITION (Google Earth Engine)
   ↓
   LST, NDVI, NDBI satellite data
   ↓
2. ML CLASSIFICATION (Scikit-learn)
   ↓
   Random Forest → Heat Zone Classification
   ↓
3. CNN ANALYSIS (TensorFlow/Keras)
   ↓
   HeatDetectorCNN → Pixel-level heat intensity
   ↓
4. GAN GENERATION (TensorFlow/Keras)
   ↓
   GreenRedesignGAN → Green space visualization
   ↓
5. VISUALIZATION (Streamlit + Folium + Geemap)
   ↓
   Interactive dashboard with maps & metrics
```

---

## 📦 Installation

All technologies are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Access at: **http://localhost:8502**

---

**Created:** February 10, 2026  
**Project:** GreenReArchitect v1.0  
**Status:** Active & Running ✅
