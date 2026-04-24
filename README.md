# Urbanization Predictor (Real Estate Sim)

Welcome to the **Urbanization Predictor** (Golden Liquid Edition)! This application leverages Machine Learning to analyze satellite imagery and predict the urban growth potential of a given area based on building density and vegetation coverage.

## Features

- **Predictive ML Pipeline**: A Random Forest classifier trained to label images as having `Low`, `Medium`, or `High` growth potential based on pixel densities.
- **Vegetation Distillation**: Automatically highlights and extracts green zones (vegetation/parks) from uploaded satellite images to display them in an isolated view.
- **Golden Liquid UI**: A sleek, dynamic front-end powered by SVG gooey filters and interactive cursor trails.
- **Automated API Training Pipeline**: A newly added coordinator script that automatically fetches satellite imagery from Mapbox, generates heuristic labels, and trains robust models without manual dataset uploading.

---

## 🛠️ Setup & Installation

1. **Clone the Repository**
2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Web Dashboard

To launch the web application (Golden Liquid UI) with prediction capabilities:

```bash
python app.py
```
*Navigate to `http://127.0.0.1:5000` in your browser.*

---

## 🧠 Model Training

You have two options for training the machine learning model: the automated API pipeline or the manual dataset method.

### Option 1: Automated API Training Pipeline (Recommended)
This pipeline uses the **ESRI World Imagery API** to automatically fetch real satellite map samples, generate heuristic segmentation masks, and retrain the model.

1. **No API Keys Required!** 
   - The pipeline now utilizes an open, free-tier-friendly API endpoint. No billing info or accounts are needed.

2. **Run the Automated Trainer**
   ```bash
   python -m ml.automated_train
   ```
   *This script fetches real tiles around randomized coordinates, downloads images to `data/raw/images`, creates heuristic masks in `data/raw/masks`, and then triggers the training script.*

### Option 2: Manual Dataset Method
1. Add your dataset (images and masks) manually to `data/raw/images` and `data/raw/masks`.
2. Train the model:
   ```bash
   python -m ml.train_model
   ```

---

## 🧪 Testing Predictions via CLI

To quickly test the model on the command line without the web dashboard:
```bash
python -m ml.predict
```
*(Make sure to have a model trained first! The script tests against default sample images `test.jpg`, `test1.jpg`, and `test2.jpg`).*