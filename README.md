# ✈️ Flight Delay Prediction System

An end-to-end machine learning application that predicts **weather-related flight delay risk** using historical flight records and weather conditions.

## Business Question

**Can we estimate the probability that a scheduled flight will be delayed because of weather, using flight timing, route, and weather characteristics?**

## Solution

The project combines historical flight and weather data, engineers temporal and weather features, trains an **XGBoost classifier**, and serves predictions through a **Streamlit** web application deployed to **Azure App Service**.

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- Category Encoders
- XGBoost
- Plotly
- Streamlit
- Azure App Service
- GitHub Actions

## Project Structure

```text
flight-delay-predictor/
├── app.py
├── flight_predictor.py
├── model_training.py
├── requirements.txt
├── azure-deploy.yml
├── startup.txt
├── .gitignore
│
├── models/
│   └── flight_delay_predictor.pkl
│
├── data/
│   └── test_data.csv
│
├── notebooks/
│   └── 01_eda_and_data_cleaning.ipynb
│
└── docs/
    └── project_report.md
```

## Machine Learning Pipeline

```text
Historical Flight + Weather Data
              ↓
        Data Cleaning
              ↓
       Feature Engineering
              ↓
     Target / Class Preparation
              ↓
       Target Encoding
              ↓
       Feature Selection (RFE)
              ↓
          XGBoost Model
              ↓
       Probability Prediction
              ↓
        Streamlit Web App
              ↓
          Azure App Service
```

## Key Features

### Temporal
- Month
- Day of week
- Hour of departure
- Season
- Time category

### Weather
- Weather type
- Weather severity
- Precipitation
- Precipitation category
- Heavy-rain indicator
- Snow-storm indicator

### Flight
- Origin airport
- Destination airport
- Scheduled departure time

## Model

The training workflow uses:
- `TargetEncoder` for categorical variables
- `StandardScaler` for selected numerical variables
- RFE with Random Forest for feature selection
- XGBoost for final classification
- F1-score, ROC-AUC, accuracy, and classification report for evaluation

The saved model pipeline is stored in `models/flight_delay_predictor.pkl`.

## Run Locally

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Azure Deployment

The repository contains:
- `azure-deploy.yml` for GitHub Actions deployment
- `startup.txt` for the Streamlit startup command

The workflow is configured for Python 3.12 and Azure App Service.

## Important Note

`data/test_data.csv` is used by the Streamlit application to provide selectable flight records. The application loads the trained model from `models/flight_delay_predictor.pkl`.

## Documentation

Detailed project notes are available in `docs/project_report.md`.

## Status

Portfolio-ready end-to-end ML application with local execution and Azure deployment configuration.
