# ✈️ Flight Delay Prediction System
## Predicting weather-related flight-delay risk by combining historical flight data, weather events, feature engineering, XGBoost, and an interactive Streamlit application.

# 1. 🎯 Objective

### Flight Delay Prediction System

Flight delays create significant operational challenges for airlines, airports, and passengers, with adverse weather being one of the important factors that can disrupt flight operations.

The objective of this project is to build an end-to-end machine-learning system that combines historical flight information with U.S. weather-event data to identify and predict the risk of weather-related flight delays.

The project explores whether information available before a flight—such as origin, destination, scheduled departure time, weather type, weather severity, precipitation, and seasonal patterns—can be used to estimate flight-delay risk.

The system was developed as a complete data pipeline, beginning with Hadoop/HDFS and Apache Spark/PySpark for large-scale data processing, followed by data preprocessing, flight-weather integration, feature engineering, feature selection, machine-learning model development, and deployment.

The final solution uses XGBoost as the prediction model and provides an interactive Streamlit application where users can select an origin airport, destination airport, flight date, and optional time range to obtain delay-risk predictions and flight statistics.

### Business Question

Can historical flight and weather information be used to identify the risk of a weather-related flight delay?

### Project Answer

Yes. By combining flight characteristics, weather conditions, and temporal patterns, the system can estimate weather-related flight-delay risk and deliver the prediction through an interactive application.

# 2. 📊 The Data
## 2.1 Flight Data — 2019–2023

The primary flight dataset was obtained from Kaggle:

https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023

Grain

One record represents one flight.

### Project dataset
- 1,048,575 flight records
- Historical flight information covering 2019–2023

### Important variables include:

- Flight date
- Origin airport
- Destination airport
- Scheduled departure time
- Scheduled arrival time
- Scheduled elapsed time
- Flight distance
- Delay information
- Delay causes

## 2.2 U.S. Weather Events

Weather information was obtained from:

https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events

US Weather Events (2016 - 2022) and 2023 US Weather Events combined

The weather data contains event-level information such as:

- Weather type
- Severity
- Precipitation
- Event start time
- Event end time
- Location
- Geographic information

The project processed approximately 8.6 million weather-event records during the data-preparation workflow.

## 2.3 Flight + Weather Dataset

The flight and weather datasets were cleaned and integrated to create the main project dataset:

`flight_2019_2023_data.csv`

The resulting dataset contains:

1,048,575 flight records

The objective was to enrich flight-level records with corresponding weather information.

## 2.4 Target Variable

The target variable was created from the weather-delay field:

`DELAY_DUE_WEATHER_YN`

| Value | Meaning                  |
| ----- | ------------------------ |
| `Yes` | Weather-related delay    |
| `No`  | No weather-related delay |

Original distribution

| Class            |       Records | Percentage |
| ---------------- | ------------: | ---------: |
| Weather Delay    |        10,736 |      1.02% |
| No Weather Delay |     1,037,839 |     98.98% |
| **Total**        | **1,048,575** |   **100%** |

The target was therefore highly imbalanced.

### Balanced Modeling Dataset

To address the imbalance, the weather-delay class was retained and an equal number of non-weather-delay records were sampled.

10,736 Weather Delays
+
10,736 Non-Weather Delays
=
21,472 Modeling Records

The sampling and model-training workflow are implemented in the project's training script.

## 2.5 Separate Dataset for Future-Style Testing

A different flight dataset was used for the separate testing/simulation workflow:

https://www.kaggle.com/datasets/gabrielluizone/us-domestic-flights-delay-prediction-2013-2018

Instead of using this dataset for training, the project selected its 2017 flight records and combined them with the corresponding weather data.

The year was then shifted:

`2017 → 2024`

to create a simulated 2024 prediction scenario.

### Data volumes
| Dataset           |   Records |
| ----------------- | --------: |
| 2017 Flight Data  |   344,450 |
| 2017 Weather Data | 1,227,786 |
| Combined Dataset  |   624,694 |

### Important

The simulated 2024 dataset is not actual 2024 flight data.

It is a transformed historical dataset used to demonstrate how the trained prediction pipeline can be applied to a future-style dataset.
