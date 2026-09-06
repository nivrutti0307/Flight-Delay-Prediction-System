# ✈️ Flight Delay Prediction System
### Predicting weather-related flight-delay risk by combining historical flight data, weather events, feature engineering, XGBoost, and an interactive Streamlit application.

# 1. 🎯 Objective

### Flight Delay Prediction System

Flight delays create significant operational challenges for airlines, airports, and passengers, with adverse weather being one of the important factors that can disrupt flight operations.

The objective of this project is to build an end-to-end machine-learning system that combines historical flight information with U.S. weather-event data to identify and predict the risk of weather-related flight delays.

The project explores whether information available before a flight—such as origin, destination, scheduled departure time, weather type, weather severity, precipitation, and seasonal patterns—can be used to estimate flight-delay risk.

The system was developed as a complete data pipeline, beginning with Hadoop/HDFS and Apache Spark/PySpark for large-scale data processing, followed by data preprocessing, flight-weather integration, feature engineering, feature selection, machine-learning model development, and deployment.

The final solution uses XGBoost as the prediction model and provides an interactive Streamlit application where users can select an origin airport, destination airport, flight date, and optional time range to obtain delay-risk predictions and flight statistics.

> ### Business Question

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

# 3. ⚙️ Methodology
## 3.1 Overall Architecture

                    ┌─────────────────────┐
                    │   Flight Data       │
                    └──────────┬──────────┘
                               │
                               │
                    ┌──────────▼──────────┐
                    │   Weather Data      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Hadoop / HDFS       │
                    │ Spark / PySpark     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Flight + Weather    │
                    │ Integration         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Selection   │
                    │ RFE                 │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ XGBoost Classifier  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction Pipeline │
                    └──────────┬──────────┘
                               │
                         ┌─────▼─────┐
                         │ Streamlit │
                         └─────┬─────┘
                               │
                         ┌─────▼─────┐
                         │   Azure   │
                         └───────────┘

## 3.2 Big Data Processing

The initial stage of the project used:

- Hadoop
- HDFS
- Apache Spark
- PySpark

The purpose was to establish a scalable data-processing workflow for the large flight and weather datasets.

The pipeline then moved into Python-based preprocessing and machine-learning development.

## 3.3 Data Preprocessing

The preprocessing stage included:

- Date conversion
- Time transformation
- Missing-value handling
- Data cleaning
- Feature preparation
- Flight/weather integration
- Removal of irrelevant target-related columns

## 3.4 Feature Engineering

The project engineered features from three major areas.

✈️ Flight Features
- ORIGIN
- DEST
- CRS_DEP_TIME
  
📅 Temporal Features
- MONTH
- DAY_OF_WEEK
- HOUR_OF_DAY
- SEASON
- TIME_CATEGORY
  
🌧️ Weather Features
- Type
- Severity
- SEVERITY_SCORE
- Precipitation(in)
- PRECIP_CAT
- HEAVY_RAIN
- SNOW_STORM

The feature-engineering function creates seasonal categories, time categories, weather severity scores, precipitation categories, heavy-rain indicators, and snow-storm indicators.

## 3.5 Categorical Encoding

Categorical variables were transformed using Target Encoding.

The categorical variables include:

- ORIGIN
- DEST
- Type
- Severity
- SEASON
- TIME_CATEGORY
- PRECIP_CAT

Numerical features were standardized using:

`StandardScaler`

The preprocessing pipeline is fitted on the training data and subsequently applied to test data.

## 3.6 Feature Selection

Recursive Feature Elimination (RFE) was used with a:

`RandomForestClassifier`

to select the most relevant features.

The project selected:

`Top 12 Features`

for the final XGBoost model.

## 3.7 Model Development

### Multiple machine-learning algorithms were explored during the project.

### The final model uses:

> `XGBoost Classifier`

The model was configured with optimized parameters including:

`learning_rate = 0.05`

`max_depth = 6`

`n_estimators = 1200`

`subsample = 0.7`

`colsample_bytree = 0.7`

`reg_alpha = 1.0`

`reg_lambda = 1.5`

`gamma = 0.1`

The model also uses class weighting through `scale_pos_weight`.

## 3.8 Prediction Threshold

Instead of blindly using the default classification threshold, the project evaluates the precision-recall relationship to determine an operating threshold.

The training workflow searches for a threshold that provides at least approximately 70% precision where possible; otherwise it selects a threshold based on F1 optimization.

This threshold is stored with the trained prediction pipeline.

## 3.9 Model Serialization

The complete prediction object—including preprocessing components, selected features, model and threshold—is saved using:

`flight_delay_predictor.pkl`

This allows the Streamlit application to load the trained prediction pipeline without retraining the model.

# 4. 📈 Key Findings
## Finding 1 — Weather-related delays are highly imbalanced

Only:

>  10,736 out of 1,048,575 flights

were classified as weather-related delays.

That represents approximately:

> 1.02%

of the total flight records.

This makes class imbalance an important modeling challenge.

## Finding 2 — Balanced sampling was required

The original dataset contained:

`1,037,839 No Weather Delay`

`10,736 Weather Delay`

A balanced modeling dataset was created using:

`10,736 Weather Delay`

`10,736 No Weather Delay`

for a total of:

> 21,472 records

This ensured that the minority class was adequately represented during model development.

## Finding 3 — XGBoost achieved meaningful classification performance

The final XGBoost implementation achieved:

| Metric   |      Score |
| -------- | ---------: |
| Accuracy | **67.82%** |
| F1 Score | **67.88%** |
| ROC-AUC  | **73.74%** |

These metrics are calculated on the held-out portion of the balanced modeling dataset.

## Finding 4 — Multiple feature types contribute to prediction

The prediction pipeline combines:

`Route`
+
`Flight Schedule`
+
`Temporal Patterns`
+
`Weather Type`
+
`Weather Severity`
+
`Precipitation`

This allows the model to consider both flight characteristics and environmental conditions.

## Finding 5 — The pipeline can process separate future-style data

The project does not rely solely on the original 2019–2023 dataset for demonstrating deployment behavior.

A separate historical flight dataset was transformed into a simulated 2024 scenario and passed through the prediction pipeline.

This demonstrates how the deployed model can process previously unseen records.

# 5. 💡 Recommendations
## ✈️ Airline Operations Team

Use flight-delay risk predictions as an additional operational signal during adverse weather.

**Recommended action**:

> Flag higher-risk flights for additional operational monitoring and contingency planning.

## 🛫 Airport Operations Team

Aggregate predictions by:

- Airport
- Route
- Departure period

**Recommended action**:

> Identify periods where predicted weather-related risk is concentrated and prepare operational resources accordingly.

## 👨‍✈️ Flight Planning Team

Use route- and time-specific weather risk as an additional planning input.

**Recommended action**:

> Review high-risk routes and departure windows during severe weather conditions.

## 👨‍💻 Data Engineering Team

Move the current batch-oriented workflow toward continuously updated data ingestion.

**Recommended action**:

> Integrate live flight and weather APIs with a scheduled or streaming data pipeline.

## 🤖 ML Engineering Team

Validate the model against genuinely future labeled observations.

**Recommended action**:

Use actual future flight/weather records to evaluate:

- Generalization
- Calibration
- Precision
- Recall
- F1
- ROC-AUC

# 6. ⚠️ Limitations & Assumptions
## 1. Simulated 2024 data

The 2024 dataset is created by transforming historical 2017 records.

Therefore:

> It is a simulation and not actual 2024 operational data.

## 2. Different training and testing datasets

The primary modeling data comes from the 2019–2023 flight/weather dataset.

A separate historical dataset is used for the future-style prediction workflow.

This separation demonstrates application of the trained pipeline to a different dataset.

## 3. Different target definitions

The primary target is:

`DELAY_DUE_WEATHER_YN`

The separate flight dataset contains general flight-delay information.

Therefore, predictions generated on the simulated 2024 dataset should not be reported as validated 2024 weather-delay accuracy without an equivalent ground-truth weather-delay target.

## 4. Class balancing changes the class distribution

The original data has approximately:

`1.02% Weather Delay`

`98.98% No Weather Delay`

while the modeling dataset is balanced:

`50% Weather Delay`

`50% No Weather Delay`

Therefore, model probabilities should not automatically be interpreted as real-world population probabilities.

## 5. Weather-event matching

Weather events are event-level observations and may not perfectly capture the exact atmospheric conditions experienced by an individual flight at the precise time of departure.

## 6. No actual 2024 ground truth

Because the 2024 dataset is simulated, the project cannot claim real-world 2024 predictive accuracy.

## 7. Model validation

The model's reported accuracy, F1 and ROC-AUC come from a stratified train/test split of the balanced modeling dataset.

Future work should include true time-based and external validation using future labeled data.

# 7. 📁 Repository Guide


Flight-Delay-Prediction-System/
│
├── README.md
├── requirements.txt
├── setup.py
├── setup.sh
│
├── app.py
├── flight_predictor.py
├── model_training_code.py
│
├── flight_delay_predictor.pkl
├── test_data.csv
│
├── azure-deploy.yml
├── startup.txt
│
├── notebooks/
│   ├── data_ingestion.ipynb
│   ├── data_cleaning.ipynb
│   ├── EDA.ipynb
│   ├── model_planning.ipynb
│   ├── test_data_cleaning.ipynb
│   └── test_data_eda.ipynb
│
└── docs/
    └── project documentation


## Key Files

| File                         | Purpose                          |
| ---------------------------- | -------------------------------- |
| `app.py`                     | Streamlit web application        |
| `flight_predictor.py`        | Prediction engine                |
| `model_training_code.py`     | Model training and evaluation    |
| `flight_delay_predictor.pkl` | Serialized prediction pipeline   |
| `test_data.csv`              | Separate test/simulation dataset |
| `requirements.txt`           | Python dependencies              |
| `azure-deploy.yml`           | Azure deployment configuration   |
| `startup.txt`                | Azure startup command            |
| `setup.py`                   | Project setup automation         |


The project report describes the application as a Streamlit web application with interactive route selection, date/time filtering, risk categorization, probability visualization and route statistics.

# 8. 🖥️ Application
## Flight Search Interface

The Streamlit application allows users to select:

- Origin airport
- Destination airport
- Flight date
- Optional time range

[click here to go to dashboard]([.Dashboard.pdf](https://github.com/nivrutti0307/Flight-Delay-Prediction-System/blob/main/Dashboard.pdf))

## Flight Search Results

After selecting the route and date, the application displays flight information and delay statistics.

The application also provides:

- Color-coded delay-risk categories
- Probability information
- Flight statistics
- Route-level statistics
- Visual prediction information

The documented application architecture includes these interactive prediction and visualization features.

## 🔄 Application Workflow

             USER INPUT
                 │
       ┌─────────┼─────────┐
       ↓         ↓         ↓
    Origin   Destination   Date
                 │
                 ↓
            Time Filter
                 │
                 ↓
       Load Model + Data
                 │
                 ↓
       Feature Engineering
                 │
                 ↓
          XGBoost Model
                 │
                 ↓
             Prediction
                 │
        ┌────────┴────────┐
        ↓                 ↓
 Delay Probability    Risk Category
        │                 │
        └────────┬────────┘
                 ↓
          Streamlit UI
          
## 🛠️ Technology Stack

| Category               | Technology                          |
| ---------------------- | ----------------------------------- |
| Programming            | Python                              |
| Big Data               | Hadoop, HDFS                        |
| Distributed Processing | Apache Spark, PySpark               |
| Data Analysis          | Pandas, NumPy                       |
| Machine Learning       | Scikit-learn                        |
| ML Models              | XGBoost, Random Forest              |
| Feature Selection      | RFE                                 |
| Encoding               | Category Encoders / Target Encoding |
| Scaling                | StandardScaler                      |
| Visualization          | Plotly                              |
| Web Application        | Streamlit                           |
| Model Persistence      | Joblib                              |
| Cloud                  | Microsoft Azure                     |

## 🚀 Installation
**Requirements**
- Python 3.12+
- Git
- pip

**Clone the repository**

> `git clone <YOUR-GITHUB-REPOSITORY-URL>`

> `cd Flight-Delay-Prediction-System`

Create virtual environment

> `python -m venv .venv`

Activate environment

Windows

> `.venv\Scripts\activate`

Install dependencies

> `pip install -r requirements.txt`

Run the application

> `streamlit run app.py`

The project documentation specifies Python 3.12+, Streamlit execution through app.py, and an Azure App Service deployment configuration.

## ☁️ Azure Deployment

The application is configured for deployment using:

Microsoft Azure App Service

Runtime

> `Python 3.12`

Startup command

> Bash
> `streamlit run app.py --server.port 8000 --server.address 0.0.0.0`

The repository also includes Azure deployment configuration and a startup command for cloud deployment.

## 🔮 Future Improvements
### Data Engineering
- Live flight-data ingestion
- Live weather API integration
- Apache Kafka streaming
- Airflow orchestration
- Automated ETL pipeline
- Cloud data warehouse
### Machine Learning
- Time-based validation
- Model calibration
- More advanced ensemble models
- Hyperparameter optimization
- Model monitoring
- Concept-drift detection
### Application
- Live flight status
- Live weather conditions
- Mobile-responsive interface
- Flight notifications
- API endpoint for predictions
- User-specific alerts

These improvements would move the current project closer to a production-grade aviation intelligence platform.

## 🎯 Final Project Outcome

The Flight Delay Prediction System demonstrates an end-to-end workflow combining:

BIG DATA
Hadoop / HDFS / Spark
        ↓
DATA ENGINEERING
Flight + Weather Integration
        ↓
MACHINE LEARNING
Feature Engineering + RFE + XGBoost
        ↓
MODEL PIPELINE
Prediction + Risk Classification
        ↓
APPLICATION
Streamlit
        ↓
CLOUD
Azure App Service


### Final Model Performance

| Metric   | Performance |
| -------- | ----------: |
| Accuracy |  **67.82%** |
| F1 Score |  **67.88%** |
| ROC-AUC  |  **73.74%** |

