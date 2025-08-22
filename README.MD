## DATA ##
Kagle data >>https://www.kaggle.com/datasets/muhammedderric/fitness-classification-dataset-synthetic/data


Real-World Style Fitness Classification Dataset (Synthetic)

Dataset Description

This synthetic dataset simulates a real-world binary classification problem where the goal is to predict whether a person is fit (is_fit = 1) or not fit (is_fit = 0) based on various health and lifestyle features.

The dataset contains 2000 samples with a mixture of numerical and categorical features, some of which include noisy, inconsistent, or missing values to reflect real-life data challenges. This design enables users, especially beginners, to practice data preprocessing, feature engineering, and building classification models such as neural networks.

Features have both linear and non-linear relationships with the target variable. Some features have complex interactions and the target is generated using a sigmoid-like function with added noise, making it a challenging but realistic task. The dataset also includes mixed data types (e.g., the "smokes" column contains both numeric and string values) and some outliers are present.

This dataset is ideal for users wanting to improve skills in cleaning messy data, encoding categorical variables, handling missing values, detecting outliers, and training classification models including neural networks.

Column Descriptions
Column Name	Description
age	Age of the individual in years (integer)
height_cm	Height in centimeters (integer)
weight_kg	Weight in kilograms (integer, contains some outliers)
heart_rate	Resting heart rate in beats per minute (float)
blood_pressure	Systolic blood pressure in mmHg (float)
sleep_hours	Average hours of sleep per day (float, may contain NaNs)
nutrition_quality	Daily nutrition quality score between 0 and 10 (float)
activity_index	Physical activity level score between 1 and 5 (float)
smokes	Smoking status (mixed types: 0, 1, "yes", "no")
gender	Gender of individual, either 'M' or 'F'
is_fit	Target variable: 1 if the person is fit, 0 otherwise

Dataset Statistics
Total samples: 2000
Features: 10 (9 predictive features + 1 target)
Target distribution: Approximately 60% not fit (0), 40% fit (1)
Missing values: ~8% missing values in sleep_hours column
Data types: Mixed (integers, floats, strings)
Outliers: Present in weight_kg column (~2% of samples)
Data Quality Issues (Intentional)

This dataset intentionally includes several data quality issues to simulate real-world scenarios:

Mixed data types: The 'smokes' column contains both numeric (0, 1) and string ("yes", "no") values
Missing values: The 'sleep_hours' column has approximately 8% missing values
Outliers: The 'weight_kg' column contains some extreme values (very low or very high weights)
Noise: All features contain some level of noise to make the classification task more realistic
Suggested Data Preprocessing Steps
Handle mixed data types: Convert the 'smokes' column to a consistent format
Deal with missing values: Impute or remove missing values in 'sleep_hours'
Outlier detection: Identify and handle outliers in 'weight_kg'
Feature engineering: Consider creating BMI from height and weight
Encoding: One-hot encode categorical variables like 'gender'
Scaling: Normalize or standardize numerical features for neural networks

Potential Use Cases
Binary classification: Predict fitness status
Data preprocessing practice: Clean and prepare messy data
Feature engineering: Create new meaningful features
Model comparison: Compare different classification algorithms
Neural network training: Practice building and tuning neural networks
Exploratory data analysis: Understand relationships between health metrics
Model Performance Expectations
Due to the synthetic nature and intentional noise, expect:

Baseline accuracy: ~60% (majority class)
Good models: 75-85% accuracy
Excellent models: 85-90% accuracy
The dataset is designed to be challenging but achievable, making it perfect for learning and experimentation.


## Folder Stuture ##
fitness_prediction/
|── ftness_venv                  # create virtual env.
│── .github/                     # GitHub workflows for CI/CD (later)
│   └── workflows/
│       └── ci-cd.yml            # GitHub Actions pipeline config
│
│── data/                        # Raw and processed datasets
│   ├── raw/                     # Original CSVs (unmodified)
│   │   └── ipl_matches.csv
│   └── processed/                # Cleaned and feature-engineered datasets
│
│── notebooks/                   # Jupyter notebooks for EDA & experiments
│   └── 01_EDA.ipynb
│   └── 02_Model_Training.ipynb
│
│── src/                         # All source code (main project logic)
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data cleaning & feature engineering
│   ├── model_training.py        # Model training script
│   ├── predict.py               # Prediction script (for inference)
│   └── utils/                   # Utility/helper functions
│       └── logger.py
│       └── config.py
│
│── models/                      # Saved models & artifacts
│   └── model.pkl
│── app/
|   |--__init__.py
│   ├── main.py
│   ├── schemas.py
│   |── model_utils.py
|
│── templates/                      # Saved models & artifacts
│   ├── index.html
│   └── results.html
│── static/                     # Automation scripts (optional)
│   └── style.css
|
│── tests/                       # Unit & integration tests
│   ├── test_data_preprocessing.py
│   ├── test_train.py
│   └── test_predict.py
│
│── scripts/                     # Automation scripts (optional)
│   └── run_pipeline.sh          # Run preprocessing, training & evaluation
│
│── requirements.txt             # Python dependencies
│── setup.py                     # Make package installable (optional)
│── config.yaml                  # Config file (hyperparams, paths)
│── README.md                    # Project documentation
│── .gitignore                   # Ignore unnecessary files


## Create virtual env ##
 1. Reach to the correct directory
 cd C:\Projects\fitness_prediction> 

 2. initiate vitual env
 python -m venv fitness_venv

 3. actiavte virtual env to correct directory
 .\fitness_venv\Scripts\Activate

  4. install the required packegaes to local env
  pip install requirements.txt

  5 collection on installed packages
   pip freeze > requirements.txt

## Run the pipelines ##
1. Run the data preprocessing pipline from local of path "C:\Projects\fitness_prediction> " as 
python src\data_preprocessing.py

2. Run the modeltraining and evaluation pipline from same path "C:\Projects\fitness_prediction> " as
python src\model_training.py

3. To check wheather app is working on correct server we can run from same path "C:\Projects\fitness_prediction> " with powershell console as
uvicorn app.main:app --reload --port 8080 --log-level debug

4. we can access the web app on UI and through Fast API
 a)http://127.0.0.1:8080 
 b)python src/predict.py   from same path as "C:\Projects\fitness_prediction> "

## 🏥 Use Case: Health Insurance Risk Assessment ##
🎯 Objective

Leverage the fitness classification dataset to predict an individual’s health risk profile (fit vs. not fit) and translate it into an actionable insurance risk score. This score helps insurers in premium pricing, underwriting decisions, preventive care programs, and fraud detection.

🔍 Problem Context

In the health insurance industry, one of the biggest challenges is accurately assessing a customer’s risk level at the time of policy issuance. Traditional approaches rely heavily on manual assessments, lengthy medical tests, or flat pricing models, which can lead to:

Overcharging healthy customers (causing dissatisfaction).

Undercharging high-risk customers (increasing claims cost).

Missed opportunities for early intervention through wellness programs.

💡 Proposed Solution

Using this dataset, we can build an ML pipeline that predicts an individual’s fitness status (is_fit) and converts it into a risk score. This score can be used by insurance companies to:

Premium Pricing – Adjust premiums based on health risk tiers.

Early Interventions – Recommend preventive healthcare programs (e.g., nutrition, gym subsidies).

Fraud Detection – Flag inconsistent or unrealistic self-reported health data.

Faster Underwriting – Automate initial screening and reduce manual workload.b


## Data ##

As Windows system has issue to directly load the data from kaggle so I have local downloaded the data, unzipped and then pasted the fitness_dataset.csv file to C:\Projects\fitness_prediction\data\raw directory


Got it 👍 You want to document your **FastAPI-based ML app** so it looks professional and standard for GitHub (and later portfolio/resume). Here are some **points you can include**:

---

### 📌 About the App

* **Purpose**: This is a **Fitness Prediction Web Application** that predicts fitness-related outcomes (e.g., BMI category, calorie needs, health risks) based on user inputs such as age, gender, height, and weight.
* **Tech Stack**:

  * **FastAPI** for building RESTful APIs
  * **HTML + Jinja2 Templates** for the user interface
  * **Scikit-learn / ML model** for predictions
  * **Uvicorn** as the ASGI server

---

### 📌 Key Features

* **User-Friendly Web Form**: Input details like age, height, weight, gender, etc., via a simple HTML form.
* **REST API Endpoint**: Supports JSON-based POST requests for predictions, useful for integration with other applications.
* **Fast and Lightweight**: Powered by FastAPI, which provides asynchronous, high-performance APIs.
* **Scalable Design**: Built with MLOps principles in mind (can be extended to CI/CD pipelines, Dockerization, cloud deployment).
* **Error Handling**: Provides structured JSON error responses for invalid inputs.

---

### 📌 API Endpoints

* `GET /` → Returns the home page with input form.
* `POST /predict` → Accepts form input or JSON request body and returns predictions.

---

### 📌 Example API Request

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 30, "weight": 73, "height": 158, "gender": "Male"}'
```

---

### 📌 Future Enhancements

* Add **model monitoring & logging** (MLflow / Prometheus).
* Automate training & retraining with **CI/CD pipeline**.
* Deploy to **Azure Web App / AWS Lambda / GCP Cloud Run**.
* Containerize with **Docker + Kubernetes** for scaling.

---

👉 You can put this in your **README.md** or GitHub project description.

Would you like me to **write a full professional README.md template** (with installation, usage, pipeline, deployment instructions) for this project?
