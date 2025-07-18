# DreamSeeker V1: Personalized Solo Activity Recommendation System

## 🎯 Project Goal
To build an intelligent recommendation system that suggests personalized solo activities and experiences based on user preferences, mood, and interests. The system aims to enhance the quality of solitary experiences by providing tailored bucket list recommendations.

## 💼 Business Objective
To create a scalable platform that helps individuals discover and engage in meaningful solo activities, addressing the growing market of solo leisure and self-development while promoting mental well-being through personalized activity recommendations.

## 🎓 Project Objective
Develop a robust machine learning system that:
- Processes and analyzes user activity data to identify patterns and preferences
- Implements custom machine learning algorithms from scratch for transparent and efficient recommendations
- Delivers accurate and personalized activity suggestions through an interactive interface

## 📊 Key Accomplishments

### 1. Data Infrastructure
- **PostgreSQL Integration**: Implemented a robust PostgreSQL database for efficient data storage and retrieval
- **Optimized Schema Design**: Created a normalized database structure for activity and user preference data
- **Direct NumPy Integration**: Established direct data pipeline from PostgreSQL to NumPy arrays, bypassing pandas for improved performance

### 2. Machine Learning Implementation
- **Custom Logistic Regression**: Built from scratch using only NumPy, achieving:
  - Efficient matrix operations for faster computations
  - Custom gradient descent optimization
  - Robust model evaluation metrics
- **NumPy-Focused Architecture**: 
  - Deliberately omitted pandas to optimize memory usage and processing speed
  - Direct array operations for enhanced computational efficiency
  - Streamlined data transformations using pure NumPy operations

### 3. Visualization & Interface
- **Streamlit Dashboard**:
  - Interactive user interface for real-time recommendations
  - Dynamic parameter adjustment capabilities
  - Responsive design for various device sizes
- **Matplotlib Integration**:
  - Custom visualization components for model metrics
  - Performance tracking plots
  - User preference analysis charts

## 🏗️ Project Structure

```
DreamSeeker/
├── data/
│   └── raw/
│       └── bucket_list_extended.csv         # Cleaned dataset with user activity features
├── db/
│   └── init_postgres.sql                    # SQL script to create PostgreSQL table and load data
├── src/
│   ├── data_loader.py                       # Loads data from PostgreSQL into NumPy arrays
│   ├── feature_engineering.py               # Applies transformations to raw input features
│   ├── model_evaluation.py                  # Evaluates all 3 models (accuracy, precision, recall)
│   └── ml_algorithms/
│       ├── logistic_regression.py           # From-scratch using NumPy
│       ├── decision_tree.py                 # From-scratch using NumPy
│       └── random_forest.py                 # From-scratch using NumPy
├── main.py                                  # Pipeline entry point
└── README.md                                # Project overview and documentation
```

---

## ✅ Data Pipeline Summary

> I first designed a structured CSV dataset containing relevant user and activity attributes such as mood, activity type, and interest score. This dataset was then cleaned and preprocessed within a Python script to ensure schema consistency, eliminate missing values, and encode categorical variables if necessary.

> Once formatted, I successfully imported the cleansed CSV into a **PostgreSQL database** using a SQL initialization script. The choice to transition from CSV to PostgreSQL was made to enable **faster data retrieval, improved scalability**, and future **compatibility with production-level MLOps infrastructure** (e.g., Redis caching, real-time API ingestion).

> After verifying successful data ingestion, I queried the PostgreSQL table directly from Python and loaded the results into **NumPy arrays**. This output format allows me to perform further **feature engineering**, exploratory data analysis, and build **machine learning models from scratch using only NumPy**, aligning with the goal of building a recommendation system without external libraries.

---

## ⚙️ Feature Engineering Steps

> Feature engineering transforms raw data into meaningful inputs that improve model performance. All transformations will be coded manually using only NumPy and pandas where necessary.

### 🧼 1. Data Cleaning

* Remove duplicates and missing entries
* Ensure numerical values are valid (e.g. no NaNs in interest score)

### 🎭 2. Categorical Encoding

* `activity_type`, `category`, `user_mood`: encoded using one-hot encoding

### 📊 3. Normalization / Scaling

* `user_interest_score`: scaled between 0 and 1 using Min-Max normalization

### 🔗 4. Feature Assembly

* Combine categorical and numeric features into a final NumPy array `X`
* Target label (`y`) is the binary recommendation column (`label`)

### 🧪 5. Data Split

* Split into train/test sets using manual indexing (no sklearn)

---

## 🔍 ML Algorithms Used (From Scratch with NumPy)

| Algorithm           | Type           | Reason for Selection                          |
| ------------------- | -------------- | --------------------------------------------- |
| Logistic Regression | Classification | Lightweight, interpretable baseline           |
| Decision Tree       | Classification | Handles non-linear logic and mixed data types |
| Random Forest       | Ensemble       | Robust performance, prevents overfitting      |

These models will be implemented manually and compared using classification metrics (accuracy, precision, recall, F1). The best-performing model will be selected for final deployment.

---

## 🛠️ Developer Tools & Technologies

### Version 1 (Completed)
- **Core Technologies**:
  - Python 3.10+
  - PostgreSQL (Database)
  - NumPy (2.2.6) - Core numerical computations
  - Streamlit (1.47.0) - Web interface
  - Matplotlib (3.10.3) - Data visualization
  - psycopg2 (2.9.10) - PostgreSQL adapter
  - requests (2.32.4) - HTTP client


- **Version Control & Collaboration**:
  - Git
  - GitHub

### Version 2 (Current)
- **Infrastructure & Containerization**:
  - Docker
  - Docker Compose

- **Data Pipeline & Orchestration**:
  - Prefect - Workflow automation
  - Redis - Caching layer
  - PostgreSQL - Primary database
  - EvidentlyAI - ML monitoring

- **API Integration**:
  - Google Places API
  - Ticketmaster API
  - Eventbrite API
  - FastAPI - Backend service

- **MLOps & Monitoring**:
  - MLflow - Model tracking & versioning
  - GitHub Actions - CI/CD pipeline

---

## 🛠️ Next Steps