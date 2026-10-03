# Student Performance Analytics and Prediction System

## Project Overview
A complete, runnable college mini project that analyzes student performance data and predicts high-performing students using machine learning. Built with Python, FastAPI, and Streamlit.

## Problem Statement
Each student shall independently select a real-world problem/domain and develop a Data Analytics and Visualization-based web application using a dataset of at least 1,000 records. The project includes data preprocessing and wrangling, exploratory data analysis, meaningful visualizations, implementation and comparison of at least two Machine Learning models, and a user-friendly web interface.

## Objectives
- Download and analyze real-world student performance data
- Perform data preprocessing, cleaning, and feature engineering
- Conduct exploratory data analysis with meaningful visualizations
- Implement and compare Logistic Regression and Random Forest models
- Build a FastAPI backend for analytics and predictions
- Create an interactive Streamlit frontend for visualization and prediction
- Avoid data leakage by not using exam scores as input features

## Dataset
**Dataset:** Students Performance in Exams  
**Source:** [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)  
**Records:** ~1000 student records  
**Downloaded using:** KaggleHub

### Features
- gender
- race/ethnicity
- parental level of education
- lunch
- test preparation course
- math score
- reading score
- writing score

### Target Variable
**performance** - Binary classification  
- High Performer: average_score >= 70
- Normal Performer: average_score < 70

**Note:** Input features for ML do NOT include math score, reading score, writing score, or average_score to avoid data leakage.

## Technology Stack
- Python 3.11+
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- Joblib
- FastAPI, Uvicorn, Pydantic
- Streamlit, Requests
- KaggleHub

## Project Structure
```
student-performance-mini-project/
├── data/
│   ├── raw/StudentsPerformance.csv
│   └── processed/processed_data.csv
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
├── reports/
│   ├── figures/ (visualizations)
│   └── *.csv (report tables)
├── backend/
│   ├── main.py
│   └── schemas.py
├── frontend/
│   └── app.py
├── preprocessing.py
├── eda.py
├── train.py
├── download_data.py
├── requirements.txt
└── README.md
```

## Installation

### Create virtual environment
```bash
python -m venv venv
```

### Activate virtual environment (Windows)
```bash
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Running Instructions

### 1. Download dataset
```bash
python download_data.py
```

### 2. Run preprocessing
```bash
python preprocessing.py
```

### 3. Run EDA and generate visualizations
```bash
python eda.py
```

### 4. Train ML models
```bash
python train.py
```

### 5. Run FastAPI backend (in one terminal)
```bash
uvicorn backend.main:app --reload
```
Swagger UI available at: http://127.0.0.1:8000/docs

### 6. Run Streamlit frontend (in another terminal)
```bash
streamlit run frontend/app.py
```
App available at: http://localhost:8501

## API Endpoints
- `GET /` - API info
- `GET /api/health` - Health check
- `GET /api/analytics` - Dataset analytics and statistics
- `GET /api/models` - Model metrics and comparison
- `POST /api/predict` - Make prediction

## ML Models & Evaluation
Models trained using only categorical features to avoid data leakage.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | (actual from training) | ... | ... | ... | ... |
| Random Forest | (actual from training) | ... | ... | ... | ... |

**Best Model:** Selected based on F1-score.

## Key Results
- Total students: 1000
- High Performer threshold: average_score >= 70
- Performance distribution and group-wise averages computed from actual data
- All visualizations saved in `reports/figures/`

## Limitations
- Binary classification with fixed threshold (70)
- Limited features (only demographic/categorical)
- Small dataset (1000 records)
- Does not guarantee actual academic performance

## Future Scope
- Try other ML algorithms
- Include additional features
- Hyperparameter tuning
- Explainable AI (SHAP values)

## Screenshots
1. Dashboard
2. Data & Preprocessing
3. EDA & Visualizations
4. Machine Learning
5. Prediction
6. FastAPI Swagger UI
