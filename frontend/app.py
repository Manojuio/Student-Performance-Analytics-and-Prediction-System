import streamlit as st
import pandas as pd
import requests
import os
import json
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Student Performance Analytics & Prediction System", layout="wide", initial_sidebar_state="expanded")

# Improved white theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1e3a8a !important;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: #1f2937 !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    h3 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    .stMetric {
        background: linear-gradient(145deg, #f8fafc 0%, #ffffff 100%) !important;
        padding: 1.5rem 1rem !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.06) !important;
        text-align: center;
    }
    
    .stMetric label {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 0.025em !important;
    }
    
    [data-testid="metric-value"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e3a8a !important;
    }
    
    .stDataFrame {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }
    
    .stDataFrame [data-testid="stDataFrameContainer"] {
        border-radius: 8px !important;
    }
    
    .element-container {
        margin-bottom: 1rem;
    }
    
    .stAlert {
        border-radius: 8px !important;
        padding: 1rem 1.5rem !important;
    }
    
    .stSuccess {
        background-color: #f0fdf4 !important;
        border: 1px solid #86efac !important;
        color: #166534 !important;
    }
    
    .stInfo {
        background-color: #eff6ff !important;
        border: 1px solid #93c5fd !important;
        color: #1e40af !important;
    }
    
    .stMarkdown {
        line-height: 1.6;
    }
    
    .css-1d391kg {
        background-color: #ffffff !important;
    }
    
    .css-1544g2n {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Student Performance Analytics and Prediction System")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_data.csv")
    df = pd.read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame()
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except:
        metrics = {}
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Select Page", [
            "Dashboard", 
            "Data & Preprocessing", 
            "EDA & Visualizations", 
            "Machine Learning", 
            "Prediction", 
            "About"
        ], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("## Project Info")
        st.info("**Dataset:** 1000 Student Records\n\n**Models:** Logistic Regression & Random Forest\n\n**Threshold:** Avg Score ≥ 70")
    
    if page == "Dashboard":
        st.header("Dashboard - Overview & Key Metrics")
        
        # KPI Cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Total Students", f"{len(df):,}")
        with col2:
            st.metric("Avg Math", f"{df['math_score'].mean():.2f}" if not df.empty else "N/A")
        with col3:
            st.metric("Avg Reading", f"{df['reading_score'].mean():.2f}" if not df.empty else "N/A")
        with col4:
            st.metric("Avg Writing", f"{df['writing_score'].mean():.2f}" if not df.empty else "N/A")
        with col5:
            st.metric("Overall Avg", f"{df['average_score'].mean():.2f}" if not df.empty else "N/A")
        with col6:
            st.metric("High Performers", f"{(df['performance']=='High Performer').mean()*100:.1f}%" if not df.empty else "N/A")
        
        st.divider()
        
        # Performance Distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Performance Distribution")
            perf_data = df['performance'].value_counts().reset_index()
            perf_data.columns = ['Performance', 'Count']
            perf_data['Percentage'] = (perf_data['Count'] / perf_data['Count'].sum() * 100).round(2)
            fig = px.pie(perf_data, values='Count', names='Performance', 
                        title='High Performer vs Normal Performer',
                        color='Performance',
                        color_discrete_map={'High Performer': '#1e40af', 'Normal Performer': '#94a3b8'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(perf_data, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Average Score Distribution")
            fig = px.histogram(df, x='average_score', nbins=40, 
                             title='Distribution of Average Scores',
                             color_discrete_sequence=['#1e40af'])
            fig.add_vline(x=df['average_score'].mean(), line_dash="dash", 
                         annotation_text=f"Mean: {df['average_score'].mean():.2f}", 
                         annotation_position="top")
            fig.update_layout(xaxis_title="Average Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("Key Analytical Findings")
        findings_col1, findings_col2 = st.columns(2)
        with findings_col1:
            st.markdown("**Academic Performance Insights:**")
            st.write("• Students who completed test preparation show significantly higher average scores")
            st.write("• Standard lunch students outperform those with free/reduced lunch")
            st.write("• Parental education level positively influences student performance")
            st.write("• Reading and writing scores show strong correlation")
        with findings_col2:
            st.markdown("**Dataset Summary:**")
            st.write(f"• Total records analyzed: {len(df)}")
            st.write(f"• High Performers (≥70 avg): {(df['performance']=='High Performer').sum()} ({(df['performance']=='High Performer').mean()*100:.2f}%)")
            st.write(f"• Normal Performers (<70 avg): {(df['performance']=='Normal Performer').sum()} ({(df['performance']=='Normal Performer').mean()*100:.2f}%)")
            st.write("• No missing values or duplicates in processed data")
    
    elif page == "Data & Preprocessing":
        st.header("Data & Preprocessing")
        
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        col4.metric("Duplicate Records", int(df.duplicated().sum()))
        
        st.divider()
        st.subheader("Dataset Columns")
        cols_df = pd.DataFrame({'Column Name': df.columns.tolist()})
        st.dataframe(cols_df, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("Sample Records (First 10)")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.divider()
        st.subheader("Descriptive Statistics")
        desc = df.describe().round(2)
        st.dataframe(desc, use_container_width=True)
        
        st.divider()
        st.subheader("Data Preprocessing Workflow")
        steps_df = pd.DataFrame({
            'Step': ['1. Load Dataset', '2. Data Inspection', '3. Clean Column Names', 
                    '4. Handle Duplicates/Missing', '5. Feature Engineering', '6. Create Target Variable', '7. Save Processed Data'],
            'Description': [
                'Loaded via KaggleHub (spscientist/students-performance-in-exams)',
                'Checked shape, dtypes, missing values, duplicates',
                'Standardized to lowercase with underscores',
                'Removed duplicates if present, handled missing values',
                'Created average_score = (math+reading+writing)/3',
                'performance: High Performer if average_score >= 70',
                'Saved to data/processed/processed_data.csv'
            ]
        })
        st.dataframe(steps_df, use_container_width=True, hide_index=True)
    
    elif page == "EDA & Visualizations":
        st.header("Exploratory Data Analysis & Visualizations")
        
        st.subheader("Individual Score Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='math_score', nbins=30, marginal='box',
                             title='Math Score Distribution', color_discrete_sequence=['#2563eb'])
            fig.update_layout(xaxis_title="Math Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(df, x='reading_score', nbins=30, marginal='box',
                             title='Reading Score Distribution', color_discrete_sequence=['#3b82f6'])
            fig.update_layout(xaxis_title="Reading Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x='writing_score', nbins=30, marginal='box',
                             title='Writing Score Distribution', color_discrete_sequence=['#60a5fa'])
            fig.update_layout(xaxis_title="Writing Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(df, x='average_score', nbins=30, marginal='box',
                             title='Average Score Distribution', color_discrete_sequence=['#1e40af'])
            fig.update_layout(xaxis_title="Average Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("Performance by Categories")
        categories = [
            ('gender', 'Gender'),
            ('race_ethnicity', 'Race/Ethnicity'),
            ('parental_education', 'Parental Education Level'),
            ('lunch', 'Lunch Type'),
            ('test_preparation', 'Test Preparation Course')
        ]
        for cat_col, cat_name in categories:
            st.subheader(f"{cat_name} vs Average Score")
            grp = df.groupby(cat_col)['average_score'].mean().reset_index()
            grp['average_score'] = grp['average_score'].round(2)
            fig = px.bar(grp, x=cat_col, y='average_score', 
                        title=f'{cat_name} Comparison',
                        text='average_score',
                        color_discrete_sequence=['#1e3a8a'])
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title=cat_name, yaxis_title="Average Score")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(grp, use_container_width=True, hide_index=True)
            st.divider()
        
        st.subheader("Correlation Heatmap")
        corr_cols = ['math_score', 'reading_score', 'writing_score', 'average_score']
        corr = df[corr_cols].corr().round(3)
        fig = px.imshow(corr, text_auto=True, aspect='auto',
                       title='Score Correlation Matrix',
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Machine Learning":
        st.header("Machine Learning - Models, Evaluation & Comparison")
        
        if metrics:
            res = metrics['results']
            st.subheader("Model Performance Metrics")
            comp_df = pd.DataFrame(res).T
            comp_df = comp_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].round(4)
            comp_df = comp_df.reset_index()
            comp_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.divider()
            # Best model highlight
            best = metrics['best_model']
            best_f1 = metrics['best_f1_score']
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Model", best)
            col2.metric("Best F1-Score", f"{best_f1:.4f}")
            col3.metric("Selection Criteria", "F1-Score")
            
            st.divider()
            st.subheader("Visual Model Comparison")
            fig = go.Figure()
            metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            for model_name in res.keys():
                fig.add_trace(go.Bar(name=model_name, 
                                    x=metrics_list, 
                                    y=[res[model_name][m] for m in metrics_list]))
            fig.update_layout(barmode='group', title='Model Comparison Across Metrics',
                            xaxis_title='Metric', yaxis_title='Score')
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("Model Configuration & Methodology")
            config_df = pd.DataFrame({
                'Aspect': ['ML Algorithms', 'Train-Test Split', 'Random State', 'Splitting Strategy',
                          'Preprocessing', 'Features Used', 'Target', 'Leakage Prevention'],
                'Details': [
                    'Logistic Regression, Random Forest Classifier',
                    '80% Training / 20% Testing',
                    '42 (for reproducibility)',
                    'Stratified Split',
                    'OneHotEncoder + ColumnTransformer + Pipeline',
                    'gender, race_ethnicity, parental_education, lunch, test_preparation',
                    'performance (High Performer/Normal Performer)',
                    'No exam scores used as input features'
                ]
            })
            st.dataframe(config_df, use_container_width=True, hide_index=True)
            
            st.divider()
            st.subheader("Evaluation Metrics Definitions")
            metrics_def = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Definition': [
                    'Ratio of correctly predicted observations to total observations',
                    'TP / (TP + FP) - Correctly identified positives among predicted positives',
                    'TP / (TP + FN) - Correctly identified positives among actual positives',
                    '2*(Precision*Recall)/(Precision+Recall) - Harmonic mean balancing precision and recall',
                    'Area Under ROC Curve - Measures ability to distinguish between classes'
                ]
            })
            st.dataframe(metrics_def, use_container_width=True, hide_index=True)
        else:
            st.error("Model metrics not found. Please run train.py to train the models.")
    
    elif page == "Prediction":
        st.header("Student Performance Prediction")
        st.markdown("**Enter student details below to predict performance classification.**")
        
        with st.form("prediction_form", clear_on_submit=False):
            st.subheader("Student Information")
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", options=["female", "male"], help="Select student's gender")
                race_ethnicity = st.selectbox("Race/Ethnicity", 
                                             options=["group A", "group B", "group C", "group D", "group E"],
                                             help="Select student's ethnic group")
                parental_education = st.selectbox("Parental Education Level",
                                               options=["some high school", "high school", "some college", 
                                                       "associate's degree", "bachelor's degree", "master's degree"],
                                               help="Highest education level of parents")
            with col2:
                lunch = st.selectbox("Lunch Type", options=["standard", "free/reduced"],
                                   help="Type of lunch received")
                test_preparation = st.selectbox("Test Preparation Course", 
                                              options=["none", "completed"],
                                              help="Whether test preparation course was completed")
            
            st.divider()
            predict_btn = st.form_submit_button("🔮 Predict Performance", use_container_width=True, type="primary")
        
        if predict_btn:
            payload = {
                "gender": gender,
                "race_ethnicity": race_ethnicity,
                "parental_education": parental_education,
                "lunch": lunch,
                "test_preparation": test_preparation
            }
            try:
                with st.spinner("Making prediction..."):
                    response = requests.post(f"{API_URL}/api/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    st.divider()
                    st.subheader("Prediction Results")
                    
                    pred = result['prediction']
                    prob = result['probability']
                    model_used = result['model']
                    
                    # Display result with styling
                    if pred == "High Performer":
                        st.success(f"**Prediction: HIGH PERFORMER**")
                    else:
                        st.info(f"**Prediction: NORMAL PERFORMER**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction", pred)
                    with col2:
                        st.metric("Probability", f"{prob:.4f}")
                    with col3:
                        st.metric("Model Used", model_used)
                    
                    # Probability bar
                    st.subheader("Confidence Level")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probability of High Performer"},
                        gauge={'axis': {'range': [0, 1]},
                               'bar': {'color': "#1e40af"},
                               'steps': [
                                   {'range': [0, 0.5], 'color': "#e5e7eb"},
                                   {'range': [0.5, 1], 'color': "#bfdbfe"}],
                               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prob}}))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.caption("**Disclaimer:** This prediction is based on statistical patterns from the training dataset. It does not guarantee actual academic performance. This is for educational/analytical purposes only.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to FastAPI server at {API_URL}. Please ensure the backend is running.")
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
    
    elif page == "About":
        st.header("About This Project")
        
        st.subheader("Project Title")
        st.write("**Student Performance Analytics and Prediction System**")
        
        st.divider()
        st.subheader("Problem Statement")
        st.markdown("Develop a Data Analytics and Visualization-based web application using a dataset of at least 1,000 records. The project must include data preprocessing and wrangling, exploratory data analysis, meaningful visualizations, implementation and comparison of at least two Machine Learning models, and a user-friendly web interface for presenting the analytical and prediction results.")
        
        st.divider()
        st.subheader("Objectives")
        obj_df = pd.DataFrame({
            'Objective': [
                'Real-world Data Analysis',
                'Data Preprocessing',
                'Exploratory Data Analysis',
                'Data Visualization',
                'Machine Learning',
                'Model Comparison',
                'Web Application Development'
            ],
            'Description': [
                'Analyze student performance using real Kaggle dataset (1000+ records)',
                'Clean, transform and prepare data for analysis',
                'Discover patterns and insights through statistical analysis',
                'Create meaningful interactive visualizations',
                'Implement Logistic Regression and Random Forest models',
                'Compare models using standard evaluation metrics',
                'Build user-friendly FastAPI + Streamlit interface'
            ]
        })
        st.dataframe(obj_df, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset Name:** Students Performance in Exams")
            st.markdown("**Source:** [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)")
            st.markdown("**Records:** ~1000")
            st.markdown("**Download Method:** KaggleHub")
        with col2:
            st.markdown("**Features:** gender, race/ethnicity, parental education, lunch, test preparation, math/reading/writing scores")
            st.markdown("**Target:** performance (High Performer if avg score ≥ 70)")
            st.markdown("**Data Type:** Real-world educational data")
        
        st.divider()
        st.subheader("Technology Stack")
        tech_df = pd.DataFrame({
            'Category': ['Data Processing', 'Visualization', 'Machine Learning', 'Backend', 'Frontend', 'Utilities'],
            'Technologies': [
                'Pandas, NumPy',
                'Matplotlib, Seaborn, Plotly',
                'Scikit-learn, Joblib',
                'FastAPI, Uvicorn, Pydantic',
                'Streamlit, Requests',
                'KaggleHub'
            ]
        })
        st.dataframe(tech_df, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("Project Architecture")
        st.code("""Kaggle Dataset → KaggleHub → CSV → Pandas → Preprocessing + Wrangling
                ↓
                EDA + Visualization → ML Models (LR & RF) → Model Comparison
                ↓
                FastAPI (Backend/API) → Streamlit (Frontend/UI)""", language="text")
        
        st.divider()
        st.subheader("Limitations & Future Scope")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Limitations:**")
            st.write("• Binary classification with fixed threshold (70)")
            st.write("• Limited demographic features")
            st.write("• Dataset size relatively small (1000 records)")
            st.write("• Predictions based on correlations, not causation")
        with col2:
            st.markdown("**Future Scope:**")
            st.write("• Hyperparameter tuning for better performance")
            st.write("• Explore additional ML algorithms")
            st.write("• Feature importance interpretation (SHAP)")
            st.write("• Expand dataset for better generalization")

if __name__ == "__main__":
    main()
