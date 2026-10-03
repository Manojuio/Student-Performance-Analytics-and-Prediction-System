import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    df = pd.read_csv(PROC_PATH)
    feature_cols = ['gender', 'race_ethnicity', 'parental_education', 'lunch', 'test_preparation']
    X = df[feature_cols]
    y = df['performance']
    preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_test_bin = (y_test == 'High Performer').astype(int)
    results = {}
    # LR
    lr = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 0]  # P(High Performer)
    results['Logistic Regression'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_lr)),
        'precision': float(precision_score(y_test, y_pred_lr, pos_label='High Performer')),
        'recall': float(recall_score(y_test, y_pred_lr, pos_label='High Performer')),
        'f1_score': float(f1_score(y_test, y_pred_lr, pos_label='High Performer')),
        'roc_auc': float(roc_auc_score(y_test_bin, y_proba_lr))
    }
    # RF
    rf = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 0]  # P(High Performer)
    results['Random Forest'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'precision': float(precision_score(y_test, y_pred_rf, pos_label='High Performer')),
        'recall': float(recall_score(y_test, y_pred_rf, pos_label='High Performer')),
        'f1_score': float(f1_score(y_test, y_pred_rf, pos_label='High Performer')),
        'roc_auc': float(roc_auc_score(y_test_bin, y_proba_rf))
    }
    joblib.dump(lr, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    best_model_name = max(results, key=lambda k: results[k]['f1_score'])
    metrics = {'results': results, 'best_model': best_model_name, 'best_f1_score': results[best_model_name]['f1_score'],
               'feature_columns': feature_cols, 'classes': ['High Performer', 'Normal Performer']}
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    # plots
    cm = confusion_matrix(y_test, y_pred_lr, labels=['High Performer','Normal Performer'])
    plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['High Performer','Normal Performer'], yticklabels=['High Performer','Normal Performer']); plt.title('Confusion Matrix - Logistic Regression'); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'confusion_matrix_lr.png'), dpi=300); plt.close()
    cm = confusion_matrix(y_test, y_pred_rf, labels=['High Performer','Normal Performer'])
    plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['High Performer','Normal Performer'], yticklabels=['High Performer','Normal Performer']); plt.title('Confusion Matrix - Random Forest'); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'confusion_matrix_rf.png'), dpi=300); plt.close()
    plt.figure(figsize=(7,6)); fpr,tpr,_=roc_curve(y_test_bin,y_proba_lr); plt.plot(fpr,tpr,label=f'Logistic Regression (AUC={results["Logistic Regression"]["roc_auc"]:.3f})'); fpr,tpr,_=roc_curve(y_test_bin,y_proba_rf); plt.plot(fpr,tpr,label=f'Random Forest (AUC={results["Random Forest"]["roc_auc"]:.3f})'); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title('ROC Curve'); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'roc_curve.png'), dpi=300); plt.close()
    try:
        cat_feats = preprocessor.named_transformers_['cat'].get_feature_names_out(feature_cols)
    except:
        cat_feats = preprocessor.named_transformers_['cat'].get_feature_names_out()
    imp = pd.DataFrame({'feature': cat_feats, 'importance': rf.named_steps['classifier'].feature_importances_}).sort_values('importance', ascending=False).head(15)
    plt.figure(figsize=(9,6)); sns.barplot(data=imp, x='importance', y='feature'); plt.title('Feature Importance (Random Forest)'); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'feature_importance.png'), dpi=300); plt.close()
    pd.DataFrame(results).T[['accuracy','precision','recall','f1_score','roc_auc']].plot(kind='bar', figsize=(10,6)); plt.title('Model Comparison'); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'model_comparison.png'), dpi=300); plt.close()
    print(json.dumps(metrics, indent=2))

if __name__=='__main__': main()
