import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    df = pd.read_csv(PROC_PATH)
    print("=== EDA ===")
    print(f"Dataset overview: shape {df.shape}")
    print("\nDescriptive statistics:")
    print(df.describe())
    
    # Score distributions
    for col in ['math_score', 'reading_score', 'writing_score', 'average_score']:
        plt.figure(figsize=(8,5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{col.replace('_',' ').title()} Distribution")
        plt.xlabel(col.replace('_',' ').title())
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{col}_distribution.png"), dpi=300)
        plt.close()
    
    # Performance distribution
    plt.figure(figsize=(6,5))
    sns.countplot(data=df, x='performance')
    plt.title('High Performer vs Normal Performer')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'performance_distribution.png'), dpi=300)
    plt.close()
    
    # Category comparisons
    for cat in ['gender', 'race_ethnicity', 'parental_education', 'lunch', 'test_preparation']:
        plt.figure(figsize=(9,5))
        sns.barplot(data=df, x=cat, y='average_score', ci=None)
        plt.title(f'{cat.replace("_"," ").title()} vs Average Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'{cat}_vs_avg.png'), dpi=300)
        plt.close()
    
    # Correlation heatmap
    corr_cols = ['math_score', 'reading_score', 'writing_score', 'average_score']
    plt.figure(figsize=(7,5))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # Save groupwise averages for report
    for cat in ['gender', 'race_ethnicity', 'parental_education', 'lunch', 'test_preparation']:
        grp = df.groupby(cat)['average_score'].mean().reset_index()
        grp.to_csv(os.path.join(BASE_DIR, 'reports', f'{cat}_avg.csv'), index=False)
    
    df.describe().to_csv(os.path.join(BASE_DIR, 'reports', 'descriptive_stats.csv'))
    print(f"Visualizations saved to {FIG_DIR}")
    print("EDA complete.")

if __name__ == "__main__":
    main()
