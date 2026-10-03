import pandas as pd
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "StudentsPerformance.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")

def main():
    # 1. Load dataset
    df = pd.read_csv(RAW_PATH)
    
    print("=== DATA PREPROCESSING ===")
    # 2. Display shape
    print(f"Shape: {df.shape}")
    # 3. Display columns
    print(f"Columns: {list(df.columns)}")
    # 4. Check missing values
    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing}")
    # 5. Check duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    # 6. Check data types
    print(f"\nData types:\n{df.dtypes}")
    # 7. Check categorical values
    print("\nCategorical values:")
    for col in df.select_dtypes(include='object').columns:
        print(f"  {col}: {df[col].unique()}")
    
    # 8. Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={
        'race/ethnicity': 'race_ethnicity',
        'parental_level_of_education': 'parental_education',
        'test_preparation_course': 'test_preparation'
    })
    print(f"\nCleaned columns: {list(df.columns)}")
    
    # 9. Perform necessary preprocessing
    # Remove duplicates if any
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    
    # No missing values in this dataset
    df = df.dropna() if df.isnull().sum().sum() > 0 else df
    
    # 10. Feature engineering
    # Create average_score
    df['average_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3.0
    # Create target performance
    df['performance'] = np.where(df['average_score'] >= 70, 'High Performer', 'Normal Performer')
    
    print(f"\n--- After preprocessing ---")
    print(f"Final records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nPerformance distribution:")
    print(df['performance'].value_counts())
    print(df['performance'].value_counts(normalize=True) * 100)
    
    # Save processed dataset
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved processed data to: {PROCESSED_PATH}")
    
    # Generate statistics
    print("\n=== STATISTICS ===")
    print(f"Total students: {len(df)}")
    print(f"Average math score: {df['math_score'].mean():.2f}")
    print(f"Average reading score: {df['reading_score'].mean():.2f}")
    print(f"Average writing score: {df['writing_score'].mean():.2f}")
    print(f"Overall average score: {df['average_score'].mean():.2f}")
    print(f"High performer count: {(df['performance']=='High Performer').sum()}")
    print(f"Normal performer count: {(df['performance']=='Normal Performer').sum()}")
    hp_pct = (df['performance']=='High Performer').mean()*100
    print(f"High performer percentage: {hp_pct:.2f}%")

if __name__ == "__main__":
    main()
