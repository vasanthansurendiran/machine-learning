import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_and_prepare_data():
    print("--- Starting Telco Data Preprocessing ---")
    # Load larger, professional dataset
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    df = pd.read_csv(url)

    # ==========================================
    # RUBRIC SECTION 1: Data Pre-processing
    # ==========================================
    # Drop completely useless ID column
    df = df.drop('customerID', axis=1)
    
    # 'TotalCharges' contains hidden blank spaces for brand new customers. 
    # Force it to numeric and fill those blanks with 0 to prevent model failure.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # ==========================================
    # RUBRIC SECTION 2: Feature Engineering
    # ==========================================
    # Combine individual streaming services into one binary 'Is_Streamer' feature
    df['Is_Streamer'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)

    # FIXED: Encode ALL text categories into numbers, bypassing the modern Pandas string bug
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object', 'string', 'category']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # Export the cleaned data for the main model pipeline
    clean_path = 'clean_churn_data.csv'
    df.to_csv(clean_path, index=False)
    print(f"--- Preprocessing Complete. Clean data saved to {clean_path} ---")

if __name__ == "__main__":
    clean_and_prepare_data()