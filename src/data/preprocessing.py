import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # 1. Fix TotalCharges (string → numeric)
    # -----------------------------
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # -----------------------------
    # 2. Drop missing values
    # -----------------------------
    df = df.dropna()

    # -----------------------------
    # 3. Remove duplicates
    # -----------------------------
    df = df.drop_duplicates()

    # -----------------------------
    # 4. Drop irrelevant columns
    # -----------------------------
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # -----------------------------
    # 5. Convert target variable
    # -----------------------------
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # -----------------------------
    # 6. Clean categorical inconsistencies
    # -----------------------------
    df = df.replace({
        'No internet service': 'No',
        'No phone service': 'No'
    })

    return df