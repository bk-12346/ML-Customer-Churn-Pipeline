import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize the categories
    df = df.replace({
        "No internet service": "No",
        "No phone service": "No"
    })

    # feature 1: Tenure Group
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins = [-1, 12, 24, 48, 72],
        labels = ['new', 'short_term', 'mid-term', 'long-term']
    )

    # feature 2: Number of Services
    services = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    df['num_services'] = df[services].apply(
        lambda x: (x=='Yes').sum(), axis = 1    # axis=1 is because we want to move horizontally over a complete row before going to next customer
    )

    # feature 3: Support Indicator
    df['has_support'] = (
        (df['OnlineSecurity']=='Yes') |
        (df['TechSupport']=='Yes')
    ).astype(int)

    # feature 4: Engagement
    df['is_engaged'] = (
        (df['StreamingTV']=='Yes') |
        (df['StreamingMovies']=='Yes')
    ).astype(int)

    # feature 5: Cost Efficiency
    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)   # +1 is for Laplace smoothig - to avoid division by 0 for new customers

    # feature 6: Contract Risk
    df['is_monthly_contract'] = (
        df['Contract'] == 'Month-to-month'
    ).astype(int)

    # feature 7: Payment Risk
    df['uses_electronic_check'] = (
        df['PaymentMethod'] == 'Electronic check'
    ).astype(int)

    # feature 8: Family Status
    df['has_family'] = (
        (df['Partner'] == 'Yes') |
        (df['Dependents'] == 'Yes')
    ).astype(int)

    return df