from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

def split_data(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.2,
        random_state=42,
        stratify=y      # used because this is a VERY imbalanced dataset
    )

    return X_train, X_test, y_train, y_test

def get_feature_types(X):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    return categorical_cols, numerical_cols

def build_preprocessor(categorical_cols, numerical_cols):
    preprocessor = ColumnTransformer(
        transformers = [
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )

    return preprocessor

def prepare_data(df):
    X_train, X_test, y_train, y_test = split_data(df)

    categorical_cols, numerical_cols = get_feature_types(X_train)

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def train_XGB(X_train, y_train):
    model = XGBClassifier(
        n_estimators = 200,
        max_depth = 5,
        learning_rate = 0.1,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state=42,
        eval_metric = 'logloss'
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print ("Accuracy: ", accuracy_score(y_test, y_pred))
    print ("\nClassification Report\n", classification_report(y_test, y_pred))
