import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def train_loan_model(data_path, target_col="loan_status"):
    # Load data
    df = pd.read_csv(data_path)

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Remove leading/trailing spaces from string values
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Ensure target column name is correct
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Found columns: {df.columns.tolist()}")

    # Convert target to numeric if it's text
    if df[target_col].dtype == "object":
        df[target_col] = df[target_col].map({"Approved": 1, "Rejected": 0, "Y": 1, "N": 0, "Yes": 1, "No": 0})

    # Separate features and target

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Fill missing numeric values with median, categorical with mode
    for col in X.columns:
        if X[col].dtype == "object":
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, X_test, y_test, y_pred, report, cm
