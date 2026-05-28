import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from src.preprocessing import (
    preprocess_data
)

df = pd.read_csv(
    "data/credit_risk_dataset.csv"
)

df = preprocess_data(df)

X = df.drop(
    "loan_status",
    axis=1
)

y = df["loan_status"]

joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

param_grid = {

    "n_estimators":[
        50,
        100,
        150
    ],

    "max_depth":[
        5,
        10,
        15
    ],

    "min_samples_split":[
        10,
        20,
        30
    ],

    "min_samples_leaf":[
        5,
        10,
        20
    ],

    "max_features":[
        "sqrt"
    ]
}

rf = RandomForestClassifier(

    random_state=42,

    n_jobs=-1
)

search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_grid,

    n_iter=15,

    cv=5,

    scoring="f1",

    random_state=42,

    n_jobs=-1
)

search.fit(
    X_train,
    y_train
)

best_params = search.best_params_

best_model = RandomForestClassifier(

    **best_params,

    random_state=42,

    n_jobs=-1,

    max_leaf_nodes=100
)

best_model.fit(
    X_train,
    y_train
)

joblib.dump(
    best_model,
    "models/rf_model.pkl",
    compress=9
)

joblib.dump(
    best_params,
    "models/best_params.pkl"
)

print("\nBest Parameters\n")

print(best_params)