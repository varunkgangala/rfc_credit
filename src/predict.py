import pandas as pd
import joblib

model = joblib.load(
    "models/rf_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)


def predict_risk(data):

    df = pd.DataFrame([data])

    categorical_columns = [

        "person_home_ownership",

        "loan_intent",

        "loan_grade",

        "cb_person_default_on_file"
    ]

    df = pd.get_dummies(

        df,

        columns=categorical_columns,

        drop_first=True
    )

    df = df.reindex(

        columns=feature_columns,

        fill_value=0
    )

    prediction = model.predict(df)[0]

    probability = model.predict_proba(
        df
    )[0][1]

    return prediction, probability