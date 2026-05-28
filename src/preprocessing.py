import pandas as pd


def preprocess_data(df):

    df = df.copy()

    df["person_emp_length"] = (
        df["person_emp_length"]
        .fillna(
            df["person_emp_length"].median()
        )
    )

    df["loan_int_rate"] = (
        df["loan_int_rate"]
        .fillna(
            df["loan_int_rate"].median()
        )
    )

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

    return df