import os
import subprocess
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

if not os.path.exists("models/rf_model.pkl"):

    os.makedirs(
        "models",
        exist_ok=True
    )

    subprocess.run(
        ["python", "-m", "src.train_model"]
    )

from src.preprocessing import (
    preprocess_data
)

from src.predict import (
    predict_risk
)

from src.visualization import (

    class_distribution,

    income_distribution,

    age_distribution,

    loan_distribution,

    correlation_heatmap,

    feature_importance_plot
)

st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide"
)

st.title(
    "Credit Risk Classification using Random Forest"
)

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/credit_risk_dataset.csv"
    )

df = load_data()

model = joblib.load(
    "models/rf_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

best_params = joblib.load(
    "models/best_params.pkl"
)

page = st.sidebar.selectbox(

    "Navigation",

    [

        "Dataset Overview",

        "Missing Values",

        "EDA",

        "Hyperparameter Tuning",

        "Model Performance",

        "Feature Importance",

        "Prediction"
    ]
)

if page == "Dataset Overview":

    st.header(
        "Dataset Overview"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10)
    )

    info_df = pd.DataFrame({

        "Column":
        df.columns,

        "Datatype":
        df.dtypes.astype(str)
    })

    st.subheader(
        "Column Information"
    )

    st.dataframe(
        info_df
    )

    st.subheader(
        "Statistical Summary"
    )

    st.dataframe(
        df.describe()
    )

elif page == "Missing Values":

    st.header(
        "Missing Values Analysis"
    )

    missing = (
        df.isnull()
        .sum()
    )

    missing_df = pd.DataFrame({

        "Column":
        missing.index,

        "Missing Values":
        missing.values
    })

    st.dataframe(
        missing_df
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    missing.sort_values(
        ascending=False
    ).plot.bar(ax=ax)

    ax.set_title(
        "Missing Values"
    )

    st.pyplot(fig)

elif page == "EDA":

    st.header(
        "Exploratory Data Analysis"
    )

    st.subheader(
        "Loan Status Distribution"
    )

    st.pyplot(
        class_distribution(df)
    )

    st.subheader(
        "Income Distribution"
    )

    st.pyplot(
        income_distribution(df)
    )

    st.subheader(
        "Age Distribution"
    )

    st.pyplot(
        age_distribution(df)
    )

    st.subheader(
        "Loan Amount Distribution"
    )

    st.pyplot(
        loan_distribution(df)
    )

    st.subheader(
        "Correlation Heatmap"
    )

    st.pyplot(
        correlation_heatmap(df)
    )

elif page == "Hyperparameter Tuning":

    st.header(
        "Best Hyperparameters"
    )

    st.json(
        best_params
    )

elif page == "Model Performance":

    st.header(
        "Model Performance"
    )

    processed_df = preprocess_data(df)

    X = processed_df.drop(
        "loan_status",
        axis=1
    )

    y = processed_df[
        "loan_status"
    ]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:,1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc = roc_auc_score(
        y_test,
        probabilities
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    c2.metric(
        "Precision",
        f"{precision:.4f}"
    )

    c3.metric(
        "Recall",
        f"{recall:.4f}"
    )

    c4.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    c5.metric(
        "ROC AUC",
        f"{roc:.4f}"
    )

    st.subheader(
        "Confusion Matrix"
    )

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(

        y_test,

        predictions,

        ax=ax
    )

    st.pyplot(fig)

    st.subheader(
        "ROC Curve"
    )

    fig, ax = plt.subplots()

    RocCurveDisplay.from_predictions(

        y_test,

        probabilities,

        ax=ax
    )

    st.pyplot(fig)

    st.subheader(
        "Precision Recall Curve"
    )

    fig, ax = plt.subplots()

    PrecisionRecallDisplay.from_predictions(

        y_test,

        probabilities,

        ax=ax
    )

    st.pyplot(fig)

elif page == "Feature Importance":

    st.header(
        "Feature Importance"
    )

    st.pyplot(

        feature_importance_plot(

            model,

            feature_columns
        )
    )

elif page == "Prediction":

    st.header(
        "Predict Credit Risk"
    )

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    person_income = st.number_input(
        "Income",
        value=50000
    )

    person_home_ownership = st.selectbox(

        "Home Ownership",

        [
            "RENT",
            "OWN",
            "MORTGAGE",
            "OTHER"
        ]
    )

    person_emp_length = st.number_input(
        "Employment Length",
        value=5.0
    )

    loan_intent = st.selectbox(

        "Loan Intent",

        [

            "EDUCATION",

            "MEDICAL",

            "VENTURE",

            "PERSONAL",

            "HOMEIMPROVEMENT",

            "DEBTCONSOLIDATION"
        ]
    )

    loan_grade = st.selectbox(

        "Loan Grade",

        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ]
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        value=10000
    )

    loan_int_rate = st.number_input(
        "Interest Rate",
        value=10.0
    )

    loan_percent_income = st.number_input(
        "Loan Percent Income",
        value=0.20
    )

    cb_person_default_on_file = st.selectbox(

        "Previous Default",

        [
            "Y",
            "N"
        ]
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length",
        value=5
    )

    if st.button(
        "Predict Risk"
    ):

        data = {

            "person_age":
            person_age,

            "person_income":
            person_income,

            "person_home_ownership":
            person_home_ownership,

            "person_emp_length":
            person_emp_length,

            "loan_intent":
            loan_intent,

            "loan_grade":
            loan_grade,

            "loan_amnt":
            loan_amnt,

            "loan_int_rate":
            loan_int_rate,

            "loan_percent_income":
            loan_percent_income,

            "cb_person_default_on_file":
            cb_person_default_on_file,

            "cb_person_cred_hist_length":
            cb_person_cred_hist_length
        }

        prediction, probability = predict_risk(
            data
        )

        if prediction == 1:

            st.error(
                f"High Credit Risk\n\nProbability: {probability:.2%}"
            )

        else:

            st.success(
                f"Low Credit Risk\n\nConfidence: {(1 - probability):.2%}"
            )

