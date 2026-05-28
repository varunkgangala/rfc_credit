import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def class_distribution(df):

    fig, ax = plt.subplots()

    df["loan_status"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Loan Status Distribution"
    )

    return fig


def income_distribution(df):

    fig, ax = plt.subplots()

    ax.hist(
        df["person_income"],
        bins=30
    )

    ax.set_title(
        "Income Distribution"
    )

    return fig


def age_distribution(df):

    fig, ax = plt.subplots()

    ax.hist(
        df["person_age"],
        bins=30
    )

    ax.set_title(
        "Age Distribution"
    )

    return fig


def loan_distribution(df):

    fig, ax = plt.subplots()

    ax.hist(
        df["loan_amnt"],
        bins=30
    )

    ax.set_title(
        "Loan Amount Distribution"
    )

    return fig


def correlation_heatmap(df):

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    corr = (
        df.select_dtypes(
            include="number"
        )
        .corr()
    )

    sns.heatmap(
        corr,
        cmap="coolwarm",
        ax=ax
    )

    return fig


def feature_importance_plot(
    model,
    columns
):

    importance = pd.Series(
        model.feature_importances_,
        index=columns
    )

    importance = (
        importance
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    importance.plot.bar(
        ax=ax
    )

    ax.set_title(
        "Top 15 Important Features"
    )

    return fig