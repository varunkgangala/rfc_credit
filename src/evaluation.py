from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score
)


def evaluate_model(
    y_test,
    predictions,
    probabilities
):

    return {

        "Accuracy":
        accuracy_score(
            y_test,
            predictions
        ),

        "Precision":
        precision_score(
            y_test,
            predictions
        ),

        "Recall":
        recall_score(
            y_test,
            predictions
        ),

        "F1":
        f1_score(
            y_test,
            predictions
        ),

        "ROC_AUC":
        roc_auc_score(
            y_test,
            probabilities
        )
    }