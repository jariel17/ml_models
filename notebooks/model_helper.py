import mlflow
import mlflow.sklearn as mlf_skl
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_and_log_model(model_name, model_pipeline, X_train, X_test, y_train, y_test, label_encoder):
    with mlflow.start_run(run_name=model_name):
        # Fit model
        model_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log parameters
        mlflow.log_params(model_pipeline.get_params())
        
        # Confusion matrix visualization
        cm = confusion_matrix(y_test, y_pred)
        cm_labels = label_encoder.classes_
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{model_name} - Confusion Matrix")
        fig.tight_layout()
        
        # Save figure as artifact
        fig_path = f"../mlflow/artifacts/{model_name}_confusion_matrix.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        plt.close(fig)
        
        # Log model
        input_example = X_train.iloc[:3]
        mlf_skl.log_model(
            sk_model=model_pipeline, 
            name=model_name, 
            input_example=input_example)
        
        print(f"{model_name} logged successfully → Recall: {rec:.3f}, ROC-AUC: {roc_auc:.3f}")
