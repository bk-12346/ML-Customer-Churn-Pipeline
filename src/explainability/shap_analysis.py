import shap
import pandas as pd

def run_shap_analysis(model, X_train, feature_names):
    # Convert to DataFrame (VERY IMPORTANT)
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    # Use TreeExplainer with correct setting
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )

    shap_values = explainer.shap_values(X_train_df)

    return explainer, shap_values, X_train_df