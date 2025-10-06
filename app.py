# Streamlit Frontend for Bias Detection Pipeline
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import tempfile
import os

# Import your backend functions (assuming they're in a file called bias_backend.py)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)

# Include all your backend functions here (or import them)
# def binarize_target(y: pd.Series, label_col: str) -> pd.Series:
#     # Your existing binarize_target function
#     y_norm = y.astype(str).str.strip().str.lower()

#     PAIRS = [
#         ({"yes", "y"}, {"no", "n"}),
#         ({"true", "t"}, {"false", "f"}),
#         ({"known"}, {"unknown", "unk"}),
#         ({"1"}, {"0"}),
#         ({"success", "approved", "positive", "pos", "subscribed"}, {"fail", "failed", "rejected", "negative", "neg", "not_subscribed"}),
#         ({"yes"}, {"unknown"}),
#         ({"true"}, {"unknown"}),
#         ({"known"}, {"missing"}),
#         ({"yes"}, {"missing"}),
#         ({"1"}, {"unknown", "missing"}),
#     ]

#     uniques = set(y_norm.unique())

#     if uniques <= {"0", "1"}:
#         return y_norm.astype(int)

#     pair_found = None
#     for pos_set, neg_set in PAIRS:
#         if (uniques & (pos_set | neg_set)) and (uniques & pos_set) and (uniques & neg_set):
#             pair_found = (pos_set, neg_set)
#             break

#     if pair_found is not None:
#         pos_set, neg_set = pair_found
#         mapping = {v: 1 for v in pos_set}
#         mapping.update({v: 0 for v in neg_set})
#         mask = y_norm.isin(pos_set | neg_set)
#         y_kept = y_norm.where(mask)
#         y_mapped = y_kept.map(mapping)
#         keep_index = y_mapped.dropna().index
#         return y_mapped.loc[keep_index].astype(int)

#     broad_map = {
#         "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
#         "success": 1, "approved": 1, "positive": 1, "pos": 1, "subscribed": 1, "known": 1,
#         "no": 0, "n": 0, "false": 0, "f": 0, "0": 0,
#         "fail": 0, "failed": 0, "rejected": 0, "negative": 0, "neg": 0, "not_subscribed": 0,
#         "unknown": 0, "unk": 0, "missing": 0
#     }
#     y_mapped = y_norm.map(broad_map)
#     keep_index = y_mapped.dropna().index
#     if len(keep_index) == 0:
#         raise ValueError(
#             f"Target '{label_col}' has unsupported values {sorted(uniques)} after normalization; "
#             "no recognizable binary pair found to map to 0/1."
#         )
#     return y_mapped.loc[keep_index].astype(int)


def handle_multiclass_target(y: pd.Series, label_col: str, target_class=None):
    """
    Handle both binary and multi-class targets
    """
    unique_vals = y.dropna().unique()

    # If already binary or can be binarized, use existing logic
    if len(unique_vals) <= 2:
        return binarize_target_original(y, label_col)

    # Multi-class: convert to binary (target_class vs others)
    if target_class is None:
        # This shouldn't happen in UI, but fallback to most common
        target_class = pd.Series(y).value_counts().index[0]

    # Convert to binary: target_class = 1, others = 0
    y_binary = (y == target_class).astype(int)
    return y_binary


def binarize_target_original(y: pd.Series, label_col: str) -> pd.Series:
    """Original binary target handling (renamed)"""
    # Your existing binarize_target code goes here
    y_norm = y.astype(str).str.strip().str.lower()

    PAIRS = [
        ({"yes", "y"}, {"no", "n"}),
        ({"true", "t"}, {"false", "f"}),
        ({"known"}, {"unknown", "unk"}),
        ({"1"}, {"0"}),
        # ... rest of your existing pairs
    ]

    uniques = set(y_norm.unique())

    if uniques <= {"0", "1"}:
        return y_norm.astype(int)

    pair_found = None
    for pos_set, neg_set in PAIRS:
        if (
            (uniques & (pos_set | neg_set))
            and (uniques & pos_set)
            and (uniques & neg_set)
        ):
            pair_found = (pos_set, neg_set)
            break

    if pair_found is not None:
        pos_set, neg_set = pair_found
        mapping = {v: 1 for v in pos_set}
        mapping.update({v: 0 for v in neg_set})
        mask = y_norm.isin(pos_set | neg_set)
        y_kept = y_norm.where(mask)
        y_mapped = y_kept.map(mapping)
        keep_index = y_mapped.dropna().index
        return y_mapped.loc[keep_index].astype(int)

    broad_map = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "t": 1,
        "1": 1,
        "success": 1,
        "approved": 1,
        "positive": 1,
        "pos": 1,
        "subscribed": 1,
        "known": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "f": 0,
        "0": 0,
        "fail": 0,
        "failed": 0,
        "rejected": 0,
        "negative": 0,
        "neg": 0,
        "not_subscribed": 0,
        "unknown": 0,
        "unk": 0,
        "missing": 0,
    }
    y_mapped = y_norm.map(broad_map)
    keep_index = y_mapped.dropna().index
    if len(keep_index) == 0:
        raise ValueError(
            f"Target '{label_col}' has unsupported values {sorted(uniques)} after normalization; "
            "no recognizable binary pair found to map to 0/1."
        )
    return y_mapped.loc[keep_index].astype(int)


def load_dataset_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, low_memory=False, dtype=str)


def select_columns(
    df: pd.DataFrame, label_col: str, sensitive_col: str, feature_cols: list = None
):
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in [label_col, sensitive_col]]

    y_raw = df[label_col].copy()
    target_class = getattr(select_columns, "target_class", None)
    y_bin = handle_multiclass_target(y_raw, label_col, target_class)
    keep_idx = y_bin.index

    X = df.loc[keep_idx, feature_cols].copy()
    s = df.loc[keep_idx, sensitive_col].copy()
    y = y_bin

    return X, y, s


def split_data(X, y, s, test_size=0.3, random_state=42):
    return train_test_split(
        X, y, s, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            ("num", StandardScaler(with_mean=False), num_cols),
        ],
        remainder="drop",
    )
    return pre


def train_model(X_train, y_train):
    pre = build_preprocessor(X_train)
    clf = LogisticRegression(solver="lbfgs", max_iter=3000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def compute_fairness(y_true, y_pred, sensitive):
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    dp_diff = demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    )
    dp_ratio = demographic_parity_ratio(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    )
    eo_diff = equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    )

    return {
        "group_metrics": mf.by_group.to_dict(),
        "overall": {"accuracy": accuracy_score(y_true, y_pred)},
        "disparities": {
            "demographic_parity_difference": dp_diff,
            "demographic_parity_ratio": dp_ratio,
            "equalized_odds_difference": eo_diff,
        },
    }


def classify_bias(
    disparities: dict,
    dp_diff_threshold=0.2,
    dp_ratio_lower=0.6,
    dp_ratio_upper=1.5,
    eo_diff_threshold=0.2,
):
    dpd = abs(disparities["demographic_parity_difference"])
    dpr = disparities["demographic_parity_ratio"]
    eod = abs(disparities["equalized_odds_difference"])

    reasons = []
    if dpd > dp_diff_threshold:
        reasons.append(f"abs(DP difference)>{dp_diff_threshold}")
    if (dpr < dp_ratio_lower) or (dpr > dp_ratio_upper):
        reasons.append(f"DP ratio not in [{dp_ratio_lower},{dp_ratio_upper}]")
    if eod > eo_diff_threshold:
        reasons.append(f"abs(EO difference)>{eo_diff_threshold}")

    return (len(reasons) > 0), reasons


def create_plot(plot_type, metrics, title=""):
    fig, ax = plt.subplots(figsize=(8, 4))

    if plot_type == "selection_rate":
        by_group = metrics["group_metrics"]["selection_rate"]
        groups = list(by_group.keys())
        vals = [by_group[g] for g in groups]

        ax.bar(groups, vals, color="#4C78A8")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Selection rate P(ŷ=1)")
        ax.set_title(title or "Selection rate by group")
        ax.set_xticklabels(groups, rotation=30, ha="right")

    elif plot_type == "accuracy":
        by_group = metrics["group_metrics"]["accuracy"]
        groups = list(by_group.keys())
        vals = [by_group[g] for g in groups]

        ax.bar(groups, vals, color="#72B7B2")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(title or "Accuracy by group")
        ax.set_xticklabels(groups, rotation=30, ha="right")

    elif plot_type == "disparities":
        disp = metrics["disparities"]
        labels = ["|DP diff|", "DP ratio", "|EO diff|"]
        values = [
            float(abs(disp["demographic_parity_difference"])),
            float(disp["demographic_parity_ratio"]),
            float(abs(disp["equalized_odds_difference"])),
        ]

        ax.bar(labels, values, color=["#F58518", "#E45756", "#54A24B"])
        # Threshold lines
        ax.axhline(
            0.2, color="#F58518", linestyle="--", linewidth=1, label="DP diff thresh"
        )
        ax.axhline(
            0.2, color="#54A24B", linestyle="--", linewidth=1, label="EO diff thresh"
        )
        ax.axhline(
            0.6, color="#E45756", linestyle="--", linewidth=1, label="DP ratio lower"
        )
        ax.axhline(
            1.5, color="#E45756", linestyle="--", linewidth=1, label="DP ratio upper"
        )

        ymax = max(values + [1.5, 0.2, 0.2]) * 1.15
        ax.set_ylim(0, ymax)
        ax.set_title(title or "Disparity summary")
        ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    return fig


def run_bias_detection_streamlit(
    df,
    label_col,
    sensitive_col,
    feature_cols=None,
    test_size=0.3,
    random_state=42,
    target_class=None,
):
    try:
        select_columns.target_class = target_class
        X, y, s = select_columns(df, label_col, sensitive_col, feature_cols)
        X_train, X_test, y_train, y_test, s_train, s_test = split_data(
            X, y, s, test_size, random_state
        )

        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_fairness(y_test, y_pred, s_test)
        is_biased, reasons = classify_bias(metrics["disparities"])

        return {
            "biased": is_biased,
            "reasons": reasons,
            "metrics": metrics,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Streamlit App
def main():
    st.set_page_config(
        page_title="Bias Detection Pipeline",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("⚖️ Dataset Bias Detection Pipeline")
    st.markdown(
        "Upload a dataset and detect potential bias across sensitive attributes using fairness metrics."
    )

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", type="csv", help="Upload your dataset in CSV format"
    )

    if uploaded_file is not None:
        # Load and display basic info
        try:
            df = load_dataset_from_upload(uploaded_file)

            # Display dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")

            # Show dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Column selection
            st.sidebar.subheader("Column Selection")

            # Label column selection
            label_col = st.sidebar.selectbox(
                "Select Target/Label Column",
                options=df.columns.tolist(),
                help="Choose the column that contains the outcome you want to predict",
            )
            # Check for multi-class targets
            target_class = None
            if label_col:
                unique_labels = df[label_col].dropna().unique()
                num_classes = len(unique_labels)

                if num_classes > 2:
                    st.sidebar.warning(
                        f"⚠️ Multi-class target detected: {num_classes} classes"
                    )

                    multiclass_option = st.sidebar.radio(
                        "Multi-class handling:",
                        [
                            "Auto: Most frequent class vs Others",
                            "Select specific target class",
                        ],
                        help="Fairness analysis works best with binary targets",
                    )

                    if multiclass_option == "Select specific target class":
                        target_class = st.sidebar.selectbox(
                            "Target class (will be '1', others become '0'):",
                            options=unique_labels.tolist(),
                            help="Choose which class to focus on for bias detection",
                        )
                    else:
                        # Auto-select most frequent class
                        target_class = pd.Series(df[label_col]).value_counts().index[0]
                        st.sidebar.info(f"Using '{target_class}' vs Others")

                elif num_classes < 2:
                    st.sidebar.error("❌ Target column has only 1 unique value")
                else:
                    st.sidebar.success("✅ Binary target detected")

            # Sensitive column selection
            sensitive_col = st.sidebar.selectbox(
                "Select Sensitive Attribute",
                options=[col for col in df.columns.tolist() if col != label_col],
                help="Choose the column representing the sensitive attribute (e.g., race, gender, age)",
            )

            # Feature column selection (optional)
            available_features = [
                col
                for col in df.columns.tolist()
                if col not in [label_col, sensitive_col]
            ]

            feature_selection = st.sidebar.radio(
                "Feature Selection",
                ["Use all available features", "Select specific features"],
            )

            if feature_selection == "Select specific features":
                feature_cols = st.sidebar.multiselect(
                    "Select Feature Columns",
                    options=available_features,
                    default=available_features,
                    help="Choose which columns to use as features for the model",
                )
            else:
                feature_cols = None

            # Advanced settings
            st.sidebar.subheader("Advanced Settings")

            test_size = st.sidebar.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Proportion of data to use for testing",
            )

            random_state = st.sidebar.number_input(
                "Random State",
                min_value=0,
                max_value=1000,
                value=42,
                help="Set for reproducible results",
            )

            # Run analysis button
            if st.sidebar.button("🔍 Run Bias Detection", type="primary"):

                # Show column info
                st.subheader("Analysis Configuration")
                config_col1, config_col2 = st.columns(2)

                with config_col1:
                    st.info(f"**Target Column:** {label_col}")
                    st.info(f"**Sensitive Attribute:** {sensitive_col}")

                with config_col2:
                    if feature_cols:
                        st.info(f"**Selected Features:** {len(feature_cols)} columns")
                    else:
                        st.info(
                            f"**Using All Features:** {len(available_features)} columns"
                        )
                    st.info(
                        f"**Test Split:** {int(test_size*100)}% / {int((1-test_size)*100)}%"
                    )

                # Run the analysis
                with st.spinner("Running bias detection analysis..."):
                    result = run_bias_detection_streamlit(
                        df=df,
                        label_col=label_col,
                        sensitive_col=sensitive_col,
                        feature_cols=feature_cols,
                        test_size=test_size,
                        random_state=random_state,
                        target_class=target_class,
                    )

                if result["success"]:
                    # Display main result
                    st.subheader("🎯 Analysis Results")
                    if target_class is not None:
                        st.info(f"**Analysis Focus:** '{target_class}' vs All Others")
                    # Bias classification result
                    if result["biased"]:
                        st.error("🚨 **Dataset appears BIASED**")
                        st.write("**Reasons:**")
                        for reason in result["reasons"]:
                            st.write(f"• {reason}")
                    else:
                        st.success("✅ **Dataset appears STABLE**")
                        st.write(
                            "No significant bias detected based on the selected thresholds."
                        )

                    # Metrics overview
                    metrics = result["metrics"]

                    # Overall accuracy
                    st.subheader("📊 Performance Metrics")
                    overall_acc = metrics["overall"]["accuracy"]
                    st.metric(
                        "Overall Accuracy",
                        f"{overall_acc:.3f}",
                        f"{overall_acc*100:.1f}%",
                    )

                    # Group-level metrics table
                    st.subheader("👥 Group-Level Metrics")

                    group_data = []
                    for group in metrics["group_metrics"]["selection_rate"].keys():
                        group_data.append(
                            {
                                "Group": group,
                                "Selection Rate": f"{metrics['group_metrics']['selection_rate'][group]:.3f}",
                                "Accuracy": f"{metrics['group_metrics']['accuracy'][group]:.3f}",
                            }
                        )

                    group_df = pd.DataFrame(group_data)
                    st.dataframe(group_df, use_container_width=True)

                    # Disparity metrics
                    st.subheader("⚖️ Fairness Disparities")

                    disp = metrics["disparities"]
                    disp_col1, disp_col2, disp_col3 = st.columns(3)

                    with disp_col1:
                        st.metric(
                            "Demographic Parity Difference",
                            f"{disp['demographic_parity_difference']:.3f}",
                            help="Closer to 0 is better",
                        )

                    with disp_col2:
                        st.metric(
                            "Demographic Parity Ratio",
                            f"{disp['demographic_parity_ratio']:.3f}",
                            help="Closer to 1 is better",
                        )

                    with disp_col3:
                        st.metric(
                            "Equalized Odds Difference",
                            f"{disp['equalized_odds_difference']:.3f}",
                            help="Closer to 0 is better",
                        )

                    # Visualizations
                    st.subheader("📈 Visualizations")

                    # Create tabs for different plots
                    tab1, tab2, tab3 = st.tabs(
                        ["Selection Rate", "Accuracy", "Disparities"]
                    )

                    with tab1:
                        fig1 = create_plot(
                            "selection_rate", metrics, "Selection Rate by Group"
                        )
                        st.pyplot(fig1)

                    with tab2:
                        fig2 = create_plot("accuracy", metrics, "Accuracy by Group")
                        st.pyplot(fig2)

                    with tab3:
                        bias_status = "Biased" if result["biased"] else "Stable"
                        fig3 = create_plot(
                            "disparities", metrics, f"Disparity Summary — {bias_status}"
                        )
                        st.pyplot(fig3)

                    # Raw results (expandable)
                    with st.expander("🔧 Raw Analysis Results"):
                        st.json(result)

                else:
                    st.error(f"❌ Analysis failed: {result['error']}")
                    st.info("Please check your column selections and data format.")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format.")

    else:
        # Welcome message
        st.info(
            "👆 Please upload a CSV file in the sidebar to begin the bias detection analysis."
        )

        # Instructions
        st.subheader("📋 How to Use")
        st.markdown(
            """
        1. **Upload your CSV file** using the file uploader in the sidebar
        2. **Select your target column** (the outcome you want to predict)
        3. **Choose a sensitive attribute** (e.g., race, gender, age group)
        4. **Optionally select specific features** or use all available columns
        5. **Adjust advanced settings** if needed (test size, random state)
        6. **Click "Run Bias Detection"** to analyze your dataset
        
        The pipeline will:
        - Automatically handle different label formats (yes/no, true/false, 1/0, etc.)
        - Train a simple logistic regression model
        - Compute fairness metrics across groups
        - Classify your dataset as "Biased" or "Stable"
        - Generate visualizations showing group differences
        """
        )

        st.subheader("🎯 Fairness Metrics Explained")
        st.markdown(
            """
        - **Demographic Parity Difference**: Difference between highest and lowest group selection rates (closer to 0 is better)
        - **Demographic Parity Ratio**: Ratio of lowest to highest group selection rates (closer to 1 is better)
        - **Equalized Odds Difference**: Difference in true positive and false positive rates across groups (closer to 0 is better)
        """
        )


if __name__ == "__main__":
    main()
