import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "Iris.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FEATURE_COLUMNS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET_COLUMN = "Species"

TEST_SIZE = 0.2      # 20% of the data is kept aside for testing
RANDOM_STATE = 42    # fixed number so results are the same every time it run
MAX_DEPTH = 3        # limits tree depth so it stays simple and does not overfit
DPI = 300


def print_heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def save_figure(fig, filename):
    """Save a figure into outputs/figures/ and close it."""
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {path.relative_to(PROJECT_ROOT)}")


# LOADING AND SPLITTING THE DATA
def load_data():
    """Read Iris.csv and return the features (X) and the labels (y)."""
    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found at {DATA_FILE}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    needed = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in CSV: {missing}")
        sys.exit(1)

    # Shorten "Iris-setosa" to "setosa"
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.replace("Iris-", "", regex=False)

    X = df[FEATURE_COLUMNS]   # the four measurements (inputs)
    y = df[TARGET_COLUMN]     # the species (what we want to predict)
    return X, y


def split_data(X, y):
    """Split into training and test sets.
    stratify=y keeps the same species proportions in both sets."""
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


# TRAINING AND EVALUATION
def train_model(X_train, y_train):
    """Create and train the decision tree."""
    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, class_names):
    """Calculate accuracy, precision and recall and save them to a CSV file."""
    predictions = model.predict(X_test)

    # Accuracy = correct predictions / all predictions
    accuracy = accuracy_score(y_test, predictions)

    # Precision and recall for EACH species
    precision, recall, _, support = precision_recall_fscore_support(
        y_test, predictions, labels=class_names, zero_division=0)

    # Macro average = simple average of the three species' scores
    macro_precision = precision.mean()
    macro_recall = recall.mean()

    print_heading("MODEL EVALUATION (on the unseen test set)")
    print(f"Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"Precision (macro average): {macro_precision:.4f}")
    print(f"Recall    (macro average): {macro_recall:.4f}")

    # Table of per-species results
    metrics = pd.DataFrame({
        "Species": class_names,
        "Precision": precision,
        "Recall": recall,
        "Support (test samples)": support,
    })
    macro_row = pd.DataFrame({
        "Species": ["macro average"],
        "Precision": [macro_precision],
        "Recall": [macro_recall],
        "Support (test samples)": [support.sum()],
    })
    accuracy_row = pd.DataFrame({
        "Species": ["overall accuracy"],
        "Precision": [accuracy],
        "Recall": [accuracy],
        "Support (test samples)": [support.sum()],
    })
    metrics = pd.concat([metrics, macro_row, accuracy_row], ignore_index=True)

    print("\n--- Precision and recall per species ---")
    print(metrics.round(4).to_string(index=False))

    path = TABLES_DIR / "decision_tree_metrics.csv"
    metrics.round(4).to_csv(path, index=False)
    print(f"\n  Saved table : {path.relative_to(PROJECT_ROOT)}")

    return predictions


def show_confusion_matrix(y_test, predictions, class_names):
    """Print and plot the confusion matrix."""
    matrix = confusion_matrix(y_test, predictions, labels=class_names)
    matrix_table = pd.DataFrame(matrix,
                                index=[f"actual {c}" for c in class_names],
                                columns=[f"predicted {c}" for c in class_names])
    print("\n--- Confusion matrix (rows = actual, columns = predicted) ---")
    print(matrix_table.to_string())

    path = TABLES_DIR / "decision_tree_confusion_matrix.csv"
    matrix_table.to_csv(path)
    print(f"\n  Saved table : {path.relative_to(PROJECT_ROOT)}")

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", linewidths=1,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 18},
                cbar_kws={"label": "Number of flowers"}, ax=ax)
    ax.set_title("Decision Tree Confusion Matrix (Test Set)", fontweight="bold")
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("Actual species")
    save_figure(fig, "decision_tree_confusion_matrix.png")


def cross_validate(X, y):
    """Extra check: repeat training/testing 5 times on different splits.
    This shows whether the result depends on one lucky split."""
    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print("\n--- 5-fold cross-validation accuracy ---")
    print("Scores for each fold:", [round(float(s), 4) for s in scores])
    print(f"Mean accuracy: {scores.mean():.4f}  (+/- {scores.std():.4f})")


# ---------------------------------------------------------------------------
# 5. VISUALIZING THE MODEL
# ---------------------------------------------------------------------------
def plot_tree_diagram(model, class_names):
    """Draw the decision tree so we can see the rules it learned."""
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(model, feature_names=FEATURE_COLUMNS, class_names=class_names,
              filled=True, rounded=True, fontsize=12, ax=ax)
    ax.set_title("Decision Tree Learned from the Iris Training Data",
                 fontsize=20, fontweight="bold")
    save_figure(fig, "decision_tree_diagram.png")


def plot_feature_importance(model):
    """Bar chart showing which measurements the tree relied on most."""
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importance = importance.sort_values()

    print("\n--- Feature importance ---")
    print(importance.sort_values(ascending=False).round(4).to_string())

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance.index, importance.values, color="#4c72b0", edgecolor="black")
    for i, value in enumerate(importance.values):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=12)
    ax.set_title("Feature Importance in the Decision Tree", fontweight="bold")
    ax.set_xlabel("Importance (higher = used more by the tree)")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, max(importance.values) * 1.15)
    save_figure(fig, "decision_tree_feature_importance.png")


def main():
    print_heading("IRIS DATASET - DECISION TREE CLASSIFIER (COS 103)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    class_names = sorted(y.unique())
    print(f"Samples: {len(X)}   Features: {len(FEATURE_COLUMNS)}   "
          f"Classes: {class_names}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training samples: {len(X_train)}   Test samples: {len(X_test)}")
    print(f"Tree settings: max_depth={MAX_DEPTH}, random_state={RANDOM_STATE}")

    model = train_model(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    predictions = evaluate_model(model, X_test, y_test, class_names)
    show_confusion_matrix(y_test, predictions, class_names)
    cross_validate(X, y)

    print_heading("CREATING VISUALIZATIONS")
    plot_tree_diagram(model, class_names)
    plot_feature_importance(model)

    print_heading("DONE")


if __name__ == "__main__":
    main()