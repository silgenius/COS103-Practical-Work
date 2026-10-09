"""
Exploratory Data Analysis (EDA) on the Iris Dataset
===================================================
Course : COS 103 (100 Level), University of Ibadan, Nigeria

What this program does
----------------------
1. Loads the Iris dataset from data/Iris.csv
2. Inspects its structure (head, tail, shape, types, missing values, duplicates)
3. Calculates descriptive statistics (overall and per species)
4. Creates and saves 10 visualizations as high-resolution PNG files
5. Saves summary tables as CSV files

How to run (from the project root folder):
    python src/iris_eda.py          (Windows)
    python3 src/iris_eda.py         (Linux / macOS)
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots to files without opening windows (works everywhere)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# 2. SETTINGS AND FILE PATHS
# ---------------------------------------------------------------------------
# Path(__file__) is this script's location. Its parent is "src", and the parent
# of "src" is the project root. This means the script works no matter which
# folder you start it from, and no paths need manual editing.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "Iris.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

# The four measurement columns in the CSV, and the nicer names we use on plots
RENAME_MAP = {
    "SepalLengthCm": "Sepal Length (cm)",
    "SepalWidthCm": "Sepal Width (cm)",
    "PetalLengthCm": "Petal Length (cm)",
    "PetalWidthCm": "Petal Width (cm)",
}
FEATURES = list(RENAME_MAP.values())
SPECIES_ORDER = ["setosa", "versicolor", "virginica"]

# One fixed colour per species so every plot uses the same colours
SPECIES_COLOURS = {
    "setosa": "#1f77b4",      # blue
    "versicolor": "#ff7f0e",  # orange
    "virginica": "#2ca02c",   # green
}

DPI = 300  # resolution of saved PNG files


def set_plot_theme():
    """Apply one consistent look to every plot."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })


def print_heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def save_figure(fig, filename):
    """Save a figure into outputs/figures/ and close it to free memory."""
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {path.relative_to(PROJECT_ROOT)}")


def save_table(table, filename, keep_index=True):
    """Save a pandas table into outputs/tables/ as a CSV file."""
    path = TABLES_DIR / filename
    table.to_csv(path, index=keep_index)
    print(f"  Saved table : {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 3. LOADING THE DATA
# ---------------------------------------------------------------------------
def load_data():
    """Read the CSV file and tidy up the column and species names."""
    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found at {DATA_FILE}")
        print("Make sure 'Iris.csv' is inside the 'data' folder.")
        sys.exit(1)

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as error:
        print(f"ERROR: Could not read the CSV file: {error}")
        sys.exit(1)

    # Check that the expected columns exist
    expected = list(RENAME_MAP.keys()) + ["Species"]
    missing_columns = [col for col in expected if col not in df.columns]
    if missing_columns:
        print(f"ERROR: These columns are missing from the CSV: {missing_columns}")
        sys.exit(1)

    # The 'Id' column is only a row number, so we drop it
    if "Id" in df.columns:
        df = df.drop(columns="Id")

    # Rename columns and shorten "Iris-setosa" to "setosa"
    df = df.rename(columns=RENAME_MAP)
    df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)
    return df


# ---------------------------------------------------------------------------
# 4. INITIAL INSPECTION
# ---------------------------------------------------------------------------
def inspect_data(df):
    """Section A: look at the structure and quality of the data."""
    print_heading("A. INITIAL INSPECTION")

    print("\n--- First 5 rows ---")
    print(df.head().to_string())

    print("\n--- Last 5 rows ---")
    print(df.tail().to_string())

    print("\n--- Shape (rows, columns) ---")
    print(df.shape)

    print("\n--- Column names ---")
    print(list(df.columns))

    print("\n--- Data types ---")
    print(df.dtypes.to_string())

    print("\n--- Dataset information (.info()) ---")
    df.info()

    # Missing values
    print("\n--- Missing values per column ---")
    missing_summary = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().mean() * 100).round(2),
    })
    print(missing_summary.to_string())
    save_table(missing_summary, "missing_values_summary.csv")

    # Duplicates
    duplicate_count = int(df.duplicated().sum())
    print(f"\n--- Duplicate rows ---\nNumber of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        # duplicated() marks the 2nd, 3rd... copy; keep=False shows ALL copies
        print("All rows that appear more than once (row index on the left):")
        print(df[df.duplicated(keep=False)].to_string())

    # Species counts
    print("\n--- Observations per species ---")
    species_counts = df["Species"].value_counts().reindex(SPECIES_ORDER)
    species_table = species_counts.rename_axis("Species").reset_index(name="Count")
    print(species_table.to_string(index=False))
    save_table(species_table, "species_counts.csv", keep_index=False)

    print(f"\nNumber of unique species: {df['Species'].nunique()}")
    print(f"Species names: {df['Species'].unique().tolist()}")


# ---------------------------------------------------------------------------
# 5. SUMMARY STATISTICS
# ---------------------------------------------------------------------------
def build_summary_table(data):
    """Return count, mean, median, std, min, max, Q1, Q3 and IQR for each feature."""
    summary = pd.DataFrame({
        "count": data.count(),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),          # sample standard deviation (n - 1)
        "min": data.min(),
        "Q1_25%": data.quantile(0.25),
        "Q3_75%": data.quantile(0.75),
        "max": data.max(),
    })
    summary["IQR"] = summary["Q3_75%"] - summary["Q1_25%"]
    return summary.round(4)


def summary_statistics(df):
    """Section B: overall and species-wise descriptive statistics."""
    print_heading("B. SUMMARY STATISTICS")

    # Display options so wide tables are not cut off
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    # Overall statistics
    overall = build_summary_table(df[FEATURES])
    print("\n--- Overall descriptive statistics ---")
    print(overall.to_string())
    save_table(overall, "overall_descriptive_statistics.csv")

    # Species-wise statistics
    pieces = []
    for species in SPECIES_ORDER:
        subset = df[df["Species"] == species][FEATURES]
        table = build_summary_table(subset)
        table.insert(0, "Species", species)
        table.index.name = "Feature"
        pieces.append(table.reset_index())
    species_stats = pd.concat(pieces, ignore_index=True)

    print("\n--- Species-wise descriptive statistics ---")
    for species in SPECIES_ORDER:
        print(f"\nSpecies: {species}")
        part = species_stats[species_stats["Species"] == species]
        print(part.drop(columns="Species").to_string(index=False))
    save_table(species_stats, "species_descriptive_statistics.csv", keep_index=False)

    # Correlation matrix
    correlation = df[FEATURES].corr().round(4)
    print("\n--- Correlation matrix (Pearson) ---")
    print(correlation.to_string())
    save_table(correlation, "correlation_matrix.csv")

    return correlation


# ---------------------------------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------------------------------
def plot_species_count(df):
    """1. Bar chart of the number of samples per species."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(data=df, x="Species", order=SPECIES_ORDER, hue="Species",
                  hue_order=SPECIES_ORDER, palette=SPECIES_COLOURS,
                  legend=False, ax=ax)
    # Write the count on top of every bar
    for bar in ax.patches:
        ax.annotate(f"{int(bar.get_height())}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_title("Number of Samples per Species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, df["Species"].value_counts().max() * 1.15)
    save_figure(fig, "species_count.png")


def plot_histograms(df):
    """2. Histograms of the four numerical features."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, feature in zip(axes.flatten(), FEATURES):
        sns.histplot(df[feature], bins=15, kde=True, color="#4c72b0", ax=ax)
        mean_value = df[feature].mean()
        ax.axvline(mean_value, color="red", linestyle="--", linewidth=2,
                   label=f"Mean = {mean_value:.2f}")
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.legend()
    fig.suptitle("Histograms of Iris Features", fontsize=22, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "feature_histograms.png")


def plot_boxplots(df):
    """3. Box plots of every feature, split by species."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, feature in zip(axes.flatten(), FEATURES):
        sns.boxplot(data=df, x="Species", y=feature, order=SPECIES_ORDER,
                    hue="Species", hue_order=SPECIES_ORDER,
                    palette=SPECIES_COLOURS, legend=False, ax=ax)
        ax.set_title(f"{feature} by Species")
        ax.set_xlabel("Species")
        ax.set_ylabel(feature)
    fig.suptitle("Box Plots of Iris Features by Species (dots = outliers)",
                 fontsize=20, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "boxplots_by_species.png")


def plot_scatter(df, x_feature, y_feature, title, filename):
    """4 and 5. Scatter plot of two features, coloured by species."""
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(data=df, x=x_feature, y=y_feature, hue="Species",
                    hue_order=SPECIES_ORDER, palette=SPECIES_COLOURS,
                    style="Species", style_order=SPECIES_ORDER,
                    s=140, alpha=0.85, edgecolor="black", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.legend(title="Species", loc="best")
    save_figure(fig, filename)


def plot_pairplot(df):
    """6. Pair plot: every feature against every other feature."""
    grid = sns.pairplot(df, vars=FEATURES, hue="Species", hue_order=SPECIES_ORDER,
                        palette=SPECIES_COLOURS, diag_kind="kde",
                        height=3.2, plot_kws={"alpha": 0.8, "s": 45})
    grid.figure.suptitle("Pair Plot of Iris Features by Species",
                         fontsize=24, fontweight="bold", y=1.03)
    grid.legend.set_title("Species")
    save_figure(grid.figure, "pairplot.png")


def plot_correlation_heatmap(correlation):
    """7. Heatmap of the correlation matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=1, square=True,
                annot_kws={"size": 14},
                cbar_kws={"label": "Correlation coefficient"}, ax=ax)
    ax.set_title("Correlation Heatmap of Iris Features")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_figure(fig, "correlation_heatmap.png")


def plot_violins(df):
    """8. Violin plots: shape of each feature's distribution per species."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, feature in zip(axes.flatten(), FEATURES):
        sns.violinplot(data=df, x="Species", y=feature, order=SPECIES_ORDER,
                       hue="Species", hue_order=SPECIES_ORDER,
                       palette=SPECIES_COLOURS, inner="quartile",
                       legend=False, ax=ax)
        ax.set_title(f"{feature} by Species")
        ax.set_xlabel("Species")
        ax.set_ylabel(feature)
    fig.suptitle("Violin Plots of Iris Features by Species",
                 fontsize=22, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "violinplots.png")


def plot_mean_comparison(df):
    """9. Grouped bar chart comparing the mean of each feature for each species."""
    means = df.groupby("Species")[FEATURES].mean().reindex(SPECIES_ORDER)
    positions = np.arange(len(FEATURES))   # one group of bars per feature
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, species in enumerate(SPECIES_ORDER):
        bars = ax.bar(positions + i * bar_width, means.loc[species],
                      width=bar_width, label=species,
                      color=SPECIES_COLOURS[species], edgecolor="black")
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=11)
    ax.set_xticks(positions + bar_width)
    ax.set_xticklabels(FEATURES)
    ax.set_title("Mean Value of Each Feature by Species")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean Measurement (cm)")
    ax.set_ylim(0, means.to_numpy().max() * 1.15)
    ax.legend(title="Species")
    save_figure(fig, "species_mean_comparison.png")


def create_all_plots(df, correlation):
    """Section 3: create every visualization."""
    print_heading("C. CREATING VISUALIZATIONS")
    set_plot_theme()
    plot_species_count(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_scatter(df, "Sepal Length (cm)", "Sepal Width (cm)",
                 "Sepal Length vs Sepal Width", "sepal_scatter.png")
    plot_scatter(df, "Petal Length (cm)", "Petal Width (cm)",
                 "Petal Length vs Petal Width", "petal_scatter.png")
    plot_pairplot(df)
    plot_correlation_heatmap(correlation)
    plot_violins(df)
    plot_mean_comparison(df)


# ---------------------------------------------------------------------------
# 7. MAIN PROGRAM
# ---------------------------------------------------------------------------
def main():
    print_heading("IRIS DATASET - EXPLORATORY DATA ANALYSIS (COS 103)")
    print(f"Project folder : {PROJECT_ROOT}")
    print(f"Dataset file   : {DATA_FILE.relative_to(PROJECT_ROOT)}")

    # Create output folders if they do not exist yet
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    inspect_data(df)
    correlation = summary_statistics(df)
    create_all_plots(df, correlation)

    print_heading("DONE")
    print(f"Figures saved in: {FIGURES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Tables saved in : {TABLES_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
