# Data folder

This folder contains the dataset used by the project.

| File | Description |
|------|-------------|
| `Iris.csv` | The Iris flower dataset (150 rows, 6 columns) |

## Columns in `Iris.csv`

| Column | Meaning |
|--------|---------|
| `Id` | Row number (not a measurement; ignored in the analysis) |
| `SepalLengthCm` | Sepal length in centimetres |
| `SepalWidthCm` | Sepal width in centimetres |
| `PetalLengthCm` | Petal length in centimetres |
| `PetalWidthCm` | Petal width in centimetres |
| `Species` | Flower species (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) |

## Source

The Iris dataset was introduced by the British statistician and biologist Ronald A. Fisher
in his 1936 paper *"The use of multiple measurements in taxonomic problems"*. The measurements
were originally collected by Edgar Anderson. This copy uses the common CSV layout (with an `Id`
column) that is widely distributed on Kaggle and in the UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/53/iris).

## Important

Do not rename or move `Iris.csv`. The script `src/iris_eda.py` looks for it at `data/Iris.csv`.
