- [DataAnalysisLibrary](#dataanalysislibrary)
  - [Project Structure](#project-structure)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Example Usage](#example-usage)
      - [Create an instance of DataAnalysisVisualizer](#create-an-instance-of-dataanalysisvisualizer)
      - [Create scatter plots](#create-scatter-plots)
  - [Feature Engineering](#feature-engineering)
    - [DataFramePreprocessor](#dataframepreprocessor)
      - [Example Usage](#example-usage-1)
    - [FeatureSelector](#featureselector)
      - [Example Usage](#example-usage-2)
  - [K-Means Clustering](#k-means-clustering)
    - [`plot_elbow`](#plot_elbow)
      - [Parameters](#parameters)
      - [Example Usage](#example-usage-3)
  - [Installation](#installation)
  - [License](#license)

# DataAnalysisLibrary

DataAnalysisLibrary is a Python library aimed at providing useful functions and tools for performing data analysis. It includes a variety of functions for data visualization and analysis using popular Python libraries such as Seaborn and Matplotlib.

## Project Structure

The project is organized as follows:

- `DataAnalysis`: This folder contains the core library files and modules.
  - `eda.py`: Module for exploratory data analysis (EDA) functions.
  - `feature_engineering.py`: Module for feature engineering functions.
  - `kmeans.py`: Module for K-means clustering functions.

- `Setup.py`: Python setup script for installing the library.
- `environment.yaml`: Conda environment file for specifying dependencies.

## Exploratory Data Analysis (EDA)

The `eda.py` module provides a `DataAnalysisVisualizer` class with functions to create various types of plots:

- `draw_scatterplot`: Create scatter plots for multiple x-columns against a single y-column.
- `draw_histplot`: Create histogram plots for specified columns.
- `draw_countplot`: Create count plots for specified columns.
- `draw_pairplot`: Create a pairplot for specified columns.
- `draw_heatmap`: Create a heatmap for a correlation matrix.
- `draw_boxplot`: Create box plots for specified columns.

You can customize the appearance of these plots by passing various optional parameters.

### Example Usage

Here's an example of how to use the `DataAnalysisVisualizer` class to create scatter plots:

```python
from DataAnalysisLibrary.eda import DataAnalysisVisualizer
```

#### Create an instance of DataAnalysisVisualizer
```python
visualizer = DataAnalysisVisualizer()
```

#### Create scatter plots
```python
visualizer.draw_scatterplot(dataframe, ['x1', 'x2', 'x3'], 'y', hue_column='hue_column', figsize=(12, 6))
```

## Feature Engineering

The `feature_engineering.py` module provides classes and functions for performing feature engineering on datasets. It includes features for data preprocessing, transformation, and feature selection.

### DataFramePreprocessor

The `DataFramePreprocessor` class helps perform preprocessing on data frames. It allows you to apply various transformations to specific columns in a DataFrame, including:

- `MinMaxScaler`: Scale numerical features to a specified range.
- `LabelEncoder`: Encode categorical features as numerical values.
- `OneHotEncoder`: Convert categorical features into binary columns (one-hot encoding).

You can use these transformers individually or in combination to preprocess your data efficiently.

#### Example Usage

```python
from feature_engineering import DataFramePreprocessor
import pandas as pd

# Load your dataset (e.g., 'titanic.csv')
titanic = pd.read_csv('titanic.csv')

# Initialize the DataFramePreprocessor
dfp = DataFramePreprocessor()

# Define columns for different transformers
numeric_cols = ['Parch', 'Age', 'Fare']
onehot_cols = ['SibSp', 'Sex']
label_cols = ['Pclass']

# Apply transformations to specific columns
df, transformers = dfp.fit_transform_multiple_transformer(
    titanic, [MinMaxScaler(), OneHotEncoder(), LabelEncoder()],
    [numeric_cols, onehot_cols, label_cols]
)
```

### FeatureSelector

The `FeatureSelector` class provides methods for feature selection and analysis. It includes the following functionality:

- `get_permutation_importance`: Calculate permutation importance for features based on a fitted estimator.
- `get_vif_dataframe`: Calculate Variance Inflation Factor (VIF) for features to assess multicollinearity.
- `vif_analysis`: Perform VIF analysis to identify and remove features with high multicollinearity.

You can use these methods to analyze the importance of features in your dataset and identify potential issues with multicollinearity.

#### Example Usage

```python
from feature_engineering import FeatureSelector
import pandas as pd

# Load your dataset (e.g., 'titanic_train.csv')
titanic = pd.read_csv('titanic_train.csv')

# Initialize the FeatureSelector
fs = FeatureSelector()

# Define a formula for your model and perform VIF analysis
formula = "Survived ~ " + " + ".join(titanic.columns.difference(['Survived']))
fs.vif_analysis(formula, titanic)
```

These classes and functions provide essential tools for preparing your data and selecting relevant features for machine learning models.

## K-Means Clustering

The `kmeans.py` module provides a function for performing K-Means clustering and visualizing the "elbow" method to determine the optimal number of clusters.

### `plot_elbow`

The `plot_elbow` function helps in identifying the optimal number of clusters for K-Means clustering. It does so by fitting K-Means models to the data for a range of cluster numbers and plotting the inertia (within-cluster sum of squares) values.

#### Parameters

- `data` (array-like): The data on which to perform K-Means clustering.
- `max_clusters` (int, optional): The maximum number of clusters to consider. Default is 10.
- `random_state` (int or None, optional): The seed used by the random number generator. Default is `None`.
- `n_init` (int, optional): The number of times K-Means will be run with different centroid seeds. Default is 10.

#### Example Usage

```python
from kmeans import plot_elbow
import numpy as np

# Generate sample data
data = np.random.rand(100, 2)

# Plot the elbow method to determine the optimal number of clusters
plot_elbow(data, max_clusters=10, random_state=42, n_init=10)
```


## Installation

You can install DataAnalysisLibrary using pip:

```
pip install git+https://github.com/tooha289/DataAnalysisLibrary.git
```

For Conda users, you can create a Conda environment using the provided `environment.yaml` file:

```
conda env create -f environment.yaml
conda activate data-analysis-env
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Note**: This library is under development, and additional features and improvements will be added in the future. We welcome contributions and feedback from the community.