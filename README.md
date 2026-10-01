# Machine-Learning-Based-Crop-Rainfall-Prediction-and-Plant-Health-Classification

A comprehensive machine learning project that combines **crop rainfall prediction, plant health classification, and crop-condition clustering** using multiple supervised and unsupervised machine learning techniques.

The project analyzes agricultural and environmental datasets to understand relationships between rainfall, crop production, environmental conditions, soil properties, and plant health. Multiple machine learning algorithms are trained and compared using appropriate evaluation metrics, followed by hyperparameter tuning for selected models.

---

##  Project Overview

Agriculture is highly dependent on environmental conditions such as rainfall, temperature, humidity, soil nutrients, and crop characteristics. Accurate prediction of rainfall and identification of plant health conditions can support better agricultural planning and resource management.

This project implements three major machine learning components:

### 1.  Crop Rainfall Prediction

A supervised **regression problem** where:

* **Target:** `Annual_Rainfall`
* Agricultural, environmental, crop, and seasonal features are used as predictors.
* Multiple regression algorithms are trained and compared.
* Data preprocessing includes missing-value handling, feature scaling, categorical encoding, and feature engineering.

### 2.  Plant Health Classification

A supervised **multi-class classification problem** where:

* **Target:** `health_class`
* Plant/environmental characteristics are used to classify plant health conditions.
* Multiple classification algorithms are compared.
* Hyperparameter tuning is performed for Random Forest, SVM, and MLP models.

### 3.  Crop-Condition Clustering

An unsupervised learning component that groups agricultural observations according to:

* Nitrogen (`N`)
* Phosphorus (`P`)
* Potassium (`K`)
* Temperature
* Humidity
* pH
* Rainfall

Both **K-Means** and **Agglomerative Hierarchical Clustering** are implemented.

---

#  Objectives

The main objectives of the project are:

* Predict annual rainfall using agricultural and environmental information.
* Analyze relationships between rainfall and crop-related variables.
* Classify plant health conditions using environmental and plant characteristics.
* Compare multiple machine learning algorithms.
* Apply appropriate preprocessing techniques for numerical and categorical features.
* Evaluate regression models using R², RMSE, and MAE.
* Evaluate classification models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* Perform hyperparameter tuning using `GridSearchCV`.
* Identify meaningful agricultural groups using clustering.
* Visualize high-dimensional agricultural data using PCA and t-SNE.
* Analyze cluster quality using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.

---

#  Project Architecture

```text
                    ┌───────────────────────────┐
                    │     Agricultural Data     │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
       Crop Yield Data      Plant Health Data   Crop Recommendation
              │                   │                   │
              ▼                   ▼                   ▼
        Data Cleaning        Data Cleaning       Data Cleaning
              │                   │                   │
              ▼                   ▼                   ▼
       Feature Engineering   Preprocessing       Standardization
              │                   │                   │
              ▼                   ▼                   ▼
       Rainfall Regression   Classification      Clustering
              │                   │                   │
              ▼                   ▼                   ▼
       Model Comparison     Model Comparison    K-Means
              │                   │              Hierarchical
              ▼                   ▼                   │
       Hyperparameter       Hyperparameter            ▼
          Tuning               Tuning          PCA / t-SNE
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
                     Agricultural Insights
```

---

#  Datasets

The project uses three datasets.

## 1. Crop Yield Dataset

**File:**

```text
Crop Yeild Data.csv
```

The uploaded version contains **19,689 records and 13 columns**.

### Features

| Feature           | Description                     |
| ----------------- | ------------------------------- |
| `Crop`            | Crop name                       |
| `Crop_Year`       | Crop production year            |
| `Season`          | Agricultural season             |
| `State`           | State where crop was cultivated |
| `Area`            | Cultivated area                 |
| `Production`      | Crop production                 |
| `Annual_Rainfall` | Annual rainfall                 |
| `Fertilizer`      | Fertilizer usage                |
| `Pesticide`       | Pesticide usage                 |
| `Yield`           | Crop yield                      |
| `Avg_Temperature` | Average temperature             |
| `Max_Temperature` | Maximum temperature             |
| `Min_Temperature` | Minimum temperature             |

### Dataset Usage

This dataset is primarily used for the **rainfall regression task**.

The target variable is:

```text
Annual_Rainfall
```

`Annual_Rainfall` is removed from the input features because it is the prediction target.

The project also excludes `yield` from the regression features to avoid uncertainty/leakage concerns described in the notebook.

---

# 2. Crop Recommendation Dataset

**File:**

```text
Crop_recommendation.csv
```

Dataset size:

```text
2,200 records
8 columns
```

### Features

| Feature       | Description        |
| ------------- | ------------------ |
| `N`           | Nitrogen content   |
| `P`           | Phosphorus content |
| `K`           | Potassium content  |
| `temperature` | Temperature        |
| `humidity`    | Humidity           |
| `ph`          | Soil pH            |
| `rainfall`    | Rainfall           |
| `label`       | Crop label         |

### Usage

This dataset is used for **unsupervised clustering analysis**.

The clustering features are:

```text
N
P
K
temperature
humidity
ph
rainfall
```

The crop `label` is not used as a clustering feature.

---

# 3. Custom Crop Health Historical Dataset

**File:**

```text
Custom_Crops_yield_Historical_Dataset.csv
```

Dataset size:

```text
2,000 records
14 columns
```

### Features

| Feature             | Description              |
| ------------------- | ------------------------ |
| `record_id`         | Unique record identifier |
| `plant_type`        | Type of plant            |
| `habitat`           | Plant habitat            |
| `season`            | Season                   |
| `air_humidity_pct`  | Air humidity percentage  |
| `soil_humidity_pct` | Soil humidity percentage |
| `ethylene_ppm`      | Ethylene concentration   |
| `co2_ppm`           | CO₂ concentration        |
| `o2_ppm`            | O₂ concentration         |
| `ldr_on_plant`      | LDR measurement          |
| `leaf_color_index`  | Leaf color index         |
| `temperature_c`     | Temperature in Celsius   |
| `health_score`      | Plant health score       |
| `health_class`      | Plant health category    |

### Classification Target

```text
health_class
```

The project removes:

```text
record_id
health_score
```

from the classification input features.

`record_id` is an identifier and does not provide meaningful predictive information.

`health_score` is removed to avoid **target leakage**, because it is directly related to the plant health classification.

---

#  Machine Learning Methodology

## Phase 1 — Data Loading

The project loads all required datasets using Pandas.

```python
import pandas as pd

DATA_FILE_1 = pd.read_csv("Crop Yeild Data.csv")
DATA_FILE_2 = pd.read_csv("Custom_Crops_yield_Historical_Dataset.csv")
DATA_FILE_3 = pd.read_csv("Crop_recommendation.csv")
```

---

# Phase 2 — Data Auditing

The datasets are inspected for:

* Dataset shape
* Data types
* Missing values
* Duplicate records
* Descriptive statistics
* Target distributions

Example:

```python
DATA_FILE_1.info()

DATA_FILE_1.isnull().sum()

DATA_FILE_1.duplicated().sum()

DATA_FILE_1.describe()
```

This ensures that the datasets are understood before model training.

---

#  Rainfall Prediction Pipeline

## Problem Type

```text
Supervised Learning
        ↓
Regression
        ↓
Predict Annual_Rainfall
```

### Target

```text
Annual_Rainfall
```

### Feature Preparation

The project removes:

```python
regression_drop = ["Annual_Rainfall", "yield"]
```

High-cardinality identifiers such as:

```text
dist_code
state_code
```

are also removed when present.

---

#  Feature Engineering

An engineered feature is created:

```text
fertilizer_per_area
```

It is calculated as:

```text
fertilizer_per_area =
Fertilizer / Area
```

This provides a normalized representation of fertilizer usage relative to cultivated area.

---

#  Regression Preprocessing

The project separates numerical and categorical features.

### Numerical Features

Numerical preprocessing includes:

```text
Median Imputation
       ↓
StandardScaler
```

### Categorical Features

Categorical preprocessing includes:

```text
Most-Frequent Imputation
       ↓
One-Hot Encoding
```

The preprocessing is implemented using:

```python
ColumnTransformer
Pipeline
```

This prevents preprocessing inconsistencies between training and testing data.

---

#  Regression Algorithms

The project evaluates multiple regression algorithms, including:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Decision Tree Regressor
6. Random Forest Regressor
7. Gradient Boosting Regressor
8. Support Vector Regression (SVR)
9. K-Nearest Neighbors Regressor
10. Polynomial Regression

The models are trained using a common preprocessing pipeline.

---

#  Regression Evaluation

The following metrics are used:

### R² Score

Measures how much of the variation in the target variable is explained by the model.

Higher values generally indicate better explanatory performance.

### RMSE

Root Mean Squared Error measures the magnitude of prediction errors while giving greater weight to larger errors.

```text
RMSE = √Mean Squared Error
```

Lower values indicate smaller prediction errors.

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted rainfall.

Lower values indicate smaller errors.

---

#  Plant Health Classification

## Problem Type

```text
Supervised Learning
        ↓
Multi-Class Classification
        ↓
Predict health_class
```

### Target

```text
health_class
```

The classification features include environmental and plant-related variables such as:

```text
plant_type
habitat
season
air_humidity_pct
soil_humidity_pct
ethylene_ppm
co2_ppm
o2_ppm
ldr_on_plant
leaf_color_index
temperature_c
```

---

#  Classification Preprocessing

The project automatically separates:

```text
Numerical Features
Categorical Features
```

### Numerical Pipeline

```text
Missing Value Imputation
        ↓
Median
        ↓
StandardScaler
```

### Categorical Pipeline

```text
Missing Value Imputation
        ↓
Most Frequent
        ↓
One-Hot Encoding
```

The two pipelines are combined using:

```python
ColumnTransformer
```

---

#  Classification Algorithms

The project evaluates the following classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors Classifier
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Machine
6. Random Forest Classifier
7. AdaBoost Classifier
8. Gradient Boosting Classifier
9. Bagging Classifier
10. MLP Classifier

---

#  Classification Evaluation

Each model is evaluated using:

### Accuracy

Percentage of correctly classified observations.

### Precision

Measures how many observations predicted as a class actually belong to that class.

### Recall

Measures how many actual observations of a class are correctly identified.

### F1-Score

Harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall
          ───────────────────
          Precision + Recall
```

### ROC-AUC

The project calculates weighted multi-class ROC-AUC using the One-vs-Rest approach.

---

#  Confusion Matrix

A confusion matrix is generated for every classification model.

It provides a detailed view of:

```text
Actual Class
     ↓
Predicted Class
```

This helps identify which plant health categories are being confused by the models.

---

#  Hyperparameter Tuning

Three classification models are further tuned using `GridSearchCV`.

## Random Forest

The following parameters are searched:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

Search values include:

```text
n_estimators:
100, 200, 300

max_depth:
None, 5, 10, 20

min_samples_split:
2, 5

min_samples_leaf:
1, 2
```

---

# Support Vector Machine

The SVM search includes:

```text
C:
0.1, 1, 10, 100

kernel:
linear, rbf

gamma:
scale, auto
```

---

# MLP Classifier

The MLP search includes:

### Hidden Layers

```text
(50,)
(100,)
(100, 50)
```

### Activation Functions

```text
relu
tanh
```

### Learning Rate

```text
0.001
0.01
```

---

#  Cross-Validation

The hyperparameter tuning uses:

```text
5-Fold Cross-Validation
```

The optimization metric is:

```text
Weighted F1-Score
```

This is particularly useful for multi-class classification because it considers the performance across different classes while accounting for class support.

---

#  Crop Clustering

The project also performs unsupervised learning using:

```text
K-Means Clustering
Agglomerative Hierarchical Clustering
```

The following features are used:

```text
N
P
K
temperature
humidity
ph
rainfall
```

---

#  Feature Standardization

Before clustering, the features are standardized using:

```python
StandardScaler()
```

This is important because the features have different numerical ranges.

For example:

```text
N, P, K       → nutrient measurements
temperature   → temperature
humidity      → percentage
rainfall      → rainfall amount
```

Standardization prevents variables with larger numerical ranges from dominating distance-based clustering.

---

#  K-Means Clustering

The project evaluates different values of `K` using the **Elbow Method**.

The tested range is:

```text
K = 2 to 10
```

The project selects:

```text
K = 5
```

based on the observed elbow in the inertia curve.

The K-Means model is implemented with:

```python
KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
```

---

#  K-Means Evaluation

K-Means is evaluated using:

### Silhouette Score

Measures how well observations fit within their assigned cluster compared with neighboring clusters.

Higher values generally indicate better-defined clusters.

### Davies-Bouldin Index

Measures similarity between clusters.

Lower values generally indicate better-separated clusters.

### Calinski-Harabasz Index

Measures between-cluster separation relative to within-cluster dispersion.

Higher values generally indicate better-defined clustering.

---

#  Agglomerative Hierarchical Clustering

The project also implements:

```python
AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)
```

The Ward linkage method attempts to minimize within-cluster variance when merging observations.

---

#  Dendrogram

A hierarchical clustering dendrogram is generated using:

```python
linkage(
    X_dendrogram,
    method="ward"
)
```

The dendrogram visualizes how observations/groups are progressively merged based on distance.

---

#  Dimensionality Reduction

The project uses two dimensionality-reduction techniques for visualization.

## PCA

Principal Component Analysis is used to transform the standardized features into two dimensions.

```text
7-dimensional feature space
          ↓
        PCA
          ↓
2-dimensional visualization
```

PCA is used to visualize both K-Means and hierarchical clustering results.

---

#  t-SNE

t-SNE is also used to visualize local structures in the clustering results.

Configuration:

```text
Components = 2
Perplexity = 30
Random State = 42
```

t-SNE can reveal local groupings that may not be obvious in the PCA representation.

---

#  Cluster Profile Analysis

After K-Means clustering, the project calculates the mean values of:

```text
N
P
K
temperature
humidity
ph
rainfall
```

for every cluster.

This creates a **cluster profile** that can be used to understand the characteristics of different agricultural groups.

A heatmap is also generated to visualize these cluster profiles.

---

#  Exploratory Data Analysis

The project includes several EDA techniques.

## Numerical Distributions

Histograms are generated for numerical variables to understand:

* Distribution
* Skewness
* Spread
* Possible outliers

---

## Correlation Heatmap

Correlation heatmaps are used to examine relationships between numerical variables.

For the crop recommendation dataset, the clustering analysis identified a relatively strong positive relationship between:

```text
Phosphorus (P)
Potassium (K)
```

with a correlation of approximately:

```text
0.74
```

Other environmental and soil variables show varying degrees of linear relationship.

---

#  Rainfall Distribution

The rainfall dataset shows a positively/right-skewed annual rainfall distribution.

Most observations are concentrated around lower-to-moderate rainfall values, while a smaller number of observations have substantially larger rainfall values.

This indicates the presence of a long right tail and extreme observations.

---

#  Feature Relationships

The rainfall analysis includes scatter plots such as:

```text
Average Temperature vs Annual Rainfall
Area vs Annual Rainfall
```

These visualizations help investigate whether simple linear relationships exist between agricultural/environmental variables and rainfall.

---

#  Data Leakage Prevention

Data leakage is explicitly considered during model preparation.

For plant health classification:

```text
health_score
```

is removed from the input features because it can provide information directly related to the target `health_class`.

Similarly:

```text
record_id
```

is removed because it is only an identifier.

For rainfall prediction:

```text
Annual_Rainfall
```

is excluded from the input variables because it is the prediction target.

---

#  Train-Test Split

## Regression

The rainfall dataset is split using:

```text
80% Training
20% Testing
```

with:

```python
random_state = 42
```

---

## Classification

The plant health dataset also uses:

```text
80% Training
20% Testing
```

with stratification:

```python
stratify=y_cls
```

This helps maintain the class distribution between the training and testing datasets.

---

#  Reproducibility

The project uses:

```python
RANDOM_STATE = 42
```

for reproducible model training and data splitting wherever supported.

---

#  Project Structure

A recommended repository structure is:

```text
Machine-Learning-Based-Crop-Rainfall-Prediction-and-Plant-Health-Classification/
│
├── README.md
│
├── code_l2G4.ipynb
├── code_clustering_l2G5.ipynb
│
├── Crop Yeild Data.csv
├── Custom_Crops_yield_Historical_Dataset.csv
├── Crop_recommendation.csv
│
├── requirements.txt
│
├── models/
│   ├── rainfall_model.pkl
│   └── plant_health_model.pkl
│
├── results/
│   ├── regression_results.csv
│   ├── classification_results.csv
│   └── clustering_results.csv
│
└── images/
    ├── correlation_heatmap.png
    ├── rainfall_distribution.png
    ├── confusion_matrix.png
    ├── elbow_curve.png
    ├── pca_clusters.png
    ├── dendrogram.png
    └── tsne_clusters.png
```

> Dataset/model/output folders can be added to the repository depending on GitHub file-size requirements.

---

#  Technologies Used

## Programming Language

* Python

## Data Processing

* NumPy
* Pandas

## Data Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* SciPy

## Model Persistence

* Joblib

---

#  Python Libraries

The major libraries used in the project are:

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
joblib
```

---

#  Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Prudhvi770/Machine-Learning-Based-Crop-Rainfall-Prediction-and-Plant-Health-Classification.git
```

Move into the project directory:

```bash
cd Machine-Learning-Based-Crop-Rainfall-Prediction-and-Plant-Health-Classification
```

---

# 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib jupyter
```

Alternatively, if `requirements.txt` is included:

```bash
pip install -r requirements.txt
```

---

#  Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
code_l2G4.ipynb
```

for:

```text
Rainfall Regression
+
Plant Health Classification
```

Open:

```text
code_clustering_l2G5.ipynb
```

for:

```text
Crop Recommendation Dataset Analysis
+
K-Means Clustering
+
Hierarchical Clustering
+
PCA
+
t-SNE
```

---

#  Complete Workflow

```text
                 START
                   │
                   ▼
            Load Agricultural
                Datasets
                   │
                   ▼
          Data Quality Analysis
                   │
          ┌────────┼────────┐
          │        │        │
          ▼        ▼        ▼
       Missing  Duplicate  Data Types
       Values    Records
          │
          ▼
      Exploratory
       Analysis
          │
     ┌────┴───────────────┐
     │                    │
     ▼                    ▼
Rainfall Prediction   Plant Health
    Regression         Classification
     │                    │
     ▼                    ▼
Preprocessing         Preprocessing
     │                    │
     ▼                    ▼
Multiple Models      Multiple Models
     │                    │
     ▼                    ▼
Evaluation           Evaluation
     │                    │
     ▼                    ▼
Model Tuning         Model Tuning
     │                    │
     └──────────┬─────────┘
                │
                ▼
       Crop Feature Clustering
                │
        ┌───────┴────────┐
        ▼                ▼
     K-Means       Hierarchical
                    Clustering
        │                │
        └───────┬────────┘
                ▼
          PCA / t-SNE
          Visualization
                │
                ▼
          Final Analysis
```

---

# 📊 Model Comparison

The project generates comparison tables for both supervised learning tasks.

## Regression Comparison

The regression comparison contains:

```text
Model
R² Score
RMSE
MAE
```

Models are compared based on their predictive performance.

---

## Classification Comparison

The classification comparison contains:

```text
Model
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

The classification results are sorted by:

```text
F1-Score
```

for analysis.

---

#  Model Improvement

The project does not stop at baseline model training.

Selected models are further optimized using:

```text
GridSearchCV
```

with:

```text
5-fold cross-validation
```

The tuned models are then evaluated on the held-out test set.

For Random Forest, the project additionally compares:

```text
Before Tuning
        vs
After Tuning
```

for:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

---

#  Key Findings Supported by the Analysis

### Rainfall Prediction

The rainfall dataset contains substantial variation in:

* Annual rainfall
* Crop area
* Production
* Fertilizer usage
* Pesticide usage
* Yield
* Temperature

The exploratory analysis shows that several agricultural variables are strongly skewed, with some extreme observations.

---

### Plant Health Classification

The classification task uses both numerical environmental measurements and categorical plant/environment information.

The preprocessing pipeline handles both types of features through:

```text
Numerical → Imputation + Scaling

Categorical → Imputation + One-Hot Encoding
```

The project compares ten classification algorithms and subsequently performs hyperparameter tuning for selected models.

---

### Crop Clustering

The clustering analysis uses seven agricultural/environmental features.

The Elbow Method suggests:

```text
K = 5
```

as a suitable number of clusters in the implemented analysis.

PCA, hierarchical clustering, and t-SNE are used to investigate the resulting groups from different perspectives.

---

#  Limitations

The current implementation has several limitations:

* Rainfall prediction performance depends strongly on the available historical agricultural variables.
* Extreme rainfall observations may affect regression models.
* Dataset distributions may not represent all agricultural regions or future climate conditions.
* Clustering results depend on the selected features and scaling method.
* PCA and t-SNE are primarily visualization techniques and should not be interpreted as direct evidence of real-world agricultural categories.
* Plant health classification is based on the available dataset features and labels.
* Model performance on the provided datasets does not guarantee equivalent performance on unseen real-world agricultural data.
* Additional external validation would be required before applying the system to operational agricultural decision-making.

---

#  Future Enhancements

Possible future improvements include:

###  Rainfall Prediction

* Add historical weather time-series data.
* Include regional climate information.
* Incorporate satellite/weather API data.
* Test advanced ensemble methods.
* Perform time-series forecasting.
* Investigate outlier treatment and target transformation.

###  Plant Health Classification

* Add real plant images and computer vision models.
* Combine sensor data with image-based plant disease detection.
* Experiment with XGBoost and other gradient-boosting approaches.
* Apply class balancing where necessary.
* Deploy the classifier as a web application.

###  Agricultural Clustering

* Test DBSCAN and Gaussian Mixture Models.
* Compare different clustering initialization strategies.
* Analyze cluster stability.
* Add geographical information.
* Build crop-specific cluster profiles.

###  Deployment

The trained models can potentially be deployed using:

```text
Streamlit
Flask
FastAPI
```

A future application could provide:

```text
User Input
    ↓
Preprocessing
    ↓
Trained ML Model
    ↓
Prediction
    ↓
Visualization / Recommendation
```


# 📚 Main Machine Learning Concepts

This project demonstrates practical implementation of:

* Supervised Learning
* Regression
* Multi-Class Classification
* Unsupervised Learning
* K-Means Clustering
* Agglomerative Hierarchical Clustering
* Feature Engineering
* Feature Scaling
* One-Hot Encoding
* Missing Value Imputation
* Train-Test Split
* Cross-Validation
* Grid Search
* PCA
* t-SNE
* Confusion Matrix
* Model Evaluation
* Hyperparameter Optimization
* Data Visualization


