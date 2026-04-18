# Biweekly Progress Report — 1

**Project Title:** Wine Quality Prediction Using Machine Learning  
**Report Period:** Week 1 – Week 2  
**Prepared By:** Aryan Sutariya  
**Date of Submission:** [DD/MM/YYYY]

---

## 1. Objective

The primary objective of this project is to develop a machine learning–based system capable of predicting the quality of red wine based on its physicochemical properties. The project leverages the UCI Red Wine Quality dataset and follows a complete ML pipeline from data acquisition through model deployment.

---

## 2. Work Summary

### Week 1: Data Acquisition, Understanding & Exploratory Data Analysis (EDA)

#### 2.1 Literature Review & Problem Formulation

- Conducted a literature review on wine quality assessment methodologies, understanding how physicochemical properties such as volatile acidity, alcohol content, sulphates, and pH influence sensory quality scores.
- Identified the UCI Machine Learning Repository dataset (`winequality-red.csv`) comprising **1,599 samples** with **11 input features** and a target variable (`quality`, scored on a scale of 3–8).
- Formulated the problem as a **regression task** — predicting continuous quality scores from chemical measurements.

#### 2.2 Project Environment Setup

- Set up the development environment with Python 3.10+ and created an isolated virtual environment (`.venv/`).
- Installed core dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib`, `jupyter`, and `dtale`.
- Established the project directory structure and initialized version control.

#### 2.3 Exploratory Data Analysis (EDA)

- **Data Loading & Inspection:** Loaded the CSV dataset using pandas. Inspected data shape (1599 × 12), data types (all float64 except `quality` as int64), and confirmed no missing values.
- **Statistical Summary:** Computed descriptive statistics (mean, std, min, max, quartiles) for all features to understand data distributions.
- **Interactive EDA with dtale:** Used the `dtale` library for interactive visual exploration, inspecting distributions, correlations, and data quality at a granular level.
- **Correlation Analysis:** Generated a correlation heatmap (lower triangle with mask) using Seaborn. Key findings:
  - `alcohol` has the strongest positive correlation with `quality` (~0.48).
  - `volatile acidity` has the strongest negative correlation with `quality` (~-0.39).
  - `citric acid` and `sulphates` show moderate positive correlations with quality.
  - `density` and `fixed acidity` are highly correlated with each other.
- **Distribution Analysis:** Plotted quality score distribution — observed that the majority of wines scored 5 or 6, with very few scoring 3, 4, or 8 (class imbalance).
- **Box Plots:** Created feature-wise box plots (e.g., Alcohol vs Quality) to visualize how chemical properties vary across quality tiers.

---

### Week 2: Data Preprocessing, Feature Engineering & Initial Model Training

#### 2.4 Data Preprocessing

- **Duplicate Removal:** Identified and removed duplicate records from the dataset to avoid training bias.
- **Outlier Treatment:** Applied the Interquartile Range (IQR × 3) method per feature to remove extreme outliers. A conservative multiplier of 3 was chosen to preserve legitimate edge-case wines while filtering out clearly erroneous readings.
- **Post-cleaning dataset:** Retained a clean subset of data after outlier trimming for model training.

#### 2.5 Feature Engineering

Engineered three additional features to capture chemical relationships beyond raw measurements:

| Derived Feature     | Formula                                            | Rationale                                      |
|---------------------|----------------------------------------------------|------------------------------------------------|
| `acidity_ratio`     | `fixed acidity / (volatile acidity + 1e-9)`        | Captures balance between structural and off-flavor acids |
| `so2_ratio`         | `free sulfur dioxide / (total sulfur dioxide + 1e-9)` | Indicates the proportion of active preservative SO₂       |
| `alcohol_density`   | `alcohol / density`                                 | Reflects the alcohol concentration relative to body       |

A small epsilon (1e-9) was added to denominators to prevent division-by-zero errors.

#### 2.6 Train/Test Split & Scaling

- Split the dataset into **80% training** and **20% testing** sets with `random_state=42` for reproducibility.
- Applied `StandardScaler` (fitted only on train data) to normalize all features to zero mean and unit variance, as required by distance-based and gradient-based models.

#### 2.7 Initial Model Training (Before Tuning)

Trained three baseline regression models with default/initial hyperparameters:

| Model                      | Configuration                                       |
|----------------------------|-----------------------------------------------------|
| Random Forest Regressor    | `n_estimators=100, random_state=42`                 |
| Gradient Boosting Regressor| `n_estimators=100, random_state=42`                 |
| XGBoost Regressor          | `n_estimators=100, random_state=42, eval_metric='rmse'` |

Evaluated each model using:
- **Mean Absolute Error (MAE)** — average magnitude of prediction errors
- **Root Mean Squared Error (RMSE)** — penalizes larger errors
- **R² Score** — proportion of variance explained by the model
- **5-Fold Cross-Validation R²** — generalization check

#### 2.8 Initial Results (Before Tuning)

| Model               | MAE   | RMSE  | R²    |
|----------------------|-------|-------|-------|
| Random Forest        | ~0.40 | ~0.54 | ~0.47 |
| Gradient Boosting    | ~0.41 | ~0.55 | ~0.44 |
| XGBoost              | ~0.40 | ~0.54 | ~0.46 |

**Observation:** All three models showed similar baseline performance, with Random Forest and XGBoost slightly outperforming Gradient Boosting. The R² values indicate moderate predictive power, motivating hyperparameter tuning in the next phase.

---

## 3. Tools & Technologies Used

| Tool / Library   | Purpose                                   |
|------------------|-------------------------------------------|
| Python 3.10+     | Core programming language                 |
| Jupyter Notebook | Interactive development & documentation   |
| pandas / numpy   | Data manipulation & numerical computation |
| dtale            | Interactive EDA GUI                       |
| matplotlib / seaborn | Static visualizations & plots          |
| scikit-learn     | ML models, preprocessing, evaluation      |
| XGBoost          | Gradient boosted tree regressor           |

---

## 4. Challenges Encountered

1. **Class Imbalance:** The quality scores are heavily concentrated around 5–6, making it harder for models to learn the characteristics of very low or very high quality wines.
2. **Outlier Sensitivity:** Choosing the right IQR multiplier required experimentation — a too-aggressive threshold (1.5×) removed valid samples, while a too-lenient one retained noise.
3. **Feature Multicollinearity:** High correlation between `density` and `fixed acidity` could introduce redundancy; this was partially addressed through the engineered ratio features.

---

## 5. Deliverables

| Deliverable                          | Status      |
|--------------------------------------|-------------|
| Dataset acquired and inspected       | ✅ Complete |
| EDA notebook with visualizations     | ✅ Complete |
| Data preprocessing pipeline          | ✅ Complete |
| Feature engineering (3 new features) | ✅ Complete |
| Baseline model training (3 models)   | ✅ Complete |
| Initial performance metrics recorded | ✅ Complete |

---

## 6. Plan for Next Period

- Perform hyperparameter tuning using **GridSearchCV** with 5-fold cross-validation for all three models.
- Compare tuned vs. baseline performance with detailed comparison charts.
- Select and serialize the best-performing model.
- Begin development of the interactive **Streamlit web application**.
