# Netflix Data Analysis

A comprehensive data science project analyzing Netflix titles using exploratory data analysis (EDA), machine learning classification, regression modeling, and time series forecasting.

## Project Overview

This project provides a structured, progressive learning path through data science techniques using Netflix's content dataset. The analysis is organized into three difficulty levels, each building on fundamental concepts to explore more advanced methodologies.

## Dataset

**Source:** Netflix Titles Dataset

- **Format:** CSV
- **Location:** `data/netflix_titles.csv`
- **Key Columns:**
  - `show_id`: Unique identifier
  - `type`: Movie or TV Show
  - `title`: Content title
  - `director`: Director(s)
  - `cast`: Cast members
  - `country`: Production country
  - `date_added`: Date added to Netflix
  - `release_year`: Original release year
  - `rating`: Content rating (PG, PG-13, R, etc.)
  - `duration`: Duration in minutes or seasons
  - `listed_in`: Genre(s)
  - `description`: Plot description

## Project Structure

```
Netflix Data Analysis/
├── data/
│   └── netflix_titles.csv          # Netflix dataset
├── level1/
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   └── ml.ipynb                    # Basic ML Classification
├── level2/
│   └── tasks.ipynb                 # Regression & Clustering
├── level3/
│   └── tasks.ipynb                 # Time Series Analysis
└── README.md                         # This file
```

## Level-by-Level Breakdown

### Level 1: Fundamentals

#### **EDA Notebook** (`level1/eda.ipynb`)

Introduction to data exploration and visualization:

- Load and inspect Netflix dataset
- Handle missing values (director, cast, country)
- Data type conversions and feature engineering (date_added → year_added)
- Visualize distribution of Movies vs TV Shows
- Analyze content by genre, rating, and release year
- Summary statistics and insights

**Skills:** Data loading, cleaning, exploratory analysis, pandas, matplotlib/seaborn

#### **ML Notebook** (`level1/ml.ipynb`)

Introduction to machine learning classification:

- Prepare data for modeling (handle missing values, encoding)
- Feature engineering (extract numeric duration values)
- One-hot encode categorical variables (genres, ratings)
- Train-test split (80-20)
- Build and train Logistic Regression classifier
- Classify content as Movie vs TV Show
- Model evaluation

**Skills:** Feature engineering, categorical encoding, train-test splitting, scikit-learn, Logistic Regression

---

### Level 2: Intermediate

#### **Tasks Notebook** (`level2/tasks.ipynb`)

Advanced ML techniques with multiple modeling approaches:

**Task 1 - Regression Analysis:**

- Predict content duration using release year, rating, type, and genre
- Feature preprocessing and one-hot encoding
- Train-test split and model training
- Regression model evaluation (MSE, R² score)

**Task 2 - Classification with Advanced Models:**

- Multi-class or binary classification problems
- Random Forest Classifier for improved accuracy
- Comparison of model performance
- Feature importance analysis

**Task 3 - Clustering Analysis:**

- K-Means clustering to segment content
- Principal Component Analysis (PCA) for dimensionality reduction
- Visualization of clusters
- Interpretation of content segments

**Skills:** Regression, ensemble methods (Random Forest), clustering, PCA, feature importance, advanced model evaluation

---

### Level 3: Advanced

#### **Tasks Notebook** (`level3/tasks.ipynb`)

Time series analysis and forecasting:

- Parse and prepare time series data (date_added)
- Create monthly content addition time series
- Stationarity testing (Augmented Dickey-Fuller test)
- Time series decomposition (trend, seasonality, residuals)
- Moving average smoothing
- ARIMA model building and training
- Forecast future content additions
- Model validation and error metrics (MSE)

**Skills:** Time series analysis, stationarity testing, seasonal decomposition, ARIMA modeling, forecasting

---

## Installation & Setup

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- statsmodels

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### Running the Notebooks

1. Navigate to the project directory:

   ```bash
   cd "d:\projects\Netflix Data Analysis"
   ```

2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open notebooks in order:
   - Start with `level1/eda.ipynb`
   - Progress to `level1/ml.ipynb`
   - Move to `level2/tasks.ipynb`
   - Finish with `level3/tasks.ipynb`

## Key Techniques Demonstrated

| Technique                               | Level | Notebook |
| --------------------------------------- | ----- | -------- |
| Data Cleaning & Handling Missing Values | 1     | EDA      |
| Exploratory Data Analysis (EDA)         | 1     | EDA      |
| Categorical Encoding                    | 1     | ML       |
| Train-Test Splitting                    | 1     | ML       |
| Logistic Regression                     | 1     | ML       |
| Regression Modeling                     | 2     | Tasks    |
| Random Forest Classification            | 2     | Tasks    |
| K-Means Clustering                      | 2     | Tasks    |
| PCA & Dimensionality Reduction          | 2     | Tasks    |
| Time Series Decomposition               | 3     | Tasks    |
| ARIMA Forecasting                       | 3     | Tasks    |
| Stationarity Testing                    | 3     | Tasks    |

## Learning Outcomes

By completing this project, you will:

- ✅ Master data cleaning and exploratory analysis
- ✅ Understand feature engineering and preprocessing
- ✅ Build and evaluate supervised learning models
- ✅ Apply unsupervised learning techniques (clustering, PCA)
- ✅ Develop time series forecasting capabilities
- ✅ Interpret model results and extract business insights
- ✅ Follow best practices in data science workflows

## Key Insights to Explore

- How has Netflix content growth evolved over time?
- What are the most common content ratings?
- Can we accurately predict content duration from other features?
- How do genres and ratings relate to content type?
- What patterns exist in Netflix's content addition strategy?

## Technologies Used

- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn
- **Time Series:** statsmodels
- **Environment:** Jupyter Notebook, Python

## Notes

- Each notebook is self-contained and includes inline comments explaining each step
- Data preprocessing and feature engineering are embedded in each notebook for learning purposes
- Models are trained with specific random states for reproducibility (random_state=42)

## License

This educational project is provided as-is for learning purposes.

## Author

Netflix Data Analysis Project

---

**Start your journey:** Begin with `level1/eda.ipynb` to explore the Netflix dataset! 📊
