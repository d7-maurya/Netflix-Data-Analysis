# Netflix Data Analysis 🎬

> A comprehensive data science project analyzing Netflix titles using exploratory data analysis (EDA), machine learning classification, regression modeling, and time series forecasting.

## Project Overview

This project provides a **structured, progressive learning path** through essential data science techniques using Netflix's real-world content dataset. The analysis is organized into **three difficulty levels**, each building on fundamental concepts to explore increasingly advanced methodologies and machine learning algorithms.

Whether you're a beginner learning data science fundamentals or an intermediate practitioner honing your skills, this project offers hands-on experience with real-world datasets and industry-standard techniques.

## Dataset 📊

**Source:** Netflix Titles Dataset (Available on Kaggle)

| Property           | Value                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------- |
| **Format**         | CSV                                                                                       |
| **Location**       | `data/netflix_titles.csv`                                                                 |
| **Total Records**  | 8,000+ titles                                                                             |
| **Key Attributes** | Type, Title, Director, Cast, Country, Release Year, Rating, Duration, Genres, Description |

### Key Columns

- **`show_id`** - Unique identifier for each title
- **`type`** - Content type: Movie or TV Show
- **`title`** - Official content title
- **`director`** - Director(s) of the content
- **`cast`** - Lead cast members
- **`country`** - Production country/countries
- **`date_added`** - Date content was added to Netflix
- **`release_year`** - Original release year
- **`rating`** - Content rating (G, PG, PG-13, R, TV-14, TV-MA, etc.)
- **`duration`** - Duration in minutes (movies) or number of seasons (TV shows)
- **`listed_in`** - Genre(s) assigned to the content
- **`description`** - Plot summary or content description

## Project Structure 📁

```
Netflix Data Analysis/
│
├── 📄 README.md                              # Project documentation (this file)
├── 📄 data/
│   └── netflix_titles.csv                    # Netflix dataset (8,000+ titles)
│
├── 📁 level1/  [Fundamentals]
│   ├── 01_EDA_Data_Exploration.ipynb         # Data cleaning & visualization
│   └── 02_ML_Classification.ipynb            # Logistic Regression classifier
│
├── 📁 level2/  [Intermediate]
│   └── 03_Regression_and_Clustering.ipynb    # Regression, ensemble methods & clustering
│
└── 📁 level3/  [Advanced]
    └── 04_ARIMA_Time_Series_Forecasting.ipynb  # Time series analysis & forecasting
```

## Level-by-Level Breakdown 📚

### Level 1: Fundamentals 🌱

Master the essentials of data science with comprehensive EDA and your first machine learning model.

#### **Notebook 1: EDA Data Exploration**

[`level1/01_EDA_Data_Exploration.ipynb`](level1/01_EDA_Data_Exploration.ipynb)

Explore, understand, and visualize the Netflix dataset:

- ✓ Load and inspect the dataset structure
- ✓ Handle missing values in director, cast, and country columns
- ✓ Feature engineering (convert date_added to year_added)
- ✓ Visualize content distribution: Movies vs TV Shows
- ✓ Analyze trends by genre, rating, and release year
- ✓ Generate summary statistics and identify key patterns
- ✓ Create meaningful visualizations with matplotlib/seaborn

**Core Skills Covered:**

- Data loading and inspection (pandas)
- Missing value analysis and handling strategies
- Data type conversions and feature engineering
- Exploratory data analysis (EDA)
- Data visualization best practices
- Statistical summary and interpretation

---

#### **Notebook 2: ML Classification**

[`level1/02_ML_Classification.ipynb`](level1/02_ML_Classification.ipynb)

Build your first machine learning model to classify content type:

- ✓ Prepare data for modeling (encode categorical variables)
- ✓ Feature engineering (extract numeric features from duration)
- ✓ One-hot encode categorical features (genres, ratings)
- ✓ Perform train-test split (80-20 ratio)
- ✓ Train a Logistic Regression classifier
- ✓ Classify content as Movie vs TV Show
- ✓ Evaluate model performance (accuracy, precision, recall)

**Core Skills Covered:**

- Data preprocessing and feature engineering
- Categorical encoding techniques
- Train-test splitting and cross-validation concepts
- Logistic Regression classifier
- Model evaluation metrics
- scikit-learn fundamentals

---

### Level 2: Intermediate 🚀

Advance your skills with multiple supervised and unsupervised learning techniques.

#### **Notebook 3: Regression and Clustering**

[`level2/03_Regression_and_Clustering.ipynb`](level2/03_Regression_and_Clustering.ipynb)

Explore multiple machine learning algorithms in one comprehensive notebook:

**Task 1: Regression Analysis**

- Predict content duration using features like release year, rating, type, and genre
- Preprocess features and handle categorical variables
- Train and evaluate regression models
- Calculate performance metrics (Mean Squared Error, R² Score)
- Interpret regression results

**Task 2: Advanced Classification**

- Implement multi-class classification problems
- Train a Random Forest Classifier for improved accuracy
- Compare model performance with baseline models
- Analyze feature importance to understand model decisions
- Apply ensemble learning concepts

**Task 3: Clustering Analysis**

- Segment Netflix content using K-Means clustering
- Apply Principal Component Analysis (PCA) for dimensionality reduction
- Visualize clusters in 2D/3D space
- Interpret and characterize different content segments
- Determine optimal number of clusters

**Core Skills Covered:**

- Regression modeling and evaluation
- Ensemble methods (Random Forest)
- Unsupervised learning and clustering
- Principal Component Analysis (PCA)
- Feature importance analysis
- Advanced model evaluation techniques

---

### Level 3: Advanced 🔬

Master time series analysis and forecasting with state-of-the-art techniques.

#### **Notebook 4: ARIMA Time Series Forecasting**

[`level3/04_ARIMA_Time_Series_Forecasting.ipynb`](level3/04_ARIMA_Time_Series_Forecasting.ipynb)

Analyze temporal patterns and forecast future trends:

- ✓ Parse and prepare time series data (date_added column)
- ✓ Aggregate data into monthly time series
- ✓ Perform stationarity testing (Augmented Dickey-Fuller test)
- ✓ Decompose time series into trend, seasonality, and residuals
- ✓ Apply moving average smoothing techniques
- ✓ Build and train ARIMA models with appropriate parameters
- ✓ Generate forecasts for future content additions
- ✓ Validate model performance using error metrics (MSE, MAE)

**Core Skills Covered:**

- Time series data preparation and exploration
- Stationarity testing and differencing
- Time series decomposition (trend, seasonality, residuals)
- ARIMA (AutoRegressive Integrated Moving Average) modeling
- Parameter selection (p, d, q values)
- Time series forecasting and validation
- statsmodels library expertise

---

## Installation & Setup 🔧

### Prerequisites

- **Python** 3.8 or higher
- **Jupyter Notebook** or **JupyterLab**
- Package manager: `pip` or `conda`

### Required Libraries

| Library          | Purpose                                       |
| ---------------- | --------------------------------------------- |
| **pandas**       | Data manipulation and analysis                |
| **numpy**        | Numerical computing                           |
| **matplotlib**   | Static and interactive visualizations         |
| **seaborn**      | Statistical data visualization                |
| **scikit-learn** | Machine learning algorithms                   |
| **statsmodels**  | Statistical modeling and time series analysis |
| **scipy**        | Scientific computing utilities                |

### Installation Steps

1. **Clone or download the project:**

   ```bash
   cd "your project location"
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or: source venv/bin/activate  # On macOS/Linux
   ```

3. **Install required packages:**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
   ```

   Or install all at once:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks

1. **Start Jupyter:**

   ```bash
   jupyter notebook
   ```

2. **Follow this learning path:**
   - 🌱 Start with [`level1/01_EDA_Data_Exploration.ipynb`](level1/01_EDA_Data_Exploration.ipynb)
   - 🌱 Then [`level1/02_ML_Classification.ipynb`](level1/02_ML_Classification.ipynb)
   - 🚀 Progress to [`level2/03_Regression_and_Clustering.ipynb`](level2/03_Regression_and_Clustering.ipynb)
   - 🔬 Complete with [`level3/04_ARIMA_Time_Series_Forecasting.ipynb`](level3/04_ARIMA_Time_Series_Forecasting.ipynb)

3. **Execute cells in order** and read the explanatory comments throughout each notebook.

## Techniques & Algorithms 🛠️

### Comprehensive Technique Matrix

| Technique                           | Difficulty | Notebook            | Description                                              |
| ----------------------------------- | ---------- | ------------------- | -------------------------------------------------------- |
| **Data Cleaning**                   | ⭐         | EDA                 | Handle missing values, data validation, type conversion  |
| **Exploratory Data Analysis (EDA)** | ⭐         | EDA                 | Data profiling, visualization, pattern discovery         |
| **Statistical Analysis**            | ⭐         | EDA                 | Summary statistics, distributions, correlations          |
| **Feature Engineering**             | ⭐⭐       | EDA, Classification | Create meaningful features from raw data                 |
| **Categorical Encoding**            | ⭐⭐       | Classification      | One-hot encoding, label encoding, ordinal encoding       |
| **Train-Test Splitting**            | ⭐⭐       | Classification      | Data partitioning for model validation                   |
| **Logistic Regression**             | ⭐⭐       | Classification      | Binary classification algorithm                          |
| **Regression Modeling**             | ⭐⭐       | Regression          | Linear & polynomial regression for continuous prediction |
| **Random Forest Classification**    | ⭐⭐⭐     | Regression          | Ensemble learning for improved accuracy                  |
| **K-Means Clustering**              | ⭐⭐⭐     | Clustering          | Unsupervised segmentation algorithm                      |
| **PCA & Dimensionality Reduction**  | ⭐⭐⭐     | Clustering          | Reduce features while preserving variance                |
| **Time Series Decomposition**       | ⭐⭐⭐     | ARIMA               | Separate trend, seasonality, and residuals               |
| **Stationarity Testing (ADF)**      | ⭐⭐⭐     | ARIMA               | Test and transform time series for ARIMA                 |
| **ARIMA Forecasting**               | ⭐⭐⭐     | ARIMA               | AutoRegressive Integrated Moving Average modeling        |
| **Model Evaluation Metrics**        | ⭐⭐⭐     | All                 | Accuracy, precision, recall, F1, MSE, MAE, RMSE          |

## Learning Outcomes ✅

By completing this project, you will master:

### Foundational Skills

- ✅ Data cleaning and handling missing values
- ✅ Exploratory data analysis (EDA) techniques
- ✅ Data visualization best practices
- ✅ Statistical analysis and interpretation

### Machine Learning Skills

- ✅ Feature engineering and preprocessing
- ✅ Supervised learning (classification & regression)
- ✅ Unsupervised learning (clustering, dimensionality reduction)
- ✅ Ensemble methods for improved model performance
- ✅ Model evaluation and validation strategies

### Advanced Analytics Skills

- ✅ Time series analysis and decomposition
- ✅ Forecasting with ARIMA models
- ✅ Parameter tuning and hyperparameter optimization
- ✅ Interpretation of machine learning results

### Professional Development

- ✅ End-to-end machine learning workflows
- ✅ Best practices in data science projects
- ✅ Reproducible analysis with random state management
- ✅ Clear code documentation and commenting

## Key Questions to Explore 🔍

This project helps answer important business and analytical questions:

1. **Content Growth & Strategy**
   - How has Netflix content volume evolved over time?
   - What is the trend in content additions (increasing, decreasing, seasonal)?
   - When did Netflix add the most content to its platform?

2. **Content Composition**
   - What proportion of Netflix's library is movies vs TV shows?
   - What are the most common content ratings on the platform?
   - How has the rating distribution changed over time?

3. **Content Characteristics**
   - Can we predict content duration from other features?
   - What features are most predictive of content type?
   - How do genres and ratings relate to content type?

4. **Content Segmentation**
   - Can we meaningfully segment Netflix content into clusters?
   - What characteristics define each cluster?
   - How do audience-facing genres relate to algorithmic clusters?

5. **Future Forecasting**
   - Can we predict Netflix's future content additions?
   - Are there seasonal patterns in content releases?
   - What are the expected trends for the next quarter/year?

## Technologies Used 💻

### Data Science Stack

| Category             | Tools                             |
| -------------------- | --------------------------------- |
| **Data Processing**  | Pandas, NumPy, SciPy              |
| **Machine Learning** | scikit-learn                      |
| **Time Series**      | statsmodels                       |
| **Visualization**    | Matplotlib, Seaborn               |
| **Environment**      | Jupyter Notebook/Lab, Python 3.8+ |

### Key Libraries & Versions

```
pandas >= 1.3.0        # Data manipulation and analysis
numpy >= 1.20.0        # Numerical computing
matplotlib >= 3.4.0    # Static and interactive plotting
seaborn >= 0.11.0      # Statistical data visualization
scikit-learn >= 0.24.0 # Machine learning algorithms
statsmodels >= 0.13.0  # Statistical models and tests
scipy >= 1.7.0         # Scientific computing
```

## Project Notes 📝

### Code Quality & Best Practices

- ✓ Each notebook is **self-contained** with clear, sequential workflow
- ✓ **Inline comments** explain each step and why it's performed
- ✓ **Markdown cells** provide context and learning objectives
- ✓ Data preprocessing and feature engineering embedded for educational clarity
- ✓ Models trained with **`random_state=42`** for reproducibility across runs
- ✓ Functions are modular and can be adapted for similar datasets

### Dataset Notes

- The dataset contains some missing values (director, cast, country) which are handled appropriately in each notebook
- Duration is formatted differently for movies (minutes) vs TV shows (seasons)
- Content may have multiple genres separated by commas

### Reproducibility

All notebooks use fixed random seeds to ensure consistent results across multiple runs:

```python
np.random.seed(42)
random.seed(42)
from sklearn.model_selection import train_test_split
train_test_split(..., random_state=42)
```

### Recommended Environment

- **OS:** Windows, macOS, or Linux
- **Python:** 3.8, 3.9, or 3.10
- **RAM:** 4GB minimum (8GB+ recommended)
- **Disk:** 500MB free space for dataset and notebooks

## Troubleshooting 🛠️

### Common Issues & Solutions

| Issue                               | Solution                                               |
| ----------------------------------- | ------------------------------------------------------ |
| **ModuleNotFoundError**             | Install missing packages: `pip install <package_name>` |
| **Jupyter not found**               | Install Jupyter: `pip install jupyter`                 |
| **Dataset file not found**          | Ensure `netflix_titles.csv` is in the `data/` folder   |
| **Memory errors on large datasets** | Reduce data or increase available RAM                  |
| **Plot display issues**             | Add `%matplotlib inline` at the start of notebooks     |
| **Random state not working**        | Ensure `random_state` parameter is set in all models   |

### Helpful Tips

- **Progressive Execution:** Always run notebook cells in order from top to bottom
- **Kernel Reset:** If you encounter errors, try "Kernel → Restart & Clear Output"
- **Variable Inspection:** Use `print()` or `df.head()` to inspect intermediate results
- **Help Documentation:** Use `help(function_name)` or `function_name?` in Jupyter cells

---

## Quick Start Guide 🚀

**Fastest way to get started:**

```bash
# 1. Navigate to project
cd "your project location"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

# 4. Launch Jupyter
jupyter notebook

# 5. Open level1/01_EDA_Data_Exploration.ipynb and start learning!
```

---

## License & Attribution 📄

This educational project is provided **as-is** for learning and educational purposes. The Netflix dataset used is sourced from publicly available data and is used here for educational demonstration.

### Attribution

- **Project:** Netflix Data Analysis - Data Science Learning Path
- **Dataset:** Netflix Titles (Kaggle)
- **Purpose:** Educational - Data Science & Machine Learning Training

### Fair Use

This project demonstrates data science techniques on a real-world dataset. For any production use or redistribution of the dataset, please refer to the original dataset's terms of use on Kaggle.

---

## Contributing 🤝

Have ideas for improvements? Found a bug? Feel free to:

- Report issues with notebook execution
- Suggest additional analysis or techniques
- Improve documentation or explanations
- Optimize code for clarity and performance

---

## Connect & Learn More 📖

### Next Steps After Completing This Project

1. **Apply to new datasets** - Use the same techniques on Kaggle datasets
2. **Explore advanced topics** - Deep learning, neural networks, NLP
3. **Build portfolio projects** - Create your own end-to-end analyses
4. **Join communities** - Engage with other data scientists on Kaggle, GitHub

### Related Resources

- [Kaggle Datasets](https://www.kaggle.com/datasets) - Find more datasets
- [Scikit-learn Documentation](https://scikit-learn.org/) - ML library reference
- [Statsmodels Guide](https://www.statsmodels.org/) - Time series guide
- [Matplotlib & Seaborn](https://matplotlib.org/) - Visualization tutorials

---

## Project Statistics 📊

- **Total Notebooks:** 4
- **Total Learning Levels:** 3
- **Estimated Time to Complete:** 10-15 hours
- **Difficulty Progression:** Beginner → Advanced
- **Hands-on Coding:** 100%

---

## Start Your Data Science Journey! 🎬

**Begin here:** Open [`level1/01_EDA_Data_Exploration.ipynb`](level1/01_EDA_Data_Exploration.ipynb) in Jupyter and start exploring the Netflix dataset!

> "The best way to learn data science is by doing. This project gives you the real-world experience you need." - Happy Learning! 🚀
