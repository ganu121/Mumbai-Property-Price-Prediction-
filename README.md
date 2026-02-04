# Mumbai Property Price Prediction  
*(Take-Home Assignment: Data Understanding, EDA & FastAPI)*

---

## 📌 Project Overview
This project focuses on **understanding Mumbai property price data**, performing **exploratory data analysis (EDA)**, building a **minimal machine learning model**, and exposing predictions through a **FastAPI application**.

The goal is **clarity of data understanding and reasoning**, rather than achieving high predictive accuracy.

---

## 📊 Dataset Understanding

- **Dataset**: `data/Assignment Data Scientist.xlsx`
- **Description**: Quarterly property price data across multiple localities in Mumbai (with some Hyderabad records removed).

### Dataset Shape
- **Total rows**: 8,404  
- **Total columns**: 9  
- **Data types**: All object types initially

### Original Columns
| Column Name     | Non-Null Count | Type   |
|-----------------|----------------|--------|
| Locality        | 8158           | Object |
| Quarter         | 8152           | Object |
| Price Range     | 7536           | Object |
| Average Price   | 7536           | Object |
| Q-o-Q           | 6941           | Object |
| Growth Type     | 197            | Object |
| City            | 8393           | Object |
| Type            | 8175           | Object |
| Unnamed: 8      | 198            | Object |

---

## 🧠 Column Explanation

- **Locality (categorical)**  
  Area within Mumbai city (e.g., Bhandup, Charkop Gaon).  
  *Hyderabad localities were dropped due to scope mismatch and inconsistent patterns.*

- **Quarter (categorical → engineered)**  
  Quarter + Year (e.g., *Apr–Jun 2024*).

- **Price Range (numerical → engineered)**  
  Minimum–maximum price per sqft (e.g., *16,266–29,066*).

- **Average Price (numerical)**  
  Average price per sqft for a given locality and quarter.

- **Q-o-Q (numerical)**  
  Quarter-on-quarter growth percentage.

- **Growth Type (numerical)**  
  Growth percentage (only available for Hyderabad → dropped).

- **City (categorical)**  
  City name (*Mumbai retained, Hyderabad dropped*).

- **Type (categorical)**  
  Property type (e.g., *Residential – Multi Storey Apartment*).

- **Unnamed: 8 (categorical)**  
  Hyderabad-specific data → dropped.

---

## 🧹 Data Cleaning & Preprocessing

### 1. Data Loading
- Loaded using `pandas.read_excel`
- Initial inspection to understand structure and missing values

### 2. Cleaning Steps
- **Missing Values**
  - Dropped rows with missing `Average Price` and `Price Range`
  - Imputed missing `Q-o-Q` values with the mean

- **Filtering**
  - Removed all Hyderabad city records
  - Dropped rows where `City` was null

- **Dropped Columns**
  - `Unnamed: 8`
  - `Growth Type`
  - `City`
  - `Type`
  - `Price Range`

- **Type Conversion**
  - Cleaned `Q-o-Q` and converted to float
  - Converted `Average Price` to integer

- **Feature Engineering**
  - Split `Quarter` into:
    - `Year` (integer)
    - `Quarter` (numerical: 1–4)

- **Categorical Encoding**
  - Encoded `Locality` using Label Encoding (`Locality_LabelEncoded`)

---

## 📌 Feature Classification

- **Numerical Features**
  - `Average Price`
  - `Q-o-Q`
  - `Year`
  - `Quarter`

- **Categorical Features**
  - `Locality`

---

## 🎯 Target Variable

- **Target**: `Average Price`  
- **Reason**: The objective is to predict the **average price per sqft** for a given Mumbai locality and time period.

---

## 📈 Exploratory Data Analysis (EDA)

### Summary
> *Univariate analysis revealed right-skewed property prices and predominantly low-to-moderate quarter-on-quarter growth, highlighting the presence of premium localities alongside stable market behavior. Bivariate analysis showed weak dependency between price levels and short-term growth, while temporal trends confirmed consistent long-term price appreciation with mild seasonal variations.*

### Key Observations
- **Average Price Distribution**
  - Right-skewed
  - Majority between ₹15,000–₹30,000
  - Long tail up to ₹70,000+ (premium localities)

- **Q-o-Q Growth Distribution**
  - Mostly between 0%–4%
  - Occasional spikes up to ~14%

- **Year-wise Trend**
  - Stable prices from 2018–2020
  - Clear upward trend from 2021 onward
  - Sharp rise in 2023–2024

- **Price vs Growth**
  - Weak correlation
  - Growth is locality- and period-specific

- **Seasonality**
  - Q2 (Apr–Jun) shows higher growth
  - Mild decline in Q3 and Q4

---

## 🏙️ Locality-Level Insights

- **Locality Pricing**
  - Premium areas consistently command higher prices
  - Emerging localities show stronger growth momentum

- **Area vs Price**
  - Absolute prices are driven more by locality than short-term growth

- **Outliers**
  - High-price localities (Malabar Hill, Napean Sea Road, Walkeshwar) are genuine
  - High Q-o-Q spikes often occur in redevelopment or mid-priced areas

---

## 🤖 Machine Learning Model

### Model Used
- **Linear Regression**
  - Chosen for simplicity and interpretability
  - Suitable for continuous target prediction

> *Note*: For better accuracy, ensemble models (e.g., Random Forest) would be more appropriate since locality strongly influences price and linear models fail to capture this effect.

### Training Setup
- Dropped `Locality_LabelEncoded` (categorical → poor fit for linear models)
- **Features (X)**:
  - `Year`
  - `Quarter`
  - `Q-o-Q`
- **Target (y)**:
  - `Average Price`
- **Train/Test Split**:
  - 80% / 20%
  - `random_state = 42`

### Evaluation Metric
- **R² Score**: `0.0431`

> Only ~4.3% variance explained, reinforcing that the model is intentionally minimal and explanatory.

---

## 🔮 Sample Prediction

### Input
```json
{
  "Year": [2025, 2025, 2026],
  "Quarter": [1, 4, 4],
  "Q-o-Q": [0.02, 0.03, 0.025]
}
