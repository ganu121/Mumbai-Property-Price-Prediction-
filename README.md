# Mumbai Property Price Prediction  
*(Take-Home Assignment: Data Understanding, EDA & FastAPI)*

---
## Data Understanding :
    Dataset: data\Assignment Data Scientist.xlsx
    The dataset contains quarterly property price information across Mumbai localities.

    RangeIndex: 8404 entries, 0 to 8403
    Data columns (total 9 columns):
    #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
    0   Locality       8158 non-null   object
    1   Quarter        8152 non-null   object
    2   Price Range    7536 non-null   object
    3   Average Price  7536 non-null   object
    4   Q-o-Q          6941 non-null   object
    5   Growth Type    197 non-null    object
    6   City           8393 non-null   object
    7   Type           8175 non-null   object
    8   Unnamed: 8     198 non-null    object
    dtypes: object(9)

### Explain: 
- What each column represents:
    Locality (categorical)                  - area in mumbai city and Hydrabad city. (ex: Bhandup, Charkop Gaon etc..) 
                                                (Droped Hydrabad city data: out of scope and had inconsistant data compared to mumbai data)
    Quarter (categorical -> engineered)     - quarter + Year. (ex : Apr-Jun 2024)
    Price Range (numerical > engineered)    - min-max price range per sqft. (ex: 16,266-29,066)
    Average Price (numerical)               - Average Price per sqft in mumbai localities for the perticular quarter.
    Q-o-Q (numerical)           - Quarter-on-quarter growth percentage
    Growth Type (numerical)     - percentage growth (But this column data is only for Hydrabad so dropping it)
    City (categorical)          - City name (Mumbai (and Hydrabad 204 rows which is dropped))
    Type (categorical)          - Type of property (ex: Residential - Multi Storey Apartment)
    Unnamed: 8 (categorical)    - only Hydrabad data regarding property type ( Dropped column )

### 1. Data Loading and Initial Inspection
    The data was loaded using `pandas.read_excel`, and an initial inspection was performed to understand its structure and identify missing values.

- Data quality issues or assumptions:
- Missing values, duplicates, outliers :
    ### 2. Data Cleaning and Preprocessing
    -   **Handling Missing Values**: Rows with missing 'Average Price' and 'Price Range' were dropped. Missing 'Q-o-Q' values were imputed with the mean.
    -   **Filtering Data**: Rows pertaining to 'Hyderabad' city were removed as the scope is limited to Mumbai. Rows with `City` being null were also dropped.
    -   **Dropping Irrelevant Columns**: Columns such as 'Unnamed: 8', 'Growth Type', 'City', 'Type', and 'Price Range' were dropped due to irrelevance, redundant information, or lack of significant data after filtering.
    -   **Data Type Conversion**: 'Q-o-Q' was cleaned by replacing non-numeric characters and converted to a float. 'Average Price' was converted to an integer.
    -   **Feature Engineering**: The 'Quarter' column was separated into 'Year' (integer) and 'Quarter' (numerical representation of quarter, e.g., 1 for Jan-Mar).
    -   **Categorical Encoding**: The 'Locality' column was converted into numerical labels using Label Encoding (`Locality_LabelEncoded`).

- Numerical vs categorical features 
    Numerical: Average Price, Q-o-Q, Year, Quarter
    Categorical: Locality

- Target variable chosen and why : 
    'Average Price' is chosen for target variale because the model needs to predict the average price per sqft of land in mumbai locality
 

2. Exploratory Data Analysis (EDA) 
 
- Univariate and bivariate analysis 

    #### Final Consolidated EDA Summary
    "Univariate analysis revealed right-skewed property prices and predominantly low-to-moderate quarter-on-quarter growth, highlighting the presence of premium localities alongside stable market behavior. Bivariate analysis showed weak dependency between price levels and short-term growth, while temporal trends confirmed consistent long-term price appreciation with mild seasonal variations."

    #### Key Observations:
    -   **Distribution of Average Property Prices**: Right-skewed, with most prices between ₹15,000 - ₹30,000, and a long tail extending to ₹70,000+ (indicating premium localities).
    -   **Distribution of Quarter-on-Quarter (Q-o-Q) Growth**: Mostly between 0% and 4%, with occasional spikes up to ~14%.
    -   **Year-wise Data Distribution**: Increasing records from 2018 to 2023, stabilizing in 2024, indicating good temporal coverage.
    -   **Average Price vs Q-o-Q Growth**: Widely scattered points with no clear linear pattern, suggesting growth is locality- and period-specific.
    -   **Year-wise Average Property Price Trend**: Stable from 2018–2020, with a clear upward trend from 2021 and a sharp increase in 2023–2024.
    -   **Quarter-wise Q-o-Q Growth Trend**: Peaks in Q2 (Apr–Jun) with slight decline in Q3 and Q4, showing mild seasonality.

- Insights on locality pricing, area vs price, suspicious values 
    #### Insights on Locality Pricing:
    Locality plays a critical role in price determination. Premium areas command consistently higher prices, while emerging localities exhibit stronger growth momentum, indicating potential future appreciation.

    #### Area (Locality) vs Price Relationship:
    The analysis suggests that absolute price levels are locality-driven rather than growth-driven, highlighting the importance of location-specific factors over short-term market movements.

    #### Suspicious / Noteworthy Values Analysis (Outliers):
    Outliers in 'Average Price' (e.g., Malabar Hill, Napean Sea Road, Walkeshwar) represent genuine premium segments, not data anomalies. High Q-o-Q growth values (e.g., Wadala West, Pant Nagar, Grant Road) are typically in mid-priced or redevelopment areas, influenced by infrastructure or lower base prices.


3. Minimal ML Model 

- Use one model 

    #### Model Used
    -   **Linear Regression**: Chosen for its simplicity, interpretability, and suitability for continuous target variable prediction (Average Price). 
    (for accurate predictions we need to use advanced ensemble models like random forest because locality is a catagorical column and it varies prices significantly which is not captured by linear models)

- Basic preprocessing 

    #### Training and Evaluation
    -   dropping column 'Locality_LabelEncoded' because its catagorical which is bad input for linear models
    -   **Features (X)**: 'Year', 'Quarter', 'Q-o-Q'
    -   **Target (y)**: 'Average Price'
    -   **Split**: Data split into 80% training and 20% testing sets (`random_state=42`).

- Show one metric (R² or RMSE) 

    -   **R-squared (R2)**: A measure of how well the model fits the data. The model achieved an R2 score of 0.0431. This indicates that only about 4.3% of the variance in property prices can be explained by the selected features.

- Predict price or price_per_sqft :
    input:
    {
        'Year': [2025, 2025, 2026],
        'Quarter': [1, 4, 4],
        'Q-o-Q': [0.02, 0.03, 0.025] # Example Q-o-Q growth values
    }

    output:
    Predicted Average Prices:
    [24717.04838386 26081.7613782  26625.98059881]


4. Minimal FastAPI 
Endpoint: 
POST  `/predict`


README Expectations 
- Dataset understanding 
- EDA insights 
- Target variable 
- Model and metric 
- Steps to run FastAPI 
---

## 4. FastAPI Application

A minimal FastAPI application is used to expose the trained model.

### Endpoint
**POST** `/predict`

### Input (via HTML form)
NOTE: inappropriate input suggested in assignment document. Going ahead with the suggested input + essential input fields from assignment and assuming the inputs for fields 'Year', 'Quarter', 'Q-o-Q' if not assigned manualy by the user.
`input:` 
- Locality
- Bedrooms
- Bathrooms
- Furnishing
- Area (sqft)  
- Year  
- Quarter  
- Q-o-Q growth  


### Output
- Predicted property price (numeric)

### Files
- `eda_and_model.ipynb` → Workflow steps (data cleaning, EDA, model training)
- `main.py` → FastAPI app  
- `linear_regression_model.pkl` → saved model  
- `templates/index.html` → simple webpage UI  
- `requirements.txt` - reqired libraries
- `README.md` - project details

### Run Instructions
```bash
pip install -r requirements.txt
uvicorn main:app --reload
- Access in Browser:
API → http://127.0.0.1:8000

