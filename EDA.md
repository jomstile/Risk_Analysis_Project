# Exploratory Data Analysis
## This section provides a preliminary dataset analysis, focusing on identifying missing values, validating data types, examining the overall schema, and performing initial descriptive statistics.

### We'll start by importing all necessary Python libraries and specifying the path to our dataset. To run locally remember to update the file path in the setup section to match your local environment.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/input/insurance-data-personal-auto-line-of-business/synthetic_insurance_data.csv")
```
### We use the .head() method to display the first five rows of this dataset, providing a quick glance at what the dataset holds.

| Age | Is_Senior | Marital_Status | Married_Premium_Discount | Prior_Insurance | Prior_Insurance_Premium_Adjustment | Claims_Frequency | Claims_Severity | Claims_Adjustment | Policy_Type | ... | Time_Since_First_Contact | Conversion_Status | Website_Visits | Inquiries | Quotes_Requested | Time_to_Conversion | Credit_Score | Premium_Adjustment_Credit | Region | Premium_Adjustment_Region |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 47 | 0 | Married | 86 | 1-5 years | 50 | 0 | Low | 0 | Full Coverage | ... | 100 | 5 | 1 | 2 | 99 | 704 | -50 | Suburban | 50 |
| 37 | 0 | Married | 86 | 1-5 years | 50 | 0 | Low | 0 | Full Coverage | ... | 220 | 5 | 1 | 2 | 99 | 726 | -50 | Urban | 10 |
| 49 | 0 | Married | 86 | 1-5 years | 50 | 1 | Low | 50 | Full Coverage | ... | 280 | 4 | 4 | 1 | 99 | 772 | -50 | Urban | 10 |
| 62 | 1 | Married | 86 | >5 years | 0 | 1 | Low | 50 | Full Coverage | ... | 416 | 2 | 2 | 2 | 80 | 9 | -50 | Urban | 10 |
| 36 | 0 | Single | 0 | >5 years | 0 | 2 | Low | 100 | Full Coverage | ... | 141 | 8 | 4 | 2 | 106 | 625 | 0 | Suburban | 50 |

## Next, we assess data completeness by using the .isnull().sum() sequence to identify the total number of null entries present in each column.
```python
df.isnull().sum()
```

| Age | 0 |
|-|-|
| Is_Senior | 0 |
| Marital_Status | 0 |
| Married_Premium_Discount | 0 |
| Prior_Insurance | 0 |
| Prior_Insurance_Premium_Adjustment | 0 |
| Claims_Frequency | 0 |
| Claims_Severity | 0 |
| Claims_Adjustment | 0 |
| Policy_Type | 0 |
| Policy_Adjustment | 0 |
| Premium_Amount | 0 |
| Safe_Driver_Discount | 0 |
| Multi_Policy_Discount | 0 |
| Bundling_Discount | 0 |
| Total_Discounts | 0 |
| Source_of_Lead | 0 |
| Time_Since_First_Contact | 0 |
| Conversion_Status | 0 |
| Website_Visits | 0 |
| Inquiries | 0 |
| Quotes_Requested | 0 |
| Time_to_Conversion | 0 |
| Credit_Score | 0 |
| Premium_Adjustment_Credit | 0 |
| Region | 0 |
| Premium_Adjustment_Region | 0 |
dtype:int64

### As a final step in our initial quality checks, we check for data redundancy by summing the output of the .duplicated() method to count all duplicate rows.
```python
df.duplicated().sum()
0
```

# The following Python function is designed to execute exploratory data analysis and is self-documented for ease of use and understanding.
```python
"""    
    This EDA process includes:
    1. Initial Data Inspection (info, descriptive statistics).
    2. Categorical Feature Summary (value counts and proportions).
    3. Target Variable Analysis ('Conversion_Status' distribution and overall rate).
    4. Univariate Analysis for key numerical columns (histograms and box plots).
    5. Bivariate Analysis against 'Conversion_Status' (numerical box plots, categorical bar plots).
    6. Correlation Analysis (heatmap for features relevant to conversion).

    Args:
        df (pd.DataFrame): The input DataFrame for analysis.

    Considerations:
       The data utilized is synthetic and was selected solely for its function as a training or demonstration medium. The focus of this exercise is the technical execution and the effective application of Python's data manipulation and analysis tools.
    """
def perform_eda(df):
    df.info()
    # Displays column data types, non-null values, and memory usage df.info()
    print(df.describe().T)
    # Generates descriptive statistics for numerical columns (mean, std, min, max, quartiles)
    print(df.describe().T)

    print("\n--- 1. Categorical Feature Summary ---")
    categorical_cols = df.select_dtypes(include=['object']).columns
    # Selects all columns with 'object' dtype
    for col in categorical_cols:
        print(f"\n{col} Value Counts:")
        # Display the normalized counts (proportions) for each unique category
        print(df[col].value_counts(normalize=True).round(3))

    print("\n--- 2. Target Variable Analysis ---")
    # Checks if the target variable is present in the DataFrame
    if 'Conversion_Status' in df.columns:
        # Calculates and prints the overall conversion rate
        conversion_rate = df['Conversion_Status'].mean()
        print(f"Overall Conversion Rate: {conversion_rate:.2%}")

        # Plots the distribution of the target variable
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Conversion_Status', data=df, palette='pastel')
        plt.title('Distribution of Conversion Status (0=No, 1=Yes)')
        plt.show()
    else:
        print("Erron in 2")

    print("\n--- 3. Univariate Analysis ---")
    # List of key numerical columns expected for distribution analysis
    numerical_cols = ['Age', 'Premium_Amount', 'Credit_Score', 'Total_Discounts', 'Time_Since_First_Contact']
    
    existing_numerical_cols = [col for col in numerical_cols if col in df.columns]

    if existing_numerical_cols:
        # Sets up subplots for combined histogram and box plot for each feature
        fig, axes = plt.subplots(len(existing_numerical_cols), 2, figsize=(15, 4 * len(existing_numerical_cols)))
        
        # Handle case where there is only one row of plots
        if len(existing_numerical_cols) == 1:
            axes = np.array([axes])

        for i, col in enumerate(existing_numerical_cols):
            # Histogram
            sns.histplot(df[col], kde=True, ax=axes[i, 0], color='skyblue')
            axes[i, 0].set_title(f'Histogram of {col}')
            
            # Box Plot
            sns.boxplot(x=df[col], ax=axes[i, 1], color='lightcoral')
            axes[i, 1].set_title(f'Box Plot of {col}')
        plt.tight_layout()
        plt.show()
    else:
        print("Error in 3")

    print("\n--- 4. Bivariate Analysis ---")
    if 'Conversion_Status' in df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        numerical_cols_bivariate = ['Age', 'Premium_Amount', 'Credit_Score', 'Website_Visits', 'Quotes_Requested', 'Time_to_Conversion']
        existing_num_bivar = [col for col in numerical_cols_bivariate if col in df.columns]
        
        for i, col in enumerate(existing_num_bivar):
            if i < len(axes): 
                sns.boxplot(x='Conversion_Status', y=col, data=df, ax=axes[i], palette='Set2')
                axes[i].set_title(f'{col} vs. Conversion Status')
            
        plt.tight_layout()
        plt.show()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        categorical_cols_bivariate = ['Marital_Status', 'Prior_Insurance', 'Claims_Severity', 'Policy_Type', 'Source_of_Lead', 'Region']
        existing_cat_bivar = [col for col in categorical_cols_bivariate if col in df.columns]

        for i, col in enumerate(existing_cat_bivar):
            if i < len(axes):
                # Calculate the mean conversion rate for each category
                conversion_pivot = df.groupby(col)['Conversion_Status'].mean().sort_values(ascending=False)
                sns.barplot(x=conversion_pivot.index, y=conversion_pivot.values, ax=axes[i], palette='viridis')
                axes[i].set_title(f'Conversion Rate by {col}')
                axes[i].set_ylabel('Conversion Rate')
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    print("\n--- 5. Correlation Analysis (Heatmap) ---")
    
    if 'Conversion_Status' in df.columns:
        # Selects only numerical columns for calculation
        numerical_df = df.select_dtypes(include=np.number)
        
        corr_matrix = numerical_df.corr()
        
        plt.figure(figsize=(12, 10))
        target_corr = corr_matrix['Conversion_Status'].sort_values(ascending=False)

        # Filters for features with absolute correlation > 0.1 with the target
        relevant_features = target_corr[abs(target_corr) > 0.1].index.tolist()
        if 'Conversion_Status' not in relevant_features:
            relevant_features.append('Conversion_Status') 

        sns.heatmap(
            numerical_df[relevant_features].corr(), 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            linewidths=.5, 
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Correlation Heatmap for Features Relevant to Conversion Status')
        plt.show()
    else:
        print("Error in 5")


if __name__ == '__main__':
    
    try:
        df = pd.read_csv('/input/insurance-data-personal-auto-line-of-business/synthetic_insurance_data.csv')
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns.")

        # Run the EDA function
        perform_eda(df)

    except FileNotFoundError:
        print("ERROR: File 'your_dataset_file.csv' not found.")
        print("Please modify the script to load your actual dataset file path and name.")
    except Exception as e:
        print(f"An error occurred during data loading or processing: {e}")
    
# The final call outside of the `if __name__ == '__main__':` block is redundant if the script is run directly.
# I've commented it out to prevent double-execution if the user is running the script as intended.
# perform_eda(df)
```

--- 1. Initial Data Structure and Summary ---

## To get an initial overview of the dataset's structure, we'll examine the output of the .info() function.

| # | Column | Non-Null Count | Dtype |
|---|---|---|---|
| 0 |Age| 10000 | int64 |
| 1 |Is_Senior| 10000 | int64 |
| 2 |Marital_Status| 10000 | object |
| 3 |Married_Premium_Discount| 10000 | int64 |
| 4 |Prior_Insurance| 10000 | object |
| 5 |Prior_Insurance_Premium_Adjustment| 10000 | int64 |
| 6 |Claims_Frequency| 10000 | int64 |
| 7 |Claims_Severity| 10000 | object |
| 8 |Claims_Adjustment| 10000 | int64 |
| 9 |Policy_Type| 10000 | object |
| 10 |Policy_Adjustment| 10000 | int64 |
| 11 |Premium_Amount| 10000 | int64 |
| 12 |Safe_Driver_Discount| 10000 | int64 |
| 13 |Multi_Policy_Discount | 10000 | int64 |
| 14 |Bundling_Discount | 10000 | int64 |
| 15 |Total_Discounts | 10000 | int64 |
| 16 |Source_of_Lead | 10000 | object |
| 17 |Time_Since_First_Contact | 10000 | int64 |
| 18 |Conversion_Status | 10000 | int64 |
| 19 |Website_Visits | 10000 | int64 |
| 20 |Inquiries | 10000 | int64 |
| 21 |Quotes_Requested | 10000 | int64 |
| 22 |Time_to_Conversion | 10000 | int64 |
| 23 |Credit_Score | 10000 | int64 |
| 24 |Premium_Adjustment_Credit | 10000 | int64 |
| 25 |Region | 10000 | object |
| 26 |Premium_Adjustment_Region | 10000 | int64 |
dtypes: int64(21), object(6)
memory usage: 2.1+ MB

## We will now inspect the summary statistics for the numerical features of this dataset by executing the .describe() function.

|count|mean|std|min|
|-|-|-|-|
|Age|10000.0|39.9917|14.050358|18.0|   
|Is_Senior|10000.0|0.1593|0.365974|0.0|   
|Married_Premium_Discount||10000.0|42.1314|42.993376|0.0|  
|Prior_Insurance_Premium_Adjustment|  10000.0    47.6250   34.354438     0.0   
|Claims_Frequency|10000.0|0.4972|0.716131|0.0|   
|Claims_Adjustment|10000.0|36.7800|65.910288|0.0|   
|Policy_Adjustment|10000.0 |-79.8600|97.955806|-200.0|   
|Premium_Amount|10000.0|2219.5714|148.521132|1800.0| 
|Safe_Driver_Discount|10000.0|0.1999|0.399945|0.0|
|Multi_Policy_Discount|10000.0|0.3051|0.460473|0.0|   
|Bundling_Discount|10000.0|0.0972|0.296245|0.0|   
|Total_Discounts|10000.0|30.1100|33.689782|0.0|   
|Time_Since_First_Contact|10000.0|15.4780|8.677975|1.0|  
|Conversion_Status|10000.0|0.5767|0.494107|0.0|   
|Website_Visits|10000.0|5.0229|2.238231|0.0|   
|Inquiries|10000.0|1.9969|1.415588|0.0|   
|Quotes_Requested|10000.0|1.9969|0.817409|1.0|   
|Time_to_Conversion|10000.0|46.0732|45.448450|1.0|  
|Credit_Score|10000.0|714.2534|49.749487|530.0|   
|Premium_Adjustment_Credit|10000.0|-11.3200|48.704156|-50.0|  
|Premium_Adjustment_Region|10000.0|64.3250|39.232618|0.0|   

||25%|50%|75%|max|
|-|-|-|-|-|
|Age|29.0|39.0|50.0|90.0|  
|Is_Senior|0.0|0.0|0.0|1.0|  
|Married_Premium_Discount|0.0|0.0|86.0|86.0|  
|Prior_Insurance_Premium_Adjustment|0.0|50.0|50.0|100.0|  
|Claims_Frequency|0.0|0.0|1.0|5.0|  
|Claims_Adjustment|0.0|0.0|50.0|800.0|  
|Policy_Adjustment|-200.0|0.0|0.0|0.0|  
|Premium_Amount|2100.0|2236.0|2336.0|2936.0|  
|Safe_Driver_Discount|0.0|0.0|0.0|1.0|  
|Multi_Policy_Discount|0.0|0.0|1.0|1.0|  
|Bundling_Discount|0.0|0.0|0.0|1.0|  
|Total_Discounts|0.0|50.0|50.0|150.0|  
|Time_Since_First_Contact|8.0|16.0|23.0|30.0| 
|Conversion_Status|0.0|1.0|1.0|1.0|  
|Website_Visits|3.0|5.0|6.0|16.0|  
|Inquiries|1.0|2.0|3.0|9.0|  
|Quotes_Requested|1.0|2.0|3.0|3.0|  
|Time_to_Conversion|6.0|12.0|99.0|99.0|  
|Credit_Score|681.0|715.0|748.0|850.0| 
|Premium_Adjustment_Credit|-50.0|-50.0|50.0|50.0|  
|Premium_Adjustment_Region|50.0|50.0|100.0|100.0| 

## This is the categorical feature summary, which aids in our understanding of the unique labels and potential groupings within the data.
### Marital_Status Value Counts:
|Marital_Status| |
|-|-|
|Married|0.490|
|Single|0.326|
|Widowed|0.092|
|Divorced|0.092|
Name: proportion, dtype: float64

### Prior_Insurance Value Counts:
|Prior_Insurance| |
|-|-|
|1-5 years|0.526|
|>5 years|0.261|
|<1 year|0.213|
Name: proportion, dtype: float64

### Claims_Severity Value Counts:
|Claims_Severity| |
|-|-|
|Low|0.700|
|Medium|0.204|
|High|0.096|
Name: proportion, dtype: float64

### Policy_Type Value Counts:
|Policy_Type| |
|-|-|
|Full Coverage|0.601|
|Liability-Only|0.399|
Name: proportion, dtype: float64

### Source_of_Lead Value Counts:
|Source_of_Lead| |
|-|-|
|Online|0.604|
|Agent|0.300|
|Referral|0.096|
Name: proportion, dtype: float64

### Region Value Counts:
|Region| |
|-|-|
|Urban|0.492|
|Suburban|0.302|
|Rural|0.206|
Name: proportion, dtype: float64

## This section is dedicated to understanding Conversion_Status, which serves as the section's target variable.
##### Overall Conversion Rate: 57.67%

<img width="549" height="393" alt="download" src="https://github.com/user-attachments/assets/3e2e0516-41d4-48d3-91a8-df71ac80595d" />

## --- 3. Univariate Analysis (Key Numerical Distributions) ---

<img width="1489" height="1989" alt="download" src="https://github.com/user-attachments/assets/dba9cb7a-f50a-455f-9257-67199f7868c2" />

## --- 4. Bivariate Analysis (Features vs. Conversion) ---

<img width="1789" height="990" alt="download" src="https://github.com/user-attachments/assets/480af328-86d2-4e58-9bdc-aa1cc8ffb407" />

<img width="1790" height="990" alt="download" src="https://github.com/user-attachments/assets/78aa3b0b-883c-4c9f-9c20-0c74bbcc927a" />

## --- 5. Correlation Analysis (Heatmap) ---

<img width="944" height="836" alt="download" src="https://github.com/user-attachments/assets/73f9f5f8-eb4b-4cf5-bd53-81220243cb9c" />






