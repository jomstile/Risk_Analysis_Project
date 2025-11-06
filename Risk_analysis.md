```python
def preprocess_claims_features(df):
    if 'Claims_Frequency' not in df.columns:
        print("Error: 'Claims_Frequency' column is required for claims grouping.")
        return df
    claims_order = ['0 Claims', '1 Claim', '2+ Claims']

    df['Claims_Group'] = df['Claims_Frequency'].apply(lambda x:
        '0 Claims' if x == 0
        else ('1 Claim' if x == 1
        else '2+ Claims')
    ).astype(pd.CategoricalDtype(categories=claims_order, ordered=True))

    return df

def analyze_claims_risk_drivers(df):
    
    print("\n--- Starting Claims-Specific Risk Driver Analysis ---")

    if 'Claims_Group' not in df.columns or 'Claims_Severity' not in df.columns:
        print("Required claims columns (Claims_Group, Claims_Severity) are missing. Skipping analysis.")
        return

    print("\n1. Analysis of Claims Groups by Key Numerical Features (Means):")

    numerical_drivers = ['Age', 'Credit_Score', 'Premium_Amount', 'Time_Since_First_Contact']
    
    numerical_drivers = [col for col in numerical_drivers if col in df.columns]

    if numerical_drivers:
        claims_group_means = df.groupby('Claims_Group')[numerical_drivers].mean().sort_index()
        print("\nMean of Numerical Drivers by Claims Frequency Group:")
        print(claims_group_means.to_string(float_format="{:.1f}".format))

        claims_severity_means = df.groupby('Claims_Severity')[numerical_drivers].mean().sort_index()
        print("\nMean of Numerical Drivers by Claims Severity:")
        print(claims_severity_means.to_string(float_format="{:.1f}".format))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, col in enumerate(numerical_drivers):
            if i < len(axes):
                sns.barplot(x='Claims_Group', y=col, data=df, ax=axes[i], palette='Reds_d', order=df['Claims_Group'].cat.categories)
                axes[i].set_title(f'Average {col} by Claims Frequency Group')
                axes[i].set_ylabel(f'Average {col}')

        plt.tight_layout()
        plt.show()
    else:
        print("Warning: None of the key numerical columns were found for analysis.")


    print("\n2. Analysis of Claims Groups by Key Categorical Features (Proportions):")

    categorical_drivers = ['Marital_Status', 'Policy_Type', 'Region']
    
    categorical_drivers = [col for col in categorical_drivers if col in df.columns]

    for driver in categorical_drivers:
        claim_vs_driver_counts = df.groupby(['Claims_Group', driver]).size().unstack(fill_value=0)
        
        claim_vs_driver_proportions = claim_vs_driver_counts.div(claim_vs_driver_counts.sum(axis=1), axis=0) * 100

        print(f"\nDistribution of {driver} within each Claims Frequency Group (%):")
        print(claim_vs_driver_proportions.to_string(float_format="{:.1f}%".format))
        
        ax = claim_vs_driver_proportions.plot(
            kind='bar',
            stacked=True,
            figsize=(10, 6),
            cmap='tab20',
            title=f'Distribution of {driver} across Claims Frequency Groups'
        )
        plt.ylabel("Percentage (%)")
        plt.xlabel("Claims Frequency Group")
        plt.xticks(rotation=0)
        ax.legend(title=driver, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


    print("\n3. Summary: Claims Severity Distribution by Region:")
    if 'Region' in df.columns and 'Claims_Severity' in df.columns:
        severity_by_region = pd.crosstab(df['Region'], df['Claims_Severity'], normalize='index') * 100
        print(severity_by_region.to_string(float_format="{:.1f}%".format))

        ax = severity_by_region.plot(
            kind='bar',
            stacked=True,
            figsize=(10, 6),
            cmap='magma',
            title='Claims Severity Distribution by Region'
        )
        plt.ylabel("Percentage of Claims Severity within Region (%)")
        plt.xlabel("Region")
        plt.xticks(rotation=45)
        ax.legend(title='Claims Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

preprocess_claims_features(df)
analyze_claims_risk_drivers(df)
```

## 1. Analysis of Claims Groups by Key Numerical Features (Means):

#### Mean of Numerical Drivers by Claims Frequency Group:
|Claims_Group|Age|Credit_Score|Premium_Amount|Time_Since_First_Contact|
|-|-|-|-|-|                                                          
|0 Claims|40.0|714.4|2183.2|15.5|
|1 Claim|40.3|713.9|2255.1|15.5|
|2+ Claims|39.1|714.7|2348.6|15.5|

#### Mean of Numerical Drivers by Claims Severity:

|Claims_Severity|Age|Credit_Score|Premium_Amount|Time_Since_First_Contact|
|-|-|-|-|-|
|High|40.0|716.5|2274.0|15.0|
|Low|40.0|713.8|2207.8|15.5|
|Medium|40.1|714.7|2234.5|15.5|

<img width="1590" height="1190" alt="download" src="https://github.com/user-attachments/assets/ccc04dc1-0e7b-4516-bb0e-e5e8bc7302d1" />

## 2. Analysis of Claims Groups by Key Categorical Features (Proportions):

#### Distribution of Marital_Status within each Claims Frequency Group (%):

|Claims_Group|Divorced|Married|Single|Widowed|
|-|-|-|-|-|                                      
|0 Claims|9.6%|48.7%|32.7%|9.0%|
|1 Claim|8.3%|49.4%|32.3%|10.0%|
|2+ Claims|9.4%|49.8%|32.7%|8.1%|

<img width="989" height="590" alt="download" src="https://github.com/user-attachments/assets/26c43071-fef1-42fc-9293-26173b5582cf" />

#### Distribution of Policy_Type within each Claims Frequency Group (%):
|Claims_Group|Policy_Type|Full Coverage|Liability-Only|
|-|-|-|-|                              
|0 Claims|60.2%|39.8%|
|1 Claim|59.2%|40.8%|
|2+ Claims|61.8%|38.2%|

<img width="987" height="590" alt="download" src="https://github.com/user-attachments/assets/905a5cc1-d9dd-4d37-809b-b487be3ed093" />

#### Distribution of Region within each Claims Frequency Group (%):
|Claims_Group|Region|Rural|Suburban|Urban|
|-|-|-|-|                        
|0 Claims|20.5%|30.3%|49.2%|
|1 Claim|20.7%|29.6%|49.7%|
|2+ Claims|20.4%|32.0%|47.6%|

<img width="988" height="590" alt="download" src="https://github.com/user-attachments/assets/3950bfbf-81a4-4785-ac63-b4d9f240b51f" />

#### 3. Summary: Claims Severity Distribution by Region:
|Region|Claims_Severity|High|Low|Medium|
|-|-|-|-|-|                             
|Rural|10.3%|69.3%|20.5%|
|Suburban|8.9%|70.0%|21.1%|
|Urban|9.7%|70.4%|19.9%|

<img width="988" height="590" alt="download" src="https://github.com/user-attachments/assets/7645c56c-6469-4ff2-b1e2-6b8e31da17b8" />

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuration for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
np.random.seed(42) # For reproducibility

def load_data(file_path):
    """Loads the dataset from the specified file path and adds synthetic date data."""
    try:
        df = pd.read_csv(file_path)
        
        # --- SYNTHETIC TIME DATA FOR TIME SERIES ANALYSIS ---
        # Assuming policies were initiated over the last 12 months for demonstration
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31')
        
        # Assign a random date within the 12-month window to each policy
        df['Policy_Date'] = pd.to_datetime(np.random.choice(
            pd.date_range(start_date, end_date), size=len(df)
        ))
        
        print(f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns (including synthetic Policy_Date).")
        return df
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

def analyze_linear_relationship(df):
    """
    SECTION 1: Linear Regression to model Claims_Adjustment (Severity Proxy) based on Claims_Frequency.
    """
    print("\n--- 1. Linear Modeling: Claims Adjustment vs. Frequency ---")

    if 'Claims_Frequency' not in df.columns or 'Claims_Adjustment' not in df.columns:
        print("Required columns for linear analysis are missing. Skipping.")
        return

    # Select and prepare data
    X = df[['Claims_Frequency']]
    y = df['Claims_Adjustment']

    # Use a simple scaling of the target variable for better model fitting visualization, 
    # though not strictly necessary for simple linear regression coefficients.
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Train the model
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    y_pred_scaled = model.predict(X_scaled)
    
    # R-squared on the scaled data (proxy for fit)
    r_squared = model.score(X_scaled, y_scaled)

    print(f"Model Coefficient (Scaled): {model.coef_[0][0]:.3f}")
    print(f"Model Intercept (Scaled): {model.intercept_[0]:.3f}")
    print(f"R-squared Score: {r_squared:.3f}")
    
    print("\nInterpretation: A positive coefficient indicates that as Claims Frequency increases, the average Claims Adjustment amount tends to increase.")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.regplot(x=X['Claims_Frequency'], y=y, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('Linear Relationship: Claims Adjustment vs. Claims Frequency')
    plt.xlabel('Claims Frequency (Count)')
    plt.ylabel('Claims Adjustment Amount')
    plt.text(0.95, 0.05, f'R-squared = {r_squared:.2f}', 
             transform=plt.gca().transAxes, 
             ha='right', 
             bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

def perform_cluster_analysis(df):
    """
    SECTION 2: Cluster Analysis (K-Means) to segment customers based on claims profile.
    """
    print("\n--- 2. Cluster Analysis: K-Means Segmentation ---")

    required_cols = ['Claims_Frequency', 'Claims_Adjustment', 'Claims_Severity']
    if not all(col in df.columns for col in required_cols):
        print("Required columns for cluster analysis are missing. Skipping.")
        return
        
    # --- Data Preprocessing for Clustering ---
    df_cluster = df.copy()

    # Encode categorical severity (High=3, Medium=2, Low=1) for numerical distance calculation
    severity_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df_cluster['Claims_Severity_Enc'] = df_cluster['Claims_Severity'].map(severity_map).fillna(1)
    
    features = ['Claims_Frequency', 'Claims_Adjustment', 'Claims_Severity_Enc']
    X_cluster = df_cluster[features].copy()

    # Standardize features (Crucial for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # --- Find Optimal K using Elbow Method ---
    inertia = []
    # Test K from 1 to 10
    k_range = range(1, 8)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.show()
    
    # --- Run K-Means with chosen K (e.g., K=3 or K=4 based on the elbow) ---
    # Using K=4 for illustration of different risk segments
    OPTIMAL_K = 4
    print(f"Running K-Means with Optimal K = {OPTIMAL_K}")
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init='auto')
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

    # --- Cluster Profile Summary ---
    cluster_summary = df_cluster.groupby('Cluster')[features].mean()
    print("\nCluster Profiles (Averages):")
    print(cluster_summary.to_string(float_format="{:.2f}".format))
    
    # --- Visualization ---
    # Visualize clusters using the two most important claims features
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='Claims_Frequency', 
        y='Claims_Adjustment', 
        hue='Cluster', 
        data=df_cluster, 
        palette='Spectral', 
        s=100
    )
    plt.title(f'Customer Clusters (K={OPTIMAL_K}) based on Claims Profile')
    plt.xlabel('Claims Frequency')
    plt.ylabel('Claims Adjustment')
    plt.legend(title='Risk Cluster')
    plt.show()

def analyze_time_series(df):
    """
    SECTION 3: Simulated Time-Series Analysis to observe and forecast Claims Frequency.
    """
    print("\n--- 3. Simulated Time-Series Analysis ---")

    if 'Policy_Date' not in df.columns or 'Claims_Frequency' not in df.columns:
        print("Required time or claims columns are missing. Skipping.")
        return

    # Aggregate Claims Frequency by month
    df_ts = df.set_index('Policy_Date').resample('M')['Claims_Frequency'].mean().reset_index()
    df_ts.rename(columns={'Claims_Frequency': 'Avg_Claims_Frequency'}, inplace=True)
    
    # Set the month as index for time series modeling
    df_ts.set_index('Policy_Date', inplace=True)

    # --- Simple Forecasting Demonstration (Rolling Mean) ---
    WINDOW_SIZE = 3
    # Create a simple 3-month rolling mean for prediction/smoothing
    df_ts['Rolling_Mean'] = df_ts['Avg_Claims_Frequency'].rolling(window=WINDOW_SIZE, center=False).mean()

    # Shift the rolling mean to represent a forecast (predicting next period based on current window)
    df_ts['Forecast'] = df_ts['Rolling_Mean'].shift(1)
    
    # Fill initial NaNs for better visualization
    df_ts.fillna(method='bfill', inplace=True) 

    print("\nSimulated Monthly Claims Frequency Data and 1-Month Rolling Mean Forecast:")
    print(df_ts.to_string(float_format="{:.3f}".format))

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df_ts.index, df_ts['Avg_Claims_Frequency'], label='Actual Avg Claims Frequency', marker='o', color='blue')
    plt.plot(df_ts.index, df_ts['Forecast'], label=f'{WINDOW_SIZE}-Month Rolling Forecast', linestyle='--', color='red')
    plt.title('Simulated Time Series: Monthly Average Claims Frequency')
    plt.xlabel('Policy Month')
    plt.ylabel('Average Claims Frequency')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("--- Starting Advanced Claims Risk Analysis Script ---")
    
    # -------------------------------------------------------------------------
    # ACTION REQUIRED: Replace 'your_dataset_file.csv' with your actual file path.
    # The script automatically adds a synthetic 'Policy_Date' column.
    # ------------------------------------------------------------------------- 
    df = load_data(DATA_FILE)
    
    if df is not None:
        
        # 1. Linear Modeling
        analyze_linear_relationship(df.copy())

        # 2. Cluster Analysis (K-Means)
        perform_cluster_analysis(df.copy())
        
        # 3. Simulated Time Series Analysis
        analyze_time_series(df.copy())
        
    print("\nAdvanced claims analysis script finished.")
```
### --- 1. Linear Modeling: Claims Adjustment vs. Frequency ---
**Model Coefficient (Scaled):** 0.804
**Model Intercept (Scaled):** -0.000
**R-squared Score:** 0.646
Interpretation: A positive coefficient indicates that as Claims Frequency increases, the average Claims Adjustment amount tends to increase.
<img width="850" height="547" alt="download" src="https://github.com/user-attachments/assets/7ac0753d-4564-42cd-ab20-e49ad5ddfdad" />

### --- 2. Cluster Analysis: K-Means Segmentation ---
<img width="713" height="470" alt="download" src="https://github.com/user-attachments/assets/1d95cc6a-f860-483a-ad9f-d16d3f2d2f70" />
Running K-Means with Optimal K = 4

#### Cluster Profiles (Averages):
|Cluster|Claims_Frequency|Claims_Adjustment|Claims_Severity_Enc|
|-|-|-|-|                                                          
|0|1.68|245.23|2.62|
|1|0.00|0.00|1.00|
|2|0.00|0.00|2.32|
|3|1.22|69.81|1.18|

<img width="850" height="701" alt="download" src="https://github.com/user-attachments/assets/3cf54882-90dd-497e-bde3-b5326709fdc5" />

### --- 3. Simulated Time-Series Analysis ---

Simulated Monthly Claims Frequency Data and 1-Month Rolling Mean Forecast:
|Policy_Date|Avg_Claims_Frequency|Rolling_Mean|Forecast|
|-|-|-|-|                                              
|2024-01-31|0.481|0.494|0.494|
|2024-02-29|0.486|0.494|0.494|
|2024-03-31|0.513|0.494|0.494|
|2024-04-30|0.522|0.507|0.494|
|2024-05-31|0.492|0.509|0.507|
|2024-06-30|0.520|0.511|0.509|
|2024-07-31|0.448|0.487|0.511|
|2024-08-31|0.472|0.480|0.487|
|2024-09-30|0.517|0.479|0.480|
|2024-10-31|0.525|0.505|0.479|
|2024-11-30|0.466|0.503|0.505|
|2024-12-31|0.523|0.505|0.503|

<img width="1190" height="590" alt="download" src="https://github.com/user-attachments/assets/2f435a48-fdcf-43ce-a79b-b13c47cb062b" />

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # Required for ANOVA testing

# Configuration for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Significance level for p-value tests
ALPHA = 0.05 

def load_data(file_path):
    """Loads the dataset from the specified file path."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns.")
        return df
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

def quantify_numerical_influence(df):
    """
    SECTION 1: Quantify the linear influence (Correlation) of Numerical Drivers
    on Claims Frequency and Claims Adjustment (Severity Proxy).
    """
    print("\n--- 1. Numerical Drivers: Correlation Analysis ---")

    # Features to test influence of
    numerical_drivers = ['Age', 'Credit_Score', 'Premium_Amount', 'Total_Discounts', 'Website_Visits']
    # Targets to be influenced
    claims_targets = ['Claims_Frequency', 'Claims_Adjustment']

    # Filter for existing columns
    numerical_drivers = [col for col in numerical_drivers if col in df.columns]
    claims_targets = [col for col in claims_targets if col in df.columns]
    
    if not numerical_drivers or not claims_targets:
        print("Warning: Missing required columns for numerical influence analysis. Skipping.")
        return

    # Calculate the Pearson correlation matrix
    correlation_matrix = df[numerical_drivers + claims_targets].corr()
    
    # Extract correlations of drivers with claims targets
    claims_correlations = correlation_matrix.loc[numerical_drivers, claims_targets]
    
    # Add absolute correlation for easy sorting
    claims_correlations['Abs_Frequency_Corr'] = claims_correlations['Claims_Frequency'].abs()
    claims_correlations['Abs_Adjustment_Corr'] = claims_correlations['Claims_Adjustment'].abs()
    
    print("\nCorrelation of Numerical Drivers with Claims Metrics:")
    print("Interpretation: Closer to 1 or -1 means stronger linear influence.")
    print(claims_correlations.sort_values(by='Abs_Frequency_Corr', ascending=False).to_string(float_format="{:.3f}".format))
    
    # Visualization: Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        claims_correlations.drop(columns=['Abs_Frequency_Corr', 'Abs_Adjustment_Corr']),
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5, 
        cbar_kws={'label': 'Pearson Correlation'}
    )
    plt.title('Correlation of Numerical Features with Claims Metrics')
    plt.tight_layout()
    plt.show()

def quantify_categorical_influence(df):
    """
    SECTION 2: Quantify the statistical influence (ANOVA) of Categorical Drivers
    on Claims Adjustment (Severity Proxy).
    """
    print("\n--- 2. Categorical Drivers: ANOVA (Influence on Claims Adjustment) ---")

    # Features to test influence of
    categorical_drivers = ['Marital_Status', 'Region', 'Policy_Type']
    claims_target = 'Claims_Adjustment'
    
    # Filter for existing columns and ensure no NaNs in target
    categorical_drivers = [col for col in categorical_drivers if col in df.columns]
    if claims_target not in df.columns or not categorical_drivers:
        print("Warning: Missing required columns for categorical influence analysis. Skipping.")
        return
        
    df_clean = df.dropna(subset=[claims_target])

    anova_results = []

    for driver in categorical_drivers:
        # 1. Group the target variable by the categories
        # Ensure we handle potential errors if a driver column has non-numeric data for ANOVA
        try:
            groups = [
                df_clean[df_clean[driver] == category][claims_target].values
                for category in df_clean[driver].unique()
            ]
        
            # 2. Perform one-way ANOVA test (requires SciPy)
            # ANOVA tests if the mean Claims Adjustment is significantly different across the categories
            f_stat, p_value = stats.f_oneway(*groups)
            
            # 3. Determine significance
            is_significant = p_value < ALPHA
            
            anova_results.append({
                'Driver': driver,
                'F_Statistic': f_stat,
                'P_Value': p_value,
                f'Significant_at_{ALPHA*100:.0f}%': is_significant
            })
        except Exception as e:
            print(f"Skipping ANOVA for {driver}: {e}")
            continue


    results_df = pd.DataFrame(anova_results).sort_values(by='F_Statistic', ascending=False)
    
    print("\nANOVA Test Results for Claims Adjustment:")
    print("Interpretation: A low P-Value (< 0.05) and high F-Statistic means the category has a SIGNIFICANT influence on the average Claims Adjustment amount.")
    
    # FIX: Use 'formatters' with callable functions for column-specific string formatting
    custom_formatters = {
        "F_Statistic": lambda x: f"{x:.2f}",
        "P_Value": lambda x: f"{x:.5f}"
    }
    
    # Only use formatters for columns that exist in results_df
    final_formatters = {k: v for k, v in custom_formatters.items() if k in results_df.columns}

    print(results_df.to_string(index=False, formatters=final_formatters))
    
    # Visualization: Mean Claims Adjustment by significant groups
    significant_drivers = results_df[results_df[f'Significant_at_{ALPHA*100:.0f}%'] == True]['Driver'].tolist()
    
    if significant_drivers:
        print(f"\nVisualizing Means for Significant Drivers ({', '.join(significant_drivers)}):")
        fig, axes = plt.subplots(1, len(significant_drivers), figsize=(6 * len(significant_drivers), 5))
        axes = np.array([axes]).flatten() # Ensure axes is iterable even if only one plot

        for i, driver in enumerate(significant_drivers):
            # Ensure the driver column exists before grouping
            if driver in df.columns:
                mean_adj = df.groupby(driver)[claims_target].mean().sort_values(ascending=False)
                sns.barplot(x=mean_adj.index, y=mean_adj.values, ax=axes[i], palette='YlOrRd')
                axes[i].set_title(f'Mean Claims Adjustment by {driver}')
                axes[i].set_ylabel('Mean Claims Adjustment')
                axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo categorical drivers were found to have a statistically significant influence on Claims Adjustment.")


if __name__ == '__main__':
    print("--- Starting Claims Risk Driver Statistical Modeling Script ---")
    
    # -------------------------------------------------------------------------
    # ACTION REQUIRED: Replace 'your_dataset_file.csv' with your actual file path.
    # ------------------------------------------------------------------------- 
    df = load_data(DATA_FILE)
    
    if df is not None:
        
        # 1. Quantify linear influence of numerical drivers
        quantify_numerical_influence(df.copy())

        # 2. Quantify statistical influence of categorical drivers using ANOVA
        quantify_categorical_influence(df.copy())
        
    print("\nStatistical claims driver analysis script finished.")
```
### --- 1. Numerical Drivers: Correlation Analysis ---

Correlation of Numerical Drivers with Claims Metrics:
Interpretation: Closer to 1 or -1 means stronger linear influence.

|Premium_Amount|Claims_Frequency|Claims_Adjustment|Abs_Frequency_Corr|Abs_Adjustment_Corr|
|-|-|-|-|-|
||0.355|0.439|0.355|0.439|
|Age|-0.006|-0.008|0.006|0.008|
|Website_Visits|0.005|0.005|0.005|0.005|
|Total_Discounts|0.003|-0.003|0.003|0.003|
|Credit_Score|0.002|0.006|0.002|0.006|

<img width="780" height="590" alt="download" src="https://github.com/user-attachments/assets/35b64ba0-7731-4663-94d3-b70899460e39" />

### --- 2. Categorical Drivers: ANOVA (Influence on Claims Adjustment) ---
ANOVA Test Results for Claims Adjustment:
**Interpretation: A low P-Value (< 0.05) and high F-Statistic means the category has a SIGNIFICANT influence on the average Claims Adjustment amount.**

|Driver|F_Statistic|P_Value|Significant_at_5%|
|-|-|-|-|
|Marital_Status|1.03|0.37857|False|
|Policy_Type|0.40 0.52795|False|
|Region|0.08|0.92231|False|

**No categorical drivers were found to have a statistically significant influence on Claims Adjustment.**













