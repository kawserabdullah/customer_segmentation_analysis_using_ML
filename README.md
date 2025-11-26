## 🧮 Customer Segmentation Using Credit Card Data (Advanced ML Clustering Project)

A Python-based **machine learning project** focused on unsupervised clustering of ~9,000 credit card customers using real-world behavioral financial data. The goal is to segment customers using **KMeans clustering, PCA dimensionality reduction, feature scaling, and statistical analysis,** enabling banks to recommend customized **loans, credit upgrades, saving plans, and wealth management services.**

---

## 🎯 Project Objective
Use unsupervised machine learning to:

    - Understand customer behavior through 18 credit card usage features
    - Build optimal clusters using KMeans & PCA
    - Identify customer personas based on purchasing patterns, repayment behavior, and credit usage
    - Enable targeted financial product recommendations

This segmentation supports risk assessment, product personalization, and strategic decision-making.

---

## 🧭 Overview
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Language-Python-blue?style=flat-square&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Environment-Jupyter%20Notebook-orange?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

<details>
<summary><b>🧩 Project Steps</b></summary>

### 1. Import Libraries & Load Data
   - Imported essential ML libraries:
   - sklearn.cluster for KMeans
   - sklearn.preprocessing for StandardScaler
   - sklearn.decomposition for PCA
   - scipy for statistical checks

Loaded dataset and performed initial inspection.

### 2. ML-Oriented Data Cleaning & Feature Engineering
   - Imputed missing values using median strategy to avoid distribution distortion.
   - Removed non-predictive identifier column CUSTID.
   - Standardized features using StandardScaler to ensure equal weight for all ML dimensions.
   - Applied log transformations for skew-heavy financial variables.
   - Created a final ML-ready dataset saved as Clustered_Customer_Data.csv.

### 3. Exploratory Data Analysis (ML perspective)
   - Visualized feature distributions to detect outliers and skewness.
   - Generated correlation matrix heatmaps to understand multicollinearity.
   - Identified strongly influential behavioral features for clustering:
          - Purchases
          - Payment behavior
          - Cash advance patterns
          - Frequency-based features

### 4. Machine Learning — Clustering with KMeans
Extensively tested and tuned KMeans:

#### ✔️ Optimal number of clusters identified using:
   - Elbow Method (Within-Cluster-Sum-of-Squares)
   - Silhouette Score
   - Cluster Compactness vs. Separation Analysis

Chose optimal cluster count (typically 4) based on best silhouette performance.

#### ✔️ Final Model:
<pre>kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = kmeans.fit_predict(scaled_df)</pre>

#### ✔️ Cluster Output:
   - Added labels back to the dataset
   - Visualized cluster separation through PCA plots

### 5. Dimensionality Reduction — PCA
Applied PCA to convert 18 features → 2 principal components for visualization:
   - PCA1 captured spending & balance behavior
   - PCA2 captured transaction frequency & advance usage
Used scatterplots to visualize cluster boundaries clearly.

### 6. Cluster Interpretation (ML + Business Insight)
Each ML cluster was interpreted through:

   - Centroid analysis
   - Feature weight importance
   - Spending-to-payment ratios
   - Transaction behavior patterns

Generated a full customer persona mapping for each cluster.

</details>

---

<details>
<summary><b>📊 Key Machine Learning Insights</b></summary>

### 🔹 Cluster 1: Low-Interaction Customers
   - Lowest purchases, low credit limit usage
   - Low engagement → Suitable for basic savings plans

### 🔹 Cluster 2: High-Spending Installment Users
   - High purchase amounts
   - Heavy on installment usage
   - Good candidates for EMI products, consumer loans

### 🔹 Cluster 3: Cash-Advance Dependent Group
   - High cash advance usage, multiple cash transactions
   - Potential risk profile → monitoring, credit counseling

### 🔹 Cluster 4: Premium High-Value Customers
   - High full-payment percentage
   - Large credit limits, high spenders
   - Ideal for premium credit cards, wealth management, investments

### ML Performance Insights
   - PCA visualization clearly showed four separable and coherent clusters
   - Silhouette scores confirmed stable segmentation
   - Frequency features strongly influenced cluster separation

</details>

---

<details>
<summary><b>🧰 Tools & ML Libraries Used</b></summary>

| Tool | Purpose |
|------|----------|
| **Python** | Core scripting and analysis language |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numeric operations and array handling |
| **Seaborn & Matplotlib** | Exploratory visualizations |
| **Scikit-Learn** | Scaling, KMeans clustering, PCA |
| **StandardScaler** | Feature normalization |
| **KMeans** | ML clustering algorithm |
| **PCA** | Dimensionality reduction |
| **Jupyter Notebook** | Experimentation & documentation |
| **Dataset Source** | Kaggle Credit Card Customer Behavior Dataset (~9,000 entries, columns: 18 behavioral variables summarizing 6-month usage patterns for active cardholders) | |

</details>

---

<details>
<summary><b>⚙️ Setup Instructions</b></summary>

1. Clone this repository:
   
   ```git clone https://github.com/kawserabdullah/data_analysis_using_Python.git```
2. Install dependencies:
   
   ```pip3 install pandas numpy seaborn matplotlib jupyter```
   
   *(In case you are using Linux like me and pip3 is not installed out of the box then please use
   ```sudo apt install python3-pip``` before using the above command.)*  
3. Launch Jupyter Notebook:
   
   ```jupyter notebook```  
4. Open and run `Movie_Correlation_Analysis.ipynb`.  
5. Replace the dataset path in `pd.read_csv('movies.csv')` if using a custom location.

</details>

---

<details>
<summary><b>❌ Error Handeling</b></summary>

*Do all the steps below in "Data Cleaning & Preparation" step.*

1. Checked for missing values: ```for col in df.columns: print(f'{col} - {np.mean(df[col].isnull())*100:.2f}%')``` (Identified high nulls in budget (~28%), gross (~2%), etc.)

2. Handled missing data: Dropped rows with nulls in key columns like budget and gross using ```df = df.dropna(subset=['budget', 'gross'])```, or filled with medians/means (e.g., ```df['budget'].fillna(df['budget'].median(), inplace=True)```).

3. Corrected data types: Converted budget and gross to float/int after removing any formatting (e.g., ```df['budget'] = df['budget'].astype('int64')```).

4. Extracted accurate year: Created a new 'yearcorrect' column from 'released' date string: ```df['yearcorrect'] = df['released'].astype(str).str[:4]``` or using ```pd.to_datetime(df['released']).dt.year```.

5. Removed duplicates: ```df.drop_duplicates()``` (no impact in this dataset).

*In my case, Error mainly arrived because of not cleaning the dataset properly.*

</details>

---

👤 My GitHub Link
🌐 [https://github.com/kawserabdullah/](https://github.com/kawserabdullah/)

⭐ If you found this project useful, consider starring the repository!

