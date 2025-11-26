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

</details>

---

<details>
<summary><b>📊 Key Insights</b></summary>

- Budget has the strongest positive correlation with gross revenue (~0.74).
- Votes also show a significant relationship (~0.61), reinforcing the link between audience engagement and earnings.
- Runtime moderately correlates (~0.25), while IMDb score has only a weak influence (~0.09).
- Company, director, and star correlations remain low (0.15–0.22), disproving assumptions of brand impact.
- Genre and country contribute minimal effect (0.04–0.10).
- Long-term trends (1980–2020) show gradual revenue growth but no single dominant factor outside investment and audience metrics.

</details>

---

<details>
<summary><b>🧰 Tools Used</b></summary>

| Tool | Purpose |
|------|----------|
| **Python** | Core scripting and analysis language |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numeric operations and array handling |
| **Seaborn & Matplotlib** | Data visualization and correlation plotting |
| **Jupyter Notebook** | Interactive environment for development and documentation |
| **Dataset Source** | Kaggle Movie Industry Dataset (~7,666 entries, columns: name, genre, budget, gross, votes, year, etc.) |

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

