# 🛡️ Insurance Customer Analytics | Machine Learning & Data Privacy

### Customer Similarity, Insurance Benefit Prediction, and Privacy-Preserving Data Transformation

**Python • Pandas • NumPy • Scikit-learn • kNN • Linear Regression • Data Obfuscation**

## 📌 Project Overview

Insurance companies can use machine learning to better understand their customers, identify similar profiles, and estimate the likelihood of insurance benefit claims.

However, predictive analytics must also consider the protection of sensitive customer information.

This project was developed for **Sure Tomorrow**, a fictional insurance company, to investigate four machine learning applications:

1. Identifying customers with similar characteristics.
2. Predicting whether a customer will receive an insurance benefit.
3. Predicting the number of insurance benefits a customer may receive.
4. Protecting customer data through a reversible mathematical transformation without compromising Linear Regression predictions.

The project combines exploratory data analysis, distance-based algorithms, classification, regression, feature scaling, and linear algebra.

## 🎯 Project Objectives

- Explore and preprocess customer insurance data.
- Identify similar customers using k-Nearest Neighbors.
- Compare Euclidean and Manhattan distance metrics.
- Evaluate the influence of feature scaling on distance-based algorithms.
- Build a binary classifier to predict insurance benefit receipt.
- Compare kNN classification against a random baseline.
- Implement Linear Regression using the Normal Equation.
- Evaluate regression performance with RMSE and R².
- Apply matrix-based data obfuscation.
- Verify that the transformation preserves Linear Regression predictions.

## 🛠️ Technologies Used

| Technology | Application |
|---|---|
| Python | Programming and analysis |
| Pandas | Data manipulation |
| NumPy | Matrix operations and numerical calculations |
| Scikit-learn | Nearest neighbors, classification, preprocessing, and metrics |
| Seaborn | Exploratory data visualization |
| Matplotlib | Visualization support |
| Jupyter Notebook | Interactive analysis and documentation |

## 📂 Dataset Description

The project uses `insurance_us.csv`, containing **5,000 customer records**.

The dataset includes the following variables:

| Feature | Description |
|---|---|
| `gender` | Customer gender |
| `age` | Customer age |
| `income` | Customer income |
| `family_members` | Number of family members |
| `insurance_benefits` | Number of insurance benefits received |

For binary classification, an additional target variable was created:

`insurance_benefits_received`

This variable indicates whether a customer received at least one insurance benefit.

## 🧹 Data Preparation and Exploration

The preprocessing workflow included:

- Inspecting data types and descriptive statistics.
- Checking for missing values and duplicate records.
- Standardizing column names.
- Converting customer age to an integer data type.
- Examining feature distributions and relationships.
- Preparing scaled and unscaled feature sets.

Exploratory visualization was performed using Seaborn's pairplot to investigate potential relationships between customer characteristics.

## 🔍 Task 1 — Finding Similar Customers

### Objective

Identify customers with similar demographic and financial characteristics.

### Methodology

The k-Nearest Neighbors algorithm was used to retrieve the closest customer records based on:

- Gender
- Age
- Income
- Number of family members

Two distance metrics were evaluated:

**Euclidean Distance**

\[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

**Manhattan Distance**

\[
d(x,y)=\sum_{i=1}^{n}|x_i-y_i|
\]

Both metrics were tested on original and scaled features.

### Key Finding

Feature scaling significantly influenced the nearest-neighbor results.

Without scaling, income dominated distance calculations because its numerical magnitude was substantially larger than that of the other variables.

After applying `MaxAbsScaler`, the features contributed on a more comparable scale.

**Business Application:** Customer similarity analysis can support segmentation and personalized marketing strategies.

## 🤖 Task 2 — Predicting Insurance Benefit Receipt

### Objective

Predict whether a customer will receive at least one insurance benefit.

### Target Variable

```python
df['insurance_benefits_received'] = (
    df['insurance_benefits'] > 0
).astype(int)
```

### Methodology

The classification workflow included:

1. Creating the binary target variable.
2. Examining class distribution.
3. Splitting data into training and testing sets.
4. Evaluating a random dummy baseline.
5. Training kNN classifiers with values of k from 1 to 10.
6. Comparing original and scaled features.
7. Evaluating performance using F1-score and confusion matrices.

### Evaluation Metric

The F1-score combines precision and recall:

\[
F1=2\frac{Precision \times Recall}{Precision+Recall}
\]

### Key Finding

The random baseline produced relatively low F1-scores.

The kNN classifiers achieved better results for selected values of k, particularly after feature scaling.

This demonstrates the importance of preprocessing for distance-based classification.

## 📈 Task 3 — Predicting the Number of Insurance Benefits

### Objective

Estimate the number of insurance benefits received by a customer using Linear Regression.

### Methodology

A custom `MyLinearRegression` class was implemented using NumPy and the Normal Equation.

The model computes regression coefficients using matrix operations and a pseudoinverse.

The regression model was evaluated using:

- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### Feature Scaling Comparison

The regression model was trained using both original and scaled data.

**Key Finding:** Feature scaling changed the model coefficients but did not meaningfully change predictive performance.

This is consistent with the mathematical properties of ordinary least squares regression under invertible linear feature transformations.

## 🔐 Task 4 — Protecting Customer Data with Matrix Obfuscation

### Objective

Apply a mathematical transformation to customer features while preserving the predictions generated by Linear Regression.

### Methodology

A random invertible matrix \(P\) was generated.

The original feature matrix \(X\) was transformed as follows:

\[
X'=XP
\]

Because \(P\) is invertible, the original features can be reconstructed:

\[
X=X'P^{-1}
\]

### Analytical Verification

The relationship between the original and transformed regression weights was derived mathematically.

For an invertible feature transformation, the corresponding regression coefficients can compensate for the transformation, preserving predictions.

### Computational Verification

The notebook:

1. Generated an invertible transformation matrix.
2. Transformed the customer features.
3. Reconstructed the original feature values.
4. Trained regression models on original and transformed data.
5. Compared the resulting predictions.
6. Verified numerical equivalence using `np.allclose()`.

### Key Finding

Linear Regression predictions remained equivalent within floating-point precision after applying the invertible transformation.

**Important:** This transformation is a demonstration of data obfuscation, not a substitute for production-grade encryption or a comprehensive privacy protection system.

## 💡 Key Findings

| Analysis | Main Insight |
|---|---|
| Customer similarity | Feature scaling affects nearest-neighbor results |
| kNN classification | Scaled features improved classification performance |
| Linear Regression | Scaling changed coefficients without meaningfully changing predictive performance |
| Data obfuscation | Invertible matrix transformations preserved regression predictions |

## 💼 Business Applications

**Customer Segmentation**

Identify similar customer profiles to support marketing initiatives.

**Claims Analytics**

Estimate whether customers are likely to receive insurance benefits.

**Predictive Modeling**

Use regression to analyze relationships between customer characteristics and benefit counts.

**Privacy-Aware Analytics**

Explore mathematical approaches to transforming customer data while retaining selected predictive properties.

## ⚠️ Limitations

- The analysis uses a fictional insurance business scenario.
- Model performance is specific to the available dataset.
- The binary target represents observed benefit receipt rather than a verified future claim outcome.
- Linear Regression may not be the ideal choice for non-negative count predictions.
- Matrix obfuscation alone does not guarantee privacy or security.
- Additional validation would be required before deploying these methods in a real insurance environment.

## 🚀 How to Run the Project

1. Clone or download this repository.
2. Install the required Python libraries.
3. Place `insurance_us.csv` in a local `data/` directory.
4. Update the dataset path in the notebook.
5. Open the notebook in Jupyter Notebook or JupyterLab.
6. Run the cells sequentially.

### Install Dependencies

```bash
pip install numpy pandas scikit-learn seaborn matplotlib notebook
```

## 📊 Conclusion

This project explores how machine learning and linear algebra can support different analytical tasks in the insurance industry.

The analysis demonstrates the importance of feature scaling for nearest-neighbor algorithms, compares classification models against a baseline, and implements Linear Regression using matrix operations.

It also illustrates how invertible mathematical transformations can preserve regression predictions while changing the representation of customer data.

The project showcases practical skills in **data preprocessing, classification, regression, model evaluation, linear algebra, and privacy-aware data analysis**.

## 👩‍💻 Author

**Andressa Miranda**

Computer Science | Data Analytics | Python | Machine Learning

[GitHub Profile](https://github.com/Andressa202)

*Developed as part of the TripleTen Data Science program.*
