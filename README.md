# house-price-prediction-minor-project-by-parth-knatwa-and-pritesh-choyal
# 🏠 House Price Prediction using Random Forest Regression

## 📌 Overview

This project predicts house sale prices using Machine Learning techniques on the Ames Housing Dataset. The model is trained on 81 housing-related features such as lot area, neighborhood, building type, quality ratings, and other property characteristics.

The project covers the complete Machine Learning pipeline from data preprocessing to model evaluation.

---

## 🎯 Objective

To build a predictive model capable of estimating house prices based on property features.

---

## 📊 Dataset Information

- Dataset: Ames Housing Dataset
- Total Records: 1460
- Features: 81
- Target Variable: `SalePrice`

---

## ⚙️ Machine Learning Pipeline

### 1. Data Preprocessing
- Handling missing values
- Removing unnecessary features
- Encoding categorical variables
- Feature selection

### 2. Exploratory Data Analysis (EDA)
- Dataset inspection
- Distribution analysis
- Correlation analysis

### 3. Model Building
- Random Forest Regressor

### 4. Model Evaluation
- Mean Absolute Error (MAE)
- R² Score

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── houseprice.ipynb
├── train.csv
├── README.md
│
└── outputs/
```

---

## 📈 Results

| Metric | Value |
|----------|----------|
| MAE | 35,788 |
| R² Score | 0.634 |

### Interpretation

The model explains approximately 63.4% of the variation in house prices and achieves an average prediction error of around 35,788.

---

## 🚀 Future Improvements

- Hyperparameter Tuning
- XGBoost Regressor
- Gradient Boosting Models
- Feature Engineering
- Model Deployment using Streamlit

---

## 👨‍💻 Authors

**Parth Khatwa** and **Pritesh Choyal**

BCA Students, Medi-Caps University, Indore


---

⭐ If you found this project useful, consider giving it a star.

