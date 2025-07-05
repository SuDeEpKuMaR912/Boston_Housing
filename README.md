# Boston Housing Price Predictor

This is an end-to-end machine learning project to predict housing prices using the **Housing Dataset** from the UCI Machine Learning Repository.

The project demonstrates a full workflow: from data loading and visualization to model selection, cross-validation, and final evaluation on test data.

---

## 📂 Project Structure
Boston_Housing/

├── data_housing.csv

├── Boston_Price_predictor.py

├── Model_Usage.py

├── Housing.joblib

├── README.md


---

## 🧠 Models Implemented

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Each model is evaluated using:

- RMSE (Root Mean Squared Error)
- Cross-validation (10-fold)
- Final RMSE on unseen test data

---

## 📊 Model Comparison

| Model                   | Mean RMSE (Cross-Validation) | Std. Deviation | Final Test RMSE |
|------------------------|------------------------------|----------------|-----------------|
| Linear Regression       | 5.03                         | 1.05           | ~4.83           |
| Decision Tree Regressor | 4.20                         | 0.62           | 0.00 *(overfit)*|
| Random Forest Regressor | **3.28**                     | **0.68**       | **2.97**        |

✅ **RandomForestRegressor** performs the best in terms of balance between bias and variance.

---

## 🧪 Final Evaluation

### ✅ RandomForestRegressor was the best-performing model:

- **Cross-validation RMSE Mean:** `~3.28`
- **Cross-validation RMSE Std Dev:** `~0.68`
- **Final RMSE on Test Data:** `~2.97`

---

## 🔧 Key ML Concepts Applied

- Data Cleaning and Imputation (`SimpleImputer`)
- Feature Scaling (`StandardScaler`)
- **Pipeline** creation for modular preprocessing
- Stratified sampling (`StratifiedShuffleSplit`)
- Cross-validation (`cross_val_score`)
- Model persistence using `joblib`

---

## 📊 Visualizations

- Histograms of features
- Correlation matrix
- Scatter plots of strongly correlated features
- Prediction vs actual comparison

---

## 💻 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/SuDeEpKuMaR912/Boston_Housing.git
    ```

2. Navigate to the project folder and install requirements (if any).

3. Run the script:
    ```bash
    python Boston_Price_predictor.py
    ```

---

## 📁 Dataset Source

**[Boston Housing Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/Housing)**

---

## 📦 Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- joblib

You can install them using:

```bash
pip install -r requirements.txt

