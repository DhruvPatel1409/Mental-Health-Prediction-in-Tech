# Mental Health Prediction - README

## Dataset Preprocessing Steps
### 1. Loading the Dataset
- The dataset is read from `survey.csv` using `pandas.read_csv()`.
- The dataset contains various attributes related to an individual's mental health status in the workplace.
- Initial inspection of the dataset is performed using `.head()`, `.info()`, and `.describe()` functions to understand its structure and statistical properties.

### 2. Handling Missing Values
- Certain columns such as `Timestamp` and `comments` are deemed unnecessary and are **dropped**.
- Categorical columns with missing values are **imputed using mode**:
  - `state`: Missing values are filled with the most frequently occurring state.
  - `work_interfere`: Missing values are filled with the most frequent category.
- The `.isnull().sum()` function is used to verify that all missing values have been handled appropriately.

### 3. Encoding Categorical Features
- Since the dataset contains categorical variables, they need to be converted into numerical format for machine learning models.
- The following encoding techniques are used:
  - `LabelEncoder`: Converts categorical labels into integers.
- The final dataset contains a mix of encoded categorical and numerical features.

### 4. Feature Scaling
- Numerical features are standardized using `StandardScaler` to ensure consistent scaling.
- Standardization helps improve model convergence and performance by keeping feature distributions uniform.

### 5. Splitting Data
- The dataset is split into **training (80%)** and **testing (20%)** sets using `train_test_split()` from `sklearn.model_selection`.
- A random seed (`random_state=42`) is used to ensure reproducibility of results.

---

## Model Selection Rationale
Several machine learning models were trained and compared to identify the most suitable one for predicting mental health status.

### 1. Logistic Regression
- A simple **Logistic Regression** model was implemented as a baseline.
- It provides an initial benchmark for performance comparison.

### 2. Random Forest Classifier (RFC) - **Final Selected Model**
- Hyperparameter tuning was performed using **GridSearchCV** to find the best configuration.
- Achieved the **highest accuracy** among all models.
- Handles feature importance effectively and works well with structured data.
- Provides robust performance and generalization across different test cases.

### 3. Gradient Boosting Classifier
- Sequentially trains weak models to correct the errors of previous ones.
- Showed good performance but slightly lower accuracy compared to RFC.

### 4. Decision Tree Classifier
- A simple, interpretable model used for comparison.
- Had lower accuracy and was prone to overfitting, making it less ideal for the final selection.

### 5. Model Evaluation Criteria
- **Accuracy Score**: Measures overall prediction correctness.
- **Confusion Matrix**: Provides insight into misclassifications.
- **ROC-AUC Score**: Evaluates model performance across different threshold levels.
- **Classification Report**: Includes precision, recall, and F1-score.

Since **Random Forest Classifier (RFC) achieved the highest accuracy**, it was selected as the final model for inference.

---

## How to Run the Inference Script
### Prerequisites
Before running the inference script, ensure you have the necessary dependencies installed.

#### 1. Install Required Packages
```bash
pip install pandas numpy scikit-learn xgboost lime matplotlib seaborn
```

#### 2. Run the Inference Script
```bash
python inference.py --input "input_data.csv"
```
- `--input`: Path to the CSV file containing new data for prediction.
- The script loads the pre-trained **Random Forest Classifier** and applies preprocessing steps before making predictions.

#### 3. Output
- The script outputs a CSV file containing the predicted mental health status.
- Predictions can also be displayed directly on the console.


## UI/CLI Usage Instructions

### Streamlit Web UI Usage
If a graphical UI (built with Streamlit) is available, use:
```bash
streamlit run app.py
```
- Open the browser and navigate to the provided `localhost` link.
- The UI allows users to enter input values via a form and get instant predictions.

For further details, refer to the full notebook: `predict_mental_health.ipynb`.

