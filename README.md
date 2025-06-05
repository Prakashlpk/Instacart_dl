# Deep Learning Based Product Reordering Prediction

## Project Overview

This project implements a deep learning solution to predict whether a user will reorder a product based on their past purchase behavior using the Instacart Market Basket Analysis dataset. The model uses a Deep Neural Network (DNN) classifier to provide binary classification and probability predictions for reordering behavior.

## Business Use Cases

- **Personalized Product Recommendation**: Suggest products users are likely to reorder
- **Inventory Management**: Predict future demand for products based on reorders
- **Customer Retention**: Identify and target customers likely to churn based on changes in reorder patterns
- **Marketing Optimization**: Trigger personalized marketing campaigns based on reorder likelihood

## Dataset

The project uses the **Instacart Market Basket Analysis** dataset containing:

- `orders.csv`: Customer order details with timing information
- `order_products__train.csv`: Products in training orders (with reordered flags)
- `order_products__prior.csv`: Products in prior orders (for historical features)
- `products.csv`: Product metadata
- `aisles.csv`: Aisle metadata
- `departments.csv`: Department metadata

**Key Variables**: user_id, product_id, order_id, aisle_id, department_id, reordered (0/1), order_dow, order_hour_of_day, days_since_prior_order

## Features Engineered

### User-based Features
- `user_total_orders`: Maximum order number for each user
- `user_avg_days_between_orders`: Average days between consecutive orders
- `user_days_since_last_order`: Days since the user's last order

### Product-based Features  
- `prod_total_orders`: Total number of times product was ordered
- `prod_reorders`: Total number of times product was reordered
- `prod_reorder_ratio`: Ratio of reorders to total orders

### User-Product Interaction Features
- `up_orders`: Number of times user ordered this specific product
- `up_first_order`: Order number when user first ordered this product
- `up_last_order`: Order number when user last ordered this product
- `up_reorders`: Number of times user reordered this specific product
- `up_order_rate`: Ratio of user-product orders to user's total orders

## Model Architecture

**Deep Neural Network (DNN) Classifier:**
- Input Layer: 11 features (after preprocessing)
- Hidden Layer 1: 256 neurons + ReLU + BatchNorm + Dropout(0.3)
- Hidden Layer 2: 128 neurons + ReLU + BatchNorm + Dropout(0.25)
- Hidden Layer 3: 64 neurons + ReLU + BatchNorm + Dropout(0.2)
- Hidden Layer 4: 32 neurons + ReLU + BatchNorm + Dropout(0.15)
- Output Layer: 1 neuron + Sigmoid activation

**Training Configuration:**
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam (learning_rate=0.001)
- Metrics: Accuracy, Precision, Recall, AUC
- Early Stopping: patience=10, monitor='val_loss'
- Batch Size: 512
- Max Epochs: 50

## Installation & Setup

### Prerequisites

pip install -r requirements.txt

### Required Dependencies

pandas
numpy
tensorflow
scikit-learn
imbalanced-learn
matplotlib
streamlit
joblib


### Dataset Setup
1. Download the Instacart dataset files:
   - orders.csv
   - order_products__train.csv
   - order_products__prior.csv
   - products.csv
   - aisles.csv
   - departments.csv

2. Place all CSV files in the project directory

## Usage Instructions

### 1. Training the Model
jupyter notebook instacart_project.ipynb

This script will:
- Load and merge all dataset files
- Perform feature engineering
- Apply preprocessing (log transformation, encoding, scaling)
- Handle class imbalance using undersampling
- Train the deep learning model
- Save the trained model as `instacart_model.keras`
- Save the scaler as `scaler.pkl`
- Generate evaluation metrics and business insights

### 2. Running the Streamlit App

streamlit run streamlit_app.py


The web interface allows users to:
- Input feature values for a user-product pair
- Get reorder probability predictions

## Data Preprocessing Steps

1. **Data Integration**: Merge orders, products, and metadata tables
2. **Missing Value Handling**: Fill missing `days_since_prior_order` with 0
3. **Feature Engineering**: Create user, product, and interaction features
4. **Skewness Correction**: Apply log transformation to skewed numerical features
5. **Encoding**: Label encode categorical variables (product_name, aisle, department)
6. **Class Imbalance**: Apply RandomUnderSampler with sampling_strategy=0.666
7. **Feature Scaling**: StandardScaler normalization for neural network input

## Model Performance

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives  
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification breakdown

### Target Performance
- F1-Score > 0.65 on validation data
- Balanced performance across precision and recall

## Business Impact Analysis

The model provides insights for four key business areas:

1. **Product Recommendations**: Identifies high-probability reorder products
2. **Demand Forecasting**: Predicts inventory needs by department/aisle
3. **Customer Retention**: Flags customers at high churn risk
4. **Marketing Segmentation**: Categorizes customers by reorder intent


## Key Features

- **Reproducibility**: Fixed random seeds (SEED=42) for consistent results
- **Modular Design**: Separate preprocessing, modeling, and evaluation phases
- **Production Ready**: Saved model and scaler for deployment
- **Interactive Interface**: Streamlit app for real-time predictions
- **Business Intelligence**: Comprehensive analysis of model outputs

## Model Deployment

The trained model can be deployed using:
**Streamlit Web App**
