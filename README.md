# DonorChoose
DonorsChoose.org, a platform dedicated to funding classroom projects, faces significant challenges in efficiently and consistently vetting an increasing number of project proposals.
# 🎯 DonorsChoose Project Approval Prediction

## 📌 Problem Statement
DonorsChoose is a crowdfunding platform where teachers request funding for classroom projects. The goal of this project is to build a machine learning model that predicts whether a project will be approved based on project details, resource requests, teacher information, and textual descriptions.

---

## 🎯 Objective
To develop a binary classification model that predicts:
- **1 → Approved**
- **0 → Not Approved**

---

## 📊 Target Metric
Since the dataset is **imbalanced**, accuracy alone is not reliable.

We focus on:
- **ROC-AUC Score**
- **Recall (especially for minority class - not approved projects)**
- **F1-score**

---

## 🧹 Data Preprocessing & Feature Engineering

### 🔹 Data Sources
- `train_data_df` → Project-level data
- `resources_df` → Resource-level data

### 🔹 Data Preparation Steps
- Aggregated resource-level data to project-level
- Merged datasets on project ID
- Handled missing values
- Dropped irrelevant columns (IDs, redundant text fields)

---

### 🔹 Feature Engineering

#### 📌 Text Features
- Combined essay fields
- Extracted:
  - Word count
  - Essay length
- (Optional: TF-IDF considered but optimized for memory)

#### 📌 Resource Features
- Total cost of project
- Average price
- Number of resources

#### 📌 Teacher Features
- Number of previously posted projects
- New vs experienced teacher indicator

#### 📌 Date Features
- Month of submission
- Day of week

#### 📌 Derived Features
- Cost-based features (high impact)
- Log transformation applied for skewed variables

---

## 📊 Exploratory Data Analysis (EDA)

Key insights:
- Majority of projects are approved (~85%)
- Strong class imbalance present
- Lower cost projects have higher approval probability
- Teacher experience positively influences approval
- Certain subject categories have higher approval rates

---

## ⚠️ Key Challenge: Class Imbalance

- Class 1 (Approved) dominates the dataset
- Initial models were biased towards predicting approvals

### ✅ Solution:
- Applied `class_weight='balanced'` for tree models
- Used `scale_pos_weight` in XGBoost
- Evaluated using recall and ROC-AUC instead of accuracy

---

## 🤖 Models Used

### 1. Logistic Regression
- Baseline model

### 2. Decision Tree
- Captures non-linear relationships
- Initially overfitted

### 3. Random Forest
- Improved generalization over Decision Tree

### 4. XGBoost (Best Model)
- Handled complex patterns
- Performed best after imbalance handling

---

## 📈 Model Performance

| Model | Accuracy | ROC-AUC | Key Observation |
|------|----------|--------|----------------|
| Logistic Regression | ~Baseline | Moderate | Biased toward majority |
| Decision Tree | ~0.84 | ~0.67 | Improved balance |
| Random Forest | ~0.85 | ~0.68 | Better generalization |
| XGBoost | ~0.85 | ~0.69+ | Best overall performance |

---

## 🔍 Feature Importance (XGBoost)

Top features:
- **Total Cost (most important)**
- Subject categories (Literacy, Math & Science)
- Word count (essay quality proxy)
- Teacher experience
- Resource price

---

## 💡 Key Insights & Recommendations

### 📌 Insights
- Cost is the strongest factor influencing approval
- Experienced teachers have higher approval rates
- Detailed project descriptions increase approval likelihood
- Subject categories impact approval probability

---

### 📌 Recommendations
- Optimize project cost to increase approval chances
- Encourage teachers to provide detailed descriptions
- Provide guidance for first-time teachers
- Focus funding on high-impact subject categories

---

## 🚀 Deployment

### Approach
- Model serialized using `pickle`
- Deployed using a simple API (Flask/FastAPI)

### Steps:
1. Save trained model
2. Create API endpoint
3. Load model and predict based on user input
4. Return approval probability

---

## 📦 Project Structure
