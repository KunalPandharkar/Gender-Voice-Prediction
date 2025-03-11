# 🎙️ **Voice-Based Gender Classification Using Machine Learning**



## 📌 **Project Overview**
This project builds a **machine learning model** to classify **gender (Male/Female)** based on **voice frequency features**. Using **Random Forest and SVM**, the model analyzes acoustic signals and predicts gender with high accuracy.

## 🚀 **Features**
✅ **Feature Selection & Preprocessing** – Removes redundant features and normalizes data.  
✅ **Machine Learning Models** – Implements **SVM & Random Forest** for classification.  
✅ **Model Evaluation** – Compares accuracy and selects the best model.  
✅ **Model Deployment** – Saves trained model using **Pickle** for future use.  

## 🛠 **Technologies Used**
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn  
- **Machine Learning Models:** Support Vector Machine (SVM), Random Forest Classifier  
- **Data Preprocessing:** Feature Selection, Normalization, Correlation Analysis  
- **Model Deployment:** Pickle, Google Colab, Google Drive API  

## 📂 **Dataset**
- The dataset consists of **voice frequency features** such as **mean frequency, spectral entropy, skewness, kurtosis, and modulation index**.  
- **Target Variable:** `label` (Male/Female)  

## 📊 **Model Performance**
| Model  | Training Accuracy | Testing Accuracy |
|--------|------------------|-----------------|
| **SVM** | 72%  | 73%  |
| **Random Forest** | 100%  | **98%**  |

## 📌 **Installation & Usage**
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/KunalPandharkar/Voide-Gender-Prediction.git
cd voice-gender-classification
```
### 2️⃣ **Install Dependencies**
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```
### 3️⃣ **Load & Preprocess Dataset**
```python
import pandas as pd  
gen_data = pd.read_csv('drive/MyDrive/Dataset/gender_voice_dataset.csv')  
```
### 4️⃣ **Train Model**
```python
from sklearn.ensemble import RandomForestClassifier  
forest = RandomForestClassifier(n_estimators=500, random_state=42).fit(X_train, y_train)
```
### 5️⃣ **Save Model**
```python
import pickle  
pickle.dump(forest, open('voice_model.pickle', 'wb'))  
```
### 6️⃣ **Load & Predict**
```python
loaded_model = pickle.load(open('voice_model.pickle', 'rb'))  
result = loaded_model.predict(X_test)
```

## 📌 **Results & Insights**
- **Random Forest performed best with 98% accuracy**, making it the ideal choice for voice-based gender classification.  
- **Feature selection improved model performance** by removing redundant attributes.  
- **The model can be integrated into speech recognition or voice assistant applications.**  

## 📩 **Future Enhancements**
🔹 Deploy the model using a **Flask API for real-time predictions**.  
🔹 Train the model on a **larger, diverse dataset** for better generalization.  
🔹 Optimize feature engineering for improved accuracy.  

---
