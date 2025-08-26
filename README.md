# Analyzing and Predicting Water Quality for Safe Consumption

## 📌 Project Overview
This project focuses on **analyzing and predicting water quality for safe consumption** using machine learning techniques.  
The goal is to classify water as **potable (safe to drink)** or **non-potable** based on its physicochemical properties.

We implemented a complete machine learning pipeline, including **data preprocessing, EDA, feature engineering, model training, evaluation, and deployment** using Streamlit.

---

## 📊 Dataset
- **Source:** [Kaggle - Water Quality and Potability Dataset](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability/data)  
- **Size:** 3,276 samples, 10 columns  
- **Features:**  
  - pH  
  - Hardness  
  - Solids  
  - Chloramines  
  - Sulfate  
  - Conductivity  
  - Organic Carbon  
  - Trihalomethanes  
  - Turbidity  
  - Potability (Target: 1 = Safe, 0 = Not Safe)  

Missing values were handled using **median imputation**. Outliers were treated using **IQR capping**.  
Class imbalance was addressed using **SMOTE (Synthetic Minority Over-sampling Technique)**.

---

## 🛠️ Project Workflow
1. Data Collection & Cleaning  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering & Scaling  
4. Model Development (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost)  
5. Model Evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC)  
6. Deployment using **Streamlit Cloud**  

---

## 📈 Results
- **Best Model:** Support Vector Machine (SVM)  
- **Accuracy:** 93%  
- **AUC Score:** 0.967  

Other strong performers: Random Forest (90% accuracy) and XGBoost (88% accuracy).  

---

## 🚀 Deployment
The model is deployed using **Streamlit**.  
You can run the app locally with:
```bash
streamlit run app.py
```

---

## 📂 Repository Structure
```
Water-Potability-Prediction/
│── Potability_Prediction.ipynb        # Jupyter Notebook with ML pipeline
│── water_potability_dataset.csv       # Dataset (from Kaggle)
│── document.docx                      # Detailed project report
│── README.md                          # Project documentation
│── requirements.txt                   # Required libraries
│── .gitignore                         # Ignore unnecessary files
│── app.py (optional)                  # Streamlit app for deployment
```

---

## 🔮 Future Scope
- Incorporate larger and more diverse datasets  
- Add new features (temperature, bacterial levels, geographic location)  
- Try advanced deep learning models (ANN, CNN)  
- IoT-based real-time monitoring system  

---

## 👩‍💻 Contributors
- Paladugu Rishitha Naga Sri  
- Mahanthi Teja  
- Parupudi Nikhitha  
- Pamarthi Sri Chakra Pranay  
- Shaik Waseem Ahamad  

Under the guidance of **K. Meenakshi & M. Ruthumma**

---

## 📜 License
This project is open-source and available for academic/research purposes.
