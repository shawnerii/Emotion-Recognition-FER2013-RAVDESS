# ✈️ Airline Passenger Satisfaction Prediction

## 📖 Overview
This project analyzes and predicts airline passenger satisfaction based on flight experience, travel details, and service quality metrics.  
It includes:
- Data preprocessing & feature engineering
- Supervised learning models for classification
- Unsupervised clustering for segmentation
- Model evaluation & comparison
- Interactive Gradio web app for predictions

---

## 📂 Project Structure
ML_Project/
│── data/                 # Raw and processed datasets
│── notebooks/            # Jupyter notebooks for EDA, training, evaluation
│── models/               # Saved model and transformer .pkl files
│── app/                  # Gradio frontend code
│── figures/              # Dataset overview & workflow diagram
│── README.md             # Project documentation

---

## 🧠 Models Used
### **Supervised Models**
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- MLP (Neural Network)
- Extra Trees

### **Unsupervised Models**
- KMeans
- DBSCAN
- Gaussian Mixture Model (GMM)
- Spectral Clustering

---

## 📊 Evaluation Metrics
- **Classification:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Clustering:** Silhouette Score, Davies–Bouldin Index, Inertia

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/airline-satisfaction-prediction.git
cd airline-satisfaction-prediction/app
```

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Gradio app
python app.py

The app will be available at:
http://127.0.0.1:7860

---

The dataset is sourced from:
Kaggle - Airline Passenger Satisfaction

---

📌 Key Insights
	•	Class (Business/Eco) and Online boarding are top predictors of satisfaction.
	•	Inflight services (WiFi, entertainment, comfort) strongly influence satisfaction.
	•	Economy class and personal travel correlate with higher dissatisfaction.
	•	Gender, age, and gate location have negligible impact.
