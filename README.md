# Cross-Modal Emotion Recognition (FER2013 & RAVDESS)

This project implements deep learning models (CNN, LSTM, and CNN-LSTM hybrid) to perform **unimodal emotion recognition** using two benchmark datasets:  
- **FER2013** (Facial Expression Recognition): 35,000+ grayscale facial images in 7 emotion categories.  
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song): Speech audio clips labeled with 8 emotions.  

The project also includes a **Gradio-based user interface** for real-time emotion detection from user-provided images or audio files.

---

## Features
- Trained and evaluated CNN, LSTM, and CNN-LSTM architectures for both datasets.
- Cross-modality performance comparison revealing modality-specific strengths.
- Real-time prediction through a simple, interactive web interface.
- Detailed analysis via classification reports and confusion matrices.

---

## Datasets
- **FER2013:** [Kaggle Link](https://www.kaggle.com/datasets/msambare/fer2013)  
- **RAVDESS:** [Official Link](https://zenodo.org/record/1188976)

---

## How to Run the App
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Emotion-Recognition-FER2013-RAVDESS.git
   cd Emotion-Recognition-FER2013-RAVDESS

2.	Navigate to the app directory:
    cd app

3.	Run the Gradio UI:
    python app.py

4.	Open your browser and go to:
    http://127.0.0.1:7860

The interface will allow you to:
	•	Select a trained model.
	•	Upload a face image or audio sample.
	•	View the predicted emotion and confidence score.