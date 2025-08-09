import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# Load all reusable artifacts
models_dir = "../models"
model_files = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "Random Forest": "Random_Forest.pkl",
    "Gradient Boosting": "Gradient_Boosting.pkl",
        "Naive Bayes (KNN used here)": "KNN.pkl",  # fallback name
        "MLP": "MLP.pkl",
    "XGBoost": "XGBoost.pkl",
    "Extra Trees": "Extra_Trees.pkl"
}

encoder_cols = joblib.load(os.path.join(models_dir, "encoder_columns.pkl"))
numeric_cols = joblib.load(os.path.join(models_dir, "numeric_cols.pkl"))
scaler = joblib.load(os.path.join(models_dir, "standard_scaler.pkl"))
service_features = joblib.load(os.path.join(models_dir, "service_features.pkl"))
pca = joblib.load(os.path.join(models_dir, "pca_transformer.pkl"))

# Define preprocessing function
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df)
    for col in encoder_cols:
        if col not in df_encoded:
            df_encoded[col] = 0
    df_encoded = df_encoded[encoder_cols]

    # Scale numeric features
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

    # --- Feature Engineering ---
    # Service rating variance
    df_encoded["Service Rating Variance"] = df_encoded[service_features].std(axis=1)

    # Had Delay
    df_encoded["Had Delay"] = ((df_encoded["Arrival Delay in Minutes"] + df_encoded["Departure Delay in Minutes"]) > 0).astype(int)

    # Delay Ratio
    df_encoded["Delay Ratio"] = (df_encoded["Arrival Delay in Minutes"] + df_encoded["Departure Delay in Minutes"]) / df_encoded["Flight Distance"]

    # Total Service Score
    df_encoded["Total Service Score"] = df_encoded[service_features].sum(axis=1)

    return df_encoded

# Define prediction function
def predict(
    model_name, gender, customer_type, age, travel_type, travel_class,
    flight_distance, wifi, time_convenient, online_booking, gate_location, food,
    boarding, comfort, entertainment, onboard_service, legroom, baggage,
    checkin, inflight_service, cleanliness, dep_delay, arr_delay
):
    try:
        # Pack inputs into a dictionary
        input_data = {
            "Gender": gender,
            "Customer Type": customer_type,
            "Age": age,
            "Type of Travel": travel_type,
            "Class": travel_class,
            "Flight Distance": flight_distance,
            "Inflight wifi service": wifi,
            "Departure/Arrival time convenient": time_convenient,
            "Ease of Online booking": online_booking,
            "Gate location": gate_location,
            "Food and drink": food,
            "Online boarding": boarding,
            "Seat comfort": comfort,
            "Inflight entertainment": entertainment,
            "On-board service": onboard_service,
            "Leg room service": legroom,
            "Baggage handling": baggage,
            "Checkin service": checkin,
            "Inflight service": inflight_service,
            "Cleanliness": cleanliness,
            "Departure Delay in Minutes": dep_delay,
            "Arrival Delay in Minutes": arr_delay
        }

        # Preprocess input
        processed_df = preprocess_input(input_data)

        # Load model
        model_path = os.path.join(models_dir, model_files[model_name])
        model = joblib.load(model_path)

        # Ensure feature alignment
        processed_df = processed_df[model.feature_names_in_]

        # Predict
        prediction = model.predict(processed_df)[0]
        return "Satisfied" if prediction == 1 else "Not Satisfied"

    except Exception as e:
        return f"Error: {str(e)}"

# Define input UI components (you can adjust defaults if needed)
inputs = [
    gr.Dropdown(list(model_files.keys()), label="Select Model"),
    gr.Dropdown(["Male", "Female"], label="Gender"),
    gr.Dropdown(["Loyal Customer", "disloyal Customer"], label="Customer Type"),
    gr.Slider(7, 85, label="Age"),
    gr.Dropdown(["Personal Travel", "Business travel"], label="Type of Travel"),
    gr.Dropdown(["Eco", "Eco Plus", "Business"], label="Class"),
    gr.Slider(30, 5000, label="Flight Distance"),
    gr.Slider(0, 5, label="Inflight wifi service"),
    gr.Slider(0, 5, label="Departure/Arrival time convenient"),
    gr.Slider(0, 5, label="Ease of Online booking"),
    gr.Slider(0, 5, label="Gate location"),
    gr.Slider(0, 5, label="Food and drink"),
    gr.Slider(0, 5, label="Online boarding"),
    gr.Slider(0, 5, label="Seat comfort"),
    gr.Slider(0, 5, label="Inflight entertainment"),
    gr.Slider(0, 5, label="On-board service"),
    gr.Slider(0, 5, label="Leg room service"),
    gr.Slider(0, 5, label="Baggage handling"),
    gr.Slider(0, 5, label="Checkin service"),
    gr.Slider(0, 5, label="Inflight service"),
    gr.Slider(0, 5, label="Cleanliness"),
    gr.Slider(0, 300, label="Departure Delay in Minutes"),
    gr.Slider(0, 300, label="Arrival Delay in Minutes"),
]

# Launch the interface
gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Text(label="Predicted Satisfaction"),
    title="Airline Satisfaction Predictor",
    description="Select a trained model and enter flight features to predict customer satisfaction.",
    flagging_mode="never"
).launch()


# import gradio as gr
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # Load all reusable artifacts
# models_dir = "../models"
# model_files = {
#     "Logistic Regression": "Logistic_Regression.pkl",
#     "Decision Tree": "Decision_Tree.pkl",
#     "Random Forest": "Random_Forest.pkl",
#     "Gradient Boosting": "Gradient_Boosting.pkl",
#     "SVM": "SVM.pkl",
#     "Naive Bayes (KNN used here)": "KNN.pkl",  # fallback name
#     "KNN": "KNN.pkl",
#     "MLP": "MLP.pkl",
#     "XGBoost": "XGBoost.pkl",
#     "Extra Trees": "Extra_Trees.pkl"
# }

# encoder_cols = joblib.load(os.path.join(models_dir, "encoder_columns.pkl"))
# numeric_cols = joblib.load(os.path.join(models_dir, "numeric_cols.pkl"))
# scaler = joblib.load(os.path.join(models_dir, "standard_scaler.pkl"))
# service_features = joblib.load(os.path.join(models_dir, "service_features.pkl"))
# pca = joblib.load(os.path.join(models_dir, "pca_transformer.pkl"))

# # Define preprocessing function
# def preprocess_input(data: dict) -> pd.DataFrame:
#     df = pd.DataFrame([data])

#     # One-hot encode categorical variables
#     df_encoded = pd.get_dummies(df)
#     for col in encoder_cols:
#         if col not in df_encoded:
#             df_encoded[col] = 0
#     df_encoded = df_encoded[encoder_cols]

#     # Scale numeric features
#     df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

#     # --- Feature Engineering ---
#     # Service rating variance
#     df_encoded["Service Rating Variance"] = df_encoded[service_features].std(axis=1)

#     # Had Delay
#     df_encoded["Had Delay"] = ((df_encoded["Arrival Delay in Minutes"] + df_encoded["Departure Delay in Minutes"]) > 0).astype(int)

#     # Delay Ratio
#     df_encoded["Delay Ratio"] = (df_encoded["Arrival Delay in Minutes"] + df_encoded["Departure Delay in Minutes"]) / df_encoded["Flight Distance"]

#     # Total Service Score
#     df_encoded["Total Service Score"] = df_encoded[service_features].sum(axis=1)

#     return df_encoded

# # Define prediction function
# def predict(
#     model_name, gender, customer_type, age, travel_type, travel_class,
#     flight_distance, wifi, time_convenient, online_booking, gate_location, food,
#     boarding, comfort, entertainment, onboard_service, legroom, baggage,
#     checkin, inflight_service, cleanliness, dep_delay, arr_delay
# ):
#     try:
#         # Pack inputs into a dictionary
#         input_data = {
#             "Gender": gender,
#             "Customer Type": customer_type,
#             "Age": age,
#             "Type of Travel": travel_type,
#             "Class": travel_class,
#             "Flight Distance": flight_distance,
#             "Inflight wifi service": wifi,
#             "Departure/Arrival time convenient": time_convenient,
#             "Ease of Online booking": online_booking,
#             "Gate location": gate_location,
#             "Food and drink": food,
#             "Online boarding": boarding,
#             "Seat comfort": comfort,
#             "Inflight entertainment": entertainment,
#             "On-board service": onboard_service,
#             "Leg room service": legroom,
#             "Baggage handling": baggage,
#             "Checkin service": checkin,
#             "Inflight service": inflight_service,
#             "Cleanliness": cleanliness,
#             "Departure Delay in Minutes": dep_delay,
#             "Arrival Delay in Minutes": arr_delay
#         }

#         # Preprocess input
#         processed_df = preprocess_input(input_data)

#         # Load model
#         model_path = os.path.join(models_dir, model_files[model_name])
#         model = joblib.load(model_path)

#         # Ensure feature alignment
#         processed_df = processed_df[model.feature_names_in_]

#         # Predict
#         prediction = model.predict(processed_df)[0]
#         return "Satisfied" if prediction == 1 else "Not Satisfied"

#     except Exception as e:
#         return f"Error: {str(e)}"

# # Define input UI components (you can adjust defaults if needed)
# inputs = [
#     gr.Dropdown(list(model_files.keys()), label="Select Model"),
#     gr.Dropdown(["Male", "Female"], label="Gender"),
#     gr.Dropdown(["Loyal Customer", "disloyal Customer"], label="Customer Type"),
#     gr.Slider(7, 85, label="Age"),
#     gr.Dropdown(["Personal Travel", "Business travel"], label="Type of Travel"),
#     gr.Dropdown(["Eco", "Eco Plus", "Business"], label="Class"),
#     gr.Slider(30, 5000, label="Flight Distance"),
#     gr.Slider(0, 5, label="Inflight wifi service"),
#     gr.Slider(0, 5, label="Departure/Arrival time convenient"),
#     gr.Slider(0, 5, label="Ease of Online booking"),
#     gr.Slider(0, 5, label="Gate location"),
#     gr.Slider(0, 5, label="Food and drink"),
#     gr.Slider(0, 5, label="Online boarding"),
#     gr.Slider(0, 5, label="Seat comfort"),
#     gr.Slider(0, 5, label="Inflight entertainment"),
#     gr.Slider(0, 5, label="On-board service"),
#     gr.Slider(0, 5, label="Leg room service"),
#     gr.Slider(0, 5, label="Baggage handling"),
#     gr.Slider(0, 5, label="Checkin service"),
#     gr.Slider(0, 5, label="Inflight service"),
#     gr.Slider(0, 5, label="Cleanliness"),
#     gr.Slider(0, 300, label="Departure Delay in Minutes"),
#     gr.Slider(0, 300, label="Arrival Delay in Minutes"),
# ]

# # Launch the interface
# gr.Interface(
#     fn=predict,
#     inputs=inputs,
#     outputs=gr.Text(label="Predicted Satisfaction"),
#     title="Airline Satisfaction Predictor",
#     description="Select a trained model and enter flight features to predict customer satisfaction.",
#     flagging_mode="never"
# ).launch()