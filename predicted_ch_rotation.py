import pandas as pd
import joblib

# Load the saved model
print("Loading trained model...")
model = joblib.load("ch_rotation_model.pkl")
print("Model loaded successfully.")

# Load new data
new_data = pd.read_csv("newest_dataset.csv")  # Should contain TrafficLoad, PacketReceived, ResidualEnergy, DistanceToBS
print("New dataset loaded successfully.")

# Select features
features = new_data[['TrafficLoad', 'PacketReceived', 'ResidualEnergy', 'DistanceToBS']]

# Predict Optimal CH Rotation
print("Performing predictions on new data...")
predictions = model.predict(features)

# Add predictions to the dataframe
new_data['Predicted_CH_Rotation'] = predictions
print("Predictions added to the dataset.")

# Save the results to a new file
new_data.to_csv("predicted_ch_rotation1.csv", index=False)
print("Predicted results saved as 'predicted_ch_rotation1.csv' ")

