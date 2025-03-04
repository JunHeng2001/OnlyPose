import cv2
from poseFunctions import *
import time
import joblib

import requests
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

class BicepCurlClassifierTorch(nn.Module):
    def __init__(self, input_dim, lstm_units=64, dropout_rate=0.2):
        super(BicepCurlClassifierTorch, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_units, lstm_units // 2)
        self.fc2 = nn.Linear(lstm_units // 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take output from the last time step
        x = self.dropout(lstm_out)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)
    
# Load the trained PyTorch model
imu_model = BicepCurlClassifierTorch(input_dim=7)  # Initialize model with correct input_dim
imu_model.load_state_dict(torch.load(r"C:\Users\nalyn\Desktop\cs3237\bicep_curl_model_state.pth"))  # Replace with your model path
imu_model.eval()

#----------------------------------------------------------------------------------------

# Replace with your bot's token and chat ID
bot_token = "7123271616:AAGq8DyZ1IDw_7M4jG0IdHoOxQxDDyUOQjk"
chat_id = "1939482390"

# Feedback
correct_imu = "Well done! Wrist done right"
wrong_imu =  "Try again, please straighten your wrist"

wrong_back = "Rep not counted, please straighten your back"
wrong_arm = "Rep not counted, please pin your elbows to the side"
correct_cam = "Well done! Rep #"
wrong_cam = "Rep not counted, please straighten your back and pin your elbows to the side!"

#----------------------------------------------------------------------------------------

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("Message sent successfully.")
    else:
        print(f"Failed to send message. Error: {response.status_code}")

#----------------------------------------------------------------------------------------

# Define route to handle incoming IMU data
@app.route('/predict/imu', methods=['POST'])

def predict_imu():
    data = request.json  # Get JSON data
    # print(data)
    print('Data received')
    imu_data = np.array(data["sensor_data"])  # Convert to numpy array
    
    # Ensure data is reshaped to match model's expected input (12x7)
    imu_tensor = torch.tensor(imu_data).float().unsqueeze(0)  # Shape (1, 12, 7)
    # print("IMU Tensor Shape:", imu_tensor.shape)

    print('Processing IMU Data')
    # Make prediction
    with torch.no_grad():
        output = imu_model(imu_tensor)
    
    # Process the output (e.g., apply softmax or threshold)
    prediction = (output > 0.5).int().item()  # Convert sigmoid output to binary  

    if prediction == 0:
        send_telegram_message(bot_token, chat_id, wrong_imu)        # Tele    ------------------------

    # Print the prediction to the server console
    if prediction == 0:
        print('Straighten wrist')
    else:
        print('Good wrist form!')

    # Send the prediction back to ESP32
    print()
    return jsonify({"prediction": prediction})

#----------------------------------------------------------------------------------------

def connect_to_camera(url):
    while True:
        try:
            print("hi")
            response = requests.get(url[:-10], timeout=2)  # Set a 2-second timeout
            if response.status_code == 200:
                print("Camera available")
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        print("Successfully connected to camera.")
                        return cap
                    else:
                        print("No frames received. Retrying...")
                else:
                    print("Camera not found. Retrying...")
                cap.release()  # Ensure resources are released if the connection fails
        except requests.RequestException as e:
            print(f"Error checking camera availability: {e}")
        time.sleep(2)  # Wait before retrying

def video_pose_detection():
    coordinates = "Not in-frame"
    counter = 0
    predictions = None
    elbow_pred = None
    uparm_state = "good"
    back_state = "good"
    last_time = time.time()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    back_model = joblib.load(r"C:\Users\nalyn\Desktop\cs3237\back_model.pkl")
    uparm_model = joblib.load(r"C:\Users\nalyn\Desktop\cs3237\uparm_model.pkl")

    # cap = cv2.VideoCapture(0)
    URL = "http://192.168.191.73:81/stream"    # ESP32-CAM   Malc's hotspot
    # URL = "http://172.20.10.9:81/stream"    # ESP32-CAM   Alyn's hotspot
    cap = connect_to_camera(URL)
    set_resolution(URL, index=7)

    ESP32_out = "http://"
    ESP32_out += "192.168.191.79"  # ESP32's IP    Malc's hotspot
    # ESP32_out += "172.20.10.5"  # ESP32's IP    Alyn's hotspot

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            cap.release()
            cap = connect_to_camera(URL)  # Reconnect if the stream fails
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            current_time = time.time()
            if current_time - last_time >= 0.2:
                last_time = current_time
                coordinates = body_coordinates(landmarks)

                if coordinates == "Not in-frame" or coordinates == "Front":
                    body_side = coordinates

                else:
                    side, shoulder, hip, elbow, wrist, index = coordinates
                    body_side = f"{side} side"

                    back_angle = find_angle(shoulder, hip, np.array([hip[0], 0]))
                    uparm_angle = find_angle(shoulder, elbow, hip)
                    wrist_angle = find_angle(elbow, wrist, index)
                    elbow_angle = find_angle(shoulder, elbow, wrist)

                    feature = np.array([[back_angle, uparm_angle, wrist_angle, elbow_angle]])
                    back_pred = str(back_model.predict(feature)[0])
                    uparm_pred = str(uparm_model.predict(feature)[0])
                    if back_pred == "b":
                        back_state = "bad"
                    if uparm_pred == "b":
                        uparm_state = "bad"

                    if elbow_angle > 160 and elbow_pred is None:
                        elbow_pred = "down"
                    elif elbow_angle < 90 and elbow_pred == "down":
                        elbow_pred = "up"
                    elif elbow_angle > 160 and elbow_pred == "up" and back_state == "good" and uparm_state == "good":
                        elbow_pred = "down"
                        counter += 1
                        print("Correct")
                        try:
                            requests.get(f"{ESP32_out}/LED4_ON", timeout=2)    # Turn on LED
                        except requests.RequestException:
                            print("No ESP32 output")
                        # correct_cam += str(counter)   # doesn't work
                        send_telegram_message(bot_token, chat_id, correct_cam)      # Tele    ------------------------

                    elif elbow_angle > 160 and elbow_pred == "up" and (back_state == "bad" or uparm_state == "bad"):
                        try:
                            requests.get(f"{ESP32_out}/LED2_ON", timeout=2)    # Turn on LED
                        except requests.RequestException:
                            print("No ESP32 output")
                        if back_state == "bad" and uparm_state == "bad":
                            print("All wrong")
                            send_telegram_message(bot_token, chat_id, wrong_cam)    # Tele    ------------------------
                        elif back_state == "bad" and uparm_state == "good":
                            print("Back wrong")
                            send_telegram_message(bot_token, chat_id, wrong_back)   # Tele    ------------------------
                        elif back_state == "good" and uparm_state == "bad":
                            print("Arm wrong")
                            send_telegram_message(bot_token, chat_id, wrong_arm)    # Tele    ------------------------

                        back_pred = "a"
                        back_state = "good"
                        uparm_pred = "a"
                        uparm_state = "good"
                        elbow_pred = "down"

                    predictions = f"{back_state}, {uparm_state}, {elbow_pred}"

            cv2.putText(frame, body_side, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, predictions, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
            cv2.putText(frame, f"Reps: {counter}, {elbow_pred}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Prediction", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close() 

# Run the server
if __name__ == '__main__':
    # Start the Flask app in a separate thread or process to avoid blocking
    import threading
    flask_thread = threading.Thread(target=app.run, kwargs={'host': "0.0.0.0", 'port': 5000})
    flask_thread.start()

    # Start video pose detection
    video_pose_detection()