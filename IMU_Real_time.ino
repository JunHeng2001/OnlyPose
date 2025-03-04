#include <WiFi.h>
#include <HTTPClient.h>
#include <MPU6500_WE.h>
#include <Wire.h>

#define MPU6500_ADDR 0x68

// Wi-Fi credentials
const char* ssid = "Malcolm";
const char* password = "mala1733";

// Server URL
const String serverURL_imu = "http://192.168.191.48:5000/predict/imu";

// MPU6500 instance
MPU6500_WE myMPU6500 = MPU6500_WE(MPU6500_ADDR);

// IMU data buffer
float imuData[12][7];  // Buffer for 12 samples, 7 features per sample
int sampleIndex = 0;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  Wire.begin();
  if (!myMPU6500.init()) {
    Serial.println("MPU6500 does not respond");
    while (1);
  }
  Serial.println("MPU6500 is connected");

  // Calibration
  calibrateMPU6500();

  // Sensor settings
  myMPU6500.enableGyrDLPF();
  myMPU6500.setGyrDLPF(MPU6500_DLPF_6);
  myMPU6500.setSampleRateDivider(5);
  myMPU6500.setGyrRange(MPU6500_GYRO_RANGE_250);
  myMPU6500.setAccRange(MPU6500_ACC_RANGE_2G);
  myMPU6500.enableAccDLPF(true);
  myMPU6500.setAccDLPF(MPU6500_DLPF_6);
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // Read IMU data
    xyzFloat accel = myMPU6500.getGValues();
    xyzFloat gyr = myMPU6500.getGyrValues();
    float resultantG = myMPU6500.getResultantG(accel);

    // Store the sample
    imuData[sampleIndex][0] = accel.x;
    imuData[sampleIndex][1] = accel.y;
    imuData[sampleIndex][2] = accel.z;
    imuData[sampleIndex][3] = gyr.x;
    imuData[sampleIndex][4] = gyr.y;
    imuData[sampleIndex][5] = gyr.z;
    imuData[sampleIndex][6] = resultantG;
    sampleIndex++;

    // Check if we have collected 12 samples
    if (sampleIndex == 12) {
      sendDataToServer();
      sampleIndex = 0;  // Reset for the next batch of samples
    }

    delay(100);  // Adjust delay as needed
  }
}

void sendDataToServer() {
  HTTPClient http;
  http.begin(serverURL_imu);
  http.addHeader("Content-Type", "application/json");

  // Create JSON payload
  String payload = "{\"sensor_data\":[";
  for (int i = 0; i < 12; i++) {
    payload += "[";
    for (int j = 0; j < 7; j++) {
      payload += String(imuData[i][j]);
      if (j < 6) payload += ",";
    }
    payload += "]";
    if (i < 11) payload += ",";
  }
  payload += "]}";

  // Send POST request
  int httpResponseCode = http.POST(payload);
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("IMU Response: " + response);
  } else {
    Serial.println("Error in sending POST request");
  }
  http.end();
}

void calibrateMPU6500() {
  Serial.println("Position your MPU6500 flat and don't move it - calibrating...");
  delay(1000);
  myMPU6500.autoOffsets();
  Serial.println("Calibration done!");
}

