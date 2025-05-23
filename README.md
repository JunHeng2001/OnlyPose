# OnlyPose

Our project solution, OnlyPose, will focus on one type of workout, the dumbbell curl. Taking inspiration from IPPT ELISS machines that are used by the SAF for fitness testing, it will monitor the user’s form during the workout and provide instantaneous feedback to the user. For example, during a push up, it might detect that the back is not straight and thus plays an audio message “Back not straightened” and the rep will not be counted. This will help the user to correct the form for the next repetition (rep), which will reduce the risk of injury, eliminate the cost of finding a personal trainer and promote self-awareness of proper form. 

# Overview

<div align="center">
  <img src="Screenshot 2025-04-28 220030.png" alt="Screenshot" />
</div>


The system consists of 3 main components: sensors (in purple), a data server (in yellow) and user feedback mechanisms (in green).
Data on the user’s posture is collected by the sensors, which include an IMU and a camera. For this project, we are using the MPU6500 as the IMU and the ESP32-CAM as the camera. The data server, referred to as the “HTTP server”, gathers and processes this data using ML models to predict whether the user’s posture is correct or wrong.

<div align="center">
  <img src="Screenshot 2025-04-28 222934.png" alt="Screenshot" />
</div>

These predictions control the activation of LEDs and send real-time text messages to the user via Telegram, providing immediate visual feedback, and highlighting necessary corrections when the form is incorrect.

<div align="center">
  <img src="Screenshot 2025-04-28 222054.png" alt="Screenshot" />
</div>

For the prototype, we have mounted the circuit board directly onto a dumbbell. Cardboard spacers are placed beneath the circuit board to provide sufficient space for the user to grip the handlebar comfortably. The flat surface of the cardboard also ensures an optimal position for calibrating the IMU sensor. Once the IMU sensor is calibrated, an LED indicator will turn green, signaling that the dumbbell is ready for use. To power the device, a power bank is attached to the right side of the dumbbell, reducing the need for a stationary power supply and enhancing portability.

# Conclusion
OnlyPose is a smart IoT-based solution designed to monitor exercise form during dumbbell curls and provide real-time corrective feedback. By leveraging an IMU sensor and ESP32-CAM, coupled with ML models, we demonstrated that IoT technology can bridge the gap between expensive personal training and self-guided fitness routines. OnlyPose demonstrates how an affordable, simple-to-use interface for dumbbells can revolutionize the fitness industry with gadgets that enable real-time monitoring and improvements to the user’s form, thereby reducing the risk of injury and improving their overall  workout efficiency.
