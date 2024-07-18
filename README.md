# Drowsiness Detection System

This project implements a drowsiness detection system using OpenCV and dlib. It monitors eye movements and alerts the user if drowsiness is detected.

## Features

- Real-time video stream processing
- Face detection using dlib
- Eye aspect ratio (EAR) calculation
- Alarm sound when drowsiness is detected
- Optional integration with Tesla API for activating autopilot and hazard lights (requires a valid access token)

## Requirements

- Python 3.x
- OpenCV
- dlib
- imutils
- pygame (for playing the alarm sound)
- Tesla API access token (optional)

## Usage

1. Clone this repository.
2. Install the required dependencies (`pip install -r requirements.txt`).
3. Run the script (`python drowsiness_detection.py`).
4. Press 'q' to exit the application.

## Configuration

- Adjust the `thresh` and `frame_check` values in the script to fine-tune the drowsiness detection sensitivity.
- Replace `<access_token>` with your actual Tesla API access token if you want to activate autopilot and hazard lights.

## Acknowledgments

- dlib
- OpenCV
- Tesla API
