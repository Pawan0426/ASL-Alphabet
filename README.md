# ASL Alphabet Recognition using Random Forest

This project utilizes machine learning to recognize American Sign Language (ASL) alphabet signs using the Random Forest classifier. The application uses real-time hand gesture recognition through the webcam, providing immediate feedback on predicted signs.

## Dataset
The ASL Alphabet dataset used in this project can be found [here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/code). It contains individual folders for each ASL alphabet letter, including 'A' to 'Z', 'space', 'delete', and 'nothing'.

## Requirements
To run the project, ensure you have the following libraries installed:
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- Pickle (`pickle`)

You can install these libraries using pip:


## Usage
1. **Setup**: Clone the repository and install the required libraries.
   
2. **Model Loading**: The trained Random Forest model (`asl_rf_model.pkl`) is loaded using Pickle. It predicts the ASL signs based on hand landmarks detected by MediaPipe.

3. **Real-time Prediction**: Run the script `cv2Signlan.py` to open your webcam and start predicting ASL signs based on your hand gestures.

4. **Recording**: The script records the webcam feed and overlays predicted labels on the video frames. It saves the output to `output.avi`.

5. **Termination**: Press 'q' on your keyboard to exit the application.

## Additional Notes
- The model was trained on hand landmarks extracted by MediaPipe's hand tracking solution, ensuring accurate recognition of ASL gestures in real-time.
- Feel free to customize the model or integrate other machine learning techniques for further improvement.

## Contact
For any questions or suggestions, feel free to reach out to the repository owner.

