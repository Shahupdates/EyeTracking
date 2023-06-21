# NLP Eye Tracking

This project combines eye tracking data and Natural Language Processing (NLP) techniques to train a machine learning model, specifically a RandomForestClassifier from Scikit-learn. The goal of this project is to integrate eye tracking with NLP for analysis and prediction.

## Files

The project consists of the following files:

- `eyetrackingnlp.py`: Contains the EyeTrackingNLP class, which handles the processing of eye tracking data, text data, model training, and prediction.
- `eye_tracking_data_generator.py`: A script used to generate eye tracking data from videos using the Haar cascade classifier for eye detection.
- `README.md`: The project documentation you are currently reading.

## EyeTrackingNLP.py

The `eyetrackingnlp.py` file implements the EyeTrackingNLP class, which provides the core functionality for integrating eye tracking with NLP. It includes the following methods:

- `__init__(self)`: Initializes the EyeTrackingNLP instance, loading the Haar cascade classifier for eye detection and creating a RandomForestClassifier model.
- `process_data(self, data)`: Processes raw eye tracking data by scaling the fixation points.
- `process_eye_tracking_data(self, eye_tracking_data)`: Processes raw eye tracking data using the Haar cascade classifier for eye detection.
- `process_text_data(self, text)`: Tokenizes and performs POS tagging on the input text data.
- `combine_data(self, processed_eye_data, processed_text_data)`: Combines processed eye tracking data and processed text data.
- `train_model(self, combined_data, labels)`: Trains the RandomForestClassifier model using the combined eye tracking and text data.
- `predict(self, new_data)`: Makes predictions using the trained RandomForestClassifier model.

## Eye Tracking Data Generation

To generate eye tracking data from videos, you can use the `eye_tracking_data_generator.py` script. This script captures video frames, detects eyes using the Haar cascade classifier, and generates eye tracking data. The generated data can be used for training the EyeTrackingNLP model.

To use the `eye_tracking_data_generator.py` script, ensure you have the necessary dependencies installed (e.g., OpenCV). Execute the script and provide the path to the video folder as a command-line argument.

Example usage:
``` python eye_tracking_data_generator.py --video_folder path_to_video_folder ```

## Pre-trained Landmark Detection Model

In the future, you may consider using a pre-trained landmark detection model to improve eye tracking accuracy. A landmark detection model can provide more precise eye region localization, leading to better eye tracking results.

To generate a pre-trained landmark detection model, you would need to collect a dataset of images with annotated eye landmarks and use tools like dlib's shape_predictor_trainer to train the model. The exact steps and process depend on the specific tools and requirements.

Using a pre-trained landmark detection model with eye tracking would involve integrating the model into the eye tracking pipeline. The landmarks detected by the model can be used to extract eye regions for further analysis and tracking.

Please note that the use of a pre-trained landmark detection model is not implemented in the current version of the project, but it can be a potential enhancement for improved eye tracking accuracy.

## Usage

To use the EyeTrackingNLP class, follow these steps:

1. Import the necessary modules and classes:
```python
import cv2
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
```

2. Create an instance of the EyeTrackingNLP class:
```
etnlp = EyeTrackingNLP()
```

4. Process eye tracking data and text data:
```
processed_eye_data = etnlp.process_eye_tracking_data(eye_tracking_data)
processed_text_data = etnlp.process_text_data(text)
```

5. Combine the processed data:
```
combined_data = etnlp.combine_data(processed_eye_data, processed_text_data)
```

6. Train the model:
```
etnlp.train_model(combined_data, labels)
```

Please note that you need to have the necessary dependencies installed, such as OpenCV, pytesseract, and nltk,
you can install them using pip: 
```
pip install opencv-python pygaze nltk pytesseract scikit-learn
```

If you encounter any issues during the installation, please refer to the respective package's documentation for further guidance.

# License
This project is licensed under the MIT License.

Feel free to modify and customize the README.md as needed, and let me know if you require any further assistance!


7. Make predictions:
```
predictions = etnlp.predict(new_data)
```
