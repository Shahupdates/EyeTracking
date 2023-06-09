# EyeTrackingNLP
This project utilizes eye tracking data and Natural Language Processing (NLP) to train a machine learning model, specifically a RandomForestClassifier from Scikit-learn. The purpose of this project is to explore and demonstrate the integration of eye tracking data with NLP.

# Overview
The project consists of a class EyeTrackingNLP which:

Interfaces with an eye tracker to capture gaze data.
Processes the eye tracking data.
Processes text data using NLTK for tokenization and POS tagging.
Combines eye tracking and text data.
Trains a RandomForestClassifier with the combined data.
Predicts labels for new data using the trained model.
Installation
Ensure you have the following packages installed:

Python 3.8+, OpenCV, pyGaze, NLTK, pytesseract, Scikit-learn
You can install these packages using pip:
```
pip install opencv-python pygaze nltk pytesseract scikit-learn
```

# Usage
To use this project:

Import the necessary modules and EyeTrackingNLP class.
Initialize the EyeTrackingNLP class.
Use the methods provided by the class to process eye tracking and text data, combine the data, train the model, and make predictions.
Please note that the project currently uses a dummy eye tracker and the OCR from pytesseract to extract text from video frames. Replace these with your own implementations as required.

# Limitations and Future Work
Currently, the project uses a dummy eye tracker and OCR to extract text. In the future, this project could be adapted to use real eye tracking devices and more accurate OCR methods.
The project assumes that the eye tracking data and text data can be processed separately. A future improvement could be to determine which part of the text the user is looking at based on the eye tracking data.
The project currently uses a RandomForestClassifier. Future work could explore the use of different machine learning models.

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License

# MIT - https://choosealicense.com/licenses/mit/
