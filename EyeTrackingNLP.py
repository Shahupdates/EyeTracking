import cv2
import pytesseract
import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.ensemble import RandomForestClassifier

class EyeTrackingNLP:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.model = RandomForestClassifier()


    def process_data(self, data):
        processed_data = []
        for timestamp, fixation_point in data:
            x, y = fixation_point
            # Scale the fixation points
            new_x = x / 1000
            new_y = y / 1000
            processed_data.append((timestamp, (new_x, new_y)))
        return processed_data

    def process_eye_tracking_data(self, eye_tracking_data):
        """
        Processes raw eye tracking data using the EyeTracker's process_data method.

        Args:
        eye_tracking_data: A list of raw eye tracking data.

        Returns:
        A list of processed eye tracking data.
        """
        processed_data = self.eye_tracker.process_data(eye_tracking_data)
        return processed_data

    def process_text_data(self, text):
        """
        Tokenizes a text and performs POS tagging.

        Args:
        text: A string containing the text data to be processed.

        Returns:
        A list of tuples where each tuple contains a token and its POS tag.
        """
        tokenized_text = word_tokenize(text)
        pos_tagged_text = pos_tag(tokenized_text)
        return pos_tagged_text

    def combine_data(self, processed_eye_data, processed_text_data):
        """
        Combines processed eye tracking data and processed text data.

        Args:
        processed_eye_data: A list of processed eye tracking data.
        processed_text_data: A list of tuples where each tuple contains a token and its POS tag.

        Returns:
        A list of tuples where each tuple contains a piece of eye tracking data and a token/POS tag pair.
        """
        combined_data = list(zip(processed_eye_data, processed_text_data))
        return combined_data

    def train_model(self, combined_data, labels):
        """
        Trains the RandomForestClassifier using combined eye tracking and text data.

        Args:
        combined_data: A list of tuples where each tuple contains a piece of eye tracking data and a token/POS tag pair.
        labels: A list of labels corresponding to the combined_data.

        """
        features, _ = zip(*combined_data)
        self.model.fit(features, labels)

    def predict(self, new_data):
        """
        Uses the trained RandomForestClassifier to make predictions.

        Args:
        new_data: A list of new combined eye tracking and text data to be predicted.

        Returns:
        A list of predictions made by the model.
        """
        features, _ = zip(*new_data)
        predictions = self.model.predict(features)
        return predictions


def process_video(etnlp, labels, text_box, result_box):
    # Initialize the video capture from the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If a frame was successfully captured
        if ret:
            # Convert the frame to grayscale for eye detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect eyes using the Haar cascade classifier
            eyes = etnlp.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process the eye tracking data and the text data
            processed_eye_data = eyes
            text = text_box.get()
            processed_text_data = etnlp.process_text_data(text)

            # Combine the data and train the model
            combined_data = etnlp.combine_data(processed_eye_data, processed_text_data)
            etnlp.train_model(combined_data, labels)

            # Display the frame with eye tracking rectangles
            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the processed frame in the GUI
            cv2.imshow('Eye Tracking', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to capture frame")
            break

    # After the loop, release the cap object
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Initialize the EyeTrackingNLP instance
    etnlp = EyeTrackingNLP()

    root = tk.Tk()

    # Create a textbox for input
    text_label = tk.Label(root, text='Enter your text:')
    text_label.pack()
    text_box = tk.Entry(root, width=50)
    text_box.pack()

    # Create a display box for results
    result_label = tk.Label(root, text='Results:')
    result_label.pack()
    result_box = scrolledtext.ScrolledText(root, width=50, height=10)
    result_box.pack()

    # Create start and stop buttons
    start_button = tk.Button(root, text='Start', command=lambda: threading.Thread(target=process_video,
                                                                                 args=(etnlp, labels)).start())
    start_button.pack()

    stop_button = tk.Button(root, text='Stop', command=lambda: cv2.destroyAllWindows())
    stop_button.pack()

    # Start the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    text = "This is a simple test sentence"
    labels = ['category_1', 'category_2', 'category_1',
              'category_2', 'category_1', 'category_2']
    eye_tracking_data = [(0.01, (50, 100)), (0.02, (60, 110)), (0.03, (65, 115)),
                         (0.04, (70, 120)), (0.05, (75, 125)), (0.06, (80, 130))]
    main()
