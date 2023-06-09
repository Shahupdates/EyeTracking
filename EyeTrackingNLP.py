import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from eye_tracking_library import EyeTracker
from sklearn.ensemble import RandomForestClassifier

class EyeTrackingNLP:
    def __init__(self):
        self.eye_tracker = EyeTracker('dummy')
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
    
    
def main():
    # Initialize the EyeTrackingNLP instance
    etnlp = EyeTrackingNLP()

    # Initialize the video capture from the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If a frame was successfully captured
        if ret:
            # Process the frame with the EyeTracker
            eye_data = etnlp.eye_tracker.sample()

            # Use pytesseract to extract text from the frame
            text = pytesseract.image_to_string(frame)

            # Process the eye tracking data and the text data
            processed_eye_data = etnlp.process_eye_tracking_data(eye_data)
            processed_text_data = etnlp.process_text_data(text)

            # Combine the data and train the model
            combined_data = etnlp.combine_data(processed_eye_data, processed_text_data)
            etnlp.train_model(combined_data, labels)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to capture frame")
            break

    # After the loop release the cap object
    cap.release()
    cv2.destroyAllWindows()    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    text = "This is a simple test sentence"
    labels = ['category_1', 'category_2', 'category_1', 
          'category_2', 'category_1', 'category_2']
    eye_tracking_data = [(0.01, (50, 100)), (0.02, (60, 110)), (0.03, (65, 115)), 
                     (0.04, (70, 120)), (0.05, (75, 125)), (0.06, (80, 130))]
    main()
