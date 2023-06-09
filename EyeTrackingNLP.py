import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from eye_tracking_library import EyeTracker
from sklearn.ensemble import RandomForestClassifier

class EyeTrackingNLP:
    def __init__(self):
        self.eye_tracker = EyeTracker()
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

if __name__ == "__main__":
    etnlp = EyeTrackingNLP()
    processed_eye_data = etnlp.process_eye_tracking_data(eye_tracking_data)
    processed_text_data = etnlp.process_text_data(text)
    combined_data = etnlp.combine_data(processed_eye_data, processed_text_data)
    etnlp.train_model(combined_data, labels)
