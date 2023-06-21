import cv2
import dlib
import os


# Load the pre-trained face and landmark detection models
face_detector = dlib.get_frontal_face_detector()
current_directory = os.path.dirname(os.path.abspath(__file__))

landmark_predictor = dlib.shape_predictor('path_to_landmark_model.dat')

def extract_eye_regions(frame, landmarks):
    left_eye_points = [36, 37, 38, 39, 40, 41]
    right_eye_points = [42, 43, 44, 45, 46, 47]

    # Extract left and right eye regions from the landmarks
    left_eye_region = landmarks[left_eye_points[0]:left_eye_points[-1]+1]
    right_eye_region = landmarks[right_eye_points[0]:right_eye_points[-1]+1]

    # Convert the eye regions to bounding rectangles
    left_eye_rect = cv2.boundingRect(left_eye_region)
    right_eye_rect = cv2.boundingRect(right_eye_region)

    # Extract the eye regions from the frame
    left_eye_img = frame[left_eye_rect[1]:left_eye_rect[1]+left_eye_rect[3],
                         left_eye_rect[0]:left_eye_rect[0]+left_eye_rect[2]]
    right_eye_img = frame[right_eye_rect[1]:right_eye_rect[1]+right_eye_rect[3],
                          right_eye_rect[0]:right_eye_rect[0]+right_eye_rect[2]]

    return left_eye_img, right_eye_img

def generate_eye_tracking_data(video_folder):
    # Iterate through all videos in the folder
    for video_file in os.listdir(video_folder):
        if video_file.endswith('.mp4'):
            video_folder = os.path.join(current_directory, 'video_folder')

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = face_detector(gray)

                for face in faces:
                    # Detect landmarks for the face
                    landmarks = landmark_predictor(gray, face)

                    # Extract eye regions from the frame
                    left_eye_img, right_eye_img = extract_eye_regions(frame, landmarks)

                    # Perform eye tracking analysis on the eye regions
                    # Add your eye tracking algorithm code here

                    # Display the eye regions with bounding rectangles
                    cv2.imshow('Left Eye', left_eye_img)
                    cv2.imshow('Right Eye', right_eye_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

# Specify the folder containing the videos
video_folder = 'path_to_video_folder'

# Generate eye tracking data from the videos in the folder
generate_eye_tracking_data(video_folder)
