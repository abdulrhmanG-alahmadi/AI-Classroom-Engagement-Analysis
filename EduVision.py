import os
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


def detect_persons(video_path, output_dir):
    """
    Detect persons in the video and save cropped images.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the cropped person images.
    """
    # Set the device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pre-trained Faster R-CNN model with ResNet50-FPN backbone
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame rate of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the number of frames to skip (process every 5th frame)
    skip_frames = int(frame_rate / 5)
    
    # Initialize variables for frame ID and person count
    frame_id = 0
    person_count = 0
    
    # Set the margin for cropping the person images
    margin = 20

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames are available

        # Process every 5th frame
        if frame_id % skip_frames == 0:
            # Convert the frame to a tensor and move it to the device
            input_tensor = to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            
            # Perform person detection using the model
            with torch.no_grad():
                prediction = model(input_tensor)

            # Process the detected persons
            for element in range(len(prediction[0]['boxes'])):
                box = prediction[0]['boxes'][element].cpu().numpy()
                score = prediction[0]['scores'][element].cpu().numpy()
                label_idx = prediction[0]['labels'][element].cpu().numpy()
                
                # Check if the detected object is a person with a confidence score above 0.5
                if score > 0.5 and label_idx == 1:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Add margin to the bounding box coordinates
                    x1 -= margin
                    y1 -= margin
                    x2 += margin
                    y2 += margin
                    
                    # Ensure the coordinates are within the frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # Crop the person image from the frame
                    cropped_image = frame[y1:y2, x1:x2]
                    person_count += 1
                    
                    # Save the cropped person image
                    image_path = os.path.join(output_dir, f'person_frame_{frame_id}_{person_count}.jpg')
                    cv2.imwrite(image_path, cropped_image)

        frame_id += 1

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()


def classify_face_orientations(input_dir, output_dir):
    """
    Classify face orientations in the input images and save the results.
    
    Args:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the classified images.
        
    Returns:
        dict: Dictionary containing the counts of each face orientation class.
    """
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Initialize MediaPipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    # Initialize a dictionary to store the counts of each face orientation class
    class_counts = {}

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Convert the image from BGR to RGB color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks using MediaPipe FaceMesh
            results = face_mesh.process(image_rgb)
            
            # Create a copy of the image for drawing
            image_drawn = image.copy()

            # Process the detected face landmarks
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get the image dimensions
                    img_h, img_w, _ = image.shape
                    
                    # Initialize lists to store 2D and 3D face landmarks
                    face_2d = []
                    face_3d = []
                    
                    # Extract specific face landmarks (nose tip, chin, left and right eyes, left and right ears)
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, landmark.z])

                    # Check if face landmarks are available
                    if face_2d and face_3d:
                        # Convert the face landmarks to NumPy arrays
                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)
                        
                        # Estimate the focal length based on image width
                        focal_length = img_w
                        
                        # Define the camera matrix
                        cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                               [0, focal_length, img_h / 2],
                                               [0, 0, 1]], dtype='double')
                        
                        # Define the distortion coefficients
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)
                        
                        # Solve the Perspective-n-Point (PnP) problem to estimate the head pose
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        
                        # Convert the rotation vector to a rotation matrix
                        rmat, jac = cv2.Rodrigues(rot_vec)
                        
                        # Extract the Euler angles (pitch, yaw, roll) from the rotation matrix
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

                        # Classify the face orientation based on the yaw and pitch angles
                        if y < -8:
                            text = "Looking_Left"
                        elif y > 8:
                            text = "Looking_Right"
                        elif x < -8:
                            text = "Looking_Down"
                        elif x > 8:
                            text = "Looking_Up"
                        else:
                            text = "Looking_Forward"

                        # Create the output directory for the classified image
                        class_directory = os.path.join(output_dir, text)
                        if not os.path.exists(class_directory):
                            os.makedirs(class_directory)

                        # Generate the new filename with the face orientation label
                        new_filename = f"{text}_{filename}"
                        save_path = os.path.join(class_directory, new_filename)
                        
                        # Draw the face mesh landmarks on the image
                        mp_drawing.draw_landmarks(
                            image=image_drawn,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                        
                        # Save the classified image with the face mesh landmarks
                        cv2.imwrite(save_path, image_drawn)
                        
                        # Update the count of the face orientation class
                        class_counts[text] = class_counts.get(text, 0) + 1

    # Close all windows
    cv2.destroyAllWindows()
    
    return class_counts


def detect_hand_raised(input_dir, output_dir):
    """
    Detect hand raised in the input images and save the results.
    
    Args:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the images with hand raised detected.
        
    Returns:
        int: Count of images with hand raised.
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize the count of images with hand raised
    hand_raised_counts = 0

    def calculate_angle(a, b, c):
        """
        Calculate the angle between three points.
        
        Args:
            a (tuple): Coordinates of point a.
            b (tuple): Coordinates of point b.
            c (tuple): Coordinates of point c.
            
        Returns:
            float: Angle in degrees.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    # Process each image in the input directory
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        left_stage, right_stage = None, None
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                try:
                    landmarks = results.pose_landmarks.landmark
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    # Calculate the angles between shoulder, elbow, and wrist for left and right arms
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    
                    # Get the image dimensions
                    image_height, image_width, _ = image.shape
                    
                    # Convert the normalized wrist coordinates to pixel coordinates
                    left_wrist_pixel = (int(left_wrist[0] * image_width), int(left_wrist[1] * image_height))
                    right_wrist_pixel = (int(right_wrist[0] * image_width), int(right_wrist[1] * image_height))

                    # Create a copy of the image for drawing the skeleton
                    image_with_skeleton = image.copy()
                    
                    # Draw the pose landmarks and connections on the image
                    mp_drawing.draw_landmarks(image_with_skeleton, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Add text annotations for the left and right arm angles and wrist coordinates
                    cv2.putText(image_with_skeleton, f'Left Angle: {left_angle:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image_with_skeleton, f'Right Angle: {right_angle:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image_with_skeleton, f'Left Wrist: {left_wrist_pixel}', left_wrist_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image_with_skeleton, f'Right Wrist: {right_wrist_pixel}', right_wrist_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                    # Check if the left wrist is above a certain threshold and the previous stage was "down"
                    if left_wrist[1] < 0.3 and left_stage == 'down':
                        left_stage = "up"
                        output_path = os.path.join(output_dir, filename)
                        cv2.imwrite(output_path, image_with_skeleton)
                        hand_raised_counts += 1
                    elif left_angle > 160:
                        left_stage = "down"
                    
                    # Check if the right wrist is above a certain threshold and the previous stage was "down"
                    if right_wrist[1] < 0.3 and right_stage == 'down':
                        right_stage = "up"
                        output_path = os.path.join(output_dir, filename)
                        cv2.imwrite(output_path, image_with_skeleton)
                        hand_raised_counts += 1
                    elif right_angle > 160:
                        right_stage = "down"
                except:
                    pass
    return hand_raised_counts


def detect_phones(input_dir, output_dir):
    """
    Detect phones in the input images and save the results.
    
    Args:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the images with phones detected.
        
    Returns:
        dict: Dictionary containing the counts of images with phones detected.
    """
    # Set the device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pre-trained Faster R-CNN model with ResNet50-FPN backbone
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Define the class mapping for the phone class
    class_mapping = {77: "Phone"}
    
    # Initialize a dictionary to store the counts of images with phones detected
    phone_counts = {}

    # Process each class directory in the input directory
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            # Process each image in the class directory
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(class_path, filename)
                    
                    # Convert the image to a tensor and move it to the device
                    input_image = to_tensor(cv2.imread(image_path)).unsqueeze(0).to(device)
                    
                    # Perform object detection using the model
                    with torch.no_grad():
                        prediction = model(input_image)
                    
                    # Read the original image
                    original_image = cv2.imread(image_path)
                    
                    # Initialize a list to store the detected objects
                    detected_objects = []

                    # Process the detected objects
                    for element in range(len(prediction[0]['boxes'])):
                        box = prediction[0]['boxes'][element].cpu().numpy()
                        score = prediction[0]['scores'][element].cpu().numpy()
                        label_idx = prediction[0]['labels'][element].cpu().item()

                        # Check if the detected object is a phone with a confidence score above 0.5
                        if label_idx in class_mapping and score > 0.5:
                            detected_objects.append(class_mapping[label_idx])
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Draw a rectangle around the detected phone
                            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add a label with the class name and confidence score
                            cv2.putText(original_image, f'{class_mapping[label_idx]}: {score:.2f}',
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Check if any phones were detected
                    if detected_objects:
                        # Create a string representation of the detected objects
                        detected_objects_str = "_".join(sorted(set(detected_objects)))
                        
                        # Create the output directory for the detected objects
                        output_directory = os.path.join(output_dir, f"{class_name}_Detected_{detected_objects_str}")
                        if not os.path.exists(output_directory):
                            os.makedirs(output_directory)
                        
                        # Generate the new filename with the detected objects
                        new_filename = f"{class_name}_{detected_objects_str}_{filename}"
                        output_path = os.path.join(output_directory, new_filename)
                        
                        # Save the image with the detected phones
                        cv2.imwrite(output_path, original_image)
                        
                        # Update the count of images with phones detected
                        phone_counts[new_filename] = phone_counts.get(new_filename, 0) + 1

    return phone_counts


def count_files(directory):
    """
    Count the number of files in the given directory, assuming they are images.
    
    Args:
        directory (str): Directory to count the files in.
        
    Returns:
        int: Number of image files in the directory.
    """
    return len([name for name in os.listdir(directory) if name.endswith(('.png', '.jpg', '.jpeg'))])


def plot_final_chart(counts_dict):
    """
    Plot the final chart showing the counts of classified images and detected phones.
    
    Args:
        counts_dict (dict): Dictionary containing the counts of classified images and detected phones.
    """
    # Define the paths for the classified and detected images
    path_classified = 'Classified_Images'
    path_detected = 'Classified_Images_Detected'
    path_hand_raised = os.path.join(path_classified, 'Hand_Raised')
    
    # Define the class names
    class_names = ['Looking_Down', 'Looking_Forward', 'Looking_Left', 'Looking_Right', 'Looking_Up']
    
    # Count the number of classified images for each class
    classified_counts = {class_name: count_files(os.path.join(path_classified, class_name))
                         for class_name in class_names}
    
    # Count the number of images with phones detected for each class
    detected_counts = {class_name + '_Detected_Phone': count_files(os.path.join(path_detected, class_name + '_Detected_Phone'))
                       for class_name in class_names}

    # --- First figure ---

    # Define the categories and counts for the first figure
    categories = class_names
    total_classified = [classified_counts[class_name] for class_name in class_names]
    total_detected = [detected_counts[class_name + '_Detected_Phone'] for class_name in class_names]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the bar width
    bar_width = 0.35
    
    # Set the index for the bars
    index = range(len(categories))

    # Create the bars for classified images and detected phones
    rects1 = ax.bar(index, total_classified, bar_width, color='royalblue', label='Classified Images')
    rects2 = ax.bar([p + bar_width for p in index], total_detected, bar_width, color='seagreen', label='Detected Phone')

    # Add labels and title to the axis
    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')
    ax.set_title('EduVision - Figure 1')
    
    # Set the tick locations and labels for the x-axis
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels(categories)
    
    # Add a legend
    ax.legend()

    def add_labels(rects):
        """
        Add labels to the top of each bar.
        
        Args:
            rects (list): List of bar rectangles.
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Add labels to the bars
    add_labels(rects1)
    add_labels(rects2)

    # Adjust the layout and display the figure
    plt.tight_layout()
    plt.show()

    # --- Second figure ---

    # Count the number of images with hand raised in the classified images
    hand_raised_classified = count_files(path_hand_raised)
    
    # Assuming no images with hand raised and phone detected (can be updated if needed)
    hand_raised_detected = 0
    
    # Calculate the counts for distracted, focused, and hand raised categories
    distracted_classified = sum([classified_counts[class_name] for class_name in ['Looking_Up', 'Looking_Down', 'Looking_Left', 'Looking_Right']])
    distracted_detected = sum([detected_counts[class_name + '_Detected_Phone'] for class_name in ['Looking_Up', 'Looking_Down', 'Looking_Left', 'Looking_Right']])
    distracted_with_phone = detected_counts['Looking_Down_Detected_Phone']
    focused_classified = classified_counts['Looking_Forward']
    focused_detected = detected_counts['Looking_Forward_Detected_Phone']

    # Define the categories and counts for the second figure
    categories = ['Distracted', 'Distracted with Phone', 'Focused', 'Hand Raised']
    totals_classified = [distracted_classified, distracted_with_phone, focused_classified, hand_raised_classified]
    totals_detected = [distracted_detected, distracted_with_phone, focused_detected, hand_raised_detected]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the bars for classified images and detected phones
    rects1 = ax.bar(range(len(categories)), totals_classified, bar_width, color='royalblue', label='Classified Images')
    rects2 = ax.bar([p + bar_width for p in range(len(categories))], totals_detected, bar_width, color='seagreen', label='Detected Phone')

    # Add labels and title to the axis
    ax.set_xlabel('Categories')
    ax.set_ylabel('Counts')
    ax.set_title('EduVision - Figure 2')
    
    # Set the tick locations and labels for the x-axis
    ax.set_xticks([p + bar_width / 2 for p in range(len(categories))])
    ax.set_xticklabels(categories)
    
    # Add a legend
    ax.legend()

    # Add labels to the bars
    add_labels(rects1)
    add_labels(rects2)

    # Adjust the layout and display the figure
    plt.tight_layout()
    plt.show()


def main_video_processing(video_path):
    """
    Main function to process the video and generate the final chart.
    
    Args:
        video_path (str): Path to the input video file.
    """
    # Define the output directories
    output_dir = 'Output_Segmentation'
    classified_dir = 'Classified_Images'
    hand_raised_dir = os.path.join(classified_dir, 'Hand_Raised')
    detected_dir = 'Classified_Images_Detected'

    # Create the output directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(classified_dir):
        os.makedirs(classified_dir)
    if not os.path.exists(hand_raised_dir):
        os.makedirs(hand_raised_dir)
    if not os.path.exists(detected_dir):
        os.makedirs(detected_dir)

    # Detect persons in the video and save the cropped images
    detect_persons(video_path, output_dir)
    
    # Classify face orientations in the cropped images
    face_counts = classify_face_orientations(output_dir, classified_dir)
    
    # Detect hand raised in the cropped images
    hand_raised_counts = detect_hand_raised(output_dir, hand_raised_dir)
    
    # Detect phones in the classified images
    phone_counts = detect_phones(classified_dir, detected_dir)

    # Combine the counts into a final dictionary
    final_counts = {**face_counts, 'Hand_Raised': hand_raised_counts, **phone_counts}
    
    # Plot the final chart
    plot_final_chart(final_counts)


if __name__ == '__main__':
    # Specify the path to your input video file
    video_path = 'your_video_path.mp4'
    
    # Call the main function to process the video
    main_video_processing(video_path)
'''

This code includes thorough comments explaining each function and important sections of the code. The comments provide an overview of what each function does, the input and output parameters, and any relevant details about the implementation.

The code structure and logic remain unchanged, and the comments are added to enhance readability and understanding of the code.
'''