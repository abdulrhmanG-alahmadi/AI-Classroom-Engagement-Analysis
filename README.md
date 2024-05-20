# AI-Classroom-Engagement-Analysis
EduVision is a computer vision-based project that analyzes classroom engagement by processing video recordings of educational sessions. The project utilizes deep learning models and computer vision techniques to detect and classify various aspects of student engagement, such as face orientations, hand raising, and phone usage.

# Key Features:

* Person Detection: The project uses a pre-trained Faster R-CNN model with a ResNet50-FPN backbone to detect persons in the video frames and save cropped person images.
* Face Orientation Classification: MediaPipe's FaceMesh is employed to detect face landmarks and estimate head pose. The face orientations are classified into different categories, such as looking forward, left, right, up, or down.
* Hand Raising Detection: MediaPipe's Pose estimation is used to detect hand raising gestures by analyzing the angles between the shoulder, elbow, and wrist joints.
* Phone Detection: The project utilizes the Faster R-CNN model to detect the presence of phones in the cropped person images.
* Data Visualization: The classified images and detected phones are analyzed to generate insightful charts and statistics, providing a visual representation of classroom engagement.
