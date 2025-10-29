import cv2

def grab_frame(video_path, time_sec, label):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Calculate the frame number to grab
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * time_sec)
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    if ret:
        # Save the frame as an image file
        cv2.imwrite(f'frame_at_{time_sec}sec_{label}.jpg', frame)
        print("Frame saved successfully.")
    else:
        print("Failed to grab frame.")
    
    # Release the video capture object
    cap.release()

# Example usage
video_path_right = r"scene3d/stereo_utils/acq_data/04_08_2025/cholocystectomy/camera0_20250408_122624_chunk.avi"
video_path_left  = r"scene3d/stereo_utils/acq_data/04_08_2025/cholocystectomy/camera1_20250408_122624_chunk.avi"

sec = 1860  # Time in seconds to grab the frame
grab_frame(video_path_right, sec, "right")
grab_frame(video_path_left, sec, "left")
