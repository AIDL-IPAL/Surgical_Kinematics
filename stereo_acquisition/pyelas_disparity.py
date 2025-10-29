import cv2
import numpy as np
import sys
# print("sys.path =", sys.path)

def disparity_pyelas(frame_left, frame_right):
    import elas as pyelas
    left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    elas = pyelas.Elas()
    return elas.process(left_gray, right_gray)  # returns (left_disp, right_disp)

def normalize_disp(disp):
    dmin, dmax = np.min(disp), np.max(disp)
    return ((disp - dmin) / (dmax - dmin) * 255).astype(np.uint8) if dmax > dmin else np.zeros_like(disp, dtype=np.uint8)


def main():
    # Paths to the two video files.
    video_path_left  = r"scene3d/acq_data/04_08_2025/cholocystectomy/camera0_20250408_122624_chunk.avi"
    video_path_right = r"scene3d/acq_data/04_08_2025/cholocystectomy/camera1_20250408_122624_chunk.avi"

    # Open both video files.
    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)
    
    if not cap_left.isOpened():
        print(f"Error opening left video file: {video_path_left}")
        return
    if not cap_right.isOpened():
        print(f"Error opening right video file: {video_path_right}")
        return
    
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    width  = 2*int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define output video writer for the disparity (depth) video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_video_path = 'scene3d/stereo_utils/depth_results_tmp/disparity_video_PyElas/disp_ex.mp4'
    depth_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # Process only the seconds 5 to 15 of video.
    start_frame = int(fps * 5)
    end_frame = int(fps * 15)
    frame_count = start_frame
    
    # Set the starting frame position for both videos.
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while frame_count < end_frame:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            break
        
        # Compute disparity using PyElas.
        # from stereo_utils.disparity import disparity_pyelas
        disp_left, disp_right = disparity_pyelas(frame_left, frame_right)
        disp_norm_left = normalize_disp(disp_left)
        disp_norm_right = normalize_disp(disp_right)        # Normalize the disparity map to 0-255 for visualization.

        # Apply a color map for better visualization.
        depth_color_l = cv2.applyColorMap(disp_norm_left, cv2.COLORMAP_JET)
        depth_color_r = cv2.applyColorMap(disp_norm_right, cv2.COLORMAP_JET)

        # Combine the two disparity maps into a single image.
        depth_color = cv2.hconcat([depth_color_l, depth_color_r])
        # Display the depth map.
        cv2.imshow("Depth Map (Color)", depth_color)
        
        # Write the depth frame to the output video.
        depth_writer.write(depth_color)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap_left.release()
    cap_right.release()
    depth_writer.release()
    cv2.destroyAllWindows()
    print(f"Processed frames from {start_frame} to {frame_count}. Depth video saved to {out_video_path}")

if __name__ == '__main__':
    main()