import sys, os, cv2
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox,
    QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtCore import QTimer, Qt
import sounddevice as sd
import soundfile as sf
import numpy as np

# manual usb device driver setup
DEVICE_INDEX = 1  # Set manually as needed
# aditya: left camera is 2, right camera is 1
Lcam_int = 2
Rcam_int = 1
print("Available audio devices:")
print(sd.query_devices())

if DEVICE_INDEX is not None:
    sd.default.device = (DEVICE_INDEX, None)

######## File Settings ########
from pathlib import Path
script_path = Path(__file__).resolve()
PROJ_ROOT = script_path.parents[0]  # Adjust parent level if needed
print("Project root:", PROJ_ROOT)
SAVE_DIR = input("Enter the directory to save data: stereo_utils/<dir> (OR press enter for 'acq_data'): ") or "acq_data"
dir = os.path.join(PROJ_ROOT, SAVE_DIR)
dir = os.path.join(os.getcwd(), dir, datetime.now().strftime('%m_%d_%Y'))
print("Saving data to:", dir)

def get_experiment_name():
    global experiment_name
    experiment_name = datetime.now().strftime('%H_%M')  # Default value
    try:
        experiment_name = input("Enter experiment name (15 seconds to respond): ")
    except EOFError:
        pass

import threading
input_thread = threading.Thread(target=get_experiment_name)
input_thread.start()
input_thread.join(timeout=15)

if not experiment_name:
    experiment_name = datetime.now().strftime('%H_%M_%d')
output_dir = os.path.join(dir, experiment_name)
index = 1
while os.path.exists(output_dir):
    output_dir = os.path.join(dir, f"{experiment_name}_{index}")
    index += 1
if not os.path.exists(output_dir):
    print("Directory does not exist... creating directory")
    os.makedirs(output_dir)

print("Saving to directory path:", output_dir)
print("Reminder to share data when finished (consider zip compression)")

# Camera Setup: remember to run camera grab script for numbers. Check left vs right camera.
cap1 = cv2.VideoCapture(Lcam_int) # left camera
cap2 = cv2.VideoCapture(Rcam_int) # right camera
# For macOS users, you might need:
# cap1 = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
# cap2 = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap1.isOpened():
    print("Error: Camera 1 failed to open.")
if not cap2.isOpened():
    print("Error: Camera 2 failed to open.")

# Global counter for still images
image_counter = 0

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Dashboard")

        # Main horizontal layout for the entire window
        main_layout = QHBoxLayout(self)

        # Left side container (for feeds and buttons)
        left_container = QWidget()
        left_vbox = QVBoxLayout(left_container)
        main_layout.addWidget(left_container)

        # Layout for camera feeds (inside the left container)
        feed_layout = QHBoxLayout()
        self.label_cam0 = QLabel("Camera 0 Feed")
        self.label_cam1 = QLabel("Camera 1 Feed")
        feed_layout.addWidget(self.label_cam0)
        feed_layout.addWidget(self.label_cam1)
        left_vbox.addLayout(feed_layout)

        # Audio properties
        self.audio_recording = False
        self.audio_samplerate = 48000
        self.audio_channels = 1
        self.audio_file = None
        self.audio_level = 0
        self.audio_stream = None

        # Set display dimensions
        screen_rect = QDesktopWidget().availableGeometry(self)
        self.setGeometry(screen_rect)
        self.setMinimumSize(screen_rect.width(), screen_rect.height())

        self.label_cam0.setFixedSize(screen_rect.width() // 2, screen_rect.height())
        self.label_cam1.setFixedSize(screen_rect.width() // 2, screen_rect.height())

        # Button layout (inside the left container)
        self.button_layout = QHBoxLayout()
        self.grab_button = QPushButton("Grab Frame")
        self.start_record_button = QPushButton("Start Recording Chunk")
        self.stop_record_button = QPushButton("Stop Recording Chunk")
        self.stop_record_button.setEnabled(False)
        self.exit_button = QPushButton("Exit")
        
        self.button_layout.addWidget(self.grab_button)
        self.button_layout.addWidget(self.start_record_button)
        self.button_layout.addWidget(self.stop_record_button)
        self.button_layout.addWidget(self.exit_button)
        left_vbox.addLayout(self.button_layout)

        # Right side: Vertical audio bar
        self.audio_bar = QProgressBar()
        self.audio_bar.setRange(0, 100)
        self.audio_bar.setValue(0)
        self.audio_bar.setTextVisible(False)
        self.audio_bar.setOrientation(Qt.Vertical)
        main_layout.addWidget(self.audio_bar)
        
        # Connect buttons
        self.grab_button.clicked.connect(self.on_grab_button_clicked)
        self.start_record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.exit_button.clicked.connect(self.close_app)
        
        # Set up a QTimer for live preview updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(30)
        
        # Recording state and video writer holders
        self.recording = False
        self.video_writer0 = None
        self.video_writer1 = None
        self.record_filename0 = ""
        self.record_filename1 = ""
        self.fps = 30.0

        # Start audio monitoring
        self.initialize_audio_monitoring()

    def initialize_audio_monitoring(self):
        """Start audio monitoring for UI feedback. Runs continuously."""
        def audio_callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            # Calculate audio magnitude (RMS) and update the level for the UI
            rms = np.sqrt(np.mean(indata**2))
            # Scale to 0-100 for the progress bar, with some gain
            self.audio_level = int(rms * 300)
            
            if self.audio_recording and self.audio_file:
                self.audio_file.write(indata)

        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.audio_samplerate,
                channels=self.audio_channels,
                callback=audio_callback
            )
            self.audio_stream.start()
            print("Audio monitoring started.")
        except Exception as e:
            print(f"Error starting audio monitoring: {e}")
            QMessageBox.critical(self, "Audio Error", f"Could not start audio monitoring: {e}")

    def update_display(self):
        """Grab frames from both cameras, update preview, and if recording, write frames."""
        ret0, frame0 = cap1.read()
        ret1, frame1 = cap2.read()
        if not ret0:
            # Handle camera 0 failure
            self.label_cam0.setText("Camera 0 Signal Lost")
            if self.recording and self.video_writer0:
                # Write a black frame to keep video sync
                h, w, _ = self.label_cam0.size().height(), self.label_cam0.size().width(), 3
                black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                self.video_writer0.write(black_frame)
            # try:
            #     cap1 = cv2.VideoCapture(Lcam_int)
            # except:
            #     pass
        if not ret1:
            # Handle camera 1 failure
            self.label_cam1.setText("Camera 1 Signal Lost")
            if self.recording and self.video_writer1:
                # Write a black frame to keep video sync
                h, w, _ = self.label_cam1.size().height(), self.label_cam1.size().width(), 3
                black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                self.video_writer1.write(black_frame)
            # try:
            #     cap2 = cv2.VideoCapture(Rcam_int)
            # except:
            #     pass

        # If recording is active, write the frames that were successfully read.
        if self.recording:
            if ret0 and self.video_writer0:
                self.video_writer0.write(frame0)
            if ret1 and self.video_writer1:
                self.video_writer1.write(frame1)
        
        if not ret0 and not ret1:
            self.label_cam0.setText("Camera 0 Signal Lost")
            self.label_cam1.setText("Camera 1 Signal Lost")
            print("Both cameras failed. Check Cable/USB connections and device integer settings.")
            return # Both cameras failed, nothing to display
        
        # Update audio bar
        self.audio_bar.setValue(self.audio_level)

        # Convert to QImage for each feed
        if ret0:
            rgb_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            h0, w0, ch0 = rgb_frame0.shape
            bytes_per_line0 = ch0 * w0
            qimg0 = QImage(rgb_frame0.data, w0, h0, bytes_per_line0, QImage.Format_RGB888)
            self.label_cam0.setPixmap(QPixmap.fromImage(qimg0))

        # Convert frames for display (BGR -> RGB)
        if ret1:
            rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            h1, w1, ch1 = rgb_frame1.shape
            bytes_per_line1 = ch1 * w1
            qimg1 = QImage(rgb_frame1.data, w1, h1, bytes_per_line1, QImage.Format_RGB888)
            self.label_cam1.setPixmap(QPixmap.fromImage(qimg1))
        
    def on_grab_button_clicked(self):
        """Snap and save still images with a confirmation dialog."""
        global image_counter
        ret0, frame0 = cap1.read()
        ret1, frame1 = cap2.read()
        if not ret0 or not ret1:
            print("Failed to grab frame(s)")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename0 = os.path.join(output_dir, f'camera0_{timestamp}_{image_counter}.jpg')
        filename1 = os.path.join(output_dir, f'camera1_{timestamp}_{image_counter}.jpg')
        
        reply = QMessageBox.question(
            self,
            "Confirm Save",
            f"Save still images to:\n{filename0}\nand\n{filename1}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            cv2.imwrite(filename0, frame0)
            cv2.imwrite(filename1, frame1)
            print(f"Images saved: {filename0}, {filename1}")
            image_counter += 1
        else:
            print("Still images not saved.")
    
    def start_recording(self):
        """Start recording a chunk of video from both cameras."""
        # Determine frame dimensions from one camera
        ret, frame = cap1.read()
        if not ret:
            print("Warning: Camera 0 not available for recording.")
            frame = None

        # Try to get frame from camera 1 if camera 0 failed
        ret1, frame1 = cap2.read()
        if not ret1:
            print("Warning: Camera 1 not available for recording.")
            frame1 = None

        # Determine frame dimensions from whichever camera is available
        if frame is not None:
            height, width, _ = frame.shape
        elif frame1 is not None:
            height, width, _ = frame1.shape
        else:
            height, width = 480, 640  # fallback default

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.record_filename0 = os.path.join(output_dir, f'camera0_{timestamp}_chunk.avi')
        self.record_filename1 = os.path.join(output_dir, f'camera1_{timestamp}_chunk.avi')
        self.audio_filename = os.path.join(output_dir, f'audio_{timestamp}_chunk.wav')

        # Start audio capture by opening a sound file
        try:
            self.audio_file = sf.SoundFile(self.audio_filename, mode='w', samplerate=self.audio_samplerate, channels=self.audio_channels)
            self.audio_recording = True
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Could not open audio file for writing: {e}")
            return

        # Define codec and create VideoWriter objects for each available camera
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # adjust codec if necessary
        if frame is not None:
            self.video_writer0 = cv2.VideoWriter(self.record_filename0, fourcc, self.fps, (width, height))
        else:
            self.video_writer0 = None
        if frame1 is not None:
            self.video_writer1 = cv2.VideoWriter(self.record_filename1, fourcc, self.fps, (width, height))
        else:
            self.video_writer1 = None

        if (self.video_writer0 is None and self.video_writer1 is None) and not self.audio_recording:
            QMessageBox.warning(self, "Error", "No audio or camera available for recording.")
            return

        self.recording = True
        self.start_record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        print("Recording started...")

    def stop_recording(self):
        """Stop recording the video chunk and ask for confirmation to save or discard."""
        if not self.recording:
            return
        
        self.recording = False
        self.audio_recording = False
        
        # Release the video writers
        if self.video_writer0 is not None:
            self.video_writer0.release()
        if self.video_writer1 is not None:
            self.video_writer1.release()
        self.video_writer0 = None
        self.video_writer1 = None
        
        # Close the audio file
        if self.audio_file is not None:
            self.audio_file.close()
            self.audio_file = None
            print("Audio file finalized.")
        
        self.start_record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        print("Recording stopped.")
        
        # Ask user for confirmation to keep the recording. Enter key confirms save.
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirm Save")
        msg_box.setText(f"Keep the recorded chunk?\nVideo 0: {self.record_filename0}\nVideo 1: {self.record_filename1}\nAudio: {self.audio_filename}")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)  # Sets 'Yes' as the default action for Enter key
        
        reply = msg_box.exec_()

        if reply == QMessageBox.No:
            # Remove the files if user discards
            try:
                if os.path.exists(self.record_filename0): os.remove(self.record_filename0)
                if os.path.exists(self.record_filename1): os.remove(self.record_filename1)
                if os.path.exists(self.audio_filename): os.remove(self.audio_filename)
                print("Recorded chunk discarded.")
            except Exception as e:
                print("Error discarding files:", e)
        else:
            print("Recorded chunk saved.")
        
    def close_app(self):
        """Cleanup and close the application."""
        if self.recording:
            self.stop_recording() # Ensure recording is stopped and files are handled
        
        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            print("Audio stream stopped.")

        self.timer.stop()
        cap1.release()
        cap2.release()
        self.close()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
