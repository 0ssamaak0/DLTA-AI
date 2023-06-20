import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QSlider, QLineEdit, QVBoxLayout, QHBoxLayout, QDialog, QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5 import QtWidgets
import qdarktheme


class VideoFrameExtractor(QDialog):
    def __init__(self, mute = None, notification = None):
        super().__init__()
        self.mute = mute
        self.notification = notification
        # set minimum window size
        self.setMinimumSize(500, 300)

        self.setWindowTitle("Open Video as Frames")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)


        self.sampling_max = 100
        # Initialize variables
        self.vid_path = None
        self.sampling_rate = 1
        self.start_frame = 1
        self.end_frame = None
        self.fps = None
        self.stop = False
        self.path_name = None

        font = QFont()
        font.setBold(True)

        # Create widgets
        self.file_label = QLabel("Select a video file:")
        self.file_button = QPushButton("Open Video")
        self.file_button.clicked.connect(self.select_file)

        self.sampling_label = QLabel("Sampling rate:")
        self.sampling_slider = QSlider()
        self.sampling_slider.setOrientation(1)
        self.sampling_slider.setRange(1, self.sampling_max)
        self.sampling_slider.setValue(1)
        self.sampling_slider.setEnabled(False)
        self.sampling_slider.valueChanged.connect(self.update_sampling_rate)
        self.sampling_edit = QLineEdit(str(self.sampling_slider.value()))
        self.sampling_edit.setFont(QFont('', 5))
        self.sampling_edit.setAlignment(Qt.AlignCenter)
        self.sampling_edit.setEnabled(False)
        self.sampling_edit.textChanged.connect(self.update_sampling_slider)
        self.sampling_time_label = QLabel("hh:mm:ss")
        self.sampling_time_label.setFont(font)
        self.sampling_time_label.setAlignment(Qt.AlignRight)

        self.start_label = QLabel("Start frame:")
        self.start_slider = QSlider()
        self.start_slider.setOrientation(1)
        self.start_slider.setRange(0, 1000)
        self.start_slider.setValue(0)
        self.start_slider.setEnabled(False)
        self.start_slider.valueChanged.connect(self.update_start_frame)
        self.start_edit = QLineEdit(str(self.start_slider.value()))
        self.start_edit.setFont(QFont('', 5))
        self.start_edit.setAlignment(Qt.AlignCenter)
        self.start_edit.setEnabled(False)
        self.start_edit.textChanged.connect(self.update_start_slider)
        self.start_time_label = QLabel("hh:mm:ss")
        self.start_time_label.setFont(font)
        self.start_time_label.setAlignment(Qt.AlignRight)

        self.end_label = QLabel("End frame:")
        self.end_slider = QSlider()
        self.end_slider.setOrientation(1)
        self.end_slider.setRange(0, 1)
        self.end_slider.setValue(1)
        self.end_slider.setEnabled(False)
        self.end_slider.valueChanged.connect(self.update_end_frame)
        self.end_edit = QLineEdit(str(self.end_slider.value()))
        self.end_edit.setFont(QFont('', 5))
        self.end_edit.setAlignment(Qt.AlignCenter)
        self.end_edit.setEnabled(False)
        self.end_edit.textChanged.connect(self.update_end_slider)
        self.end_time_label = QLabel("hh:mm:ss")
        self.end_time_label.setFont(font)
        self.end_time_label.setAlignment(Qt.AlignRight)

        self.extract_button = QPushButton("Extract Frames")
        self.extract_button.clicked.connect(self.extract_frames)
        self.extract_button.setEnabled(False)

        self.stop_button = QPushButton("Stop")
        self.stop_button.pressed.connect(self.stop_extraction)
        self.stop_button.setEnabled(False)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(50, 150, 300, 20)
        self.progress_bar.setFormat("Waiting for extraction...")
        self.progress_bar.setValue(0)


        # Create layouts
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)

        sampling_layout = QHBoxLayout()
        inner_sampling_layout = QVBoxLayout()
        inner_sampling_layout.addWidget(self.sampling_label)
        inner_sampling_layout.addWidget(self.sampling_time_label)
        sampling_layout.addLayout(inner_sampling_layout)
        inner_sampling_layout = QVBoxLayout()
        inner_sampling_layout.addWidget(self.sampling_edit)
        inner_sampling_layout.addWidget(self.sampling_slider)
        sampling_layout.addLayout(inner_sampling_layout)


        range_layout = QHBoxLayout()
        

        start_layout = QHBoxLayout()
        inner_start_layout = QVBoxLayout()
        inner_start_layout.addWidget(self.start_label, alignment=Qt.AlignLeft)
        inner_start_layout.addWidget(self.start_time_label, alignment=Qt.AlignLeft)
        start_layout.addLayout(inner_start_layout)
        inner_start_layout = QVBoxLayout()
        inner_start_layout.addWidget(self.start_edit)
        inner_start_layout.addWidget(self.start_slider)
        start_layout.addLayout(inner_start_layout)


        end_layout = QHBoxLayout()
        inner_end_layout = QVBoxLayout()
        inner_end_layout.addWidget(self.end_label)
        inner_end_layout.addWidget(self.end_time_label)
        end_layout.addLayout(inner_end_layout)
        inner_end_layout = QVBoxLayout()
        inner_end_layout.addWidget(self.end_edit)
        inner_end_layout.addWidget(self.end_slider)
        end_layout.addLayout(inner_end_layout)

        range_layout.addLayout(start_layout)
        end_layout.setContentsMargins(20, 0, 0, 0)
        range_layout.addLayout(end_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.extract_button)
        button_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(file_layout)

        range_layout.setContentsMargins(0, 20, 0, 0)
        main_layout.addLayout(range_layout)

        main_layout.addLayout(sampling_layout)

        main_layout.addLayout(button_layout)

        main_layout.addWidget(self.progress_bar)

        # Set the main layout
        self.setLayout(main_layout)

    def select_file(self):
        # Open a file dialog to select a video file
        file_path, _ = QFileDialog.getOpenFileName(self, "Video to Frames", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.vid_path = file_path
            self.file_label.setText(f"Selected video file: {self.vid_path}")
            self.sampling_slider.setEnabled(True)
            self.sampling_edit.setEnabled(True)
            self.start_slider.setEnabled(True)
            self.start_edit.setEnabled(True)
            self.end_slider.setEnabled(True)
            self.end_edit.setEnabled(True)
            self.extract_button.setEnabled(True)
            self.stop_button.setEnabled(True)

            # Set the stop button to red
            self.stop_button.setStyleSheet("background-color: red; color: white;")
            
            # Open the video file
            vidcap = cv2.VideoCapture(self.vid_path)
            self.fps = vidcap.get(cv2.CAP_PROP_FPS)

            # Set the maximum value of the start and end sliders to the total number of frames in the video
            self.max_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Set the start and end sliders to the maximum value
            self.start_slider.setMaximum(self.max_frame)
            self.start_time_label.setText(self.get_time_string(0))
            # update startedit and start time
            self.end_slider.setMaximum(self.max_frame)
            self.end_slider.setValue(self.max_frame)
            # update endedit and end time
            self.end_edit.setText(str(self.end_slider.value()))
            self.end_time_label.setText(self.get_time_string(self.max_frame / self.fps))
            # update sampling 
            self.sampling_time_label.setText(self.get_time_string(1 / self.fps))
            self.sampling_slider.setMaximum(self.max_frame // 10)
            self.sampling_slider.setValue(self.max_frame // 100)
            self.sampling_max = self.max_frame // 10
            
        else:
            self.file_label.setText("No video is selected")
            self.sampling_slider.setEnabled(False)
            self.sampling_edit.setEnabled(False)
            self.start_slider.setEnabled(False)
            self.start_edit.setEnabled(False)
            self.end_slider.setEnabled(False)
            self.end_edit.setEnabled(False)

    def update_sampling_rate(self, value):
        # Update the sampling rate when the slider is moved
        self.sampling_rate = value
        self.sampling_edit.setText(str(value))

    def update_sampling_slider(self, text):
        # Update the sampling rate when the edit box is changed
        try:
            value = int(text)
            if value < 1:
                value = 1
            elif value > self.sampling_max:
                value = self.sampling_max
            self.sampling_rate = value
            self.sampling_slider.setValue(value)
            if self.fps:
                self.sampling_time_label.setText(self.get_time_string(value / self.fps))
                if self.end_frame is not None:
                    self.progress_bar.setFormat(f"Will Extract {(self.end_frame - self.start_frame) // self.sampling_rate} Frames")
        except ValueError:
            pass

    def update_start_frame(self, value):
        # Update the start frame when the slider is moved
        self.start_frame = value
        self.start_edit.setText(str(value))

    def update_start_slider(self, text):
        # Update the start frame when the edit box is changed
        try:
            value = int(text)
            if value < 0:
                value = 0
            elif self.end_frame is not None and value > self.end_frame:
                self.start_slider.setValue(self.end_frame)
                value = self.end_frame
            self.start_frame = value
            self.start_slider.setValue(value)
            if self.fps:
                self.start_time_label.setText(self.get_time_string(value / self.fps))
                if self.end_frame is not None:
                    self.progress_bar.setFormat(f"Will Extract {(self.end_frame - self.start_frame) // self.sampling_rate} Frames")
        except ValueError:
            pass

    def update_end_frame(self, value):
        # Update the end frame when the slider is moved
        self.end_frame = value
        self.end_edit.setText(str(value))

    def update_end_slider(self, text):
        # Update the end frame when the edit box is changed
        try:
            value = int(text)
            if self.start_frame is not None and value < self.start_frame:
                value = self.start_frame
            self.end_frame = value
            self.end_slider.setValue(value)
            if self.fps:
                self.end_time_label.setText(self.get_time_string(value / self.fps))
                if self.end_frame is not None:
                    self.progress_bar.setFormat(f"Will Extract {(self.end_frame - self.start_frame) // self.sampling_rate} Frames")
        except ValueError:
            pass

    def extract_frames(self):
        # Call the vid_to_frames function with the selected parameters
        try:
            self.path_name = self.vid_to_frames(self.vid_path, self.sampling_rate, self.start_frame, self.end_frame)
        except ValueError as e:
            self.progress_bar.setFormat(str(e))
            return
        self.close()
        return self.path_name
        
    
    def stop_extraction(self):
        # stop the extraction process
        self.stop = True

    def get_time_string(self, seconds, separator=":"):
        # Convert seconds to hh:mm:ss format
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}{separator}{int(m):02d}{separator}{int(s):02d}"


    def vid_to_frames(self, vid_path, sampling_rate, start_frame, end_frame):
        """
        Extracts frames from a video file and saves them as JPEG images.

        Args:
            vid_path (str): Path to the video file.
            sampling_rate (int): How often to save a frame. For example, if sampling_rate = 2, every other frame will be saved.
            start_frame (int): Starting frame number.
            end_frame (int): Ending frame number.
        """
        # Check if the path exists
        if not os.path.exists(vid_path):
            raise ValueError("Video path does not exist")

        # Create a directory to store the frames
        frames_path = "".join([vid_path.split(".")[0], "_frames"])

        # if the directory does not exist, create it
        if not os.path.exists(frames_path):
            os.mkdir(frames_path)
        # if the directory exists, delete all the files it contains
        else:
            for file in os.listdir(frames_path):
                os.remove(os.path.join(frames_path, file))

        # Open the video file
        vidcap = cv2.VideoCapture(vid_path)
        # if the video file does not exist, raise an error

        # Set the starting frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Get the total number of frames in the video
        n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total number of frames: {n_frames}")

        # Initialize counters
        count = start_frame
        success = True

        # set progress bar Format
        while success:
            success, image = vidcap.read()
            if count % sampling_rate == 0:
                # Get the time in the video corresponding to the current frame
                time_in_sec = count / self.fps
                time_str = self.get_time_string(time_in_sec, separator="_")

                # Save the image with the time in the file name
                indented_count = str(count).zfill(len(str(n_frames)))
                cv2.imwrite(f"{frames_path}/frame_{indented_count}_time_{time_str}.jpg", image)
            
            self.progress_bar.setValue(int(((count - start_frame) / (end_frame - start_frame)) * 100))
            self.progress_bar.setFormat(f"{int(((count - start_frame) / (end_frame - start_frame)) * 100)}%")

            count += 1
            if count >= end_frame:
                self.progress_bar.setValue(100)
                break

            QtWidgets.QApplication.processEvents()
            if self.stop:
                self.stop = False
                self.progress_bar.setFormat("Extraction stopped")
                self.progress_bar.setValue(0)
                break
            
                    # Show a notification if the model explorer is not the active window
        try:
            if not self.mute:
                if not self.isActiveWindow():
                    self.notification(f"Video Extraction Completed")
        except:
            pass
        return frames_path


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     qdarktheme.setup_theme()
#     window = VideoFrameExtractor()
#     window.show()
#     sys.exit(app.exec_())