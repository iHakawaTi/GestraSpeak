import cv2
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import sys
from PySide2.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                               QHBoxLayout, QPushButton, QListWidget, QComboBox)
from PySide2.QtGui import QImage, QPixmap, QFont
from PySide2.QtCore import Qt, QTimer, QObject, Signal


class Speaker(QObject):
    # Signal emitted when speaking is complete
    speaking_complete = Signal()

    def __init__(self):
        super().__init__()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.words_to_speak = []
        self.current_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.speak_next_word)

    def speak_word_list(self, words, delay=100):
        """Speak a list of words with delay between them"""
        if words:
            self.words_to_speak = words
            self.current_index = 0
            # Start with first word immediately
            self.speak_next_word()
            # Then set up timer for subsequent words
            self.timer.start(delay)  # delay in milliseconds

    def speak_next_word(self):
        """Speak the next word in the queue"""
        if self.current_index < len(self.words_to_speak):
            word = self.words_to_speak[self.current_index]
            self.engine.say(word)
            self.engine.runAndWait()
            self.current_index += 1
        else:
            self.timer.stop()
            self.speaking_complete.emit()


class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Sign Language Translator")

        # Initialize video capture and models
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        self.offset = 20
        self.imgSize = 300
        self.labels = ["NO", "YES", "OKAY", "THANK YOU"]
        self.words = []

        # Initialize speaker
        self.speaker = Speaker()
        self.speaker.speaking_complete.connect(self.on_speaking_complete)
        self.speaking = False

        # Create UI elements
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.translation_label = QLabel("Detected Sign:")
        self.translation_output = QLabel("")
        self.translation_output.setFont(QFont("Arial", 20, QFont.Bold))

        self.spoken_words_label = QLabel("List of Words:")
        self.spoken_words_list = QListWidget()

        # Dropdown for speak options
        self.speak_option_label = QLabel("Speak:")
        self.speak_option = QComboBox()
        self.speak_option.addItems(["Current Word", "All Words"])

        # Delay control - updated to smaller intervals
        self.delay_label = QLabel("Delay (ms):")
        self.delay_input = QComboBox()
        self.delay_input.addItems(["50", "100", "200", "500"])
        self.delay_input.setCurrentIndex(1)  # default to 100ms

        # Buttons
        self.add_button = QPushButton("Add Word")
        self.speak_button = QPushButton("Speak")
        self.clear_button = QPushButton("Clear")

        # Timer for video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Connect buttons
        self.add_button.clicked.connect(self.add_word)
        self.speak_button.clicked.connect(self.speak_words)
        self.clear_button.clicked.connect(self.clear_words)

        # Layout setup
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.image_label)

        translation_layout = QHBoxLayout()
        translation_layout.addWidget(self.translation_label)
        translation_layout.addWidget(self.translation_output)

        # Add speak option to translation layout
        options_layout = QHBoxLayout()
        options_layout.addWidget(self.speak_option_label)
        options_layout.addWidget(self.speak_option)
        options_layout.addWidget(self.delay_label)
        options_layout.addWidget(self.delay_input)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.speak_button)
        button_layout.addWidget(self.clear_button)

        spoken_layout = QVBoxLayout()
        spoken_layout.addWidget(self.spoken_words_label)
        spoken_layout.addWidget(self.spoken_words_list)
        spoken_layout.addLayout(options_layout)
        spoken_layout.addLayout(button_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addLayout(translation_layout)
        main_layout.addLayout(spoken_layout)

        self.setLayout(main_layout)

    def update_frame(self):
        success, img = self.cap.read()
        if not success:
            return

        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - self.offset:y + h + self.offset,
                      x - self.offset:x + w + self.offset]

            if imgCrop.size > 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = self.imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                    wGap = math.ceil((self.imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = self.imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                    hGap = math.ceil((self.imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                predicted_label = self.labels[index]
                self.translation_output.setText(predicted_label)

                # Draw bounding box and label
                cv2.putText(imgOutput, predicted_label, (x - self.offset, y - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(imgOutput, predicted_label, (x - self.offset, y - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(imgOutput, (x - self.offset, y - self.offset),
                              (x + w + self.offset, y + h + self.offset), (127, 127, 127), 2)

        # Display the image
        h, w, ch = imgOutput.shape
        bytes_per_line = ch * w
        q_image = QImage(imgOutput.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def add_word(self):
        current_text = self.translation_output.text()
        if current_text and not self.speaking:
            self.words.append(current_text)
            self.spoken_words_list.addItem(current_text)
            print(f"Added word: {current_text}")

    def speak_words(self):
        if self.speaking:
            return

        selected_option = self.speak_option.currentText()
        delay = int(self.delay_input.currentText())

        try:
            if selected_option == "Current Word":
                current_text = self.translation_output.text()
                if current_text:
                    self.speaking = True
                    self.speaker.engine.say(current_text)
                    self.speaker.engine.runAndWait()
                    self.speaking = False
            else:  # "All Words"
                if self.words:
                    self.speaking = True
                    self.speak_button.setEnabled(False)
                    self.speaker.speak_word_list(self.words, delay)
        except Exception as e:
            print(f"TTS Error: {e}")
            self.speaking = False
            self.speak_button.setEnabled(True)

    def on_speaking_complete(self):
        """Called when speaking all words is complete"""
        self.speaking = False
        self.speak_button.setEnabled(True)

    def clear_words(self):
        if not self.speaking:
            self.words.clear()
            self.spoken_words_list.clear()
            print("Cleared all words")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())