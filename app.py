from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from pygame import mixer
import time

app = Flask(__name__)

mixer.init()

notes = {
    'C': mixer.Sound('static/C4.mp3'), 
    'D': mixer.Sound('static/D4.mp3'), 
    'E': mixer.Sound('static/E4.mp3'),
    'F': mixer.Sound('static/F4.mp3'),
    'G': mixer.Sound('static/G4.mp3'), 
    'A': mixer.Sound('static/A4.mp3'),
    'B': mixer.Sound('static/B4.mp3')
}

key_regions = {
    'C': (50, 100),  
    'D': (100, 150),
    'E': (150, 200), 
    'F': (200, 250),
    'G': (250, 300),
    'A': (300, 350),
    'B': (350, 400)
}

mediapipe_hands = mp.solutions.hands
hand_detector = mediapipe_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mediapipe_drawing = mp.solutions.drawing_utils

last_played = {}  

def generate_frames():
    webcam = cv2.VideoCapture(0)
    
    while True:
        frame_captured, video_frame = webcam.read()
        if not frame_captured:
            break
            
        mirrored_frame = cv2.flip(video_frame, 1)
        
        for note, (x_start, x_end) in key_regions.items():
            cv2.rectangle(mirrored_frame, (x_start, 280), (x_end, 430), (255, 255, 255), 2)
            cv2.putText(mirrored_frame, note, (x_start + 10, 330), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        rgb_image = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
        
        hand_detection_results = hand_detector.process(rgb_image)
        
        if hand_detection_results.multi_hand_landmarks:
            for detected_hand in hand_detection_results.multi_hand_landmarks:
                mediapipe_drawing.draw_landmarks(
                    mirrored_frame,
                    detected_hand,
                    mediapipe_hands.HAND_CONNECTIONS,
                    mediapipe_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mediapipe_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                index_finger_tip = detected_hand.landmark[8]
                h, w, _ = mirrored_frame.shape
                finger_x, finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                for note, (x_start, x_end) in key_regions.items():
                    if (280 <= finger_y <= 430 and x_start <= finger_x <= x_end):
                        current_time = time.time()
                        if note not in last_played or (current_time - last_played[note]) > 0.5:
                            notes[note].play()
                            last_played[note] = current_time
        
        success, encoded_frame = cv2.imencode('.jpg', mirrored_frame)
        frame_bytes = encoded_frame.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
