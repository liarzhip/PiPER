from dataclasses import dataclass
import cv2
import mediapipe as mp
import numpy as np

PALM_IDS = (0, 5, 9, 13, 17)

@dataclass
class HandDetection:
    u: int
    v: int
    handedness: str
    score: float

class HandDetector:
    def __init__(self, min_det=0.6, min_track=0.6):
        self.mp_hands = mp.solutions.hands
        self.draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )

    def detect(self, bgr):
        debug = bgr.copy()
        result = self.hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None, debug

        lm = result.multi_hand_landmarks[0]
        h, w = bgr.shape[:2]
        pts = np.array([[lm.landmark[i].x*w, lm.landmark[i].y*h] for i in PALM_IDS])
        u, v = pts.mean(axis=0)
        u, v = int(np.clip(round(u),0,w-1)), int(np.clip(round(v),0,h-1))

        handedness, score = "Unknown", 0.0
        if result.multi_handedness:
            c = result.multi_handedness[0].classification[0]
            handedness, score = c.label, float(c.score)

        self.draw.draw_landmarks(debug, lm, self.mp_hands.HAND_CONNECTIONS)
        cv2.circle(debug, (u,v), 8, (0,255,255), -1)
        return HandDetection(u,v,handedness,score), debug

    def close(self):
        self.hands.close()
