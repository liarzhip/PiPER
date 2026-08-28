from dataclasses import dataclass

import cv2
import mediapipe as mp


@dataclass
class HandDetection:
    # 掌心二维中心
    u: int
    v: int

    # 21个 MediaPipe landmark 像素坐标
    # landmarks[i] = (u, v)
    landmarks: list

    handedness: str
    score: float


class HandDetector:
    def __init__(
        self,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.drawer = (
            mp.solutions.drawing_utils
        )

    def detect(self, image_bgr):
        debug = image_bgr.copy()
        rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, debug

        h, w = image_bgr.shape[:2]
        hand_landmarks = (
            result.multi_hand_landmarks[0]
        )
        pixels = []
        for lm in hand_landmarks.landmark:

            u = int(
                round(lm.x * (w - 1))
            )

            v = int(
                round(lm.y * (h - 1))
            )

            u = max(
                0,
                min(w - 1, u)
            )

            v = max(
                0,
                min(h - 1, v)
            )

            pixels.append(
                (u, v)
            )

        # ----------------------------------------------------------
        # Palm center in image
        #
        # Wrist + 4 MCP
        # ----------------------------------------------------------
        palm_ids = [
            0,
            5,
            9,
            13,
            17,
        ]
        palm_u = int(
            sum(
                pixels[i][0]
                for i in palm_ids
            )
            / len(palm_ids)
        )
        palm_v = int(
            sum(
                pixels[i][1]
                for i in palm_ids
            )
            / len(palm_ids)
        )

        # ----------------------------------------------------------
        # Handedness
        # ----------------------------------------------------------
        handedness = "Unknown"
        score = 0.0
        if result.multi_handedness:

            classification = (
                result
                .multi_handedness[0]
                .classification[0]
            )

            handedness = (
                classification.label
            )

            score = float(
                classification.score
            )

        # ----------------------------------------------------------
        # Draw MediaPipe landmarks
        # ----------------------------------------------------------
        self.drawer.draw_landmarks(
            debug,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
        )

        # 掌心中心
        cv2.circle(
            debug,
            (palm_u, palm_v),
            8,
            (0, 255, 255),
            -1,
        )
        detection = HandDetection(
            u=palm_u,
            v=palm_v,
            landmarks=pixels,
            handedness=handedness,
            score=score,
        )
        return detection, debug

    def close(self):
        self.hands.close()