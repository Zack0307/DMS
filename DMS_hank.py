import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import threading
# import transforms3d
import collections
import base64
import os
from google import genai
from google.genai import types
from collections import deque
from datetime import datetime
from mediapipe.python.solutions import face_mesh, drawing_utils, drawing_styles
from flask import Flask, Response, request, jsonify, render_template
from utils.face_geometry import (  
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)
from utils.drawing import Drawing
from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from utils.blendshape_calculator import BlendshapeCalculator
from gemini_chatbot import API_KEY
from flask_cors import CORS  # 引入 CORS

# points of the face model that will be used for SolvePnP later
points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

#API key
client = genai.Client(api_key=API_KEY)


class AngleBuffer:
    def __init__(self, size=40):
        self.size = size
        self.buffer = collections.deque(maxlen=size)

    def add(self, angles):
        self.buffer.append(angles)

    def get_average(self):
        if not self.buffer:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(self.buffer, axis=0)

USER_FACE_WIDTH = 140  # [mm]

## Camera Parameters (not currently used in calculations)
# NOSE_TO_CAMERA_DISTANCE: The distance from the tip of the nose to the camera lens in millimeters.
# Intended for future use where accurate physical distance measurements may be necessary.
NOSE_TO_CAMERA_DISTANCE = 600  # [mm]

## Configuration Parameters
# PRINT_DATA: Enable or disable the printing of data to the console for debugging.
PRINT_DATA = False
# SHOW_ALL_FEATURES: If True, display all facial landmarks on the video feed.
SHOW_ALL_FEATURES = True
# ENABLE_HEAD_POSE: Enable the head position and orientation estimator.
ENABLE_HEAD_POSE = True

# SERVER_PORT: Port number for the server to listen on.
SERVER_PORT = 7070

## Blink Detection Parameters
# SHOW_ON_SCREEN_DATA: If True, display blink count and head pose angles on the video feed.
SHOW_ON_SCREEN_DATA = True

# BLINK_THRESHOLD: Eye aspect ratio threshold below which a blink is registered.
BLINK_THRESHOLD = 0.51

# EYE_AR_CONSEC_FRAMES: Number of consecutive frames below the threshold required to confirm a blink.
EYE_AR_CONSEC_FRAMES = 2

## Head Pose Estimation Landmark Indices
# These indices correspond to the specific facial landmarks used for head pose estimation.
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291
PHONE_DIST_TH   = 0.15       # 手靠近耳朵距離閾值 (相對寬度)

## MediaPipe Model Confidence Parameters
# These thresholds determine how confidently the model must detect or track to consider the results valid.
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

## Angle Normalization Parameters
# MOVING_AVERAGE_WINDOW: The number of frames over which to calculate the moving average for smoothing angles.
MOVING_AVERAGE_WINDOW = 10

# Initial Calibration Flags
# initial_pitch, initial_yaw, initial_roll: Store the initial head pose angles for calibration purposes.
# calibrated: A flag indicating whether the initial calibration has been performed.
initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False

# User-configurable parameters
DEFAULT_WEBCAM = 0  # Default webcam number

#If set to false it will wait for your command (hittig 'r') to start logging.
IS_RECORDING = False  # Controls whether data is being logged

# Command-line arguments for camera source

# Iris and eye corners landmarks indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # Left eye Left Corner
L_H_RIGHT = [133]  # Left eye Right Corner
R_H_LEFT = [362]  # Right eye Left Corner
R_H_RIGHT = [263]  # Right eye Right Corner

# Blinking Detection landmark's indices.
# P0, P3, P4, P5, P8, P11, P12, P13
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

# Mouth landmarks for Yawn Detection
MOUTH_POINTS = [0, 17, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Face Selected points indices for Head Pose Estimation
_indices_pose = [1, 33, 61, 199, 263, 291]

class DMSSystem:
    def __init__(self):
        # ... (rest of the __init__ method)
        self.head_pose = "N/A"
        self.avg_ear = 0.0
        self.data_lock = threading.Lock()
        self.data_queue = deque(maxlen=1) # Buffer for the latest data

        # MediaPipe 初始化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hand_mesh = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_mesh = self.mp_hand_mesh.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.show_3d = True
        
        # 眼部關鍵點索引 (MediaPipe Face Mesh)
        
        # EAR 相關參數
        self.EAR_THRESHOLD = 0.51
        self.EAR_CONSECUTIVE_FRAMES = 2
        self.ear_counter = 0
        self.ear_history = deque(maxlen=30)
        
        # 疲勞檢測參數
        self.fatigue_score = 0
        self.attention_score = 100
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.yawn_count = 0
        self.yawn_counter = 0
        self.YAWN_THRESHOLD = 0.5
        self.YAWN_CONSECUTIVE_FRAMES = 3 # Number of consecutive frames to confirm a yawn
        
        # 頭部姿態參數
        self.head_pose_history = deque(maxlen=10)
        self.angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
        
        # 系統狀態
        self.start_time = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # 警報系統
        self.alerts = []
        self.last_alert_time = {}
        self.alert_cooldown = 3  # 秒
        
        # 攝影機 3d video
        self.pcf = PCF()
        self.blendshape_calulator = BlendshapeCalculator()
        self.live_link_face = PyLiveLinkFace(fps = 10, filter_size = 4)
        self.outputFrame = None
        self.running = True
        


    def euclidean_distance_3D(self, points):
        """Calculates the Euclidean distance between two points in 3D space.

        Args:
            points: A list of 3D points.

        Returns:
            The Euclidean distance between the two points.

            # Comment: This function calculates the Euclidean distance between two points in 3D space.
        """

        # Get the three points.
        P0, P3, P4, P5, P8, P11, P12, P13 = points

        # Calculate the numerator.
        numerator = (
            np.linalg.norm(P3 - P13) ** 3
            + np.linalg.norm(P4 - P12) ** 3
            + np.linalg.norm(P5 - P11) ** 3
        )

        # Calculate the denominator.
        denominator = 3 * np.linalg.norm(P0 - P8) ** 3

        # Calculate the distance.
        distance = numerator / denominator

        return distance

    def blinking_ratio(self, landmarks):
        """Calculates the blinking ratio of a person.

        Args:
            landmarks: A facial landmarks in 3D normalized.

        Returns:
            The blinking ratio of the person, between 0 and 1, where 0 is fully open and 1 is fully closed.

        """

        # Get the right eye ratio.
        right_eye_ratio = self.euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])

        # Get the left eye ratio.
        left_eye_ratio = self.euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])

        # Calculate the blinking ratio.
        ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

        return ratio

    def mouth_aspect_ratio(self, landmarks):
        """Calculates the mouth aspect ratio (MAR) for yawn detection.
        Args:
            landmarks: A facial landmarks in 3D normalized.
        Returns:
            The mouth aspect ratio.
        """
        # Get the coordinates of the mouth landmarks
        # Inner mouth landmarks (vertical)
        inner_up = landmarks[13]  # Upper lip inner
        inner_down = landmarks[14]  # Lower lip inner

        # Outer mouth landmarks (horizontal)
        left_corner = landmarks[61]  # Right mouth corner
        right_corrner = landmarks[291] # Left mouth corner

        # Calculate the Euclidean distances
        vertical_dist = np.linalg.norm(inner_up - inner_down)
        horizontal_dist = np.linalg.norm(left_corner - right_corrner)

        # Calculate MAR
        mar = vertical_dist / horizontal_dist
        return mar


    def calculate_head_pose(self, landmarks, image_size):
        # Scale factor based on user's face width (assumes model face width is 150mm)
        scale_factor = USER_FACE_WIDTH / 150.0
        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        # Chin
            (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     # Left eye left corner
            (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      # Right eye right corner
            (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    # Left Mouth corner
            (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      # Right mouth corner
        ])
        

        # Camera internals
        focal_length = image_size[1]
        center = (image_size[1]/2, image_size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype = "double"
        )

        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1))

        # 2D image points from landmarks, using defined indices
        image_points = np.array([
            landmarks[NOSE_TIP_INDEX],            # Nose tip
            landmarks[CHIN_INDEX],                # Chin
            landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
            landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
            landmarks[LEFT_MOUTH_CORNER_INDEX],      # Left mouth corner
            landmarks[RIGHT_MOUTH_CORNER_INDEX]      # Right mouth corner
        ], dtype="double")


            # Solve for pose
        (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv.Rodrigues(rotation_vector)

        # Combine rotation matrix and translation vector to form a 3x4 projection matrix
        projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

        # Decompose the projection matrix to extract Euler angles
        _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
        pitch, yaw, roll = euler_angles.flatten()[:3]


        # Normalize the pitch angle
        pitch = self.normalize_pitch(pitch)

        return pitch, yaw, roll
    
    # https://github.com/JimWest/MeFaMo
    def calculate_rotation(self, face_landmarks, pcf: PCF, image_shape):
        frame_width, frame_height, channels = image_shape
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        dist_coeff = np.zeros((4, 1))

        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark[:468]]

        )
        # print(landmarks.shape)
        landmarks = landmarks.T

        metric_landmarks, pose_transform_mat = get_metric_landmarks(
            landmarks.copy(), pcf
        )

        model_points = metric_landmarks[0:3, points_idx].T
        image_points = (
            landmarks[0:2, points_idx].T
            * np.array([frame_width, frame_height])[None, :]
        )

        success, rotation_vector, translation_vector = cv.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeff,
            flags=cv.SOLVEPNP_ITERATIVE,
        )

        return pose_transform_mat, metric_landmarks, rotation_vector, translation_vector
    
    def normalize_pitch(self, pitch):
        """
        Normalize the pitch angle to be within the range of [-90, 90].

        Args:
            pitch (float): The raw pitch angle in degrees.

        Returns:
            float: The normalized pitch angle.
        """
        # Map the pitch angle to the range [-180, 180]
        if pitch > 180:
            pitch -= 360

        # Invert the pitch angle for intuitive up/down movement
        pitch = -pitch

        # Ensure that the pitch is within the range of [-90, 90]
        if pitch < -90:
            pitch = -(180 + pitch)
        elif pitch > 90:
            pitch = 180 - pitch
            
        pitch = -pitch

        return pitch
        
    
    def analyze_fatigue(self, ear_left, ear_right):
        """分析疲勞狀態"""
        avg_ear = (ear_left + ear_right) / 2.0
        self.ear_history.append(avg_ear)
        
        # 檢測眨眼
        if ear_left < self.EAR_THRESHOLD:  #ear_left 只有暫時的 不確定變數名稱
            self.ear_counter += 1
        else:
            if self.ear_counter >= self.EAR_CONSECUTIVE_FRAMES:
                self.blink_count += 1
                self.last_blink_time = time.time()
                self.add_alert("檢測到疲勞指標", "warning")
            self.ear_counter = 0
        
        # 檢測打哈欠
        mar = self.mouth_aspect_ratio(self.mesh_points)
        if mar > self.YAWN_THRESHOLD:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= self.YAWN_CONSECUTIVE_FRAMES:
                self.yawn_count += 1
                self.add_alert("檢測到打哈欠", "warning")
            self.yawn_counter = 0

        # 計算疲勞分數
        if len(self.ear_history) >= 30:
            avg_ear_30 = sum(self.ear_history) / len(self.ear_history)
            if avg_ear_30 < self.EAR_THRESHOLD:
                self.fatigue_score = min(100, self.fatigue_score + 2)
            else:
                self.fatigue_score = max(0, self.fatigue_score - 0.5)
        
        # 更新注意力分數
        time_since_blink = time.time() - self.last_blink_time
        if time_since_blink > 5:  # 5秒沒有眨眼
            self.attention_score = max(0, self.attention_score - 1)
        else:
            self.attention_score = min(100, self.attention_score + 1)  # 加快注意力分數變化
        
        return avg_ear
        
    
    #later fix
    def car_speed(self, speed):
        """分析車速"""
        if speed > 120:
            self.add_alert("車速過快", "warning")
        elif speed < 20:
            self.add_alert("車速過慢", "warning")
        else:
            self.add_alert("車速正常", "info")
        
        return speed

    def analyze_head_pose(self, image, img_w, img_h, nose_3D_point, nose_2D_point):  #pitch, yaw, roll
        """分析頭部姿態"""
        # create the camera matrix
        focal_length = 1 * img_w

        cam_matrix = np.array(
            [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
        )

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        head_pose_points_2D = np.delete(self.head_pose_points_3d, 2, axis=1)
        head_pose_points_3D = self.head_pose_points_3d.astype(np.float64)
        head_pose_points_2D = head_pose_points_2D.astype(np.float64)
        # Solve PnP
        success, rot_vec, trans_vec = cv.solvePnP(
            head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
        )
        # Get rotational matrix
        rotation_matrix, jac = cv.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

        # Get the y rotation degree
        angle_x = angles[0] * 360
        angle_y = angles[1] * 360
        z = angles[2] * 360

        # if angle cross the values then
        threshold_angle = 10
        speed = self.car_speed(60)  # Example speed, replace with actual speed data
        # See where the user's head tilting
        if angle_y > threshold_angle:
            face_looks = "Left"
            if speed > 60:
                self.add_alert(f"Head{face_looks}", "warning")
        elif angle_y < -threshold_angle:
            face_looks = "Right"
            if speed > 60:
                self.add_alert(f"Head{face_looks}", "warning")
        elif angle_x < -threshold_angle:
            face_looks = "Down"
            if speed > 60:
                self.add_alert(f"Head{face_looks}", "warning")
        elif angle_x > threshold_angle:
            face_looks = "Up"
            if speed > 60:
                self.add_alert(f"Head{face_looks}", "warning")
        else:
            face_looks = "Forward"
            if speed > 60:
                self.add_alert(f"Head{face_looks}", "warning")
        # if SHOW_ON_SCREEN_DATA:
        #     cv.putText(
        #         image,
        #         f"Face Looking at {face_looks}",
        #         (img_w - 600, 80),
        #         cv.FONT_HERSHEY_TRIPLEX,
        #         0.8,
        #         (0, 255, 0),
        #         2,
        #         cv.LINE_AA,
        #     )
        # Display the nose direction
        nose_3d_projection, jacobian = cv.projectPoints(
            nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
        )

        p1 = nose_2D_point
        p2 = (
            int(nose_2D_point[0] + angle_y * 10),
            int(nose_2D_point[1] - angle_x * 10),
        )

        cv.line(image, p1, p2, (255, 0, 255), 3)

        return face_looks
    

    def add_alert(self, message, level="info"):
        """添加警報"""
        current_time = time.time()
        alert_key = f"{message}_{level}"
        
        # 檢查冷卻時間
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.alert_cooldown:
                return
        
        self.last_alert_time[alert_key] = current_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.alerts.append({
            "time": timestamp,
            "level": level
        })
        
        # 限制警報數量
        if len(self.alerts) > 20:
            self.alerts.pop(0)

    def draw_eye_landmarks(self, image, landmarks, eye_indices):
        """繪製眼部關鍵點"""
        h, w = image.shape[:2]
        points = []
        
        for idx in eye_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                points.append((x, y))
                cv.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        if len(points) > 0:
            cv.polylines(image, [np.array(points)], True, (0, 255, 0), 1)

    def draw_gaze(self, image, landmarks, eye_indices):
        """繪製眼睛的視線方向"""
        h, w = image.shape[:2]
        
        # 計算瞳孔中心（使用眼部關鍵點的平均位置）
        pupil_x = int(np.mean([landmarks[idx].x for idx in eye_indices]) * w)
        pupil_y = int(np.mean([landmarks[idx].y for idx in eye_indices]) * h)
        
        # 選擇眼睛外側的關鍵點（例如 eye_indices[3]）
        outer_x = int(landmarks[eye_indices[3]].x * w)
        outer_y = int(landmarks[eye_indices[3]].y * h)
        
        # 計算視線方向的終點
        dx = outer_x - pupil_x
        dy = outer_y - pupil_y
        length = 100  # 視線長度（像素）
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            end_x = pupil_x + int(dx * length / magnitude)
            end_y = pupil_y + int(dy * length / magnitude)
            # 繪製視線（藍色線條）
            cv.line(image, (pupil_x, pupil_y), (end_x, end_y), (255, 0, 0), 2)

    # def draw_status_panel(self, image):
    #     """繪製狀態面板"""
    #     h, w = image.shape[:2]
        
    #     # 背景面板
    #     overlay = image.copy()
    #     cv.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    #     cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
    #     # 狀態資訊
    #     y_offset = 35
    #     font = cv.FONT_HERSHEY_SIMPLEX
        
    #     # 系統狀態
    #     status_color = (0, 255, 0)  # 綠色
    #     if self.fatigue_score > 70:
    #         status_color = (0, 0, 255)  # 紅色
    #     elif self.fatigue_score > 40:
    #         status_color = (0, 255, 255)  # 黃色
        
    #     cv.putText(image, "DMS System Status", (20, y_offset), font, 0.6, (255, 255, 255), 2)
    #     y_offset += 25
        
    #     # 疲勞指數
    #     cv.putText(image, f"Fatigue Level: {self.fatigue_score:.1f}%", 
    #                 (20, y_offset), font, 0.5, status_color, 1)
    #     y_offset += 20
        
    #     # 注意力指數
    #     cv.putText(image, f"Attention: {self.attention_score:.1f}%", 
    #                 (20, y_offset), font, 0.5, (255, 255, 255), 1)
    #     y_offset += 20
        
    #     # 眨眼次數
    #     cv.putText(image, f"Blink Count: {self.blink_count}", 
    #                 (20, y_offset), font, 0.5, (255, 255, 255), 1)
    #     y_offset += 20
        
    #     # FPS
    #     cv.putText(image, f"FPS: {self.fps:.1f}", 
    #                 (20, y_offset), font, 0.5, (255, 255, 255), 1)
    #     y_offset += 20
        
    #     # 運行時間
    #     if self.start_time:
    #         runtime = time.time() - self.start_time
    #         minutes = int(runtime // 60)
    #         seconds = int(runtime % 60)
    #         cv.putText(image, f"Runtime: {minutes:02d}:{seconds:02d}", 
    #                     (20, y_offset), font, 0.5, (255, 255, 255), 1)

    # def draw_alerts(self, image):
    #     """繪製警報"""
    #     if not self.alerts:
    #         return
        
    #     h, w = image.shape[:2]
    #     y_start = h - 30
        
    #     # 顯示最近的3個警報
    #     recent_alerts = self.alerts[-3:]
    #     for i, alert in enumerate(reversed(recent_alerts)):
    #         y_pos = y_start - (i * 25)
    #         color = (0, 255, 255) if alert["level"] == "warning" else (255, 255, 255)
    #         # 使用英文顯示避免編碼問題
    #         text = f"{alert['time']}: {alert['message']}"
    #         # 嘗試編碼處理
    #         try:
    #             cv.putText(image, text, (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    #         except:
    #             # 如果中文顯示失敗，使用英文替代
    #             english_text = f"{alert['time']}: Alert - {alert['level']}"
    #             cv.putText(image, english_text, (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def process_frame(self, image):
        """處理單幀圖像"""
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img_h, img_w = rgb_image.shape[:2]
        face_results = self.face_mesh.process(rgb_image)
        
        if face_results.multi_face_landmarks:#先檢測有無臉部以及手部
            with self.data_lock:
                self.mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in face_results.multi_face_landmarks[0].landmark])
                self.mesh_points_3d = np.array([[lm.x, lm.y, lm.z] 
                    for lm in face_results.multi_face_landmarks[0].landmark])
                self.head_pose_points_3d = np.multiply(self.mesh_points_3d[_indices_pose], [img_w, img_h, 1])
                self.head_pose_points_2d = self.mesh_points[_indices_pose]
                self.nose_3d_points = np.multiply(self.head_pose_points_3d[0], [1, 1, 3000])
                self.nose_2d_points = self.head_pose_points_2d[0]
                ear_left = self.blinking_ratio(self.mesh_points_3d)
                ear_right = ear_left # blinking_ratio calculates for both eyes
                self.avg_ear = self.analyze_fatigue(ear_left, ear_right)

                # 計算頭部姿態
                pitch, yaw, roll = self.calculate_head_pose(self.mesh_points, (img_h, img_w))
                self.angle_buffer.add([pitch, yaw, roll])
                pitch, yaw, roll = self.angle_buffer.get_average()
                self.head_pose = self.analyze_head_pose(image, img_w, img_h, self.nose_3d_points, self.nose_2d_points)
        else:
            self.add_alert("未檢測到駕駛員", "warning")
        
        return image
    
    def calculate_fps(self):
        """計算FPS"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def run_2d(self, camera_id=DEFAULT_WEBCAM):
        """運行DMS系統並產生2D影像串流"""
        self.cap = cv.VideoCapture(camera_id)

        if not self.cap.isOpened():
            print("錯誤：無法打開攝影機")
            return

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv.CAP_PROP_FPS, 30)

        self.is_running = True
        self.start_time = time.time()
        self.add_alert("DMS系統啟動", "info")

        print("DMS系統運行中...")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                key = cv.waitKey(1) & 0xFF
                if not ret:
                    print("錯誤：無法讀取攝影機畫面")
                    break

                processed_frame = self.process_frame(frame)

                # 將影像編碼為 JPEG
                (flag, encodedImage) = cv.imencode(".jpg", processed_frame)
                frame = encodedImage.tobytes()
                if not flag:
                    continue

                # 產生影像串流
                yield(b'--frame\r\n' 
                      b'Content-Type: image/jpeg\r\n\r\n' + 
                      frame +
                      b'\r\n')

        except GeneratorExit:
            print("Streaming client disconnected.")
        finally:
            self.cleanup()

    def reset_status(self):
        """重置統計數據"""
        self.fatigue_score = 0
        self.attention_score = 100
        self.blink_count = 0
        self.avg_ear = 0
        self.yawn_count = 0
        self.ear_history.clear()
        self.head_pose_history.clear()
        self.alerts.clear()
        self.start_time = time.time()

    def cleanup(self):
        """清理資源"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv.destroyAllWindows()
        print("DMS系統已關閉")

dms = DMSSystem()

app = Flask(__name__, template_folder='templates', static_folder='static') # 指定 template_folder 和 static_folder
CORS(app)
dms.run_2d()  # 啟動 DMS 2d 系統
# 首頁路由，用於提供 HTML 儀表板
@app.route('/')
def index():
    return render_template('DMS_hank.html')

@app.route('/membership')
def show_membership_page():
    return render_template('membership.html')

@app.route('/login')
def show_login_page():
    return render_template('login.html')


# 影像串流路由
@app.route('/video_feed')
def video_feed():
    return Response(dms.run_2d(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Gemini 
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # 你的 Google Studio 程式碼片段
        model = "gemini-2.5-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_message),
                ],
            ),
        ]
        tools = [
            types.Tool(googleSearch=types.GoogleSearch()),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(),
            tools=tools,
        )

        # 這裡改用 client.models.generate_content_stream()
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            # 串流回應，將所有文字合併
            response_text += chunk.text

        # 回傳完整的 Gemini 回應
        return jsonify({"reply": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# 資料 API 路由
@app.route('/data')
def data():
    with dms.data_lock:
        return jsonify({
            'fatigue_level': dms.fatigue_score,
            'attention_level': dms.attention_score,
            'blink_count': dms.blink_count,
            'yawn_count': dms.yawn_count,
            'head_pose': dms.head_pose,
            'ear': dms.avg_ear,
            'alerts': dms.alerts
        })

# 重置統計數據的 API 路由
@app.route('/reset_status', methods=['POST'])
def reset_status():
    dms.reset_status()
    return jsonify({"status": "success", "message": "統計數據已重置"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) # 使用 threaded=True 允許並發請求