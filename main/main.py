#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import time
from collections import deque
import math
import cv2
import numpy as np
from joblib import load
import mediapipe as mp
import pygame

# --- 1. 사운드 관리 클래스 ---
class SoundManager:
    """
    Pygame 믹서를 초기화하고, 코드 사운드를 로드하며,
    채널을 관리하여 사운드를 재생합니다.
    """
    def __init__(self, chord_labels_data):
        print("사운드 매니저 초기화 중...")
        pygame.mixer.init()
        pygame.mixer.set_num_channels(24)
        self.sounds = self._load_sounds(chord_labels_data)
        print("사운드 매니저 초기화 완료.")

    def _load_sounds(self, chord_labels):
        sounds = {}
        sound_dir = "sounds/"
        print("코드 사운드 로드 중...")
        for label_info in chord_labels:
            chord_name = label_info["label"]
            file_path = f"{sound_dir}{chord_name}.wav"
            try:
                sounds[chord_name] = pygame.mixer.Sound(file_path)
                print(f" - '{file_path}' 로드 성공")
            except (pygame.error, FileNotFoundError):
                print(f" - 경고: '{file_path}' 파일을 찾을 수 없거나 로드할 수 없습니다.")
        return sounds

    def play_stroke(self, chord_name):
        """
        스트로크 시, 유효한 코드 사운드를 재생합니다.
        채널이 꽉 찼으면 가장 오래된 소리를 중지하고 재생합니다.
        """
        if chord_name != "None" and chord_name in self.sounds:
            sound_to_play = self.sounds[chord_name]
            channel = pygame.mixer.find_channel(True)
            if channel:
                channel.play(sound_to_play)


# --- 2. 코드 분류기 클래스 ---
class ChordClassifier:
    """
    ML 모델을 로드하고, 왼손 랜드마크를 받아 코드를 분류합니다.
    """
    def __init__(self, model_path, labels_data, use_z=False, smooth_frames=5):
        print("코드 분류기 초기화 중...")
        try:
            self.pipe = load(model_path)
            self.idx2label = {d["label_id"]: d["label"] for d in labels_data}
        except FileNotFoundError as e:
            print(f"오류: 모델 또는 라벨 파일 로드 실패. {e}")
            raise
            
        self.use_z = use_z
        expected_features = 63 if self.use_z else 42
        print(f" - Z축 사용: {self.use_z} (예상 특성 개수: {expected_features})")
        
        self.prob_buf = deque(maxlen=max(1, smooth_frames))
        print("코드 분류기 초기화 완료.")

    @staticmethod
    def _extract_feature(hand_landmarks, handedness_label, mirror_left=True, use_z=False):
        """ML 모델 입력을 위한 특징을 추출합니다. (Static Method)"""
        lm = hand_landmarks.landmark
        xs = np.array([p.x for p in lm], dtype=np.float32)
        ys = np.array([p.y for p in lm], dtype=np.float32)
        zs = np.array([p.z for p in lm], dtype=np.float32)

        if mirror_left and handedness_label == "Left":
            xs = -xs

        xs -= xs[0]
        ys -= ys[0]
        if use_z:
            zs -= zs[0]

        scale = max(xs.max() - xs.min(), ys.max() - ys.min())
        if scale < 1e-6: scale = 1.0
        xs /= scale
        ys /= scale
        if use_z: zs /= scale

        feats = np.concatenate([xs, ys]) if not use_z else np.concatenate([xs, ys, zs])
        return feats

    def classify(self, hand_landmarks, handedness_label):
        """랜드마크를 받아 코드를 분류하고, 스무딩을 적용합니다."""
        feats = self._extract_feature(
            hand_landmarks, handedness_label, use_z=self.use_z
        )
        if feats is None:
            return "None", "Chord: None"

        feats = feats.reshape(1, -1)
        
        try:
            proba = self.pipe.predict_proba(feats)[0]
        except ValueError as e:
            print(f"!!! 분류 오류: {e}")
            self.reset()
            return "None", "Chord: Error"

        self.prob_buf.append(proba)
        
        proba_smooth = np.mean(self.prob_buf, axis=0)
        cls_id = int(np.argmax(proba_smooth))
        cls_name = self.idx2label.get(cls_id, str(cls_id))
        conf = float(proba_smooth[cls_id])
        
        current_chord = cls_name
        chord_text = f"Chord: {cls_name} ({conf*100:.1f}%)"
        
        return current_chord, chord_text

    def reset(self):
        """확률 버퍼를 초기화합니다."""
        self.prob_buf.clear()


# --- 3. 스트럼 감지기 클래스 ---
class StrumDetector:
    """
    오른손 검지 끝의 속도를 추적하여 스트로크를 감지합니다.
    """
    def __init__(self, cooldown, threshold_callback):
        self.cooldown = cooldown
        self.get_sensitivity_threshold = threshold_callback
        self.prev_y = None
        self.last_stroke_time = 0
        print("스트럼 감지기 초기화 완료.")

    def detect(self, hand_landmarks, frame_h, frame_w):
        """
        오른손 랜드마크를 받아 스트로크 여부를 감지합니다.
        반환값: (stroke_detected (bool), stroke_text (str))
        """
        stroke_text = ""
        stroke_detected = False
        current_time = time.time()
        
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        index_tip_x_px, index_tip_y_px = int(index_tip.x * frame_w), int(index_tip.y * frame_h)
        
        stroke_threshold = self.get_sensitivity_threshold()

        if self.prev_y is not None and (current_time - self.last_stroke_time) > self.cooldown:
            velocity_y = index_tip_y_px - self.prev_y
            
            if velocity_y > stroke_threshold:
                stroke_text = "Downstroke!"
                stroke_detected = True
                self.last_stroke_time = current_time
        
        self.prev_y = index_tip_y_px
        return stroke_detected, stroke_text

    def reset(self):
        """오른손이 감지되지 않으면 속도 추적을 리셋합니다."""
        self.prev_y = None


# --- 4. 기타 렌더러 클래스 ---
class GuitarRenderer:
    """
    Pose 모델을 실행하고, 어깨 위치에 맞춰 기타 이미지를 오버레이합니다.
    """
    def __init__(self, image_path, min_conf=0.7):
        print("기타 렌더러 초기화 중...")
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf
        )
        try:
            self.guitar_img_original = cv2.imread(image_path, -1)
            if self.guitar_img_original is None: raise FileNotFoundError(image_path)
            if self.guitar_img_original.shape[2] != 4:
                raise ValueError("기타 이미지는 알파 채널(투명도)이 있는 4채널 PNG여야 합니다.")
                
            # --- [수정] ---
            # 초기화 시 투명도를 고정하지 않고 원본 알파 채널 유지
            # opacity = 160
            # alpha_channel = self.guitar_img_original[:, :, 3].astype(np.float32) * (opacity/255.0)
            # self.guitar_img_original[:, :, 3] = alpha_channel.astype(np.uint8)
            
            self.guitar_img_original = cv2.flip(self.guitar_img_original, 1)
            print("기타 이미지 로드 성공.")

        except (FileNotFoundError, ValueError) as e:
            print(f"오류: 기타 이미지('{image_path}') 로드 실패. {e}")
            self.guitar_img_original = None

        print("기타 렌더러 초기화 완료.")

    def process_pose(self, image_rgb):
        """Pose 모델을 실행합니다."""
        return self.pose.process(image_rgb)

    # [수정] current_opacity 인자 추가
    def draw_overlay(self, image, pose_results, current_opacity):
        """
        Pose 결과에 따라 기타 이미지를 그립니다.
        """
        if self.guitar_img_original is None:
            return

        h, w, _ = image.shape
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]

            if all(lm.visibility > 0.5 for lm in [left_shoulder, right_shoulder]):
                left_shoulder_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                right_shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))

                shoulder_center_x = (left_shoulder_px[0] + right_shoulder_px[0]) // 2
                shoulder_center_y = (left_shoulder_px[1] + right_shoulder_px[1]) // 2
                shoulder_width = int(math.dist(left_shoulder_px, right_shoulder_px))
                
                if shoulder_width <= 0: return

                guitar_center_x = shoulder_center_x + int(shoulder_width * 0.15) 
                guitar_center_y = shoulder_center_y + int(shoulder_width * 0.65)
                guitar_width = int(shoulder_width * 3.0)
                
                base_angle = math.degrees(math.atan2(right_shoulder_px[1] - left_shoulder_px[1], 
                                                     right_shoulder_px[0] - left_shoulder_px[0]))
                final_angle = -base_angle + 60
                
                if guitar_width > 0:
                    scale = guitar_width / self.guitar_img_original.shape[1]
                    resized_height = int(self.guitar_img_original.shape[0] * scale)
                    if resized_height > 0:
                        resized_guitar = cv2.resize(self.guitar_img_original, (guitar_width, resized_height), interpolation=cv2.INTER_AREA)
                        M = cv2.getRotationMatrix2D((guitar_width / 2, resized_height / 2), final_angle, 1)
                        rotated_guitar = cv2.warpAffine(resized_guitar, M, (guitar_width, resized_height))

                        # [수정] current_opacity 인자 전달
                        self._alpha_blend(image, rotated_guitar, guitar_center_x, guitar_center_y, current_opacity)

    # [수정] current_opacity 인자 추가
    def _alpha_blend(self, background, overlay, center_x, center_y, current_opacity):
        """
        알파 채널을 사용해 이미지를 오버레이합니다. (Helper method)
        """
        h, w, _ = background.shape
        rh, rw, _ = overlay.shape

        y1 = center_y - rh // 2
        y2 = y1 + rh
        x1 = center_x - rw // 2
        x2 = x1 + rw

        y1_c, y2_c = max(0, y1), min(h, y2)
        x1_c, x2_c = max(0, x1), min(w, x2)
        
        overlay_y1, overlay_x1 = max(0, -y1), max(0, -x1)
        overlay_y2 = overlay_y1 + (y2_c - y1_c)
        overlay_x2 = overlay_x1 + (x2_c - x1_c)

        if overlay_y2 <= overlay_y1 or overlay_x2 <= overlay_x1:
            return

        # [수정] 슬라이더의 투명도 값을 (0-1) 스케일로 변환
        user_alpha_scale = current_opacity / 255.0

        # [수정] 이미지 자체의 알파 채널과 사용자의 투명도 설정을 모두 곱함
        alpha_s = (overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0) * user_alpha_scale
        alpha_l = 1.0 - alpha_s

        roi = background[y1_c:y2_c, x1_c:x2_c]
        
        for c in range(0, 3):
            overlay_rgb = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c]
            roi[:, :, c] = (alpha_s * overlay_rgb + alpha_l * roi[:, :, c])

    def close(self):
        """Pose 모델 리소스를 해제합니다."""
        self.pose.close()


# --- 5. 메인 애플리케이션 클래스 ---
class AirGuitarApp:
    """
    모든 컴포넌트를 통합하고 메인 루프를 실행하는
    메인 애플리케이션 클래스입니다.
    """
    WINDOW_NAME = 'Real-time Air Guitar'

    def __init__(self, args):
        print("Air Guitar 애플리케이션 시작 중...")
        self.args = args
        self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            raise IOError(f"카메라 {args.camera}를 열 수 없습니다.")
        
        cv2.namedWindow(self.WINDOW_NAME)

        try:
            with open(args.labels, "r") as f:
                labels_data = json.load(f)
        except FileNotFoundError:
            print(f"오류: 라벨 파일 '{args.labels}'를 찾을 수 없습니다.")
            raise

        # --- 컴포넌트 초기화 ---
        self.sound_manager = SoundManager(labels_data)
        
        use_z_flag = not args.dont_use_z
        
        self.chord_classifier = ChordClassifier(
            args.model, labels_data, use_z_flag, args.smooth
        )
        self.strum_detector = StrumDetector(
            cooldown=0.2, 
            threshold_callback=self.get_sensitivity
        )
        self.guitar_renderer = GuitarRenderer(
            args.guitar_img, args.min_det_conf
        )
        
        self.hands_model = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=args.min_det_conf,
            min_tracking_confidence=args.min_track_conf
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # --- [수정] 토글 상태 변수들 (기본값: 모두 켜기) ---
        self.show_all_visuals = True  # 'a' (마스터 토글)
        self.show_sensitivity_ui = True # 's' (트랙바, 민감도 텍스트)
        self.show_opacity_slider = True # 'e' (투명도 트랙바) [신규]
        self.show_markers = True      # 'm' (손 마커)
        self.show_guitar = True       # 'g' (기타 이미지)
        self.show_chord_text = True   # 'c' (코드 텍스트)
        self.show_stroke_text = True  # 't' (스트로크 텍스트)
        
        self.current_sensitivity = 30 # 기본값
        self.current_opacity = 160    # 기본값 [신규]

        # [수정] 도움말 업데이트
        print("--- Air Guitar 실행 (종료: q, 전체: a, 민감도: s, 마커: m, 투명도: e, 스트로크: t, 기타: g, 코드: c) ---")

        # [수정] 기본값이 켜기이므로, 트랙바를 처음부터 생성
        cv2.createTrackbar('Sensitivity', self.WINDOW_NAME, self.current_sensitivity, 100, self._on_sensitivity_change)
        cv2.createTrackbar('Opacity', self.WINDOW_NAME, self.current_opacity, 255, self._on_opacity_change) # [신규]

    def _on_sensitivity_change(self, val):
        """트랙바 콜백: 현재 민감도 값을 업데이트합니다."""
        self.current_sensitivity = val

    # [신규] 투명도 트랙바 콜백
    def _on_opacity_change(self, val):
        """트랙바 콜백: 현재 투명도 값을 업데이트합니다."""
        self.current_opacity = val

    def get_sensitivity(self):
        """저장된 민감도 값을 반환합니다."""
        return self.current_sensitivity

    def run(self):
        """메인 애플리케이션 루프를 실행합니다."""
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("카메라 프레임 읽기 실패.")
                break
                
            # image = cv2.flip(image, 1) # 비-거울 모드
            h, w, _ = image.shape

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pose_results = self.guitar_renderer.process_pose(image_rgb)
            hand_results = self.hands_model.process(image_rgb)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # [수정] draw_overlay에 current_opacity 값 전달
            if self.show_all_visuals and self.show_guitar:
                self.guitar_renderer.draw_overlay(image, pose_results, self.current_opacity)
            
            chord, chord_text, stroke, stroke_text = self._process_hands(
                image, hand_results
            )

            if stroke:
                self.sound_manager.play_stroke(chord)

            self._draw_ui(image, chord_text, stroke_text)

            cv2.imshow(self.WINDOW_NAME, image)
            
            # --- [수정] 키보드 입력 처리 ---
            key = cv2.waitKey(5) & 0xFF
            trackbar_state_changed = False
            
            if key == ord('q'):
                break # 'q' (종료)
            
            elif key == ord('a'): # 'a' (마스터 토글)
                self.show_all_visuals = not self.show_all_visuals
                trackbar_state_changed = True 
                print(f"모든 시각 요소: {'ON' if self.show_all_visuals else 'OFF'}")

            elif key == ord('s'): # 's' (민감도 UI 토글)
                self.show_sensitivity_ui = not self.show_sensitivity_ui
                trackbar_state_changed = True
                print(f"민감도 UI: {'ON' if self.show_sensitivity_ui else 'OFF'}")

            elif key == ord('m'): # 'm' (마커 토글)
                self.show_markers = not self.show_markers
                print(f"손 마커 표시: {'ON' if self.show_markers else 'OFF'}")

            # [신규] 'e' 키로 투명도 슬라이더 토글
            elif key == ord('e'):
                self.show_opacity_slider = not self.show_opacity_slider
                trackbar_state_changed = True
                print(f"투명도 슬라이더: {'ON' if self.show_opacity_slider else 'OFF'}")

            elif key == ord('t'): # 't' (스트로크 텍스트 토글)
                self.show_stroke_text = not self.show_stroke_text
                print(f"스트로크 텍스트 표시: {'ON' if self.show_stroke_text else 'OFF'}")

            elif key == ord('g'): # 'g' (기타 토글)
                self.show_guitar = not self.show_guitar
                print(f"기타 이미지 표시: {'ON' if self.show_guitar else 'OFF'}")

            elif key == ord('c'): # 'c' (코드 텍스트 토글)
                self.show_chord_text = not self.show_chord_text
                print(f"코드 텍스트 표시: {'ON' if self.show_chord_text else 'OFF'}")
            
            # [수정] 'a', 's', 'e' 키가 눌렸을 때만 창을 새로고침
            if trackbar_state_changed:
                cv2.destroyWindow(self.WINDOW_NAME)
                cv2.namedWindow(self.WINDOW_NAME)
                
                # 민감도 트랙바 생성 로직
                if self.show_all_visuals and self.show_sensitivity_ui:
                    cv2.createTrackbar('Sensitivity', self.WINDOW_NAME, self.current_sensitivity, 100, self._on_sensitivity_change)
                
                # [신규] 투명도 트랙바 생성 로직
                if self.show_all_visuals and self.show_opacity_slider:
                    cv2.createTrackbar('Opacity', self.WINDOW_NAME, self.current_opacity, 255, self._on_opacity_change)


    def _process_hands(self, image, hand_results):
        """
        Hands 모델 결과를 처리하여 코드와 스트로크를 감지합니다.
        (손 역할이 반대로 수정됨)
        """
        chord_hand_detected = False
        strum_hand_detected = False
        
        current_chord = "None"
        chord_text = "Chord: None"
        stroke_detected = False
        stroke_text = ""
        h, w, _ = image.shape

        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                
                # [수정] 마스터 토글('a')과 마커 토글('m') 모두 확인
                if self.show_all_visuals and self.show_markers:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                
                handedness_label = hand_results.multi_handedness[i].classification[0].label

                if handedness_label == "Right": # 오른손(사용자 실제 손) -> 코드 잡기
                    chord_hand_detected = True
                    current_chord, chord_text = self.chord_classifier.classify(
                        hand_landmarks, handedness_label
                    )
                elif handedness_label == "Left": # 왼손(사용자 실제 손) -> 스트럼
                    strum_hand_detected = True
                    stroke_detected, stroke_text = self.strum_detector.detect(
                        hand_landmarks, h, w
                    )

        if not strum_hand_detected:
            self.strum_detector.reset()
            
        if not chord_hand_detected:
            self.chord_classifier.reset()
            current_chord = "None" 
            chord_text = "Chord: None"

        return current_chord, chord_text, stroke_detected, stroke_text

    def _draw_ui(self, image, chord_text, stroke_text):
        """화면에 텍스트 UI를 그립니다."""
        
        # 마스터 토글('a')이 꺼져있으면 아무것도 그리지 않음
        if not self.show_all_visuals:
            return

        h, w, _ = image.shape

        # 's' 키로 토글
        if self.show_sensitivity_ui:
            cv2.putText(image, f'Sensitivity: {self.get_sensitivity()}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 'c' 키로 토글
        if self.show_chord_text:
            cv2.putText(image, chord_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 180, 80), 2, cv2.LINE_AA)
        
        # 't' 키로 토글
        if self.show_stroke_text and stroke_text:
            stroke_ui_text = stroke_text
            color = (0, 255, 0)
            
            (text_w, text_h), _ = cv2.getTextSize(stroke_ui_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)
            text_x = (w - text_w) // 2
            text_y = (h + text_h) // 2
            cv2.putText(image, stroke_ui_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5, cv2.LINE_AA)

    def cleanup(self):
        """애플리케이션 종료 시 모든 리소스를 해제합니다."""
        print("--- 프로그램 종료 ---")
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        self.hands_model.close()
        self.guitar_renderer.close()


# --- 6. 메인 실행 블록 ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./artifacts/mlp.joblib", help="코드 분류기 모델 경로")
    ap.add_argument("--labels", default="./artifacts/labels.json", help="모델 라벨 json 파일 경로")
    ap.add_argument("--guitar_img", default="3.png", help="기타 오버레이 이미지 경로")
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스")
    ap.add_argument("--dont_use_z", action="store_true", help="ML 모델이 Z축을 사용하지 *않는다면* 설정 (기본값: Z축 사용)")
    ap.add_argument("--smooth", type=int, default=5, help="코드 예측 스무딩 프레임 수")
    ap.add_argument("--min_det_conf", type=float, default=0.7, help="최소 감지 신뢰도")
    ap.add_argument("--min_track_conf", type=float, default=0.7, help="최소 추적 신뢰도")
    args = ap.parse_args()

    app_initialized = False
    try:
        app = AirGuitarApp(args)
        app_initialized = True
        app.run()
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
    finally:
        if app_initialized:
            app.cleanup()