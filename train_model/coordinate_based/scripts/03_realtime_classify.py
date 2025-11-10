#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from joblib import load


# ---- 전처리: 훈련과 동일 ----
def extract_feature_from_result(res, mirror_left=True, use_z=False):
    """res: mediapipe result (이미 max_num_hands=1 가정)"""
    if not res.multi_hand_landmarks:
        return None

    lm = res.multi_hand_landmarks[0].landmark

    handed = None
    if res.multi_handedness:
        handed = res.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

    xs = np.array([p.x for p in lm], dtype=np.float32)
    ys = np.array([p.y for p in lm], dtype=np.float32)
    zs = np.array([p.z for p in lm], dtype=np.float32)

    # 왼손이면 x 반전(훈련 스크립트의 기본과 동일)
    if mirror_left and handed == "Left":
        xs = -xs

    # 손목(0) 기준 평행이동
    xs -= xs[0]
    ys -= ys[0]
    if use_z:
        zs -= zs[0]

    # 스케일 정규화(박스 크기)
    scale = max(xs.max() - xs.min(), ys.max() - ys.min())
    if scale < 1e-6:
        scale = 1.0
    xs /= scale
    ys /= scale
    if use_z:
        zs /= scale

    feats = np.concatenate([xs, ys]) if not use_z else np.concatenate([xs, ys, zs])
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/mlp.joblib")
    ap.add_argument("--labels", default="artifacts/labels.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument(
        "--use_z", action="store_true", help="훈련을 z 포함으로 했을 때만 켜세요"
    )
    ap.add_argument("--smooth", type=int, default=8, help="최근 N프레임 확률 이동평균")
    ap.add_argument("--min_det_conf", type=float, default=0.5)
    ap.add_argument("--min_track_conf", type=float, default=0.5)
    ap.add_argument("--no_draw", action="store_true", help="랜드마크 그리지 않기")
    args = ap.parse_args()

    # 모델 & 라벨 매핑
    pipe = load(args.model)
    with open(args.labels, "r") as f:
        labels = json.load(f)
    idx2label = {d["label_id"]: d["label"] for d in labels}

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # 한 손만
        model_complexity=0,  # 속도↑
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    prob_buf = deque(maxlen=max(1, args.smooth))
    last_pred = None
    t0, frames = time.time(), 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames += 1

            # MediaPipe는 RGB 입력
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            h, w = frame_bgr.shape[:2]

            # 분류
            text = "No hand"
            color = (64, 64, 64)
            if res.multi_hand_landmarks:
                feats = extract_feature_from_result(
                    res, mirror_left=True, use_z=args.use_z
                )
                if feats is not None:
                    feats = feats.reshape(1, -1)
                    proba = pipe.predict_proba(feats)[0]
                    prob_buf.append(proba)
                    proba_smooth = np.mean(prob_buf, axis=0)
                    cls_id = int(np.argmax(proba_smooth))
                    cls_name = idx2label.get(cls_id, str(cls_id))
                    conf = float(proba_smooth[cls_id])
                    last_pred = (cls_name, conf)
                    text = f"{cls_name}  {conf*100:.1f}%"
                    color = (40, 180, 80)  # green-ish

                # 시각화(선택)
                if not args.no_draw:
                    mp_draw.draw_landmarks(
                        frame_bgr,
                        res.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            # FPS
            dt = time.time() - t0
            fps = frames / dt if dt > 0 else 0.0

            # 오버레이
            cv2.rectangle(frame_bgr, (10, 10), (310, 80), (0, 0, 0), -1)
            cv2.putText(
                frame_bgr,
                text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                f"FPS:{fps:.1f}",
                (220, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Chord Classifier (q to quit)", frame_bgr)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()


# 실행코드
# python scripts/03_realtime_classify.py --use_z
