#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset/C, D, G 폴더의 이미지를 돌며 MediaPipe Hands로 21개 랜드마크(x,y,(z)) 추출.
- 왼손은 좌우 반전(mirror)로 정규화해 좌우 일관성 확보
- (x,y)는 손목(랜드마크 0) 기준으로 평행이동 후, 박스 크기(max(width,height))로 스케일 정규화
- 기본은 x,y만 사용(2D). --use_z로 z까지 포함 가능
출력: data/hand_landmarks.csv
"""
import os, argparse, glob, csv
import cv2
import numpy as np
import mediapipe as mp

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def extract_one(img_bgr, mirror_left=True, use_z=False):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0].landmark
        handed_label = None
        if res.multi_handedness and len(res.multi_handedness) > 0:
            handed_label = res.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

        xs = np.array([p.x for p in lm], dtype=np.float32)
        ys = np.array([p.y for p in lm], dtype=np.float32)
        zs = np.array([p.z for p in lm], dtype=np.float32)

        # 좌우 일관성: 왼손이면 x 반전
        if mirror_left and handed_label == "Left":
            xs = -xs

        # 손목(0) 기준 평행이동
        xs -= xs[0]; ys -= ys[0]; 
        if use_z: zs -= zs[0]

        # 스케일 정규화(박스 크기)
        scale = max(xs.max() - xs.min(), ys.max() - ys.min())
        if scale < 1e-6: 
            scale = 1.0
        xs /= scale; ys /= scale
        if use_z: zs /= scale

        # 피처 구성: [x0..x20, y0..y20(, z0..z20)]
        feats = np.concatenate([xs, ys]) if not use_z else np.concatenate([xs, ys, zs])
        return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", help="이미지 루트 폴더")
    ap.add_argument("--out",  default="data/hand_landmarks.csv")
    ap.add_argument("--classes", nargs="+", default=["C","D","G"])
    ap.add_argument("--use_z", action="store_true", help="z 좌표 포함")
    ap.add_argument("--no_mirror_left", action="store_true", help="왼손 좌우 반전 비활성화")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 컬럼 이름
    dim = 3 if args.use_z else 2
    headers = ["filepath","label","label_id"]
    for ax in ("x","y","z")[:dim]:
        headers += [f"{ax}{i}" for i in range(21)]

    cls_to_idx = {c:i for i,c in enumerate(sorted(args.classes))}
    total, ok, miss = 0, 0, 0

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
        for c in sorted(args.classes):
            class_dir = os.path.join(args.root, c)
            if not os.path.isdir(class_dir):
                print(f"[WARN] Not a directory: {class_dir}")
                continue
            for path in glob.glob(os.path.join(class_dir, "*")):
                if not path.lower().endswith(VALID_EXTS): 
                    continue
                total += 1
                img = cv2.imread(path)
                if img is None:
                    miss += 1
                    continue
                feats = extract_one(img, mirror_left=(not args.no_mirror_left), use_z=args.use_z)
                if feats is None:
                    miss += 1
                    continue
                row = [path, c, cls_to_idx[c]] + feats.tolist()
                w.writerow(row)
                ok += 1

    print(f"[DONE] total={total}, extracted={ok}, failed={miss}, saved={args.out}")
    print(f"[INFO] class_to_idx = {cls_to_idx}")

if __name__ == "__main__":
    main()


# 실행코드 
# python scripts/01_extract_landmarks.py --use_z