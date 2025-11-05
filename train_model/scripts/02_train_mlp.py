#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV를 읽어 MLP로 C/D/G 분류 학습.
- 80/10/10 stratified split
- 표준화 + MLPClassifier(early_stopping)
- 모델 저장: artifacts/mlp.joblib, 레이블 매핑 저장: artifacts/labels.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/hand_landmarks.csv")
    ap.add_argument("--model_out", default="artifacts/mlp.joblib")
    ap.add_argument("--labels_out", default="artifacts/labels.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    df = pd.read_csv(args.csv)
    y = df["label_id"].values.astype(int)
    # 피처 컬럼 자동 탐색 (x*, y*, z* 순)
    feat_cols = [c for c in df.columns if c.startswith(("x", "y", "z"))]
    X = df[feat_cols].values.astype(np.float32)

    # 80/10/10 split (test 10%, val 10%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=args.seed
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=tuple(args.hidden),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=64,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=1 / 9,  # train:val:test=8:1:1
                    n_iter_no_change=20,  #  20 epoch 연속으로 향상되지 않으면, 학습이 500 에포크에 도달하지 않았더라도 훈련을 중단
                    random_state=args.seed,
                    verbose=True,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)  # 학습

    def eval_split(name, Xs, ys):
        pred = pipe.predict(Xs)
        acc = accuracy_score(ys, pred)
        print(f"\n[{name}] acc = {acc:.4f}")
        print(classification_report(ys, pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(ys, pred))

    eval_split("TEST", X_test, y_test)

    # 파이프라인 객체 저장
    dump(pipe, args.model_out)
    # 레이블 매핑 복원용
    labels = (
        df[["label", "label_id"]]
        .drop_duplicates()
        .sort_values("label_id")
        .to_dict("records")
    )
    with open(args.labels_out, "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] model => {args.model_out}")
    print(f"[SAVED] labels => {args.labels_out}")


if __name__ == "__main__":
    main()


# 실행코드
# python scripts/02_train_mlp.py --hidden 512 256 128 64
