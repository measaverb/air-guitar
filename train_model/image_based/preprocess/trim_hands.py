import os
from glob import glob

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

pad_portion = 0.25

images = glob("data/raw/*/*.png")
images.sort()


def trim_hand_by_coordinates(image, pad_portion, h, w, lm, square=False):
    max_x = 0
    max_y = 0
    min_x = w
    min_y = h
    for _, ld in enumerate(lm.landmark):
        x, y = int(ld.x * w), int(ld.y * h)
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y

    x_pad = int((max_x - min_x) * pad_portion)
    y_pad = int((max_y - min_y) * pad_portion)

    box_min_x = min_x - x_pad
    box_min_y = min_y - y_pad
    box_max_x = max_x + x_pad
    box_max_y = max_y + y_pad

    if square:
        box_w = box_max_x - box_min_x
        box_h = box_max_y - box_min_y
        box_size = max(box_w, box_h)
        center_x = (box_max_x + box_min_x) // 2
        center_y = (box_max_y + box_min_y) // 2

        box_min_x = center_x - box_size // 2
        box_min_y = center_y - box_size // 2
        box_max_x = box_min_x + box_size
        box_max_y = box_min_y + box_size

    pad_left = max(0, -box_min_x)
    pad_top = max(0, -box_min_y)
    pad_right = max(0, box_max_x - w)
    pad_bottom = max(0, box_max_y - h)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    crop_min_x = box_min_x + pad_left
    crop_min_y = box_min_y + pad_top
    crop_max_x = box_max_x + pad_left
    crop_max_y = box_max_y + pad_top

    image = image[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

    return image


for image in images:
    img = cv2.imread(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    chord_name = image.split("/")[-2]

    with mp_hands.Hands(
        static_image_mode=True,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.1,
    ) as hands:
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                img = trim_hand_by_coordinates(img, pad_portion, h, w, lm, square=True)
        else:
            print(f"No hand detected in {image}, skipping...")
            continue

    os.makedirs(f"data/processed/{chord_name}", exist_ok=True)
    cv2.imwrite(f"data/processed/{chord_name}/{os.path.basename(image)}", img)
