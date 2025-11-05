import os

import cv2

# === 설정 ===
base_dir = "dataset"
chords = ["C", "D", "G"]
max_images = 30  # 각 코드당 30장

# 폴더가 없으면 자동 생성
for chord in chords:
    os.makedirs(os.path.join(base_dir, chord), exist_ok=True)

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("=== Guitar Chord Dataset Creator ===")
print("C / D / G 키를 눌러 사진을 찍습니다.")
print("q 키를 누르면 종료합니다.")
print("==============================")

# 각 코드별 저장된 파일 개수 확인
count = {}
for chord in chords:
    files = [f for f in os.listdir(os.path.join(base_dir, chord)) if f.endswith(".png")]
    count[chord] = len(files)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("Create Dataset - Press C / D / G", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # q 누르면 종료
        break

    for chord in chords:
        if key == ord(chord.lower()) or key == ord(chord.upper()):
            if count[chord] < max_images:
                # 파일명: 0001.png 형식
                filename = f"{count[chord]+1:04d}.png"
                save_path = os.path.join(base_dir, chord, filename)
                cv2.imwrite(save_path, frame)
                count[chord] += 1
                print(f"[{chord}] {filename} 저장 완료 ({count[chord]}/{max_images})")
            else:
                print(f"[{chord}] 이미 {max_images}장 저장 완료!")

cap.release()
cv2.destroyAllWindows()
