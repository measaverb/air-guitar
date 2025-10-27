import time
import cv2
import mediapipe as mp
import numpy as np
import math


# --- 1. 초기 설정 ---
# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
# Pose 감지 모델 인스턴스 생성 (기본값 사용)
pose = mp_pose.Pose()

# 기타 이미지 불러오기 (배경이 투명한 PNG 파일)
try:
    # cv2.IMREAD_UNCHANGED (-1) 플래그는 알파 채널(투명도)까지 포함하여 4채널로 불러옴
    guitar_img = cv2.imread('3.png', -1)
    
    # 파일이 존재하지 않거나 읽을 수 없으면 guitar_img는 None이 됨
    if guitar_img is None:
        raise FileNotFoundError
    
    # guitar_img.shape[2]는 이미지의 채널 수를 의미. 4가 아니면(BGR만 있으면, 투명도가 없는 사진이라면) 오류
    if guitar_img.shape[2] != 4:
        print(f"오류: '3.png' 파일에 알파 채널(투명도)이 없습니다.")
        print("배경이 투명한 4채널 PNG 파일을 사용하세요.")
        exit()
        
    # 투명도 설정 (0~255)
    opacity = 160 # 160으로 지정
    guitar_img[:, :, 3] = (guitar_img[:, :, 3].astype(np.float32) * (opacity/255.0)).astype(np.uint8)

    # 기타 이미지 좌우 반전
    guitar_img = cv2.flip(guitar_img, 1)
    

except FileNotFoundError:
    # try 블록에서 FileNotFoundError가 발생하거나 guitar_img가 None일 때 실행됨
    print("오류: '3.png' 파일을 찾을 수 없습니다.")
    print("스크립트와 동일한 폴더에 파일이 있는지 확인하세요.")
    exit()

# 웹캠 열기 (0번 카메라)
cap = cv2.VideoCapture(0)

# 🎥 저장
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('airguitar_output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# --- 2. 메인 루프 ---
# cap.isOpened()가 True인 동안 (카메라가 정상 연결된 동안) 무한 반복
while cap.isOpened():
    # 카메라에서 프레임(image)과 성공 여부(success)를 읽어옴
    success, image = cap.read()
    if not success:
        print("카메라 프레임을 읽는 데 실패했습니다.")
        continue # 다음 루프로 건너뜀

    # --- 3. 이미지 전처리 ---
    
    # 웹캠 이미지를 BGR을 RGB로 변환 OpenCV(BGR) -> MediaPipe(RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지의 높이(h), 너비(w)를 나중에 좌표 계산을 위해 저장
    h, w, _ = image.shape
    
    # MediaPipe Pose 모델로 이미지 처리 (자세 감지 수행)
    results = pose.process(image)

    # 이미지를 다시 BGR로 변환 (OpenCV로 화면에 표시하기 위함)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- 4. 랜드마크 처리 (어깨만 사용) ---
    # 감지 결과(results)에 랜드마크가 존재하는지 확인
    if results.pose_landmarks:
        # 랜드마크 좌표(x, y, z, visibility)가 담긴 리스트
        landmarks = results.pose_landmarks.landmark

        # 좌/우 어깨 랜드마크 객체를 가져옴
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # 두 어깨가 모두 화면에 잘 보이는지 확인 (visibility > 0.5)
        if all(lm.visibility > 0.5 for lm in [left_shoulder, right_shoulder]):
            
            # 랜드마크의 정규화된 좌표(0.0~1.0)를 픽셀 좌표(0~w, 0~h)로 변환
            left_shoulder_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))

            # --- 4a. 기타 중심 위치 계산 ---
            # 양 어깨의 중심점(x, y) 계산
            shoulder_center_x = (left_shoulder_px[0] + right_shoulder_px[0]) // 2
            shoulder_center_y = (left_shoulder_px[1] + right_shoulder_px[1]) // 2

            # 양 어깨 사이의 거리(너비)를 유클리드 거리로 계산
            shoulder_width = int(math.dist(left_shoulder_px, right_shoulder_px))
            
            # 기타의 중심 X좌표: 어깨 중심에서 어깨너비의 15%만큼 오른쪽으로 이동
            guitar_center_x = shoulder_center_x + int(shoulder_width * 0.15) 
            
            # 기타의 중심 Y좌표: 어깨 중심에서 어깨너비의 65%만큼 아래로 이동
            guitar_center_y = shoulder_center_y + int(shoulder_width * 0.65)
         
            # --- 4b. 기타 크기 계산 ---
            # 기타의 너비: 어깨너비의 3배로 설정
            guitar_width = int(shoulder_width * 3.0)

            # --- 4c. 기타 각도 계산 ---
            # 양 어깨를 잇는 선의 기본 각도(degree)를 계산
            # math.atan2(y변화량, x변화량) -> 라디안 값 반환
            base_angle = math.degrees(math.atan2(right_shoulder_px[1] - left_shoulder_px[1], 
                                                 right_shoulder_px[0] - left_shoulder_px[0]))
            
          
            # 60도를 더해 기타의 기본 기울기를 설정
            final_angle = -base_angle + 60
          
            # --- 5. 기타 이미지 변형 (리사이즈 및 회전) ---
            
            # 원본 기타 이미지의 비율을 유지하면서 계산된 'guitar_width'에 맞게 리사이즈
            scale = guitar_width / guitar_img.shape[1] # 너비 비율
            resized_height = int(guitar_img.shape[0] * scale) # 높이도 동일 비율로 조절
            
            # 계산된 너비나 높이가 0 이하이면 (너무 작으면) 처리 중단
            if resized_height <= 0 or guitar_width <= 0:
                continue
                
            # OpenCV를 사용해 기타 이미지 리사이즈 (INTER_AREA는 축소 시 품질 좋음)
            resized_guitar = cv2.resize(guitar_img, (guitar_width, resized_height), interpolation=cv2.INTER_AREA)

            # 2D 회전 변환 행렬(M)을 계산
            # (중심점, 각도, 배율)
            M = cv2.getRotationMatrix2D((guitar_width / 2, resized_height / 2), final_angle, 1)
            
            # warpAffine 함수를 사용해 회전 변환 행렬 M을 이미지에 적용
            rotated_guitar = cv2.warpAffine(resized_guitar, M, (guitar_width, resized_height))

            # --- 6. 기타 이미지 오버레이 (알파 블렌딩) ---
            
            # 회전된 기타 이미지의 실제 높이(rh), 너비(rw)
            rh, rw, _ = rotated_guitar.shape

            # 기타 이미지를 배치할 원본 이미지(웹캠)의 좌표(ROI) 계산
            # (중심점을 기준으로 이미지 크기의 절반씩 빼고 더함)
            y1 = guitar_center_y - rh // 2
            y2 = y1 + rh
            x1 = guitar_center_x - rw // 2
            x2 = x1 + rw

            # 1. 최종 ROI(붙여넣을 위치)의 범위를 화면(h, w) 내로 제한 (Clamping)
            y1_c = max(0, y1)  # (c = clamped)
            y2_c = min(h, y2)
            x1_c = max(0, x1)
            x2_c = min(w, x2)

            # 2. 잘라낼 기타 이미지(overlay)의 범위 계산
            # (화면에서 잘린 만큼 기타 이미지도 동일하게 잘라줌)
            overlay_y1 = max(0, -y1) # y1이 -20이면 20 (즉, 20px 위에서부터 자름)
            overlay_y2 = overlay_y1 + (y2_c - y1_c) # 잘린 최종 높이만큼
            overlay_x1 = max(0, -x1)
            overlay_x2 = overlay_x1 + (x2_c - x1_c)

            # 3. 알파 채널 분리 및 슬라이싱
            # 회전된 기타 이미지에서 계산된 범위만큼 잘라내고, 4번째 채널(알파)을 가져옴
            alpha_s = rotated_guitar[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
            # 역 알파 채널 (배경용) (1.0 - 투명도)
            alpha_l = 1.0 - alpha_s

            # 4. 알파 블렌딩 적용
            # ROI 영역이 유효할 때만 (크기가 0이 아닐 때)
            if alpha_s.shape[0] > 0 and alpha_s.shape[1] > 0:
                for c in range(0, 3):  # B, G, R 3개 채널에 대해 반복
                    # 붙여넣을 이미지 = (알파 * 기타) + (역알파 * 원본배경)
                    image[y1_c:y2_c, x1_c:x2_c, c] = (alpha_s * rotated_guitar[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
                                                      alpha_l * image[y1_c:y2_c, x1_c:x2_c, c])
                    
            # +) 스트러밍 포인트 하이라이트
            if 'guitar_center_x' in locals() and 'guitar_center_y' in locals():
                # 하이라이트 직사각형 크기 (기타 전체 중 일부)
                # shoulder_width 기준으로 적당히 비율 조절
                highlight_w = int(shoulder_width * 0.9)   # 가로 폭
                highlight_h = int(shoulder_width * 0.4)   # 세로 폭 (좁게)
                
                # 기타 중심을 기준으로 하이라이트 위치 설정
                # 중심보다 약간 아래(몸통 부분)로 내림
                hl_center_x = guitar_center_x - int(shoulder_width * 0.25)
                hl_center_y = guitar_center_y + int(shoulder_width * 0.25)

                # 좌상단 / 우하단 좌표 계산
                hl_x1 = hl_center_x - highlight_w // 2
                hl_x2 = hl_center_x + highlight_w // 2
                hl_y1 = hl_center_y - highlight_h // 2
                hl_y2 = hl_center_y + highlight_h // 2

                # 화면 범위 안으로 제한
                hl_x1 = max(0, hl_x1); hl_x2 = min(w-1, hl_x2)
                hl_y1 = max(0, hl_y1); hl_y2 = min(h-1, hl_y2)

                # 반투명 스트러밍 지점 오버레이
                overlay = image.copy()
                cv2.rectangle(overlay, (hl_x1, hl_y1), (hl_x2, hl_y2), (0, 255, 0), -1)
                alpha = 0.3 
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                # 테두리 및 텍스트 표시
                cv2.rectangle(image, (hl_x1, hl_y1), (hl_x2, hl_y2), (0, 220, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "ACTIVE ZONE", (hl_x1+10, hl_y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            
            # --- 6b. 오른손 손끝 추적 및 속도 기반 소리 트리거 ---
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]

                if all(lm.visibility > 0.5 for lm in [right_wrist, right_index]):
                    # 픽셀 좌표로 변환
                    wrist_px = (int(right_wrist.x * w), int(right_wrist.y * h))
                    index_px = (int(right_index.x * w), int(right_index.y * h))

                    # 손끝 좌표 계산 (손목~검지 방향)
                    fingertip = (
                        int((wrist_px[0] + index_px[0]) / 2),
                        int((wrist_px[1] + index_px[1]) / 2)
                    )

                    # 속도 계산 (이전 프레임과 비교)
                    if 'prev_tip' not in locals():
                        prev_tip = fingertip
                        prev_time = time.time()
                        velocity = 0
                    else:
                        now = time.time()
                        dt = now - prev_time
                        dist_px = math.hypot(fingertip[0]-prev_tip[0], fingertip[1]-prev_tip[1])
                        velocity = dist_px / dt if dt > 0 else 0
                        prev_tip = fingertip
                        prev_time = now

                    # 손끝 표시
                    cv2.circle(image, fingertip, 6, (0, 255, 255), -1)
                    cv2.putText(image, f"speed:{velocity:.0f}", (fingertip[0]+10, fingertip[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                    # === rule based: 특정 영역 + 특정 속도 ===
                    # (예: 기타 중심 기준 직사각형 안쪽)
                    if 'guitar_center_x' in locals() and 'guitar_center_y' in locals():
                        rect_half_w = shoulder_width * 1.2
                        rect_half_h = shoulder_width * 0.4
                        if (abs(fingertip[0] - guitar_center_x) < rect_half_w and
                            abs(fingertip[1] - guitar_center_y) < rect_half_h):
                            if velocity > 900:  # px/s 임계값, 환경에 따라 조정
                                cv2.putText(image, "STRUM!", (guitar_center_x-40, guitar_center_y-60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                                # 소리 재생 대신 텍스트
                                cv2.putText(image, "STRUM!", (hl_x1 + 20, hl_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

    # --- 7. 결과 표시 및 종료 ---
    # 'Virtual Guitar'라는 이름의 창에 최종 이미지를 표시
    cv2.imshow('Virtual Guitar', image)
    
    # 5ms 동안 키 입력을 기다림
    # ESC 키(27)가 눌리면 루프 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- 8. 자원 해제 ---
# 웹캠 사용 해제
cap.release()
out.release()  # 저장
# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()