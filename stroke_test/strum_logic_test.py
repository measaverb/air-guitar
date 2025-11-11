import cv2
import mediapipe as mp
import time
import pygame
import collections
import math

WINDOW_NAME = 'Air Guitar – Detect + Center Needed Distances (Trackbars)'
SOUND_FILE  = 'strum.wav'

# --- 트랙바 초기값 ---
INITIAL_SENS_ZX100 = 20    # z-민감도(배수*100). 낮을수록 민감(작은 속도도 반응)
INITIAL_PINCH_PX   = 45    # 핀치 ON 기준(px). 엄지-검지 거리가 이 값 이하일 때만 감지
INITIAL_UPBIAS_PCT = 30    # Up(위로) 방향 z-임계 완화 비율(%)
INITIAL_NEED_DOWN  = 36    # Down 최소 y 변위(px) 시작값(현장 튜닝 권장)
INITIAL_NEED_UP    = 50    # Up   최소 y 변위(px) 시작값(현장 튜닝 권장)

# --- 사운드 초기화 ---
pygame.init()
pygame.mixer.init()
try:
    strum_sound = pygame.mixer.Sound(SOUND_FILE)
except pygame.error:
    print(f"오류: '{SOUND_FILE}' 파일을 찾을 수 없습니다.")
    raise SystemExit

def play_strum():
    try:
        strum_sound.play()
    except:
        pass

# --- MediaPipe Hands: 오른손만 추적 ---
mp_solutions = mp.solutions
mp_hands = mp_solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)
mp_drawing = mp_solutions.drawing_utils

def get_right_hand(results):
    """멀티핸드 결과에서 '오른손' 랜드마크만 반환. 없으면 None"""
    if not results or (not results.multi_hand_landmarks) or (not results.multi_handedness):
        return None
    for i, handed in enumerate(results.multi_handedness):
        if handed.classification[0].label.lower() == 'right':
            return results.multi_hand_landmarks[i]
    return None

# --- 카메라 ---
cap = cv2.VideoCapture(0)

# --- 스무딩/감지 파라미터 ---
EMA_ALPHA_POS       = 0.35   # y좌표(검지 끝) EMA
EMA_ALPHA_SPEED     = 0.25   # |vy| 스케일(적응형 z 분모) EMA
SAME_DIR_COOLDOWN   = 0.22   # 같은 방향 연속 확정 최소 간격(초)
CONFIRM_WIN_MS      = 160    # 2-Stage 확정 윈도우(ms) (100→160으로 완화)
VERT_RATIO_MIN      = 0.60   # 수직성 검사: |vy| / sqrt(vx^2+vy^2) ≥ 0.60
DOWN_GUARD_MS       = 120    # Down 직후 Up 강화 시간(ms)
UP_STRONG_MULT      = 1.4    # 보호시간 내 Up z-문턱 가중치 배수

# --- 진폭 비율 보정(적응형 필요변위) ---
AMP_RATIO_UP        = 0.90   # Up 필요변위 ≥ 0.90 * 최근 Down 진폭 EMA
AMP_RATIO_DOWN      = 0.60   # Down 필요변위 ≥ 0.60 * 최근 Up   진폭 EMA
EMA_AMP_ALPHA       = 0.25   # 진폭 EMA 업데이트 비율

# --- 내부 상태 ---
state            = 'idle'
smoothed_y       = None          # EMA y
prev_y           = None
prev_time_y      = None          # y/z 계산용 시간
prev_time_xy     = None          # vx,vy 계산용 시간 (분리 중요!)
last_dir_time    = {'down': 0.0, 'up': 0.0}
recent_dirs      = collections.deque(maxlen=6)  # +1:down, -1:up
pinch_active     = False
ema_down_amp     = 20.0          # 최근 Down 진폭 EMA
ema_up_amp       = 20.0          # 최근 Up   진폭 EMA
pending_up       = None          # {'t0','y0','neg_cnt','disp'}
pending_down     = None          # {'t0','y0','pos_cnt','disp'}

# FPS
t0 = time.time()
frames = 0
fps_text = "FPS: --"

# --- 트랙바 콜백 ---
def on_trackbar(_): pass

# --- UI(트랙바) ---
cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar('Sensitivity(zx100)', WINDOW_NAME, INITIAL_SENS_ZX100, 100, on_trackbar)  # 0~100 → 0.5~2.0 매핑
cv2.createTrackbar('UpBias(%)',  WINDOW_NAME, INITIAL_UPBIAS_PCT, 60,  on_trackbar)  # Up z-임계 완화
cv2.createTrackbar('PinchPx', WINDOW_NAME, INITIAL_PINCH_PX,   200, on_trackbar)  # 핀치 ON 문턱(px)
cv2.createTrackbar('NeedDown(px)', WINDOW_NAME, INITIAL_NEED_DOWN,  200, on_trackbar)  # Down 최소 변위(px)
cv2.createTrackbar('NeedUp(px)', WINDOW_NAME, INITIAL_NEED_UP,    200, on_trackbar)  # Up   최소 변위(px)
cv2.createTrackbar('Adaptive(0/1)',  WINDOW_NAME, 1,  1,   on_trackbar)  # 1=적응형 보정 켜기
cv2.createTrackbar('DispRaw(0/1)', WINDOW_NAME, 1, 1,   on_trackbar)  # 1=raw y로 변위 계산

def compute_z_thresholds():
    """
    z-기반 1차 트리거 임계 계산.
    Sensitivity: 전체 민감도 배수(낮을수록 민감).
    UpBias(%): Up 방향은 기본적으로 덜 가중하도록 완화(낮은 임계 허용).
    """
    raw = cv2.getTrackbarPos('Sensitivity(zx100)', WINDOW_NAME)  # 0~100
    mult = 0.5 + (raw / 100.0) * 1.5                              # 0.5~2.0
    up_bias_pct = cv2.getTrackbarPos('UpBias(%)', WINDOW_NAME)    # 0~60
    up_relax    = 1.0 - (up_bias_pct / 100.0)                     # 1.0(완화X)~0.4(최대 완화)
    th_down_z   = mult
    th_up_z     = max(0.35, mult * up_relax)                      # Up만 완화
    return th_down_z, th_up_z

def draw_needed_center(image, need_down, need_up):
    """
    화면 중앙에 Down/Up '필요 y거리(px)'를 크게 시각화.
    - Down: 중앙 기준 아래로 초록 화살표
    - Up  : 중앙 기준 위로 하늘색 화살표
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    # Down: 중앙→아래 need_down
    y_down = min(h-1, cy + int(need_down))
    cv2.arrowedLine(image, (cx - 25, cy), (cx - 25, y_down), (0, 200, 0), 4, tipLength=0.15)
    cv2.putText(image, f'Down need: {int(need_down)}px',
                (cx - 160, cy + int(need_down) + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)

    # Up: 중앙→위 need_up
    y_up = max(0, cy - int(need_up))
    cv2.arrowedLine(image, (cx + 25, cy), (cx + 25, y_up), (0, 128, 255), 4, tipLength=0.15)
    cv2.putText(image, f'Up need: {int(need_up)}px',
                (cx - 40, max(24, cy - int(need_up) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)

while cap.isOpened():
    ok, image = cap.read()
    if not ok:
        break

    # 미러링(연주 직관성↑)
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # --- 트랙바 읽기 ---
    sens_raw   = cv2.getTrackbarPos('Sensitivity(zx100)', WINDOW_NAME)
    upbias     = cv2.getTrackbarPos('UpBias(%)', WINDOW_NAME)
    pinch_px   = cv2.getTrackbarPos('PinchPx', WINDOW_NAME)
    base_need_down = max(4, cv2.getTrackbarPos('NeedDown(px)', WINDOW_NAME))
    base_need_up   = max(4, cv2.getTrackbarPos('NeedUp(px)', WINDOW_NAME))
    use_adapt  = (cv2.getTrackbarPos('Adaptive(0/1)', WINDOW_NAME) == 1)
    use_rawdisp= (cv2.getTrackbarPos('DispRaw(0/1)', WINDOW_NAME) == 1)

    pinch_on   = max(5, pinch_px)      # ON 문턱
    pinch_off  = pinch_on + 10         # OFF 문턱(히스테리시스)
    th_down_z, th_up_z = compute_z_thresholds()

    # Down 우세 시 Up z-임계 강화 (D-D 사이 미세 U 억제)
    if recent_dirs:
        down_ratio = sum(1 for d in recent_dirs if d == +1) / len(recent_dirs)
        th_up_z *= (1.0 + 0.8 * down_ratio)

    # Down 직후 보호시간 동안 Up 더 강화
    elapsed_since_down = time.time() - last_dir_time['down']
    if elapsed_since_down * 1000.0 < DOWN_GUARD_MS:
        th_up_z *= UP_STRONG_MULT

    # --- Hands 추론 ---
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results   = hands.process(image_rgb)
    hand      = get_right_hand(results)

    stroke_text = ""
    current_time = time.time()

    pinch_dist = None
    ix = iy = tx = ty = None

    if hand:
        # 검지/엄지 좌표(px)
        tip_i = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        tip_t = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        ix, iy = int(tip_i.x * w), int(tip_i.y * h)
        tx, ty = int(tip_t.x * w), int(tip_t.y * h)

        # 핀치 거리 → 게이트
        pinch_dist = math.hypot(ix - tx, iy - ty)
        if pinch_active:
            if pinch_dist >= pinch_off:
                pinch_active = False
        else:
            if pinch_dist <= pinch_on:
                pinch_active = True

        # --- y 위치/속도(EMA) : 시간 스탬프 분리/클램프 ---
        tip_y_px = iy
        if smoothed_y is None:
            smoothed_y = tip_y_px
            prev_time_y = current_time
            velocity_y  = 0.0
            speed_ema   = 1.0
        else:
            dt_y = max(1/120.0, current_time - prev_time_y)  # 최소 1/120초
            smoothed_y = EMA_ALPHA_POS * tip_y_px + (1.0 - EMA_ALPHA_POS) * smoothed_y
            velocity_y = (smoothed_y - prev_y) / dt_y if (prev_y is not None) else 0.0
            prev_time_y = current_time
            abs_v      = abs(velocity_y)
            speed_ema  = EMA_ALPHA_SPEED * max(1.0, abs_v) + \
                         (1.0 - EMA_ALPHA_SPEED) * (speed_ema if 'speed_ema' in locals() else max(1.0, abs_v))
        prev_y = smoothed_y

        # --- 수직성 검사용 vx, vy : raw 좌표 & 별도 시간 ---
        if 'prev_ix' not in locals():
            prev_ix, prev_iy = ix, iy
            prev_time_xy = current_time

        dt2 = max(1/120.0, current_time - (prev_time_xy if prev_time_xy else current_time))
        vx = (ix - prev_ix) / dt2
        vy = (iy - prev_iy) / dt2
        prev_ix, prev_iy = ix, iy
        prev_time_xy = current_time

        v_mag  = math.hypot(vx, vy) + 1e-6
        vert_ratio = abs(vy) / v_mag
        z = velocity_y / max(1.0, speed_ema)

        # --- 필요 y거리(px): 베이스 vs 적응형(최근 반대방향 진폭 EMA) 중 더 큰 값 ---
        if use_adapt:
            eff_need_up = max(base_need_up,   AMP_RATIO_UP   * ema_down_amp)
            eff_need_down = max(base_need_down, AMP_RATIO_DOWN * ema_up_amp)
        else:
            eff_need_up   = base_need_up
            eff_need_down = base_need_down

        # --- 중앙 오버레이(가이드) ---
        draw_needed_center(image, need_down=eff_need_down, need_up=eff_need_up)

        # --- 2-Stage 판정: 핀치 ON일 때만 ---
        if pinch_active:
            # 변위 기준(y 선택: raw 또는 smoothed)
            current_y_for_disp = (iy if use_rawdisp else smoothed_y)

            # Down 프리트리거/확정
            if pending_down is None:
                if (z > th_down_z) and (vert_ratio >= VERT_RATIO_MIN):
                    pending_down = {'t0': current_time, 'y0': current_y_for_disp,
                                    'pos_cnt': 1, 'disp': 0.0}
            else:
                dt_win_ms = (current_time - pending_down['t0']) * 1000.0
                if z > 0:  # 아래 방향 프레임 카운트
                    pending_down['pos_cnt'] += 1
                # 누적 아래 변위: 현재 - 시작
                pending_down['disp'] = max(0.0, current_y_for_disp - pending_down['y0'])

                if (dt_win_ms <= CONFIRM_WIN_MS and
                    pending_down['pos_cnt'] >= 2 and
                    pending_down['disp'] >= eff_need_down and
                    vert_ratio >= VERT_RATIO_MIN and
                    (current_time - last_dir_time['down']) > SAME_DIR_COOLDOWN):
                    # Down 확정
                    state = 'down'
                    stroke_text = "Downstroke!"
                    play_strum()
                    last_dir_time['down'] = current_time
                    recent_dirs.append(+1)
                    # Down 진폭 EMA 업데이트(Up 필요거리 보정에 사용)
                    ema_down_amp = EMA_AMP_ALPHA * pending_down['disp'] + (1 - EMA_AMP_ALPHA) * ema_down_amp
                    pending_down = None
                elif dt_win_ms > CONFIRM_WIN_MS:
                    pending_down = None

            # Up 프리트리거/확정
            if pending_up is None:
                if (-z > th_up_z) and (vert_ratio >= VERT_RATIO_MIN):
                    pending_up = {'t0': current_time, 'y0': current_y_for_disp,
                                  'neg_cnt': 1, 'disp': 0.0}
            else:
                dt_win_ms = (current_time - pending_up['t0']) * 1000.0
                if z < 0:  # 위 방향 프레임 카운트
                    pending_up['neg_cnt'] += 1
                # 누적 위 변위: 시작 - 현재
                pending_up['disp'] = max(0.0, pending_up['y0'] - current_y_for_disp)

                if (dt_win_ms <= CONFIRM_WIN_MS and
                    pending_up['neg_cnt'] >= 3 and
                    pending_up['disp'] >= eff_need_up and
                    vert_ratio >= VERT_RATIO_MIN and
                    (current_time - last_dir_time['up']) > SAME_DIR_COOLDOWN):
                    # Up 확정
                    state = 'up'
                    stroke_text = "Upstroke!"
                    play_strum()
                    last_dir_time['up'] = current_time
                    recent_dirs.append(-1)
                    # Up 진폭 EMA 업데이트(Down 필요거리 보정에 사용)
                    ema_up_amp = EMA_AMP_ALPHA * pending_up['disp'] + (1 - EMA_AMP_ALPHA) * ema_up_amp
                    pending_up = None
                elif dt_win_ms > CONFIRM_WIN_MS:
                    pending_up = None

        else:
            # 핀치 OFF → 엔진 리셋
            state = 'idle'
            pending_up   = None
            pending_down = None

        # --- 시각화: 핀치/랜드마크/디버그 ---
        if pinch_dist is not None:
            cv2.line(image, (ix, iy), (tx, ty), (255, 255, 0), 2)
            cv2.circle(image, (ix, iy), 5, (0, 255, 255), -1)
            cv2.circle(image, (tx, ty), 5, (0, 255, 255), -1)
            midx, midy = (ix + tx) // 2, (iy + ty) // 2
            cv2.putText(image, f'{pinch_dist:.0f}px', (midx + 8, midy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f'PINCH: {"ON" if pinch_active else "OFF"}',
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 220, 255) if pinch_active else (128,128,128), 2)

        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

        if stroke_text:
            color = (0,255,0) if stroke_text == "Downstroke!" else (0,0,255)
            cv2.putText(image, stroke_text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5, cv2.LINE_AA)

        # 디버그: z 및 임계값
        cv2.putText(image, f'z:{z:+.2f}  zTHd:{th_down_z:.2f}  zTHu:{th_up_z:.2f}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)

    else:
        # 손 미검출 → 상태 리셋
        prev_y = None
        smoothed_y = None
        prev_time_y = None
        prev_time_xy = None
        pending_up = None
        pending_down = None
        pinch_active = False
        recent_dirs.clear()
        speed_ema = None

    # 상단 HUD: 베이스/실제 필요거리 동시 표시
    eff_need_down = (max(base_need_down, AMP_RATIO_DOWN * ema_up_amp) if use_adapt else base_need_down)
    eff_need_up   = (max(base_need_up,   AMP_RATIO_UP   * ema_down_amp) if use_adapt else base_need_up)
    hud_text = (f'Sens:{sens_raw}  UpBias:{upbias}%  '
                f'NeedD:{base_need_down}px({int(eff_need_down)} eff)  '
                f'NeedU:{base_need_up}px({int(eff_need_up)} eff)  '
                f'Adapt:{1 if use_adapt else 0}  DispRaw:{1 if use_rawdisp else 0}')
    cv2.putText(image, hud_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

    # FPS
    frames += 1
    if frames % 30 == 0:
        dt_all = time.time() - t0
        fps = frames / dt_all if dt_all > 0 else 0.0
        fps_text = f'FPS: {fps:.1f}'
    cv2.putText(image, fps_text, (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # 표시/종료
    cv2.imshow(WINDOW_NAME, image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
pygame.quit()