import os
import time

import cv2
import numpy as np
import mediapipe as mp


# =========================
# 모델 파일 설정
# =========================
# face_landmarker.task 파일을 이 코드와 같은 폴더에 넣어야 합니다.
# 다운로드 주소:
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")


def check_model_file():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "\nface_landmarker.task 파일이 없습니다.\n"
            "아래 폴더에 face_landmarker.task 파일을 넣어주세요.\n\n"
            f"{BASE_DIR}\n"
        )


# =========================
# Blendshape 점수 가져오기
# =========================

def get_score(blendshape_dict, name):
    return blendshape_dict.get(name, 0.0)


# =========================
# Blendshape 기반 감정 점수 계산
# =========================

def estimate_emotions(blendshape_dict):
    """
    MediaPipe Face Landmarker는 angry, sad 같은 감정을 직접 주는 것이 아니라
    browDownLeft, jawOpen, mouthSmileLeft 같은 표정 계수(blendshape)를 줍니다.
    여기서는 여러 blendshape 값을 조합해서 감정 점수처럼 계산합니다.
    """

    # =========================
    # 주요 blendshape 값 추출
    # =========================

    brow_down = max(
        get_score(blendshape_dict, "browDownLeft"),
        get_score(blendshape_dict, "browDownRight")
    )

    eye_squint = max(
        get_score(blendshape_dict, "eyeSquintLeft"),
        get_score(blendshape_dict, "eyeSquintRight")
    )

    mouth_press = max(
        get_score(blendshape_dict, "mouthPressLeft"),
        get_score(blendshape_dict, "mouthPressRight")
    )

    nose_sneer = max(
        get_score(blendshape_dict, "noseSneerLeft"),
        get_score(blendshape_dict, "noseSneerRight")
    )

    brow_inner_up = get_score(blendshape_dict, "browInnerUp")

    jaw_open = get_score(blendshape_dict, "jawOpen")

    eye_wide = max(
        get_score(blendshape_dict, "eyeWideLeft"),
        get_score(blendshape_dict, "eyeWideRight")
    )

    mouth_smile = max(
        get_score(blendshape_dict, "mouthSmileLeft"),
        get_score(blendshape_dict, "mouthSmileRight")
    )

    cheek_squint = max(
        get_score(blendshape_dict, "cheekSquintLeft"),
        get_score(blendshape_dict, "cheekSquintRight")
    )

    mouth_frown = max(
        get_score(blendshape_dict, "mouthFrownLeft"),
        get_score(blendshape_dict, "mouthFrownRight")
    )

    mouth_pucker = get_score(blendshape_dict, "mouthPucker")
    mouth_shrug_upper = get_score(blendshape_dict, "mouthShrugUpper")

    # =========================
    # 감정 점수 계산
    # =========================

    # 웃음
    happy = (
        mouth_smile * 0.65
        + cheek_squint * 0.35
    )

    # 놀람
    surprise = (
        jaw_open * 0.45
        + eye_wide * 0.35
        + brow_inner_up * 0.20
    )

    # 화남 / 찡그림
    # 기존 평균 방식보다 화남 관련 특징을 더 강하게 반영
    angry = (
        brow_down * 0.45
        + eye_squint * 0.25
        + mouth_press * 0.20
        + nose_sneer * 0.10
    )

    # browInnerUp이 크면 화남보다는 슬픔/놀람 쪽이므로 약간 감점
    angry = angry - brow_inner_up * 0.15

    # 화남 점수 증폭
    angry = angry * 3.0

    # 슬픔
    sad = (
        mouth_frown * 0.45
        + brow_inner_up * 0.35
        + mouth_pucker * 0.20
    )

    # 혐오
    disgust = (
        nose_sneer * 0.55
        + mouth_shrug_upper * 0.30
        + eye_squint * 0.15
    )

    # 두려움
    fear = (
        eye_wide * 0.35
        + brow_inner_up * 0.30
        + jaw_open * 0.35
    )

    emotion_scores = {
        "angry": angry,
        "disgust": disgust,
        "fear": fear,
        "happy": happy,
        "sad": sad,
        "surprise": surprise,
    }

    # 0~1 범위로 제한
    for key in emotion_scores:
        emotion_scores[key] = float(np.clip(emotion_scores[key], 0.0, 1.0))

    max_emotion_score = max(emotion_scores.values())

    # neutral이 너무 쉽게 이기지 않도록 조정
    neutral = max(0.0, 1.0 - max_emotion_score * 3.0)
    emotion_scores["neutral"] = float(np.clip(neutral, 0.0, 1.0))

    return emotion_scores


# =========================
# AI 모니터링 화면 그리기
# =========================

def draw_ai_view(face_landmarks, emotion_scores, source_width, source_height, width=720, height=480):
    """
    AI 모니터링 화면을 생성합니다.
    얼굴 bounding box 기준으로 가로세로 비율을 유지해서 랜드마크를 그립니다.
    """

    ai_view = np.zeros((height, width, 3), dtype=np.uint8)

    # =========================
    # 얼굴 랜드마크 점 표시
    # =========================
    if face_landmarks is not None:
        points = []

        for lm in face_landmarks:
            x = lm.x * source_width
            y = lm.y * source_height
            points.append([x, y])

        points = np.array(points)

        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

        face_w = max_x - min_x
        face_h = max_y - min_y

        if face_w > 0 and face_h > 0:
            # =========================
            # 얼굴 표시 영역 조절 부분
            # =========================
            face_area_x = 80   # 얼굴 랜드마크 표시 시작 x 위치
            face_area_y = 70   # 얼굴 랜드마크 표시 시작 y 위치
            face_area_w = 280  # 추출 랜드마크 표시 영역 가로 크기
            face_area_h = 320  # 추출 랜드마크 표시 영역 세로 크기

            scale = min(face_area_w / face_w, face_area_h / face_h)

            scaled_w = face_w * scale
            scaled_h = face_h * scale

            offset_x = face_area_x + (face_area_w - scaled_w) / 2
            offset_y = face_area_y + (face_area_h - scaled_h) / 2

            for x, y in points:
                draw_x = int((x - min_x) * scale + offset_x)
                draw_y = int((y - min_y) * scale + offset_y)

                if 0 <= draw_x < width and 0 <= draw_y < height:
                    cv2.circle(ai_view, (draw_x, draw_y), 2, (0, 255, 0), -1)

    # =========================
    # 감정 막대 표시
    # =========================

    emotion_order = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    start_x = 470
    start_y = 70
    bar_w = 130
    gap = 32

    for i, emotion in enumerate(emotion_order):
        score = emotion_scores.get(emotion, 0.0)
        y = start_y + i * gap

        cv2.putText(
            ai_view,
            f"{emotion}: {score:.2f}",
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
        )

        # 막대 테두리
        cv2.rectangle(
            ai_view,
            (start_x + 120, y - 14),
            (start_x + 120 + bar_w, y),
            (80, 80, 80),
            1,
        )

        # 막대 내부
        filled_w = int(bar_w * score)

        cv2.rectangle(
            ai_view,
            (start_x + 120, y - 14),
            (start_x + 120 + filled_w, y),
            (0, 255, 0),
            -1,
        )

    return ai_view


# =========================
# 메인 실행
# =========================

def main():
    check_model_file()

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture("test_video.mp4")

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    start_time = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            # 거울 모드
            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            # OpenCV BGR → MediaPipe RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            emotion_scores = {
                "angry": 0.0,
                "disgust": 0.0,
                "fear": 0.0,
                "happy": 0.0,
                "sad": 0.0,
                "surprise": 0.0,
                "neutral": 1.0,
            }

            face_landmarks = None
            dominant_emotion = "neutral"
            dominant_score = 1.0

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]

                # 실제 카메라 화면에도 얼굴 랜드마크 표시
                for lm in face_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # blendshape 가져오기
                if result.face_blendshapes:
                    blendshapes = result.face_blendshapes[0]

                    blendshape_dict = {
                        category.category_name: category.score
                        for category in blendshapes
                    }

                    emotion_scores = estimate_emotions(blendshape_dict)

                    dominant_emotion = max(
                        emotion_scores,
                        key=emotion_scores.get,
                    )
                    dominant_score = emotion_scores[dominant_emotion]

            else:
                dominant_emotion = "no face"
                dominant_score = 0.0

            # 사람 화면에 감정 표시
            display_text = f"{dominant_emotion.upper()} : {dominant_score * 100:.1f}%"

            cv2.putText(
                frame,
                display_text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

            # AI 모니터링 화면 생성
            ai_view = draw_ai_view(face_landmarks, emotion_scores, w, h)

            cv2.imshow("Human View - Emotion Output", frame)
            cv2.imshow("AI Monitoring View - Facial Mesh & Emotion Bars", ai_view)

            # q 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()