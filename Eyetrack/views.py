from django.shortcuts import render
from django.http import JsonResponse
from .main import GazeTrackingSession
from .models import GazeTrackingResult
import cv2
import pandas as pd
import base64
import io
from PIL import Image
import numpy as np

# 전역 변수 선언
gaze_session = GazeTrackingSession()

def start_gaze_tracking_view(request):
    gaze_session.start_eye_tracking()  # 시선 추적 시작
    return JsonResponse({"message": "Gaze tracking started"}, status=200)

def apply_gradient(center, radius, color, image, text=None):
    overlay = image.copy()
    cv2.circle(overlay, center, radius, color, -1)  # 원
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    if text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness)

def assign_colors_and_numbers(section_counts):
    colors = [
        (38, 38, 255), (59, 94, 255), (55, 134, 255),
        (51, 173, 255), (26, 210, 255), (0, 255, 255)
    ]
    sorted_sections = sorted(section_counts.items(), key=lambda item: item[1], reverse=True)
    color_map = {}
    number_map = {}
    for i, (section, _) in enumerate(sorted_sections):
        color_map[section] = colors[i % len(colors)]
        number_map[section] = str(i + 1)
    return color_map, number_map

def get_feedback(section_counts):
    max_section = max(section_counts, key=section_counts.get)
    feedback = ""
    if max_section in ['B', 'E']:
        feedback = "면접관을 잘 응시하고 있습니다!"
    elif max_section in ['A', 'D']:
        feedback = "면접관의 왼쪽을 많이 응시합니다. 면접관을 응시하려고 노력해보세요."
    elif max_section in ['C', 'F']:
        feedback = "면접관의 오른쪽을 많이 응시합니다. 면접관을 응시하려고 노력해보세요."
    return feedback

def draw_heatmap(image, section_counts):
    if image is not None:
        height, width, _ = image.shape
        section_centers = {
            "A": (int(width / 6), int(height / 4)),
            "B": (int(width / 2), int(height / 4)),
            "C": (int(5 * width / 6), int(height / 4)),
            "D": (int(width / 6), int(3 * height / 4)),
            "E": (int(width / 2), int(3 * height / 4)),
            "F": (int(5 * width / 6), int(3 * height / 4))
        }

        color_map, number_map = assign_colors_and_numbers(section_counts)
        for section, count in section_counts.items():
            if count > 0 and section in section_centers:
                center = section_centers[section]
                color = color_map[section]
                number = number_map[section]
                radius = 100  # 모든 원의 반지름을 일정하게 설정
                apply_gradient(center, radius, color, image, number)

def stop_gaze_tracking_view(request):
    csv_filename = gaze_session.stop_eye_tracking()  # 섹션 및 횟수를 저장하고 시선 추적 종료
    section_data = pd.read_csv(csv_filename)
    section_counts = dict(zip(section_data["Section"], section_data["Count"]))

    image_path = "C:/KJE/IME_graduation/Back_AI_connect-main/Eyetrack/0518/image.png"
    original_image = cv2.imread(image_path)  # 이미지 로드

    if original_image is None:
        return JsonResponse({"message": "Image not found"}, status=404)

    # heatmap 그리기
    heatmap_image = original_image.copy()
    draw_heatmap(heatmap_image, section_counts)

    # 이미지를 base64로 인코딩하여 문자열로 변환
    _, buffer = cv2.imencode('.png', heatmap_image)
    encoded_image_string = base64.b64encode(buffer).decode('utf-8')

    # 피드백 생성
    feedback = get_feedback(section_counts)

    # 이미지를 PIL 형식으로 변환하여 화면에 표시
    heatmap_pil_image = Image.fromarray(cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB))
    heatmap_pil_image.show()

    # GazeTrackingResult 모델에 이미지 데이터 저장
    gaze_tracking_result = GazeTrackingResult.objects.create(
        encoded_image=encoded_image_string,
        feedback=feedback  # 피드백 저장
    )

    return JsonResponse({
        "message": "Gaze tracking stopped",
        "image_data": gaze_tracking_result.encoded_image,
        "feedback": feedback
    }, status=200)
