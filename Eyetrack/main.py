import cv2
import numpy as np
import csv
import time
import threading
import requests
from .gaze_tracking import GazeTracking

class GazeTrackingSession:
    def __init__(self):
        self.sections = {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": 0,
            "E": 0,
            "F": 0
        }
        self.section = "None"
        self.thread = None
        self.running = False

    def Section(self, where):
        if where in self.sections:
            self.sections[where] += 1
            return self.sections[where]

    def Thread_run(self):
        if not self.running:
            return
        print(self.section, ":", self.Section(self.section))
        self.thread = threading.Timer(0.2, self.Thread_run)  # 0.1초 단위 기록
        self.thread.daemon = True
        self.thread.start()

    def start_eye_tracking(self):
        self.running = True
        self.thread = self.Thread_run()

        avg_left_hor_gaze = 0
        avg_right_hor_gaze = 0
        avg_top_ver_gaze = 0
        avg_bottom_ver_gaze = 0

        total_left_hor_gaze = 0
        total_right_hor_gaze = 0
        total_top_ver_gaze = 0
        total_bottom_ver_gaze = 0

        webcam = cv2.VideoCapture(0)
        test_count = 1
        flag = 0
        gaze = GazeTracking()

        while self.running:
            key = cv2.waitKey(1)
            if key == 27:  # esc 눌러서 저장하고 종료
                self.stop_eye_tracking()
                break
            _, frame = webcam.read()
            gaze.refresh(frame)
            frame, loc1, loc2 = gaze.annotated_frame()

            text = ""
            if test_count < 50:
                cv2.circle(frame, (25, 25), 25, (0, 0, 255), -1)
                if gaze.horizontal_ratio() is not None and gaze.vertical_ratio() is not None:
                    total_left_hor_gaze += gaze.horizontal_ratio()
                    total_top_ver_gaze += gaze.vertical_ratio()
                    test_count += 1
            elif 50 <= test_count < 100:
                cv2.circle(frame, (610, 25), 25, (0, 0, 255), -1)
                if gaze.horizontal_ratio() is not None and gaze.vertical_ratio() is not None:
                    total_right_hor_gaze += gaze.horizontal_ratio()
                    total_top_ver_gaze += gaze.vertical_ratio()
                    test_count += 1
            elif 100 <= test_count < 150:
                cv2.circle(frame, (25, 450), 25, (0, 0, 255), -1)
                if gaze.horizontal_ratio() is not None and gaze.vertical_ratio() is not None:
                    total_left_hor_gaze += gaze.horizontal_ratio()
                    total_bottom_ver_gaze += gaze.vertical_ratio()
                    test_count += 1
            elif 150 <= test_count < 200:
                cv2.circle(frame, (610, 450), 25, (0, 0, 255), -1)
                if gaze.horizontal_ratio() is not None and gaze.vertical_ratio() is not None:
                    total_right_hor_gaze += gaze.horizontal_ratio()
                    total_bottom_ver_gaze += gaze.vertical_ratio()
                    test_count += 1

            else:
                if flag == 0:
                    avg_left_hor_gaze = total_left_hor_gaze / 100
                    avg_right_hor_gaze = total_right_hor_gaze / 100
                    avg_top_ver_gaze = total_top_ver_gaze / 100
                    avg_bottom_ver_gaze = total_bottom_ver_gaze / 100
                    flag = 1

                if gaze.is_blinking():
                    text = "Blinking"

                if gaze.is_top_left(avg_left_hor_gaze, avg_top_ver_gaze):
                    cv2.putText(frame, "Top Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    text = "Looking top left"
                    self.section = "A"
                elif gaze.is_top_center(avg_top_ver_gaze, avg_right_hor_gaze, avg_left_hor_gaze):
                    cv2.putText(frame, "Top Center", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    text = "Looking top center"
                    self.section = "B"
                elif gaze.is_top_right(avg_right_hor_gaze, avg_top_ver_gaze):
                    cv2.putText(frame, "Top Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    text = "Looking top right"
                    self.section = "C"
                elif gaze.is_bottom_left(avg_left_hor_gaze, avg_top_ver_gaze):
                    cv2.putText(frame, "Bottom Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    text = "Looking bottom left"
                    self.section = "D"
                elif gaze.is_bottom_center(avg_top_ver_gaze, avg_right_hor_gaze, avg_left_hor_gaze):
                    cv2.putText(frame, "Bottom Center", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    text = "Looking bottom center"
                    self.section = "E"
                elif gaze.is_bottom_right(avg_right_hor_gaze, avg_top_ver_gaze):
                    cv2.putText(frame, "Bottom Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    text = "Looking bottom right"
                    self.section = "F"
                gaze_time = int(time.time())
                save_loc1 = loc1
                save_loc2 = loc2

            cv2.imshow("Frame", frame)

        cv2.destroyAllWindows()

    def stop_eye_tracking(self):
        self.running = False
        if self.thread is not None:
            self.thread.cancel()

        csv_filename = "C:/KJE/IME_graduation/Back_AI_connect-main/Eyetrack/0518/gaze_sections.csv"

        # CSV 파일 헤더
        csv_header = ["Section", "Count"]

        # CSV 파일 쓰기 모드로 열기
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # 헤더 쓰기
            writer.writerow(csv_header)

            # 각 섹션의 횟수를 CSV 파일에 기록
            for section_name, count in self.sections.items():
                writer.writerow([section_name, count])

        print("Data saved to", csv_filename)

        # # sections 변수 초기화
        # self.sections = {
        #     "A": 0,
        #     "B": 0,
        #     "C": 0,
        #     "D": 0,
        #     "E": 0,
        #     "F": 0
        # }
        # self.section = "None"
        return csv_filename

# 이하의 코드는 프로그램 실행을 위한 메인 부분입니다.
# if __name__ == "__main__":
#     session = GazeTrackingSession()
#     session.start_eye_tracking()
