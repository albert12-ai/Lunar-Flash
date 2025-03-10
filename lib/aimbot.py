import ctypes
import cv2
import json
import math
import mss
import os
import sys
import time
import torch
import numpy as np
import win32api
from termcolor import colored
from ultralytics import YOLO

torch.backends.cudnn.benchmark = True

screensize = {'X': ctypes.windll.user32.GetSystemMetrics(0), 'Y': ctypes.windll.user32.GetSystemMetrics(1)}
screen_res_x = screensize['X']
screen_res_y = screensize['Y']

screen_x = int(screen_res_x / 2)
screen_y = int(screen_res_y / 2)

aim_height = 2
confidence = 0.45
use_trigger_bot = False

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class Aimbot:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    screen = mss.mss()
    
    with open("lib/config/config.json") as f:
        sens_config = json.load(f)
    aimbot_status = colored("ENABLED", 'green')

    def __init__(self, box_constant=320, collect_data=False, virtual_mode=False):
        self.box_constant = box_constant
        self.collect_data = collect_data
        self.virtual_mode = virtual_mode
        self.last_target = None
        self.no_detection_count = 0

        print("[INFO] Loading neural network")
        self.model = YOLO('lib/best.pt')
        if torch.cuda.is_available():
            self.model.to('cuda')
            print(colored("CUDA ACCELERATION [ENABLED]", "green"))
        else:
            print(colored("[!] CUDA ACCELERATION NOT AVAILABLE", "red"))
            print(colored("[!] Check PyTorch installation, otherwise performance will be low", "red"))

        self.conf = confidence
        self.iou = 0.05

        print("\n[INFO] PRESS 'F1' TO TOGGLE AIMBOT ON/OFF\n[INFO] PRESS 'F2' TO EXIT")
        print("[INFO] USE MOUSE SIDE BUTTON (6) TO AIM")

    @staticmethod
    def update_status_aimbot():
        if Aimbot.aimbot_status == colored("ENABLED", 'green'):
            Aimbot.aimbot_status = colored("DISABLED", 'red')
        else:
            Aimbot.aimbot_status = colored("ENABLED", 'green')
        sys.stdout.write("\033[K")
        print(f"[!] AIMBOT STATUS: [{Aimbot.aimbot_status}]", end="\r")

    @staticmethod
    def left_click():
        ctypes.windll.user32.mouse_event(0x0002)
        Aimbot.sleep(0.000)
        ctypes.windll.user32.mouse_event(0x0004)

    @staticmethod
    def sleep(duration, get_now=time.perf_counter):
        if duration == 0:
            return
        now = get_now()
        end = now + duration
        while now < end:
            now = get_now()

    @staticmethod
    def is_aimbot_enabled():
        return Aimbot.aimbot_status == colored("ENABLED", 'green')

    @staticmethod
    def is_shooting():
        return win32api.GetKeyState(0x01) in (-127, -128)

    @staticmethod
    def is_targeted():
        return win32api.GetKeyState(0x06) in (-127, -128)

    @staticmethod
    def is_right_click_pressed():
        return win32api.GetKeyState(0x02) in (-127, -128)

    def move_crosshair(self, x, y):
        if Aimbot.is_targeted():
            if Aimbot.is_right_click_pressed():
                scale = float(Aimbot.sens_config["targeting_scale"])
            else:
                scale = float(Aimbot.sens_config["xy_scale"])
            dx = int((x - screen_x) * scale)
            dy = int((y - screen_y) * scale)
            Aimbot.ii_.mi = MouseInput(dx, dy, 0, 0x0001, 0, ctypes.pointer(Aimbot.extra))
            input_obj = Input(ctypes.c_ulong(0), Aimbot.ii_)
            ctypes.windll.user32.SendInput(1, ctypes.byref(input_obj), ctypes.sizeof(input_obj))

    def start(self):
        print("[INFO] Screen capture started")
        Aimbot.update_status_aimbot()
        half_screen_width = screen_res_x / 2
        half_screen_height = screen_res_y / 2
        detection_box = {
            'left': int(half_screen_width - self.box_constant // 2),
            'top': int(half_screen_height - self.box_constant // 2),
            'width': int(self.box_constant),
            'height': int(self.box_constant)
        }

        while True:
            if not Aimbot.is_targeted():
                self.last_target = None
                self.no_detection_count = 0

            start_time = time.perf_counter()
            frame = np.array(Aimbot.screen.grab(detection_box))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        boxes = self.model.predict(source=frame, verbose=False, conf=self.conf, iou=self.iou, half=True)
                else:
                    boxes = self.model.predict(source=frame, verbose=False, conf=self.conf, iou=self.iou, half=False)
            result = boxes[0]
            target_detected = False

            if len(result.boxes.xyxy) != 0:
                least_crosshair_dist = None
                closest_detection = None
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    height = y2 - y1

                    min_box_height = 50
                    max_box_height = 300
                    norm = (height - max_box_height) / (min_box_height - max_box_height)
                    norm = max(0, min(1, norm))
                    dynamic_aim_height = round(2 + norm * (4 - 2), 2)

                    relative_head_X = int((x1 + x2) / 2)
                    relative_head_Y = int((y1 + y2) / 2 - height / dynamic_aim_height)
                    
                    own_player = x1 < 15 or (x1 < self.box_constant / 5 and y2 > self.box_constant / 1.2)
                    if own_player:
                        continue
                    crosshair_dist = math.dist((relative_head_X, relative_head_Y),
                                               (self.box_constant / 2, self.box_constant / 2))
                    if least_crosshair_dist is None or crosshair_dist < least_crosshair_dist:
                        least_crosshair_dist = crosshair_dist
                        closest_detection = {
                            "x1y1": (x1, y1),
                            "x2y2": (x2, y2),
                            "relative_head_X": relative_head_X,
                            "relative_head_Y": relative_head_Y
                        }
                if closest_detection:
                    target_detected = True
                    absolute_head_X = closest_detection["relative_head_X"] + detection_box['left']
                    absolute_head_Y = closest_detection["relative_head_Y"] + detection_box['top']
                    self.last_target = (absolute_head_X, absolute_head_Y)
                    self.no_detection_count = 0

                    cv2.circle(frame, (closest_detection["relative_head_X"], closest_detection["relative_head_Y"]),
                               5, (115, 244, 113), -1)
                    cv2.line(frame, (closest_detection["relative_head_X"], closest_detection["relative_head_Y"]),
                             (self.box_constant // 2, self.box_constant // 2), (244, 242, 113), 2)
                    
                    if use_trigger_bot and not Aimbot.is_shooting():
                        Aimbot.left_click()
                    cv2.putText(frame, "LOCKED", (x1 + 40, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                (115, 244, 113), 2)

                    if Aimbot.is_aimbot_enabled():
                        self.move_crosshair(absolute_head_X, absolute_head_Y)
            else:
                if Aimbot.is_targeted() and self.last_target is not None:
                    self.no_detection_count += 1
                    if self.no_detection_count >= 2:
                        self.last_target = None
                        self.no_detection_count = 0

            if not target_detected and Aimbot.is_targeted() and self.last_target is not None:
                if Aimbot.is_aimbot_enabled():
                    self.move_crosshair(self.last_target[0], self.last_target[1])

            fps = int(1 / (time.perf_counter() - start_time))
            if fps < 300:
                cv2.putText(frame, f"FPS: {fps}", (5, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"FPS: {fps}", (5, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2)
            cv2.imshow("Lunar Vision - GPU Accelerated", frame)
            if cv2.waitKey(1) & 0xFF == ord('0'):
                break

    @staticmethod
    def clean_up():
        print("\n[INFO] F2 PRESSED. EXITING...")
        if hasattr(Aimbot, 'screen') and Aimbot.screen:
            try:
                Aimbot.screen.close()
            except AttributeError:
                pass
        os._exit(0)

if __name__ == "__main__":
    virtual_mode = "virtual" in sys.argv
    lunar = Aimbot(collect_data="collect_data" in sys.argv, virtual_mode=virtual_mode)
    lunar.start()
