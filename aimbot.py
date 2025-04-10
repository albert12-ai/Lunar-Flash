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
from ultralytics import YOLO
import dearpygui.dearpygui as dpg

torch.backends.cudnn.benchmark = True

screensize = {
    'width': ctypes.windll.user32.GetSystemMetrics(0),
    'height': ctypes.windll.user32.GetSystemMetrics(1)
}
screen_width = screensize['width']
screen_height = screensize['height']
center_x = int(screen_width / 2)
center_y = int(screen_height / 2)

PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput)
    ]

class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", Input_I)
    ]

class Aimbot:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    screen = mss.mss()
    
    config_path = "lib/config/config.json"
    if not os.path.exists(config_path):
        default_config = {
            "xy_sens": 5.0,                     # XY sensitivity
            "targeting_sens": 5.0,               # Targeting sensitivity
            "xy_scale": 10/5.0,                  # XY scale factor
            "targeting_scale": 1000/(5.0*5.0),   # Targeting scale factor
            "toggle_key_code": 0x70,             # Toggle key (F1)
            "exit_key_code": 0x71,               # Exit key (F2)
            "aim_key_code": 0x06,                # Aim key (usually mouse button)
            "min_aim_height": 2.0,               # Minimum aim height
            "max_aim_height": 4.0,               # Maximum aim height
            "min_box_height": 50,                # Minimum box height
            "max_box_height": 300,               # Maximum box height
            "confidence": 0.45,                  # Confidence threshold for detections
            "iou": 0.05,                       # IoU threshold for detections
            "use_trigger_bot": False,            # Whether to use triggerbot
            "aiming_lerp_factor": 0.5,           # Lerp factor while aiming (csak simításra!)
            "not_aiming_lerp_factor": 0.5,       # Lerp factor while not aiming (csak simításra!)
            "box_constant": 320,                 # Detection box constant (size)
            "triggerbot_threshold": 10,          # Triggerbot activation range
            "max_aim_delta": 50,                 # Maximum aim delta in pixels
            "min_aim_delta": 5,                  # Minimum aim delta in pixels
            "max_move_pixels": 100               # Maximum pixel movement allowed
        }
        os.makedirs("lib/config", exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=4)
    
    with open(config_path) as f:
        config = json.load(f)
    
    status = "ENABLED"

    def __init__(self, collect_data=False, virtual_mode=False):
        self.collect_data = collect_data
        self.virtual_mode = virtual_mode
        self.last_target = None
        self.missed_frames = 0

        self.model = YOLO('lib/best.pt')
        if torch.cuda.is_available():
            self.model.to('cuda')

        self.sens_settings = self.config
        self.toggle_key = self.config.get("toggle_key_code", 0x70)
        self.exit_key = self.config.get("exit_key_code", 0x71)
        self.aim_key = self.config.get("aim_key_code", 0x06)
        self.min_aim_h = self.config.get("min_aim_height", 2.0)
        self.max_aim_h = self.config.get("max_aim_height", 4.0)
        self.min_box_h = self.config.get("min_box_height", 50)
        self.max_box_h = self.config.get("max_box_height", 300)
        self.conf_thresh = self.config.get("confidence", 0.45)
        self.iou_thresh = self.config.get("iou", 0.05)
        self.triggerbot_enabled = self.config.get("use_trigger_bot", False)
        self.aim_smooth = self.config.get("aiming_lerp_factor", 0.5)      # Csak simítás
        self.idle_smooth = self.config.get("not_aiming_lerp_factor", 0.5)   # Csak simítás
        self.detection_size = self.config.get("box_constant", 320)
        self.triggerbot_range = self.config.get("triggerbot_threshold", 10)
        self.max_aim_delta = self.config.get("max_aim_delta", 50)
        self.min_aim_delta = self.config.get("min_aim_delta", 5)
        self.max_move_pixels = self.config.get("max_move_pixels", 100)

        self.update_scaling()

        self.key_binding_mode = None
        self.init_gui()

    def update_scaling(self):
        xy_sens = self.sens_settings.get("xy_sens", 5.0)
        targeting_sens = self.sens_settings.get("targeting_sens", xy_sens)
        print("[INFO] In-game aiming sensitivity must match the scope sensitivity!")
        # A sensitivity a mozgás mértékét szabályozza
        self.sens_settings["xy_scale"] = 10 / xy_sens
        self.sens_settings["targeting_scale"] = 1000 / (targeting_sens * xy_sens)
    
    def init_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="Aimbot Configuration", width=400, height=900, resizable=False)
        with dpg.window(label="Settings", width=400, height=900):
            with dpg.group():
                dpg.add_text("Key Bindings")
                with dpg.group(horizontal=True):
                    dpg.add_text("Toggle key: ")
                    self.toggle_key_label = dpg.add_text(f"0x{self.toggle_key:02X}")
                    dpg.add_button(label="Set", callback=lambda: self.set_key_binding("toggle"))
                with dpg.group(horizontal=True):
                    dpg.add_text("Exit key: ")
                    self.exit_key_label = dpg.add_text(f"0x{self.exit_key:02X}")
                    dpg.add_button(label="Set", callback=lambda: self.set_key_binding("exit"))
                with dpg.group(horizontal=True):
                    dpg.add_text("Aim key: ")
                    self.aim_key_label = dpg.add_text(f"0x{self.aim_key:02X}")
                    dpg.add_button(label="Set", callback=lambda: self.set_key_binding("aim"))
            
            with dpg.group():
                dpg.add_text("Sensitivity")
                dpg.add_input_float(label="XY Sensitivity", default_value=self.sens_settings.get("xy_sens", 5.0),
                                      callback=lambda s, a: self.update_setting("xy_sens", a))
                dpg.add_input_float(label="Aiming Sensitivity", default_value=self.sens_settings.get("targeting_sens", 5.0),
                                      callback=lambda s, a: self.update_setting("targeting_sens", a))
                dpg.add_input_float(label="XY Scale", default_value=self.sens_settings["xy_scale"], readonly=True)
                dpg.add_input_float(label="ADS Scale", default_value=self.sens_settings["targeting_scale"], readonly=True)
            
            with dpg.group():
                dpg.add_text("Aimbot Parameters")
                dpg.add_input_float(label="Min Aim Height", default_value=self.min_aim_h, callback=lambda s, a: setattr(self, "min_aim_h", a))
                dpg.add_input_float(label="Max Aim Height", default_value=self.max_aim_h, callback=lambda s, a: setattr(self, "max_aim_h", a))
                dpg.add_input_float(label="Min Box Height", default_value=self.min_box_h, callback=lambda s, a: setattr(self, "min_box_h", a))
                dpg.add_input_float(label="Max Box Height", default_value=self.max_box_h, callback=lambda s, a: setattr(self, "max_box_h", a))
                dpg.add_input_float(label="Confidence", default_value=self.conf_thresh, callback=lambda s, a: setattr(self, "conf_thresh", a))
                dpg.add_input_float(label="IOU Threshold", default_value=self.iou_thresh, callback=lambda s, a: setattr(self, "iou_thresh", a))
                dpg.add_checkbox(label="Enable Triggerbot", default_value=self.triggerbot_enabled, callback=lambda s, a: setattr(self, "triggerbot_enabled", a))
                dpg.add_input_int(label="Triggerbot Range", default_value=self.triggerbot_range, callback=lambda s, a: setattr(self, "triggerbot_range", a))
                dpg.add_input_float(label="Aim Smoothing", default_value=self.aim_smooth, callback=lambda s, a: setattr(self, "aim_smooth", a))
                dpg.add_input_float(label="Idle Smoothing", default_value=self.idle_smooth, callback=lambda s, a: setattr(self, "idle_smooth", a))
                dpg.add_input_float(label="Detection Area", default_value=self.detection_size, callback=lambda s, a: setattr(self, "detection_size", a))
                dpg.add_input_float(label="Max Aim Delta", default_value=self.max_aim_delta, callback=lambda s, a: setattr(self, "max_aim_delta", a))
                dpg.add_input_float(label="Min Aim Delta", default_value=self.min_aim_delta, callback=lambda s, a: setattr(self, "min_aim_delta", a))
                dpg.add_input_int(label="Max Pixel Movement", default_value=self.max_move_pixels, callback=lambda s, a: setattr(self, "max_move_pixels", a))
            
            dpg.add_button(label="Save Configuration", callback=self.save_config)
        
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def set_key_binding(self, target):
        self.key_binding_mode = target

    def update_setting(self, key, value):
        self.sens_settings[key] = value
        if key in ["xy_sens", "targeting_sens"]:
            self.update_scaling()

    def save_config(self):
        self.update_scaling()
        config = {
            "xy_sens": self.sens_settings.get("xy_sens", 5.0),
            "targeting_sens": self.sens_settings.get("targeting_sens", 5.0),
            "xy_scale": self.sens_settings["xy_scale"],
            "targeting_scale": self.sens_settings["targeting_scale"],
            "toggle_key_code": self.toggle_key,
            "exit_key_code": self.exit_key,
            "aim_key_code": self.aim_key,
            "min_aim_height": self.min_aim_h,
            "max_aim_height": self.max_aim_h,
            "min_box_height": self.min_box_h,
            "max_box_height": self.max_box_h,
            "confidence": self.conf_thresh,
            "iou": self.iou_thresh,
            "use_trigger_bot": self.triggerbot_enabled,
            "aiming_lerp_factor": self.aim_smooth,
            "not_aiming_lerp_factor": self.idle_smooth,
            "box_constant": self.detection_size,
            "triggerbot_threshold": self.triggerbot_range,
            "max_aim_delta": self.max_aim_delta,
            "min_aim_delta": self.min_aim_delta,
            "max_move_pixels": self.max_move_pixels
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

    def run(self):
        self.toggle_status()
        half_width = screen_width / 2
        half_height = screen_height / 2

        while dpg.is_dearpygui_running():
            if self.key_binding_mode:
                for key in range(0x01, 0xFF):
                    if win32api.GetAsyncKeyState(key) & 0x8000:
                        if self.key_binding_mode == "toggle":
                            self.toggle_key = key
                            dpg.set_value(self.toggle_key_label, f"0x{key:02X}")
                        elif self.key_binding_mode == "exit":
                            self.exit_key = key
                            dpg.set_value(self.exit_key_label, f"0x{key:02X}")
                        elif self.key_binding_mode == "aim":
                            self.aim_key = key
                            dpg.set_value(self.aim_key_label, f"0x{key:02X}")
                        self.key_binding_mode = None
                        break

            if win32api.GetKeyState(self.toggle_key) & 0x8000:
                self.toggle_status()
                time.sleep(0.2)
            if win32api.GetKeyState(self.exit_key) & 0x8000:
                self.shutdown()

            detection_area = {
                'left': int(half_width - self.detection_size // 2),
                'top': int(half_height - self.detection_size // 2),
                'width': int(self.detection_size),
                'height': int(self.detection_size)
            }

            frame = np.array(self.screen.grab(detection_area))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        results = self.model.predict(source=frame, conf=self.conf_thresh, iou=self.iou_thresh, half=True)
                else:
                    results = self.model.predict(source=frame, conf=self.conf_thresh, iou=self.iou_thresh)
            
            detection = results[0]
            target_found = False

            if len(detection.boxes.xyxy) > 0:
                closest_target = None
                min_distance = float('inf')
                for box in detection.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    h = y2 - y1

                    norm_height = (h - self.max_box_h) / (self.min_box_h - self.max_box_h)
                    norm_height = max(0, min(1, norm_height))
                    aim_height = self.min_aim_h + norm_height * (self.max_aim_h - self.min_aim_h)

                    head_x = int((x1 + x2) / 2)
                    head_y = int((y1 + y2) / 2 - h / aim_height)
                    head_y = min(max(head_y, y1), y2)
                    
                    if x1 < 15 or (x1 < self.detection_size / 5 and y2 > self.detection_size / 1.2):
                        continue
                    
                    distance = math.dist((head_x, head_y), (self.detection_size/2, self.detection_size/2))
                    if distance < min_distance:
                        min_distance = distance
                        closest_target = {
                            "x": head_x + detection_area['left'],
                            "y": head_y + detection_area['top'],
                            "distance": distance
                        }

                if closest_target:
                    target_found = True
                    self.last_target = (closest_target["x"], closest_target["y"])
                    self.missed_frames = 0

                    if all([
                        self.triggerbot_enabled,
                        self.is_aiming(),
                        closest_target["distance"] <= self.triggerbot_range,
                        not self.is_firing()
                    ]):
                        self.click()

                    if self.is_active():
                        self.aim(closest_target["x"], closest_target["y"])
            
            if not target_found and self.is_aiming() and self.last_target:
                self.missed_frames += 1
                if self.missed_frames >= 2:
                    self.last_target = None
                    self.missed_frames = 0

            dpg.render_dearpygui_frame()

    def aim(self, x, y):
        """
        A cél eléréséhez szükséges mozgásvektor számítása:
        1. A sensitivity (xy_scale vagy targeting_scale) határozza meg a teljes elmozdulás nagyságát.
        2. A smooth faktor (aim_smooth vagy idle_smooth) kizárólag a simításért felel.
        3. A dinamikus maximum lépést is figyelembe vesszük.
        """
        if self.is_aiming():
            # Sensitivity faktor kiválasztása: normál vagy ADS mód szerint
            sens_factor = self.sens_settings["xy_scale"] if not self.is_ads() else self.sens_settings["targeting_scale"]
            # Smooth faktor a simításra
            smooth_factor = self.aim_smooth if self.is_ads() else self.idle_smooth
            
            # Teljes mozgás a sensitivity alapján
            raw_dx = (x - center_x) * sens_factor
            raw_dy = (y - center_y) * sens_factor
            
            # Smooth alkalmazása a sima mozgásért
            dx = raw_dx * smooth_factor
            dy = raw_dy * smooth_factor
            
            current_magnitude = math.hypot(dx, dy)
            d = math.hypot(x - center_x, y - center_y)
            norm_factor = self.detection_size / 2

            dynamic_max_delta = self.max_aim_delta * (d / norm_factor)
            dynamic_max_delta = max(dynamic_max_delta, self.min_aim_delta)
            dynamic_max_delta = min(dynamic_max_delta, self.max_move_pixels)

            if current_magnitude > dynamic_max_delta:
                scale = dynamic_max_delta / current_magnitude
                dx = int(dx * scale)
                dy = int(dy * scale)
            else:
                dx = int(dx)
                dy = int(dy)

            self.ii_.mi = MouseInput(dx, dy, 0, 0x0001, 0, ctypes.pointer(self.extra))
            input_struct = Input(ctypes.c_ulong(0), self.ii_)
            ctypes.windll.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(input_struct))

    @staticmethod
    def click():
        ctypes.windll.user32.mouse_event(0x0002)
        time.sleep(0.001)
        ctypes.windll.user32.mouse_event(0x0004)

    def is_active(self):
        return self.status == "ENABLED"

    def is_firing(self):
        return win32api.GetKeyState(0x01) < 0

    def is_aiming(self):
        return win32api.GetKeyState(self.aim_key) < 0

    def is_ads(self):
        return win32api.GetKeyState(0x02) < 0

    def toggle_status(self):
        self.status = "DISABLED" if self.status == "ENABLED" else "ENABLED"

    def shutdown(self):
        if hasattr(self, 'screen'):
            self.screen.close()
        dpg.destroy_context()
        os._exit(0)

if __name__ == "__main__":
    bot = Aimbot()
    bot.run()
