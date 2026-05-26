import os
import time
import threading
import cv2
from gpiozero import AngularServo

# Định nghĩa chân kết nối
Servo1_Pin = 23
Servo2_Pin = 24

servo1 = AngularServo(Servo1_Pin, min_pulse_width=0.5/1000, max_pulse_width=2.6/1000)
servo2 = AngularServo(Servo2_Pin, min_pulse_width=0.5/1000, max_pulse_width=2.6/1000)

currentDeg = 0
while (True):
    deg = float(input("servo1 deg: "))
    
    if currentDeg != deg:
        currentDeg = deg
        print(f"Điều khiển servo1 đến góc {deg} độ")
        servo1.angle = deg