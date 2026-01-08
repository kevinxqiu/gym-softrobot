#!/usr/bin/env python

from dynamixel_controller import DynamixelController, BaseModel
import os
import time
import numpy as np

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
    
n_motors = 1

print("Attempting to connect to dynamixels...")

num_motors = 1  # Number of motors to control
motor_list = []
for i in range(num_motors):
    motor_list.append(BaseModel(2*i))

print("Motor list: " + str(motor_list))

controller = DynamixelController("/dev/ttyUSB0", motor_list, baudrate=115200, latency_time=10)
controller.activate_controller()
controller.set_operating_mode_all("extended_position_control")
controller.torque_on()
# controller.set_profile_velocity([100,100,100])
# controller.set_profile_acceleration([5,5,5])

result_info = controller.read_info_with_unit(
    pwm_unit="percent",
    angle_unit="deg",
    current_unit="mA",
    retry=False,
    fast_read=False,
)
pos = result_info[0] + 2048
zero_pos = np.array(pos, dtype=float)
goal_pos = np.array(pos, dtype=float)
for i in range(num_motors):
    print("Dynamixel" + str(i) + ":  " + str(pos[i]))

enc_steps_per_rev = 4096
oscillation_amplitude = 1024.0
oscillation_frequency = 0.2
update_dt = 0.02

start_time = time.perf_counter()
while 1:
    if os.name == "nt" and msvcrt.kbhit():
        if getch() == chr(0x1b):
            break

    elapsed = time.perf_counter() - start_time
    offset = oscillation_amplitude * np.sin(2.0 * np.pi * oscillation_frequency * elapsed)
    goal_pos = zero_pos + offset
    controller.set_goal_position([int(value) for value in np.atleast_1d(goal_pos)])
    time.sleep(update_dt)



