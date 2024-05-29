import time
import signal
import sys
import board
import pigpio
import adafruit_icm20x
import numpy as np
import RPi.GPIO as GPIO
import tkinter as tk
from tkinter import messagebox

# Kalman filter parameters
dt = 0.01  # Sampling time in seconds
Q_angle = 0.001  # Process noise covariance
Q_bias = 0.003  # Gyroscope bias covariance
R_measure = 0.03  # Measurement noise covariance

# Other parameters
servox = 13
servoy = 12
input_min = -0.9
input_max = 0.9
output_min = 500
output_max = 2500

# Initialize IMU
i2c = board.I2C()
icm = adafruit_icm20x.ICM20948(i2c)

# Servo setup
pwm = pigpio.pi()
pwm.set_mode(servox, pigpio.OUTPUT)
pwm.set_mode(servoy, pigpio.OUTPUT)

pwm.set_PWM_frequency(servox, 50)
pwm.set_PWM_frequency(servoy, 50)

pwm.set_servo_pulsewidth(servox, 500)
pwm.set_servo_pulsewidth(servoy, 500)
time.sleep(0.5)

pwm.set_servo_pulsewidth(servox, 1500)
pwm.set_servo_pulsewidth(servoy, 1500)
time.sleep(0.5)

pwm.set_servo_pulsewidth(servox, 2500)
pwm.set_servo_pulsewidth(servoy, 2500)
time.sleep(0.5)

# Initialize Kalman filter variables
angle = np.zeros(3)  # Roll, Pitch, Yaw (in radians)
bias = np.zeros(3)   # Gyroscope bias (in radians/second)
P = np.eye(3)        # Error covariance matrix

# State transition matrix (A)
A = np.eye(3)

# Measurement matrix (H)
H = np.eye(3)

# Process noise covariance matrix (Q)
Q = np.diag([Q_angle, Q_angle, Q_bias])

# Measurement noise covariance matrix (R)
R = np.diag([R_measure, R_measure, R_measure])

# Identity matrix (I)
I = np.eye(3)

# Initialize GPIO for the buttons and LEDs
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(23, GPIO.OUT)
GPIO.output(22, GPIO.LOW)
GPIO.output(23, GPIO.HIGH)

reading_enabled = True

def reset_reference():
    global angle, bias, P
    gyro_x, gyro_y, gyro_z = icm.gyro
    angle = np.array([np.deg2rad(gyro_x), np.deg2rad(gyro_y), np.deg2rad(gyro_z)])
    bias = np.zeros(3)
    P = np.eye(3)
    print("Reference reset to current IMU readings.")

def toggle_reading(channel):
    global reading_enabled
    reading_enabled = not reading_enabled
    if reading_enabled:
        reset_reference()
        GPIO.output(22, GPIO.HIGH)  # Turn on the LED
    else:
        GPIO.output(22, GPIO.LOW)  # Turn off the LED
    print("Reading enabled:", reading_enabled)

def skalering(input_value, input_min, input_max, output_min, output_max):
    global skalert_verdi

    # Ensure the input value is numeric
    input_value = float(input_value)
    input_min = float(input_min)
    input_max = float(input_max)
    output_min = float(output_min)
    output_max = float(output_max)

    # Scale the input value to the output range
    skalert_verdi = ((input_value - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
    return skalert_verdi

def signal_handler(sig, frame):
    GPIO.output(23, GPIO.LOW)  # Turn off the LED when the script is deactivated
    GPIO.cleanup()  # Clean up GPIO on exit
    sys.exit(0)

# Detect button press and call reset_reference for GP17
GPIO.add_event_detect(17, GPIO.FALLING, callback=lambda x: reset_reference(), bouncetime=300)

# Detect button press and call toggle_reading for GP27
GPIO.add_event_detect(27, GPIO.FALLING, callback=toggle_reading, bouncetime=300)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# GUI Setup
def update_input_ranges():
    global input_min, input_max
    try:
        new_input_min = float(entry_input_min.get())
        new_input_max = float(entry_input_max.get())
        if new_input_min < new_input_max:
            input_min = new_input_min
            input_max = new_input_max
            messagebox.showinfo("Success", f"Updated input ranges:\ninput_min: {input_min}\ninput_max: {input_max}")
        else:
            messagebox.showerror("Error", "input_min must be less than input_max.")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers.")

root = tk.Tk()
root.title("IMU Settings")

frame = tk.Frame(root)
frame.pack(pady=20)

label_input_min = tk.Label(frame, text="Input Min:")
label_input_min.grid(row=0, column=0, padx=10)
entry_input_min = tk.Entry(frame)
entry_input_min.grid(row=0, column=1, padx=10)
entry_input_min.insert(0, str(input_min))

label_input_max = tk.Label(frame, text="Input Max:")
label_input_max.grid(row=1, column=0, padx=10)
entry_input_max = tk.Entry(frame)
entry_input_max.grid(row=1, column=1, padx=10)
entry_input_max.insert(0, str(input_max))

update_button = tk.Button(frame, text="Update", command=update_input_ranges)
update_button.grid(row=2, columnspan=2, pady=20)

try:
    while True:
        root.update_idletasks()
        root.update()

        if reading_enabled:
            # Read gyro, accelerometer, and magnetometer data
            gyro_x, gyro_y, gyro_z = icm.gyro

            # Convert gyro data from degrees per second to radians per second
            gyro_x_rad = np.deg2rad(gyro_x)
            gyro_y_rad = np.deg2rad(gyro_y)
            gyro_z_rad = np.deg2rad(gyro_z)

            # Predict
            angle += dt * (np.array([gyro_x_rad, gyro_y_rad, gyro_z_rad]) - bias)
            P += dt * (A @ P @ A.T + Q)

            # Update
            z = np.array([angle[0], angle[1], 0])  # Yaw is not directly measured, set to 0
            y = z - angle
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            angle += K @ y
            bias += K @ y

            # Update error covariance matrix
            P = (I - K @ H) @ P

            # Print the original gyroscope data and the filtered orientation angles
            print("Original Gyro (deg/s): X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(gyro_x, gyro_y, gyro_z))
            print("Filtered Orientation (deg): Roll: {:.2f}, Pitch: {:.2f}, Yaw: {:.2f}".format(np.rad2deg(angle[0]), np.rad2deg(angle[1]), np.rad2deg(angle[2])))

            skalering(format(np.rad2deg(angle[0])), input_min, input_max, output_min, output_max)
            gradx = skalert_verdi
            skalering(format(np.rad2deg(angle[1])), input_min, input_max, output_min, output_max)
            grady = skalert_verdi

            if gradx >= 2500 or gradx <= 500:
                print("X: IKKE I RANGE")
            else:
                pwm.set_servo_pulsewidth(servox, gradx)
            if grady >= 2500 or grady <= 500:
                print("Y: IKKE I RANGE")
            else:
                pwm.set_servo_pulsewidth(servoy, grady)
        # Sleep for a short time
        time.sleep(dt)

finally:
    GPIO.output(23, GPIO.LOW)  # Turn off the LED when the script is deactivated
    GPIO.cleanup()  # Clean up GPIO on exit
