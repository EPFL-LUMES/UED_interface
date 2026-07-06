import serial
try:
    s = serial.Serial('COM10', 115200, timeout=1)
    print("Opened ok")
    s.close()
except Exception as e:
    print("Failed:", e)
