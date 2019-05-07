import serial


class arduino_serial:
    def __init__(self, port, rate=9600):
        self.arduino = serial.Serial(port, rate)

    def writeString(self, string):
        self.arduino.write(str.encode(string))
