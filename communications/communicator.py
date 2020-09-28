import serial
import os, time
import RPi.GPIO as GPIO


class Communicator:
    __instance__ = None

    def __init__(self):
        """private constructor. Don't call from the outside"""
        if Communicator.__instance__ != None:
            raise Exception("Multiple calls to the Communicator constructor")
        else:
            Communicator.__instance__ = self
            self.__init_module__()

    def __init_module__(self):
        GPIO.setmode(GPIO.BOARD)
        self.port = serial.Serial('/dev/ttyS0', baudrate=9600, timeout=1)

        self.port.write(b'AT\r')
        rcv = self.port.read(10)
        print(rcv)
        time.sleep(1)

        self.port.write(b'AT+CMGF=1\r')
        time.sleep(3)

    def send_mesage(self, number: str, messge: str):
        cmd = f'AT+CMGS-"{number}"\r'
        cmd = cmd.encode()
        self.port.write(cmd)
        message = message + chr(26)
        message = messge.encode()
        time.sleep(3)
        self.port.reset_output_buffer()
        time.sleep(1)
        self.port.write(message)
        time.sleep(3)

    @staticmetod
    def getComunicator() -> Communicator:
        if Communicator.__instance__ == None:
            Communicator()
        return Communicator.__instance__
