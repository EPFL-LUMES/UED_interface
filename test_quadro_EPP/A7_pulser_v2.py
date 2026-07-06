import pyvisa
import time


class A7_pulser:
    def __init__(self, COM_PORT, **kwargs):
        for key, value in kwargs.items():
            if key == 'thing':
                self.thing = value

        # connect to the device
        rm = pyvisa.ResourceManager()
        self.rs232 = rm.open_resource(COM_PORT)
        # baud_rate = 9600) # also works for baud rate setting

        self.rs232.baud_rate = 115200
        self.rs232.read_termination = '\n'
        self.rs232.write_termination = ''
        self.rs232.send_end = False
        self.rs232.query_delay = 0.05  # 50ms wait between query write and read

        # internal channel indices are reversed compared to how the box is labeled.
        # channel 0 is channel D, channel 1 is channel C, etc.
        # reverse the indices here to make this transparent for the user
        self.channel = []
        for i in range(0, 4):
            self.channel.append(A7_pulser_channel(self, 3-i))

    def flush(self):
        try:
            while True:
                self.rs232.read()
        except:
            pass

    def write_register(self, addr, value):
        if type(addr) == str:
            addr_string = addr
        else:
            addr_string = "{:04X}".format(addr)

        if type(value) == str:
            value_string = value
        else:
            value_string = "{:04X}".format(value)

        self.rs232.query(addr_string + "g")
        self.rs232.read()  # returns A= XXXX\r\n

        self.rs232.query(value_string + "w")
        returnval = self.rs232.read()  # should return "D:=XXXX\r\n"

        if (returnval[:3] != "D:=") or (returnval[3:7] != value_string):
            return -1
        else:
            return 0

    def read_register(self, addr):
        if type(addr) == str:
            addr_string = addr
        else:
            addr_string = "{:04X}".format(addr)

        self.rs232.query(addr_string + "g")
        self.rs232.read()  # returns A= XXXX\r\n

        self.rs232.query("r")
        returnval = self.rs232.read()  # should return "D=XXXX\r\n"

        if (returnval[:2] != "D="):
            return -1
        else:
            return int(returnval[2:6], 16)

    # set the internal trigger rate.  rate should be expressed in Hz
    def set_trigger_rate(self, rate):
        triggerRateWord = 125000000/rate
        triggerRateWordLSB = int(triggerRateWord) & 0x0000FFFF
        triggerRateWordMSB = (int(triggerRateWord) & 0xFFFF0000) >> 16

        returnval = self.write_register(0x0003, "{:04X}".format(triggerRateWordLSB))
        if returnval != 0:
            return returnval

        return self.write_register(0x0004, "{:04X}".format(triggerRateWordMSB))

    def set_trigger_threshold(self, threshold):
        threshold_word = int(threshold*(2**12)/5.)

        return self.write_register(0x0010, "{:04X}".format(threshold_word))

    def set_gate_threshold(self, threshold):
        threshold_word = int(threshold*(2**12)/5.)

        return self.write_register(0x0020, "{:04X}".format(threshold_word))

    def close(self):
        self.rs232.close()


class A7_pulser_channel:
    def __init__(self, parent, channel, **kwargs):
        for key, value in kwargs.items():
            if key == 'thing':
                self.thing = value
        self.pulser = parent
        self.index = channel
        self.source_dict = {"OFF": 0, "TRIG": 1, "GATE": 2, "TRIG AND GATE": 3, "INTERNAL": 4}

    def set_source(self, source):
        if source not in self.source_dict:
            raise ValueError("channel source " + str(source) + " is not supported")

        addr = 0x0100 + self.index
        return self.pulser.write_register(addr, self.source_dict[source])

    def set_divider(self, divider):
        addr = 0x0110 + self.index

        return self.pulser.write_register(addr, divider)

    def set_delay(self, delay):
        # delay register is set in nanoseconds
        delayWord = int(delay*1e9)
        delayWordLSB = int(delayWord) & 0x0000FFFF
        delayWordMSB = (int(delayWord) & 0xFFFF0000) >> 16

        addr = 0x0120 + self.index
        returnval = self.pulser.write_register(addr, "{:04X}".format(delayWordLSB))
        if returnval != 0:
            return returnval

        addr = 0x0130 + self.index
        return self.pulser.write_register(addr, "{:04X}".format(delayWordMSB))

    def set_width(self, width):
        # pulse width register is set in nanoseconds
        widthWord = int(width*1e9)
        widthWordLSB = int(widthWord) & 0x0000FFFF
        widthWordMSB = (int(widthWord) & 0xFFFF0000) >> 16

        addr = 0x0140 + self.index
        returnval = self.pulser.write_register(addr, "{:04X}".format(widthWordLSB))
        if returnval != 0:
            return returnval

        addr = 0x0150 + self.index
        return self.pulser.write_register(addr, "{:04X}".format(widthWordMSB))

################################################################


if __name__ == "__main__":
    # instantiate VISA resource manager and print a list of COM ports
    rm = pyvisa.ResourceManager()
    resourceList = rm.list_resources()

    # get resource name
    for id, resourceName in enumerate(resourceList):
        print(str(id) + ": " + resourceName)

    index = int(input("Which Index? >"))

    # connect to arduino
    pulser = A7_pulser(resourceList[index])  # baud_rate = 9600) # also works for baud rate setting

    pulser.set_trigger_rate(250000)
    pulser.set_trigger_threshold(1.25)
    pulser.set_gate_threshold(1.25)
    channelA = {"source": "TRIG AND GATE", "divider": 0, "delay": 12.0e-6, "width": 9e-6}  # for pump-probe QUADRO
    channelB = {"source": "INTERNAL", "divider": 0, "delay": 500e-9, "width": 1e-6}
    channelC = {"source": "TRIG", "divider": 0, "delay": 12.0e-6, "width": 9e-6}  # for pump-probe reflectivity
    channelD = {"source": "TRIG AND GATE", "divider": 0, "delay": 1e-6, "width": 1e-6}

    for i, channel in enumerate([channelA, channelB, channelC, channelD]):
        pulser.channel[i].set_source(channel['source'])
        pulser.channel[i].set_divider(channel['divider'])
        pulser.channel[i].set_delay(channel['delay'])
        pulser.channel[i].set_width(channel['width'])

    pulser.close()
