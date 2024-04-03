import serial
import logging


# Configure serial connection
# Converts Voltage to Pressure
def vtop(v):
    v_s = 5
    p_max = 150
    p_min = 0
    p = ((v - 0.1 * v_s) * (p_max - p_min) / (0.8 * v_s)) + p_min
    return p


def remove_crud(string):
    """Return string without useless information.

     Return string with trailing zeros after a decimal place, trailing
     decimal points, and leading and trailing spaces removed.
     """
    if "." in string:
        string = string.rstrip('0')

    string = string.lstrip('0 ')
    string = string.rstrip(' .')

    return string


# Setting up Host Computer COM Port
class Chain(serial.Serial):
    def __init__(self, port):
        serial.Serial.__init__(self, port=port, baudrate=115200, stopbits=serial.STOPBITS_TWO,
                               parity=serial.PARITY_NONE, timeout=2)
        self.flushOutput()
        self.flushInput()
        logging.info('Chain created on %s', port)


# Pump Control Functions
class pump:
    def __init__(self, chain, address=0, name='Pump 11'):
        self.name = name
        self.serialcon = chain
        self.address = str(address)
        self.diameter = None
        self.flowrate = None
        self.targetvolume = None

    # Sending Commands to pump
    def write(self, command):
        self.serialcon.write((self.address + command + '\r').encode())

    # Reading Replies from pump
    def read_until(self, string):
        self.serialcon.reset_output_buffer()
        response = self.serialcon.read_until(expected=string.encode())
        return response.decode()


    def poll(self, cmd):
        self.write('poll ' + cmd)

    # Syringe Set up Related
    # Sets up Syringe Diameter
    def setdiameter(self, diameter):

        # if diameter > 35 or diameter <0.1:
        # raise PumpError('%s: diameter %s mm is out of range' %
        # (self.name, diameter))
        diameter = str(diameter)
        if len(diameter) > 5:
            if diameter[2] == '.':  # e.g. 30.2222222
                diameter = diameter[0:5]
            elif diameter[1] == '.':  # e.g. 3.222222
                diameter = diameter[0:4]
        else:
            diameter = remove_crud(diameter)
            cmd = 'diameter ' + diameter
            self.write(cmd)

    # Sets up infuse ramp speed
    def setiramp(self, startrate, startunit, endrate, endunit, ramptime):

        startrate = str(startrate)
        startunit = str(startunit)
        endrate = str(endrate)
        endunit = str(endunit)
        ramptime = str(ramptime)
        cmd = 'iramp ' + startrate + ' ' + startunit + ' ' + endrate + ' ' + endunit + ' ' + ramptime
        self.write(cmd)

    # Sets up withdraw ramp speed
    def setwramp(self, startrate, startunit, endrate, endunit, ramptime):
        startrate = str(startrate)
        startunit = str(startunit)
        endrate = str(endrate)
        endunit = str(endunit)
        ramptime = str(ramptime)
        cmd = 'wramp ' + startrate + ' ' + startunit + ' ' + endrate + ' ' + endunit + ' ' + ramptime
        self.write(cmd)

    # Sets up infuse rate
    def setirate(self, rate, unit):
        rate = str(rate)
        unit = str(unit)
        cmd = 'irate ' + rate + ' ' + unit
        self.write(cmd)

    # Sets up withdraw rate
    def setwrate(self, rate, unit):
        rate = str(rate)
        unit = str(unit)
        cmd = 'wrate ' + rate + ' ' + unit
        self.write(cmd)

    # Run pump
    def run(self):
        self.write('run')

    # Infuse
    def irun(self):
        self.write('irun')

    # Withdraw
    def wrun(self):
        self.write('wrun')

    # Stop
    def stp(self):
        self.write('stp')

    # Volume Related
    # Sets up Syringe Volume
    def svolume(self, volume, unit):
        volume = str(volume)
        unit = str(unit)
        cmd = 'svolume ' + volume + ' ' + unit
        self.write(cmd)

    # Sets up Target Volume
    def tvolume(self, volume, unit):
        volume = str(volume)
        unit = str(unit)
        cmd = 'tvolume ' + volume + ' ' + unit
        self.write(cmd)

    # Infused volume
    def ivolume(self):
        self.write('ivolume')
        resp = (self.read_until('ul'))
        iv = resp.strip('\n00: ')
        iv = iv.split()
        volume = iv[0]
        unit = iv[1]
        # iv = iv.strip('mul')
        # print('infused volume:')
        # print(str(iv))
        return float(volume), unit

    # Withdrawn volume
    def wvolume(self):
        self.write('wvolume')
        # resp: \n00:\n00:\n00<\n00:272.679\r\n00:T*
        resp = (self.read_until('ul'))
        wv = resp.strip('\n00: ')
        wv = wv.split()
        volume = wv[0]
        unit = wv[1]
        # print('withdrawn volume:')
        # print(str(wv))
        return float(volume), unit

    # Clear Infused Volume
    def civolume(self):
        self.write('civolume')

    # Clear Withdrawn Volume
    def cwvolume(self):
        self.write('cwvolume')

    # Clear Target Volume
    def ctvolume(self):
        self.write('ctvolume')

    # Time Related
    # Sets up Target Time
    def ttime(self, time):
        time = str(time)
        cmd = 'ttime ' + time
        self.write(cmd)

    # Infused Time
    def itime(self):
        self.write('wtime')
        resp = (self.read_until('seconds'))
        it = resp.strip('\n00: ')
        # print('infused volume:')
        # print(str(iv))
        return it

    # Withdrawn Time
    def wtime(self):
        self.write('wtime')
        resp = (self.read_until('seconds'))
        wt = resp.strip('\n00: ')
        # print('infused volume:')
        # print(str(iv))
        return wt

    # Clear Infused time
    def citime(self):
        self.write('citime')

    # Clear Withdrawn time
    def cwtime(self):
        self.write('cwtime')

    # Clear Target time
    def cttime(self):
        self.write('cttime')
