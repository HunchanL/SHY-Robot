import serial
import numpy as np
import time
import xlsxwriter
import math
from scipy.spatial.transform import Rotation as R

_baudrate_opt = [9600, 14400, 19200, 38400, 57600, 115200, 921600, 230400]
_databits_opt = [serial.EIGHTBITS, serial.SEVENBITS]
_parity_opt = [serial.PARITY_NONE, serial.PARITY_ODD, serial.PARITY_EVEN]
_stopbits_opt = [serial.STOPBITS_ONE, serial.STOPBITS_TWO]
_handshake_opt = [False, True]


class Chain(serial.Serial):
    # Initializing Host computer serial communication settings
    def __init__(self, port):
        serial.Serial.__init__(self, port=port, baudrate=9600, bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE
                               , parity=serial.PARITY_NONE, timeout=3)
        self.flushOutput()
        self.flushInput()

    # Changing the serial communication settings
    def com_setup(self, bd, db, p, sb):
        self.baudrate = _baudrate_opt[bd]
        self.bytesize = _databits_opt[db]
        self.parity = _parity_opt[p]
        self.stopbits = _stopbits_opt[sb]


# Magnetic Tracker Control Functions
class tracker:
    def __init__(self, chain):
        self.serialcon = chain  # Initializes Host Computer COM Port

    # Serial Write
    def write(self, command):
        self.serialcon.write(command.encode())
        self.serialcon.flushInput()

    # Serial Read
    def read(self):
        response = self.serialcon.read_until(expected='\r'.encode())
        response = response.decode()

        if response == '':
            print('No response received')
        elif 'ERROR' in response:
            print('ERROR')

        self.serialcon.flushOutput()
        return response

    # Changing Magnetic Tracker COM Port Setting
    def comm(self, bd, db, p, sb, hs):
        # self.reset(0)
        cmd = 'COMM ' + str(bd) + str(db) + str(p) + str(sb) + \
              str(hs) + '\r'

        self.write(cmd)
        time.sleep(0.1)
        self.serialcon.com_setup(bd, db, p, sb)
        resp = self.read()
        # print(str(resp))

    # Checks Tracking Console Communication
    def beep(self, opt):
        cmd = 'BEEP ' + str(opt) + '\r'
        self.write(cmd)
        resp = self.read()
        # print(str(resp))

    # Resets to the original setting
    def reset(self, opt):
        cmd = 'RESET ' + str(opt) + '\r'
        self.write(cmd)
        time.sleep(0.1)
        self.serialcon.com_setup(0, 0, 0, 0)
        resp = self.read()
        print(str(resp))


    # Initialing Magnetic Tracker
    def initialize(self):
        cmd = 'INIT \r'
        self.write(cmd)
        resp = self.read()
        print(str(resp))

    # Sets up the ports
    def port_setup(self):
        self.phsr('01')
        self.phsr('02')
        self.phsr('03')
        self.phsr('04')

    # Returns the number of assigned port handles and the port handle status for each one
    # 00 - Reports all allocated port handles (default)
    # 01 - Reports port handles that need to be freed
    # 02 - Reports port handles that are occupied, but not initialized or enabled
    # 03 - Reports port handles that are occupied and initialized, but not enabled
    # 04 - Reports enabled port handles
    def phsr(self, opt):
        ports = []
        cmd = 'PHSR ' + opt + '\r'
        self.write(cmd)
        resp = self.read()

        num_port = int(resp[:2])
        if num_port > 0:
            for i in range(2, 2 + 5 * num_port, 5):
                test = resp[i: (i + 2)]
                ports.append(resp[i: (i + 2)])
        # Reports all allocated port handles
        if opt == '00':
            print('Reporting all allocated port handles')
            print(str(resp))

        # Reports port handles that need to be freed
        elif opt == '01':
            print("Freeing Ports")
            if num_port > 0:
                for i in range(len(ports)):
                    self.phf(ports[i])

            print("Task Complete")

        # Reports port handles that are occupied, but not initialized or enabled
        elif opt == '02':
            print("Initializing Ports")

            if num_port > 0:
                for i in range(len(ports)):
                    self.pinit(ports[i])
                    print('Port ' + str(ports[i]) + 'initialized')

            print("Task Complete")

        # Reports port handles that are occupied and initialized, but not enabled
        elif opt == '03':
            print("Enabling Ports")

            if num_port > 0:
                for i in range(len(ports)):
                    self.pena(ports[i])
                    print('Port ' + str(ports[i]) + ' Enabled')

            print("Task Complete")
        # Reports enabled port handles
        elif opt == '04':
            if num_port == 0:
                print('No port to be freed')
            else:
                for i in range(len(ports)):
                    print('Enabled Port: ' + str(ports[i]))

        print(str(resp))
        return ports

    # Releases system resources from an unused port handle
    def phf(self, port):
        cmd = 'PHF ' + port + '\r'
        self.write(cmd)
        resp = self.read()
        # print(str(resp))

    # Initializes a port handle.
    def pinit(self, port):
        cmd = 'PINIT ' + port + '\r'
        self.write(cmd)
        resp = self.read()
        # print(str(resp))

    # Enables the reporting of transformations for a particular port handle.
    def pena(self, port):
        cmd = 'PENA ' + port + 'S' + '\r'
        self.write(cmd)
        resp = self.read()
        # print(str(resp))

    # Starts Tracking Mode
    def tstart(self):
        # has options
        opt = '00'
        cmd = 'TSTART ' + opt + '\r'
        self.write(cmd)
        resp = self.read()
        # print(str(resp))
        print('Start Tracking')

    # Sets up a configuration for a tool
    def ttcfg(self, port):
        cmd = 'TTCFG ' + port + '\r'
        self.write(cmd)
        resp = self.read()
        # print(str(resp))

    # Stops Tracking Mode
    def tstop(self):
        cmd = 'TSTOP \r'
        self.write(cmd)
        resp = self.read()
        # print(str(resp))

    # Returns information about the tool associated with the port handle
    def phinf(self, port, opt):
        cmd = 'PHINF ' + port + opt + '\r'
        self.write(cmd)
        resp = self.read()
        # print(str(resp))

    # Returns the latest tool transformations and system status information in text format.
    def tx(self):
        cmd = 'TX 0001\r'
        self.write(cmd)
        resp = self.read()
        # print(str(resp))
        return str(resp)


def data_formatting(raw_data):
    Q_0 = np.array([])
    Q_x = np.array([])
    Q_y = np.array([])
    Q_z = np.array([])
    T_x = np.array([])
    T_y = np.array([])
    T_z = np.array([])
    error = np.array([])
    num_frame = len(raw_data)
    num_port = raw_data[0]
    num_port = int(num_port[:2])
    port = ['0A', '0B', '0C', '0D']
    for i in range(len(raw_data)):
        t = raw_data[i]
        t = t[2:]
        t = ''.join(t)
        t = t.split('\n')
        for j in range(len(t) - 1):
            temp_d = t[j]
            # print(temp_d)
            # print(temp_d[2:9])
            check = temp_d[2:9]
            if check == 'MISSING':
                print('NAN')
                Q_0 = np.append(Q_0, 'NAN')
                Q_x = np.append(Q_x, 'NAN')
                Q_y = np.append(Q_y, 'NAN')
                Q_z = np.append(Q_z, 'NAN')
                T_x = np.append(T_x, 'NAN')
                T_y = np.append(T_y, 'NAN')
                T_z = np.append(T_z, 'NAN')
                error = np.append(error, 'NAN')
            else:
                port = np.append(port, temp_d[:2])
                Q_0 = np.append(Q_0, temp_d[2:8])
                Q_x = np.append(Q_x, temp_d[8:14])
                Q_y = np.append(Q_y, temp_d[14:20])
                Q_z = np.append(Q_z, temp_d[20:26])
                T_x = np.append(T_x, temp_d[26:33])
                T_y = np.append(T_y, temp_d[33:40])
                T_z = np.append(T_z, temp_d[40:47])
                error = np.append(error, temp_d[47:53])
    port = port[:num_port]
    return Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, num_frame, port


def rotation_format(Q):
    for i in range(len(Q)):
        q_0 = Q[i]
        q_0 = q_0.tolist()
        q_0 = q_0.replace('+', '')
        Q[i] = float(q_0) / 10000
    return Q


def translation_format(T):
    for i in range(len(T)):
        t_0 = T[i]
        t_0 = t_0.tolist()
        t_0 = t_0.replace('+', '')
        T[i] = float(t_0) / 100
    return T


def data_save(Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, e, n_frame, port):
    n_port = len(port)
    Q_0 = rotation_format(Q_0)
    Q_x = rotation_format(Q_x)
    Q_y = rotation_format(Q_y)
    Q_z = rotation_format(Q_z)
    T_x = translation_format(T_x)
    T_y = translation_format(T_y)
    T_z = translation_format(T_z)
    # e = rotation_format(e)

    Q_0 = Q_0.reshape((n_frame, n_port))
    Q_x = Q_x.reshape((n_frame, n_port))
    Q_y = Q_y.reshape((n_frame, n_port))
    Q_z = Q_z.reshape((n_frame, n_port))
    T_x = T_x.reshape((n_frame, n_port))
    T_y = T_y.reshape((n_frame, n_port))
    T_z = T_z.reshape((n_frame, n_port))

    workbook = xlsxwriter.Workbook('magdata.xlsx')
    worksheet = workbook.add_worksheet()

    label = ['Frame', 'error']

    for i in range(len(label)):
        worksheet.write(0, i, label[i])
    row = 1
    for i in range(n_frame):
        worksheet.write(i + 1, 0, i)
        worksheet.write(i + 1, 1, e[i])
        row += 1
    label = ['Port', 'Q_0', 'Q_x', 'Q_y', 'Q_z', 'T_x', 'T_y', 'T_z']
    for j in range(n_port):
        row = 1
        worksheet.write(1, (8 * j) + 2, port[j])
        for i in range(len(label)):
            worksheet.write(0, (8 * j) + i + 2, label[i])

        q_0 = Q_0[..., j]
        q_x = Q_x[..., j]
        q_y = Q_y[..., j]
        q_z = Q_z[..., j]
        t_x = T_x[..., j]
        t_y = T_y[..., j]
        t_z = T_z[..., j]
        m = (np.vstack((q_0, q_x, q_y, q_z, t_x, t_y, t_z))).transpose()

        for q, qx, qy, qz, tx, ty, tz in m:
            worksheet.write(row, (8 * j) + 3, q)
            worksheet.write(row, (8 * j) + 4, qx)
            worksheet.write(row, (8 * j) + 5, qy)
            worksheet.write(row, (8 * j) + 6, qz)
            worksheet.write(row, (8 * j) + 7, tx)
            worksheet.write(row, (8 * j) + 8, ty)
            worksheet.write(row, (8 * j) + 9, tz)
            row += 1
    workbook.close()

    return Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, e


def data_save2(Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, e, n_frame, port, wsheet):
    n_port = len(port)
    Q_0 = rotation_format(Q_0)
    Q_x = rotation_format(Q_x)
    Q_y = rotation_format(Q_y)
    Q_z = rotation_format(Q_z)
    T_x = translation_format(T_x)
    T_y = translation_format(T_y)
    T_z = translation_format(T_z)
    # e = rotation_format(e)

    Q_0 = Q_0.reshape((n_frame, n_port))
    Q_x = Q_x.reshape((n_frame, n_port))
    Q_y = Q_y.reshape((n_frame, n_port))
    Q_z = Q_z.reshape((n_frame, n_port))
    T_x = T_x.reshape((n_frame, n_port))
    T_y = T_y.reshape((n_frame, n_port))
    T_z = T_z.reshape((n_frame, n_port))

    label = ['Frame', 'error']

    for i in range(len(label)):
        wsheet.write(0, i, label[i])
    row = 1
    for i in range(n_frame):
        wsheet.write(i + 1, 0, i)
        wsheet.write(i + 1, 1, e[i])
        row += 1
    label = ['Port', 'Q_0', 'Q_x', 'Q_y', 'Q_z', 'T_x', 'T_y', 'T_z']
    for j in range(n_port):
        row = 1
        wsheet.write(1, (8 * j) + 2, port[j])
        for i in range(len(label)):
            wsheet.write(0, (8 * j) + i + 2, label[i])

        q_0 = Q_0[..., j]
        q_x = Q_x[..., j]
        q_y = Q_y[..., j]
        q_z = Q_z[..., j]
        t_x = T_x[..., j]
        t_y = T_y[..., j]
        t_z = T_z[..., j]
        m = (np.vstack((q_0, q_x, q_y, q_z, t_x, t_y, t_z))).transpose()

        for q, qx, qy, qz, tx, ty, tz in m:
            wsheet.write(row, (8 * j) + 3, q)
            wsheet.write(row, (8 * j) + 4, qx)
            wsheet.write(row, (8 * j) + 5, qy)
            wsheet.write(row, (8 * j) + 6, qz)
            wsheet.write(row, (8 * j) + 7, tx)
            wsheet.write(row, (8 * j) + 8, ty)
            wsheet.write(row, (8 * j) + 9, tz)
            row += 1

    return Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, e


def get_height(Aurora):
    data = [Aurora.tx()]
    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = data_formatting(data)
    T_x = translation_format(T_x)
    return float(T_x[1])


def get_initial_translations(Aurora):
    data = [Aurora.tx()]
    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = data_formatting(data)
    T_x0 = translation_format(T_x)
    T_y0 = translation_format(T_y)
    T_z0 = translation_format(T_z)
    Q_00 = rotation_format(Q_0)
    Q_x0 = rotation_format(Q_x)
    Q_y0 = rotation_format(Q_y)
    Q_z0 = rotation_format(Q_z)

    return T_x0, T_y0, T_z0, Q_00, Q_x0, Q_y0, Q_z0


def get_angle(r, T_x0, T_y0, T_z0, Aurora):  # For bending actuator
    data = [Aurora.tx()]
    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = data_formatting(data)
    T_x = translation_format(T_x)
    T_y = translation_format(T_y)
    T_z = translation_format(T_z)
    # Changing from absolute coordinate to relative coordinate
    # Make sure to bending toward the magnetic generator
    r_Tx = float(T_x[1]) - float(T_x0[1])  # Extension
    r_Tz = float(T_z[1]) - float(T_z0[1])  # Bending toward tracker
    theta = math.degrees(math.atan2(abs(r_Tx), r - abs(r_Tz)))
    return theta


def coor_transform_Ref2probe(Aurora):
    # Reference to Probe Coordinate Transformation
    # Output: Tx, Ty, Tz values in Ref coordinate
    data = [Aurora.tx()]
    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = data_formatting(data)
    T_x = translation_format(T_x)
    T_y = translation_format(T_y)
    T_z = translation_format(T_z)
    Q_0 = rotation_format(Q_0)
    Q_x = rotation_format(Q_x)
    Q_y = rotation_format(Q_y)
    Q_z = rotation_format(Q_z)

    Q_0 = Q_0.reshape((1, 2))
    Q_x = Q_x.reshape((1, 2))
    Q_y = Q_y.reshape((1, 2))
    Q_z = Q_z.reshape((1, 2))
    T_x = T_x.reshape((1, 2))
    T_y = T_y.reshape((1, 2))
    T_z = T_z.reshape((1, 2))

    Q_0 = Q_0.astype(float)
    Q_x = Q_x.astype(float)
    Q_y = Q_y.astype(float)
    Q_z = Q_z.astype(float)
    T_x = T_x.astype(float)
    T_y = T_y.astype(float)
    T_z = T_z.astype(float)
    # assuming 1st port = ref, 2nd port = probe
    # T_GP =  from Mag Generator coordinate to probe coordinate
    # T_GR =  from Mag Generator coordinate to ref coordinate
    # T_RG =  from ref coordinate to Mag Generator coordinate
    R_GR = R.from_quat([Q_x[0, 0], Q_y[0, 0], Q_z[0, 0], Q_0[0, 0]])
    R_GP = R.from_quat([Q_x[0, 1], Q_y[0, 1], Q_z[0, 1], Q_0[0, 1]])
    R_GR = R_GR.as_matrix()  # Rotation matrix - Reference probe
    R_GP = R_GP.as_matrix()  # Rotation matrix - Moving probe

    trans_GR = np.array([T_x[0, 0], T_y[0, 0], T_z[0, 0]])  # translation vector - Reference probe
    trans_GP = np.array([T_x[0, 1], T_y[0, 1], T_z[0, 1]])  # translation vector - Moving probe

    T_GR = np.vstack((np.hstack((R_GR, trans_GR[:, None])), [0, 0, 0, 1]))
    T_GP = np.vstack((np.hstack((R_GP, trans_GP[:, None])), [0, 0, 0, 1]))
    T_RG = np.linalg.inv(T_GR)

    T_RP = np.matmul(T_RG, T_GP)
    Tx_RP = T_RP[0, 3]
    Ty_RP = T_RP[1, 3]
    Tz_RP = T_RP[2, 3]

    return T_RP  # Tx_RP, Ty_RP, Tz_RP


def coor_transform_Base2Ref(Aurora):
    # Coordinate transformation from the base of continuum robot to Ref
    # Uses the initial X,Y position of probe tracker to define the base of continuum robot
    data = [Aurora.tx()]
    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = data_formatting(data)
    T_x = translation_format(T_x)
    T_y = translation_format(T_y)
    T_z = translation_format(T_z)
    Q_0 = rotation_format(Q_0)
    Q_x = rotation_format(Q_x)
    Q_y = rotation_format(Q_y)
    Q_z = rotation_format(Q_z)

    Q_0 = Q_0.reshape((1, 2))
    Q_x = Q_x.reshape((1, 2))
    Q_y = Q_y.reshape((1, 2))
    Q_z = Q_z.reshape((1, 2))
    T_x = T_x.reshape((1, 2))
    T_y = T_y.reshape((1, 2))
    T_z = T_z.reshape((1, 2))

    Q_0 = Q_0.astype(float)
    Q_x = Q_x.astype(float)
    Q_y = Q_y.astype(float)
    Q_z = Q_z.astype(float)
    T_x = T_x.astype(float)
    T_y = T_y.astype(float)
    T_z = T_z.astype(float)
    # assuming 1st port = ref, 2nd port = probe
    # T_GP =  from Mag Generator coordinate to probe coordinate
    # T_GR =  from Mag Generator coordinate to ref coordinate
    # T_RG =  from ref coordinate to Mag Generator coordinate
    # T_RP =  from ref coordinate to probe coordinate
    # T_BR =  from base coordinate to ref coordinate
    # T_RB =  from  coordinate to ref coordinate
    R_GR = R.from_quat([Q_x[0, 0], Q_y[0, 0], Q_z[0, 0], Q_0[0, 0]])
    R_GP = R.from_quat([Q_x[0, 1], Q_y[0, 1], Q_z[0, 1], Q_0[0, 1]])
    R_GR = R_GR.as_matrix()  # Rotation matrix - Reference probe
    R_GP = R_GP.as_matrix()  # Rotation matrix - Moving probe

    trans_GR = np.array([T_x[0, 0], T_y[0, 0], T_z[0, 0]])  # translation vector - Reference probe
    trans_GP = np.array([T_x[0, 1], T_y[0, 1], T_z[0, 1]])  # translation vector - Moving probe

    T_GR = np.vstack((np.hstack((R_GR, trans_GR[:, None])), [0, 0, 0, 1]))
    T_GP = np.vstack((np.hstack((R_GP, trans_GP[:, None])), [0, 0, 0, 1]))
    T_RG = np.linalg.inv(T_GR)

    T_RP = np.matmul(T_RG, T_GP)

    T_RB = np.array([[1, 0, 0, T_RP[0, 3]], [0, 1, 0, T_RP[1, 3]], [0, 0, 1, 0], [0, 0, 0, 1]])
    T_BR = np.linalg.inv(T_RB)

    # Tx_BR = T_BR[0,3]
    # Ty_BR = T_BR[1,3]
    # Tz_BR = T_BR[2,3]

    return T_BR, T_RP

    # T_RB = np.array([[1,0,0,-33],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    # T_BR = np.linalg.inv(T_RB)
    #
    # Tx_BR = T_BR[0,3]
    # Ty_BR = T_BR[1,3]
    # Tz_BR = T_BR[2,3]
    #
    # return T_BR


def coor_transform_Base2probe(Aurora, T_BR):
    # Base to Probe Coordinate Transformation
    # Base is the base of the continuum robot
    # Input: Probe data (Aurora), Base2Ref coordinate transformation
    # Output: Tx, Ty, Tz of probe in base coordinate
    data = [Aurora.tx()]
    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = data_formatting(data)
    T_x = translation_format(T_x)
    T_y = translation_format(T_y)
    T_z = translation_format(T_z)
    Q_0 = rotation_format(Q_0)
    Q_x = rotation_format(Q_x)
    Q_y = rotation_format(Q_y)
    Q_z = rotation_format(Q_z)

    Q_0 = Q_0.reshape((1, 2))
    Q_x = Q_x.reshape((1, 2))
    Q_y = Q_y.reshape((1, 2))
    Q_z = Q_z.reshape((1, 2))
    T_x = T_x.reshape((1, 2))
    T_y = T_y.reshape((1, 2))
    T_z = T_z.reshape((1, 2))

    Q_0 = Q_0.astype(float)
    Q_x = Q_x.astype(float)
    Q_y = Q_y.astype(float)
    Q_z = Q_z.astype(float)
    T_x = T_x.astype(float)
    T_y = T_y.astype(float)
    T_z = T_z.astype(float)
    # assuming 1st port = ref, 2nd port = probe
    # T_GP =  from Mag Generator coordinate to probe coordinate
    # T_GR =  from Mag Generator coordinate to ref coordinate
    # T_RG =  from ref coordinate to Mag Generator coordinate
    # T_RP =  from ref coordinate to probe coordinate
    # T_BR =  from base coordinate to ref coordinate
    # T_RB =  from  coordinate to ref coordinate
    R_GR = R.from_quat([Q_x[0, 0], Q_y[0, 0], Q_z[0, 0], Q_0[0, 0]])
    R_GP = R.from_quat([Q_x[0, 1], Q_y[0, 1], Q_z[0, 1], Q_0[0, 1]])
    R_GR = R_GR.as_matrix()  # Rotation matrix - Reference probe
    R_GP = R_GP.as_matrix()  # Rotation matrix - Moving probe

    trans_GR = np.array([T_x[0, 0], T_y[0, 0], T_z[0, 0]])  # translation vector - Reference probe
    trans_GP = np.array([T_x[0, 1], T_y[0, 1], T_z[0, 1]])  # translation vector - Moving probe

    T_GR = np.vstack((np.hstack((R_GR, trans_GR[:, None])), [0, 0, 0, 1]))
    T_GP = np.vstack((np.hstack((R_GP, trans_GP[:, None])), [0, 0, 0, 1]))
    T_RG = np.linalg.inv(T_GR)

    T_RP = np.matmul(T_RG, T_GP)
    T_BP = np.matmul(T_BR, T_RP)

    Tx_BP = T_BP[0, 3]
    Ty_BP = T_BP[1, 3]
    Tz_BP = T_BP[2, 3]

    return Tx_BP, Ty_BP, Tz_BP


def data_save3(current_sheet, Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, e, n_frame, port):
    n_port = len(port)
    Q_0 = rotation_format(Q_0)
    Q_x = rotation_format(Q_x)
    Q_y = rotation_format(Q_y)
    Q_z = rotation_format(Q_z)
    T_x = translation_format(T_x)
    T_y = translation_format(T_y)
    T_z = translation_format(T_z)
    # e = rotation_format(e)

    Q_0 = Q_0.reshape((n_frame, n_port))
    Q_x = Q_x.reshape((n_frame, n_port))
    Q_y = Q_y.reshape((n_frame, n_port))
    Q_z = Q_z.reshape((n_frame, n_port))
    T_x = T_x.reshape((n_frame, n_port))
    T_y = T_y.reshape((n_frame, n_port))
    T_z = T_z.reshape((n_frame, n_port))

    label = ['Frame', 'error']

    for i in range(len(label)):
        current_sheet.write(0, i + 12, label[i])
    row = 1
    for i in range(n_frame):
        current_sheet.write(i + 1, 12, i)
        current_sheet.write(i + 1, 13, e[i])
        row += 1
    label = ['Port', 'Q_0', 'Q_x', 'Q_y', 'Q_z', 'T_x', 'T_y', 'T_z']
    for j in range(n_port):
        row = 1
        current_sheet.write(1, (8 * j) + 14, port[j])
        for i in range(len(label)):
            current_sheet.write(0, (8 * j) + i + 14, label[i])

        q_0 = Q_0[..., j]
        q_x = Q_x[..., j]
        q_y = Q_y[..., j]
        q_z = Q_z[..., j]
        t_x = T_x[..., j]
        t_y = T_y[..., j]
        t_z = T_z[..., j]
        m = (np.vstack((q_0, q_x, q_y, q_z, t_x, t_y, t_z))).transpose()

        for q, qx, qy, qz, tx, ty, tz in m:
            current_sheet.write(row, (8 * j) + 15, q)
            current_sheet.write(row, (8 * j) + 16, qx)
            current_sheet.write(row, (8 * j) + 17, qy)
            current_sheet.write(row, (8 * j) + 18, qz)
            current_sheet.write(row, (8 * j) + 19, tx)
            current_sheet.write(row, (8 * j) + 20, ty)
            current_sheet.write(row, (8 * j) + 21, tz)
            row += 1
