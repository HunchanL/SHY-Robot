import time
import nidaqmx
import numpy as np
import xlsxwriter
import NDI
import pumplib

def initialize():
    # initialize nidaq
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan("Dev1/ai9", min_val=-4, max_val=4)
    task.ai_channels.add_ai_voltage_chan("Dev1/ai10", min_val=-4, max_val=4)
    task.ai_channels.add_ai_voltage_chan("Dev1/ai11", min_val=-4, max_val=4)
    task.start()

    # initialize pumps
    # PUMP 1 -- 0.5 mL SYRINGE
    pump1 = pumplib.pump(pumplib.Chain('COM16'), 0, 'pump11')
    tv = 3  # target volume
    pump1.setdiameter(3.256)
    pump1.svolume(0.5, 'ml')
    pump1.tvolume(tv, 'ml')
    pump1.setirate(0.1, 'm/m')
    pump1.setwrate(0.1, 'm/m')
    pump1.poll('remote')

    # PUMP 2 -- 0.5 mL SYRINGE
    pump2 = pumplib.pump(pumplib.Chain('COM17'), 0, 'pump11')
    tv = 3  # target volume
    pump2.setdiameter(3.256)
    pump2.svolume(0.5, 'ml')
    pump2.tvolume(tv, 'ml')
    pump2.setirate(0.1, 'm/m')
    pump2.setwrate(0.1, 'm/m')
    pump2.poll('remote')

    # PUMP 3 -- 1 mL SYRINGE
    pump3 = pumplib.pump(pumplib.Chain('COM18'), 0, 'pump11')
    tv = 3  # target volume
    pump3.setdiameter(4.61)
    pump3.svolume(1, 'ml')
    pump3.tvolume(tv, 'ml')
    pump3.setirate(0.1, 'm/m')
    pump3.setwrate(0.1, 'm/m')
    pump3.poll('remote')

    # initializing Magnetic Tracker
    port = []
    Aurora = NDI.tracker(NDI.Chain('COM20'))
    Aurora.beep(opt=1)
    Aurora.initialize()
    Aurora.comm(5, 0, 0, 0, 0)
    Aurora.port_setup()

    # Sample Data
    Aurora.tstart()
    sensor_reading = task.read()
    print("Initial Vrms Reading: ")
    print(sensor_reading)

    return task, pump1, pump2, pump3, Aurora


def run_test_linear(Aurora, pump1, task,h_initial):
    mdata = []  # mag data
    vdata = []  # volume data
    unitdata = []  # units of volume
    sdata = []  # sensor data
    hdata = []  # height data
    t = []
    label = ['Volume', 'Unit', 'Voltage', 'Extension']  # Voltage, MagX,MagY, MagZ


    pump1.civolume()
    h_threshold = h_initial + 0.1
    h_step = 0.1
    max_h = h_initial + 5 # controls the maximum height that an actuator can go to
    min_h = h_initial + 0.1
    max_v = 0

    h = h_initial
    while h <= max_h:
        h = NDI.get_height(Aurora)
        pump1.irun()
        while h <= h_threshold:
            h = NDI.get_height(Aurora)

            if h > h_threshold:
                pump1.stp()
                v_sensor = task.read()
                mdata.append(Aurora.tx())
                height = h - h_initial
                hdata.append(height)
                # print('RMS voltage :')
                # print(v_sensor)
                iv, iu = pump1.ivolume()
                vdata.append(iv)
                unitdata.append(iu)
                sdata.append(v_sensor)
                break
            time.sleep(0.5)
        print('Height: ' + str(h - h_initial) + 'Voltage: '+ str(v_sensor))
        h_threshold += h_step

    pump1.stp
    in_size = len(vdata)
    Max_vdata = iv
    time.sleep(1)
    print('Retracting...')

    h_threshold -= h_step
    while h >= min_h:
        pump1.wrun()
        h = NDI.get_height(Aurora)
        while h >= h_threshold:
            h = NDI.get_height(Aurora)
            if h < h_threshold:
                pump1.stp()
                v_sensor = task.read()
                height = h - h_initial
                hdata.append(height)
                mdata.append(Aurora.tx())
                wv, wu = pump1.wvolume()
                vdata.append(wv)
                unitdata.append(wu)
                sdata.append(v_sensor)
                break
        time.sleep(0.5)
        h_threshold -= h_step
        print('Height: ' + str(h - h_initial) + 'Voltage: ' + str(v_sensor))
    pump1.stp

    print("RESETING h_i")
    #pump1.wrun()
    time.sleep(2)
    #pump1.stp()

    Aurora.beep(1)

    return vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, hdata


def run_test_bending(Aurora, pump1, task, T_x_i, T_y_i, T_z_i):
    mdata = []  # mag data
    vdata = []  # volume data
    unitdata = []  # units of volume
    sdata = []  # sensor data
    th_data = []  # bending angle
    t = []
    label = ['Volume', 'Unit', 'Voltage', 'Bending Angle']  # Voltage, MagX,MagY, MagZ

    theta_initial = 0
    pump1.civolume()
    theta_step = 2
    theta_threshold = theta_initial + theta_step
    max_theta = theta_initial + 46 # controls maximum bending angle
    min_theta = theta_initial + 1
    max_v = 0

    theta = theta_initial
    while theta <= max_theta:
        r = 4.8
        theta = NDI.get_angle(r, T_x_i, T_y_i, T_z_i, Aurora)
        pump1.irun()
        while theta <= theta_threshold:
            theta = NDI.get_angle(r, T_x_i, T_y_i, T_z_i, Aurora)
            if theta >= theta_threshold:
                pump1.stp()
                v_sensor = task.read()
                mdata.append(Aurora.tx())
                th_data.append(theta)
                iv, iu = pump1.ivolume()
                vdata.append(iv)
                unitdata.append(iu)
                sdata.append(v_sensor)
                break
        print('theta: ' + str(theta) + ' voltage:' + str(v_sensor))
        theta_threshold += theta_step

    pump1.stp()
    in_size = len(vdata)
    Max_vdata = iv
    time.sleep(1)
    print('Retracting...')

    theta_threshold -= theta_step
    while theta >= min_theta:
        pump1.wrun()
        theta = NDI.get_angle(r, T_x_i, T_y_i, T_z_i, Aurora)
        while theta >= theta_threshold:
            theta = NDI.get_angle(r, T_x_i, T_y_i, T_z_i, Aurora)
            if theta <= theta_threshold:
                pump1.stp()
                v_sensor = task.read()
                mdata.append(Aurora.tx())
                th_data.append(theta)
                wv, wu = pump1.wvolume()
                vdata.append(wv)
                unitdata.append(wu)
                sdata.append(v_sensor)
                break
        theta_threshold -= theta_step
        print('theta: ' + str(theta) + ' voltage:' + str(v_sensor))
    pump1.stp()

    print("RESETING THETA_i")
    pump1.wrun()
    time.sleep(2)
    pump1.stp()

    Aurora.beep(1)

    return vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, th_data


def run_test_roto(Aurora, pump1, task, h_initial):
    mdata = []  # mag data
    vdata = []  # volume data
    unitdata = []  # units of volume
    sdata = []  # sensor data
    hdata = []  # height data
    t = []
    label = ['Volume', 'Unit', 'Voltage', 'Extension']  # Voltage, MagX,MagY, MagZ


    pump1.civolume()
    h_threshold = h_initial + 0.2
    h_step = 0.2

    max_h = h_initial + 5.2
    min_h = h_initial + 0.1
    max_v = 0

    h = h_initial
    v_sensor = task.read()
    mdata.append(Aurora.tx())
    height = 0
    hdata.append(height)
    vdata.append(0)
    unitdata.append('pl')
    sdata.append(v_sensor)
    while h <= max_h:
        h = NDI.get_height(Aurora)
        pump1.irun()
        while h <= h_threshold:
            h = NDI.get_height(Aurora)
            if h > h_threshold:
                pump1.stp()
                v_sensor = task.read()
                mdata.append(Aurora.tx())
                height = h - h_initial
                hdata.append(height)
                iv, iu = pump1.ivolume()
                vdata.append(iv)
                unitdata.append(iu)
                sdata.append(v_sensor)
                break
        print('Height: ' + str(h - h_initial) + 'Voltage: ' + str(v_sensor))
        h_threshold += h_step

    pump1.stp
    in_size = len(vdata)
    Max_vdata = iv
    time.sleep(1)
    print('Retracting...')

    h_threshold -= h_step
    while h >= min_h:
        pump1.wrun()
        h = NDI.get_height(Aurora)
        while h >= h_threshold:
            h = NDI.get_height(Aurora)
            if h < h_threshold:
                pump1.stp()
                v_sensor = task.read()
                height = h - h_initial
                hdata.append(height)
                mdata.append(Aurora.tx())
                wv, wu = pump1.wvolume()
                vdata.append(wv)
                unitdata.append(wu)
                sdata.append(v_sensor)
                break
        h_threshold -= h_step
        print('Height: ' + str(h - h_initial) + 'Voltage: ' + str(v_sensor))
    pump1.stp

    print("RESETING h_i")
    pump1.wrun()
    time.sleep(2)
    pump1.stp()

    Aurora.beep(1)

    return vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, hdata

def save_data_module(current_sheet, vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, hdata):
    # For data saving controlling over the height variable (translational and rototranslational)
    np_vdata = (np.array(vdata)).transpose()
    np_sdata = (np.array(sdata)).transpose()
    np_hdata = (np.array(hdata)).transpose()

    iv_data = np_vdata[0:in_size]
    wv_data = Max_vdata - np_vdata[in_size:]
    np_pro_volume = (np.hstack((iv_data, wv_data))).transpose()

    np_unitdata = (np.array(unitdata)).transpose()
    np_data = (np.vstack((np_vdata, np_unitdata, np_sdata, np_hdata))).transpose()
    np_pro_data = (np.vstack((np_pro_volume, np_unitdata, np_sdata, np_hdata))).transpose()

    for x in range(len(label)):
        current_sheet.write(0, x, label[x])

    row = 1
    col = 0

    for v_ul, unit, s_v, h_d in np_pro_data:
        current_sheet.write(row, col, v_ul)
        current_sheet.write(row, col + 1, unit)
        current_sheet.write(row, col + 2, s_v)
        current_sheet.write(row, col + 3, h_d)
        row += 1

    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = NDI.data_formatting(mdata)
    NDI.data_save3(current_sheet, Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port)


def save_data_bending(current_sheet, vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, th_data):
    np_vdata = (np.array(vdata)).transpose()
    np_sdata = (np.array(sdata)).transpose()
    np_th_data = (np.array(th_data)).transpose()

    iv_data = np_vdata[0:in_size]
    wv_data = Max_vdata - np_vdata[in_size:]
    np_pro_volume = (np.hstack((iv_data, wv_data))).transpose()

    np_unitdata = (np.array(unitdata)).transpose()
    np_data = (np.vstack((np_vdata, np_unitdata, np_sdata, np_th_data))).transpose()
    np_pro_data = (np.vstack((np_pro_volume, np_unitdata, np_sdata, np_th_data))).transpose()

    for x in range(len(label)):
        current_sheet.write(0, x, label[x])

    row = 1
    col = 0

    for v_ul, unit, s_v, th_d in np_pro_data:
        current_sheet.write(row, col, v_ul)
        current_sheet.write(row, col + 1, unit)
        current_sheet.write(row, col + 2, s_v)
        current_sheet.write(row, col + 3, th_d)
        row += 1

    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = NDI.data_formatting(mdata)
    NDI.data_save3(current_sheet, Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port)
