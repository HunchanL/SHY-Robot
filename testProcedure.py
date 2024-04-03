import time
import nidaqmx
import numpy as np
import xlsxwriter
import NDI
import continuum
import pumplib


def initialize():
    # initialize nidaq
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan("Dev2/ai9", min_val=-4, max_val=4)
    task.ai_channels.add_ai_voltage_chan("Dev2/ai10", min_val=-4, max_val=4)
    task.ai_channels.add_ai_voltage_chan("Dev2/ai11", min_val=-4, max_val=4)
    task.start()

    # initialize pumps
    # PUMP 1 -- 0.5 mL SYRINGE
    pump1 = pumplib.pump(pumplib.Chain('COM5'), 0, 'pump11')
    tv = 3  # target volume
    pump1.setdiameter(3.256)
    pump1.svolume(0.5, 'ml')
    pump1.tvolume(tv, 'ml')
    pump1.setirate(0.1, 'm/m')
    pump1.setwrate(0.1, 'm/m')
    pump1.poll('remote')

    # PUMP 2 -- 0.5 mL SYRINGE
    pump2 = pumplib.pump(pumplib.Chain('COM6'), 0, 'pump11')
    tv = 3  # target volume
    pump2.setdiameter(3.256)
    pump2.svolume(0.5, 'ml')
    pump2.tvolume(tv, 'ml')
    pump2.setirate(0.1, 'm/m')
    pump2.setwrate(0.1, 'm/m')
    pump2.poll('remote')

    # PUMP 3 -- 1 mL SYRINGE
    pump3 = pumplib.pump(pumplib.Chain('COM7'), 0, 'pump11')
    tv = 3  # target volume
    pump3.setdiameter(4.61)
    pump3.svolume(1, 'ml')
    pump3.tvolume(tv, 'ml')
    pump3.setirate(0.1, 'm/m')
    pump3.setwrate(0.1, 'm/m')
    pump3.poll('remote')

    # initializing Magnetic Tracker
    port = []
    Aurora = NDI.tracker(NDI.Chain('COM4'))
    Aurora.beep(opt=1)
    Aurora.initialize()
    Aurora.comm(5, 0, 0, 0, 0)
    Aurora.port_setup()

    # Sample Data
    Aurora.tstart()
    time.sleep(1)
    sensor_reading = task.read()
    print("Initial Vrms Reading: ")
    print(sensor_reading)

    return task, pump1, pump2, pump3, Aurora


def run_continuum(Aurora, pump1, pump2, pump3, task, T_BR, h_i, sensor_i):
    #
    mdata = []  # mag data
    x_tip_mag = []  # tip tracking
    y_tip_mag = []
    z_tip_mag = []
    vdata = []  # volume data
    unitdata = []  # units of volume
    s_module1 = []  # sensor data
    s_module2 = []
    s_module3 = []
    x_tip_model = []  # predicted tip position based on sensor reading
    y_tip_model = []
    z_tip_model = []

    label = ['Sensor 1 [V]', 'Sensor 2 [V]', 'Sensor 3 [V]', 'x Predicted [mm]', 'y Predicted [mm]', 'z Predicted [mm]',
             'x Actual [mm]', 'y Actual [mm]','z Actual [mm]', 'Error [mm]', 'Distance from Start [mm]']


    # get initial position data
    initial = 0
    pump1.civolume()
    pump2.civolume()
    pump3.civolume()
    t = 0

    #Live plotting
    while t < 65:
        t+=0.1
        T_x_live, T_y_live, T_z_live = NDI.coor_transform_Base2probe(Aurora, T_BR)
        T_live = [-float(T_x_live),
                  float(T_y_live),
                  float(T_z_live)]

        sensor = task.read()

        _, _, _, theta = continuum.bending(sensor[1] / sensor_i[1])
        _, _, _, phi = continuum.roto(sensor[2] / sensor_i[2])
        t1 = continuum.compute_transform("roto", sensor[2] / sensor_i[2])
        t2 = continuum.compute_transform("bend", sensor[1] / sensor_i[1])
        t3 = continuum.compute_transform("trans", sensor[0] / sensor_i[0])

        point0, point1, point2, point3, point4, point5 = continuum.get_transforms(t1, t2, t3, h_i)
        point_tip = point5


        x_tip_mag.append(T_live[1])
        y_tip_mag.append(T_live[0])
        z_tip_mag.append(T_live[2])


        x_tip_model.append(point_tip[0])
        y_tip_model.append(point_tip[1])
        z_tip_model.append(point_tip[2])

        s_module1.append(sensor[2])
        s_module2.append(sensor[1])
        s_module3.append(sensor[0])


        mdata.append(Aurora.tx())
        time.sleep(0.1)

    pump1.stp()
    pump2.stp()
    pump3.stp()


    Aurora.beep(1)

    return label, mdata, s_module1, s_module2, s_module3, x_tip_model, y_tip_model, z_tip_model, x_tip_mag, y_tip_mag, z_tip_mag


def save_data_continuum(current_sheet, label, mdata, s1, s2, s3, tipx, tipy, tipz, mx, my, mz):
    # tip x,y,z -> model data
    # mx,y,z -> actual magnetic tracker data
    # distance calculation

    # MODULE SENSOR DATA
    np_s1 = (np.array(s1)).transpose()
    np_s2 = (np.array(s2)).transpose()
    np_s3 = (np.array(s3)).transpose()
    # TIP PREDICT DATA
    np_tipx = (np.array(tipx)).transpose()
    np_tipy = (np.array(tipy)).transpose()
    np_tipz = (np.array(tipz)).transpose()
    # MAG DATA
    np_mx = (np.array(mx)).transpose()
    np_my = (np.array(my)).transpose()
    np_mz = (np.array(mz)).transpose()

    dist = np.sqrt((np_tipx-np_mx) ** 2 + (np_tipy - np_my) ** 2 + (np_tipz - np_mz) ** 2)
    np_dist = (np.array(dist)).transpose()
    dist_start = np.sqrt((np_mx-np_mx[0]) ** 2 + (np_my-np_my[0]) ** 2 + (np_mz-np_mz[0]) ** 2)
    np_d_s = (np.array(dist_start)).transpose()
    np_pro_data = (np.vstack((np_s1, np_s2, np_s3, np_tipx, np_tipy, np_tipz, np_mx, np_my, np_mz, np_dist, np_d_s))).transpose()
    # label = ['Sensor 1 [V]', 'Sensor 2 [V]', 'Sensor 3 [V]',
    #           'xtip ', 'ytip', 'ztip', 'xmag', 'ymag', 'zmag']  # Sensor Voltages, MagX,MagY, MagZ
    for x in range(len(label)):
        current_sheet.write(0, x, label[x])

    row = 1
    col = 0

    for s_v1, s_v2, s_v3, tx, ty, tz, magtx, magty, magtz, distance, d_s in np_pro_data:

        current_sheet.write(row, col + 0, s_v1)
        current_sheet.write(row, col + 1, s_v2)
        current_sheet.write(row, col + 2, s_v3)
        current_sheet.write(row, col + 3, tx)
        current_sheet.write(row, col + 4, ty)
        current_sheet.write(row, col + 5, tz)
        current_sheet.write(row, col + 6, magtx)
        current_sheet.write(row, col + 7, magty)
        current_sheet.write(row, col + 8, magtz)
        current_sheet.write(row, col + 9, distance)
        current_sheet.write(row, col + 10, d_s)


        row += 1

    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = NDI.data_formatting(mdata)
    NDI.data_save3(current_sheet, Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port)
