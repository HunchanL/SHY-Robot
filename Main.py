import nidaqmx
import nidaqmx.constants as nconst
import pumplib
import time
import xlsxwriter
import NDI
import test_procedure

# Main script for testing individual PCA modules

# Calls upon functions from test_procedure.py, NDI.py, pumplib.py, and nidaqmx library

# Set up using the Aurora magnetic tracker, one Harvard pump, and the one NIDAQ analog port
# for measuring using the LTC 1968 module RMS to DC converter chip.

# Set number of samples desired, and the script will run that number of test for the module
# then save the data in different sheets of the same Excel file


# initialize sensor
# task = nidaqmx.Task()
# task.ai_channels.add_ai_voltage_chan("Dev2/ai9", min_val=-10, max_val=10)
# #task.ai_channels.add_ai_voltage_chan("Dev1/ai2", min_val=-4, max_val=4)
# #task.ai_channels.add_ai_voltage_chan("Dev1/ai10", min_val=-4, max_val=4)
# task.start()
#
# # pump1 = pumplib.pump(pumplib.Chain('/dev/tty.usbmodemD3088131'), 0, 'pump11')
#
# print("INITIALIZED")
#
#
# print('init. complete')
#
# print("sensor data[Vrms]: ")
# v_sensor = task.read()
#
# print(v_sensor)
#
# while(1):
#     v_sensor = task.read()
#     time.sleep(1)
#     print(v_sensor)


sensor_num = "linear_5_240319"
workbook = xlsxwriter.Workbook(sensor_num + '.xlsx')

sheet0 = workbook.add_worksheet()
sheet1 = workbook.add_worksheet()
sheet2 = workbook.add_worksheet()
sheet3 = workbook.add_worksheet()
sheet4 = workbook.add_worksheet()

worksheet = [sheet0, sheet1, sheet2, sheet3, sheet4]

# initialize nidaq
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev2/ai9", min_val=-4, max_val=4, terminal_config = nconst.TerminalConfiguration.RSE) # ROTO
#task.ai_channels.add_ai_voltage_chan("Dev2/ai10", min_val=-4, max_val=4, terminal_config = nconst.TerminalConfiguration.RSE) # BENDING
# task.ai_channels.add_ai_voltage_chan("Dev2/ai9", min_val=-4, max_val=4, terminal_config=nconst.TerminalConfiguration.RSE)  # LINEAR
task.start()

# initialize pump
pump1 = pumplib.pump(pumplib.Chain('COM10'), 0, 'pump11')
tv = 3  # target volume
pump1.setdiameter(3.256)
pump1.svolume(0.5, 'ml')
pump1.tvolume(tv, 'ml')
pump1.setirate(0.05, 'm/m')
pump1.setwrate(0.1, 'm/m')
pump1.poll('remote')

# initializing Magnetic Tracker
port = []
Aurora = NDI.tracker(NDI.Chain('COM5'))
Aurora.beep(opt=1)
Aurora.initialize()
Aurora.comm(5, 0, 0, 0, 0)
Aurora.port_setup()

# Sample Data
Aurora.tstart()
time.sleep(1)
sensor_reading = task.read()

# For Translation & Roto
initial_data = [Aurora.tx()]
Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = NDI.data_formatting(initial_data)
T_x_i, T_y_i, T_z_i = NDI.translation_format(T_x), \
                      NDI.translation_format(T_y), \
                      NDI.translation_format(T_z)
h_initial = float(T_x_i[1])
print("Initial Vrms Reading: ")
print(sensor_reading)
print("Initial h_initial Reading: ")
print(h_initial)

# For Bending
# initial_data = [Aurora.tx()]
# Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = NDI.data_formatting(initial_data)
# T_x_i, T_y_i, T_z_i = NDI.translation_format(T_x), \
#                       NDI.translation_format(T_y), \
#                       NDI.translation_format(T_z)
# theta_initial = NDI.get_angle(1.9, T_x_i, T_y_i, T_z_i, Aurora)
# print("Initial Vrms Reading: ")
# print(sensor_reading)
# print("Initial theta_initial Reading: ")
# print(theta_initial)

samples = 1  # Set number of samples
print('Level pressure and begin' + str(samples) + 'tests? [y/n]')
response = input()
while response == 'n':
    #For Translation & Roto
    for i in range(1):
        sensor_reading = task.read()
        initial_data = [Aurora.tx()]
        Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = NDI.data_formatting(initial_data)
        T_x_i, T_y_i, T_z_i = NDI.translation_format(T_x), \
                          NDI.translation_format(T_y), \
                          NDI.translation_format(T_z)
        h_initial = float(T_x_i[1])
        print("Initial Vrms Reading: ")
        print(sensor_reading)
        print("Initial h_initial Reading: ")
        print(h_initial)
        time.sleep(1)

    #For Bending
    # for i in range (1):
    #     sensor_reading = task.read()
    #     initial_data = [Aurora.tx()]
    #     Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = NDI.data_formatting(initial_data)
    #     T_x_i, T_y_i, T_z_i = NDI.translation_format(T_x), \
    #                       NDI.translation_format(T_y), \
    #                       NDI.translation_format(T_z)
    # # h_initial = float(T_x_i[1])
    #     theta_initial = NDI.get_angle(1.9, T_x_i, T_y_i, T_z_i, Aurora)
    #     print("Initial Vrms Reading: ")
    #     print(sensor_reading)
    #     print("Initial T_x_i Reading: "+str(T_x_i[1])+" Initial T_y_i Reading: "+str(T_y_i[1]))
    #     time.sleep(1)

    print('Level pressure and begin' + str(samples) + 'tests? [y/n]')
    response = input()

if response == 'y':

    for sample in range(samples):
        print('Test' + str(sample) + 'ready \r\n')
        current_sheet = worksheet[sample]

        # # TRANSLATION MODULE TEST DATA
        # vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, hdata = test_procedure.run_test_linear(Aurora, pump1, task, h_initial)
        # # TRANSLATION DATA SAVE
        # test_procedure.save_data_module(current_sheet, vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, hdata)

        # BENDING MODULE TEST DATA
        # vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, th_data = test_procedure.run_test_bending(Aurora,pump1, task, T_x_i, T_y_i, T_z_i)
        # BENDING DATA SAVE
        # test_procedure.save_data_bending(current_sheet, vdata, sdata, unitdata, label, in_size, Max_vdata, mdata,th_data)

        # ROTO MODULE TEST DATA
        vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, hdata = test_procedure.run_test_roto(Aurora, pump1, task, h_initial)
        # # ROTO DATA SAVE
        test_procedure.save_data_module(current_sheet, vdata, sdata, unitdata, label, in_size, Max_vdata, mdata, hdata)

    pump1.stp()
    Aurora.tstop()
    Aurora.reset(0)
    # RCSensor.close_port()
    task.close()
    workbook.close()

elif response == 'a':

    T_x0, T_y0, T_z0, Q_00, Q_x0, Q_y0, Q_z0 = NDI.get_initial_translations(Aurora)
    # theta = NDI.get_angle(5.1,T_x0,T_y0,T_z0,Aurora)
    # theta = NDI.get_angle(5.1, T_x0, T_y0, T_z0, Aurora)
    # theta = NDI.get_angle(5.1, T_x0, T_y0, T_z0, Aurora)
    # theta = NDI.get_angle(5.1, T_x0, T_y0, T_z0, Aurora)
    Tx_RP, Ty_RP, Tz_RP = NDI.coordinate_transform(Aurora)

    print(Tx_RP)
    print(Ty_RP)
    print(Tz_RP)
    time.sleep(3)
    Tx_RP, Ty_RP, Tz_RP = NDI.coordinate_transform(Aurora)

    print(Tx_RP)
    print(Ty_RP)
    print(Tz_RP)
    time.sleep(3)
    Tx_RP, Ty_RP, Tz_RP = NDI.coordinate_transform(Aurora)

    print(Tx_RP)
    print(Ty_RP)
    print(Tz_RP)
    time.sleep(3)

    Aurora.tstop()
    Aurora.reset(0)
    print("Finished")

else:
    pump1.stp
    Aurora.tstop()
    Aurora.reset(0)
    # RCSensor.close_port()
    task.close()
    print('Error at Pressure Leveling and Start of Test. Closing.')
