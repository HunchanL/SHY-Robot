import nidaqmx
from matplotlib import pyplot as plt
import pumplib
import time
import xlsxwriter
import numpy as np
import NDI
import testProcedure
import continuum

sensor_num = "workspace_test"
workbook = xlsxwriter.Workbook(sensor_num + '.xlsx')

sheet1 = workbook.add_worksheet()
worksheet = [sheet1]

# INITIALIZATION --
# PORT CHANGES IN THE INITIALIZE FUNCTION
task, pump1, pump2, pump3, Aurora = testProcedure.initialize()     # NIDAQ channels, Pumps X3, Aurora tracker
sensor_i, T_BR, T_RP = continuum.baselines(task, Aurora)     # establish baseline voltages X3, and height_i

z_height = float(T_RP[2][3])
sensor = task.read()

K_R = sensor_i[0]
K_B = sensor_i[1]
K_L = sensor_i[2]

#V/V0
_, _, _, theta = continuum.bending(sensor[1] / sensor_i[1])
_, _, _, phi = continuum.roto(sensor[2] / sensor_i[2])
t1 = continuum.compute_transform("roto", sensor[2] / sensor_i[2])
t2 = continuum.compute_transform("bend", sensor[1] / sensor_i[1])
t3 = continuum.compute_transform("trans", sensor[0] / sensor_i[0])

point0, point1, point2, point3, point4, point5 = continuum.get_transforms(t1, t2, t3, z_height) # plot continuum robot depiction
point_tip = point5


# RUN TEST
# Pump is manually controlled
samples = 1     # define number of samples (run the test x number of times)

print('Ready to begin ' + str(samples) + ' tests? [y/n]')
response = input()
if response == 'y':

    for sample in range(samples):

        print('Test' + str(sample + 1) + 'ready \r\n')
        current_sheet = worksheet[sample]
        # run test script in test_procedure
        label, mdata, s1, s2, s3, tx, ty, tz, mx, my, mz = testProcedure.run_continuum(Aurora, pump1, pump2, pump3,
                                                                                       task, T_BR, z_height, sensor_i)
        # save all the data
        testProcedure.save_data_continuum(current_sheet, label, mdata, s1, s2, s3, tx, ty, tz, mx, my, mz)

    # stop all operations and reset Aurora
    pump1.stp()
    pump2.stp()
    pump3.stp()
    Aurora.tstop()
    Aurora.reset(0)
    task.close()
    workbook.close()
    print("Testing finished. Data saved.")

else:
    # last "else" is a failsafe for unintended inputs
    # stops pumps to prevent actuators from popping, closes NIDAQ task, and
    # resets Aurora, preventing a reboot
    pump1.stp
    Aurora.tstop()
    Aurora.reset(0)
    # RCSensor.close_port()
    task.close()
    print('Error at Start of Test. Closing...')
