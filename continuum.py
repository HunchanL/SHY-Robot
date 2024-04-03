import matplotlib.patches
import nidaqmx
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import NDI


def translational(sensor_reading):
    # converts sensor reading of translational actuator to extension in local coord frame
    # inputs = specific sensor reading (v/vo) [sensor_reading[0]]
    # outputs = linear translation (y coord, mm)

    # Constants obtained from the sensor calibration (Polynomial constants)
    linear_constant = [2551, -9213.4, 12472, -7512.4, 1702.6]
    const = linear_constant

    x = 0
    y = 0
    z = const[0] * sensor_reading ** 4 + const[1] * sensor_reading ** 3 + const[2] * sensor_reading ** 2 + const[3] * sensor_reading + const[4]
    theta = 0

    return x, y, z, theta


def bending(sensor_reading):
    # converts sensor reading of bending actuator to bending angle and extension in local coord frame
    # inputs = specific sensor reading (v/vo) [sensor_reading[1]]
    # outputs = linear translation (y coord, mm)

    # Constants obtained from the sensor calibration (Polynomial constants)
    bend_constant = [6623.3, -27365, 42245, -28995, 7492]
    const = bend_constant

    theta = const[0] * sensor_reading ** 4 + const[1] * sensor_reading ** 3 + const[2] * sensor_reading ** 2 + const[
        3] * sensor_reading + const[4]

    theta = theta

    theta_rad = theta * np.pi / 180

    r = 4.8  # radius
    chord1 = r*np.sin(theta_rad)
    chord2 = r-r*np.cos(theta_rad)

    x = chord2
    y = 0  # local horizontal # extension due to bending
    z = chord1  # local vertical extension due to bending

    return x, y, z, theta


def roto(sensor_reading):

    # Constants obtained from the sensor calibration (Polynomial constants)
    roto_constant = [2332.4, -8061.8, 10428, -5994.2, 1295.6]

    const = roto_constant
    h = const[0] * sensor_reading ** 4 + const[1] * sensor_reading ** 3 + const[2] * sensor_reading ** 2 + const[
        3] * sensor_reading + const[4]

    #Computing rotation using geometric model
    th_i = 94  # deg
    r = 5.1  # mm
    c = 2 * r * np.sin(th_i * np.pi / 180 / 2)
    k = 1.1
    c_i = np.sqrt(c ** 2 + k ** 2)

    c_n = np.sqrt(c_i ** 2 - (h + k) ** 2)
    alpha = 2 * np.arcsin(c_n / (2 * r))
    th_k = alpha * 180 / np.pi
    phi = th_i - th_k

    x = 0
    y = 0
    z = h

    return x, y, z, phi


def transform_x(x, y, z, theta):
    # Takes x, y, z inputs and places them in a transformation matrix of the form
    # of an x-axis rotation with y and z translation
    # [[1       0              0      x]
    #  [0   cos(theta)   -sin(theta)  y]
    #  [0   sin(theta)   cos(theta)   z]
    #  [0       0              0      1]]
    # Z- AND X-AXIS TRANSLATION, X-AXIS ROTATION

    theta = np.radians(theta)
    rotation = R.from_matrix([[1, 0, 0],
                              [0, np.cos(theta), -np.sin(theta)],
                              [0, np.sin(theta), np.cos(theta)]])
    rotation = rotation.as_matrix()
    translation = [[x],
                   [y],
                   [z]]

    transformation = np.vstack((np.hstack((rotation, translation)), [0, 0, 0, 1]))

    return transformation


def transform_y(x, y, z, theta):
    # FOR BENDING OR LINEAR
    # Z AXIS TRANSLATION, Y-AXIS ROTATION
    phi = np.radians(theta)
    rot = [[np.cos(phi), 0, np.sin(phi)],
           [0, 1, 0],
           [-np.sin(phi), 0, np.cos(phi)]]
    translation = [[x],
                   [y],
                   [z]]

    transformation = np.vstack((np.hstack((rot, translation)), [0, 0, 0, 1]))

    return transformation


def transform_z(x, y, z, phi):
    # FOR ROTOTRANSLATIONAL
    # Z AXIS TRANSLATION, X-AXIS ROTATION
    phi = np.radians(phi)
    rot = [[np.cos(phi), -np.sin(phi), 0],
           [np.sin(phi), np.cos(phi), 0],
           [0, 0, 1]]
    translation = [[x],
                   [y],
                   [z]]

    transformation = np.vstack((np.hstack((rot, translation)), [0, 0, 0, 1]))

    return transformation


def compute_transform(type, sensor):
    if type == "trans":
        x, y, z, theta = translational(sensor)
        transform = transform_z(x, y, z, theta)
    if type == "bend":  # MAY NEED TO BE CHANGED IF ORIENTED IN Y-AXIS BEND
        x, y, z, theta = bending(sensor)
        transform = transform_y(x, y, z, theta)
    if type == "roto":
        x, y, z, phi = roto(sensor)
        transform = transform_z(x, y, z, phi)
    return transform


def get_transforms(t1, t2, t3, h_i):
    # a more general computation of the forward kinematics, independent of
    # what type of module is placed.

    t_sp1 = transform_z(0, 0, 4.0, 0)  # setting the 4 mm spacer translation in local z-axis
    t_sp2 = transform_z(0, 0, 4.0, 0)  # setting the 1.96 mm spacer translation in local z-axis
    t_top = transform_z(0, 0, 0.3, 0) # Tip
    t_i = transform_z(0, 0, 0.3, 0)  # initial height of modules divided in 2


    t_circ = transform_y(4.8, 0, 0, 0)  # setting the offset circle transform for bending arc
    tb1 = np.matmul(t_i, t1)
    t1s = np.matmul(tb1, t_sp1)  # first transform, between 1 and spacer

    ts2 = np.matmul(t1s, t2)  # second transform, between 1st spacer and 2nd module
    t2s = np.matmul(ts2, t_sp2)  # third transform, between 2nd module and 2nd spacer

    ts3 = np.matmul(t2s, t3)  # fourth transform, between 2nd spacer and 3rd module
    t4s = np.matmul(ts3, t_top) #Top

    # ts3 is the transform of the tip
    point0 = [0, 0, 0]
    point1 = [t1[0][3], t1[1][3], t1[2][3]]  # end of tfrm 1
    point2 = [t1s[0][3], t1s[1][3], t1s[2][3]]  # after spacer
    point3 = [-ts2[0][3], ts2[1][3], ts2[2][3]]  # end of tfrm 2
    point4 = [-t2s[0][3], t2s[1][3], t2s[2][3]]  # after spacer
    point5 = [-ts3[0][3], ts3[1][3], ts3[2][3]]  # end of tfrm 3
    point6 = [-t4s[0][3], t4s[1][3], t4s[2][3]]  # end of tfrm 4

    return point0, point1, point2, point3, point4, point5, point6


def plot_continuum(point0, point1, point2, point3, point4, point5, point6, theta, phi, ax, ax2, ax3, ax4):
    #Plotting continuum robot
    x_new = []
    y_new = []
    z_new = []
    phi = -np.radians(phi)
    rot_phi = np.array([[np.cos(phi), -np.sin(phi), 0],
                        [np.sin(phi), np.cos(phi), 0],
                        [0, 0, 1]])

    t = np.linspace(0, theta, 100) #ploting arc for bending

    for i in t:
        r = 4.8  # radius
        x = -(r - r * np.cos(np.radians(i)))
        y = 0
        z = r * np.sin(np.radians(i))
        t_xyz = np.array([[x, y, z]]).T
        T_XYZ = rot_phi.dot(t_xyz)

        x_new.append(T_XYZ[0])
        y_new.append(T_XYZ[1])
        z_new.append(T_XYZ[2])

    linem1 = ax.plot3D([point0[0], point1[0]], [point0[1], point1[1]], [point0[2], point1[2]], color='b', linestyle="-")[0]
    linesp1 = ax.plot3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='r', linestyle="-")[0]
    # linem2 = ax.plot3D(x_new - point2[0], y_new - point2[1], z_new + point2[2], color='b', linestyle="-")[0]
    linem2 = ax.plot3D(x_new[:98], y_new[:98], z_new[:98] + point2[2], color='b', linestyle="-")[0]
    linesp2 = ax.plot3D([point3[0], point4[0]], [point3[1], point4[1]], [point3[2], point4[2]], color='r', linestyle="-")[0]
    linem3 = ax.plot3D([point4[0], point5[0]], [point4[1], point5[1]], [point4[2], point5[2]], color='b', linestyle="-")[0]
    linesp3 = ax.plot3D([point5[0], point6[0]], [point5[1], point6[1]], [point5[2], point6[2]], color='r', linestyle="-")[0]

    linem11 = ax2.plot([point0[0], point1[0]], [point0[1], point1[1]], color='b', linestyle="-", linewidth= 3)[0]
    linesp11 = ax2.plot([point1[0], point2[0]], [point1[1], point2[1]], color='r', linestyle="-", linewidth= 3)[0]
    linem22 = ax2.plot(x_new - point2[0], y_new - point2[1], color='b', linestyle="-", linewidth= 3)[0]
    linesp22 = ax2.plot([point3[0], point4[0]], [point3[1], point4[1]], color='r', linestyle="-", linewidth= 3)[0]
    linem33 = ax2.plot([point4[0], point5[0]], [point4[1], point5[1]],  color='b', linestyle="-", linewidth= 3)[0]
    linesp33 = ax2.plot([point5[0], point6[0]], [point5[1], point6[1]], color='r', linestyle="-", linewidth= 3)[0]

    linem111 = ax3.plot([point0[0], point1[0]], [point0[2], point1[2]], color='b', linestyle="-", linewidth= 3)[0]
    linesp111 = ax3.plot([point1[0], point2[0]], [point1[2], point2[2]], color='r', linestyle="-", linewidth= 3)[0]
    linem222 = ax3.plot(x_new - point2[0], z_new + point2[2], color='b', linestyle="-", linewidth= 3)[0]
    linesp222 = ax3.plot([point3[0], point4[0]], [point3[2], point4[2]], color='r', linestyle="-", linewidth= 3)[0]
    linem333 = ax3.plot([point4[0], point5[0]], [point4[2], point5[2]], color='b', linestyle="-", linewidth= 3)[0]
    linesp333 = ax3.plot([point5[0], point6[0]], [point5[2], point6[2]], color='r', linestyle="-", linewidth= 3)[0]

    linem111 = ax4.plot([point0[1], point1[1]], [point0[2], point1[2]], color='b', linestyle="-", linewidth= 3)[0]
    linesp111 = ax4.plot([point1[1], point2[1]], [point1[2], point2[2]], color='r', linestyle="-", linewidth= 3)[0]
    linem222 = ax4.plot(y_new - point2[1], z_new + point2[2], color='b', linestyle="-", linewidth= 3)[0]
    linesp222 = ax4.plot([point3[1], point4[1]], [point3[2], point4[2]], color='r', linestyle="-", linewidth= 3)[0]
    linem333 = ax4.plot([point4[1], point5[1]], [point4[2], point5[2]], color='b', linestyle="-", linewidth= 3)[0]
    linesp333 = ax4.plot([point5[1], point6[1]], [point5[2], point6[2]], color='r', linestyle="-", linewidth= 3)[0]

def baselines(task, Aurora):
    #Obtaining baseline voltages and initial height of the continuum robot
    response = 'n'
    while response != 'y':
        sensor_init = task.read()
        print(sensor_init)
        T_BR, T_RP = NDI.coor_transform_Base2Ref(Aurora)  # COORD TRANS BASE TO REF
        print("T_RP")
        print(T_RP)
        print(T_RP[2][3])
        print("Accept baseline voltages? [y/n]")
        response = input()
    return sensor_init, T_BR, T_RP


def update(frame, data_m, data_x, data_y, data_z, sdata1, sdata2, sdata3, sensor_i, task, Aurora, ax, ax2, ax3, ax4, T_BR, h_i,
           model_x, model_y, model_z, start_time, E_time):
    # include  K_R, K_B, K_L for raw voltage
    # show magnetic tracking tip
    data_m.append(Aurora.tx())
    T_x_live, T_y_live, T_z_live = NDI.coor_transform_Base2probe(Aurora, T_BR)
    T_live = [float(T_x_live),
              float(T_y_live),
              float(T_z_live)]  # live magnetic tracker data wrt base coordinate system

    sensor = task.read()
    s1flt = float(sensor[0])
    s2flt = float(sensor[1])
    s3flt = float(sensor[2])

    # Offset Calculation

    try:
        data_x.append(T_live[0])
        data_y.append(T_live[1])
        data_z.append(T_live[2])

        sdata1.append(sensor[2])
        sdata2.append(sensor[1])
        sdata3.append(sensor[0])

    except:
        pass

    data_x = data_x[-1:]
    data_y = data_y[-1:]
    data_z = data_z[-1:]
    sdata1 = sdata1[-2:]
    sdata2 = sdata2[-2:]
    sdata3 = sdata3[-2:]

    # for each frame, update the data stored on each artist.
    ax.clear()
    ax.set_xlim3d([-10, 5])
    ax.set_ylim3d([-10, 5])
    ax.set_zlim3d([0, 20])

    ax.set_xlabel('x [mm]', fontsize=16, fontname='Times New Roman')
    ax.set_ylabel('y [mm]', fontsize=16, fontname='Times New Roman')
    ax.set_zlabel('z [mm]', fontsize=16, fontname='Times New Roman')

    #x-y subplot
    ax2.clear()
    ax2.set_xlim([-10, 5])
    ax2.set_ylim([-10, 5])
    ax2.set_xlabel('x [mm]', fontsize=16, fontname='Times New Roman')
    ax2.set_ylabel('y [mm]', fontsize=16, fontname='Times New Roman')

    #x-z subplot
    ax3.clear()
    ax3.set_xlim([-10, 5])
    ax3.set_ylim([0, 25])
    ax3.set_xlabel('x [mm]', fontsize=16, fontname='Times New Roman')
    ax3.set_ylabel('z [mm]', fontsize=16, fontname='Times New Roman')

    #y-z subplot
    ax4.clear()
    ax4.set_xlim([-10, 5])
    ax4.set_ylim([0, 25])
    ax4.set_xlabel('y [mm]', fontsize=16, fontname='Times New Roman')
    ax4.set_ylabel('z [mm]', fontsize=16, fontname='Times New Roman')


    s1 = sdata1[0]
    s2 = sdata2[0]
    s3 = sdata3[0]

    # update the plot and transformations
    # data_y = X axis
    # data_x = Y axis
    # data_z = Z axis
    ax.plot3D(data_x, data_y, data_z, color='g', marker="o", markersize=4)
    ax2.plot(data_x, data_y,  color='g', marker="o", markersize=4) #x-y
    ax3.plot(data_x, data_z, color='g', marker="o", markersize=4) #x-z
    ax4.plot(data_y, data_z, color='g', marker="o", markersize=4) #y-z

    # V/V0
    _, _, _, theta = bending(s2 / sensor_i[1])
    _, _, _, phi = roto(s1 / sensor_i[2])
    t1 = compute_transform("roto", s1 / sensor_i[2])
    t2 = compute_transform("bend", s2 / sensor_i[1])
    t3 = compute_transform("trans", s3 / sensor_i[0])

    point0, point1, point2, point3, point4, point5, point6 = get_transforms(t1, t2, t3, h_i)
    plot_continuum(point0, point1, point2, point3, point4, point5, point6, theta, phi, ax, ax2, ax3, ax4)
    model_x.append(point6[0])
    model_y.append(point6[1])
    model_z.append(point6[2])

    E_time.append(time.time()-start_time)


def fig_close(event, current_sheet, label, mdata, s1, s2, s3, tipx, tipy, tipz, mx, my, mz, E_time):
    #Data Saving
    # tip x,y,z -> model data
    # mx,y,z -> actual magnetic tracker data
    # distance calculation
    print("Saving Data...")
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

    # Time Data
    np_time = (np.array(E_time)).transpose()

    dist = np.sqrt((np_tipx - np_mx) ** 2 + (np_tipy - np_my) ** 2 + (np_tipz - np_mz) ** 2)
    np_dist = (np.array(dist)).transpose()
    dist_start = np.sqrt((np_mx - np_mx[0]) ** 2 + (np_my - np_my[0]) ** 2 + (np_mz - np_mz[0]) ** 2)
    np_d_s = (np.array(dist_start)).transpose()
    np_pro_data = (
        np.vstack((np_s1, np_s2, np_s3, np_tipx, np_tipy, np_tipz, np_mx, np_my, np_mz, np_dist, np_d_s, np_time))).transpose()

    for x in range(len(label)):
        current_sheet.write(0, x, label[x])

    row = 1
    col = 0

    for s_v1, s_v2, s_v3, tx, ty, tz, magtx, magty, magtz, distance, d_s, e_t in np_pro_data:
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
        current_sheet.write(row, col + 11, e_t)
        row += 1

    Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port = NDI.data_formatting(mdata)
    NDI.data_save3(current_sheet, Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, error, n_frame, port)

