import xlwings as xw
import numpy as np
from scipy.spatial.transform import Rotation as R

def data_read(file):
    data = xw.Book(file).sheets['Sheet1']
    roto_data = data.range("A2:A816").value
    bend_data = data.range("B2:B816").value
    linear_data = data.range("C2:C816").value

    Q_0_ref = data.range("P2:P816").value
    Q_0_prb = data.range("X2:X816").value
    Q_0 = np.vstack((Q_0_ref, Q_0_prb))
    Q_0 = np.transpose(Q_0)

    Q_X_ref = data.range("Q2:Q816").value
    Q_X_prb = data.range("Y2:Y816").value
    Q_X = np.vstack((Q_X_ref, Q_X_prb))
    Q_X = np.transpose(Q_X)

    Q_Y_ref = data.range("R2:R816").value
    Q_Y_prb = data.range("Z2:Z816").value
    Q_Y = np.vstack((Q_Y_ref, Q_Y_prb))
    Q_Y = np.transpose(Q_Y)

    Q_Z_ref = data.range("S2:S816").value
    Q_Z_prb = data.range("AA2:AA816").value
    Q_Z = np.vstack((Q_Z_ref, Q_Z_prb))
    Q_Z = np.transpose(Q_Z)

    T_X_ref = data.range("T2:T816").value
    T_X_prb = data.range("AB2:AB816").value
    T_X = np.vstack((T_X_ref, T_X_prb))
    T_X = np.transpose(T_X)

    T_Y_ref = data.range("U2:U816").value
    T_Y_prb = data.range("AC2:AC816").value
    T_Y = np.vstack((T_Y_ref, T_Y_prb))
    T_Y = np.transpose(T_Y)

    T_Z_ref = data.range("V2:V816").value
    T_Z_prb = data.range("AD2:AD816").value
    T_Z = np.vstack((T_Z_ref, T_Z_prb))
    T_Z = np.transpose(T_Z)

    return roto_data, bend_data, linear_data, Q_0, Q_X, Q_Y, Q_Z, T_X, T_Y, T_Z

def coor_transform_Base2Ref(Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z):
    # Coordinate transformation from the base of continuum robot to Ref
    # Uses the initial X,Y position of probe tracker to define the base of continuum robot
    # assuming 1st port = ref, 2nd port = probe
    # T_GP =  from Mag Generator coordinate to probe coordinate
    # T_GR =  from Mag Generator coordinate to ref coordinate
    # T_RG =  from ref coordinate to Mag Generator coordinate
    # T_RP =  from ref coordinate to probe coordinate
    # T_BR =  from base coordinate to ref coordinate
    # T_RB =  from  coordinate to ref coordinate
    R_GR = R.from_quat([Q_x[0], Q_y[0], Q_z[0], Q_0[0]])
    R_GP = R.from_quat([Q_x[1], Q_y[1], Q_z[1], Q_0[1]])
    R_GR = R_GR.as_matrix()  # Rotation matrix - Reference probe
    R_GP = R_GP.as_matrix()  # Rotation matrix - Moving probe

    trans_GR = np.array([T_x[0], T_y[0], T_z[0]])  # translation vector - Reference probe
    trans_GP = np.array([T_x[1], T_y[1], T_z[1]])  # translation vector - Moving probe

    T_GR = np.vstack((np.hstack((R_GR, trans_GR[:, None])), [0, 0, 0, 1]))
    T_GP = np.vstack((np.hstack((R_GP, trans_GP[:, None])), [0, 0, 0, 1]))
    T_RG = np.linalg.inv(T_GR)

    T_RP = np.matmul(T_RG, T_GP)

    T_RB = np.array([[1, 0, 0, T_RP[0, 3]], [0, 1, 0, T_RP[1, 3]], [0, 0, 1, 0], [0, 0, 0, 1]])
    T_BR = np.linalg.inv(T_RB)

    return T_BR, T_RP

def coor_transform_Base2probe(Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z, T_BR):
    # Base to Probe Coordinate Transformation
    # Base is the base of the continuum robot
    # Input: Probe data (Aurora), Base2Ref coordinate transformation
    # Output: Tx, Ty, Tz of probe in base coordinate

    # assuming 1st port = ref, 2nd port = probe
    # T_GP =  from Mag Generator coordinate to probe coordinate
    # T_GR =  from Mag Generator coordinate to ref coordinate
    # T_RG =  from ref coordinate to Mag Generator coordinate
    # T_RP =  from ref coordinate to probe coordinate
    # T_BR =  from base coordinate to ref coordinate
    # T_RB =  from  coordinate to ref coordinate
    R_GR = R.from_quat([Q_x[0], Q_y[0], Q_z[0], Q_0[0]])
    R_GP = R.from_quat([Q_x[1], Q_y[1], Q_z[1], Q_0[1]])
    R_GR = R_GR.as_matrix()  # Rotation matrix - Reference probe
    R_GP = R_GP.as_matrix()  # Rotation matrix - Moving probe

    trans_GR = np.array([T_x[0], T_y[0], T_z[0]])  # translation vector - Reference probe
    trans_GP = np.array([T_x[1], T_y[1], T_z[1]])  # translation vector - Moving probe

    T_GR = np.vstack((np.hstack((R_GR, trans_GR[:, None])), [0, 0, 0, 1]))
    T_GP = np.vstack((np.hstack((R_GP, trans_GP[:, None])), [0, 0, 0, 1]))
    T_RG = np.linalg.inv(T_GR)

    T_RP = np.matmul(T_RG, T_GP)
    T_BP = np.matmul(T_BR, T_RP)

    Tx_BP = T_BP[0, 3]
    Ty_BP = T_BP[1, 3]
    Tz_BP = T_BP[2, 3]

    return Tx_BP, Ty_BP, Tz_BP

def translational(sensor_reading):
    # converts sensor reading of translational actuator to extension in local coord frame
    # inputs = specific sensor reading (v/vo) [sensor_reading[0]]
    # outputs = linear translation (y coord, mm)

    linear = [2551, -9213.4, 12472, -7512.4, 1702.6]
    const = linear

    offset_z = 0.1 #Necessary offset to prevent air being the sensor medium

    x = 0
    y = 0
    z = const[0] * sensor_reading ** 4 + const[1] * sensor_reading ** 3 + const[2] * sensor_reading ** 2 + const[3] * sensor_reading + const[4]
    theta = 0

    z = z + offset_z
    return x, y, z, theta


def bending(sensor_reading):
    # converts sensor reading of bending actuator to bending angle and extension in local coord frame
    # inputs = specific sensor reading (v/vo) [sensor_reading[1]]
    # outputs = linear translation (y coord, mm)


    bend = [6623.3, -27365, 42245, -28995, 7492]
    const = bend

    offset_b = 2 ##Necessary offset to prevent air being the sensor medium
    theta = const[0] * sensor_reading ** 4 + const[1] * sensor_reading ** 3 + const[2] * sensor_reading ** 2 + const[
        3] * sensor_reading + const[4]

    theta = theta + offset_b

    theta_rad = theta * np.pi / 180

    r = 4.8  # radius of 5.1 mm
    chord1 = r*np.sin(theta_rad)
    chord2 = r-r*np.cos(theta_rad)

    x = chord2
    y = 0  # local horizontal # extension due to bending
    z = chord1  # local vertical extension due to bending


    return x, y, z, theta


def roto(sensor_reading):

    roto3_060123 = [2332.4, -8061.8, 10428, -5994.2, 1295.6]

    offset_h = 0.2 #Initial offset created from the actuator
    offset_r = 0

    const = roto3_060123
    h = const[0] * sensor_reading ** 4 + const[1] * sensor_reading ** 3 + const[2] * sensor_reading ** 2 + const[
        3] * sensor_reading + const[4]

    h = h + offset_h #Necessary offset to prevent air being the sensor medium

    th_i = 94  # deg
    r = 5.1  # mm
    c = 2 * r * np.sin(th_i * np.pi / 180 / 2)
    k = 1.1
    c_i = np.sqrt(c ** 2 + k ** 2)

    c_n = np.sqrt(c_i ** 2 - (h + k) ** 2)
    alpha = 2 * np.arcsin(c_n / (2 * r))
    th_k = alpha * 180 / np.pi
    phi = th_i - th_k

    phi = phi + offset_r
    ##

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


def get_transforms(t1, t2, t3):
    # a more general computation of the forward kinematics, independent of
    # what type of module is placed.

    t_sp1 = transform_z(0, 0, 4.0, 0)  # setting the 4 mm spacer translation in local z-axis
    t_sp2 = transform_z(0, 0, 3.9, 0)  # setting the 3.9 mm spacer translation in local z-axis
    t_top = transform_z(0, 0, 0.3, 0) # Tip
    t_i = transform_z(0, 0, 0.3, 0)  # initial height of modules divided in 2

    t_circ = transform_y(4.8, 0, 0, 0)  # setting the offset circle transform for bending arc
    tb1 = np.matmul(t_i, t1)
    t1s = np.matmul(tb1, t_sp1)  # first transform, between 1 and spacer
    #tcircle = np.matmul(t1s, t_circ)  # for plotting the arc of bending actuator

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
    #pointc = [tcircle[0][3], tcircle[1][3], tcircle[2][3]]

    return point0, point1, point2, point3, point4, point5, point6


def plot_continuum(point0, point1, point2, point3, point4, point5, point6, theta, phi, ax):
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

    linem1 = ax.plot3D([point0[0], point1[0]], [point0[1], point1[1]], [point0[2], point1[2]], color='b', linestyle="-",linewidth=3.0)[0]
    linesp1 = ax.plot3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='r', linestyle="-",linewidth=3.0)[0]
    linem2 = ax.plot3D(x_new[:98], y_new[:98], z_new[:98] + point2[2], color='b', linestyle="-",linewidth=3.0)[0]
    linesp2 = ax.plot3D([point3[0], point4[0]], [point3[1], point4[1]], [point3[2], point4[2]], color='r', linestyle="-",linewidth=3.0)[0]
    linem3 = ax.plot3D([point4[0], point5[0]], [point4[1], point5[1]], [point4[2], point5[2]], color='b', linestyle="-",linewidth=3.0)[0]
    linesp3 = ax.plot3D([point5[0], point6[0]], [point5[1], point6[1]], [point5[2], point6[2]], color='r', linestyle="-",linewidth=3.0)[0]



def update(frame, sensor_i, roto_data, bend_data, linear_data, Q_0, Q_x, Q_y, Q_z, T_x, T_y, T_z , ax, ax2, ax3, ax4, T_BR, h_i,
           Tip_model_x, Tip_model_y, Tip_model_z, Tip_actual_x, Tip_actual_y, Tip_actual_z, dist):
    # include  K_R, K_B, K_L for raw voltage
    # show magnetic tracking tip
    #print(frame)
    T_BR, T_RP = coor_transform_Base2Ref(Q_0[0], Q_x[0], Q_y[0], Q_z[0], T_x[0], T_y[0], T_z[0])
    T_x_live, T_y_live, T_z_live = coor_transform_Base2probe(Q_0[frame], Q_x[frame], Q_y[frame], Q_z[frame], T_x[frame], T_y[frame], T_z[frame], T_BR)
    T_live = [-float(T_x_live),
              float(T_y_live),
              float(T_z_live)]  # live magnetic tracker data wrt base coordinate system
    # print(T_base)
    # print(T_live)
    s1flt = float(linear_data[frame])
    s2flt = float(bend_data[frame])
    s3flt = float(roto_data[frame])

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
    ax3.set_xlabel('y [mm]', fontsize=16, fontname='Times New Roman')
    ax3.set_ylabel('z [mm]', fontsize=16, fontname='Times New Roman')

    #y-z subplot
    ax4.clear()
    ax4.set_xlim([-10, 5])
    ax4.set_ylim([0, 25])
    ax4.set_xlabel('x [mm]', fontsize=16, fontname='Times New Roman')
    ax4.set_ylabel('z [mm]', fontsize=16, fontname='Times New Roman')

    s1 = roto_data[frame]
    s2 = bend_data[frame]
    s3 = linear_data[frame]


    ax.plot3D(T_live[1], T_live[0], T_live[2], color='g', marker="o", markersize=4)
    ax2.plot(T_live[1], T_live[0],  color='g', marker="o", markersize=4) #x-y
    ax3.plot(T_live[1], T_live[2], color='g', marker="o", markersize=4) #x-z
    ax4.plot(T_live[0], T_live[2], color='g', marker="o", markersize=4) #y-z

    # V/V0
    _, _, _, theta = bending(s2 / sensor_i[1])
    _, _, _, phi = roto(s1 / sensor_i[2])
    t1 = compute_transform("roto", s1 / sensor_i[2])
    t2 = compute_transform("bend", s2 / sensor_i[1])
    t3 = compute_transform("trans", s3 / sensor_i[0])

    point0, point1, point2, point3, point4, point5, point6 = get_transforms(t1, t2, t3, h_i)
    plot_continuum(point0, point1, point2, point3, point4, point5, point6, theta, phi, ax, ax2, ax3, ax4)

    Tip_model_x.append(point6[0])
    Tip_model_y.append(point6[1])
    Tip_model_z.append(point6[2])
    Tip_actual_x.append(T_live[1])
    Tip_actual_y.append(T_live[0])
    Tip_actual_z.append(T_live[2])
    dist_calc = np.sqrt((T_live[0] - point6[0]) ** 2 + (T_live[1] - point6[1]) ** 2 + (T_live[2] - point6[2]) ** 2)
    dist.append(dist_calc)

def fig_close(event, current_sheet,Tip_model_x, Tip_model_y, Tip_model_z, Tip_actual_x, Tip_actual_y, Tip_actual_z, dist):
    label = ['Tip_Ax', 'Tip_Ay', 'Tip_Az', 'Tip_Mx', 'Tip_My', 'Tip_Mz', 'Dist Error']

    # TIP Actual (mag) DATA
    np_tipx = (np.array(Tip_actual_x)).transpose()
    np_tipy = (np.array(Tip_actual_y)).transpose()
    np_tipz = (np.array(Tip_actual_z)).transpose()
    # Model DATA
    np_mx = (np.array(Tip_model_x)).transpose()
    np_my = (np.array(Tip_model_y)).transpose()
    np_mz = (np.array(Tip_model_z)).transpose()

    np_dist = (np.array(dist)).transpose()

    np_pro_data = np.vstack((np_tipx, np_tipy, np_tipz, np_mx, np_my, np_mz, np_dist)).transpose()

    for x in range(len(label)):
        current_sheet.write(0, x, label[x])
    row = 1
    col = 0

    for tx, ty, tz, magtx, magty, magtz, distance in np_pro_data:
        current_sheet.write(row, col + 0, tx)
        current_sheet.write(row, col + 1, ty)
        current_sheet.write(row, col + 2, tz)
        current_sheet.write(row, col + 3, magtx)
        current_sheet.write(row, col + 4, magty)
        current_sheet.write(row, col + 5, magtz)
        current_sheet.write(row, col + 6, distance)
        row += 1

