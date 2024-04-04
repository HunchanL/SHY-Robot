import Postprocess_compute as ppc
from matplotlib import pyplot as plt
import xlwings as xw

file = "Configuration states.xlsx"
i = 1 #Change this value to move to different frame number


data = xw.Book(file).sheets['Sheet1']
roto_data = data.range("E2:E8").value
bend_data = data.range("F2:F8").value
linear_data = data.range("G2:G8").value

act_x = data.range("H2:H8").value
act_y = data.range("I2:I8").value
act_z = data.range("J2:J8").value

_, _, _, theta = ppc.bending(bend_data[i])
_, _, _, phi = ppc.roto(roto_data[i])
t1 = ppc.compute_transform("roto", roto_data[i])
t2 = ppc.compute_transform("bend", bend_data[i])
t3 = ppc.compute_transform("trans",linear_data[i])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.clear()
ax.set_xlim3d([-10, 2])
ax.set_ylim3d([-5, 5])
ax.set_zlim3d([0, 20])

# ax.set_xlabel('x [mm]', fontsize=13, fontname='Times New Roman')
# ax.set_ylabel('y [mm]', fontsize=13, fontname='Times New Roman')
# ax.set_zlabel('z [mm]', fontsize=13, fontname='Times New Roman')

point0, point1, point2, point3, point4, point5, point6 = ppc.get_transforms(t1, t2, t3)
ppc.plot_continuum(point0, point1, point2, point3, point4, point5, point6, theta, phi, ax)
ax.plot3D(act_x[i], act_y[i], act_z[i], color='g', marker="o", markersize=4)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
ax.zaxis.set_tick_params(labelsize=20)
# plt.zticks(fontsize = 20)
plt.show()
