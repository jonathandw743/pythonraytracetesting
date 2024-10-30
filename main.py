# from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import csv
import copy



# Create a 1x2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# First subplot
axes[0].plot([1, 2, 3, 4], [1, 4, 9, 16])
axes[0].set_title("First Graph")

# Second subplot
axes[1].plot([1, 2, 3, 4], [2, 4, 6, 8])
axes[1].set_title("Second Graph")

plt.tight_layout()  # Adjusts spacing between subplots for better readability
plt.show()

exit()

fig = plt.figure()
# ax = plt.axes(projection="3d")

# ax.plot3D(xline, yline, zline, 'gray')

# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

original_ros = []
# Open the CSV file for reading
with open("points.csv", "r") as csvfile:
    csvreader = csv.reader(csvfile)

    # Skip the header row if it exists
    next(csvreader, None)

    # Iterate over each row in the CSV file
    for i, row in enumerate(csvreader):
        # Convert the row values to float and create a NumPy array
        point = np.array([float(row[0]), float(row[1]), float(row[2])])

        # Append the NumPy array to the list
        original_ros.append(point)

original_rds = [np.array([0.0, 0.0, 1.0])] * len(original_ros)
# if len(original_rds) > 3:
#     original_rds[3][2] *= -1


BH_MASS = 1.0
BH_POS = np.array([0.0, 0.0, 0.0])
DT = 0.005
G = 1.0


def sqrnorm(vec):
    return np.einsum("...i,...i", vec, vec)


# def runge_kutta(ro, rd, h2):

BREAK_ON_EH = True


# leapfrog
# ros = copy.deepcopy(original_ros)
# rds = copy.deepcopy(original_rds)

# for o_ro, o_rd in zip(ros, rds):
#     points = []

#     ro = copy.deepcopy(o_ro)
#     rd = copy.deepcopy(o_rd)

#     h2 = sqrnorm(np.cross(ro, rd))

#     for i in range(int(8 / DT)):
#         ro += rd * DT
#         acceleration = -1.5 * h2 * ro / np.power(sqrnorm(ro), 2.5)
#         rd += acceleration * DT

#         points.append(ro.copy())

#         if BREAK_ON_EH and np.linalg.norm(ro) < 1.0:
#             break

#     ax.plot(
#         [p[0] for p in points], [p[2] for p in points], [p[1] for p in points], c="purple"
#     )


# def RK4f(y,h2):
#     f = np.zeros(y.shape)

#     f[0:3] = y[3:6]
#     f[3:6] = - 1.5 * h2 * y[0:3] / np.power(sqrnorm(y[0:3]),2.5)
#     return f

# ros = copy.deepcopy(original_ros)
# rds = copy.deepcopy(original_rds)

# for o_ro, o_rd in zip(ros, rds):
#     points = []
#     ro = copy.deepcopy(o_ro)
#     rd = copy.deepcopy(o_rd)
#     # print(ro, rd)

#     h2 = sqrnorm(np.cross(ro,rd))#[:,np.newaxis]
#     for i in range(int(8 / DT)):
#         #simple step size control
#         rkstep = DT

#         # standard Runge-Kutta
#         y = np.array([ro[0], ro[1], ro[2], rd[0], rd[1], rd[2]])
#         k1 = RK4f( y, h2)
#         k2 = RK4f( y + 0.5*rkstep*k1, h2)
#         k3 = RK4f( y + 0.5*rkstep*k2, h2)
#         k4 = RK4f( y +     rkstep*k3, h2)

#         increment = rkstep/6. * (k1 + 2*k2 + 2*k3 + k4)

#         rd += increment[3:6]
#         ro += increment[0:3]

#         # print(ro, rd)

#         points.append(ro.copy())

#         if BREAK_ON_EH and np.linalg.norm(ro) < 1.0:#2 * G * BH_MASS:
#             break

#     # ax.plot(
#     #     [p[0] for p in points], [p[2] for p in points], [p[1] for p in points], c="pink"
#     # )


# def magnitude_squared(vec):
#     sum = 0
#     for e in vec:
#         sum += e * e

#     return sum

# def runge_kutte_4th_order(h, devirative_f, variable_derivative_arg, *constant_derivative_args):

#     # k_1 = h * devirative_f(x_0, y_0)
#     # k_2 = h * devirative_f(x_0 + 0.5 * h, y_0 + 0.5 * k_1)
#     # k_3 = h * devirative_f(x_0 + 0.5 * h, y_0 + 0.5 * k_2)
#     # k_4 = h * devirative_f(x_0 + h, y_0 + k_3)

#     print(variable_derivative_arg)

#     k_1 = h * devirative_f(variable_derivative_arg, *constant_derivative_args)
#     # print(k_1)
#     k_2 = h * devirative_f(variable_derivative_arg + 0.5 * k_1, *constant_derivative_args)
#     # print(k_2)
#     k_3 = h * devirative_f(variable_derivative_arg + 0.5 * k_2, *constant_derivative_args)
#     # print(k_3)
#     k_4 = h * devirative_f(variable_derivative_arg + k_3, *constant_derivative_args)
#     # print(k_4)

#     return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0

# def ro_derivative(rd):
#     return copy.deepcopy(rd)

# def rd_derivative(ro, h2):
#     return -1.5 * h2 * ro / np.power(magnitude_squared(ro), 2.5)

# def ro_rd_derivative(ro_rd, h2):
#     ro = ro_rd[0:3]
#     rd = ro_rd[3:6]
#     # a=ro_rd[3:6]
#     a=ro_derivative(rd)
#     # b=-1.5 * h2 * ro_rd[0:3] / np.power(magnitude_squared(ro_rd[0:3]), 2.5)
#     b=rd_derivative(ro,h2)
#     return np.array([a[0],a[1],a[2],b[0],b[1],b[2]])

# ros = copy.deepcopy(original_ros)
# rds = copy.deepcopy(original_rds)

# for o_ro, o_rd in zip(ros, rds):
#     points = []
#     ro = copy.deepcopy(o_ro)
#     rd = copy.deepcopy(o_rd)

#     h2 = sqrnorm(np.cross(ro,rd))#[:,np.newaxis]
#     for i in range(int(8 / DT)):

#         ro_rd = np.array([ro[0], ro[1], ro[2], rd[0], rd[1], rd[2]])

#         # ro_der = ro_derivative(rd)
#         # rd_der = rd_derivative(ro, h2)
#         # print(np.array([ro_der[0], ro_der[1], ro_der[2], rd_der[0], rd_der[1], rd_der[2]]))
#         # print(ro_rd_derivative(ro_rd, h2))

#         # print("")

#         ro_inc = runge_kutte_4th_order(DT, ro_derivative, rd)
#         rd_inc = runge_kutte_4th_order(DT, rd_derivative, ro, h2)
#         inc = np.array([ro_inc[0], ro_inc[1],ro_inc[2],rd_inc[0],rd_inc[1],rd_inc[2]])
#         # print(inc)
#         # print("---")

#         inc = runge_kutte_4th_order(DT, ro_rd_derivative, ro_rd, h2)
#         # print(inc)

#         # print("---")


#         ro += inc[0:3]
#         rd += inc[3:6]
#         # ro += runge_kutte_4th_order(DT, ro_derivative, rd)
#         # rd += runge_kutte_4th_order(DT, rd_derivative, ro, h2)

#         # print(ro, rd)

#         points.append(ro.copy())

#         if BREAK_ON_EH and np.linalg.norm(ro) < 1.0:#2 * G * BH_MASS:
#             break

#     # ax.scatter(
#     #     [p[0] for p in points], [p[2] for p in points], [p[1] for p in points], c="red"
#     # )

# def runge_kutte_4th_order(
#     h, devirative_f, variable_derivative_arg, *constant_derivative_args
# ):

#     # k_1 = h * devirative_f(x_0, y_0)
#     # k_2 = h * devirative_f(x_0 + 0.5 * h, y_0 + 0.5 * k_1)
#     # k_3 = h * devirative_f(x_0 + 0.5 * h, y_0 + 0.5 * k_2)
#     # k_4 = h * devirative_f(x_0 + h, y_0 + k_3)

#     # print(variable_derivative_arg)

#     k_1 = h * devirative_f(variable_derivative_arg, *constant_derivative_args)
#     # print(k_1)
#     k_2 = h * devirative_f(
#         variable_derivative_arg + 0.5 * k_1, *constant_derivative_args
#     )
#     # print(k_2)
#     k_3 = h * devirative_f(
#         variable_derivative_arg + 0.5 * k_2, *constant_derivative_args
#     )
#     # print(k_3)
#     k_4 = h * devirative_f(variable_derivative_arg + k_3, *constant_derivative_args)
#     # print(k_4)

#     return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0


# def ro_derivative(rd):
#     return copy.deepcopy(rd)

def magnitude_squared(vec):
    sum = 0
    for e in vec:
        sum += e * e

    return sum


def rd_derivative(ro, h2):
    return -1.5 * h2 * ro / np.power(magnitude_squared(ro), 2.5)


def ro_rd_derivative(ro_rd, h2):
    ro = ro_rd[0:3]
    rd = ro_rd[3:6]
    # a=ro_rd[3:6]
    a = ro_derivative(rd)
    # b=-1.5 * h2 * ro_rd[0:3] / np.power(magnitude_squared(ro_rd[0:3]), 2.5)
    b = rd_derivative(ro, h2)
    return np.array([a[0], a[1], a[2], b[0], b[1], b[2]])


def plot_point(ro_rd):
    points = []
    ro = copy.deepcopy(ro_rd[0])
    rd = copy.deepcopy(ro_rd[1])

    x = []
    y = []

    h2 = sqrnorm(np.cross(ro, rd))  # [:,np.newaxis]
    for i in range(int(16 / DT)):
        # h2 = sqrnorm(np.cross(ro,rd))#[:,np.newaxis]
        # print(h2)
        # print(h2)

        # ro_rd = np.array([ro[0], ro[1], ro[2], rd[0], rd[1], rd[2]])

        # ro_der = ro_derivative(rd)
        # rd_der = rd_derivative(ro, h2)
        # print(np.array([ro_der[0], ro_der[1], ro_der[2], rd_der[0], rd_der[1], rd_der[2]]))
        # print(ro_rd_derivative(ro_rd, h2))

        # print("")

        # ro_inc = runge_kutte_4th_order(DT, ro_derivative, rd)
        # rd_inc = runge_kutte_4th_order(DT, rd_derivative, ro, h2)
        # inc = np.array([ro_inc[0], ro_inc[1],ro_inc[2],rd_inc[0],rd_inc[1],rd_inc[2]])
        # print(inc)
        # print("---")

        # inc = runge_kutte_4th_order(DT, ro_rd_derivative, ro_rd, h2)
        # # print(inc)

        # print("---")

        # ro += inc[0:3]
        # rd += inc[3:6]

        ro_k1 = DT * rd
        rd_k1 = DT * rd_derivative(ro, h2)
        ro_k2 = DT * (rd + 0.5 * rd_k1)
        rd_k2 = DT * rd_derivative(ro + 0.5 * ro_k1, h2)
        ro_k3 = DT * (rd + 0.5 * rd_k2)
        rd_k3 = DT * rd_derivative(ro + 0.5 * ro_k2, h2)
        ro_k4 = DT * (rd + rd_k3)
        rd_k4 = DT * rd_derivative(ro + ro_k3, h2)
        delta_ro = (ro_k1 + 2.0 * ro_k2 + 2.0 * ro_k3 + ro_k4) / 6.0
        delta_rd = (rd_k1 + 2.0 * rd_k2 + 2.0 * rd_k3 + rd_k4) / 6.0
        ro += delta_ro
        rd += delta_rd

        x.append(np.linalg.norm(ro))
        y.append(np.linalg.norm(rd)/DT)

        # rd = rd / np.linalg.norm(rd)

        # rd = (rd / np.linalg.norm(rd)) * DT
        # ro += runge_kutte_4th_order(DT, ro_derivative, rd)
        # rd += runge_kutte_4th_order(DT, rd_derivative, ro, h2)

        # print(ro, rd)
        points.append((ro.copy(), rd.copy()))

        if len(points) > 2:
            print(
                np.linalg.norm(points[-1][0] - points[-2][0])
                - np.linalg.norm(points[1][0] - points[0][0])
            )

        if np.linalg.norm(ro) < 0.5:  # 2 * G * BH_MASS:
            break
    ax.scatter(x, y, [0]*len(x), c="orange", s=1)
    return points


ros = copy.deepcopy(original_ros)
rds = copy.deepcopy(original_rds)

for o_ro, o_rd in zip(ros, rds):
    points = plot_point((o_ro, o_rd))
    reverse_points = plot_point((points[-1][0], -points[-1][1] / np.linalg.norm(-points[-1][1])))
    # reverse_points = plot_point((points[-1][0], -points[-1][1]))
    # ax.scatter(
    #     [p[0][0] for p in points],
    #     [p[0][2] for p in points],
    #     [p[0][1] for p in points],
    #     c="red",
    #     s=1,
    # )
    # ax.scatter(
    #     [p[0][0] for p in reverse_points],
    #     [p[0][2] for p in reverse_points],
    #     [p[0][1] for p in reverse_points],
    #     c="blue",
    #     s=1,
    # )


def magnitude_squared(vec):
    sum = 0
    for e in vec:
        sum += e * e

    return sum


# def runge_kutte_4th_order(
#     h, devirative_f, variable_derivative_arg, *constant_derivative_args
# ):

#     # k_1 = h * devirative_f(x_0, y_0)
#     # k_2 = h * devirative_f(x_0 + 0.5 * h, y_0 + 0.5 * k_1)
#     # k_3 = h * devirative_f(x_0 + 0.5 * h, y_0 + 0.5 * k_2)
#     # k_4 = h * devirative_f(x_0 + h, y_0 + k_3)

#     k_1 = h * devirative_f(variable_derivative_arg, *constant_derivative_args)
#     k_2 = h * devirative_f(
#         variable_derivative_arg + 0.5 * k_1, *constant_derivative_args
#     )
#     k_3 = h * devirative_f(
#         variable_derivative_arg + 0.5 * k_2, *constant_derivative_args
#     )
#     k_4 = h * devirative_f(variable_derivative_arg + k_3, *constant_derivative_args)

#     return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0


def ro_derivative(rd):
    return rd.copy()


def rd_derivative(ro, h2):
    return -1.5 * h2 * ro / np.power(magnitude_squared(ro), 2.5)


# ros = copy.deepcopy(original_ros)
# rds = copy.deepcopy(original_rds)

# for o_ro, o_rd in zip(ros, rds):
#     points = []
#     ro = copy.deepcopy(o_ro)
#     rd = copy.deepcopy(o_rd)

#     h2 = sqrnorm(np.cross(ro, rd))  # [:,np.newaxis]
#     for i in range(int(8 / DT)):

#         # cannot do inline
#         ro_inc = runge_kutte_4th_order(DT, ro_derivative, rd)
#         rd_inc = runge_kutte_4th_order(DT, rd_derivative, ro, h2)

#         ro += ro_inc
#         rd += rd_inc

#         # print(ro, rd)

#         points.append(ro.copy())

#         if BREAK_ON_EH and np.linalg.norm(ro) < 1.0:  # 2 * G * BH_MASS:
#             break

    # ax.plot(
    #     [p[0] for p in points], [p[2] for p in points], [p[1] for p in points], c="orange"
    # )


# drawing black hole position and event horizon
points = []
for i in range(1000):
    p = np.array([0, np.sin(np.pi * 2 * i / 1000), np.cos(np.pi * 2 * i / 1000)])

    points.append(p)
ax.plot(
    [p[0] for p in points],
    [p[2] for p in points],
    [p[1] for p in points],
    c="black",
)
ax.scatter([0], [0], [0], c="black")


ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_zlabel("y")
ax.set_aspect("equal")


plt.show()
