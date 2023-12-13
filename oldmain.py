from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import csv
import copy

fig = plt.figure()
ax = plt.axes(projection="3d")

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
# DT = 0.1
G = 1.0


class Photon:
    def __init__(self, ro, rd):
        self.ro = ro
        self.rd = rd


class ConvertedPhoton:
    def __init__(self, photon):
        self.dist_to_bh = np.linalg.norm(photon.ro)
        self.photon_to_bh_norm = -photon.ro / self.dist_to_bh

        self.photon_to_bh_perp_norm = np.cross(
            np.cross(photon.rd, self.photon_to_bh_norm), self.photon_to_bh_norm
        )
        self.photon_to_bh_perp_norm /= -np.linalg.norm(self.photon_to_bh_perp_norm)

        self.converted_vel = np.array(
            [
                np.dot(photon.rd, self.photon_to_bh_perp_norm),
                np.dot(photon.rd, self.photon_to_bh_norm),
            ]
        )

    def convert(self):
        return Photon(
            -self.photon_to_bh_norm * self.dist_to_bh,
            self.photon_to_bh_perp_norm * self.converted_vel[0]
            + self.photon_to_bh_norm * self.converted_vel[1],
        )

    # def thetas(self, delta_time):
    #     newtonian_force_mag = G * BH_MASS / (self.dist_to_bh * self.dist_to_bh)
    #     theta = np.arctan2(self.converted_vel[1], self.converted_vel[0])
    #     delta_theta = abs(delta_time * newtonian_force_mag * np.sin(theta))
    #     return (theta, delta_theta)

    def curvature(self):
        newtonian_force_mag = G * BH_MASS / (self.dist_to_bh * self.dist_to_bh)
        theta = np.arctan2(self.converted_vel[1], self.converted_vel[0])
        curvature = abs(newtonian_force_mag * np.sin(theta))
        curvature /= abs(1.0 - 2.0 * G * BH_MASS / self.dist_to_bh)
        return curvature

    def apply_curvature(self, curvature, delta_time):
        acc_norm = np.array([-self.converted_vel[1], self.converted_vel[0]])
        radius = 1 / curvature
        to_centre_of_curvature = acc_norm * radius
        centre_of_curvature = np.array([0, self.dist_to_bh]) + to_centre_of_curvature
        arc_length = delta_time
        # rotate_through = arc_length / radius
        rotate_through = arc_length * curvature
        curr_angle_to = np.arctan2(
            -to_centre_of_curvature[1], -to_centre_of_curvature[0]
        )
        new_angle_to = curr_angle_to + rotate_through
        new_ro = centre_of_curvature + radius * np.array(
            [np.cos(new_angle_to), np.sin(new_angle_to)]
        )
        displacement = new_ro - np.array([0, self.dist_to_bh])

        theta = np.arctan2(self.converted_vel[1], self.converted_vel[0])
        delta_theta = curvature * delta_time
        theta += delta_theta
        new_rd = np.array([np.cos(theta), np.sin(theta)])

        displacement = (
            self.photon_to_bh_perp_norm * displacement[0]
            + self.photon_to_bh_norm * displacement[1]
        )
        new_rd = (
            self.photon_to_bh_perp_norm * new_rd[0] + self.photon_to_bh_norm * new_rd[1]
        )

        photon = self.convert()
        photon.ro += displacement
        photon.rd = new_rd
        return photon

    # def apply_theta(self, theta):
    #     self.converted_vel = np.array([np.cos(theta), np.sin(theta)])


# tested ish
def displacement_along_constant_curvature(initial_dir, curvature, distance) -> np.array:
    radius = 1 / curvature
    to_centre_of_curvature_norm = np.array([-initial_dir[1], initial_dir[0]])
    delta_theta = curvature * distance
    angle_of_point = (
        np.arctan2(initial_dir[1], initial_dir[0]) + delta_theta - np.pi * 0.5
    )
    return radius * (
        to_centre_of_curvature_norm
        + np.array([np.cos(angle_of_point), np.sin(angle_of_point)])
    )


# print(displacement_along_constant_curvature(np.array([1, 0]), 1, np.pi))


class ConvertedPhoton2:
    def __init__(self, photon):
        self.dist_to_bh = np.linalg.norm(photon.ro)
        self.photon_to_bh_norm = -photon.ro / self.dist_to_bh

        self.photon_to_bh_perp_norm = np.cross(
            np.cross(photon.rd, self.photon_to_bh_norm), self.photon_to_bh_norm
        )
        self.photon_to_bh_perp_norm /= -np.linalg.norm(self.photon_to_bh_perp_norm)

        self.converted_vel = np.array(
            [
                np.dot(photon.rd, self.photon_to_bh_perp_norm),
                np.dot(photon.rd, self.photon_to_bh_norm),
            ]
        )

    def convert(self):
        return Photon(
            -self.photon_to_bh_norm * self.dist_to_bh,
            self.photon_to_bh_perp_norm * self.converted_vel[0]
            + self.photon_to_bh_norm * self.converted_vel[1],
        )

    # def thetas(self, delta_time):
    #     newtonian_force_mag = G * BH_MASS / (self.dist_to_bh * self.dist_to_bh)
    #     theta = np.arctan2(self.converted_vel[1], self.converted_vel[0])
    #     delta_theta = abs(delta_time * newtonian_force_mag * np.sin(theta))
    #     return (theta, delta_theta)

    def curvature(self):
        newtonian_force_mag = G * BH_MASS / (self.dist_to_bh * self.dist_to_bh)
        theta = np.arctan2(self.converted_vel[1], self.converted_vel[0])
        curvature = abs(newtonian_force_mag * np.sin(theta))
        curvature /= abs(1.0 - 2.0 * G * BH_MASS / self.dist_to_bh)
        return curvature

    def get_next_photon(self, delta_time: float, accuracy: int):
        # this is so fucked
        curvature = self.curvature()
        rd_perp = np.array([-self.converted_vel[1], self.converted_vel[0]])
        radius = 1 / curvature
        to_centre_of_curvature = rd_perp * radius
        
        to_mp = displacement_along_constant_curvature(self.converted_vel, curvature, delta_time * 0.5)
        real_to_mp = (
            self.photon_to_bh_perp_norm * to_mp[0]
            + self.photon_to_bh_norm * to_mp[1]
        )
        to_est_next = displacement_along_constant_curvature(self.converted_vel, curvature, delta_time)
        real_to_est_next = (
            self.photon_to_bh_perp_norm * to_est_next[0]
            + self.photon_to_bh_norm * to_est_next[1]
        )

        theta = np.arctan2(self.converted_vel[1], self.converted_vel[0])
        delta_theta = curvature * delta_time
        theta += delta_theta
        new_rd = np.array([np.cos(theta), np.sin(theta)])
        new_rd = (
            self.photon_to_bh_perp_norm * new_rd[0] + self.photon_to_bh_norm * new_rd[1]
        )

        est_next_photon = self.convert()
        est_next_photon.ro += real_to_est_next
        est_next_photon.rd = new_rd

        for _ in range(accuracy):

            est_next_con_photon = ConvertedPhoton2(est_next_photon)
            est_next_curvature = est_next_con_photon.curvature()
            est_next_to_mp = displacement_along_constant_curvature(est_next_con_photon.converted_vel, est_next_curvature, -delta_time * 0.5)
            real_est_next_to_mp = (
                est_next_con_photon.photon_to_bh_perp_norm * est_next_to_mp[0]
                + est_next_con_photon.photon_to_bh_norm * est_next_to_mp[1]
            )

            real_to_est_next = real_to_mp - real_est_next_to_mp

            theta = np.arctan2(self.converted_vel[1], self.converted_vel[0])
            theta += curvature * delta_time * 0.5
            theta += est_next_curvature * delta_time * 0.5
            new_rd = np.array([np.cos(theta), np.sin(theta)])
            new_rd = (
                self.photon_to_bh_perp_norm * new_rd[0] + self.photon_to_bh_norm * new_rd[1]
            )

            est_next_photon = self.convert()
            est_next_photon.ro += real_to_est_next
            est_next_photon.rd = new_rd

        return est_next_photon


def sqrnorm(vec):
    return np.einsum('...i,...i',vec,vec)

BREAK_ON_EH = False







DT = 0.1

ros = copy.deepcopy(original_ros)
rds = copy.deepcopy(original_rds)


for ro, rd in zip(ros, rds):
    points = []
    photon = Photon(ro, rd)
    h2 = sqrnorm(np.cross(photon.ro,photon.rd))#[:,np.newaxis]
    for i in range(int(8 / DT)):
        photon.ro += photon.rd * DT
        accel = - 1.5 * h2 *  photon.ro / np.power(sqrnorm(photon.ro),2.5)#[:,np.newaxis]
        photon.rd += accel * DT

        # ro += rd * DT * 0.5
        # rd = new_rd(ro, rd)
        # ro += rd * DT * 0.5

        # print(np.linalg.norm(rd))

        points.append(photon.ro.copy())

        if BREAK_ON_EH and np.linalg.norm(photon.ro) < 2 * G * BH_MASS:
            break

    ax.plot(
        [p[0] for p in points], [p[2] for p in points], [p[1] for p in points], c="purple"
    )







def RK4f(y,h2):
    f = np.zeros(y.shape)
    
    f[0:3] = y[3:6]
    f[3:6] = - 1.5 * h2 * y[0:3] / np.power(sqrnorm(y[0:3]),2.5)
    return f

ros = copy.deepcopy(original_ros)
rds = copy.deepcopy(original_rds)


x = np.array([[0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5]])[:,0:3]
print(x)

for ro, rd in zip(ros, rds):
    points = []
    photon = Photon(ro, rd)
    h2 = sqrnorm(np.cross(photon.ro,photon.rd))#[:,np.newaxis]
    for i in range(int(8 / DT)):
        #simple step size control
        rkstep = DT

        # standard Runge-Kutta
        y = np.array([photon.ro[0], photon.ro[1], photon.ro[2], photon.rd[0], photon.rd[1], photon.rd[2]])
        k1 = RK4f( y, h2)
        k2 = RK4f( y + 0.5*rkstep*k1, h2)
        k3 = RK4f( y + 0.5*rkstep*k2, h2)
        k4 = RK4f( y +     rkstep*k3, h2)

        increment = rkstep/6. * (k1 + 2*k2 + 2*k3 + k4)
        
        photon.rd += increment[3:6]

        photon.ro += increment[0:3]

        points.append(photon.ro.copy())

        if BREAK_ON_EH and np.linalg.norm(photon.ro) < 2 * G * BH_MASS:
            break

    ax.plot(
        [p[0] for p in points], [p[2] for p in points], [p[1] for p in points], c="pink"
    )


# old curvature
# ros = copy.deepcopy(original_ros)
# rds = copy.deepcopy(original_rds)

# for ro, rd in zip(ros, rds):
#     points = []
#     photon = Photon(ro, rd)
#     for i in range(int(8 / DT)):
#         converted_photon = ConvertedPhoton2(photon)
#         # accuracy is redundant
#         photon = converted_photon.get_next_photon(DT, 10)
#         # print(photon.ro)

#         # ro += rd * DT * 0.5
#         # rd = new_rd(ro, rd)
#         # ro += rd * DT * 0.5

#         # print(np.linalg.norm(rd))

#         points.append(photon.ro.copy())

#         if BREAK_ON_EH and np.linalg.norm(photon.ro) < 2 * G * BH_MASS:
#             break

    # ax.plot(
    #     [p[0] for p in points], [p[2] for p in points], [p[1] for p in points], c="red"
    # )

# new curvature
# ros = copy.deepcopy(original_ros)
# rds = copy.deepcopy(original_rds)

# for ro, rd in zip(ros, rds):
#     points = []
#     photon = Photon(ro, rd)
#     for i in range(int(8 / DT)):
#         original_converted_photon = ConvertedPhoton(photon)
#         original_curvature = original_converted_photon.curvature()
#         photon = original_converted_photon.apply_curvature(original_curvature, DT)

#         average_curvature = original_curvature

#         for j in range(10):
#             estimated_next_converted_photon = ConvertedPhoton(photon)
#             estimated_next_curvature = estimated_next_converted_photon.curvature()
#             average_curvature = (original_curvature + estimated_next_curvature) / 2

#             photon = original_converted_photon.apply_curvature(average_curvature, DT)

#         # ro += rd * DT * 0.5
#         # rd = new_rd(ro, rd)
#         # ro += rd * DT * 0.5

#         # print(np.linalg.norm(rd))

#         points.append(photon.ro.copy())

#         if BREAK_ON_EH and np.linalg.norm(photon.ro) < 2 * G * BH_MASS:
#             break

    # ax.plot(
    #     [p[0] for p in points], [p[2] for p in points], [p[1] for p in points], c="blue"
    # )


# DT = 0.1

# old old
from old import new_rd

ros = copy.deepcopy(original_ros)
rds = copy.deepcopy(original_rds)

for ro, rd in zip(ros, rds):
    points = []
    for i in range(int(8 / DT)):
        ro += rd * DT * 0.5
        rd = new_rd(ro, rd, DT)
        ro += rd * DT * 0.5

        # print(np.linalg.norm(rd))

        points.append(ro.copy())
        if BREAK_ON_EH and np.linalg.norm(photon.ro) < 2 * G * BH_MASS:
            break
    ax.plot(
        [p[0] for p in points],
        [p[2] for p in points],
        [p[1] for p in points],
        c="orange",
    )


# gold standard
DT = 0.01

ros = copy.deepcopy(original_ros)
rds = copy.deepcopy(original_rds)

for ro, rd in zip(ros, rds):
    points = []
    for i in range(int(8 / DT)):
        ro += rd * DT * 0.5
        rd = new_rd(ro, rd, DT)
        ro += rd * DT * 0.5

        # print(np.linalg.norm(rd))

        points.append(ro.copy())
        if BREAK_ON_EH and np.linalg.norm(photon.ro) < 2 * G * BH_MASS:
            break
    ax.plot(
        [p[0] for p in points],
        [p[2] for p in points],
        [p[1] for p in points],
        c="green",
    )


# ros = copy.deepcopy(original_ros)
# rds = copy.deepcopy(original_rds)
# points = []

# for ro, rd in zip(ros, rds):
#     for i in range(20):
#         dist_to_bh = np.linalg.norm(ro - BH_POS)

#         photon_to_bh_norm = (BH_POS - ro) / dist_to_bh
#         photon_to_bh_perp_norm = np.cross(np.cross(rd, photon_to_bh_norm), photon_to_bh_norm)
#         photon_to_bh_perp_norm /= -np.linalg.norm(photon_to_bh_perp_norm)

#         converted_vel = np.array([np.dot(rd, photon_to_bh_perp_norm), np.dot(rd, photon_to_bh_norm)])

#         newtonian_force_mag = G * BH_MASS / (dist_to_bh * dist_to_bh)

#         theta = np.arctan2(converted_vel[1], converted_vel[0])

#         # delta_time = DT
#         # delta_theta = abs(DT * newtonian_force_mag * np.sin(theta))

#         delta_theta = np.pi * 0.01
#         delta_time = abs(delta_theta / (newtonian_force_mag * np.sin(theta)))

#         if delta_time < 0.025:
#             delta_time = 0.025
#         if delta_time > 1:
#             delta_time = 1

#         delta_theta = abs(DT * newtonian_force_mag * np.sin(theta))

#         theta += delta_theta

#         converted_vel = np.array([np.cos(theta), np.sin(theta)])

#         ro += rd * delta_time * 0.5
#         rd = photon_to_bh_perp_norm * converted_vel[0] + photon_to_bh_norm * converted_vel[1]
#         ro += rd * delta_time * 0.5

#         # print(i)
#         if i > 10:
#             print(photon_to_bh_perp_norm, photon_to_bh_norm)

#         # print(np.linalg.norm(rd))

#         points.append(ro.copy())
#     print("-----------")

# ax.scatter([p[0] for p in points], [p[2] for p in points], [p[1] for p in points], c="red")

points = []
for i in range(1000):
    p = np.array([0, np.sin(np.pi * 2 * i / 1000), np.cos(np.pi * 2 * i / 1000)]) * 2 * G * BH_MASS

    points.append(p)
    # if BREAK_ON_EH and np.linalg.norm(photon.ro) < 2 * G * BH_MASS:
        # break
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
