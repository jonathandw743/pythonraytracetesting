# from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import csv
import copy
import json
import math

DEFAULT_POINT = {"dt": 0.1, "x": 10.0, "y": 10.0, "dx": 1.0, "dy": 0.0, "col": "grey"}
RENDER_OBJECT = True

def load_points():
    with open("for_presentation.json", "r", encoding="utf-8") as f:
        points = json.load(f)
    for point in points:
        for key in DEFAULT_POINT:
            if key not in point:
                point[key] = DEFAULT_POINT[key]
    return points


def magnitude_squared(vec):
    sum = 0
    for e in vec:
        sum += e * e
    return sum


def rd_derivative(ro, h2):
    return -1.5 * h2 * ro / np.power(magnitude_squared(ro), 2.5)


c = np.array([-10.0, 0.0, 0.0])
r = 2.5


def process_point(point):
    path = []

    ro = np.array([point["x"], point["y"], 0.0])
    rd = np.array([point["dx"], point["dy"], 0.0])

    h2 = magnitude_squared(np.cross(ro, rd))
    # print(h2)

    j = 0
    while True:
        if RENDER_OBJECT:
            point["dt"] = min(np.linalg.norm(ro) * 0.2, np.linalg.norm(ro - c) - r)
            if np.linalg.norm(ro - c) - r < np.linalg.norm(ro) * 0.2:
                points = []
                for i in range(1000):
                    p = np.array([ro[0] + point["dt"] * np.cos(np.pi * 2 * i / 1000), ro[1] + point["dt"] * np.sin(np.pi * 2 * i / 1000)])
                    points.append(p)
                path_ax.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    c="grey",
                )
        else:
            point["dt"] = np.linalg.norm(ro) * 0.2

        ro_k1 = point["dt"] * rd
        rd_k1 = point["dt"] * rd_derivative(ro, h2)
        ro_k2 = point["dt"] * (rd + 0.5 * rd_k1)
        rd_k2 = point["dt"] * rd_derivative(ro + 0.5 * ro_k1, h2)
        ro_k3 = point["dt"] * (rd + 0.5 * rd_k2)
        rd_k3 = point["dt"] * rd_derivative(ro + 0.5 * ro_k2, h2)
        ro_k4 = point["dt"] * (rd + rd_k3)
        rd_k4 = point["dt"] * rd_derivative(ro + ro_k3, h2)
        delta_ro = (ro_k1 + 2.0 * ro_k2 + 2.0 * ro_k3 + ro_k4) / 6.0
        delta_rd = (rd_k1 + 2.0 * rd_k2 + 2.0 * rd_k3 + rd_k4) / 6.0
        ro += delta_ro
        rd += delta_rd
        if np.linalg.norm(ro) > np.linalg.norm(np.array([point["x"], point["y"], 0.0])):
            break
        if j > 30:
            break

        path.append((ro.copy(), rd.copy()))

        if np.linalg.norm(ro) < 1.0:
            break
        j += 1
    return path


FIG_SIZE = 10
NUM_FIGS = 1
fig, axes = plt.subplots(1, NUM_FIGS, figsize=(NUM_FIGS * FIG_SIZE, FIG_SIZE))

path_ax = axes
path_ax.set_aspect("equal")
points = []
for i in range(1000):
    p = np.array([np.cos(np.pi * 2 * i / 1000), np.sin(np.pi * 2 * i / 1000)])
    points.append(p)
path_ax.plot(
    [p[0] for p in points],
    [p[1] for p in points],
    c="black",
)
path_ax.scatter([0], [0], c="black")
if RENDER_OBJECT:
    points = []
    for i in range(1000):
        p = np.array([c[0] + r * np.cos(np.pi * 2 * i / 1000), c[1] + r * np.sin(np.pi * 2 * i / 1000)])
        points.append(p)
    path_ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        c="orange",
    )

for point in load_points():
    path = process_point(point)
    xs = [p[0][0] for p in path]
    ys = [p[0][1] for p in path]
    path_ax.scatter(
        xs, ys, c=point["col"], s=20
    )
    path_ax.plot(
        xs, ys, c=point["col"]
    )

plt.tight_layout()
plt.show()