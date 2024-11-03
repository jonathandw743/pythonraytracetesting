import numpy as np

BH_MASS = 1.0
BH_POS = np.array([0.0, 0.0, 0.0])
G = 1.0


def calc(ro, rd, DT):
    dist_to_bh = np.linalg.norm(ro - BH_POS)

    photon_to_bh_norm = (BH_POS - ro) / dist_to_bh
    photon_to_bh_perp_norm = np.cross(np.cross(rd, photon_to_bh_norm), photon_to_bh_norm)
    photon_to_bh_perp_norm /= -np.linalg.norm(photon_to_bh_perp_norm)

    converted_vel = np.array([np.dot(rd, photon_to_bh_perp_norm), np.dot(rd, photon_to_bh_norm)])

    newtonian_force_mag = G * BH_MASS / (dist_to_bh * dist_to_bh)

    theta = np.arctan2(converted_vel[1], converted_vel[0])
    
    delta_theta = abs(DT * newtonian_force_mag * np.sin(theta))

    # delta_theta = np.pi * 0.01
    # delta_time = abs(delta_theta / (newtonian_force_mag * np.sin(theta)))

    theta += delta_theta

    converted_vel = np.array([np.cos(theta), np.sin(theta)])
    
    next_rd = photon_to_bh_perp_norm * converted_vel[0] + photon_to_bh_norm * converted_vel[1]

    return theta, delta_theta, next_rd

def new_rd(ro, rd, DT):
    dist_to_bh = np.linalg.norm(ro - BH_POS)

    photon_to_bh_norm = (BH_POS - ro) / dist_to_bh
    photon_to_bh_perp_norm = np.cross(np.cross(rd, photon_to_bh_norm), photon_to_bh_norm)
    photon_to_bh_perp_norm /= -np.linalg.norm(photon_to_bh_perp_norm)

    converted_vel = np.array([np.dot(rd, photon_to_bh_perp_norm), np.dot(rd, photon_to_bh_norm)])

    newtonian_force_mag = G * BH_MASS / (dist_to_bh * dist_to_bh)

    theta = np.arctan2(converted_vel[1], converted_vel[0])
    
    delta_theta = abs(DT * newtonian_force_mag * np.sin(theta))

    delta_theta /= abs(1.0 - 2.0 * G * BH_MASS / dist_to_bh)

    # delta_theta = np.pi * 0.01
    # delta_time = abs(delta_theta / (newtonian_force_mag * np.sin(theta)))

    theta += delta_theta

    converted_vel = np.array([np.cos(theta), np.sin(theta)])
    
    return photon_to_bh_perp_norm * converted_vel[0] + photon_to_bh_norm * converted_vel[1]

