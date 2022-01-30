"""
Miscellaneous aperture field analysis functions.

Grace E. Chesmore
2021
"""

import sys
import os
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
from DEFAULTS import PARENT_PATH

import numba as nb
from numba import jit, guvectorize
import numpy as np

from tele_geo_jit import *
from root_finder import brentq, brentq_arg, bisection_arg

import pan_mod_jit as pm
from pan_mod_jit import *
y_cent_m1 = -7201.003729431267

adj_pos_m1, adj_pos_m2 = pm.get_single_vert_adj_positions()
adj_pos_m1, adj_pos_m2 = np.array(adj_pos_m1), np.array(adj_pos_m2)

@jit(nopython=True, fastmath=True)
def root_z2(t, x_0, y_0, z_0, alpha, beta, gamma):
    # Endpoint of ray:
    x = x_0 + alpha * t
    y = y_0 + beta * t
    z = z_0 + gamma * t

    # Convert to M2 r.f.
    xm2, ym2, zm2 = tele_into_m2(x, y, z)

    # Z of mirror in M2 r.f.
    z_m2 = z2(xm2, ym2)
    return zm2 - z_m2

@jit(nopython=True, fastmath=True)
def root_z1(t, x_m2, y_m2, z_m2, alpha, beta, gamma):

    # Endpoint of ray:
    x = x_m2 + alpha * t
    y = y_m2 + beta * t
    z = z_m2 + gamma * t

    # Convert to M1 r.f.
    xm1, ym1, zm1 = tele_into_m1(x, y, z)

    # Z of mirror in M1 r.f.
    z_m1 = z1(xm1, ym1)
    return zm1 - z_m1

@jit(nopython=True, parallel=True, fastmath=True)
def ray_mirror_pts(P_rx, tg_th2, tg_F_2, theta, phi):
    theta_N = len(theta)
    phi_N = len(phi)
    theta = theta.repeat(theta_N).reshape((theta_N, theta_N)).T.flatten()
    phi = phi.repeat(phi_N)
    # Read in telescope geometry values
    th2 = tg_th2
    focal = tg_F_2

    n_pts = theta_N*theta_N
    out = np.zeros((6, n_pts))
    # P_rx = np.array([P_rx_x, P_rx_y, P_rx_z])


    for ii in nb.prange(n_pts):
        # initialize arrays
        N_hat_t = np.zeros(3)
        tan_rx_m2_t = np.zeros(3)
        tan_og_t = np.zeros(3)
        P_m2 = np.zeros(3)
        D_rxm2 = np.zeros(3) # direction of a ray pointing from receiver to mirror2
        P_m1 = np.zeros(3)

        # Define the outgoing ray's direction
        th = theta[ii]
        ph = phi[ii]
        # alpha2 = np.sin(th) * np.cos(ph)
        # beta2 = np.sin(th) * np.sin(ph)
        # gamma2 = np.cos(th)
        D_rxm2 = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        # Use a root finder to find where the ray intersects with M2

        t_m2 = brentq_arg(root_z2, focal + 1e3, focal + 13e3, 
                        (P_rx[0], P_rx[2], P_rx[2], D_rxm2[0], D_rxm2[1], D_rxm2[2]))

        # Endpoint of ray:
        P_m2 = P_rx + D_rxm2 * t_m2

        ########## M2 r.f ###########################################################

        x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m2(P_m2[0], P_m2[1], P_m2[2])
        x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m2(P_rx[0], P_rx[1], P_rx[2])

        # Normal vector of ray on M2
        norm = d_z2(x_m2_temp, y_m2_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(np.sum(norm_temp ** 2))

        # Normalized vector from RX to M2
        vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2))
        tan_rx_m2 = vec_rx_m2 / dist_rx_m2

        # Vector of outgoing ray
        tan_og = tan_rx_m2 - 2 * np.dot(tan_rx_m2, N_hat)*N_hat

        # Transform back to telescope cordinates

        # N_hat_x_temp = N_hat[0] * np.cos(np.pi) - N_hat[1] * np.sin(np.pi)
        # N_hat_y_temp = N_hat[0] * np.sin(np.pi) + N_hat[1] * np.cos(np.pi)
        # N_hat_z_temp = N_hat[2]
        N_hat_x_temp = N_hat[0] * -1.0
        N_hat_y_temp =  N_hat[1] * -1.0
        N_hat_z_temp = N_hat[2]

        N_hat_t[0] = N_hat_x_temp
        N_hat_t[1] = N_hat_y_temp * np.cos(th2) - N_hat_z_temp * np.sin(th2)
        N_hat_t[2] = N_hat_y_temp * np.sin(th2) + N_hat_z_temp * np.cos(th2)

        # tan_rx_m2_x_temp = tan_rx_m2[0] * np.cos(np.pi) - tan_rx_m2[1] * np.sin(np.pi)
        # tan_rx_m2_y_temp = tan_rx_m2[0] * np.sin(np.pi) + tan_rx_m2[1] * np.cos(np.pi)
        # tan_rx_m2_z_temp = tan_rx_m2[2]
        tan_rx_m2_x_temp = tan_rx_m2[0] * -1.0
        tan_rx_m2_y_temp = tan_rx_m2[1] * -1.0
        tan_rx_m2_z_temp = tan_rx_m2[2]

        tan_rx_m2_t[0] = tan_rx_m2_x_temp
        tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(th2) - tan_rx_m2_z_temp * np.sin(th2)
        tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(th2) + tan_rx_m2_z_temp * np.cos(th2)
        
        # tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
        # tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
        # tan_og_z_temp = tan_og[2]
        tan_og_x_temp = tan_og[0] * -1.0
        tan_og_y_temp = tan_og[1] * -1.0
        tan_og_z_temp = tan_og[2]

        tan_og_t[0] = tan_og_x_temp
        tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
        tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)

        ########## Tele. r.f ###########################################################

        # Vector of outgoing ray:
        # alpha1 = tan_og_t[0]
        # beta1 = tan_og_t[1]
        # gamma1 = tan_og_t[2]

        # Use a root finder to find where the ray intersects with M1
        t_m1 = bisection_arg(root_z1, 50, 22000,
                                 (P_m2[0], P_m2[1], P_m2[2], tan_og_t[0], tan_og_t[1], tan_og_t[2]))
        # Endpoint of ray:

        P_m1 = P_m2 + tan_og_t * t_m1

        # P_m1[0] = P_m2[0] + tan_og_t[0] * t_m1
        # P_m1[1] = P_m2[1] + tan_og_t[1] * t_m1
        # P_m1[2] = P_m2[2] + tan_og_t[2] * t_m1

        # Write out
        out[0, ii] = P_m2[0]
        out[1, ii] = P_m2[1]
        out[2, ii] = P_m2[2]

        out[3, ii] = P_m1[0]
        out[4, ii] = P_m1[1]
        out[5, ii] = P_m1[2]

    return out

@jit(nopython=True, fastmath=True)
def root_z2_pm(t, x_0, y_0, z_0, alpha, beta, gamma, a, b, c, d, e, x0, y0):
    x = x_0 + alpha * t
    y = y_0 + beta * t
    z = z_0 + gamma * t
    xm2, ym2, zm2 = tele_into_m2(
        x, y, z
    )  # Convert ray's endpoint into M2 coordinates

    if z_0 != 0:
        z /= np.cos(np.arctan(1 / 3))
    xm2_err, ym2_err, zm2_err = tele_into_m2(
        x, y, z
    )  # Convert ray's endpoint into M2 coordinates

    # x_temp = xm1_err * np.cos(np.pi) + zm1_err * np.sin(np.pi)
    x_temp = xm2_err * -1.0 

    xpc = x_temp - x0
    ypc = ym2_err - y0

    z_err = (
        a
        + b * xpc
        + c * (ypc)
        + d * (xpc ** 2 + ypc ** 2)
        + e * (xpc * ypc)
        )   

    z_m2 = z2(xm2, ym2)  # Z of mirror in M2 coordinates

    root = zm2 - (z_m2 + z_err)
    return root

@jit(nopython=True, fastmath=True)
def root_z1_pm(t, x_m2, y_m2, z_m2, alpha, beta, gamma, a, b, c, d, e, x0, y0):
    x = x_m2 + alpha * t
    y = y_m2 + beta * t
    z = z_m2 + gamma * t
    xm1, ym1, zm1 = tele_into_m1(
        x, y, z
    )  # take ray end coordinates and convert to M1 coordinates

    xm1_err, ym1_err, zm1_err = tele_into_m1(x, y, z)

    # x_temp = xm1_err * np.cos(np.pi) + zm1_err * np.sin(np.pi)
    # y_temp = ym1_err
    # z_temp = -xm1_err * np.sin(np.pi) + zm1_err * np.cos(np.pi)
    x_temp = xm1_err * -1.0 
    y_temp = ym1_err
    z_temp =  zm1_err * -1.0

    xpc = x_temp - x0
    ypc = y_temp - y0

    z_err = (
        a
        + b * xpc
        + c * (ypc)
        + d * (xpc ** 2 + ypc ** 2)
        + e * (xpc * ypc)
    )

    z_m1 = z1(xm1, ym1)  # Z of mirror 1 in M1 coordinates
    root = zm1 - (z_m1 + z_err)
    return root

@jit(nopython=True, parallel=True, fastmath=True)
def aperature_fields_from_panel_model(
    panel_model1, panel_model2, P_rx, tg_th_1,
    tg_th2, tg_z_ap, tg_th_fwhp, tg_F_2, theta, phi, rxmirror
    ):
    theta_N = len(theta)
    phi_N = len(phi)
    th_mean = np.mean(theta)
    ph_mean = np.mean(phi)
    theta = theta.repeat(theta_N).reshape((theta_N, theta_N)).T.flatten()
    phi = phi.repeat(phi_N)

    th_1 = tg_th_1
    th2 = tg_th2
    z_ap = tg_z_ap * 1e3
    horn_fwhp = tg_th_fwhp
    focal = tg_F_2
    # Step 1:  grid the plane of rays shooting out of receiver feed

    col_m2 = adj_pos_m2[0]
    row_m2 = adj_pos_m2[1]
    x_adj_m2 = adj_pos_m2[4]
    y_adj_m2 = adj_pos_m2[3]
    x_panm_m2 = rxmirror[0, :].reshape(phi_N, phi_N)
    y_panm_m2 = rxmirror[2, :].reshape(phi_N, phi_N)
    pan_id_m2 = identify_panel(x_panm_m2, y_panm_m2, x_adj_m2, y_adj_m2, col_m2, row_m2)

    col_m1 = adj_pos_m1[0]
    row_m1 = adj_pos_m1[1]
    x_adj_m1 = adj_pos_m1[2]
    y_adj_m1 = adj_pos_m1[3]
    x_panm_m1 = rxmirror[3, :].reshape(phi_N, phi_N)
    y_panm_m1 = rxmirror[4, :].reshape(phi_N, phi_N)
    pan_id_m1 = identify_panel(x_panm_m1, y_panm_m1 - y_cent_m1, x_adj_m1, y_adj_m1, col_m1, row_m1)

    row_panm_m2 = np.ravel(pan_id_m2[0, :, :])
    col_panm_m2 = np.ravel(pan_id_m2[1, :, :])
    row_panm_m1 = np.ravel(pan_id_m1[0, :, :])
    col_panm_m1 = np.ravel(pan_id_m1[1, :, :])

    # # Step 2: calculate the position + local surface normal for the dish
    n_pts = theta_N*theta_N
    out = np.zeros((17, n_pts))
    out[4, :] = y_cent_m1

    for ii in nb.prange(n_pts):
        # initialize arrays
        N_hat_t = np.zeros(3)
        tan_rx_m2_t = np.zeros(3)
        tan_og_t = np.zeros(3)
        tan_m2_m1_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        i_row = row_panm_m2[ii]
        i_col = col_panm_m2[ii]
        i_panm = np.where((panel_model2[0, :] == i_row) & (panel_model2[1, :] == i_col))

        if len(i_panm[0]) != 0:

            th = theta[ii]
            ph = phi[ii]
            r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

            alpha = r_hat[0]
            beta = r_hat[1]
            gamma = r_hat[2]

            # Receiver feed position [mm] (in telescope reference frame):
            x_0 = P_rx[0]
            y_0 = P_rx[1]
            z_0 = P_rx[2]
            a = panel_model2[2][i_panm].item()
            b = panel_model2[3][i_panm].item()
            c = panel_model2[4][i_panm].item()
            d = panel_model2[5][i_panm].item()
            e = panel_model2[6][i_panm].item()
            f = panel_model2[7][i_panm].item()
            x0 = panel_model2[8][i_panm].item()
            y0 = panel_model2[9][i_panm].item()


            t_m2 = brentq_arg(root_z2_pm, focal + 1000, focal + 12000, 
                            (x_0, y_0, z_0, alpha, beta, gamma, a, b, c, d, e, x0, y0))

            # Location of where ray hits M2
            x_m2 = x_0 + alpha * t_m2
            y_m2 = y_0 + beta * t_m2
            z_m2 = z_0 + gamma * t_m2

            # Using x and y in M2 coordiantes, find the z err:

            P_m2 = np.array([x_m2, y_m2, z_m2])

            ###### in M2 coordinates ##########################
            x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m2(
                x_m2, y_m2, z_m2
            )  # P_m2 temp
            x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m2(x_0, y_0, z_0)  # P_rx temp
            norm = d_z2(x_m2_temp, y_m2_temp)
            norm_temp = np.array([-norm[0], -norm[1], 1])
            N_hat = norm_temp / np.sqrt(np.sum(norm_temp ** 2))
            vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array(
                [x_rx_temp, y_rx_temp, z_rx_temp]
            )
            dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2))
            tan_rx_m2 = vec_rx_m2 / dist_rx_m2

            # Outgoing ray
            tan_og = tan_rx_m2 - 2 * np.dot(tan_rx_m2, N_hat)*N_hat

            # Transform back to telescope cordinates
            # N_hat_x_temp = N_hat[0] * np.cos(np.pi) - N_hat[1] * np.sin(np.pi)
            # N_hat_y_temp = N_hat[0] * np.sin(np.pi) + N_hat[1] * np.cos(np.pi)
            N_hat_x_temp = N_hat[0] * -1.0 
            N_hat_y_temp = N_hat[1] * -1.0
            N_hat_z_temp = N_hat[2]

            N_hat_t[0] = N_hat_x_temp
            N_hat_t[1] = N_hat_y_temp * np.cos(th2) - N_hat_z_temp * np.sin(th2)
            N_hat_t[2] = N_hat_y_temp * np.sin(th2) + N_hat_z_temp * np.cos(th2)



            # tan_rx_m2_x_temp = tan_rx_m2[0] * np.cos(np.pi) - tan_rx_m2[1] * np.sin(np.pi)
            # tan_rx_m2_y_temp = tan_rx_m2[0] * np.sin(np.pi) + tan_rx_m2[1] * np.cos(np.pi)
            tan_rx_m2_x_temp = tan_rx_m2[0] * -1.0
            tan_rx_m2_y_temp = tan_rx_m2[1] * -1.0
            tan_rx_m2_z_temp = tan_rx_m2[2]

            tan_rx_m2_t[0] = tan_rx_m2_x_temp
            tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(th2) - tan_rx_m2_z_temp * np.sin(
                th2
            )
            tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(th2) + tan_rx_m2_z_temp * np.cos(
                th2
            )


            # tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
            # tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
            tan_og_x_temp = tan_og[0] * -1.0
            tan_og_y_temp = tan_og[1] * -1.0
            tan_og_z_temp = tan_og[2]

            tan_og_t[0] = tan_og_x_temp
            tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
            tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)
            ##################################################

            alpha = tan_og_t[0]
            beta = tan_og_t[1]
            gamma = tan_og_t[2]

            i_row = row_panm_m1[ii]
            i_col = col_panm_m1[ii]
            i_panm = np.where(
                (panel_model1[0, :] == i_row) & (panel_model1[1, :] == i_col)
            )
            if len(i_panm[0]) != 0:
                a = panel_model1[2][i_panm].item()
                b = panel_model1[3][i_panm].item()
                c = panel_model1[4][i_panm].item()
                d = panel_model1[5][i_panm].item()
                e = panel_model1[6][i_panm].item()
                f = panel_model1[7][i_panm].item()
                x0 = panel_model1[8][i_panm].item()
                y0 = panel_model1[9][i_panm].item()

                t_m1 = brentq_arg(root_z1_pm, 500, 22000, 
                            (P_m2[0], P_m2[1], P_m2[2], alpha, beta, gamma, a, b, c, d, e, x0, y0))

                # Location of where ray hits M1
                x_m1 = P_m2[0] + alpha * t_m1
                y_m1 = P_m2[1] + beta * t_m1
                z_m1 = P_m2[2] + gamma * t_m1
                P_m1 = np.array([x_m1, y_m1, z_m1])

                ###### in M1 cordinates ##########################
                x_m1_temp, y_m1_temp, z_m1_temp = tele_into_m1(
                    x_m1, y_m1, z_m1
                )  # P_m2 temp
                x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m1(
                    P_m2[0], P_m2[1], P_m2[2]
                )  # P_rx temp
                norm = d_z1(x_m1_temp, y_m1_temp)
                norm_temp = np.array([-norm[0], -norm[1], 1])
                N_hat = norm_temp / np.sqrt(np.sum(norm_temp ** 2))
                vec_m2_m1 = np.array([x_m1_temp, y_m1_temp, z_m1_temp]) - np.array(
                    [x_m2_temp, y_m2_temp, z_m2_temp]
                )
                dist_m2_m1 = np.sqrt(np.sum(vec_m2_m1 ** 2))
                tan_m2_m1 = vec_m2_m1 / dist_m2_m1

                # Outgoing ray
                tan_og = tan_m2_m1 - 2 * np.dot(tan_m2_m1, N_hat)*N_hat

                # Transform back to telescope cordinates
                # N_x_temp = N_hat[0] * np.cos(np.pi) + N_hat[2] * np.sin(np.pi)
                # N_y_temp = N_hat[1]
                # N_z_temp = -N_hat[0] * np.sin(np.pi) + N_hat[2] * np.cos(np.pi)
                N_x_temp = N_hat[0] * -1.0
                N_y_temp = N_hat[1]
                N_z_temp = N_hat[2] * -1.0

                N_hat_t[0] = N_x_temp
                N_hat_t[1] = N_y_temp * np.cos(th_1) - N_z_temp * np.sin(th_1)
                N_hat_t[2] = N_y_temp * np.sin(th_1) + N_z_temp * np.cos(th_1)

                # tan_m2_m1_x_temp = tan_m2_m1[0] * np.cos(np.pi) + tan_m2_m1[2] * np.sin(np.pi)
                # tan_m2_m1_y_temp = tan_m2_m1[1]
                # tan_m2_m1_z_temp = -tan_m2_m1[0] * np.sin(np.pi) + tan_m2_m1[2] * np.cos(np.pi)
                tan_m2_m1_x_temp = tan_m2_m1[0] * -1.0 
                tan_m2_m1_y_temp = tan_m2_m1[1]
                tan_m2_m1_z_temp = tan_m2_m1[2] * -1.0

                tan_m2_m1_t[0] = tan_m2_m1_x_temp
                tan_m2_m1_t[1] = tan_m2_m1_y_temp * np.cos(th_1) - tan_m2_m1_z_temp * np.sin(th_1)
                tan_m2_m1_t[2] = tan_m2_m1_y_temp * np.sin(th_1) + tan_m2_m1_z_temp * np.cos(th_1)

                # tan_og_x_temp = tan_og[0] * np.cos(np.pi) + tan_og[2] * np.sin(np.pi)
                # tan_og_y_temp = tan_og[1]
                # tan_og_z_temp = -tan_og[0] * np.sin(np.pi) + tan_og[2] * np.cos(np.pi)
                tan_og_x_temp = tan_og[0] * -1.0
                tan_og_y_temp = tan_og[1]
                tan_og_z_temp = tan_og[2] * -1.0

                tan_og_t[0] = tan_og_x_temp
                tan_og_t[1] = tan_og_y_temp * np.cos(th_1) - tan_og_z_temp * np.sin(th_1)
                tan_og_t[2] = tan_og_y_temp * np.sin(th_1) + tan_og_z_temp * np.cos(th_1)

                ##################################################

                dist_m1_ap = np.abs((z_ap - P_m1[2]) / tan_og_t[2])
                total_path_length = t_m2 + t_m1 + dist_m1_ap
                # total_path_length = dist_rx_m2 + dist_m2_m1 + dist_m1_ap
                pos_ap = P_m1 + dist_m1_ap * tan_og_t

                # # Estimate theta
                # de_ve = np.arctan(tan_rx_m2_t[2] / (-tan_rx_m2_t[1]))
                # de_ho = np.arctan(
                #     tan_rx_m2_t[0] / np.sqrt(tan_rx_m2_t[1] ** 2 + tan_rx_m2_t[2] ** 2)
                # )

                # Write out
                out[0, ii] = x_m2
                out[1, ii] = y_m2
                out[2, ii] = z_m2

                out[3, ii] = x_m1
                out[4, ii] = y_m1
                out[5, ii] = z_m1

                out[6, ii] = N_hat_t[0]
                out[7, ii] = N_hat_t[1]
                out[8, ii] = N_hat_t[2]

                out[9, ii] = pos_ap[0]
                out[10, ii] = pos_ap[1]
                out[11, ii] = pos_ap[2]

                out[12, ii] = tan_og_t[0]
                out[13, ii] = tan_og_t[1]
                out[14, ii] = tan_og_t[2]

                out[15, ii] = total_path_length
                out[16, ii] = np.exp(
                    (-0.5)
                    * ((th - th_mean) ** 2 + (ph - ph_mean) ** 2)
                    / (horn_fwhp / (np.sqrt(8 * np.log(2)))) ** 2
                )
    return out

if __name__ == "__main__":
    '''
    test the code--------------------------------------------------------------
    '''
    from time import time
    import ap_field as ap
    tele_geo = initialize_telescope_geometry()
    P_rx = np.array([0.0, 0.0, 0.0])
    tg_th2 = tele_geo.th2
    tg_F_2 = tele_geo.F_2
    theta_a, theta_b, theta_N = -np.pi / 2 - 0.28, -np.pi / 2 + 0.28, 64
    phi_a, phi_b, phi_N = np.pi / 2 - 0.28, np.pi / 2 + 0.28, 64
    theta = np.linspace(theta_a, theta_b, theta_N)
    phi = np.linspace(phi_a, phi_b, phi_N)


    ################## verify the ray_mirror_pts function--------------------------
    t1 = time()
    result_jit = ray_mirror_pts(P_rx, tg_th2, tg_F_2, theta, phi)
    t2 = time()
    print(f'Function ray_mirror_pts executed in {(t2-t1):.8f}s')

    t1 = time()
    result_jit = ray_mirror_pts(P_rx, tg_th2, tg_F_2, theta, phi)
    t2 = time()
    print(f'Function ray_mirror_pts executed in {(t2-t1):.8f}s (used cache)')

    t1 = time()
    result = ap.ray_mirror_pts(P_rx, tele_geo, theta, phi)
    t2 = time()
    print(f'Function ap.ray_mirror_pts executed in {(t2-t1):.8f}s')

    print(result_jit-result)

    if np.allclose(result, result_jit, rtol=1e-11, atol=1e-11):
        print("OH Yes! ray_mirror_pts DONE")
    else:
        print("FML check the ray_mirror_pts function again")

    ################## verify the aperature_fields_from_panel_model function--------------
    save = 0  # Option to save adjuster offsets to .txt file
    adj_1_A = np.random.randn(1092) * 35
    adj_2_A = np.random.randn(1092) * 35
    panel_model2 = pm.panel_model_from_adjuster_offsets(
        2, adj_2_A, 1, save
    )  # Panel Model on M2
    panel_model1 = pm.panel_model_from_adjuster_offsets(
        1, adj_1_A, 1, save
    )  # Panel Model on M1
    rxmirror = result

    tg_th_1, tg_th2, tg_z_ap = tele_geo.th_1, tele_geo.th2, tele_geo.z_ap
    tg_th_fwhp, tg_F_2 = tele_geo.th_fwhp, tele_geo.F_2

    t1 = time()    
    apfield_jit = aperature_fields_from_panel_model(panel_model1, panel_model2, \
                                        P_rx, tg_th_1, tg_th2, tg_z_ap, tg_th_fwhp, \
                                        tg_F_2, theta, phi, rxmirror
                                        )
    t2 = time()
    print(f'Function aperature_fields_from_panel_model executed in {(t2-t1):.8f}s')

    t1 = time()    
    apfield_jit = aperature_fields_from_panel_model(panel_model1, panel_model2, \
                                        P_rx, tg_th_1, tg_th2, tg_z_ap, tg_th_fwhp, \
                                        tg_F_2, theta, phi, rxmirror
                                        )
    t2 = time()
    print(f'Function aperature_fields_from_panel_model executed in {(t2-t1):.8f}s (used cache)')

    t1 = time()    
    apfield = ap.aperature_fields_from_panel_model(panel_model1, panel_model2, \
                                        P_rx, tele_geo, theta, phi, rxmirror
                                        )
    t2 = time()
    print(f'Function ap.aperature_fields_from_panel_model executed in {(t2-t1):.8f}s')

    print(apfield_jit-apfield)

    if np.allclose(apfield, apfield_jit, rtol=1e-11, atol=1e-11):
        print("OH Yes! aperature_fields_from_panel_model DONE")
    else:
        print("FML check the aperature_fields_from_panel_model function again")

    