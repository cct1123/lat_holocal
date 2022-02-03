"""
Miscellaneous aperture field analysis functions.

Grace E. Chesmore
2021

vectorized by 
Chun Tung Cheung
(January 2022)
"""
import numpy as np
from time import time

import sys
from pathlib import Path
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
# # from DEFAULTS import PARENT_PATH
import sim.tele_geo_ar as tg
from numerical.root_finder import bisection, brentq, brentq_arg, bisection_arg

import sim.pan_mod_ar as pm
from sim.pan_mod_ar import *
y_cent_m1 = -7201.003729431267

adj_pos_m1, adj_pos_m2 = pm.get_single_vert_adj_positions()
adj_pos_m1, adj_pos_m2 = np.array(adj_pos_m1), np.array(adj_pos_m2)


def root_z2(t, x_0, y_0, z_0, alpha, beta, gamma):
    # Endpoint of ray:
    x = x_0 + alpha * t
    y = y_0 + beta * t
    z = z_0 + gamma * t

    # Convert to M2 r.f.
    xm2, ym2, zm2 = tg.tele_into_m2(x, y, z)
    # Z of mirror in M2 r.f.
    z_m2 = tg.z2(xm2, ym2)
    return zm2 - z_m2

def root_z1(t, x_m2, y_m2, z_m2, alpha, beta, gamma):
    # print(x_m2, y_m2, z_m2, alpha, beta, gamma)
    # Endpoint of ray:
    x = x_m2 + alpha * t
    y = y_m2 + beta * t
    z = z_m2 + gamma * t

    # Convert to M1 r.f.
    xm1, ym1, zm1 = tg.tele_into_m1(x, y, z)

    # Z of mirror in M1 r.f.
    z_m1 = tg.z1(xm1, ym1)
    # print(zm1 - z_m1)
    return zm1 - z_m1

def ray_mirror_pts(P_rx, tg_th2, tg_F_2, theta, phi, estima=[]):
    '''
    Trace rays from the receiver to mirror 1 and mirror 2 of LAT
    This function serves as an estimator to find out on what panels
    the rays will fall.
    Arguemnts:
        P_rx         = Receiver feed position [mm] (in telescope reference frame)
                       np array of shape (3, )
        tg_th_1      = angle of the mirror 1
        tg_th2       = angle of the mirror 2
        theta        = np.array of the theta of rays
        phi          = np.array of the phi of rays
        estima       = [np.array, np.array], a list containing the estimations of
                       the 2 paths (from RX to M2 and from M2 to M1)
    Return:
        out          = np array of shape (6, number of rays)
                       details see below
    '''

    theta_N = len(theta)
    phi_N = len(phi)
    theta = theta.repeat(theta_N).reshape((theta_N, theta_N)).T.flatten()
    phi = phi.repeat(phi_N)

    # Read in telescope geometry values
    th2 = tg_th2
    focal = tg_F_2
    
    n_pts = theta_N*theta_N
    # initialize arrays
    N_hat_t = np.zeros((3, n_pts))
    tan_rx_m2_t = np.zeros((3, n_pts))
    tan_og_t = np.zeros((3, n_pts))
    P_m2 = np.zeros((3, n_pts), dtype=np.float64)
    D_rxm2 = np.zeros((3, n_pts)) # direction of a ray pointing from receiver to mirror2
    P_m1 = np.zeros((3, n_pts))
    ones = np.ones(n_pts)
    out = np.zeros((6, n_pts))

    # Define the outgoing ray's direction
    D_rxm2 = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    # Use a root finder to find where the ray intersects with M2
    if len(estima) !=0 :
        estimator = estima[0]
        lb = estimator - 1e3
        ub = estimator + 1e3
    else:
        lb = ones * (focal + 1e3)
        ub = ones * (focal + 13e3)

    def root_z2_fr(t):
        return root_z2(t, P_rx[0], P_rx[1], P_rx[2], D_rxm2[0], D_rxm2[1], D_rxm2[2])

    t_m2 = bisection(root_z2_fr, lb, ub)

    # Endpoint of ray:
    P_m2[0] = P_rx[0] + D_rxm2[0] * t_m2
    P_m2[1] = P_rx[1] + D_rxm2[1] * t_m2
    P_m2[2] = P_rx[2] + D_rxm2[2] * t_m2

    ########## M2 r.f ###########################################################

    x_m2_temp, y_m2_temp, z_m2_temp = tg.tele_into_m2(P_m2[0],P_m2[1], P_m2[2])
    x_rx_temp, y_rx_temp, z_rx_temp = tg.tele_into_m2(P_rx[0], P_rx[1], P_rx[2])

    # Normal vector of ray on M2
    norm_x, norm_y = tg.d_z2(x_m2_temp, y_m2_temp)
    norm_z = np.ones(n_pts)
    normalizer = 1.0 / np.sqrt(norm_x**2 + norm_y**2 + 1.0)
    N_hat = normalizer*np.array([-norm_x, -norm_y, norm_z])

    # Normalized vector from RX to M2
    vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array([x_rx_temp, y_rx_temp, z_rx_temp])[:,None]
    dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2, axis=0))
    tan_rx_m2 = vec_rx_m2 / dist_rx_m2

    # Vector of outgoing ray
    tan_og = tan_rx_m2 - 2 * np.sum(tan_rx_m2*N_hat, axis=0)*N_hat

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

    # Vector of outgoing ray:
    tan_og_t[0] = tan_og_x_temp
    tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
    tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)

    ########## Tele. r.f ###########################################################
    # Use a root finder to find where the ray intersects with M1
    if len(estima) !=0 :
        estimator = estima[1]
        lb = estimator - 1e3
        ub = estimator + 1e3
    else:
        lb = ones*50.0
        ub = ones*22000.0
    def root_z1_fr(t):
        return root_z1(t, P_m2[0], P_m2[1], P_m2[2], tan_og_t[0], tan_og_t[1], tan_og_t[2])

    t_m1 = bisection(root_z1_fr, lb, ub)

    # Endpoint of ray:
    # P_m1 = P_m2 + tan_og_t * t_m1
    P_m1[0] = P_m2[0] + tan_og_t[0] * t_m1
    P_m1[1] = P_m2[1] + tan_og_t[1] * t_m1
    P_m1[2] = P_m2[2] + tan_og_t[2] * t_m1

    # Write out
    out[0:3] = P_m2 # coordinates of intersection points on mirror 2
    out[3:6] = P_m1 # coordinates of intersection points on mirror 1
    return out

def root_z2_pm(t, x_0, y_0, z_0, alpha, beta, gamma, a, b, c, d, e, x0, y0):
    x = x_0 + alpha * t
    y = y_0 + beta * t
    z = z_0 + gamma * t
     # Convert ray's endpoint into M2 coordinates
    xm2, ym2, zm2 = tg.tele_into_m2(x, y, z) 

    if z_0 != 0:
        z /= np.cos(np.arctan(1 / 3))
    # Convert ray's endpoint into M2 coordinates
    xm2_err, ym2_err, _ = tg.tele_into_m2(x, y, z)  

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

    z_m2 = tg.z2(xm2, ym2)  # Z of mirror in M2 coordinates

    root = zm2 - (z_m2 + z_err)
    return root

def root_z1_pm(t, x_m2, y_m2, z_m2, alpha, beta, gamma, a, b, c, d, e, x0, y0):
    x = x_m2 + alpha * t
    y = y_m2 + beta * t
    z = z_m2 + gamma * t
    # take ray end coordinates and convert to M1 coordinates
    xm1, ym1, zm1 = tg.tele_into_m1(x, y, z)  

    xm1_err, ym1_err, _ = tg.tele_into_m1(x, y, z)

    # x_temp = xm1_err * np.cos(np.pi) + zm1_err * np.sin(np.pi)
    # y_temp = ym1_err
    # z_temp = -xm1_err * np.sin(np.pi) + zm1_err * np.cos(np.pi)
    x_temp = xm1_err * -1.0 
    y_temp = ym1_err
    # z_temp =  zm1_err * -1.0

    xpc = x_temp - x0
    ypc = y_temp - y0

    z_err = (
        a
        + b * xpc
        + c * (ypc)
        + d * (xpc ** 2 + ypc ** 2)
        + e * (xpc * ypc)
    )

    z_m1 = tg.z1(xm1, ym1)  # Z of mirror 1 in M1 coordinates
    root = zm1 - (z_m1 + z_err)
    return root

def aperature_fields_from_panel_model(
    panel_model1, panel_model2, P_rx, tg_th_1,
    tg_th2, tg_z_ap, tg_th_fwhp, theta, phi, rxmirror
    ):
    '''
    Trace rays from the receiver to the aperature of LAT
    with errors in panels of both mirror 1 and mirror 2
    Arguemnts:
        panel_model1 = surface error parameters of panels of miirror 1
        panel_model2 = surface error parameters of panels of miirror 2
        P_rx         = Receiver feed position [mm] (in telescope reference frame)
                       np array of shape (3, )
        tg_th_1      = angle of the mirror 1
        tg_th2       = angle of the mirror 2
        tg_z_ap      = z coordinate of the aperture
        tg_th_fwhp   = FWHP of the receiver horn
        theta        = np.array of the theta of rays
        phi          = np.array of the phi of rays
        rxmirror     = estimated positions of intersection points,  
                       result of 'ray_mirror_pts' function
    Return:
        out          = np array of shape (17, num of valid rays)
                       details see below
    '''
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

    n_pts = theta_N*theta_N
    i_panm2 = np.empty(n_pts, dtype=np.int16)
    i_panm2.fill(32_767)
    i_panm1 = np.empty(n_pts, dtype=np.int16)
    i_panm1.fill(32_767)
    for ii in range(n_pts):
        i_row2 = row_panm_m2[ii]
        i_col2 = col_panm_m2[ii]
        i_row1 = row_panm_m1[ii]
        i_col1 = col_panm_m1[ii]
        temp2 = np.where((panel_model2[0, :] == i_row2) & (panel_model2[1, :] == i_col2))[0]
        temp1 = np.where((panel_model1[0, :] == i_row1) & (panel_model1[1, :] == i_col1))[0]
        if len(temp2) != 0 and len(temp1) != 0:
            i_panm2[ii] =  temp2[0]
            i_panm1[ii] =  temp1[0]
    valid_ray_index = np.where(i_panm2 != 32_767)[0]

    n_pts = len(valid_ray_index)
    i_panm2 = i_panm2[valid_ray_index]
    i_panm1 = i_panm1[valid_ray_index]
    theta = theta[valid_ray_index]
    phi = phi[valid_ray_index]
    rxmirror = np.transpose(np.transpose(rxmirror)[valid_ray_index])

    # initialize arrays
    D_rxm2 = np.zeros((3, n_pts))
    P_m2 = np.zeros((3, n_pts))
    P_m1 = np.zeros((3, n_pts))
    N_hat_t = np.zeros((3, n_pts))
    tan_rx_m2_t = np.zeros((3, n_pts))
    tan_og_t = np.zeros((3, n_pts))
    tan_m2_m1_t = np.zeros((3, n_pts))
    tan_og_t = np.zeros((3, n_pts))
    out = np.zeros((17, n_pts))

    # trace the rays from the receiver to mirror 2
    D_rxm2 = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]

    # surface error parameters of panels
    a = panel_model2[2][i_panm2]
    b = panel_model2[3][i_panm2]
    c = panel_model2[4][i_panm2]
    d = panel_model2[5][i_panm2]
    e = panel_model2[6][i_panm2]
    # f = panel_model2[7][i_panm2]
    x0 = panel_model2[8][i_panm2]
    y0 = panel_model2[9][i_panm2]
    
    estimator = np.sqrt(np.sum(rxmirror[0:3]**2, axis=0))
    lb = estimator-1e3
    ub = estimator+1e3
    # lb = ones * (focal + 1e3)
    # ub = ones * (focal + 12e3)
    def root_z2_pm_fr(t):
        return root_z2_pm(t, P_rx[0], P_rx[1], P_rx[2], 
                                D_rxm2[0], D_rxm2[1], D_rxm2[2], 
                                a, b, c, d, e, x0, y0)
    t_m2 = bisection(root_z2_pm_fr, lb, ub, tol=1e-12, iter_nmax=100)

    # Location of where ray hits M2
    P_m2[0] = P_rx[0] + D_rxm2[0] * t_m2
    P_m2[1] = P_rx[1] + D_rxm2[1] * t_m2
    P_m2[2] = P_rx[2] + D_rxm2[2] * t_m2

    # Using x and y in M2 coordiantes, find the z err:
    

    ###### in M2 coordinates ##########################
    x_m2_temp, y_m2_temp, z_m2_temp = tg.tele_into_m2(P_m2[0], P_m2[1], P_m2[2])  # P_m2 temp
    x_rx_temp, y_rx_temp, z_rx_temp = tg.tele_into_m2(P_rx[0], P_rx[1], P_rx[2])  # P_rx temp

    norm_x, norm_y = tg.d_z2(x_m2_temp, y_m2_temp)
    norm_z = np.ones(n_pts)
    normalizer = 1.0 / np.sqrt(norm_x**2 + norm_y**2 + 1.0)
    N_hat = normalizer*np.array([-norm_x, -norm_y, norm_z])

    vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array([x_rx_temp, y_rx_temp, z_rx_temp])[:,None]
    dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2, axis=0))
    tan_rx_m2 = vec_rx_m2 / dist_rx_m2

    # Vector of outgoing ray
    tan_og = tan_rx_m2 - 2 * np.sum(tan_rx_m2*N_hat, axis=0)*N_hat
    
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
    tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(th2) - tan_rx_m2_z_temp * np.sin(th2)
    tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(th2) + tan_rx_m2_z_temp * np.cos(th2)

    # tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
    # tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
    tan_og_x_temp = tan_og[0] * -1.0
    tan_og_y_temp = tan_og[1] * -1.0
    tan_og_z_temp = tan_og[2]

    tan_og_t[0] = tan_og_x_temp
    tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
    tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)
    ##################################################

    # surface error parameters of panels
    a = panel_model1[2][i_panm1]
    b = panel_model1[3][i_panm1]
    c = panel_model1[4][i_panm1]
    d = panel_model1[5][i_panm1]
    e = panel_model1[6][i_panm1]
    # f = panel_model1[7][i_panm1]
    x0 = panel_model1[8][i_panm1]
    y0 = panel_model1[9][i_panm1]


    estimator = np.sqrt(np.sum((rxmirror[3:6] - rxmirror[0:3])**2, axis=0))
    lb = estimator-1e3
    ub = estimator+1e3
    # lb = ones*50.0
    # ub = ones*22000.0
    def root_z1_pm_fr(t):
        return root_z1_pm(t, 
                          P_m2[0], P_m2[1], P_m2[2], \
                          tan_og_t[0], tan_og_t[1], tan_og_t[2],\
                          a, b, c, d, e, x0, y0)
    t_m1 = bisection(root_z1_pm_fr, lb, ub)

    # Location of where ray hits M1
    P_m1[0] = P_m2[0] + tan_og_t[0] * t_m1
    P_m1[1] = P_m2[1] + tan_og_t[1] * t_m1
    P_m1[2] = P_m2[2] + tan_og_t[2] * t_m1

    ###### in M1 cordinates ##########################
    x_m1_temp, y_m1_temp, z_m1_temp = tg.tele_into_m1(P_m1[0], P_m1[1], P_m1[2])  # P_m2 temp
    x_m2_temp, y_m2_temp, z_m2_temp = tg.tele_into_m1(P_m2[0], P_m2[1], P_m2[2])  # P_rx temp

    norm_x, norm_y = tg.d_z1(x_m1_temp, y_m1_temp)
    norm_z = np.ones(n_pts)
    normalizer = 1.0 / np.sqrt(norm_x**2 + norm_y**2 + 1.0)
    N_hat = normalizer*np.array([-norm_x, -norm_y, norm_z])

    vec_m2_m1 = np.array([x_m1_temp, y_m1_temp, z_m1_temp]) - np.array([x_m2_temp, y_m2_temp, z_m2_temp])
    dist_m2_m1 = np.sqrt(np.sum(vec_m2_m1 ** 2, axis=0))
    tan_m2_m1 = vec_m2_m1 / dist_m2_m1

    # Vector of outgoing ray
    tan_og = tan_m2_m1 - 2 * np.sum(tan_m2_m1*N_hat, axis=0)*N_hat

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

    out[0:3] = P_m2 # coordinates of intersection points on mirror 2
    out[3:6] = P_m1 # coordinates of intersection points on mirror 1
    out[6:9] = N_hat_t # directions of rays pointing towards mirror 1
    out[9:12] = pos_ap # coordinates of intersection points on aperture
    out[12:15] = tan_og_t # directions of rays pointing away from mirror 1

    out[15] = total_path_length # total path length of rays
    out[16] = np.exp(
                    (-0.5)
                    * ((theta - th_mean) ** 2 + (phi - ph_mean) ** 2)
                    / (horn_fwhp / (np.sqrt(8 * np.log(2)))) ** 2
                    ) # intensity modification from receiver feedhorn 
    return out

    
if __name__ == "__main__":
    '''
    test the code--------------------------------------------------------------
    '''
    module_path = "E://Holography/holosim-ml"
    if module_path not in sys.path:
        sys.path.append(module_path)
        
    # from time import time
    import ap_field as ap
    import pan_mod as pm_orig
    tele_geo = tg.initialize_telescope_geometry()
    P_rx = np.array([-50, 209, 50])
    tg_th2 = tele_geo.th2
    tg_F_2 = tele_geo.F_2
    theta_a, theta_b, theta_N = -np.pi / 2 - 0.28, -np.pi / 2 + 0.28, 64
    phi_a, phi_b, phi_N = np.pi / 2 - 0.28, np.pi / 2 + 0.28, 64
    theta = np.linspace(theta_a, theta_b, theta_N)
    phi = np.linspace(phi_a, phi_b, phi_N)


    ################## verify the ray_mirror_pts function--------------------------
    t1 = time()
    rmpts_test = ray_mirror_pts(P_rx, tg_th2, tg_F_2, theta, phi)
    t2 = time()
    print(f'Function ray_mirror_pts executed in {(t2-t1):.8f}s')

    t1 = time()
    P_rx_est = np.array([0, 0, 0])
    rmpts_test_est = ray_mirror_pts(P_rx_est, tg_th2, tg_F_2, theta, phi)
    t2 = time()
    print(f'Function ray_mirror_pts (cached) executed in {(t2-t1):.8f}s')

    t1 = time()
    estima = np.zeros((2, phi_N*theta_N))
    estima[0] = np.sqrt(np.sum(rmpts_test[0:3]**2, axis=0))
    estima[1] = np.sqrt(np.sum((rmpts_test[3:6] - rmpts_test[0:3])**2, axis=0))
    P_rx_est = np.array([0, 0, 0])
    rmpts_test_est = ray_mirror_pts(P_rx_est, tg_th2, tg_F_2, theta, phi, estima=estima)
    t2 = time()
    print(f'Function ray_mirror_pts (cached)(with estimator) executed in {(t2-t1):.8f}s')

    t1 = time()
    rmpts = ap.ray_mirror_pts(P_rx, tele_geo, theta, phi)
    t2 = time()
    print(f'Function ap.ray_mirror_pts executed in {(t2-t1):.8f}s')
    # print(rmpts-rmpts_test)

    if np.allclose(rmpts, rmpts_test, rtol=1e-12, atol=1e-12):
        print("OH Yes! ray_mirror_pts DONE")
    else:
        print("FML check the ray_mirror_pts function again")
    # ################## verify the aperature_fields_from_panel_model function--------------
    save = 0 
    adj_1_A = np.random.randn(1092) * 35
    adj_2_A = np.random.randn(1092) * 35
    panel_model2 = pm.panel_model_from_adjuster_offsets(
        2, adj_2_A, 1, save
    )  # Panel Model on M2
    panel_model1 = pm.panel_model_from_adjuster_offsets(
        1, adj_1_A, 1, save
    )  # Panel Model on M1
    panel_model2_orig = pm_orig.panel_model_from_adjuster_offsets(
        2, adj_2_A, 1, save
    )  # Panel Model on M2
    panel_model1_orig = pm_orig.panel_model_from_adjuster_offsets(
        1, adj_1_A, 1, save
    )  # Panel Model on M1
    rxmirror = rmpts

    tg_th_1, tg_th2, tg_z_ap = tele_geo.th_1, tele_geo.th2, tele_geo.z_ap
    tg_th_fwhp, tg_F_2 = tele_geo.th_fwhp, tele_geo.F_2

    t1 = time()    
    apfield_test = aperature_fields_from_panel_model(panel_model1, panel_model2, \
                                        P_rx, tg_th_1, tg_th2, tg_z_ap, tg_th_fwhp, \
                                        theta, phi, rxmirror
                                        )
    t2 = time()
    print(f'Function aperature_fields_from_panel_model executed in {(t2-t1):.8f}s')

    t1 = time()    
    apfield_raw = ap.aperature_fields_from_panel_model(panel_model1_orig, panel_model2_orig, \
                                        P_rx, tele_geo, theta, phi, rxmirror
                                        )
    fp = np.where(apfield_raw[15,:]!=0)
    apfield = np.zeros((len(fp), 17))
    apfield = apfield_raw.T[fp]
    apfield = apfield.T
    t2 = time()
    print(f'Function ap.aperature_fields_from_panel_model executed in {(t2-t1):.8f}s')
    # print(apfield_test-apfield)

    if np.allclose(apfield, apfield_test, rtol=1e-12, atol=1e-12):
        print("OH Yes! aperature_fields_from_panel_model DONE")
    else:
        print("FML check the aperature_fields_from_panel_model function again")

    