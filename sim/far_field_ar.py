"""
Far field simulation.

Grace E. Chesmore
May 2021

Chun Tung Cheung
March 2022
"""

import numpy as np


def far_field_sim(ap_field, msmt_geo):

    # Break out many quantities from msmt_geo
    N_scan = msmt_geo.N_scan
    de_ang = msmt_geo.de_ang
    lambda_ = msmt_geo.lambda_

    x_tow = msmt_geo.x_tow
    y_tow = msmt_geo.y_tow
    z_tow = msmt_geo.z_tow

    x_phref = msmt_geo.x_phref
    y_phref = msmt_geo.y_phref
    z_phref = msmt_geo.z_phref

    x_rotc = msmt_geo.x_rotc
    y_rotc = msmt_geo.y_rotc
    z_rotc = msmt_geo.z_rotc

    el0 = msmt_geo.el0
    az0 = msmt_geo.az0

    # Break out the geometric coordinates from ap_fields
    # Location of points on the aperture plane
    # in rotation centered coordinates
    x_ap = ap_field[9, :] / 1e3 - x_rotc
    y_ap = ap_field[10, :] / 1e3 - y_rotc
    z_ap = ap_field[11, :] / 1e3 - z_rotc
    N_apscan = len(x_ap)
    # x_ap_tile = np.reshape(np.tile(x_ap, N_scan), (N_scan,len(x_ap)))
    # y_ap_tile = np.reshape(np.tile(y_ap, N_scan), (N_scan,len(x_ap)))
    # z_ap_tile = np.reshape(np.tile(z_ap, N_scan), (N_scan,len(x_ap)))

    # Propagation vector of the sample points (tan_og)
    k_x = ap_field[12, :]
    k_y = ap_field[13, :]
    k_z = ap_field[14, :]

    pathl = ap_field[15, :] / 1e3  # Path length convert to meters
    ampl = np.sqrt(ap_field[16, :])  # Amplitude

    k = 2.0 * np.pi / lambda_  # Wavenumber [1/m]

    # az, el angles for rotation
    el_cur = np.linspace(el0-N_scan*de_ang, el0+N_scan*de_ang, N_scan)
    az_cur = np.linspace(az0-N_scan*de_ang, az0+N_scan*de_ang, N_scan)
    el_cur, az_cur = np.meshgrid(el_cur, az_cur)
    el_cur = np.ravel(el_cur)
    az_cur = np.ravel(az_cur)
    
    # Complex fields
    ima = np.complex(0, 1)
    Fcomplex = ampl * np.exp(ima * pathl * k)

    Npts = len(x_ap)
    out = np.zeros((3, N_scan * N_scan * 4), dtype=complex)


    # Elevation rotation (about x axis)
    x_temp = np.outer(np.ones(len(el_cur)), x_ap)
    y_temp = np.outer(np.cos(el_cur), y_ap) - np.outer(np.sin(el_cur), z_ap)
    z_temp = np.outer(np.sin(el_cur), y_ap) + np.outer(np.cos(el_cur), z_ap)
    cosazmatrx = np.outer(np.cos(az_cur), np.ones(N_apscan)) 
    sinazmatrx = np.outer(np.sin(az_cur), np.ones(N_apscan)) 
    x_apr = cosazmatrx * x_temp + sinazmatrx * z_temp
    y_apr = y_temp
    z_apr = -sinazmatrx * x_temp + cosazmatrx * z_temp

    # Evaluate the distance to the phase reference if prompted to do so

    x_temp = x_phref
    y_temp = np.cos(el_cur) * y_phref - np.sin(el_cur) * z_phref
    z_temp = np.sin(el_cur) * y_phref + np.cos(el_cur) * z_phref

    x_phrefr = np.cos(az_cur) * x_temp + np.sin(az_cur) * z_temp
    y_phrefr = y_temp
    z_phrefr = -np.sin(az_cur) * x_temp + np.cos(az_cur) * z_temp

    r_phref = np.sqrt(
        (x_phrefr - x_tow) ** 2
        + (y_phrefr - y_tow) ** 2
        + (z_phrefr - z_tow) ** 2
    )

    # Evaluate r
    r = np.sqrt((x_apr - x_tow)*(x_apr - x_tow) + (y_apr - y_tow)*(y_apr - y_tow)+ (z_apr - z_tow)*(z_apr - z_tow))
    z_dot_rhat = (z_apr - z_tow) * (-1) / r

    out[0] = np.ravel(az_cur)
    out[1] = np.ravel(el_cur)
    out[2] = np.ravel(
                    np.exp(-ima * r_phref * k)
                    * np.sum(
                        (Fcomplex * np.exp(ima * k * r) / (4 * np.pi * r))
                        * ((ima * k + 1 / r) * z_dot_rhat + ima * k * np.outer(np.ones(len(el_cur)), k_z)), axis=1)
                    / Npts
                )

    return out
