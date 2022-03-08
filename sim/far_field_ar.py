"""
Far field simulation.

Grace E. Chesmore
May 2021

Chun Tung Cheung
March 2022
"""

import numpy as np
import numba as nb

def far_field_sim_unjit(ap_field, msmt_geo, rx):
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
    
    k = 2.0 * np.pi / lambda_  # Wavenumber [1/m]

    # Break out the geometric coordinates from ap_fields
    # Location of points on the aperture plane
    # in rotation centered coordinates
    x_ap = ap_field[9, :] / 1e3 - x_rotc
    y_ap = ap_field[10, :] / 1e3 - y_rotc
    z_ap = ap_field[11, :] / 1e3 - z_rotc
    
    radius = 3.0 # [m]
    validindex = np.where(x_ap*x_ap+y_ap*y_ap<=radius*radius)
    x_ap = x_ap[validindex]
    y_ap = y_ap[validindex]
    z_ap = z_ap[validindex]

    # Propagation vector of the sample points (tan_og)
    # k_x = ap_field[12, :][validindex]
    # k_y = ap_field[13, :][validindex]
    k_z = ap_field[14, :][validindex]

    pathl = ap_field[15, :][validindex] / 1e3  # Path length convert to meters
    ampl = np.sqrt(ap_field[16, :])[validindex]  # Amplitude
    
    N_apscan = len(x_ap)
    # az, el angles for rotation
    el_cur = np.linspace(el0-N_scan*de_ang, el0+(N_scan-1)*de_ang, 2*N_scan, endpoint=True)
    az_cur = np.linspace(az0-N_scan*de_ang, az0+(N_scan-1)*de_ang, 2*N_scan, endpoint=True)
    el_cur, az_cur  = np.meshgrid(el_cur, az_cur)
    el_cur = np.ravel(el_cur)
    az_cur = np.ravel(az_cur)
    
    # Complex fields
    ima = np.complex(0, 1)
    Fcomplex = ampl * np.exp(ima * pathl * k)

    Npts = len(x_ap)
    out = np.zeros((3, N_scan * N_scan * 4), dtype=complex)
    oneselcur = np.ones(len(el_cur))

    # Elevation rotation (about x axis)
    sinelcur = np.sin(el_cur)
    coselcur = np.cos(el_cur)
    x_temp = np.outer(oneselcur, x_ap)
    y_temp = np.outer(coselcur, y_ap) - np.outer(sinelcur, z_ap)
    z_temp = np.outer(sinelcur, y_ap) + np.outer(coselcur, z_ap)

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
        (x_phrefr - x_tow) * (x_phrefr - x_tow)
        + (y_phrefr - y_tow) * (y_phrefr - y_tow) 
        + (z_phrefr - z_tow) * (z_phrefr - z_tow)
    )

    # Evaluate r
    r = np.sqrt((x_apr - x_tow)*(x_apr - x_tow) + (y_apr - y_tow)*(y_apr - y_tow)+ (z_apr - z_tow)*(z_apr - z_tow))
    z_dot_rhat = (z_apr - z_tow) * (-1) / r

    out[0] = az_cur
    out[1] = el_cur
    # out[2] = (
    #             np.exp(-ima * r_phref * k)
    #             * np.mean(
    #                 (Fcomplex * np.exp(ima * k * r) / (4 * np.pi * r))
    #                 * ((ima * k + 1 / r) * z_dot_rhat + ima * k * np.outer(oneselcur, k_z)), axis=1)
    #         )

    
    field_far_integrand = Fcomplex * np.exp(ima * k * r) * z_dot_rhat  / (r * pathl)
    field_far = np.mean(field_far_integrand, axis=1) * k / (2*np.pi*1j)
    out[2] = field_far

    return out

@nb.jit(nopython=True, parallel=False, fastmath=True)
def far_field_sim_jit(ap_field, N_scan, de_ang,lambda_,x_tow,y_tow,z_tow,x_phref,y_phref,z_phref,x_rotc,y_rotc,z_rotc,el0,az0,rx):
    # Break out the geometric coordinates from ap_fields
    # Location of points on the aperture plane
    # in rotation centered coordinates
    x_ap = ap_field[9, :] / 1e3 - x_rotc
    y_ap = ap_field[10, :] / 1e3 - y_rotc
    z_ap = ap_field[11, :] / 1e3 - z_rotc

    
    radius = 30000.0 # [m]
    validindex = np.where(x_ap*x_ap+y_ap*y_ap<=radius*radius)
    x_ap = x_ap[validindex]
    y_ap = y_ap[validindex]
    z_ap = z_ap[validindex]
    N_apscan = len(x_ap)

    # Propagation vector of the sample points (tan_og)
    # k_x = ap_field[12, :][validindex]
    # k_y = ap_field[13, :][validindex]
    k_z = ap_field[14, :][validindex]

    pathl = ap_field[15, :][validindex] / 1e3  # Path length convert to meters
    ampl = np.sqrt(ap_field[16, :])[validindex]  # Amplitude

    k = 2.0 * np.pi / lambda_  # Wavenumber [1/m]

    # az, el angles for rotation
    el_cur = np.zeros(4*N_scan*N_scan, dtype=np.float64)
    az_cur = np.zeros(4*N_scan*N_scan, dtype=np.float64)
    for i_ang in nb.prange(-N_scan, N_scan, 1):
        for j_ang in nb.prange(-N_scan, N_scan, 1):
            i_out = (i_ang + N_scan) * 2 * N_scan + (j_ang + N_scan)
            az_cur[i_out] = az0 + (i_ang) * de_ang
            el_cur[i_out] = el0 + (j_ang) * de_ang

    # Complex fields
    Fcomplex = ampl * np.exp(1j * pathl * k)

    # Npts = len(x_ap)
    out = np.zeros((3, N_scan * N_scan * 4), dtype=np.complex128)
    oneselcur = np.ones(len(el_cur))

    # Elevation rotation (about x axis)
    sinelcur = np.sin(el_cur)
    coselcur = np.cos(el_cur)
    sinazcur = np.sin(az_cur)
    cosazcur = np.cos(az_cur)

    x_temp = np.outer(oneselcur, x_ap)
    y_temp = np.outer(coselcur, y_ap) - np.outer(sinelcur, z_ap)
    z_temp = np.outer(sinelcur, y_ap) + np.outer(coselcur, z_ap)

    cosazmatrx = np.outer(cosazcur, np.ones(N_apscan)) 
    sinazmatrx = np.outer(sinazcur, np.ones(N_apscan)) 
    x_apr = cosazmatrx * x_temp + sinazmatrx * z_temp
    y_apr = y_temp
    z_apr = -sinazmatrx * x_temp + cosazmatrx * z_temp

    # # Evaluate the distance to the phase reference if prompted to do so
    # x_temp = x_phref
    # y_temp = coselcur * y_phref - sinelcur * z_phref
    # z_temp = sinelcur * y_phref + coselcur * z_phref

    # x_phrefr = cosazcur * x_temp + sinazcur * z_temp
    # y_phrefr = y_temp
    # z_phrefr = -sinazcur * x_temp + cosazcur * z_temp

    # r_phref = np.sqrt(
    #     (x_phrefr - x_tow) * (x_phrefr - x_tow)
    #     + (y_phrefr - y_tow) * (y_phrefr - y_tow) 
    #     + (z_phrefr - z_tow) * (z_phrefr - z_tow)
    # )

    # Evaluate r
    r = np.sqrt((x_apr - x_tow)*(x_apr - x_tow) + (y_apr - y_tow)*(y_apr - y_tow)+ (z_apr - z_tow)*(z_apr - z_tow))
    z_dot_rhat = (z_apr - z_tow) * (-1) / r

    out[0] = az_cur
    out[1] = el_cur
    # integrand =  (Fcomplex * np.exp(1j * k * r) / (4 * np.pi * r))* ((1j * k + 1 / r) * z_dot_rhat + 1j * k * np.outer(oneselcur, k_z))
    # for ii in nb.prange(4*N_scan*N_scan):
    #     out[2][ii] = np.mean(integrand[ii]) 
    # out[2] = np.exp(-1j * r_phref * k) * out[2]

    field_far_integrand = Fcomplex * np.exp(1j * k * r) * z_dot_rhat  / (r * pathl)
    for ii in nb.prange(4*N_scan*N_scan):
        out[2][ii] = np.mean(field_far_integrand[ii]) # reuse the array in the memory
    out[2] = out[2] * k / (2*np.pi*1j)

    return out

def far_field_sim(ap_field, msmt_geo, rx):
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
    return far_field_sim_jit(ap_field, N_scan, de_ang,lambda_,x_tow,y_tow,z_tow,x_phref,y_phref,z_phref,x_rotc,y_rotc,z_rotc,el0,az0,rx)

if __name__ == "__main__":
    import time
    import sys
    import os

    module_path = os.path.abspath(os.path.join('.'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from DEFAULTS import PARENT_PATH

    import sim.ap_field_ar as ap
    import sim.pan_mod_ar as pm
    import sim.tele_geo_ar as tg
    import sim.ap_field_ar as ap
    
    # create panel offsets randomly
    save = 0 
    error_rms = 0
    adj_1_A = np.random.randn(1092) * error_rms
    adj_2_A = np.random.randn(1092) * error_rms
    panel_model2 = pm.panel_model_from_adjuster_offsets(
        2, adj_2_A, 1, save
    )  # Panel Model on M2
    panel_model1 = pm.panel_model_from_adjuster_offsets(
        1, adj_1_A, 1, save
    )  # Panel Model on M1

    # ###################### define telescope geometry
    freq = 101.28 # frequency of signal source [GHz]
    wavelength = (30.0 / freq) * 0.01 # [m]
    wavelength = wavelength * 1e3 # [mm]
    k = 2 * np.pi / wavelength
    P_source = [0, -7200.0, 1e6] # unit [mm]
    st_th_fwhp = 30.0/180.0*np.pi 

    tele_geo = tg.TelescopeGeometry()
    tele_geo.N_scan = 100  # pixels in 1D of grid
    # tele_geo.de_ang = 0.2 / 60 * np.pi / 180  # angle increment of telescope scan [rad]
    arcmin_to_rad = 1.0 / 60 * np.pi / 180.0
    tele_geo.de_ang = 60.0/tele_geo.N_scan / (252.0/60.0) * arcmin_to_rad   # angle increment of telescope scan [rad]

    tele_geo.lambda_ = wavelength / 1e3 # [m]
    tele_geo.k = 2 * np.pi / tele_geo.lambda_  # wavenumber [1/m]
    # tele_geo.th_fwhp = 44 * np.pi / 180  # full width half power of source feed [rad]
    tele_geo.z_tow = 1e3 #[m]
    # Azimuth and Elevation center [rad]
    tele_geo.az0 = 0.0
    tele_geo.el0 = np.arctan(-tele_geo.y_tow / tele_geo.z_tow)
    # el0 = 0.005
    # position of the receiver feed [mm]
    P_rx = np.array([0, 209.920654, 0])
    # P_rx = np.array([0, 0, 0])
    [tele_geo.rx_x, tele_geo.rx_y, tele_geo.rx_z] = P_rx/1e3

    # arrays of pointing angles of rays
    angle_size = 0.29
    theta_a, theta_b, theta_N = -np.pi / 2 - angle_size, -np.pi / 2 + angle_size, 100
    phi_a, phi_b, phi_N = np.pi / 2 - angle_size, np.pi / 2 + angle_size, 100
    theta = np.linspace(theta_a, theta_b, theta_N)
    phi = np.linspace(phi_a, phi_b, phi_N)



    # get parameters from telescope geometry
    tg_th_1, tg_th2, tg_z_ap = tele_geo.th_1, tele_geo.th2, tele_geo.z_ap
    tg_th_fwhp, tg_F_2 = tele_geo.th_fwhp, tele_geo.F_2

    # ray tracing
    rxmirror = ap.ray_mirror_pts(P_rx, tg_th2, tg_F_2, theta, phi)
    apout = ap.aperature_fields_from_panel_model(panel_model1, panel_model2, \
                                        P_rx, tg_th_1, tg_th2, tg_z_ap, tg_th_fwhp, \
                                        theta, phi, rxmirror
                                        )
                                    
    def timeit(func):
        def wrapper(*arg):
            time_start = time.time()
            print(f"Starting testing function : {func.__name__}---------")
            output = func(*arg)
            time_end = time.time()
            print(f"Function Execution time : {(time_end-time_start):.5f}s----------------")
            return output
        return wrapper
        
    module_path = "E://Holography/holosim-ml"
    if module_path not in sys.path:
        sys.path.append(module_path)
    import far_field as ff

    # original
    print("Original -------------------")
    farbeam_or = timeit(ff.far_field_sim)(apout, tele_geo, None) 
    print("Under test -----------------------")
    farbeam_ut = timeit(far_field_sim)(apout, tele_geo, None)

    if np.allclose(farbeam_or, farbeam_ut, rtol=1e-12, atol=1e-12):
        print("OH Yes! DONE")
    else:
        print("FML Discrepancy found! Check the function again")
        diff = farbeam_ut-farbeam_or
        print(f"Number of deviated entities: {len(np.ravel(np.where(diff>1e-12)))}")
        print(f"Deviated entities' difference: {diff[np.where(diff>1e-12)]}")
        # print(f"Difference: {diff}")
        
