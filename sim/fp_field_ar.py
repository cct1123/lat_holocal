''''
Simulate the diffraction beam pattern on the focal plane

Chun Tung Cheung
February 2022

'''
import numpy as np
import sys
from pathlib import Path
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)

from DEFAULTS import PARENT_PATH
import sim.ap_field_ar as ap
import sim.tele_geo_ar as tg
import sim.pan_mod_ar as pm

def source_horn_radiation_pattern(th, st_th_fwhp):
    # theta_fwhp = 30/180.0*np.pi # FWHP of 30 degree
    theta_fwhp = st_th_fwhp
    theta_m = 0
    intensity = np.exp(-0.5*((th-theta_m)/(theta_fwhp/np.sqrt(8*np.log(2))))**2)
    return intensity

def focal_fields(panel_model1, panel_model2, P_rx_x, P_rx_y, P_rx_z, P_source, tg_rotc, tg_th_1, tg_th2, tg_F_2, tg_z_ap, tg_th_fwhp, st_th_fwhp, theta, phi, k):
    '''
    To simulate the diffraction pattern on the focal plane 

    Arguemnts:
        panel_model1 = surface error parameters of panels of miirror 1
        panel_model2 = surface error parameters of panels of miirror 2
        P_rx_x       = array of x coordinate of receiver feed position [mm] (in telescope reference frame)
        P_rx_y       = y coordinate of receiver feed position [mm] (in telescope reference frame) (float)
        P_rx_z       = array of z coordinate of receiver feed position [mm] (in telescope reference frame)
        P_source     = position of source transmitter, np.array([x, y, z])  unit: [mm]
        tg_rotc      = center of rotation, np.array([x, y, z])  [mm]
        tg_th_1      = angle of the mirror 1
        tg_th2       = angle of the mirror 2
        tg_F_2       = focal length ?? [mm]
        tg_z_ap      = z coordinate of the aperture
        tg_th_fwhp   = FWHP of the receiver horn
        st_th_fwhp   = FWHP of the source transmitter horn
        theta        = np.array of the theta of rays
        phi          = np.array of the phi of rays
        k            = wavenumber 2pi/lambda [1/mm]
    Return:
        out          = the field on focal plane, 
                       np array of shape ( len(P_rx_x), len(P_rx_z)) np.complex128
    '''
    num_rx_x= len(P_rx_x)
    num_rx_z = len(P_rx_z)
    out = np.zeros((num_rx_x, num_rx_z), dtype=np.complex128)
    aperture_radius = 3000 # [mm]
    for ii in range(num_rx_x):
        for jj in range(num_rx_z):
            P_rx = np.array([P_rx_x[ii], P_rx_y, P_rx_z[jj]])
            rxmirror = ap.ray_mirror_pts(P_rx, tg_th2, tg_F_2, theta, phi)
            apout = ap.aperature_fields_from_panel_model(panel_model1, panel_model2, \
                                                            P_rx, tg_th_1, tg_th2, tg_z_ap, tg_th_fwhp, \
                                                            theta, phi, rxmirror
                                                            )

            # get coordinates of points on aperature
            x_aprotc = apout[9] - tg_rotc[0]  # coordinates of pts on aperature relative to the rotation center [mm]
            y_aprotc = apout[10] - tg_rotc[1]  # coordinates of pts on aperature relative to the rotation center [mm]
            z_aprotc = apout[11] - tg_rotc[2]  # coordinates of pts on aperature relative to the rotation center [mm]
            inap_bool = ((np.square(x_aprotc) + np.square(y_aprotc)) < np.square(aperture_radius))
            valid_indices = np.where(inap_bool) #Ignore rays that missed the aperture plane
            x_ap = apout[9][valid_indices]
            y_ap = apout[10][valid_indices]
            z_ap = apout[11][valid_indices]
            x_aprotc = x_aprotc[valid_indices]
            y_aprotc = y_aprotc[valid_indices]
            z_aprotc = z_aprotc[valid_indices]

            dir_outgo_x = apout[12][valid_indices] # directions of rays pointing away from mirror 1
            dir_outgo_y = apout[13][valid_indices] # directions of rays pointing away from mirror 1
            dir_outgo_z = apout[14][valid_indices] # directions of rays pointing away from mirror 1
            
            intensity = apout[16][valid_indices]
            pathl = apout[15][valid_indices]
            ampl = np.sqrt(intensity)
            field_ap = ampl * np.exp(1j * k * pathl)

            [x_tow, y_tow, z_tow] = P_source  # unit: [mm]
            x_aptow = x_ap - x_tow
            y_aptow = y_ap - y_tow
            z_aptow = z_ap - z_tow
            xy_aptow_sq = x_aptow*x_aptow + y_aptow*y_aptow
            r_aptow = np.sqrt(xy_aptow_sq + z_aptow*z_aptow)
            elr = np.arctan(np.sqrt(xy_aptow_sq)/z_aptow)
            field_sour = np.sqrt(source_horn_radiation_pattern(elr, st_th_fwhp)) * np.exp(1j * k * r_aptow)

            z_dot_r_aptow = - z_aptow / r_aptow
            z_dot_r_apm1 = dir_outgo_z / np.sqrt(dir_outgo_x*dir_outgo_x + dir_outgo_y*dir_outgo_y + dir_outgo_z*dir_outgo_z)
            
            # # Fresnel Kirchhoff diffraction formula 
            # field_fp_integrand = field_sour * field_ap / (r_aptow * pathl) * (z_dot_r_apm1 - z_dot_r_aptow)
            # field_fp = np.sum(field_fp_integrand) / 2.0 * k / (2*np.pi*1j)

            # First Rayleigh-Sommerfeld diffraction formula 
            field_fp_integrand = field_sour * field_ap / (r_aptow * pathl) * z_dot_r_apm1
            field_fp = np.sum(field_fp_integrand) * k / (2*np.pi*1j) / len(field_fp_integrand)

            out[ii][jj] = field_fp
    return out

if __name__ == "__main__":
    import time
    import os
    import pickle
    import shutil
    
    # propeties of the source transmitter
    freq = 101.28 # frequency of signal source [GHz]
    wavelength = (30.0 / freq) * 0.01 # [m]
    wavelength = wavelength * 1e3 # [mm]
    k = 2 * np.pi / wavelength
    P_source = [0, -7200.0, 1e6] # unit [mm]
    st_th_fwhp = 30.0/180.0*np.pi 

    # scanning plane
    Npts_x = 4
    Npts_z = 4
    scanrange_x = 60 # [mm]
    scanrange_z = 60 # [mm]
    P_rx_x = np.linspace(-scanrange_x/2.0, scanrange_x/2.0, Npts_x)
    P_rx_y = 209.920654 # mm, position of focus point with rf source located 1km away
    P_rx_z = np.linspace(-scanrange_z/2.0, scanrange_z/2.0, Npts_z)

    # default telescope geometry
    tele_geo = tg.TelescopeGeometry()

    # arrays of pointing angles of rays
    theta_a, theta_b, theta_N = -np.pi / 2 - 0.29, -np.pi / 2 + 0.29, 64
    phi_a, phi_b, phi_N = np.pi / 2 - 0.29, np.pi / 2 + 0.29, 64
    theta = np.linspace(theta_a, theta_b, theta_N)
    phi = np.linspace(phi_a, phi_b, phi_N)

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

    # get parameters from telescope geometry
    tg_th_1, tg_th2, tg_z_ap = tele_geo.th_1, tele_geo.th2, tele_geo.z_ap
    tg_th_fwhp, tg_F_2 = tele_geo.th_fwhp, tele_geo.F_2
    tg_rotc = np.array([tele_geo.x_rotc, tele_geo.y_rotc, tele_geo.z_rotc]) * 1e3 # [mm]
    
    print("Simulation Starts------------------")
    t1 = time.time()
    field_fp = focal_fields(panel_model1, panel_model2, P_rx_x, P_rx_y, P_rx_z, P_source, tg_rotc, tg_th_1, tg_th2, tg_F_2, tg_z_ap, tg_th_fwhp, st_th_fwhp, theta, phi, k)
    t2 = time.time()
    print(f'Function focal_fields executed in {(t2-t1):.8f}s')
    print("Simulation Ended------------------")

    filename = "focal_field_test"
    path = f"{PARENT_PATH}/data/tung"  # path where you save far-field simulations.
    if not os.path.exists(path):
        os.makedirs(path)
    shutil.copy2(Path(__file__), f"{path}/script_backup.py")
    with open(f"{path}/{filename}.pys", "wb") as file:
        pickle.dump(field_fp, file)
    # with open(f"{path}/{filename}.pys", "rb") as file:
    #     field_fp = pickle.load(file)

    ############# plot the field #######################################
    # import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    field_fp = field_fp / np.max(np.abs(field_fp)) # normalize the field
    plot_1 = go.Heatmap(
                        x=P_rx_x, 
                        y=P_rx_z,
                        z = np.angle(field_fp),
                        colorscale='Twilight',
                        showscale=True, colorbar=dict(len=0.8, x=0.44),
                        showlegend = False,
                        # hoverinfo='name', 
                        name="phase"
                        )
    plot_2 = go.Heatmap( 
                        x=P_rx_x, 
                        y=P_rx_z,
                        z = 20*np.log10(np.abs(field_fp)),
                        # z= np.abs(field_fp*field_fp),
                        colorscale='Magma',
                        showscale=True, 
                        colorbar=dict(len=0.8, x=1), 
                        showlegend = False,
                        # hoverinfo='name', 
                        name="power"
                        )
    layout = go.Layout(title='Beam on focal plane', autosize=True,
                    width=1000, height=500, 
                    margin=dict(l=50, r=50, b=65, t=90),
                    xaxis1 = dict(title="x [mm]"),
                    yaxis1 = dict(scaleanchor = 'x', title="y [mm]"),
                    xaxis2 = dict(scaleanchor = "x", title="x [mm]"),
                    yaxis2 = dict(scaleanchor = "y"),
                    )

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False, shared_xaxes=True, subplot_titles=["Phase [rad]", "Power [dB]"])
    fig.add_trace(plot_1, row=1, col=1)
    fig.add_trace(plot_2, row=1, col=2)
    fig.update_layout(layout)
    fig.show()
