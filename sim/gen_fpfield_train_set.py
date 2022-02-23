"""
Parallelized holography simulation of the LATRt using geometric
theory of diffraction. Output is a .txt file with X,Y,amplitude and phase of beam.
Grace E. Chesmore, July 2021

Output are :
            .pys files with adjuster errors of mirror 1 and 2, and field on focal plane, 
            .png showing the diffraction pattern on focal plane
            a backup script
            a .txt describing the keys of the .pys files
Chun Tung Cheung, February 2021

To Run (from terminal): 
mpiexec -n 5 "D:/ProgramData/Miniconda3/envs/holoenv/python.exe" "./sim/gen_fpfield_train_set.py"
"""

import sys
import time
import numpy as np
from mpi4py import MPI

from pathlib import Path
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)

from DEFAULTS import PARENT_PATH
import sim.tele_geo_ar as tg
import sim.pan_mod_ar as pm
import sim.fp_field_ar as fp

################## parameters for MPI4PY ##################
# Some initial parameters 
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
status = MPI.Status()
t_i = time.perf_counter()

def enum(*seq):
    enums = dict(zip(seq, range(len(seq))))
    return type("Enum", (), enums)

Tags = enum("READY", "CALC", "DONE", "ERROR", "EXIT")

################## parallelization ##################

# rank zero is the main thread. It manages the others and passes messages them
if rank == 0:

    import os
    import pickle
    import shutil
    from tqdm import tqdm
    from datetime import datetime
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    ################## parameters for simulation ##################
    # propeties of the source transmitter
    freq = 101.28 # frequency of signal source [GHz]
    wavelength = (30.0 / freq) * 0.01 # [m]
    wavelength = wavelength * 1e3 # [mm]
    k = 2 * np.pi / wavelength
    P_source = [0, -7200.0, 1e6] # unit [mm]
    st_th_fwhp = 30.0/180.0*np.pi 

    # scanning plane
    Npts_x = 64
    Npts_z = 64
    scanrange_x = 90 # [mm]
    scanrange_z = 90 # [mm]
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

    # get parameters from telescope geometry
    tg_th_1, tg_th2, tg_z_ap = tele_geo.th_1, tele_geo.th2, tele_geo.z_ap
    tg_th_fwhp, tg_F_2 = tele_geo.th_fwhp, tele_geo.F_2
    tg_rotc = np.array([tele_geo.x_rotc, tele_geo.y_rotc, tele_geo.z_rotc]) * 1e3 # [mm]

    # parameters for generating training set
    train_set_size = 10

    ################## parameters for saving and backup ##################
    filename_base = "simdata"
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    # current_time = datetime.now().strftime("%Y_%m_%d")
    path = f"{PARENT_PATH}/data/tung/{current_time}"  # path where you save far-field simulations.
    if not os.path.exists(path):
        os.makedirs(path)
    # backup this script
    shutil.copy2(Path(__file__), f"{path}/script_backup.py")

    ################## initialize plots ##################
    layout = go.Layout(title='Beam on focal plane', autosize=True,
                    width=1000, height=500, 
                    margin=dict(l=50, r=50, b=65, t=90),
                    xaxis1 = dict(title="x [mm]"),
                    yaxis1 = dict(scaleanchor = 'x', title="y [mm]"),
                    xaxis2 = dict(scaleanchor = "x", title="x [mm]"),
                    yaxis2 = dict(scaleanchor = "y"),
                    )
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False, 
                        shared_xaxes=True, subplot_titles=["Phase [rad]", "Intensity [dB]"])
    fig.update_layout(layout)

    ################## communicate with threads ##################
    # result_compile_fpf = np.zeros((train_set_size, Npts_x, Npts_z))
    # result_compile_adj1 = np.zeros((train_set_size, 77 * 5))
    # result_compile_adj2 = np.zeros((train_set_size, 69 * 5))

    # nworkers is the number of sub threads
    pbar = tqdm(total=train_set_size)
    nworkers = size - 1
    for ii in range(train_set_size):
        ## receive messages from the workers
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        ## send back new calculation
        send = {
                "idx" : ii,
                "P_rx_x" : P_rx_x,
                "P_rx_y" : P_rx_y,
                "P_rx_z" : P_rx_z,
                "P_source" : P_source,
                "tg_rotc" : tg_rotc,
                "tg_th_1" : tg_th_1,
                "tg_th2" : tg_th2,
                "tg_F_2" : tg_F_2,
                "tg_z_ap" : tg_z_ap,
                "tg_th_fwhp" : tg_th_fwhp,
                "st_th_fwhp" : st_th_fwhp,
                "theta" : theta,
                "phi" : phi,
                "k" : k
                }
                
        comm.send(send, dest=source, tag=Tags.CALC)
        if tag == Tags.DONE:
            idx = msg["idx"]
            # result_compile_adj1[idx] = msg["result"]["adj_err_1"]
            # result_compile_adj2[idx] = msg["result"]["adj_err_2"]
            # result_compile_fpf[idx] = msg["result"]["field_fp"]
            # savedata = dict(
            #                 adj_err_1=result_compile_adj1[idx],
            #                 adj_err_2=result_compile_adj2[idx],
            #                 field_fp=result_compile_fpf[idx], 
            #                 )

            # save data
            savedata = dict(
                            adj_err_1=msg["result"]["adj_err_1"],
                            adj_err_2=msg["result"]["adj_err_2"],
                            field_fp=msg["result"]["field_fp"], 
                            )

            with open(f"{path}/{filename_base}_{str(idx)}.pys", "wb") as file:
                pickle.dump(savedata, file)

            # save plots
            field_fp = msg["result"]["field_fp"]
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
            fig.data = []
            fig.add_trace(plot_1, row=1, col=1)
            fig.add_trace(plot_2, row=1, col=2)
            fig.write_image(f"{path}/{filename_base}_{str(idx)}.png")

            # update the progress bar
            pbar.update(1)
        elif tag == Tags.ERROR:
            print(msg["error"])

    for _ in range(nworkers):
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        comm.send(None, dest=source, tag=Tags.EXIT)
        if tag == Tags.DONE:
            idx = msg["idx"]
            # result_compile_adj1[idx] = msg["result"]["adj_err_1"]
            # result_compile_adj2[idx] = msg["result"]["adj_err_2"]
            # result_compile_fpf[idx] = msg["result"]["field_fp"]
            # savedata = dict(
            #                 adj_err_1=result_compile_adj1[idx],
            #                 adj_err_2=result_compile_adj2[idx],
            #                 field_fp=result_compile_fpf[idx], 
            #                 )

            # save data
            savedata = dict(
                            adj_err_1=msg["result"]["adj_err_1"],
                            adj_err_2=msg["result"]["adj_err_2"],
                            field_fp=msg["result"]["field_fp"], 
                            )

            with open(f"{path}/{filename_base}_{str(idx)}.pys", "wb") as file:
                pickle.dump(savedata, file)

            # save plots
            field_fp = msg["result"]["field_fp"]
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
            fig.data = []
            fig.add_trace(plot_1, row=1, col=1)
            fig.add_trace(plot_2, row=1, col=2)
            fig.write_image(f"{path}/{filename_base}_{str(idx)}.png")

            # update the progress bar
            pbar.update(1)
        elif tag == Tags.ERROR:
            print(msg["error"])

    with open(f"{path}/{filename_base}_keys.txt", "w") as file:
                file.write(str(list(savedata.keys())))

else:

    comm.send(None, dest=0, tag=Tags.READY)

    while True:
        msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == Tags.CALC:
            ################# get the parameters from main thread #########################
            idx = msg["idx"]
            P_rx_x = msg["P_rx_x"]
            P_rx_y = msg["P_rx_y"]
            P_rx_z = msg["P_rx_z"]
            P_source = msg["P_source"]
            tg_rotc = msg["tg_rotc"]
            tg_th_1 = msg["tg_th_1"]
            tg_th2 = msg["tg_th2"]
            tg_F_2 = msg["tg_F_2"]
            tg_z_ap = msg["tg_z_ap"]
            tg_th_fwhp = msg["tg_th_fwhp"]
            st_th_fwhp = msg["st_th_fwhp"]
            theta = msg["theta"]
            phi = msg["phi"]
            k = msg["k"]


            ################## create panel offsets randomly #########################
            adj_err_1_A = np.random.randn(77 * 5) * 40 # unit [um]
            adj_err_2_A = np.random.randn(69 * 5) * 40 # unit [um]
            panel_model2 = pm.panel_model_from_adjuster_offsets(
                2, adj_err_2_A, 1, 0
            )  # Panel Model on M2
            panel_model1 = pm.panel_model_from_adjuster_offsets(
                1, adj_err_1_A, 1, 0
            )  # Panel Model on M1

            ################### perform simulation to get the field on focal plane ##################
            field_fp = fp.focal_fields(
                                        panel_model1, panel_model2, \
                                        P_rx_x, P_rx_y, P_rx_z, P_source, \
                                        tg_rotc, tg_th_1, tg_th2, tg_F_2, tg_z_ap, tg_th_fwhp, \
                                        st_th_fwhp, theta, phi, k
                                        )

            # msg["result"] = [field_fp]
            msg["result"] = dict(
                                 adj_err_1=adj_err_1_A, 
                                 adj_err_2=adj_err_2_A, 
                                 field_fp=field_fp
                                 )
            # msg["result"] = dict(epicresult=f"fun{idx}")
            # send message that you're done with this calculation.
            comm.send(msg, dest=0, tag=Tags.DONE)

        elif tag == Tags.EXIT:
            print(f"Thread {rank} is bailing")
            break