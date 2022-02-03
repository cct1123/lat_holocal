"""
Telescope geometry definitions.

Grace E. Chesmore
May 2021
"""
import numpy as np
import numba as nb
import plotly.graph_objects as go

class TelescopeGeometry():
    F_2 = 7000.0
    th_1 = np.arctan(1.0 / 2.0)  # Primary mirror tilt angle
    th_2 = np.arctan(1.0 / 3.0)  # Secondary mirror tilt angle
    th2 = (-np.pi / 2) - th_2
    th_fwhp = 44.0 * np.pi / 180.0  # Full width half power [rad]
    N_scan = 100  # Pixels in 1D of grid
    de_ang = 1 / 60.0 * np.pi / 180.0  # Far-field angle increment, arcmin = 1/60 degree

    # Receiver feed position [m]
    rx_x = 0.0
    rx_y = 0.0
    rx_z = 0.0

    # Phase reference [m]
    x_phref = 0.0
    y_phref = -7.2
    z_phref = 0.0
    
    # Center of rotation [m]
    x_rotc = 0.0
    y_rotc = -7.2
    z_rotc = 0.0

    # Aperture plane [m]
    x_ap = 3.0
    y_ap = -7.2
    z_ap = 4.0

    # Source wavelength [m]
    lambda_ = (30.0 / 100.0) * 0.01  
    # Wavenumber [1/m]
    k = 2 * np.pi / lambda_  
    # Source position (tower) [m]
    x_tow = 0.0
    y_tow = -7.2
    z_tow = 1e3

    # Azimuth and Elevation center [rad]
    az0 = 0.0
    el0 = np.arctan(-(y_tow-y_rotc) / z_tow)

    # Matrix Coefficients defining mirror surfaces
    # Primary Mirror
    a1 = np.zeros((7, 7))
    a1[0, :] = [0, 0, -57.74022, 1.5373825, 1.154294, -0.441762, 0.0906601]
    a1[1, :] = [0, 0, 0, 0, 0, 0, 0]
    a1[2, :] = [-72.17349, 1.8691899, 2.8859421, -1.026471, 0.2610568, 0, 0]
    a1[3, :] = [0, 0, 0, 0, 0, 0, 0]
    a1[4, :] = [1.8083973, -0.603195, 0.2177414, 0, 0, 0, 0]
    a1[5, :] = [0, 0, 0, 0, 0, 0, 0]
    a1[6, :] = [0.0394559, 0, 0, 0, 0, 0, 0]
    # Secondary Mirror
    a2 = np.zeros((8, 8))
    a2[0, :] = [0, 0, 103.90461, 6.6513025, 2.8405781, -0.7819705, -0.0400483, 0.0896645]
    a2[1, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    a2[2, :] = [115.44758, 7.3024355, 5.7640389, -1.578144, -0.0354326, 0.2781226, 0, 0]
    a2[3, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    a2[4, :] = [2.9130983, -0.8104051, -0.0185283, 0.2626023, 0, 0, 0, 0]
    a2[5, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    a2[6, :] = [-0.0250794, 0.0709672, 0, 0, 0, 0, 0, 0]
    a2[7, :] = [0, 0, 0, 0, 0, 0, 0, 0]

    R_N = 3000  # [mm]

    def __init__(self):
        pass

    def set_source_position(self, x, y, z):
        self.x_tow = x
        self.y_tow = y
        self.z_tow = z

    def set_receiver_position(self, x, y, z):
        self.rx_x = x
        self.rx_y = y
        self.rx_z = z

    def set_point_direction(self, el, az):
        self.el0 += el
        self.az0 += az

    def reset_geometry(self):
        F_2 = 7000.0
        th_1 = np.arctan(1.0 / 2.0)  # Primary mirror tilt angle
        th_2 = np.arctan(1.0 / 3.0)  # Secondary mirror tilt angle
        th2 = (-np.pi / 2) - th_2
        th_fwhp = 44.0 * np.pi / 180.0  # Full width half power [rad]
        N_scan = 100  # Pixels in 1D of grid
        de_ang = 1 / 60.0 * np.pi / 180.0  # Far-field angle increment, arcmin = 1/60 degree

        # Receiver feed position [m]
        rx_x = 0.0
        rx_y = 0.0
        rx_z = 0.0

        # Phase reference [m]
        x_phref = 0.0
        y_phref = -7.2
        z_phref = 0.0
        
        # Center of rotation [m]
        x_rotc = 0.0
        y_rotc = -7.2
        z_rotc = 0.0

        # Aperture plane [m]
        x_ap = 3.0
        y_ap = -7.2
        z_ap = 4.0

        # Source wavelength [m]
        lambda_ = (30.0 / 100.0) * 0.01  
        # Wavenumber [1/m]
        k = 2 * np.pi / lambda_  
        # Source position (tower) [m]
        x_tow = 0.0
        y_tow = -7.2
        z_tow = 1e3

        # Azimuth and Elevation center [rad]
        az0 = 0.0
        el0 = np.arctan(-(y_tow-y_rotc) / z_tow)

    def init_drawing(self):
        plot_layout = go.Layout(title='SO Large Aperture Telescope', autosize=False,
                  width=1000, height=600,
                  margin=dict(l=65, r=50, b=65, t=90),
                  scene = dict(
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                zaxis=dict(visible=False),
                                aspectmode='data'),
                )
        self.figure = go.Figure(layout=plot_layout)

    def draw_setup(self, show_figure=False):
        if not hasattr(self, "figure"):
            self.init_drawing()
        miiror_size = 6000.0/np.cos(np.arctan(1 / 2))
        msf = miiror_size/2.0

        # positions of mirror 1 surface
        x_m1 = np.linspace(-msf,msf,10, endpoint=True) # [mm]
        y_m1 = np.linspace(-msf,msf,10, endpoint=True) # [mm]
        x_m1, y_m1 = np.meshgrid(x_m1, y_m1)
        height_1 = z1(x_m1, y_m1)
        x_m1, y_m1, z_m1 = m1_into_tele(x_m1, y_m1, height_1)

        # positions of mirror 2 surface
        x_m2 = np.linspace(-msf,msf,10, endpoint=True) # [mm]
        y_m2 = np.linspace(-msf,msf,10, endpoint=True) # [mm]
        x_m2, y_m2 = np.meshgrid(x_m2, y_m2)
        height_2 = z2(x_m2, y_m2)
        x_m2, y_m2, z_m2 = m2_into_tele(x_m2, y_m2, height_2)

        # positions of focal plane 
        x_fp = np.linspace(-2000,2000,100, endpoint=True) # [mm]
        y_fp = np.linspace(-2000,2000,100, endpoint=True) # [mm]
        x_fp, y_fp = np.meshgrid(x_fp, x_fp)
        z_fp = np.zeros_like(x_fp)
        r = np.sqrt(x_fp**2 + y_fp**2)
        z_fp = np.where(r<2000,z_fp,np.nan)
        x_fp,y_fp,z_fp = y_fp, z_fp, x_fp

        plot_m1 = go.Surface(
                                    x = x_m1,
                                    y = y_m1,
                                    z = z_m1,
                                    surfacecolor=height_1, 
                                    colorscale='Teal',
                                    showscale= False, 
                                    name = 'Mirror 1',
                                    text="Mirror 1", 
                                    showlegend = True, 
                                    hoverinfo="name"
                                )

        plot_m2 = go.Surface(
                                    x = x_m2,
                                    y = y_m2,
                                    z = z_m2,
                                    surfacecolor=-height_2,
                                    colorscale='Teal',
                                    showscale= False, 
                                    name = 'Mirror 2',
                                    showlegend = True, 
                                    hoverinfo="name"
                                )

        plot_fp = go.Surface(
                                    x = x_fp,
                                    y = y_fp,
                                    z = z_fp,
                                    surfacecolor=z_fp*0 + 0.6,
                                    colorscale='Cividis',
                                    showscale= False, 
                                    opacity=0.4,
                                    name = 'Focal Plane',
                                    showlegend = True, 
                                    hoverinfo="name"
                                )

        plot_setup = [plot_m1, plot_m2, plot_fp]
        self.figure.add_traces(plot_setup)

        if show_figure:
            self.show_figure()

    def draw_rays(self, rays_data, num_rayshow=500, show_figure=False):
        if not hasattr(self, "figure"):
            self.init_drawing()
        ######### draw the rays obtained from the simulation #############
        apfield_t = rays_data
        amp_min = np.min(apfield_t[16,:])
        amp_max = np.max(apfield_t[16,:])
        x_foc, y_foc, z_foc = self.rx_x, self.rx_y, self.rx_z

        plot_rays = []
        x_a_list, y_a_list, z_a_list = [], [], []
        amp_list = []
        for ii in np.unique(np.random.randint(low=0, high=len(apfield_t[15]), size=num_rayshow)):
            if apfield_t[15][ii] !=0 :
                x_m1, y_m1, z_m1 = apfield_t[3][ii], apfield_t[4][ii], apfield_t[5][ii]
                x_m2, y_m2, z_m2 = apfield_t[0][ii], apfield_t[1][ii], apfield_t[2][ii]
                x_a, y_a, z_a = apfield_t[9][ii], apfield_t[10][ii], apfield_t[11][ii]


                xline = [ x_a, x_m1, x_m2, x_foc]
                yline = [ y_a, y_m1, y_m2, y_foc]
                zline = [ z_a, z_m1, z_m2, z_foc]
            
                amp = apfield_t[16,:][ii]

                # my_color = ('rgba('+str(np.random.randint(50, high = 200))+','+
                #             str(np.random.randint(50, high = 200))+','+
                #             str(np.random.randint(50, high = 200)))

                ray_line =go.Scatter3d(
                                    x=xline, 
                                    y=yline, 
                                    z=zline,
                                    mode='lines',
                                    line=dict(
                                            color=[amp,amp,amp,amp],
                                            cmax=amp_max, 
                                            cmin=amp_min,
                                            width=5, 
                                            colorscale='Purpor', 
                                            ), 
                                    opacity=0.4,
                                    showlegend = False,
                                    hoverinfo='none', 
                                    )

                plot_rays.append(ray_line)
                x_a_list.append(x_a)
                y_a_list.append(y_a)
                z_a_list.append(z_a)
                amp_list.append(amp)

        plot_ap = [go.Scatter3d(
                                x = x_a_list, y = y_a_list, z = z_a_list, 
                                mode = 'markers', 
                                marker=dict(symbol="circle", size=5, colorscale='Purpor', color=amp_list), 
                                opacity=0.4,
                                showlegend = True,
                                hoverinfo='name', 
                                name="rays"
                            )
                ]
        self.figure.add_traces(plot_rays + plot_ap)
        if show_figure:
            self.show_figure()
            
    def show_figure(self):
        self.figure.show()
# compatible class for older version of codes
class initialize_telescope_geometry(TelescopeGeometry):
    def __init__(self):
        super().__init__()

# calculate parameters
tele_geo = TelescopeGeometry()
a1 = tele_geo.a1
a2 = tele_geo.a2
R_N = tele_geo.R_N
th1 = tele_geo.th_1
th2 = tele_geo.th2
costh1 = np.cos(th1)
sinth1 = np.sin(th1)
costh2 = np.cos(th2)
sinth2 = np.sin(th2)

# These functions define the mirror surfaces,
# and the normal vectors on the surfaces.
@nb.jit(nopython=True, parallel=False, fastmath=True)
def z1(x, y):
    xrn = x / R_N
    yrn = y / R_N
    amp = np.zeros_like(x)
    xrn0 = np.ones_like(x)
    yrn0 = np.copy(xrn0)
    for ii in nb.prange(0, 7):
        yrn0 = yrn0 * 0.0 + 1.0
        for jj in nb.prange(0, 7):
            amp += a1[ii, jj] * (xrn0) * (yrn0)
            yrn0 = yrn0 * yrn
        xrn0 = xrn0 * xrn
    return amp

@nb.jit(nopython=True, parallel=False, fastmath=True)
def z2(x, y):
    xrn = x / R_N
    yrn = y / R_N
    amp = np.zeros_like(x)
    xrn0 = np.ones_like(x)
    yrn0 = np.copy(xrn0)
    for ii in nb.prange(0, 8):
        yrn0 = yrn0 * 0.0 + 1.0
        for jj in nb.prange(0, 8):
            amp += a2[ii, jj] * (xrn0) * (yrn0)
            yrn0 = yrn0 * yrn
        xrn0 = xrn0 * xrn
    return amp

# def d_z1(x, y):
#     amp_x = 0
#     amp_y = 0
#     for ii in range(7):
#         for jj in range(7):
#             amp_x += (
#                 a1[ii, jj] * (ii / R_N) * ((x / R_N) ** (ii - 1)) * ((y / R_N) ** jj)
#             )
#             amp_y += (
#                 a1[ii, jj] * ((x / R_N) ** ii) * (jj / R_N) * ((y / R_N) ** (jj - 1))
#             )
#     return amp_x, amp_y

def d_z1(x, y):
    amp_x = np.zeros_like(x)
    amp_y = np.copy(amp_x)
    xrn = x / R_N
    yrn = y / R_N
    xrn0 = np.ones_like(x)
    yrn0 = np.copy(xrn0)
    for ii in range(7):
        yrn0 = yrn0 * 0.0 + 1.0
        for jj in range(7):
            amp_x = amp_x + a1[ii, jj] * (ii / R_N) * (xrn0/xrn) * (yrn0)
            amp_y = amp_y + a1[ii, jj] * (xrn0) * (jj / R_N) * (yrn0/yrn) 
            yrn0 = yrn0 * yrn
        xrn0 = xrn0 * xrn
    return amp_x, amp_y

# def d_z2(x, y):
#     amp_x = 0
#     amp_y = 0
#     for ii in range(8):
#         for jj in range(8):
#             amp_x += (
#                 a2[ii, jj] * (ii / R_N) * ((x / R_N) ** (ii - 1)) * ((y / R_N) ** jj)
#             )
#             amp_y += (
#                 a2[ii, jj] * ((x / R_N) ** ii) * (jj / R_N) * ((y / R_N) ** (jj - 1))
#             )
#     return amp_x, amp_y

def d_z2(x, y):
    amp_x = np.zeros_like(x)
    amp_y = np.copy(amp_x)
    xrn = x / R_N
    yrn = y / R_N
    xrn0 = np.ones_like(x)
    yrn0 = np.copy(xrn0)
    for ii in range(8):
        yrn0 = yrn0 * 0.0 + 1.0
        for jj in range(8):
            amp_x = amp_x + a2[ii, jj] * (ii / R_N) * (xrn0/xrn) * (yrn0)
            amp_y = amp_y + a2[ii, jj] * (xrn0) * (jj / R_N) * (yrn0/yrn) 
            yrn0 = yrn0 * yrn
        xrn0 = xrn0 * xrn
    return amp_x, amp_y

# Coordinate transfer functions. Transferring
# coordinates between telescope reference frame
# and mirror reference frame, and vice versa.
def m1_into_tele(x, y, z):
    xx = x * np.cos(np.pi) + z * np.sin(np.pi)
    yy = y
    zz = -x * np.sin(np.pi) + z * np.cos(np.pi)

    x_rot1 = xx
    y_rot1 = yy * np.cos(th1) - zz * np.sin(th1) - 7200
    z_rot1 = (yy * np.sin(th1) + zz * np.cos(th1)) - 3600
    return x_rot1, y_rot1, z_rot1


def m2_into_tele(x, y, z):
    x_temp = x * np.cos(np.pi) - y * np.sin(np.pi)
    y_temp = x * np.sin(np.pi) + y * np.cos(np.pi)
    z_temp = z

    x_rot2 = x_temp
    y_rot2 = (y_temp * np.cos(th2) - z_temp * np.sin(th2)) - 4800 - 7200
    z_rot2 = y_temp * np.sin(th2) + z_temp * np.cos(th2)
    return x_rot2, y_rot2, z_rot2

# @nb.jit(nopython=True, parallel=False, fastmath=True)
def tele_into_m1(x, y, z):
    zshift = z + 3600.0
    yshift = y + 7200.0
    # x_temp = x
    # y_temp = yshift * np.cos(-th1) - zshift * np.sin(-th1)
    # z_temp = yshift * np.sin(-th1) + zshift * np.cos(-th1)
    # x2 = x_temp * np.cos(np.pi) + z_temp * np.sin(np.pi)
    # y2 = y_temp
    # z2 = -x_temp * np.sin(np.pi) + z_temp * np.cos(np.pi)
    # return x2, y2, z2

    y_temp = yshift * costh1 + zshift * sinth1
    z_temp = -yshift * sinth1 + zshift * costh1
    return -x, y_temp, -z_temp

# @nb.jit(nopython=True, parallel=False, fastmath=True)
def tele_into_m2(x, y, z):
    yshift = y + 4800.0 + 7200.0
    # x_temp = x
    # y_temp = yshift * np.cos(-th2) - z * np.sin(-th2)
    # z_temp = yshift * np.sin(-th2) + z * np.cos(-th2)
    # x2 = x_temp * np.cos(-np.pi) - y_temp * np.sin(-np.pi)
    # y2 = x_temp * np.sin(-np.pi) + y_temp * np.cos(-np.pi)
    # z2 = z_temp
    # return x2, y2, z2

    y_temp = yshift * costh2 + z * sinth2
    z_temp = -yshift * sinth2 + z * costh2
    return -x, -y_temp, z_temp

def tele_geo_init(x, y, z, el, az):
    tele_geo = initialize_telescope_geometry()
    tele_geo.rx_x = x
    tele_geo.rx_y = y
    tele_geo.rx_z = z
    tele_geo.el0 += el
    tele_geo.az0 += az
    return tele_geo

if __name__ == "__main__":
    '''
    for code test
    '''
    tele_geo = TelescopeGeometry()
    tele_geo.draw_setup(show_figure=True)
    