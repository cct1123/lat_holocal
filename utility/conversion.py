import numpy as np

def sky_to_focal(x_sky, y_sky):
    """
    convert sky coordinates (x_sky, y_sky) to focal plane coordinates (x_fp, y_fp)
    ref : https://github.com/patogallardo/zemax_tools/tree/master/design_analysis/CD
    arguments:
        x_sky : x coordinate on sky unit:[rad]
        y_sky : y coordinate on sky unit:[rad]
    return:
        x_fp, y_fp : coordinates on focal plane unit:[mm]
    """
    # x_fp = 4.206667 * x_sky  #252.4/60.0
    # y_fp = 4.205000 * y_sky - 7.6 #252.3/60.0
    x_fp = 252.4*180.0/np.pi * x_sky  
    # y_fp = 252.3*180.0/np.pi * y_sky - 7.6 
    y_fp = 252.3*180.0/np.pi * y_sky 
    return x_fp, y_fp

def sky_to_aperture(x_sky, y_sky, wavelength):
    """
    convert sky coordinates (x_sky, y_sky) to aperture plane coordinates (x_ap, y_ap)
    ref : https://github.com/patogallardo/zemax_tools/tree/master/design_analysis/CD
    arguments:
        x_sky : 1d np.array, x coordinate on sky unit:[rad]
        y_sky : 1d np.array, y coordinate on sky unit:[rad]
        wavelength : wavelength, unit:[mm]
    return:
        x_fp, y_fp : 1d np.arrays, coordinates on focal plane unit:[mm]
    """
    # x_skyrange = np.abs(np.max(x_sky) - np.min(x_sky)) * 1.0/60.0/np.pi*180.0
    x_skyrange = np.abs(np.max(x_sky) - np.min(x_sky))
    x_skynum = len(x_sky)
    x_skydelta = x_skyrange / (x_skynum - 1)  # increment in azimuthal angle
    x_apdelta = wavelength / (x_skydelta*x_skynum)  # increment in x spatial coordinates
    x_aprange = wavelength / x_skydelta

    # y_skyrange = np.abs(np.max(y_sky) - np.min(y_sky)) * 1.0/60.0/np.pi*180.0
    y_skyrange = np.abs(np.max(y_sky) - np.min(y_sky)) 
    y_skynum = len(y_sky)
    y_skydelta = y_skyrange / (y_skynum - 1)  # increment in evalational angle
    y_apdelta = wavelength / y_skyrange  # increment in x spatial coordinates
    y_aprange = wavelength / y_skydelta

    x_ap = np.linspace(-x_aprange/2.0, x_aprange/2.0, x_skynum)
    y_ap = np.linspace(-y_aprange/2.0, y_aprange/2.0, y_skynum)
    # x_ap = np.linspace(-int(x_skynum/2.0), int(x_skynum/2.0), x_skynum) * x_apdelta
    # y_ap = np.linspace(-int(x_skynum/2.0), int(x_skynum/2.0), x_skynum) * x_apdelta
    return x_ap, y_ap

def grid_to_ticks(grid):
    '''
    convert 2D grid to 1D ticks
    '''
    # determine according to the orientation
    if np.sum(grid[0] - grid[1]) == 0:
        return grid[0]
    else:
        return grid.T[0]

def flat_sqgrid_to_ticks(grid):
    '''
    convert 1D flatten square grid to 1D ticks
    '''
    ptnum = len(grid)
    size = int(np.sqrt(ptnum))
    if size*size != ptnum:
        raise AttributeError("Please input a flatten square grid!")
    if np.sum(grid[0:size] - grid[size:2*size]) == 0:
        return grid[0:size]
    else:
        return grid[0:-1:size]

def flat_sqgrid_to_2dgrid(grid):
    '''
    convert 1D flatten square grid to 2D grid
    '''
    ptnum = len(grid)
    size = int(np.sqrt(ptnum))
    if size*size != ptnum:
        raise AttributeError("Please input a flatten square grid!")
    return np.reshape(grid, (size, size))