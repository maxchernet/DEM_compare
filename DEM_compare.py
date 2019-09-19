"""

Validation of high resolution satellite derived DEM and 'ground-truth' LIDAR.

Two images are stacked to common resolution, extent and projection using functionality of GDAL.
The issue is that geographical projection of satellite can be not perfect. This is addressed
by using opencv translation function to correct possible error in geographical position.
Finally two sets of data are compared and plotted to a file.

Author: Maxim Chernetskiy, UCL, Mullard Space Science Laboratory, 2019

Version 0.5

Input files have to be georeferenced TIFF images with Digital Elevation Models (DEMs).
Input images can have different projections, spatial resolutions and extents.

Example of usage:
python DEM_compare.py --f_in_sat data/satDEM.tif --f_in_lidar data/lidarDEM.tif

"""

import numpy as np
import gdal
import subprocess as sp
import os, sys
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker
import cv2
import argparse


def open_geo_file(in_file):
    """
    Open geo file and get image and projection

    :param str in_file: Input file
    :return: Projection and Image data
    :rtype: str, array
    """

    if os.path.isfile(in_file):
        gds = gdal.Open(in_file)
        srs = gds.GetProjection()
        print('Raster size: %d X %d' % (gds.RasterXSize, gds.RasterYSize))
        print('Raster projection: %s' % srs)
        img = gds.GetRasterBand(1).ReadAsArray()
    else:
        print "File %s does not exist" % in_file
        sys.exit()

    return srs, img


def gdal_merge(file_in_dst, file_in_srs, file_out):
    """
    Use gdal_merge.py to stack two images to common resolution and extent.
    This function get file_in_dst and file_in_srs, resize file_in_srs to extent of
    file_in_dst and save both into one file as separate bands.

    :param str file_in_dst:
    :param str file_in_srs:
    :param str file_out:
    :return: None
    """

    # Get upper left X and Y geo-coordinates
    gds = gdal.Open(file_in_dst)
    ulx, xs, xxx, uly, yyy, ys = gds.GetGeoTransform()
    # Get lower right coordinates
    lrx = ulx + xs * gds.RasterXSize
    lry = uly + ys * gds.RasterYSize
    # Extent to which image will be resized
    extent_str = '%f %f %f %f' % (ulx, uly, lrx, lry)

    if os.path.isfile(file_out):
        print('Temporary file exists. Removing...')
        sp.check_output('rm ' + file_out, shell=True)
    print('Resizing to extent %s' % extent_str)
    # file_resized = 'data/' + file_in_srs.split('.')[0] + '_resized.tif'
    retval = sp.check_output('gdal_merge.py -of gtiff -separate -o %s %s %s -ul_lr %s -a_nodata -2000' %
                             (file_out, file_in_srs, file_in_dst, extent_str), shell=True)
    print(retval)
    gds = None


def get_good_val(arr1, arr2, thresh = 0):
    """
    1. Get 'good' only values from arr1 and arr2. I.e. ignore negative which are negative;
    2. Save results as vectors;
    3. Get correlation coefficient, slope and interception from these two vectors.

    :param array arr1: 2D array one
    :param array arr2: 2D array two
    :param float thresh:
    :return: Two vectors without 'bad' values, correlation coefficient, slope and interception
    :rtype: array, array, float, float, float
    """

    arr1_vec = arr1[np.logical_and(arr1 > thresh, arr2 > 0)]
    arr2_vec = dem_lidar[np.logical_and(arr1 > thresh, arr2 > 0)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(arr1_vec, arr2_vec)

    return arr1_vec, arr2_vec, r_value, slope, intercept


def translate_corr(arr_dst, arr_srs):
    """
    Correct position of arr_dst according to arr_srs using opencv translation.

    :param array arr_dst: Image data to correct
    :param array arr_srs: Reference image data
    :return: Corrected version of arr_dst
    :rtype: array
    """

    r2_arr = []
    rows, cols = arr_dst.shape
    xi = np.arange(0, 100, 2)
    yi = np.arange(0, 100, 2)
    for i in xi:
        for j in yi:
            # Translation matrix - XYZ direction of translation
            M = np.float32([[1, 0, i], [0, 1, j]])
            # Do translation
            arr_dst_corr = cv2.warpAffine(arr_dst, M, (cols, rows), borderValue=-20000)
            arr_dst_vec, arr_srs_vec, r2_value, slope, intercept = get_good_val(arr_dst_corr, arr_srs)
            r2_arr = np.append(r2_arr, r2_value)
    print('Optimal r^2: %f, %d' % (np.max(r2_arr), np.argmax(r2_arr)))
    # Convert 1D index to 2D
    indx, indy = np.unravel_index(np.argmax(r2_arr), (xi.shape[0], yi.shape[0]))
    x_shift = xi[indx]
    y_shift = yi[indy]
    print('x_shift, y_shift: %d %d' % (x_shift, y_shift))

    # Do final translation using just found optimal values
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    arr_dst_corr = cv2.warpAffine(arr_dst, M, (cols, rows), borderValue=-20000)

    return arr_dst_corr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validation of high resolution satellite derived DEM and 'ground-truth' LIDAR")

    parser.add_argument('--f_in_sat', dest='f_in_sat', help='Satellite DEM file')
    parser.add_argument('--f_in_lidar', dest='f_in_lidar', help='LIDAR DEM file')

    args = parser.parse_args()

    if args.f_in_sat == None:
        print('No Satellite DEM is provided')
        sys.exit()
    if args.f_in_lidar == None:
        print('No LIDAR DEM is provided')
        sys.exit()
    f_in_sat = args.f_in_sat
    f_in_lidar = args.f_in_lidar

    dir_data = 'data/'
    dir_fig = 'fig/'

    # f_in_sat = '../pc_align_dem_02/dtm-trans_source-DEM.tif'
    # f_in_lidar = '/Volumes/lacie_data/satellite/Trondheim/dtm1/data/dtm1_33_124_141.tif'

    f_out_lidar = dir_data + f_in_lidar.split('.')[0] + '_resized.tif'

    f_lidar_warped = f_in_lidar.split('/')[-1].split('.')[0] + "warped.tif"

    # Get satellite and LIDAR geo-projections and images
    sat_srs, sat_img = open_geo_file(f_in_sat)
    lidar_srs, lidar_img = open_geo_file(f_in_lidar)

    plt.subplot(121)
    plt.imshow(sat_img, vmin=-20, vmax=140)
    plt.subplot(122)
    plt.imshow(lidar_img, vmin=0, vmax=500)
    plt.savefig(dir_fig + 'orig_images.png')

    # Warp two images to common projection
    com_str = 'gdalwarp %s %s -s_srs %s -t_srs %s' % (f_in_lidar, dir_data + f_lidar_warped, lidar_srs, sat_srs)
    sp.check_output(com_str, shell=True)

    # Stack two images together
    gdal_merge(f_in_sat, f_in_lidar, f_out_lidar)

    if os.path.isfile(f_out_lidar) == False:
        print('File %s does not exist')
        sys.exit()

    # Open resized file which has two DEMs and get these DEMs as two separate arrays
    gds = gdal.Open(f_out_lidar)
    dem_sat = gds.GetRasterBand(2).ReadAsArray()
    dem_lidar = gds.GetRasterBand(1).ReadAsArray()
    gds = None

    plt.subplot(121)
    plt.title('Satellite DEM', fontsize=18)
    plt.imshow(dem_sat, vmin=-10, vmax=160)
    plt.colorbar(fraction=0.02, pad=0.01)
    plt.subplot(122)
    plt.title('LIDAR DEM', fontsize=18)
    plt.imshow(dem_lidar, vmin=0, vmax=160)
    plt.colorbar(fraction=0.02, pad=0.01)
    plt.tight_layout()
    plt.savefig(dir_fig + 'satellite-lidar_dem.png')

    # Get vectors which have only 'good' values and correlation coefficient
    dem_sat_vec, dem_lid_vec, r_value, slope, intercept = get_good_val(dem_sat, dem_lidar)
    print('r^2=%0.4f; slope=%0.4f; intercept=%0.4f' % (r_value, slope, intercept))

    dem_sat_corr = translate_corr(dem_sat, dem_lidar)
    dem_sat_vec_corr, dem_lid_vec_corr, r_value_corr, slope_corr, intercept_corr = get_good_val(dem_sat_corr, dem_lidar)
    print('Before correction: r2=%0.4f; slope=%0.4f; intercept=%0.4f' % (r_value, slope, intercept))
    print('After correction: r2=%0.4f; slope=%0.4f; intercept=%0.4f' % (r_value_corr, slope_corr, intercept_corr))

    # Plot results of comparison
    plt.figure(figsize=(15, 6))

    plt.subplot(121)
    plt.title('Before position correction', fontsize=18)

    retval = plt.hist2d(dem_sat_vec, dem_lid_vec, (100, 100),
                        cmap=plt.get_cmap('gnuplot2_r'),
                        norm=mcolors.PowerNorm(0.1), range=[[0, 200], [0, 200]])

    cb = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=3)
    cb.locator = tick_locator
    cb.ax.tick_params(labelsize=16)
    cb.update_ticks()

    plt.subplot(122)
    plt.title('After position correction', fontsize=18)
    retval = plt.hist2d(dem_sat_vec_corr, dem_lid_vec_corr, (100, 100),
                        cmap=plt.get_cmap('gnuplot2_r'),
                        norm=mcolors.PowerNorm(0.1), range=[[0, 200], [0, 200]])

    cb = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=3)
    cb.locator = tick_locator
    cb.ax.tick_params(labelsize=16)
    cb.update_ticks()

    plt.tight_layout()

    plt.savefig(dir_fig + 'dems_scatter_plot.png')
