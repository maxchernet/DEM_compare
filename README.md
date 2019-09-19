# DEM_compare
Validation of high resolution satellite derived DEM and 'ground-truth' LIDAR

Two images are stacked to common resolution, extent and projection using functionality of GDAL.
The issue is that geographical projection of satellite can be not perfect. This is addressed
by using opencv translation function to correct possible error in geographical position.
Finally two sets of data are compared and plotted to a file.

Author: Maxim Chernetskiy, UCL, Mullard Space Science Laboratory, 2019

Version 0.5

Input files have to be georeferenced TIFF images with Digital Elevation Models (DEMs).
Input images can have different projections, spatial resolutions and extents.

Example of usage:
python DEM_compare.py --f_in_sat ../pc_align_dem_02/dtm-trans_source-DEM.tif --f_in_lidar = /Volumes/lacie_data/satellite/Trondheim/dtm1/data/dtm1_33_124_141.tif
