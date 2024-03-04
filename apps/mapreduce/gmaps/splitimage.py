from  gmaps import downloader
import os, gdal
import haversine as hs
import math


def split_image(input_filename, tile_size_x=1500, tile_size_y=1500, out_path='.', output_filename= 'tile_'):
    ds = gdal.Open(input_filename)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
            os.system(com_string)


if __name__ == "__main__":
    marcianise_top_left = [41.05162612435181, 14.288554711937884,]
    origin = marcianise_top_left
    square_edge = 4.5
    dest = hs.inverse_haversine((origin[0], origin[1]), square_edge*math.sqrt(2), hs.Direction.SOUTHEAST)

    downloader.main(origin[1], origin[0], dest[1], dest[0], 17, r'test.tif', server="Google")
    os.system('convert test.tif -crop 1500x1500 test_%d.tif')
