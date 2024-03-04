import os

from downloader import wgs_to_tile, saveTiff, getExtent
import PIL.Image as pil
import glob
import cv2
import numpy as np
import sys

import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def merge_tiles(x1, y1, x2, y2, z, tiles_dir):
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    outpic = pil.new('RGBA', (lenx * 256, leny * 256))
    print("lenx: {lenx} leny: {leny}".format(lenx=lenx, leny=leny))
    for y in range(leny):
        for x in range(lenx):
            small_pic = pil.open(tiles_dir+"/"+str(pos1y+y)+"_"+str(pos1x+x)+"_tile.jpeg")
            outpic.paste(small_pic, (x * 256, y * 256))
    print('Tiles merge completed')
    return outpic


if __name__ == '__main__':

    
    zone = 'test'
    zone_dir = "./images/"+zone

    maps = {'marcianise': [41.03699966153493, 14.282332207120104, 41.02356625889453, 14.300298406435248],
            'casapulla': [41.082515040404, 14.268917099445803, 41.066324184008195, 14.302277371886024],
            'casagiove': [41.08261546935178, 14.303388824263887, 41.07064562316231, 14.324267139938394],
            'capodrise': [41.05162612435181, 14.288554711937884, 41.03859863125092, 14.311235472131052],
            'portico_caserta': [41.06655157598264, 14.26429436809843, 41.052507617894044, 14.29476426353296],
            'puglianello': [],
            'campania': [41.677007, 13.892689, 41.087208, 15.770957],
            'test': [41.047047, 14.282363, 41.029596, 14.326394]
            }

    c_map = maps[zone]

    left = c_map[1]
    top = c_map[0]
    right = c_map[3]
    bottom = c_map[2]
    zoom = 17
    server = "Google"

    items = 4
    x_step = np.linspace(left, right, items)
    y_step = np.linspace(bottom, top, items)
    for i in range(items-1):
        for j in range(items-1):
            left = x_step[i]
            right = x_step[i+1]
            bottom = y_step[j]
            top = y_step[j+1]


            #Combine downloaded tile maps into one map
            outpic = merge_tiles(left, top, right, bottom, zoom,zone_dir+"/"+str(zoom)+"/tiles")
            outpic = outpic.convert('RGB')
            r, g, b = cv2.split(np.array(outpic))

            # Get the spatial information of the four corners of the merged map and use it for outputting
            extent = getExtent(left, top, right, bottom, zoom, server)
            gt = (extent['LT'][0], (extent['RB'][0] - extent['LT'][0]) / r.shape[1], 0, extent['LT'][1], 0,
                  (extent['RB'][1] - extent['LT'][1]) / r.shape[0])
            saveTiff(r, g, b, gt, zone_dir+"/"+str(i)+"_"+str(j)+".tif")
