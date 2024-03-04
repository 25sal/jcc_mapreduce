import glob
from downloader   import get_urls, wgs_to_tile
import os
import sys

def downloaded_tiles(tiles_dir,lenx):
    file_ids = []
    files = glob.glob(tiles_dir+"/*.jpeg")
    for filename in files:
        filename = os.path.basename(filename)
        parts = filename.split("_")
        y = int(parts[0])
        x = int(parts[1])
        i = y * lenx + x
        file_ids.append(i)
    return file_ids


if __name__ == '__main__':
    files = glob.glob("/data/temp/pietro/*.jpeg")
    print(len(files))
    zone = 'campania'
    maps = {'marcianise': [41.03699966153493, 14.282332207120104, 41.02356625889453, 14.300298406435248],
            'casapulla': [41.082515040404, 14.268917099445803, 41.066324184008195, 14.302277371886024],
            'casagiove': [41.08261546935178, 14.303388824263887, 41.07064562316231, 14.324267139938394],
            'capodrise': [41.05162612435181, 14.288554711937884, 41.03859863125092, 14.311235472131052],
            'portico_caserta': [41.06655157598264, 14.26429436809843, 41.052507617894044, 14.29476426353296],
            'puglianello': [],
            'campania': [41.677007, 13.892689, 41.087208, 15.770957]
            }

    c_map = maps[zone]

    # (left, top, right, bottom, zoom, filePath, style='s', server="Google China"):
    # Get the urls of all tiles in the extent

    # urls = get_urls(c_map[1], c_map[0], c_map[3], c_map[2], 17, "Google", 's')

    x1 = c_map[1]
    y1 = c_map[0]
    x2 = c_map[3]
    y2 = c_map[2]
    z = 17
    source = "Google"
    style = 's'
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    print("Positions：{x} X {y}".format(x=pos1x, y=pos1y))
    print("Total tiles number：{x} X {y}: {z}".format(x=lenx, y=leny, z=lenx*leny))
    '''
    for filename in files:
        basename = os.path.basename(filename)
        parts = basename.split("_")
        y = int(parts[0])
        x = int(parts[1])
        new_name = str(pos1y+y)+"_"+str(pos1x+x)+"_tile.jpeg"
        # print(basename, new_name)
        # os.rename(filename, "/data/temp/pietro/"+new_name)
    '''
