import glob
import cv2
import os

preamble = "rejoin/test_"
images = glob.glob(preamble+"*.tif")
col = 1
row = 0
last_line = 0
input_files = ""

for i in range(len(images)):

    if os.path.isfile(preamble+str(i)+".tif"):
        im = cv2.imread(preamble+str(i)+".tif")
    else:
        im = None
    if im is not None and im.shape[0] == 1500 and im.shape[1] == 1500:
        input_files += " " + preamble+str(i)+".tif"
        if col == 1:
            row += 1
        last_line = 0
    else:
        if last_line == 1:
            break
        col += 1
        last_line = 1

col -= 1
cmd = "montage " + input_files + " -geometry 1500x1500 -tile "+str(col)+"x"+str(row) + " result.tif"
print(cmd)
os.system(cmd)
