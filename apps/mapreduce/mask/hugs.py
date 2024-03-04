import cv2
import numpy as np


def hugs(filename, outfile):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,10,600,apertureSize = 3)

    # lines = cv2.HoughLines(edges,1,np.pi/180,120)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    # img.fill(255)
    img_xsize = 4500
    img_ysize = 4500
    for line in lines:
        for rho,theta in line:

            if   rho > 0 and (theta == 0 or (theta >1.55 and theta < 1.6)):
                print(rho, theta, line)

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + img_xsize*(-b))
                y1 = int(y0 + img_ysize*(a))
                x2 = int(x0 - img_xsize*(-b))
                y2 = int(y0 - img_ysize*(a))

                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imwrite(outfile,img)


if __name__ == "__main__":
    # hugs('images/Capodrise_40.png', 'images/result.png')
    # hugs('images/Merged_Image.png', 'images/result.png')
    hugs('images/casagiove_mask.png', 'images/casagiove_hugs.png')