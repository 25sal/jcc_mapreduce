from mask.contour import  extract_shapes
from mask.hugs import hugs
if __name__ == "__main__":
    hugs('images/Capodrise_40.png', 'images/result.png')
    extract_shapes('images/result.png')