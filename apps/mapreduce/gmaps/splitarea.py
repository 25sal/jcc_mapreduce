import haversine as hs
import math
from gmaps import downloader

if __name__ == "__main__":
    maps = {'marcianise': [41.05129674906493, 14.271082713309346, 41.05134529402639, 14.324705433047116]}
    c_map = maps['marcianise']
    x = hs.haversine((c_map[0], c_map[1]), (c_map[0], c_map[3]))
    print(f"distance is { x } km")

    distance = x
    origin = [c_map[0], c_map[1]]
    for i in range(math.ceil(distance/1.5)):
        for j in range(math.ceil(distance/1.5)):
            dest = hs.inverse_haversine((origin[0], origin[1]), 1.4*math.sqrt(2), hs.Direction.SOUTHEAST)

            print("origin is ", origin, "destination is ", dest)
            x = hs.haversine((origin[0], origin[1]), (origin[0], dest[1]))
            print(f"distance EAST is {x} km")
            x = hs.haversine((origin[0], origin[1]), (dest[0], origin[1]))
            print(f"distance SOUTH  {x} km")

            downloader.main(origin[1], origin[0], dest[1], dest[0], 17, 'images/test'+str(i)+'_'+str(j)+'.tif', server="Google")
            origin[1] = dest[1]

        origin[1] = c_map[1]
        dest = hs.inverse_haversine(origin, (i+1)*1.5, hs.Direction.SOUTH)
        origin[0] = dest[0]
