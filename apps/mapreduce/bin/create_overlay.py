from  gmaps.downloader import pixls_to_mercator, mercator_to_wgs
import json

zb = {'RT': [70603, 48814],'LT':[70594,48814], 'RB':[70603,48823], 'LB':[70594,48823], 'z':17}

map_file = open("mapper_input_campania_short.txt", "r")
rows = map_file.readlines()
groups = {}
for row in rows:
    json_obj = json.loads(row)
    if json_obj['group'] not in json_obj.keys():
        groups[json_obj['group']]={}
        groups[json_obj['group']]['pos1x']=json_obj['pos1x']
        groups[json_obj['group']]['pos1y']=json_obj['pos1y']
        groups[json_obj['group']]['start_row']=json_obj['start_row']
        groups[json_obj['group']]['start_col']=json_obj['start_col']

with open("results.csv", "r") as results:
    lines =results.readlines()
    js_str = "var "
    for line in lines:
        line = line.split(" ")
        line = line[1].split(",")
        group = int(line[1])
        fitness = (float(line[2])-50)/100
        lx = groups[group]['pos1x']+groups[group]['start_row']
        ly = groups[group]['pos1y']+groups[group]['start_col']


        zb = {'RT': [lx+36,ly],'LT':[lx, ly],
               'RB':[lx+36,ly+36], 'LB':[lx, ly+36], 'z':17}


        pixels = pixls_to_mercator(zb)
        coors1 = mercator_to_wgs(pixels['LT'][0],pixels['LT'][1])
        coors2 = mercator_to_wgs(pixels['RB'][0],pixels['RB'][1])
        js_str += f"pol = L.polygon([[{coors1[1]},{coors1[0]}]," \
                +f"[{coors2[1]},{coors1[0]}], [{coors2[1]},{coors2[0]}]," \
                +f"[{coors1[1]},{coors2[0]}]" \
                +f"]).addTo(map); \n"
        js_str += "pol.setStyle({" \
               + "color: 'red', " \
               + "fillColor: '#f03', " \
               + f"fillOpacity: {fitness}" \
               + "});\n"


    print(js_str)
         