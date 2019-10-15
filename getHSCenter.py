import pya
import pandas as pd 
import numpy as np 


hs_layer = [21]
gds_in = gdsin
csv_out = csvout
head = ['id','x','y']

layout = pya.Layout()
layout.read(gds_in)
cell = layout.top_cell()
count = 0
id = 0
data=[]
for i in hs_layer:
    print("layer",i)
    layer = layout.layer(i, 0)
    core_iter = layout.begin_shapes(cell, layer)

    while not core_iter.at_end():
        count += 1
        id += 1
        core_box = core_iter.shape().bbox()
        center = core_box.center()
        data = data.append([id, center.x, center.y])
        #print(center.x, center.y)
        core_iter.next()

print(count)
df = pd.DataFrame(data, columns = head)

df.to_csv(csv_out)

