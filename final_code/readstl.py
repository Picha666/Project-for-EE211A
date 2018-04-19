import re
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# record the time to start
start = time.clock()
# open the stl file
with open('/Users/px/Desktop/stl/BI8022_2010_Neckcut_Pcomcut.stl','r') as infile:
    content = infile.readlines() # read all stl data into content

# read the header of the file. The first line of the stl file is the filename
name = content[0]
print('The name of the stl file:', name)

colors = list()
vertexs = list()
for line in content:
    reg_color = 'facet normal (.*?)\n'
    color = re.findall(reg_color,line)
    if len(color):
        color = color[0].split()
    colors.extend(color)
    reg_vertex = 'vertex (.*?)\n'
    vertex = re.findall('vertex (.*?)\n',line)
    if len(vertex):
        vertex = vertex[0].split()
    vertexs.extend(vertex)
data_points = np.zeros((3, len(colors)))
data_points[0,:] = np.asarray(vertexs[0::3])
data_points[1,:] = np.asarray(vertexs[1::3])
data_points[2,:] = np.asarray(vertexs[2::3])

# plot the scattering of the vertexs in stl file
ax = plt.subplot(111, projection='3d')
ax.scatter(data_points[0,:], data_points[1,:], data_points[2,:], c=colors, marker = '^')
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

# calculate and output the runtime
end=time.clock()
print("The runtime：%.03f"%(end-start))
