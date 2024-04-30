from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal_array

# 读取dem文件
path = r"D:\Github_Repo\MLP_WaterDepth\3D.TIF"
# 将dem文件转为np.array数组
lmsdem = gdal_array.LoadFile(path)
nrows, ncols = lmsdem.shape

# 设置x轴坐标
x_array = np.zeros((nrows,ncols))
def xaxis(a,b):
  for i in range(a,b):
      x_array[i,:] = i
  return x_array
x = xaxis(0,nrows)

# 设置y轴坐标
y_array = np.zeros((nrows,ncols))
def yaxis(a,b):
  for i in range(a,b):
      y_array[:,i] = i
  return y_array
y = yaxis(0,ncols)


# 设置绘制区域的范围
region = np.s_[0:nrows,0+25:ncols]
x,y,z = x[region],y[region],lmsdem[region]


fig, ax = plt.subplots(subplot_kw = dict(projection='3d'))
ls = LightSource(270, 45)
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

#ax.set_zticks([700,705,710,715,720])
#ax.set_zticklabels(z.astype(int))
ax.view_init(20,-60)
ax.set_zlim(704,720)
# plt.axis("off")
ax.set_xticklabels([])
ax.set_yticklabels([])
#plt.show()
# plt.gca().get_xaxis().set_visible(False)
# plt.gca().get_yaxis().set_visible(False)

plt.savefig(r"C:\Users\Vitch\Desktop\3dfig.png",dpi=200)
