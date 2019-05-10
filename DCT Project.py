#import pptk
import numpy as np
import scipy.io

from scipy.fftpack import dct, idct
from scipy.spatial import distance

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D




#############################################################
#Original image before compression                          #
#############################################################

data = np.genfromtxt('C:/Users/lzhongab/Desktop/trex.dat')
    
#  Data Preprocessing
round_data = np.around(data)  # 取整
Deduplicate_data = np.array(list(set([tuple(row) for row in round_data])))  # 去除重复元素



a = np.min(Deduplicate_data, axis = 0)  # 得到每一维的最小值（都是负数）

Deduplicate_data[:, 0] = Deduplicate_data[:, 0] - a[0]  # 将点云搬移到正半轴空间
Deduplicate_data[:, 1] = Deduplicate_data[:, 1] - a[1]
Deduplicate_data[:, 2] = Deduplicate_data[:, 2] - a[2]



fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter([row[0] for row in Deduplicate_data], 
           [row[1] for row in Deduplicate_data], 
           [row[2] for row in Deduplicate_data], 
           c = [row[2] for row in Deduplicate_data])

ax.view_init(azim = 0, elev = 90)
pyplot.show()



e = np.max(Deduplicate_data, axis = 0).astype(np.int32)  # 得到每一维的最大值

threeD_matrix = np.zeros(shape = (e[0]+1, e[1]+1, e[2]+1))  # 创建一个3D矩阵

for i in range(Deduplicate_data.shape[0]):
    x = Deduplicate_data[i, :].astype(np.int32)
    threeD_matrix[x[0], x[1], x[2]] = 255             # 有点的位置赋值255



t = threeD_matrix

t = dct(t, axis = 0, norm = 'ortho')  # 3D-DCT
t = dct(t, axis = 1, norm = 'ortho')
t = dct(t, axis = 2, norm = 'ortho')  

threeD_matrixDCT = t

#quantization_step = 256
#quantizated_matrix = np.around(threeD_matrixDCT / quantization_step)  # quantization



zigzag_size = 64
#threeD_matrixDCT[zigzag_size:(threeD_matrixDCT.shape[0]), :, :] = 0
#threeD_matrixDCT[:, zigzag_size:(threeD_matrixDCT.shape[1]), :] = 0
#threeD_matrixDCT[:, :, zigzag_size:(threeD_matrixDCT.shape[2])] = 0
                   
threeD_matrixDCT = np.around(threeD_matrixDCT)

zig_mat = threeD_matrixDCT[0:zigzag_size, 0:zigzag_size, 0:zigzag_size]

#scipy.io.savemat('zig_matrix', mdict = {'zig_mat': zig_mat})

#compression_ratio = (int(data.shape[0])) / (int(len(np.nonzero(threeD_matrixDCT)[0])))*3/4

compression_ratio = (int(data.shape[0])*3) / (zigzag_size)**3


#############################################################
#Image after compression                                    #
#############################################################

#h = quantizated_matrix * quantization_step

h = threeD_matrixDCT

h = idct(h, axis = 2, norm = 'ortho')  # 3D-IDCT
h = idct(h, axis = 1, norm = 'ortho')
h = idct(h, axis = 0, norm = 'ortho')

threeD_matrixIDCT = h



#threshold = (np.max(threeD_matrixIDCT) + np.min(threeD_matrixIDCT)) /2

#threeD_matrixIDCT[threeD_matrixIDCT[:, :, :] < threshold] = 0

threeD_cor = list(np.nonzero(threeD_matrixIDCT))  # 获取非零元素的坐标

point_cloud = np.zeros((len(threeD_cor[0]), 3))

for x in range(0, len(threeD_cor[0])):
    point_cloud[x] = [threeD_cor[0][x], threeD_cor[1][x], threeD_cor[2][x]]





# 计算error rate 
threeD_matrixIDCT[threeD_matrixIDCT[:, :, :] != 0] = 255

#hamming_distance = 0
#for i in range(threeD_matrix.shape[2]):
#    if ((threeD_matrixIDCT[:, :, i] == threeD_matrix[:, :, i]).any()):
#        hamming_distance = hamming_distance + 1

oneD_matrix = threeD_matrix.flatten()
oneD_matrix_idct = threeD_matrixIDCT.flatten()

#hamming_distance = distance.hamming(oneD_matrix, oneD_matrix_idct) * oneD_matrix.shape[0]
#hamming_distance = distance.hamming(oneD_matrix, oneD_matrix_idct)



#error_rate = hamming_distance / point_cloud.shape[0]



# 计算MSE
#mse = np.sqrt(((threeD_matrix - threeD_matrixIDCT)**2).mean(axis = None))  




fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter([row[0] for row in point_cloud], 
           [row[1] for row in point_cloud], 
           [row[2] for row in point_cloud], 
           c = [row[2] for row in point_cloud])

ax.view_init(azim = 0, elev = 90)

pyplot.show()




print('\n')
print("The compression ration is:")
print(compression_ratio)
print('\n')
print("The error rate is:")
print(hamming_distance)
print('\n')
print("The MSE is:")
print(mse)

#qwe = pptk.viewer(Deduplicate_data)
#qwe = pptk.viewer(point_cloud)


