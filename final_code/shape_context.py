import re
import time
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import copy
from munkres import Munkres,print_matrix
import datetime

def plot_scatter(data, saving_name):
    print('plotting the figures......')
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(data[0,:], data[1,:], data[2,:], color = 'indigo', marker = '.')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    saveing_path = './test_plot/'
    nowTime = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
    saving_name = saveing_path + saving_name + nowTime
    plt.savefig(saving_name)
    plt.close('all')
    # plt.show()

def sample_point(data_points, sample_num, plot_name):
    print('sampling the points......')

    # start = time.clock()
    data = copy.deepcopy(data_points)
    data_num = data.shape[1]
    count = 0
    points = list()
    while count < sample_num:
        random_index = np.random.randint(data_num, size=1)
#         print(random_index)
        if not np.isnan(data[0,random_index]):
            points.extend(random_index)
            for index in range(0,data_num):
                distance_x = data[0,index] - data[0,random_index]
                distance_y = data[1,index] - data[1,random_index]
                distance_z = data[2,index] - data[2,random_index]
                distance_sqr = distance_x**2 + distance_y**2 + distance_z**2
                if distance_sqr == 0:
                    data[0,index] = np.NaN
            count = count + 1
    plot_name = 'sample_point of ' + plot_name
    plot_scatter(data_points[:,points], plot_name)
    # end = time.clock()
    # print("Runtime: %.03f"%(end - start))

    return points

def shape_bins(points):
    print('extracting the features of the shape......')

    # start = time.clock()
    N = points.shape
    bins_all = list()
    dis_Block = 5
    theta_Block = 6
    phi_Block = 12

    for i in range(N[1]):
        distances = list()
        angle_theta = list()
        angle_phi = list()
        bins = np.zeros((dis_Block, theta_Block, phi_Block))

        for j in range(N[1]):
            if j != i:
                distance_x = points[0,j] - points[0,i]
                distance_y = points[1,j] - points[1,i]
                distance_z = points[2,j] - points[2,i]
                distance = np.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
                distances.append(distance)

                theta = np.arccos(distance_z / distance)
                angle_theta.append(theta * theta_Block / (2 * pi))

                phi = np.arccos(distance_x / np.sqrt(distance_x**2 + distance_y**2))
                if distance_y < 0:
                    phi = 2 * pi - phi
                elif distance_x == 0 and distance_y == 0:
                    phi = theta
                else:
                    phi = phi
                if int(phi * phi_Block / (2 * pi)) == 12:
                    phi = phi - 0.1
                angle_phi.append(phi * phi_Block / (2 * pi))

        mean_dis = np.mean(distances)
        distances = distances / mean_dis
        block_lens = 1
        distances_log = np.log(distances / block_lens)
        for k in range(len(distances_log)):
            if distances_log[k] <= 0:
                distances_log[k] = 0
            elif distances_log[k] <= 1:
                distances_log[k] = 1
            elif distances_log[k] <= 2:
                distances_log[k] = 2
            elif distances_log[k] <= 3:
                distances_log[k] = 3
            elif distances_log[k] <= 4:
                distances_log[k] = 4
            bins[int(distances_log[k]), int(angle_theta[k]), int(angle_phi[k])] += 1
        bins = np.reshape(bins,[dis_Block * theta_Block * phi_Block])
        bins_all.append(bins)
    # end = time.clock()
    # print("Runtime: %.03f"%(end - start))

    return bins_all

def cost_matrix(bins_A, bins_B):
    print('calculating the cost matrix......')

    # start = time.clock()
    row = 0
    col = 0
    cost = np.zeros((len(bins_A), len(bins_B)))
    for bin_A in bins_A:
        col = 0
        for bin_B in bins_B:
            cost[row, col] = 0.5 * np.sum(((bin_A - bin_B) ** 2) / (bin_A + bin_B + 0.00000001))
            col = col + 1
        row = row + 1
    # end = time.clock()
    # print("Runtime: %.03f"%(end - start))

    return cost

def readstl(filepath):
    print('reading the stl file......')

    # start = time.clock()

    # open the stl file
    with open(filepath,'r') as infile:
        content = infile.readlines() # read all stl data into content

    # read the header of the file. The first line of the stl file is the filename
    # name = content[0]
    # print('The name of the stl file:', name)

    # extract the coordinates from the content
    vertexs = list()
    for line in content:
        #     print(line)
        reg_vertex = 'vertex (.*?)\n'
        vertex = re.findall('vertex (.*?)\n',line)
        if len(vertex):
            vertex = vertex[0].split()
        vertexs.extend(vertex)

    # data_points contains all the vertex points in stl file
    data_points = np.transpose(np.array(vertexs).reshape((int(len(vertexs)/3),-1))).astype(np.float)

    # data_points_avg contains all the average points of three vertexs points in stl file
    x_avg = np.mean(np.transpose(data_points[0,:].reshape((-1,3))),axis=0)
    y_avg = np.mean(np.transpose(data_points[1,:].reshape((-1,3))),axis=0)
    z_avg = np.mean(np.transpose(data_points[2,:].reshape((-1,3))),axis=0)
    data_points_avg = np.asarray([x_avg, y_avg, z_avg])

    # plot the scattering of the vertexs in stl file
    # plot_scatter(data_points)
    # plot_scatter(data_points_avg)

    # end = time.clock()
    # print("Runtime: %.03f"%(end - start))

    return data_points_avg


pi = 3.1415926535
# samplepts_num = 500

growing_filepaths = ['./20180308Chienlab/growing/FE1546_2003_Neckcut.stl',
                     './20180308Chienlab/growing/LR1592_2007_Neckcut_Pcomcut.stl',
                     './20180308Chienlab/growing/LR4789_2010_Neckcut_Pcomcut.stl',
                     './20180308Chienlab/growing/TK928_2003_Neckcut_Pcomcut.stl',
                     './20180308Chienlab/growing/XJ3252_2008_Neckcut.stl']
stable_filepaths = ['./20180308Chienlab/stable/BI8022_2011_Neckcut_Pcomcut.stl',
                    './20180308Chienlab/stable/MM8894_2011_Neckcut.stl',
                    './20180308Chienlab/stable/RS4668_2003_Neckcut.stl',
                    './20180308Chienlab/stable/VR1339_2004_Neckcut.stl',
                    './20180308Chienlab/stable/WC610_2007_Neckcut.stl']
growing_name = ['FE1546', 'LR1592', 'LR4789', 'TK928', 'XJ3252']
stale_name = ['BI8022', 'MM8894', 'RS4668', 'VR1339', 'WC610']

m = Munkres()
start_all = time.clock()

growing_totals = list()
stable_totals = list()
interaction_totals = list()

# Compute the average cost of the pairs of two growing aneurysm
for i in range(0,5):
    filepath_A = growing_filepaths[i]
    data_points_avg_A = readstl(filepath_A)
    samples_A = sample_point(data_points_avg_A, samplepts_num, growing_name[i])
    data_sample_A = data_points_avg_A[:, samples_A]
    bins_A = shape_bins(data_sample_A)
    for j in range(i+1,5):
        name = growing_name[i] + ' & ' + growing_name[j]

        samplepts_num = min(data_points_avg_A.shape[1], data_points_avg_B.shape[1])
        if samplepts_num >= 2000:
            samplepts_num = 2000
        print('The number of the samples for', name, 'is', samplepts_num)

        filepath_B = growing_filepaths[j]
        data_points_avg_B = readstl(filepath_B)
        samples_B = sample_point(data_points_avg_B, samplepts_num, growing_name[j])
        data_sample_B = data_points_avg_B[:, samples_B]
        bins_B = shape_bins(data_sample_B)
        costAB = cost_matrix(bins_A, bins_B)
        # print(costAB)
        print('doing the maximum matching......')
        indexes = m.compute(costAB.tolist())
        print('calculating the total cost......')
        total = 0
        for row, column in indexes:
            value = costAB.tolist()[row][column]
            total += value
        print('The total cost of', name, 'is',total)
        print('='*80)
        growing_totals.append(total)

print('The total number of pairs of the growing aneurysm is', len(growing_totals))
print('The average total cost of the pairs of two growing aneurysm is %.2f'%(sum(growing_totals)/len(growing_totals)))
print('='*100)

# Compute the average cost of the pairs of two stable aneurysm
for i in range(0,5):
    filepath_A = stable_filepaths[i]
    data_points_avg_A = readstl(filepath_A)
    samples_A = sample_point(data_points_avg_A, samplepts_num, stale_name[i])
    data_sample_A = data_points_avg_A[:, samples_A]
    bins_A = shape_bins(data_sample_A)
    for j in range(i+1,5):
        name = stale_name[i] + ' & ' + stale_name[j]

        samplepts_num = min(data_points_avg_A.shape[1], data_points_avg_B.shape[1])
        if samplepts_num >= 2000:
            samplepts_num = 2000
        print('The number of the samples for', name, 'is', samplepts_num)

        filepath_B = stable_filepaths[j]
        data_points_avg_B = readstl(filepath_B)
        samples_B = sample_point(data_points_avg_B, samplepts_num, stale_name[j])
        data_sample_B = data_points_avg_B[:, samples_B]
        bins_B = shape_bins(data_sample_B)
        costAB = cost_matrix(bins_A, bins_B)
        # print(costAB)
        print('doing the maximum matching......')
        indexes = m.compute(costAB.tolist())
        print('calculating the total cost......')
        total = 0
        for row, column in indexes:
            value = costAB.tolist()[row][column]
            total += value
        print('The total cost of', name, 'is',total)
        print('='*80)
        stable_totals.append(total)

print('The total number of pairs of the growing aneurysm is', len(stable_totals))
print('The average total cost of the pairs of two growing aneurysm is %.2f'%(sum(stable_totals)/len(stable_totals)))
print('='*100)

# Compute the average cost of the pairs of two stable aneurysm
for i in range(0,5):
    filepath_A = stable_filepaths[i]
    data_points_avg_A = readstl(filepath_A)
    samples_A = sample_point(data_points_avg_A, samplepts_num, stale_name[i])
    data_sample_A = data_points_avg_A[:, samples_A]
    bins_A = shape_bins(data_sample_A)
    for j in range(0,5):
        name = stale_name[i] + ' & ' + growing_name[j]

        samplepts_num = min(data_points_avg_A.shape[1], data_points_avg_B.shape[1])
        if samplepts_num >= 2000:
            samplepts_num = 2000
        print('The number of the samples for', name, 'is', samplepts_num)

        filepath_B = growing_filepaths[j]
        data_points_avg_B = readstl(filepath_B)
        samples_B = sample_point(data_points_avg_B, samplepts_num, growing_name[j])
        data_sample_B = data_points_avg_B[:, samples_B]
        bins_B = shape_bins(data_sample_B)
        costAB = cost_matrix(bins_A, bins_B)
        # print(costAB)
        print('doing the maximum matching......')
        indexes = m.compute(costAB.tolist())
        print('calculating the total cost......')
        total = 0
        for row, column in indexes:
            value = costAB.tolist()[row][column]
            total += value
        print('The total cost of', name, 'is',total)
        print('='*80)
        interaction_totals.append(total)

print('The total number of pairs of the growing aneurysm is', len(interaction_totals))
print('The average total cost of the pairs of two growing aneurysm is %.2f'%(sum(interaction_totals)/len(interaction_totals)))
print('='*100)

end_all = time.clock()
print("Total runtime: %.03f"%(end_all - start_all))


# filepath_A = './EE211A proj/stl/STL/BI8022_2010_Neckcut_Pcomcut.stl'
# data_points_avg_A = readstl(filepath_A)
# filepath_B = './EE211A proj/stl/STL/BI8022_2010_Neckcut_Pcomcut.stl'
# data_points_avg_B = readstl(filepath_B)

# start_all = time.clock()
# samples_A = sample_point(data_points_avg_A, samplepts_num)
# data_sample_A = data_points_avg_A[:, samples_A]
# samples_B = sample_point(data_points_avg_B, samplepts_num)
# data_sample_B = data_points_avg_B[:, samples_B]
# # plot_scatter(data_sample_A, 'A1')
# # plot_scatter(data_sample_B, 'B1')

# bins_A = shape_bins(data_sample_A)
# bins_B = shape_bins(data_sample_B)
# # print(bins_A[0])
# # print('*'*20)
# # print(bins_B[0])
# # print('*'*20)

# costAB = cost_matrix(bins_A, bins_B)
# print(costAB)

# print('doing the maximum matching......')
# m = Munkres()
# indexes = m.compute(costAB.tolist())
# # print(indexes)

# print('calculating the total cost......')
# total = 0
# for row, column in indexes:
#     value = costAB.tolist()[row][column]
#     total += value
# if total < 12000:
#     print('It is aneursym with cost:',total)
# else:
#     print('Not aneursym with cost:',total)
# end_all = time.clock()
# print("Total runtime: %.03f"%(end_all - start_all))
