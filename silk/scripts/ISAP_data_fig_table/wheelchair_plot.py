import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from path import Path


# PATH = "/data/wheelchair_data/inu_data/inu_lib/viz_siftpose"
PATH = "/data/wheelchair_data/inu_data/inu_lib/viz_sup13"
# PATH = "/data/wheelchair_data/inu_data/inu_lib/viz_sup13_0"
# PATH = "/data/wheelchair_data/inu_data/inu_lib/viz_sup13_1"


def main(seq):
    # Load the numpy array (adjust the path and file format as needed)
    _array = np.loadtxt(Path(PATH)/seq+'.txt')
    
    print(_array.shape) #nx12
    # Extract x and y coordinates
    _x = _array[:, 3]
    _y = _array[:, 7]
    _z = _array[:, 11]
    
    # plt.scatter(_x, _z, s=0.1, c=cmap(seq), label = '{:0}'.format(seq))
    plt.scatter(_x, _z, s=0.1, c='red', label = seq)

    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    plt.title('ours odometry')
    plt.legend()

    # ax = fig.add_subplot(111, projection='3d')
    # plt.scatter(gt_x, gt_y, gt_z, s=1, c='red', label='GT')
    # plt.scatter(_x, _y, _z, s=1, c='blue', label = '')
    # ax.scatter(gt_x, gt_y, gt_z, s=1, c='red', label='GT')
    # ax.scatter(_x, _y, _z, s=1, c='blue', label = '')
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_ylabel('Z-axis')
    # ax.set_title('3D trajectory on KITTI odometry seq 09')




# test_seqs = ["0_", "2_", "3_", "8_"]       
# test_seqs = ["0_", "2_0", "3_", "8_"] 
# test_seqs = ["0_", "2_0", "3_", "8_0"] 
test_seqs = ["0_", "2_", "3_", "8_0"] 
      
# test_seqs = ["0_", "2_", "2_0", "3_", "8_", "8_0"]       

# test_seqs = [0, 2, 3, 8]       
# test_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]       

if __name__ == '__main__':
    fig = plt.figure(figsize=(5, 5))  # Width=4 inches, Height=3 inches
    cmap=plt.cm.get_cmap('hsv', 16)

    for seq in test_seqs:
        main(seq)
        
    plt.grid(True)
    plt.savefig(Path(PATH)/'some_traj_2.png', dpi=300, bbox_inches='tight')

    print("done")
    
    
