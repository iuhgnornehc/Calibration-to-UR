import cv2
import numpy as np
import glob
from math import *
import pandas as pd
import os

# TODO:选择可成功检测角点且结果准确的数据标记
# good_picture = [1,2,3,4]
good_picture = [1,2,3,4,5,6,7,9,10,11]
file_num = len(good_picture)

# -------------------- 计算board to cam 变换矩阵 -----------------------------------
R_all_chess_to_cam_1 = []
T_all_chess_to_cam_1 = []
for i in good_picture:
    print('b2c:', i)
    # ------------ 从python中获取矩阵 -------------------------------------
    # TODO；获取motive中的标定板变换矩阵
    # RT = np.linalg.inv(np.load(f'RT_Motive_to_board/{i}.npy', allow_pickle=True))
    RT = np.load(f'RT_Motive_to_board/{i}.npy', allow_pickle=True)

    if RT is not None:
        print(RT)
        R_all_chess_to_cam_1.append(RT[:3, :3])
        T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3, 1)))


# -------------------- 计算end to base变换矩阵 -----------------------------------
R_all_end_to_base_1 = []
T_all_end_to_base_1 = []
for i in good_picture:
    print('e2b:', i)
    RT = np.load(f'RT_Base_to_End/{i}.npy', allow_pickle=True)
    # RT[:3, 3] = RT[:3, 3] * 1000

    # 眼在手外--需求逆   (end to base)
    RT = np.linalg.inv(RT)

    print(RT)
    R_all_end_to_base_1.append(RT[:3, :3])
    T_all_end_to_base_1.append(RT[:3, 3].reshape((3, 1)))


# -------------------- 手眼标定 ------------------------------------------------
if len(R_all_end_to_base_1) == len(R_all_chess_to_cam_1) and len(R_all_end_to_base_1) != 0:
    R, T = cv2.calibrateHandEye(R_all_end_to_base_1, T_all_end_to_base_1, R_all_chess_to_cam_1,
                                T_all_chess_to_cam_1)  # 手眼标定
    RT = np.column_stack((R, T))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))  # 即为base to cam变换矩阵

    print('Base to cam：')
    print(RT)
    np.save('RT.npy', RT)

    # 结果验证，原则上来说，每次结果相差较小
    # 对比恒定不变的 chess to end 矩阵
    for i in range(len(good_picture)):
        RT_end_to_base = np.column_stack((R_all_end_to_base_1[i], T_all_end_to_base_1[i]))
        RT_end_to_base = np.row_stack((RT_end_to_base, np.array([0, 0, 0, 1])))
        # print(RT_end_to_base)
        RT_chess_to_cam = np.column_stack((R_all_chess_to_cam_1[i], T_all_chess_to_cam_1[i]))
        RT_chess_to_cam = np.row_stack((RT_chess_to_cam, np.array([0, 0, 0, 1])))
        # print(RT_chess_to_cam)

        RT_cam_to_end = np.column_stack((R, T))
        RT_cam_to_end = np.row_stack((RT_cam_to_end, np.array([0, 0, 0, 1])))
        # print(RT_cam_to_end)

        RT_chess_to_base = RT_end_to_base @ RT_cam_to_end @ RT_chess_to_cam  # 即为固定的棋盘格相对于机器人基坐标系位姿
        RT_chess_to_base = np.linalg.inv(RT_chess_to_base)
        print('第', i, '次')
        print(RT_chess_to_base[:3, :])
        print('')
else:
    print("No good pic.")


