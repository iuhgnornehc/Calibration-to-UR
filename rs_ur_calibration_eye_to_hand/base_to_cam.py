import os
import cv2
import numpy as np
import scipy.io as sio
import pyrealsense2 as rs

# TODO: K:相机内参；
#       chess_board_x_num：棋盘格x方向格子数-1，即交叉点数
#       chess_board_y_num：棋盘格y方向格子数-1，即交叉点数
#       chess_board_len：单位棋盘格长度，mm

K = np.array([[606.87, 0, 326.074],
              [0, 606.394, 241.158],
              [0, 0, 1]], dtype=np.float64)  # 相机内参
chess_board_x_num = 12 - 1  # 棋盘格x方向格子数-1
chess_board_y_num = 9 - 1  # 棋盘格y方向格子数-1
chess_board_len = 30  # 单位棋盘格长度,mm


# 用来从棋盘格图片得到相机外参
def get_RT_from_chessboard(img_path, chess_board_x_num, chess_board_y_num, K, chess_board_len):
    """
    :param img_path: 读取图片路径
    :param chess_board_x_num: 棋盘格x方向格子数
    :param chess_board_y_num: 棋盘格y方向格子数
    :param K: 相机内参
    :param chess_board_len: 单位棋盘格长度,mm
    :return: 相机外参
    """
    img = cv2.imread(img_path)
    # print(img_path)
    # print(os.path.exists(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    # 粗略求角点
    ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), None)
    RT = None
    if corners is not None:
        print("Good Pic: 角点可被检测")

        # 精细求角点
        corners1 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        # 画出角点
        cv2.drawChessboardCorners(img, (chess_board_x_num, chess_board_y_num), corners1, ret)
        # print(corners)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        corner_points = np.zeros((2, corners1.shape[0]), dtype=np.float64)
        for i in range(corners1.shape[0]):
            corner_points[:, i] = corners1[i, 0, :]
        # print(corner_points)
        object_points = np.zeros((3, chess_board_x_num * chess_board_y_num), dtype=np.float64)
        flag = 0
        for j in range(chess_board_y_num):
            for k in range(chess_board_x_num):
                object_points[:2, flag] = np.array([(11 - k - 1) * chess_board_len, (8 - j - 1) * chess_board_len])
                flag += 1
        # print(object_points)
        retval, rvec, tvec = cv2.solvePnP(object_points.T, corner_points.T, K, distCoeffs=None)
        RT = np.column_stack(((cv2.Rodrigues(rvec))[0], tvec))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    else:
        print("Bad")
    return RT


# TODO:选择可成功检测角点且结果准确的数据标记
good_picture = [2, 3, 4, 6, 8, 17, 25, 31, 32, 33, 34, 35, 37, 40]
file_num = len(good_picture)

# -------------------- 计算board to cam 变换矩阵 -----------------------------------
R_all_chess_to_cam_1 = []
T_all_chess_to_cam_1 = []
for i in good_picture:
    print('b2c:', i)
    # ------------ 从python中获取矩阵 -------------------------------------
    image_path = 'IMG' + '/' + str(i) + '.png'
    RT = get_RT_from_chessboard(image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)
    if RT is not None:
        print(RT)
        R_all_chess_to_cam_1.append(RT[:3, :3])
        T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3, 1)))

# -------------------- 计算end to base变换矩阵 -----------------------------------
R_all_end_to_base_1 = []
T_all_end_to_base_1 = []
for i in good_picture:
    print('e2b:', i)
    RT = np.load(f'RT/{i}.npy', allow_pickle=True)
    RT[:3, 3] = RT[:3, 3] * 1000

    # TODO: 眼在手外--需求逆   (end to base)
    #       眼在手内--无需求逆 (base to end)，将下行注释掉
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


