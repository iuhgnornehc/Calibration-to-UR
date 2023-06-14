import cv2
import numpy as np
import scipy.io as sio
import pyrealsense2 as rs

K = np.array([[606.375, 0, 327.991],
              [0, 605.522, 247.453],
              [0, 0, 1]], dtype=np.float64)  # 相机内参
chess_board_x_num = 11  # 棋盘格x方向格子数
chess_board_y_num = 8  # 棋盘格y方向格子数
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    # 粗略求角点
    ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), None)
    # 精细求角点
    corners1 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    # 画出角点
    cv2.drawChessboardCorners(img, (chess_board_x_num, chess_board_y_num), corners1, ret)
    # print(corners)
    cv2.imshow('img', img)
    cv2.waitKey(0)
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
    return RT


def get_RT_from_chessboard_matlab(index):
    rotation = sio.loadmat('./MATLAB/rotation.mat')
    rotation = rotation['rotation']
    translation = sio.loadmat('./MATLAB/translation.mat')
    translation = translation['translation']
    RT = np.column_stack((rotation[:, :, index], translation[index, :].T))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    return RT


good_picture = range(0, 20)  # 存放可以检测出棋盘格角点的图片
# good_picture = [20]
file_num = len(good_picture)

# -------------------- 计算board to cam 变换矩阵 -----------------------------------
R_all_chess_to_cam_1 = []
T_all_chess_to_cam_1 = []
for i in good_picture:
    print('b2c:', i)
    # ------------ 从python中获取矩阵 -------------------------------------
    image_path = './IMG' + '/' + str(i) + '.png'
    RT = get_RT_from_chessboard(image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)
    # RT = np.linalg.inv(RT)
    # ------------ 从matlab中获取矩阵 ---------------------------------------
    # RT = get_RT_from_chessboard_matlab(i-20)
    # RT = np.linalg.inv(RT)
    print(RT)
    R_all_chess_to_cam_1.append(RT[:3, :3])
    T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3, 1)))

# -------------------- 计算end to base变换矩阵 -----------------------------------
R_all_end_to_base_1 = []
T_all_end_to_base_1 = []
for i in good_picture:
    print('e2b:', i)
    RT = np.load(f'./RT/{i}.npy', allow_pickle=True)
    RT[:3, 3] = RT[:3, 3] * 1000
    # RT = np.linalg.inv(RT)
    print(RT)
    R_all_end_to_base_1.append(RT[:3, :3])
    T_all_end_to_base_1.append(RT[:3, 3].reshape((3, 1)))
# -------------------- 手眼标定 ------------------------------------------------
R, T = cv2.calibrateHandEye(R_all_end_to_base_1, T_all_end_to_base_1, R_all_chess_to_cam_1,
                            T_all_chess_to_cam_1)  # 手眼标定
RT = np.column_stack((R, T))
RT = np.row_stack((RT, np.array([0, 0, 0, 1])))  # 即为cam to end变换矩阵
print('相机相对于末端的变换矩阵为：')
print(RT)
np.save('RT.npy', RT)

# 结果验证，原则上来说，每次结果相差较小
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

# # 获取相机的内参矩阵
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
# cfg = pipeline.start(config)
# device = cfg.get_device()
# name = device.get_info(rs.camera_info.name)
# print(name)
# profile = cfg.get_stream(rs.stream.depth)
# profile1 = cfg.get_stream(rs.stream.color)
# intr = profile.as_video_stream_profile().get_intrinsics()
# intr1 = profile1.as_video_stream_profile().get_intrinsics()
# extrinsics = profile1.get_extrinsics_to(profile)
# print(extrinsics)
# print("深度传感器内参：", intr)
# print("RGB相机内参:", intr1)
