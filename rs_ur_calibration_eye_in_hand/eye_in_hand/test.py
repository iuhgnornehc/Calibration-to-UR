import cv2
import numpy as np
import urx

a = (2.4805, -1.2941, -0.0208)
print(a)
R, j = cv2.Rodrigues(a)
print(R)
a, j = cv2.Rodrigues(R)
print(a)

#
# RT_b2e = np.load('INIT/rt.npy', allow_pickle=True)  # 基座到执行器的转换矩阵
# RT_b2e[0:3, 3] = RT_b2e[0:3, 3] * 1000
# RT_e2c = np.load('RT.npy', allow_pickle=True)
# RT_b2c = RT_b2e.dot(RT_e2c)
# print(RT_b2c)
#
robot = urx.Robot("192.168.1.110")
trans = robot.get_pose()
RT = np.array(trans.matrix)
print(trans)
# K = np.array([[606.375, 0, 327.991],
#               [0, 605.522, 247.453],
#               [0, 0, 1]], dtype=np.float64)  # 相机内参
#
# print(K)
# 图片路径
# img = cv2.imread('./IMG/21.png')
# a = []
# b = []
#
#
# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         a.append(x)
#         b.append(y)
#         cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0, 0, 0), thickness=1)
#         cv2.imshow("image", img)
#         print(x, y)
#
#
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# cv2.imshow("image", img)
# cv2.waitKey(0)
# print(a[0], b[0])
