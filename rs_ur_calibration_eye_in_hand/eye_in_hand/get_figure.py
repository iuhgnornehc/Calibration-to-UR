import cv2
import urx
import time
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()  # 构建一个抽象设备的管道
config = rs.config()  # 使用非默认配置文件创建配置以配置管道
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)   # 表示构建的管道使用上述配置开始流传输

# 将深度图对齐到RGB
align_to = rs.stream.color
align = rs.align(align_to)
print(f"align:{align}")
robot = urx.Robot("192.168.1.100")
time.sleep(2)


i = 13
frame = pipeline.wait_for_frames()  # 获取一帧数据
data_sz = frame.get_data_size()  # 获取图像数据

color_rs = frame.get_color_frame()
img = np.asanyarray(color_rs.get_data())
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(f'./IMG/{i}.png', img)
trans = robot.get_pose()
position = trans.pos.array
orientation = trans.orient.array
TR = np.array(trans.matrix)

np.save(f"./POSITION/{i}.npy", position)
np.save(f"./ORIENTATION/{i}.npy", orientation)
np.save(f"./RT/{i}.npy", TR)

print(TR)








