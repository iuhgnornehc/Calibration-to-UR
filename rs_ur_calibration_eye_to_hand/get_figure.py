import cv2
import urx
import time
import pyrealsense2 as rs
import numpy as np

if __name__ == '__main__':
    pipeline = rs.pipeline()  # 构建一个抽象设备的管道
    config = rs.config()  # 使用非默认配置文件创建配置以配置管道

    # TODO：input：stream通道类型, width, height, 编码类型, 帧率
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    profile = pipeline.start(config)   # 表示构建的管道使用上述配置开始流传输

    # 将深度图对齐到RGB
    align_to = rs.stream.color
    align = rs.align(align_to)
    print(f"align:{align}")
    # TODO：机械臂ip
    robot = urx.Robot("192.168.1.121")
    time.sleep(2)

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays

        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        # images = depth_colormap
        # Show images
        cv2.namedWindow('RealSense-realtime', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

    # TODO:数据标记编号
    i = 1
    frame = pipeline.wait_for_frames()  # 获取一帧数据
    data_sz = frame.get_data_size()  # 获取图像数据

    color_rs = frame.get_color_frame()
    img = np.asanyarray(color_rs.get_data())

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.imwrite(f'./IMG/{i}.png', img)
    trans = robot.get_pose()
    position = trans.pos.array
    orientation = trans.orient.array
    TR = np.array(trans.matrix)

    # print("End:\n", robot.getl())

    print("position: \n", position)
    np.save(f"./POSITION/{i}.npy", position)
    print("orientation: \n", orientation)
    np.save(f"./ORIENTATION/{i}.npy", orientation)
    print("RT: \n", TR)
    np.save(f"./RT/{i}.npy", TR)

    # print(TR)








