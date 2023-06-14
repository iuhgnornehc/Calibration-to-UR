import urx
import time
import numpy as np
import sys
from NatNetClient import NatNetClient
from scipy.spatial.transform import Rotation as R

def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len=len(arg_list)
    if arg_list_len>1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len>2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len>3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict

def receive_new_frame(data_dict):
    order_list=[ "frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
                "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged" ]
    dump_args = False
    if dump_args == True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "="
            if key in data_dict :
                out_string += data_dict[key] + " "
            out_string+="/"
        print(out_string)

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame( new_id, position, rotation ):
    # pass
    # print( "Received frame for rigid body", new_id )
    pos = None
    rot = None
    flag = False
    if new_id == '2':
        pos = position
        rot = rotation
        flag = True
    print( "Received frame for rigid body", new_id," ",position," ",rotation )
    return pos, rot, flag


if __name__ == '__main__':

    pos = []
    rot = []
    flag = False  # 是否成功接收到刚体位姿

    optionsDict = {}

    # TODO: Motive IP
    optionsDict["clientAddress"] = "127.0.0.1"
    optionsDict["serverAddress"] = "127.0.0.1"
    optionsDict["use_multicast"] = True

    # This will create a new NatNet client
    optionsDict = my_parse_args(sys.argv, optionsDict)

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.new_frame_listener = receive_new_frame
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run()
    time.sleep(3)

    # TODO： rigid body 编号
    rigid = streaming_client.get_rigid_body(2)

    if rigid is not None:
        pos = np.array(rigid.pos).reshape((3, 1))
        rot = rigid.rot
        streaming_client.shutdown()

        r = R.from_quat(rot)
        R_Motive_to_board = r.as_matrix()
        RT_Motive_to_board = np.column_stack((R_Motive_to_board, pos))
        RT_Motive_to_board = np.row_stack((RT_Motive_to_board, np.array([0, 0, 0, 1])))

        print(RT_Motive_to_board)

        # TODO：机械臂ip
        robot = urx.Robot("192.168.1.121")
        time.sleep(2)


        # TODO:数据标记编号
        i = 11

        trans = robot.get_pose()
        # position = trans.pos.array
        # orientation = trans.orient.array
        RT_Base_to_End = np.array(trans.matrix)

        # print("End:\n", robot.getl())

        print("RT_Motive_to_board: \n", RT_Motive_to_board)
        np.save(f"./RT_Motive_to_board/{i}.npy", RT_Motive_to_board)
        print("RT_Base_to_End: \n", RT_Base_to_End)
        np.save(f"./RT_Base_to_End/{i}.npy", RT_Base_to_End)

        # print(TR)








