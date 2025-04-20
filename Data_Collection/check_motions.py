import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import joblib
import os
import multiprocessing

from pynput import keyboard




import csv




motion_data_path = "./select_data/select_motion_dof_pos.pkl"
motion_name_path = "./select_data/select_motion_name.pkl"
motion_text_path = "./select_data/upper_motion_text.pkl"



curr_frame = 0
start_frame = 0
end_frame = 0
shutdown = False
restart = False

go_forward = False
go_backward = False

   
# 判断截取范围

# 直接将motion跳过到最后

# 快进 or 后退

def on_press(key):
    global command
    global start_frame
    global end_frame
    global shutdown
    global restart
    global go_forward
    global go_backward
    if isinstance(key, keyboard.KeyCode):
        if key.char == '2':
            print(curr_frame, "start frame")
            start_frame = curr_frame
        elif key.char == '3':
            print(curr_frame, "end frame")
            end_frame = curr_frame

    elif isinstance(key, keyboard.Key):
        if key == keyboard.Key.left:
            go_backward = True
        elif key == keyboard.Key.right:
            go_forward = True
        elif key == keyboard.Key.up:
            shutdown = True
        elif key == keyboard.Key.down:
            # restart
            restart = True
    #     elif key == keyboard.Key.right:
    #         command[1] -= 0.1   
listener = keyboard.Listener(
    on_press=on_press)
listener.start()            
def _load_motion():
    motion_path = motion_data_path
    print(f"Load motion from {motion_path}......")
    motion_data = joblib.load(motion_path) # shape: (num_motion, num_frames, num_dof)
    print(len(motion_data))
    return motion_data

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

motion_name = joblib.load(motion_name_path)

# flag_list = []
# for i in range(len(motion_name)):
#     flag_list.append({})
# joblib.dump(flag_list, "/home/bcj/new_unitree_rl_gym/resources/motions/experiment/KIT/complete_KIT_process/data_list.pkl")
# 1/0
motion_text  = joblib.load(motion_text_path)
motion_data = _load_motion()  


use_motion = True


config_file = 'h1_2_21dof_fix.yaml'
with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    
    control_decimation = 5

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    
    cmd = np.array(config["cmd_init"], dtype=np.float32)

# define context variables
action = np.zeros(num_actions, dtype=np.float32)
target_dof_pos = default_angles.copy()
obs = np.zeros(num_obs, dtype=np.float32)

counter = 0

# Load robot model
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = simulation_dt

# load policy
policy = torch.jit.load(policy_path)
# frame_num = 0

small_data = 0 # 15s 668个






def write_specific_row(csv_file, row_index, new_data):
    """
    修改 CSV 文件的指定行
    
    参数:
        csv_file (str): CSV 文件路径
        row_index (int): 要修改的行号（0-based，如第 5 行 → row_index=4）
        new_data (list): 新的行数据（如 ["A1", "B1", "C1"]）
    """
    # 读取整个 CSV
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        # print(rows)
    
    # 检查行号是否有效
    if row_index >= len(rows):
        raise IndexError(f"行号超出范围（最大行数：{len(rows)-1}）")
    
    # 修改指定行
    rows[row_index] = new_data
    
    # 写回文件
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerows(rows)




with open("/home/bcj/new_unitree_rl_gym/resources/motions/experiment/KIT/complete_KIT_process/mujoco/frame_text.csv", "r", encoding="utf-8") as file:
    csv_data = list(csv.reader(file))  # 转为列表



with mujoco.viewer.launch_passive(m, d) as viewer:

    i = -1
    while True:
        
        i += 1
        
        if i >= len(motion_data):
            break
        # 如果当前行非空，跳过本次循环
        if any(csv_data[i+1]):  # 检查行是否有非空值
            continue
        # print(i)
        # continue
        
        # for i in range(len(motion_name)):
        current_motion = motion_data[i]


        # 如果这个motion已经处理过了，直接跳过
        # TODO

        mujoco.mj_resetData(m, d)
        curr_frame = 0
        start_frame = 0
        end_frame = len(current_motion)
        shutdown = False

        counter = 0
        target_dof_pos = default_angles.copy()
        print("-------start motion-------", str(i), motion_name[i])
        print(len(current_motion), len(current_motion) * 0.02)



        time.sleep(1)
        while counter < (len(current_motion) * control_decimation):
            if shutdown:
                shutdown = False

                break
            if restart:
                restart = False

                counter = 0
                curr_frame = 0
                start_frame = 0
                end_frame = len(current_motion)  
                target_dof_pos = default_angles.copy()
                continue    
            if go_forward:
                go_forward = False

                counter += 100
                curr_frame += 20
                continue

            if go_backward:
                go_backward = False

                counter -= 100
                counter = max(counter, 0)
                curr_frame -= 20
                curr_frame = max(curr_frame, 0)
                continue

            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            if counter % control_decimation == 0:
                curr_frame += 1
                # Apply control signal here.
                # print(counter//10)
                cur_pos = current_motion[counter//control_decimation][0]
                indices = [5, 9]
                cur_pos = np.insert(cur_pos, indices, 0)
                # print(cur_pos)


                target_dof_pos = cur_pos
                
                target_dof_pos[:13] = default_angles[:13]


                # 特权信息

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            counter += 1


            # Rudimentary time keeping, will drift relative to wall clock.
            # time_until_next_step = m.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)
        # 输入text      
        
        new_text = input("Input text: ")
        if new_text == "":
            new_text = motion_text[i]
        # 如果输入no的话，后面就抛弃这个motion
        print("start frame: ",start_frame, " end frame: ", end_frame, "text: ", new_text)
        # 用&分割不同的文本
        # 
        
    
        write_specific_row("/home/bcj/new_unitree_rl_gym/resources/motions/experiment/KIT/complete_KIT_process/mujoco/frame_text.csv", i+1, [start_frame, end_frame, new_text])
        
        # 写一个save一次 TODO
        # save到一个表格里吧
        
        # 加一个提前终止 并且去掉这个motion
        # break
        time.sleep(1)

print(small_data)