


import time

import mujoco.viewer
import mujoco
import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import joblib
import os
import multiprocessing
# # import pygame
# mujoco.GLContext.use_egl()
import gc

# Force garbage collection


import csv
def read_frame_num(file_path):
    start_frame = []
    end_frame = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            start_frame.append(row[0])
            end_frame.append(row[1])
    return start_frame, end_frame

csv_path = './mujoco/frame_text.csv'
start_frames, end_frames = read_frame_num(csv_path)
start_frames = [int(x) for x in start_frames]
end_frames = [int(x) for x in end_frames]

# # 初始化 pygame
# pygame.init()
# # 速度控制变量
# velocity = 0.0
# velocity_step = 0.1  # 速度增量

# def update_velocity():
#     global velocity
#     keys = pygame.key.get_pressed()
#     if keys[pygame.K_UP]:  # 按上方向键加速
#         velocity += velocity_step
#     elif keys[pygame.K_DOWN]:  # 按下方向键减速
#         velocity -= velocity_step
#     elif keys[pygame.K_SPACE]:  # 按空格键停止
#         velocity = 0.0
            
def _load_motion():
    motion_path = "./select_data/select_motion_dof_pos.pkl"
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

motion_name = joblib.load("./select_data/select_motion_name.pkl")
cmd_data = joblib.load("./select_data/cmd/cmd_list.pkl")
npy_name_list = joblib.load("./select_data/npy_name_list.pkl")
robot_nums = 10
motion_data = _load_motion()  


TOTAL_NUMS = 34235
# 还是这么多motion 但是每个motion可能需要重复采集一下


motion_num_dict = joblib.load("./motion_num_dict.pkl")
# 文件名



def run_one_robot(robot_idx):
    add_history = True

    use_motion = True

    
    print(robot_idx, robot_idx * TOTAL_NUMS//robot_nums ,(robot_idx + 1) * TOTAL_NUMS//robot_nums)


    config_file = 'h1_2_21dof.yaml'
    with open(f"./select_data/mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        # xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        policy_path = "./mujoco/policy_lstm_12800.pt"
        xml_path = "./mujoco/robots/h1_2/scene_21dof.xml"


# xml_path: "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/meshes/scene_21dof.xml"

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

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
    
    # with mujoco.viewer.launch_passive(m, d) as viewer:
        # viewer.callbacks.keyboard = keyboard_callback  # 绑定键盘事件
        # viewer.add_key_callback(ord("w"). keyboard_callback)
        # Close the viewer automatically after simulation_duration wall-seconds.
    # start = time.time()
    # while viewer.is_running() and time.time() - start < simulation_duration:
    while True:

        start = robot_idx * TOTAL_NUMS//robot_nums
        end = (robot_idx + 1) * TOTAL_NUMS//robot_nums
        if robot_idx == robot_nums - 1:
            end = TOTAL_NUMS
        current_motion = motion_data[start//41]
        current_start_frame = start_frames[start//41]
        current_end_frame = end_frames[start//41]

        current_motion = current_motion[current_start_frame:current_end_frame]
    
        
        # 这里开始循环每一个motion，然后需要在这里判断这个motion需要重复多少次
        # idd = 0
        # for i in range(len(npy_name_list)):
        #     if npy_name_list[i] == "0-KIT_572_violin_left02_poses-go_backward_to_the_right-fast_24.npy":
        #         idd = i
        #         break
        
        # for i in range(idd, idd+1):
        for i in range(start, end):
            # 这里感觉会有问题 -- 如果中断之后，可能会让没跑完的多次不会继续跑
            current_motion_file = npy_name_list[i]




            # 如果已经处理过，就跳过
            
            if i % 41 == 0:
                current_motion = motion_data[i//41]
                current_motion_name = motion_name[i//41]
                current_start_frame = start_frames[i//41]
                current_end_frame = end_frames[i//41]
                current_motion = current_motion[current_start_frame:current_end_frame]
                
            current_cmd = cmd_data[i]
            
            # 处理每一个motion and cmd
            action = np.zeros(num_actions, dtype=np.float32)
            target_dof_pos = default_angles.copy()
            obs = np.zeros(num_obs, dtype=np.float32)

            counter = 0
            target_dof_pos = default_angles.copy()
            print("-------start motion-------", str(i))
            
            

            # obs包括 原来的基础上 cmd + 9；action + 9 65 + 18 = 83 - 12 = 71
            # 去掉cmd - 3 - 9
            npy_obs = np.zeros((len(current_motion), 71), dtype=np.float32)
            npy_actions = np.zeros((len(current_motion), 21), dtype=np.float32)
            npy_dof_pos = np.zeros((len(current_motion), 21), dtype=np.float32)
            npy_root_trans = np.zeros((len(current_motion), 3), dtype=np.float32)
            npy_root_rot = np.zeros((len(current_motion), 4), dtype=np.float32)
            
            print(len(current_motion))
            print(current_motion_file)
            print(current_cmd)
            print("current_motion frame len: ",current_start_frame,current_end_frame, len(current_motion))   
            
            mujoco.mj_resetData(m, d)


            # print(current_motion_file)
            # 1/0
            repeat_num = motion_num_dict[current_motion_file.split("_poses")[0] + "_poses"]
            print("repeat time", repeat_num)
            print()
            # Total_datas += repeat_num
            # Total_motions += 1 

            for r in range(repeat_num):
########
                test_file_path = "./obs/" + current_motion_file
                if r != 0:
                    test_file_path = "./obs/" + current_motion_file.split(".")[0]+"="+str(r)+".npy"
                if os.path.exists(test_file_path):
                    print("already processed",current_motion_file)
                    continue  
###############
                # 把判断的代码移动到这里！！！！！！！！！

                # test_file_path = "/home/lxz/Desktop/Motion_generator/complete_KIT_process/new_obs/" + current_motion_file
                # if r != 0:
                #     test_file_path = "/home/lxz/Desktop/Motion_generator/complete_KIT_process/new_obs/" + current_motion_file.split(".")[0]+"="+str(r)+".npy"
                
                # if test_file_path != "/home/lxz/Desktop/Motion_generator/complete_KIT_process/new_obs/0-KIT_572_violin_left02_poses-go_backward_to_the_right-fast_24.npy":
                
                #     if os.path.exists(test_file_path):
                #         print("already processed",current_motion_file)
                #         continue  



                # npy_dict = np.zeros(1)
                # if r == 0:
                #     np.save("/home/lxz/Desktop/Motion_generator/complete_KIT_process/new_obs/" + current_motion_file, npy_dict)
                # else:
                #     np.save("/home/lxz/Desktop/Motion_generator/complete_KIT_process/new_obs/" + current_motion_file.split(".")[0]+"--"+str(r)+".npy", npy_dict)
                # continue
                # 对于同一个motion， 要采集r次（如果是no就不采集了 这个是0）
                


                for j in range(len(current_motion) * 10):

                    step_start = time.time()
                    tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                    d.ctrl[:] = tau
                    # mj_step can be replaced with code that also evaluates
                    # a policy and applies a control signal before stepping the physics.
                    mujoco.mj_step(m, d)

                    counter += 1
                    if counter % control_decimation == 0:
                        # Apply control signal here.

                        # create observation
                        qj = d.qpos[7:]  # 21
                        dqj = d.qvel[6:] # 21
                        quat = d.qpos[3:7] 
                        omega = d.qvel[3:6]

                        qj = (qj - default_angles) * dof_pos_scale
                        dqj = dqj * dof_vel_scale
                        gravity_orientation = get_gravity_orientation(quat)
                        omega = omega * ang_vel_scale

                        period = 0.8
                        
                        count = counter * simulation_dt

                        cmd = current_cmd

                        if np.linalg.norm(cmd) < 0.1:
                            phase = 0
                            offset = 0
                        else:
                            phase = count % period / period
                            offset = 0.5
                        phase_left = phase
                        phase_right = (phase + offset) % 1
                        left_sin_phase = np.sin(2 * np.pi * phase_left)
                        right_sin_phase = np.sin(2 * np.pi * phase_right)

                        obs[:3] = omega
                        obs[3:6] = gravity_orientation
                        
                        # cmd[0] += 0.005
                        
                        
                        obs[6:9] = cmd * cmd_scale
                        obs[9:30] = qj
                        obs[30:51] = dqj
                        obs[51:63] = action
                        obs[63:65] = np.array([left_sin_phase, right_sin_phase])
                        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                        action = policy(obs_tensor).detach().numpy().squeeze() # actor(obs_tensor).detach().numpy().squeeze()
                        # current_motion[j//10][0][12] = 0
                        upper_action = ( current_motion[j//10][0][-9:] - default_angles[-9:])/ action_scale
                        upper_action[0] = 0
                        # frame_num += 1
                        if not use_motion:
                            whole_action =  np.concatenate([action, np.zeros(9)])
                        else:
                            whole_action =  np.concatenate([action, upper_action])
                        # transform action to target_dof_pos
                        target_dof_pos = whole_action * action_scale + default_angles


                        # print(j//10, npy_obs.shape, current_motion[j//10][0][-9:], )
                        # 更新npy
                        npy_obs[j//10] = np.concatenate([obs[:6], obs[9:63], upper_action, obs[63:]]).astype(np.float32)
                        npy_actions[j//10] = whole_action.astype(np.float32)
                        npy_dof_pos[j//10] = qj.astype(np.float32)
                        npy_root_trans[j//10] = d.qpos[0:3].astype(np.float32)
                        npy_root_rot[j//10] = d.qpos[3:7].astype(np.float32)


                        # 特权信息

                    # Pick up changes to the physics state, apply perturbations, update options from GUI.
                    # viewer.sync()

                    # Rudimentary time keeping, will drift relative to wall clock.
                    time_until_next_step = m.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        # print("sleep")
                        time.sleep(time_until_next_step)

                # 保存一个npy
                npy_dict = {
                    "obs": npy_obs,
                    "actions": npy_actions,
                    "dof_pos": npy_dof_pos,
                    "root_trans": npy_root_trans,
                    "root_rot": npy_root_rot
                }
                if r == 0:
                    np.save("./obs/" + current_motion_file, npy_dict)
                else:
                    np.save("./obs/" + current_motion_file.split(".")[0]+"="+str(r)+".npy", npy_dict)


            gc.collect()
        # print("*************************",Total_datas )
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!",Total_motions )
        break






processes = []
for i in range(robot_nums):
    p = multiprocessing.Process(target=run_one_robot, args=(i,))
    processes.append(p)
    p.start()

for process in processes:
    process.join()

# 8877
# +9368
# +9818
# +7603
# +6395
# +8242
# +8611
# +8141
# +6846
# +7648


# 0-KIT_572_violin_right09_poses-go_backward_to_the_right-fast_24.npy
# [-0.657467488804619, -0.4919139694383745, 0]
# current_motion frame len:  100 350 0
# -------start motion------- 17942