# KIT中筛选出上半身动作


import joblib
import torch
import numpy as np

# 从motion中读取真值
motion_name = joblib.load('./origin_data/motion_name.pkl')
motion_dof_pos = joblib.load('./origin_data/motion_dof_pos.pkl')
motion_rg_pos = joblib.load('./origin_data/motion_rg_pos.pkl')
motion_root_rot = joblib.load('./origin_data/motion_root_rot.pkl')
motion_root_vel = joblib.load('./origin_data/motion_root_vel.pkl')
motion_root_ang_vel = joblib.load('./origin_data/motion_root_ang_vel.pkl')

# 总计 3232

# 去掉一些下肢动作 和 上肢无意义动作 剩余
lower_motion_keywords = ["run", "running", 'walk', "walking","turn", "recoverystepping", "displace", "kick", "push", "go_over", "jump", "step", "parkour", "stomp", "seesaw", "balancing", "bend", "Drehung", 'blance', "dance","flexion", "evasion", "squat", "bow_", "flexion", "912_912_3","Dreischritte02", "synchron", "rotation", "MarcusS", "3_912_3", "conversation", "supination"]
# 835
# upper_motion_keywords = ["wave", "push", "hand", "arm", "drinking"]
upper_motion_name_idx = [
    (i, name)
    for i, name in enumerate(motion_name) 
    if not any(keyword in name.lower() for keyword in lower_motion_keywords) 
    # and any(keyword in name.lower() for keyword in upper_motion_keywords)
]

print((upper_motion_name_idx))
print(len(upper_motion_name_idx))


save_motion_name = [motion_name[i] for i, name in upper_motion_name_idx]
save_motion_dof_pos = [motion_dof_pos[i] for i, name in upper_motion_name_idx]
save_motion_rg_pos = [motion_rg_pos[i] for i, name in upper_motion_name_idx]
save_motion_root_rot = [motion_root_rot[i] for i, name in upper_motion_name_idx]
save_motion_root_vel = [motion_root_vel[i] for i, name in upper_motion_name_idx]
save_motion_root_ang_vel = [motion_root_ang_vel[i] for i, name in upper_motion_name_idx]

joblib.dump(save_motion_name, './select_data/select_motion_name.pkl')
joblib.dump(save_motion_dof_pos, './select_data/select_motion_dof_pos.pkl')
joblib.dump(save_motion_rg_pos, './select_data/select_motion_rg_pos.pkl')
joblib.dump(save_motion_root_rot, './select_data/select_motion_root_rot.pkl')
joblib.dump(save_motion_root_vel, './select_data/select_motion_root_vel.pkl')
joblib.dump(save_motion_root_ang_vel, './select_data/select_motion_root_ang_vel.pkl')


import re
def clean_motion_name(name):
    # 去掉前缀和后缀
    cleaned = re.sub(r'^\d+-KIT_\d+_|(_?\d+_poses)$', '', name)
    # 替换内部的下划线为空格
    return cleaned.replace("_", " ")

cleaned_names = [clean_motion_name(name) for i, name in upper_motion_name_idx]
print((cleaned_names))
print(len(cleaned_names))

joblib.dump(cleaned_names, './select_data/upper_motion_text.pkl')





