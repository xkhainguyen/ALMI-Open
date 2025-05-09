# KIT all upper motion

import joblib
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")
def convert_sentence(sentence):
    """
    将输入句子转换为指定格式：
    1. 保持原始句子
    2. 生成词性标注版本
    3. 添加固定的 '#0.0#0.0'
    """
    doc = nlp(sentence)
    
    # 生成带有词性标注的文本
    tagged_text = " ".join(f"{token.text}/{token.pos_}" for token in doc)
    
    # 组合最终格式
    formatted_output = f"{sentence}#{tagged_text}#0.0#0.0"
    
    return formatted_output

motion_name = joblib.load('./select_data/select_motion_name.pkl')
print(len(motion_name))
upper_text = joblib.load('./select_data/upper_motion_text.pkl')

text_list = []
cmd_list = []

lin_vel_x_range = [
    [0.2, 0.4],
    [0.4, 0.5],
    [0.5, 0.7],
]
linx_vel_y_range = [
    [0.2, 0.3],
    [0.3, 0.4],
    [0.4, 0.5],
]
ang_vel_yaw_range = [
    [0.2, 0.3],
    [0.3, 0.4],
    [0.4, 0.5],
]

fixed_vel = 0.4
fixed_ang = 0.4

# lin_vel_text[x][y] x: 0 1 2 y: 0 1 2
# x 1 前
# y 1 左
lin_vel_text = [
    ["keep standing", "go left", "go right"],
    ["go forward", "go forward to the left", "go forward to the right"],
    ["go backward", "go backward to the left", "go backward to the right"], 
]
lin_vel_speed_text = ["slowly", "moderately", "fast", ""]
ang_vel_text = ["keep standing", "turn left", "turn right"]
idx = 0

npy_name_list = []

for i in range(len(motion_name)):
# for i in range(1):
    # 8 * 3 + 1
    idx = 0
    for j in range(3):
        for k in range(3):                          
            lin_idx = [0, 1, -1]
            lin_vel_x_choose = lin_idx[j]
            lin_vel_y_choose = lin_idx[k]

            for l in range(3):
                if (j == 0 and k == 0) and l != 0:
                    continue
                lin_vel_x_flag = l
                lin_vel_y_flag = l
                lin_vel_x_val =  lin_vel_x_choose * np.random.uniform(lin_vel_x_range[lin_vel_x_flag][0], lin_vel_x_range[lin_vel_x_flag][1])
                lin_vel_y_val =  lin_vel_y_choose * np.random.uniform(linx_vel_y_range[lin_vel_y_flag][0], linx_vel_y_range[lin_vel_y_flag][1])

                cmd_list.append([float(lin_vel_x_val), float(lin_vel_y_val), 0])
    
                text_lower_direction = lin_vel_text[lin_vel_x_choose if lin_vel_x_choose != -1 else 2][lin_vel_y_choose if lin_vel_y_choose != -1 else 2]
                text_lower_speed = lin_vel_speed_text[max(lin_vel_x_flag, lin_vel_y_flag)] # 现在直接设置成二者相等了

                text_lower = text_lower_direction +  " " + text_lower_speed
                if j == 0 and k == 0:
                    text_lower_speed = ""
                    text_lower = text_lower_direction

                text_upper = upper_text[i]
                text_complete = "Robot " + text_lower + " and " + text_upper + "."
                text_format = convert_sentence(text_complete)
                text_list.append(text_format)
                tpath = "./text_no_expand/" + motion_name[i] + "-" + text_lower_direction.replace(" ","_")+ "-" + (text_lower_speed if text_lower_speed != "" else "zero") + "_" + str(idx)
                path = tpath + ".txt"
                with open(path, "w") as f:
                    f.write(text_format)
                    
                    npy_name_list.append(motion_name[i] + "-" + text_lower_direction.replace(" ","_")+ "-" + (text_lower_speed if text_lower_speed != "" else "zero") + "_" + str(idx) + ".npy")
                    idx += 1
                    # print(path)
                    # print(lin_vel_x_val, lin_vel_y_val,0)  
                # print(text_lower)
    # 8 * 1 每个方向固定速度
    for j in range(3):
        for k in range(3):
            if j == 0 and k == 0:
                continue
            lin_idx = [0, 1, -1]
            # 选择方向
            lin_vel_x_choose = lin_idx[j]
            lin_vel_y_choose = lin_idx[k]


            lin_vel_x_val =  lin_vel_x_choose * fixed_vel
            lin_vel_y_val =  lin_vel_y_choose * fixed_vel

            cmd_list.append([float(lin_vel_x_val), float(lin_vel_y_val), 0])

            text_lower_direction = lin_vel_text[lin_vel_x_choose if lin_vel_x_choose != -1 else 2][lin_vel_y_choose if lin_vel_y_choose != -1 else 2]
            text_lower = text_lower_direction

            text_upper = upper_text[i]
            text_complete = "Robot " + text_lower + " and " + text_upper + "."
            text_format = convert_sentence(text_complete)
            text_list.append(text_format)
            tpath = "./text_no_expand/" + motion_name[i] + "-" + text_lower_direction.replace(" ","_")+ "-" + "fixed"  + "_" + str(idx)
            path = tpath + ".txt"
            with open(path, "w") as f:
                f.write(text_format)  
                
                # print(path)
                # print(lin_vel_x_val, lin_vel_y_val,0)  
                npy_name_list.append(motion_name[i] + "-" + text_lower_direction.replace(" ","_")+ "-" + "fixed"  + "_" + str(idx) + ".npy")
                idx +=1
            # print(text_lower)
    
    # 2 * 3
    for j in range(3):
        for k in range(3):
            if j == 0:
                continue

            ang_idx = [0, 1, -1]
            ang_vel_yaw_choose = ang_idx[j]
            ang_vel_yaw_flag = k
            ang_vel_yaw_val =  ang_vel_yaw_choose * np.random.uniform(ang_vel_yaw_range[ang_vel_yaw_flag][0], ang_vel_yaw_range[ang_vel_yaw_flag][1])
            
            cmd_list.append([0, 0, float(ang_vel_yaw_val)])
            
            text_lower_direction = ang_vel_text[ang_vel_yaw_choose if ang_vel_yaw_choose != -1 else 2]
            text_lower_speed = lin_vel_speed_text[ang_vel_yaw_flag]
            text_lower = text_lower_direction +  " " + text_lower_speed
            text_upper = upper_text[i]
            text_complete = "Robot " + text_lower + " and " + text_upper + "."
            text_format = convert_sentence(text_complete)
            text_list.append(text_format)
            tpath = "./text_no_expand/" + motion_name[i] + "-" + text_lower_direction.replace(" ","_")+ "-" + (text_lower_speed if text_lower_speed != "" else "zero")  + "_" + str(idx)
            path = tpath + ".txt"
            with open(path, "w") as f:
                f.write(text_format)
                
                npy_name_list.append(motion_name[i] + "-" + text_lower_direction.replace(" ","_")+ "-" + (text_lower_speed if text_lower_speed != "" else "zero")  + "_" + str(idx) + ".npy")
                idx += 1
                # print(path)
                # print(0, 0,ang_vel_yaw_val)  
            # print(text_lower)

    for j in range(3):
        if j == 0:
            continue

        ang_idx = [0, 1, -1]
        ang_vel_yaw_choose = ang_idx[j]
        ang_vel_yaw_val =  ang_vel_yaw_choose * fixed_ang
        
        cmd_list.append([0, 0, float(ang_vel_yaw_val)])
        
        text_lower_direction = ang_vel_text[ang_vel_yaw_choose if ang_vel_yaw_choose != -1 else 2]
        # text_lower_speed = lin_vel_speed_text[ang_vel_yaw_flag]
        text_lower = text_lower_direction #  +  " " + text_lower_speed
        text_upper = upper_text[i]
        text_complete = "Robot " + text_lower + " and " + text_upper + "."
        text_format = convert_sentence(text_complete)
        text_list.append(text_format)
        tpath = "./text_no_expand/" + motion_name[i] + "-" + text_lower_direction.replace(" ","_")+ "-" + "fixed"  + "_" + str(idx)
        path = tpath +  ".txt"
        with open(path, "w") as f:
            f.write(text_format)
        
            npy_name_list.append(motion_name[i] + "-" + text_lower_direction.replace(" ","_")+ "-" + "fixed"  + "_" + str(idx)+ ".npy")
            idx += 1
        #     # print(path)
        #     # print(0, 0,ang_vel_yaw_val)  
        # print(text_lower)

# # 保存结果
# print(idx)
# # for i in range(800):
# # print(cmd_list)
# # print(text_list)
joblib.dump(cmd_list, "./cmd/cmd_list.pkl")
joblib.dump(npy_name_list, "./npy_name_list.pkl")