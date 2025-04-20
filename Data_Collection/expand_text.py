import joblib
import os

motion_num_dict = joblib.load("./motion_num_dict.pkl")

# 循环读取文件夹中的每一个文件，根据motion_num_dict判断是否需要扩展

folder_path = "./select_data/text_no_expand"  # 替换为你的文件夹路径

orgin_num = 0
expand_num = 0
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    if os.path.isfile(file_path):  # 确保是文件而不是子目录
        # print(file_name)
        # 1/0
        
        motion_names = file_name.split("poses")[0] + "poses"
        
        if motion_names not in motion_num_dict.keys():
            # pass
            print(motion_names)
            print("error")
        orgin_num += 1
        expand_num += motion_num_dict[motion_names]

        with open(file_path, 'r') as f:
            origin_text = f.read()

        for i in range(motion_num_dict[motion_names]):
            if i == 0:
                new_file_path = "./text/" + file_name
            else :
                new_file_path = "./text/" + file_name.split('.')[0] + "=" + str(i) + ".txt"
            with open(new_file_path, "w") as f:
                f.write(origin_text)
            

print(orgin_num)
print(expand_num)
