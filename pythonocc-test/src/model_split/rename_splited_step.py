# coding=utf-8

# 这个文件的作用：
# 整个节段模型拆分成独立板单元模型，但板单元类别还是需要和模型对应起来。
# 之前零件聚集成板单元时，随机形成的顺序号，并没有和类别关联。
# 先通过人工查看拆分后的独立板单元模型，来判断属于什么类别，类别用整数来表示。
# 然后将板单元的序号与类别编号对应起来
# 这个文件就是按照对应表，将板单元模型名称改为类别名称，这样可以通过名字就知道类别，便于后面训练测试

import json
import os
import re

# 指定要重命名文件的文件夹路径
# folder_name = "C3-JD-27_v2_1_1-split"
# folder_name = "61-v2_1_1-split"
folder_name = "X11-SE1-v2_1_1-split"
folder_path = "../../data/step/" + folder_name


def rename_split_step():
    # 读取JSON文件并解析为字典
    json_file_path = "class_occ-id_map.json"
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    map_data = data[folder_name]

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 去除文件名的 ".step" 扩展名
        name_without_extension, extension = os.path.splitext(filename)
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否以字典的某个键结尾
        for key, value in map_data.items():
            if name_without_extension.endswith('-'+key):
                # 构建新文件名，将字典的值添加到文件名前面
                new_filename = f"{value}_{filename}"

                # 构建新文件的完整路径
                new_file_path = os.path.join(folder_path, new_filename)

                # 重命名文件
                os.rename(file_path, new_file_path)
                break

def remove_str_prefix(input_str:str, sign:str):
    sign_index = input_str.find(sign)

    if sign_index!=-1:
        result = input_str[sign_index+1:]
        return result
    else:
        return input_str

def remove_prefix():
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} does not exist.")
        return

    for root, _, files in os.walk(folder_path):
        for filename in files:
            original_path = os.path.join(root, filename)
            new_filename = remove_str_prefix(filename,'_')
            if filename != new_filename:
                new_path = os.path.join(root, new_filename)
                os.rename(original_path, new_path)
                print(f"Renamed: {original_path} -> {new_path}")

if __name__ == "__main__":
    rename_split_step()
    # remove_prefix()
