# coding=utf-8
import csv
import glob
import os

from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs, STEPControl_ManifoldSolidBrep
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.DataExchange import write_step_file, read_step_file_with_names_colors


def read_step_model(input_step_file):
    # 读取原始 STEP 模型
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(input_step_file)
    if status != IFSelect_RetDone:
        raise ValueError("Failed to read the input STEP file.")

    stp_name_colors = read_step_file_with_names_colors(input_step_file)
    return stp_name_colors


def write_split_step_model(stp_name_colors: dict, order_list, key, output_file_prefix):
    # 创建新的 STEP 文件并写入形状
    step_writer = STEPControl_Writer()
    index = 1
    for shp in stp_name_colors:
        shp_name, _ = stp_name_colors[shp]
        if index in order_list:
            step_writer.Transfer(shp, STEPControl_ManifoldSolidBrep)
        index += 1

    output_file_name = output_file_prefix + "-" + str(key) + ".stp"
    status = step_writer.Write(output_file_name)
    if status != IFSelect_RetDone:
        raise ValueError("Failed to write the output STEP file.")


# 读取occ id的合并文件
def read_occ_id_merge_txt(file_name: str):
    # 初始化一个空的二维列表
    data = []
    with open(file_name, 'r') as csvfile:
        # 创建CSV读取器
        csvreader = csv.reader(csvfile)

        # 逐行读取CSV文件并将每一行转换为整数列表
        for row in csvreader:
            int_row = [int(cell) for cell in row]
            data.append(int_row)

    return data


def read_file_and_convert_to_dict(filename: str):
    result_dict = {}
    with open(filename, "r") as file:
        for line in file:
            key, value_str = line.strip().split(":")
            values = [int(x) for x in value_str.strip().split(",")]
            result_dict[key] = values

    return result_dict


def split_step_model(input_step_file, occ_id_split_path, output_step_prefix):
    # occ_merge_list = read_occ_id_merge_txt(occ_id_split_path)
    occ_ids_dict = read_file_and_convert_to_dict(occ_id_split_path)

    stp_name_colors = read_step_model(input_step_file)
    # occ_ids=[1,2]
    # key="0"
    # write_step_model_to_step(stp_name_colors, occ_ids, key, output_step_prefix)
    for index, occ_ids in occ_ids_dict.items():
        write_split_step_model(stp_name_colors, occ_ids, index, output_step_prefix)

def split_one_file():
    # output_step_file = "../../data/step/X11-SE1-v2.1_1-split.step"
    # order_list = [244,245,246,247,248,249,250,251,252,1075,1076]  # 替换为你的顺序号列表
    # split_step_model(input_step_file, order_list,output_step_file)
    file_prefix = "C3-JD-27_v2.1_1"
    input_step_file = "../../data/step/" + file_prefix + ".step"
    occ_id_split_path = file_prefix + "_occ-id_split.txt"
    output_step_prefix = "../../data/step/" + file_prefix + "_split"
    split_step_model(input_step_file, occ_id_split_path, output_step_prefix)

def split_mul_file():
    # 指定包含STEP文件的文件夹路径
    folder_path = "../../data/step/X11-SE1-v2_1_1"
    # file_prefix = "X11-SE1-v2.1_1"
    folder_path = "../../data/step/61-v2_1_1"
    file_prefix = "61-v2.1_1"
    occ_id_split_path = file_prefix + "_occ-id_split.txt"

    # 列出文件夹中所有的STEP文件
    step_files = glob.glob(os.path.join(folder_path, "*.step"))

    # 循环处理每个STEP文件
    for step_file in step_files:
        # 提取文件名（不包括路径和扩展名）
        file_name = os.path.splitext(os.path.basename(step_file))[0]

        # 构建输出文件路径
        output_step_prefix = os.path.join(folder_path, f"{file_name}_split")

        # 调用 split_step_model 函数来处理每个STEP文件
        split_step_model(step_file, occ_id_split_path, output_step_prefix)

if __name__ == "__main__":
    split_mul_file()