#!/bin/python
import csv
import os.path
import re

import ifcopenshell as ifc
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename, askopenfilenames

import parseIfc


# -------Functions-----------------
def OpenFile(_extension, _filetypes):
    filename = askopenfilename(defaultextension=_extension, filetypes=[_filetypes])
    print(filename)
    return filename


def SaveFileAs(_extension):
    filename = asksaveasfilename(defaultextension=_extension)
    print(filename)
    return filename


def get_attr_of_pset(_id, ifc_file):
    """ Get all attributes of an instance by given Id
        param _id: id of instance
        return: dict of dicts of attributes
    """
    dict_psets = {}

    try:
        defined_by_type = [x.RelatingType for x in ifc_file[_id].IsDefinedBy if x.is_a("IfcRelDefinesByType")]
        defined_by_properties = [x.RelatingPropertyDefinition for x in ifc_file[_id].IsDefinedBy if
                                 x.is_a("IfcRelDefinesByProperties")]
    except:
        dict_psets.update({ifc_file[_id].GlobalId: "No Attributes found"})
    else:
        for x in defined_by_type:
            if x.HasPropertySets:
                for y in x.HasPropertySets:
                    for z in y.HasProperties:
                        dict_psets.update({z.Name: z.NominalValue.wrappedValue})

        for x in defined_by_properties:
            if x.is_a("IfcPropertySet"):
                for y in x.HasProperties:
                    if y.is_a("IfcPropertySingleValue"):
                        dict_psets.update({y.Name: y.NominalValue.wrappedValue})
                    # this could be usefull for multilayered walls in Allplan
                    if y.is_a("IfcComplexProperty"):
                        for z in y.HasProperties:
                            dict_psets.update({z.Name: z.NominalValue.wrappedValue})
            if x.is_a("IfcElementQuantity"):
                for y in x.Quantities:
                    dict_psets.update({y[0]: y[3]})

    finally:
        dict_psets.update({"IfcGlobalId": ifc_file[_id].GlobalId})

        return dict_psets


def get_structural_storey(_id, ifc_file):
    """ Get structural (IfcBuilgingStorey) information of an instance by given Id
        param _id: id of instance
        return: dict of attributes
    """
    dict_structural = {}
    instance = ifc_file[_id]
    try:
        structure = instance.ContainedInStructure
        storey = structure[0].RelatingStructure.Name

    except:
        dict_structural.update({"Storey": "No Information found"})

    else:
        dict_structural.update({"Storey": storey})
    finally:
        return dict_structural


def test1():
    # ifc_file_path = OpenFile(".ifc", ("IFC-Files", "*.ifc"))
    ifc_file_path = "./segment.ifc"

    ifc_file = ifc.open(ifc_file_path)

    # Change IfcBuildingElement to IfcWall  if you are only interested in walls for example
    filterType = "IfcBuildingElement"
    # filterType = "IfcProduct"
    instances = ifc_file.by_type(filterType)

    excel_list = []
    project = ifc_file.by_type("IfcProject")[0].Name

    for inst in instances:
        # info_structural = get_structural_storey(inst.id())
        info_pset = get_attr_of_pset(inst.id())
        info = inst.get_info()
        print("info: ", info)
        # inst.Name=inst.Name+"-1"
        info_pset.update({"Name": inst.Name})
        info_pset.update({"IfcType": info["type"]})
        # info_pset.update(info_structural)
        info_pset.update({"Project": project})
        excel_list.append(info_pset)

    # print(excel_list)
    # df1 = pd.DataFrame(excel_list)
    # print(df1['Name'])
    # ifc_file.write("segment_1.ifc")


# --------------analyze dataframe -------------

# df2 = df1
# # define the index for analyzing the file with pivo_tables, here I use ["Name","IfcType"]
# pivot1 = df2.pivot_table(index=["Name", "IfcType"])
# # count by IfcGlobalId
# pivot2 = df2.pivot_table(index=["Name", "IfcType"], values=["IfcGlobalId"], aggfunc=[len])
#
#
# outfile = SaveFileAs(".xlsx")
# outfile="./1.xlsx"
#
# writer = pd.ExcelWriter(outfile)
# df1.to_excel(writer, "Psets")
# pivot1.to_excel(writer, 'Analyze')
# pivot2.to_excel(writer, "TypeCount")
# writer.save()


def mergeNameGuid(file_path):
    # file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    # new_file_name = file_name + "_1" + file_ext
    ifc_file = parseIfc.IfcFile(file_path)
    ifc_file.update_guid_name()
    return ifc_file


def addGuidToName(old_file_path, new_file_path):
    old_file = ifc.open(old_file_path)
    instances = old_file.by_type("IfcBuildingElement")
    for inst in instances:
        try:
            inst.Name = inst.Name + "_" + inst.Tag
        except:
            print("name:", inst.Name, "tag:", inst.Tag, "failed")
            continue
    old_file.write(new_file_path)


def write_id_guid(old_file_path, new_file_path):
    ifc_file = ifc.open(old_file_path)
    # instances = ifc_file.by_type("IfcBuildingElement")#零件
    instances = ifc_file.by_type("IfcMechanicalFastener")#螺栓

    res_list = []

    for inst in instances:
        try:
            ifc_id = extract_number(str(inst))
            tekla_guid = remove_id_prefix(inst.Tag)
            global_id = inst.GlobalId
            nominalDiameter = str(inst.NominalDiameter)
            nominalLength = str(inst.NominalLength)

            res = {"id": ifc_id,"globalId":global_id, "teklaGuid": tekla_guid, "nominalDiameter":nominalDiameter,
                   "nominalLength":nominalLength}
            res_list.append(res)
        except:
            print("name:", inst.Name, "tag:", inst.Tag, "failed")
            continue

    write_id_guid_to_csv(res_list, new_file_path)
    print("完成id清单导出")


def write_id_guid_to_csv(data_list, file_path):
    field_names = ["id","globalId", "teklaGuid","nominalDiameter","nominalLength"]
    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()

        for data in data_list:
            writer.writerow(data)


def remove_id_prefix(input_string):
    prefix = "ID"
    if input_string.startswith(prefix):
        return input_string[len(prefix):]
    else:
        return input_string


def extract_number(input_string):
    # 定义正则表达式，用于匹配数字
    pattern = r'\d+'

    # 使用findall函数查找所有匹配的数字
    numbers = re.findall(pattern, input_string)

    # 返回第一个匹配的数字（如果有的话）
    if numbers:
        return numbers[0]
    else:
        return None


def browse_files():
    file_list = askopenfilenames()
    file_var.set(file_list)


def process_files():
    file_var_str = file_var.get()
    file_list = [file.strip().strip("'") for file in file_var_str[1:-1].split(",")]
    for file in file_list:
        if file=='':
            continue
        file_path, file_name = os.path.split(file)
        file_base, file_ext = os.path.splitext(file_name)
        new_file_name = file_base + "_1" + file_ext
        new_file_path = os.path.join(file_path, new_file_name)
        addGuidToName(file, new_file_path)


def export_id_guid_list():
    file_var_str = file_var.get()
    file_list = [file.strip().strip("'") for file in file_var_str[1:-1].split(",")]
    for file in file_list:
        if file=='':
            continue
        file_path, file_name = os.path.split(file)
        file_base, file_ext = os.path.splitext(file_name)
        new_file_name = file_base + "_id_relation" + ".csv"
        new_file_path = os.path.join(file_path, new_file_name)
        write_id_guid(file, new_file_path)


# def tryTkinter():
root = tk.Tk()
root.title("将ifc中GUID合并至name")
file_var = tk.StringVar()

browse_button = tk.Button(root, text="选择文件", command=browse_files)
browse_button.pack()

process_button = tk.Button(root, text="名称添加guid", command=process_files)
process_button.pack()

export_button = tk.Button(root, text="导出清单", command=export_id_guid_list)
export_button.pack()

file_label = tk.Label(root, textvariable=file_var)
file_label.pack()

root.mainloop()

# if __name__ == '__main__':
#     tryTkinter()
