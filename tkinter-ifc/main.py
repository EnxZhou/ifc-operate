#!/bin/python
import os.path

import ifcopenshell as ifc
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
import pandas as pd

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


def modifyIfcFile(file_path):
    file_name,file_ext = os.path.splitext(os.path.basename(file_path))
    new_file_name = file_name+"_1"+file_ext
    ifc_file = parseIfc.IfcFile(file_path)
    ifc_file.update_guid_name()
    return ifc_file


def test3():
    ifc_file_path = OpenFile(".ifc", ("IFC-Files", "*.ifc"))

    newfile = modifyIfcFile(ifc_file_path)
    newFileName = SaveFileAs(".ifc")
    newfile.save_file(newFileName)

if __name__ == '__main__':
    test3()
