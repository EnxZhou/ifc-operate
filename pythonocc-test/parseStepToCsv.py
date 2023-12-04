import csv
import re
import time
import os

import pandas as pd
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_ManifoldSolidBrep
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape, topods_Vertex, TopoDS_Solid
from OCC.Extend.DataExchange import read_step_file, read_step_file_with_names_colors
from OCC.Extend.TopologyUtils import TopologyExplorer

import FaceUtil
import SolidUtil
import shapeUtil


def shape_edges_length(the_shape):
    t = TopologyExplorer(the_shape)
    props = GProp_GProps()
    edgeList = []
    edgeCount = t.number_of_edges()
    print("edge count: ", edgeCount)
    for edge in t.edges():
        brepgprop_LinearProperties(edge, props)
        edge_len = props.Mass()
        edgeList.append(round(edge_len, 2))
    return edgeList


# 体的全部顶点
def shape_points(the_shape):
    t = TopologyExplorer(the_shape)
    pointCount = t.number_of_vertices()
    print("point count: ", pointCount)
    anVertexExplorer = TopExp_Explorer(the_shape, TopAbs_VERTEX)
    vertex = []
    while anVertexExplorer.More():
        anVertex = topods_Vertex(anVertexExplorer.Current())
        aPnt = BRep_Tool.Pnt(anVertex)

        vertex.append(aPnt)
        anVertexExplorer.Next()

    pnts = []
    for v in vertex:
        coordinate = (v.X(), v.Y(), v.Z())
        if coordinate not in pnts:
            pnts.append(coordinate)

    return pnts


def step_model_explore2():
    solidList = SolidUtil.get_solids_from_step('data/step/62-2.step')
    for solid in solidList:
        solidProps = SolidUtil.get_solid_feature(solid)
        print("solid mass:", solidProps.mass)
        print("solid centre mass:", solidProps.centreOfMassX, solidProps.centreOfMassY, solidProps.centreOfMassZ)
        print("solid max face prop:", solidProps.maxFaceFeature)
        print("solid max face max edge props:", solidProps.maxFaceFeature.maxEdgeFeature)

        faceList = FaceUtil.get_faces_from_solid(solid)
        print("-------------------")
        # for face in faceList:
        #     faceProps = FaceUtil.get_face_props(face)
        #     print(faceProps.kind, "axis direct:", (faceProps.axisDirectX, faceProps.axisDirectY, faceProps.axisDirectZ))
        #     print(faceProps.kind, "axis location:",
        #           (faceProps.axisLocationX, faceProps.axisLocationY, faceProps.axisLocationZ))


def parse_step_to_df(file_name):
    stp_name_colors = read_step_file_with_names_colors(file_name)
    solid_data = []
    for shp in stp_name_colors:
        solidProps = SolidUtil.get_solid_feature(shp)
        shp_name, _ = stp_name_colors[shp]
        name, guid = parse_name_and_guid(shp_name)
        data = {
            "name": name,
            "guid": guid,
            "solidMass": solidProps.mass,
            "centreOfMassX": solidProps.centreOfMassX,
            "centreOfMassY": solidProps.centreOfMassY,
            "centreOfMassZ": solidProps.centreOfMassZ,
            "surfaceArea": solidProps.surfaceArea,
            "maxFaceMass": solidProps.maxFaceFeature.mass,
            "maxFaceCentreOfMassX": solidProps.maxFaceFeature.centreOfMassX,
            "maxFaceCentreOfMassY": solidProps.maxFaceFeature.centreOfMassY,
            "maxFaceCentreOfMassZ": solidProps.maxFaceFeature.centreOfMassZ,
            "maxFaceAxisLocationX": solidProps.maxFaceFeature.axisLocationX,
            "maxFaceAxisLocationY": solidProps.maxFaceFeature.axisLocationY,
            "maxFaceAxisLocationZ": solidProps.maxFaceFeature.axisLocationZ,
            "maxFaceAxisDirectX": solidProps.maxFaceFeature.axisDirectX,
            "maxFaceAxisDirectY": solidProps.maxFaceFeature.axisDirectY,
            "maxFaceAxisDirectZ": solidProps.maxFaceFeature.axisDirectZ,
            "maxFacePerimeter": solidProps.maxFaceFeature.perimeter,
            "maxFaceMaxEdgeLength": solidProps.maxFaceFeature.maxEdgeFeature.length,
            "maxFaceMaxEdgeCentreX": solidProps.maxFaceFeature.maxEdgeFeature.centreX,
            "maxFaceMaxEdgeCentreY": solidProps.maxFaceFeature.maxEdgeFeature.centreY,
            "maxFaceMaxEdgeCentreZ": solidProps.maxFaceFeature.maxEdgeFeature.centreZ,
            "maxFaceMinEdgeLength": solidProps.maxFaceFeature.minEdgeFeature.length,
            "maxFaceMinEdgeCentreX": solidProps.maxFaceFeature.minEdgeFeature.centreX,
            "maxFaceMinEdgeCentreY": solidProps.maxFaceFeature.minEdgeFeature.centreY,
            "maxFaceMinEdgeCentreZ": solidProps.maxFaceFeature.minEdgeFeature.centreZ,
            "maxFaceEdgeLengthAverage": solidProps.maxFaceFeature.edgeLengthAverage,
            "maxFaceEdgeLengthVariance": solidProps.maxFaceFeature.edgeLengthVariance,
            "minFaceMass": solidProps.minFaceFeature.mass,
            "minFaceCentreOfMassX": solidProps.minFaceFeature.centreOfMassX,
            "minFaceCentreOfMassY": solidProps.minFaceFeature.centreOfMassY,
            "minFaceCentreOfMassZ": solidProps.minFaceFeature.centreOfMassZ,
            "minFaceAxisLocationX": solidProps.minFaceFeature.axisLocationX,
            "minFaceAxisLocationY": solidProps.minFaceFeature.axisLocationY,
            "minFaceAxisLocationZ": solidProps.minFaceFeature.axisLocationZ,
            "minFaceAxisDirectX": solidProps.minFaceFeature.axisDirectX,
            "minFaceAxisDirectY": solidProps.minFaceFeature.axisDirectY,
            "minFaceAxisDirectZ": solidProps.minFaceFeature.axisDirectZ,
            "minFacePerimeter": solidProps.minFaceFeature.perimeter,
            "minFaceMaxEdgeLength": solidProps.minFaceFeature.minEdgeFeature.length,
            "minFaceMaxEdgeCentreX": solidProps.minFaceFeature.minEdgeFeature.centreX,
            "minFaceMaxEdgeCentreY": solidProps.minFaceFeature.minEdgeFeature.centreY,
            "minFaceMaxEdgeCentreZ": solidProps.minFaceFeature.minEdgeFeature.centreZ,
            "minFaceMinEdgeLength": solidProps.minFaceFeature.minEdgeFeature.length,
            "minFaceMinEdgeCentreX": solidProps.minFaceFeature.minEdgeFeature.centreX,
            "minFaceMinEdgeCentreY": solidProps.minFaceFeature.minEdgeFeature.centreY,
            "minFaceMinEdgeCentreZ": solidProps.minFaceFeature.minEdgeFeature.centreZ,
            "minFaceEdgeLengthAverage": solidProps.minFaceFeature.edgeLengthAverage,
            "minFaceEdgeLengthVariance": solidProps.minFaceFeature.edgeLengthVariance,
            "faceMassAverage": solidProps.faceMassAverage,
            "faceMassVariance": solidProps.faceMassVariance,
            "edgeCount": solidProps.edgeCount,
            "edgeLenSum": solidProps.edgeLenSum,
            "segment_no": file_name
        }
        solid_data.append(data)

    part_df = pd.DataFrame(solid_data)
    return part_df


def parse_step_to_csv(file_name):
    df = parse_step_to_df(file_name)
    df_to_csv(file_name, df)

    print(f"{file_name}已解析完毕")


# 因为是只解析step文件的几何属性，所有只能把自定义属性加在solid的name中，所以name为"模型名称_GUID"
def parse_name_and_guid(old_name):
    name_pattern = r'[\u4e00-\u9fa5\w]+'
    guid_pattern = r'_(?:ID)?(([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-' \
                   r'[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}))'
    pattern = r'^(' + name_pattern + ')' + guid_pattern + '?$'
    match = re.match(pattern, old_name)
    if match:
        name = match.group(1)
        if match.group(2):
            guid = match.group(2)
            if guid[:2] == 'ID':
                guid = guid[2:]
        else:
            guid = None
        return name, guid
    else:
        return old_name, None


# 将一个文件夹下所有step模型进行解析，并且合并成一个dataframe
def folder_step_to_df(folder_path, old_step_name):
    guid_index_dict = {}

    # Step 1: Read folder and extract guid, index from each "{old_step_name}.step" file
    old_file_path = os.path.join(folder_path, old_step_name+".step")
    old_df = parse_step_to_df(old_file_path)
    guid_index_dict.update(dict(zip(old_df.index, old_df['guid'])))

    # Step 2: Read all "xxx" files and update the "guid" column using the guid_index_dict
    dfs = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith(old_step_name):
            file_path = os.path.join(folder_path, file_name)
            print(f"正在解析模型：{file_name}")
            df = parse_step_to_df(file_path)
            df['guid'] = df.index.map(guid_index_dict)
            dfs.append(df)

    # Step 3: Merge all DataFrames into a single DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df


def df_to_csv(file_name, df):
    csv_filename = file_name.replace(".step", ".csv")
    df.index = df.index+1
    df.to_csv(csv_filename, index_label='index')


# 解析多个step模型的特征
def parse_mul_step_to_csv():
    file_path = "data/step/"
    file_name = "C3-JD-27_v2_1"
    full_file_path = os.path.join(file_path, file_name+".step")
    df = folder_step_to_df(file_path, file_name)
    df_to_csv(full_file_path, df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f"开始解析step文件")
    start_time = time.perf_counter()

    parse_step_to_csv('data/step/61-v2.1_1.step')

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"执行时间为 {duration:.2f} 秒")
