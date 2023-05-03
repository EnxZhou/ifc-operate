import csv
import re
import time
import os

import pandas as pd
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape, topods_Vertex,TopoDS_Solid
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
    solidList = SolidUtil.get_solids_from_step('segment-1.step')
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


def parse_step_to_csv(file_name):
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
            "maxFaceMaxEdgeCentreX": solidProps.maxFaceFeature.maxEdgeFeature.centreX,
            "maxFaceMaxEdgeCentreY": solidProps.maxFaceFeature.maxEdgeFeature.centreY,
            "maxFaceMaxEdgeCentreZ": solidProps.maxFaceFeature.maxEdgeFeature.centreZ,
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
            "minFaceMaxEdgeCentreX": solidProps.minFaceFeature.minEdgeFeature.centreX,
            "minFaceMaxEdgeCentreY": solidProps.minFaceFeature.minEdgeFeature.centreY,
            "minFaceMaxEdgeCentreZ": solidProps.minFaceFeature.minEdgeFeature.centreZ,
            "minFaceMinEdgeCentreX": solidProps.minFaceFeature.minEdgeFeature.centreX,
            "minFaceMinEdgeCentreY": solidProps.minFaceFeature.minEdgeFeature.centreY,
            "minFaceMinEdgeCentreZ": solidProps.minFaceFeature.minEdgeFeature.centreZ,
            "minFaceEdgeLengthAverage": solidProps.minFaceFeature.edgeLengthAverage,
            "minFaceEdgeLengthVariance": solidProps.minFaceFeature.edgeLengthVariance,
            "faceMassAverage": solidProps.faceMassAverage,
            "faceMassVariance": solidProps.faceMassVariance,
            "edgeCount": solidProps.edgeCount,
            "edgeLenSum": solidProps.edgeLenSum
        }
        solid_data.append(data)

    df = pd.DataFrame(solid_data)
    csv_filename = file_name.replace(".step", ".csv")
    df.to_csv(csv_filename)

    print(f"{file_name}已解析完毕，生成{csv_filename}")


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f"开始解析step文件")
    start_time = time.perf_counter()
    # step_model_explore2()
    # parse_step_to_csv('data/step/C3-JD-27_1.step')

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"执行时间为 {duration:.2f} 秒")
