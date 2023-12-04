import csv
import json
import re
import time
import os

import networkx as nx
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape, topods_Vertex, TopoDS_Solid
from OCC.Extend.DataExchange import read_step_file, read_step_file_with_names_colors
from OCC.Extend.TopologyUtils import TopologyExplorer

import FaceUtil
import SolidUtil
import shapeUtil

import multiprocessing


def step_path_to_graphml_path(file_name):
    # 获取目录名和文件名
    dir_name, basename = os.path.split(file_name)
    # 构造新的文件名
    new_filename = os.path.join(dir_name.replace('step', 'graphml'),
                                basename.replace('step', 'graphml'))

    return new_filename


def step_path_to_dist_path(file_name):
    # 获取目录名和文件名
    dir_name, basename = os.path.split(file_name)
    # 构造新的文件名
    new_filename = os.path.join(dir_name.replace('step', 'dist'),
                                basename.replace('step', 'csv'))

    return new_filename


def parseSolidAdjacent(file_name):
    graphml_path = step_path_to_graphml_path(file_name)
    stp_name_colors = read_step_file_with_names_colors(file_name)
    shapes = []
    num_of_shp = len(stp_name_colors)

    import networkx as nx
    from tqdm import tqdm
    from multiprocessing import Pool, Manager
    G = nx.Graph()

    for shp in stp_name_colors:
        shapes.append(shp)
        shp_name, _ = stp_name_colors[shp]
        solidProps = SolidUtil.get_solid_feature(shp)
        shp_attributes = solidProps.to_dict()
        G.add_nodes_from([(shp_name, shp_attributes)])

    for i in tqdm(range(num_of_shp), desc='Outer loop', leave=False):
        shp_name1, _ = stp_name_colors[shapes[i]]
        for j in tqdm(range(i + 1, num_of_shp), desc='Inner loop', leave=True):
            shp_name2, _ = stp_name_colors[shapes[j]]
            if shapeUtil.isShapeAdjacent(shapes[i], shapes[j], 0.001):
                G.add_edge(shp_name1, shp_name2)

    nx.write_graphml(G, graphml_path)


# 解析solid之间的距离
def parseSolidDist(file_name):
    dist_path = step_path_to_dist_path(file_name)
    stp_name_colors = read_step_file_with_names_colors(file_name)
    solid_list = []
    face_list = []
    num_of_shp = len(stp_name_colors)

    from tqdm import tqdm
    import csv

    with open(dist_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for shp in stp_name_colors:
            solid_list.append(shp)
            shp_name, _ = stp_name_colors[shp]
            solidProps = SolidUtil.get_solid_node(shp)
            current_solid_faces = FaceUtil.get_faces_from_solid(shp)
            for face in current_solid_faces:
                face_attributes = FaceUtil.get_face_node_feature(face).to_list()
                face_attributes.insert(0, 'node', 'face')
                writer.writerow(face_attributes)
            shp_attributes = solidProps.to_list()
            shp_attributes.insert(0, 'node', 'solid')
            writer.writerow(shp_attributes)

        for i in tqdm(range(num_of_shp), desc='Outer loop', leave=False):
            shp_name1, _ = stp_name_colors[solid_list[i]]
            for j in tqdm(range(i + 1, num_of_shp), desc='Inner loop', leave=True):
                shp_name2, _ = stp_name_colors[solid_list[j]]
                dist = shapeUtil.distOfShape(solid_list[i], solid_list[j])
                writer.writerow(['dist', shp_name1, shp_name2, dist.Value()])


def parseFaceDist(file_name):
    dist_path = step_path_to_dist_path(file_name)
    stp_name_colors = read_step_file_with_names_colors(file_name)
    solid_list = []
    face_list = []
    num_of_shp = len(stp_name_colors)

    from tqdm import tqdm
    import csv

    with open(dist_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for shp in stp_name_colors:
            solid_list.append(shp)
            shp_name, _ = stp_name_colors[shp]
            solidProps = SolidUtil.get_solid_node(shp)
            current_face_node_list = FaceUtil.get_face_node_from_solid(shp, shp_name)
            for current_face in current_face_node_list:
                face_list.append(current_face)
                face_feature = current_face.feature
                face_row = ['node', 'face',
                            current_face.solidName,
                            current_face.index,
                            face_feature.to_list()]
                writer.writerow(face_row)

            shp_feature = solidProps.to_list()
            shp_row = ['node', 'solid', shp_name, shp_feature]
            writer.writerow(shp_row)

        num_of_face = len(face_list)

        for i in tqdm(range(num_of_face), desc='Outer loop', leave=False):
            face_node1 = face_list[i]
            shp_name1 = face_node1.solidName
            for j in tqdm(range(i + 1, num_of_face), desc='Inner loop', leave=True):
                face_node2 = face_list[j]
                shp_name2 = face_node2.solidName
                if shp_name1 != shp_name2:
                    dist = shapeUtil.distOfShape(face_node1.topoDS_face,
                                                 face_node2.topoDS_face)
                    if dist.Value() < 0.001:
                        dist_row = [
                            'dist',
                            shp_name1,
                            face_node1.index,
                            shp_name2,
                            face_node2.index,
                            dist.Value()
                        ]
                        writer.writerow(dist_row)


def tryStepToDist():
    print(f"开始解析step文件")
    start_time = time.perf_counter()
    # parseFaceDist('data/step/scw-C3-JD-31-DB-252-freecad.step')
    parseFaceDist('data/step/62-1.step')
    parseSolidAdjacent('data/step/62-2.step')

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"执行时间为 {duration:.2f} 秒")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tryStepToDist()
