# coding=utf-8
import csv
import json
import os
import pickle
import time

import networkx as nx

import FaceUtil
import SolidUtil
import graphEdgeUtil
import graphNodeUtil


def dist_path_to_graphml_path(file_name):
    # 获取目录名和文件名
    dirname, basename = os.path.split(file_name)
    # 构造新的文件名
    new_filename = os.path.join(dirname.replace('dist', 'graphml'),
                                basename.replace('xlsx', 'graphml'))

    return new_filename


def dist_to_graph(file_path):
    graphml_path = dist_path_to_graphml_path(file_path)
    solid_list = SolidUtil.solid_node_from_csv(file_path)
    face_list = FaceUtil.face_node_from_xlsx(file_path)
    graph_edge_list = graphEdgeUtil.dist_edge_from_xlsx(file_path)

    G = nx.Graph()
    for solid in solid_list:
        G.add_node(solid.name, type="solid",
                   solidMass=solid.feature.mass,
                   centreOfMassX=solid.feature.centreOfMassX,
                   centreOfMassY=solid.feature.centreOfMassY,
                   centreOfMassZ=solid.feature.centreOfMassZ,
                   faceCount=solid.feature.faceCount)

    for face in face_list:
        faceName = face.solidName + '_' + str(face.index)
        G.add_node(faceName, type="face",
                   faceMass=face.feature.mass,
                   centreOfMassX=face.feature.centreOfMassX,
                   centreOfMassY=face.feature.centreOfMassY,
                   centreOfMassZ=face.feature.centreOfMassZ,
                   edgeCount=face.feature.edgeCount,
                   )
        G.add_edge(face.solidName, faceName, type="face")

    for graph_edge in graph_edge_list:
        G.add_edge(graph_edge.node1, graph_edge.node2, type="adj")

    nx.write_graphml(G, graphml_path)


def main_sub_node_to_graph(source_file_path):
    graphml_path = dist_path_to_graphml_path(source_file_path)
    solid_list = graphNodeUtil.read_main_sub_node_from_xlsx(source_file_path)
    graph_edge_list = graphEdgeUtil.read_edge_from_xlsx(source_file_path)

    G = nx.Graph()
    for solid in solid_list:
        G.add_node(solid.id, type="solid",
                   id=solid.id,
                   guid=solid.guid,
                   node_class=solid.node_class,
                   centre_of_mass_x=solid.centre_of_mass_x,
                   centre_of_mass_y=solid.centre_of_mass_y,
                   centre_of_mass_z=solid.centre_of_mass_z,
                   max_face_axis_location_x=solid.max_face_axis_location_x,
                   max_face_axis_location_y=solid.max_face_axis_location_y,
                   max_face_axis_location_z=solid.max_face_axis_location_z,
                   max_face_axis_direct_x=solid.max_face_axis_direct_x,
                   max_face_axis_direct_y=solid.max_face_axis_direct_y,
                   max_face_axis_direct_z=solid.max_face_axis_direct_z)

    for graph_edge in graph_edge_list:
        # 增加边的权重，虽说可以针对main-main，main-sub，sub-sub之间的关系进行区分
        # node1_class=G.nodes[graph_edge.node1]['node_class']
        # node2_class=G.nodes[graph_edge.node2]['node_class']
        # if node1_class==node2_class and node1_class=='main':
        #     G.add_edge(graph_edge.node1, graph_edge.node2, weight=999)
        # else:
        #     G.add_edge(graph_edge.node1, graph_edge.node2, weight=1)
        G.add_edge(graph_edge.node1, graph_edge.node2, weight=1)

    nx.write_graphml(G, graphml_path)
    return graphml_path


# 读取occ_id关系表，该表为guid，ifc-id，occ-id的对应关系
def read_occ_id_map_ifc_id(csv_file_path):
    relation_map = {}
    with open(csv_file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ifc_id = row["ifc-id"]
            occ_id = row["occ-id"]
            relation_map[occ_id] = ifc_id
    return relation_map


def extract_ifc_ids(occ_ids, relation_map):
    return [relation_map.get(occ_id, "") for occ_id in occ_ids]


def save_results_to_file(filename, data: dict):
    with open(filename, "w") as file:
        for key, ids in data.items():
            file.write(f"{key}:{ids}\n")


def combine_and_print_results(ifc_id_res, occ_id_res):
    for key, ifc_ids in ifc_id_res.items():
        print(ifc_ids)

    print("---------------------------------------------")

    for key, occ_ids in occ_id_res.items():
        print(occ_ids)


def combine_and_export_results(file_prefix, ifc_id_res, occ_id_res):
    ifc_filename = f"{file_prefix}_ifc-id_split.txt"
    occ_filename = f"{file_prefix}_occ-id_split.txt"
    save_results_to_file(ifc_filename, ifc_id_res)
    save_results_to_file(occ_filename, occ_id_res)


def export_combine_main_node(file_prefix, graphml_path, csv_file_path):
    G = nx.read_graphml(graphml_path)
    combine_node_dict = graphNodeUtil.combine_node(G)

    occ_map_ifc = read_occ_id_map_ifc_id(csv_file_path)

    ifc_id_res = {}
    occ_id_res = {}
    occ_id_dict = {}
    for key, occ_ids in combine_node_dict.items():
        ifc_ids = extract_ifc_ids(occ_ids, occ_map_ifc)
        ifc_id_res[key] = ",".join(filter(None, ifc_ids))
        occ_id_res[key] = ",".join(filter(None, occ_ids))
        occ_id_dict[key] = occ_ids

    combine_and_export_results(file_prefix, ifc_id_res, occ_id_res)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f"开始解析dist文件")
    start_time = time.perf_counter()

    # dist_to_graph("../../data/dist/62-1-dist-nogbk.xlsx")
    # file_prefix = '61-v2.1'
    # file_prefix = 'C3-JD-27_v2.1_1'
    file_prefix = 'X11-SE1-v2.1_1'
    graphml_path = main_sub_node_to_graph("../../data/dist/" + file_prefix + "_graph.xlsx")
    export_combine_main_node(file_prefix=file_prefix, graphml_path=graphml_path,
                             csv_file_path="../../data/dist/" + file_prefix + "-id关系表.csv")

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"执行时间为 {duration:.2f} 秒")
