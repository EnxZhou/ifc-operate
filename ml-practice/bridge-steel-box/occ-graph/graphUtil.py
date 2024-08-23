import os

import networkx as nx
import graphNodeUtil
import graphEdgeUtil

Dataset_path = "./dataset/node-edge/"

def add_solid_node(G, solid):
    G.add_node(int(solid.id), type="solid", **solid.__dict__)


def part_node_to_graph(source_file_path):
    graphml_path = dist_path_to_graphml_path(source_file_path)
    solid_list = graphNodeUtil.read_part_node_from_xlsx(source_file_path)
    graph_edge_list = graphEdgeUtil.read_edge_from_xlsx(source_file_path)

    G = nx.Graph()
    for solid in solid_list:
        add_solid_node(G, solid)

    for graph_edge in graph_edge_list:
        G.add_edge(graph_edge.node1, graph_edge.node2, weight=1)

    nx.write_graphml(G, graphml_path)
    return graphml_path


def dist_path_to_graphml_path(file_name):
    # 获取目录名和文件名
    dirname, basename = os.path.split(file_name)
    # 构造新的文件名
    new_filename = os.path.join(dirname.replace('dist', 'graphml'),
                                basename.replace('xlsx', 'graphml'))

    return new_filename


if __name__ == '__main__':
    # file_prefix = 'X11-SE1-v2.1_2'
    file_prefix = 'Z4-SS9-1-v2_1'
    graphml_path = part_node_to_graph(Dataset_path + file_prefix + "_graph.xlsx")
