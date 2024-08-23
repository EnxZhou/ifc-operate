import networkx as nx
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import graphUtil
from collections import defaultdict


def read_graphml(filename):
    """Reads a GraphML file and returns a NetworkX graph."""
    return nx.read_graphml(filename)


def create_color_dict(num_labels):
    # 创建一个颜色映射，使用viridis colormap
    cmap = plt.get_cmap('turbo', num_labels)

    # 生成颜色字典，键是标签，值是对应的颜色
    color_dict = {i: cmap(i) for i in range(num_labels)}

    return color_dict


# 这个函数和散点聚类重复了,可以独立成为一个通用函数
def plot_3d_from_graphml(graphml, num_labels):
    # 读取 GraphML 文件
    G = graphml

    # 创建颜色字典
    color_dict = create_color_dict(num_labels)

    # 初始化坐标和标签列表
    node_coordinates = []
    node_labels = []

    # 聚类中心点字典
    cluster_centers = defaultdict(list)

    # 遍历所有节点，提取 x, y, z 和 label 属性
    for node, data in G.nodes(data=True):
        x = data.get('centreOfMassX', 0)  # 假设 x 属性存在，否则默认为 0
        y = data.get('centreOfMassY', 0)  # 同上
        z = data.get('centreOfMassZ', 0)  # 同上
        label = data.get('label', 0)  # 同上
        node_coordinates.append((x, y, z))
        node_labels.append(label)
        # 按聚类标签收集点坐标
        cluster_centers[label].append((x, y, z))

    # 计算每个聚类的中心点
    for label, points in cluster_centers.items():
        avg_x = sum(point[0] for point in points) / len(points)
        avg_y = sum(point[1] for point in points) / len(points)
        avg_z = sum(point[2] for point in points) / len(points)
        cluster_centers[label] = (avg_x, avg_y, avg_z)

    # 创建一个新的图来绘制三维散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图，根据标签不同使用不同的颜色
    scatter = ax.scatter(*zip(*node_coordinates),
                         c=[color_dict[label] for label in node_labels], marker='x',
                         cmap='turbo', s=20, label='Nodes')

    # 绘制聚类中心点，使用与对应节点相同的颜色
    for label, center in cluster_centers.items():
        ax.scatter(*center, c=color_dict[label], marker='o', s=100)

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    # cbar.set_label('Cluster Labels')

    # 添加坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置视角，旋转45度
    ax.view_init(elev=20, azim=40)

    # 添加图例
    ax.legend()

    # 显示图表
    plt.show()


def visualize_graph1(graph):
    """Visualizes the graph."""
    nx.draw(graph, with_labels=True)
    plt.show()


def visualize_graph2(graph):
    """Visualizes the graph with node colors based on the 'label' attribute."""
    # 定义一个颜色映射字典，可以根据实际的标签值进行调整
    label_start = 1

    # 生成颜色映射，为每个大于1的整数生成一个随机颜色
    color_map = {i: np.random.rand(3) for i in range(label_start, label_start + 10)}

    # 根据节点的'label'属性值创建颜色列表
    node_colors = [color_map.get(node_label, 'grey') for node_label in nx.get_node_attributes(graph, 'label').values()]

    # 绘制图形，使用node_color参数应用颜色映射
    nx.draw(graph, with_labels=True, node_color=node_colors, cmap=plt.cm.get_cmap('tab20', max(len(node_colors), 20)))

    # 显示图形
    plt.show()


# 定义一个函数来评估不同聚类数的模型
def evaluate_clustering(graph, num_clusters_range):
    # Generate node features based on node attributes
    node_features = []
    for node in graph.nodes:
        # Extract all node attributes into a feature vector
        feature_vector = []
        for attr_key, attr_value in graph.nodes[node].items():
            if isinstance(attr_value, (int, float)):
                feature_vector.append(attr_value)  # Assume numerical attributes
            # You can add more handling for other types of attributes if needed
        node_features.append(feature_vector)

    # Convert features to numpy array
    X = np.array(node_features)

    silhouette_scores = []
    for num_clusters in num_clusters_range:
        clustering_model = SpectralClustering(
            n_clusters=num_clusters,
            affinity='nearest_neighbors',
            n_neighbors=15,
            # n_init=25,
            random_state=0)
        cluster_labels = clustering_model.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        print(f"For {num_clusters} clusters the silhouette score is {score:.2f}")
    return silhouette_scores


# 确定谱聚类聚类个数
def determine_clusters_number_graph(graph, startN, endN):
    # 评估不同聚类数的模型
    num_clusters_range = range(startN, endN,10)
    silhouette_scores = evaluate_clustering(graph, num_clusters_range)

    # 绘制肘部曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(num_clusters_range, silhouette_scores, marker='o')
    # plt.title('Elbow Method For Optimal K')
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.xlabel('聚类数量')
    plt.ylabel('轮廓系数')
    plt.xticks(num_clusters_range)
    plt.grid(True)
    plt.show()


def cluster_graph(graph, num_clusters):
    """Clusters the graph using K-Means algorithm."""
    # Generate node features based on node attributes
    node_features = []
    for node in graph.nodes:
        # Extract all node attributes into a feature vector
        feature_vector = []
        for attr_key, attr_value in graph.nodes[node].items():
            if isinstance(attr_value, (int, float)):
                feature_vector.append(attr_value)  # Assume numerical attributes
            # You can add more handling for other types of attributes if needed
        node_features.append(feature_vector)

    # Convert features to numpy array
    X = np.array(node_features)

    # Apply K-Means clustering
    clf = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
    clf.fit(X)

    # Assign cluster labels to nodes
    for i, label in enumerate(clf.labels_):
        graph.nodes[str(i + 1)]['label'] = int(label)

    return graph


def load_graphml():
    graph_file = get_graphml_file()
    g = read_graphml(graph_file)
    return g


def get_graphml_file():
    # 读取 GraphML 文件
    # file_prefix = 'X11-SE1-v2.1_2'
    file_prefix = 'Z4-SS9-1-v2_1'
    graph_file = graphUtil.Dataset_path + file_prefix + "_graph.graphml"
    return graph_file


def analyze_graph(output_csv):
    # 读取GraphML文件
    G = load_graphml()

    # 计算节点度的平均值和最大值
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)

    # 计算全局聚类系数
    global_clustering_coefficient = nx.average_clustering(G)

    # 计算平均路径长度
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        avg_path_length = nx.average_shortest_path_length(subgraph)

    # 计算连通分量数量和最大的连通分量的节点比例
    num_connected_components = nx.number_connected_components(G)
    largest_cc_size = len(max(nx.connected_components(G), key=len))
    largest_cc_ratio = largest_cc_size / G.number_of_nodes()

    # 创建统计信息表格
    data = {
        '统计信息': [
            '平均度',
            '最大度',
            '全局聚类系数',
            '平均路径长度',
            '连通分量数量',
            '最大连通分量比例'
        ],
        '数值': [
            avg_degree,
            max_degree,
            global_clustering_coefficient,
            avg_path_length,
            num_connected_components,
            largest_cc_ratio
        ],
        '描述': [
            '节点的平均连接边数',
            '单个节点的最大连接边数',
            '图的平均聚类系数',
            '最大连通分量中节点之间的平均最短路径长度',
            '图中的连通分量数量',
            '最大连通分量的节点数量占总节点数量的比例'
        ]
    }

    df = pd.DataFrame(data)

    # 将表格保存为CSV文件
    df.to_csv(output_csv, index=False)

    print(f"统计信息已保存到 {output_csv}")


if __name__ == "__main__":
    graph = load_graphml()
    # 显示图结构数据统计信息
    analyze_graph('图数据统计信息')
    # 可视化原始图
    # visualize_graph1(graph)
    # 确定聚类个数
    # determine_clusters_number_graph(graph,10,150)

    # 执行聚类算法
    num_clusters = 60  # 假设要聚类为n个集群
    clustered_graph = cluster_graph(graph, num_clusters)
    plot_3d_from_graphml(graphml=graph, num_labels=num_clusters)

    # Visualize clustered graph
    # visualize_graph2(clustered_graph)
