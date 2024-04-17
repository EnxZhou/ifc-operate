import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

label_feature = LabelEncoder()
xyz_feature = ['centreOfMassX', 'centreOfMassY', 'centreOfMassZ', 'maxFaceMass',
               'maxFaceAxisDirectX', 'maxFaceAxisDirectY', 'maxFaceAxisDirectZ',
               'faceMassAverage', 'faceMassVariance']


# xyz_feature = ['solidMass',
#                    'centreOfMassX', 'centreOfMassY', 'centreOfMassZ',
#                    'surfaceArea', 'maxFaceMass',
#                    'maxFaceCentreOfMassX', 'maxFaceCentreOfMassY', 'maxFaceCentreOfMassZ',
#                    'maxFaceAxisLocationX', 'maxFaceAxisLocationY', 'maxFaceAxisLocationZ',
#                    'maxFaceAxisDirectX', 'maxFaceAxisDirectY', 'maxFaceAxisDirectZ',
#                    'maxFacePerimeter',
#                    'maxFaceMaxEdgeCentreX', 'maxFaceMaxEdgeCentreY', 'maxFaceMaxEdgeCentreZ',
#                    'maxFaceMinEdgeCentreX', 'maxFaceMinEdgeCentreY', 'maxFaceMinEdgeCentreZ',
#                    'maxFaceEdgeLengthAverage', 'maxFaceEdgeLengthVariance', 'minFaceMass',
#                    'minFaceCentreOfMassX', 'minFaceCentreOfMassY', 'minFaceCentreOfMassZ',
#                    'minFaceAxisLocationX', 'minFaceAxisLocationY', 'minFaceAxisLocationZ',
#                    'minFaceAxisDirectX', 'minFaceAxisDirectY', 'minFaceAxisDirectZ',
#                    'minFacePerimeter',
#                    'minFaceMaxEdgeCentreX', 'minFaceMaxEdgeCentreY', 'minFaceMaxEdgeCentreZ',
#                    'minFaceMinEdgeCentreX', 'minFaceMinEdgeCentreY', 'minFaceMinEdgeCentreZ',
#                    'minFaceEdgeLengthAverage', 'minFaceEdgeLengthVariance',
#                    'faceMassAverage', 'faceMassVariance', 'edgeCount', 'edgeLenSum']


def PreprocessDataForAP():
    # xyz_feature = ['centreOfMassX', 'centreOfMassY', 'centreOfMassZ', 'maxFaceMass'
    #                ]
    # xyz_feature = [ 'centreOfMassX', 'centreOfMassY', 'centreOfMassZ']

    label_name = 'class'

    file_path = 'dataset/label-v2/xmSS9train-lm61test/'
    train_data = pd.read_csv(file_path + 'train.csv')
    data = train_data
    data.reset_index(drop=True, inplace=True)

    data.dropna(inplace=True, subset=label_name)

    scaler = MinMaxScaler()
    # 对当前需要进行归一化的字段进行归一化操作
    data[xyz_feature] = scaler.fit_transform(data[xyz_feature])

    object_feature = list(data.select_dtypes(include=['object']).columns)
    data.drop(object_feature, axis=1, inplace=True)

    return data


def AP_train(data,n_clusters=0):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.cluster import KMeans
    from sklearn.cluster import OPTICS
    has_nan = data.isna().any().any()
    if has_nan:
        print('数据存在NaN')
        return

    # xyz_data = data[xyz_feature]
    xyz_data = data

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(xyz_data)
    labels = kmeans.labels_
    print("K-Means n_clusters={}, Silhouette Coefficient: {}".format(n_clusters, silhouette_score(xyz_data, labels)))

    dbscan = OPTICS(xi=0.05, min_samples=2)
    labels = dbscan.fit_predict(xyz_data)
    # 确保labels中有多个簇
    if len(set(labels)) > 1:
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 计算噪声点
        print("OPTICS  n_clusters={}, Silhouette Coefficient: {}".format(
            n_clusters_, silhouette_score(xyz_data, labels)))
    else:
        print("No clusters found")

    agglo = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    agglo.fit(xyz_data)
    labels = agglo.labels_
    print("Agglomerative n_clusters={}, Silhouette Coefficient: {}".format(n_clusters,
                                                                           silhouette_score(xyz_data, labels)))


def cluster_train_pre(data):
    # 定义算法和参数

    startN = 2
    endN = 15
    # xyz_data=filter_correlated_features(data,0.8)
    xyz_data = remove_highly_correlated_columns(data, 0.95)
    plot_person_correlation(xyz_data)

    # 调用函数，确定聚类数量
    print("Using KMeans:")
    determine_clusters_number_kmean(xyz_data, startN, endN)

    print("\nUsing AgglomerativeClustering:")
    determine_clusters_number_agglomerative(xyz_data, startN, endN)
    return xyz_data


def remove_highly_correlated_columns(df, threshold=0.95):
    """
    删除高度相关的列
    """
    # 计算相关系数矩阵
    corr_matrix = df.corr().abs()

    # 找到高度相关的列
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # 删除高度相关的列
    df = df.drop(to_drop, axis=1)

    return df


def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    return x


def plot_person_correlation(data):
    # 1. 皮尔逊相关系数图
    correlation_matrix = data.corr()
    # 将相关系数矩阵保存为 CSV 文件
    correlation_matrix.to_csv("correlation_matrix.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Pearson Correlation Coefficient Heatmap')
    plt.savefig('person_correlation_plot.png')
    plt.show()


# 聚类算法的前处理，为了确定应该聚类成多少个
def determine_clusters_number_kmean(data, startN, endN):
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist

    wcss = []  # 平均离差
    sse = []  # SSE
    silhouette_scores = []  # 轮廓系数
    for i in range(startN, endN):
        clf = KMeans(n_clusters=i)
        clf.fit(data)
        # 计算平均离差
        m_Disp = sum(np.min(cdist(data, clf.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]
        wcss.append(m_Disp)
        # 计算簇内误差平方和（sse）
        sse.append(clf.inertia_)
        # 调用字模块metrics中的silhouette_score函数，计算轮廓系数
        silhouette_scores.append(silhouette_score(data, clf.labels_))

    # 2. 手肘图-基于平均离差
    plt.figure(figsize=(10, 6))
    plt.plot(range(startN, endN), wcss, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('The Elbow Method (Average Dissimilarity)')
    plt.show()

    # 3. 手肘图-基于SSE
    plt.figure(figsize=(10, 6))
    plt.plot(range(startN, endN), sse, 'rx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('The Elbow Method (SSE)')
    plt.show()

    # 4. 轮廓系数图
    # 这里我们使用之前计算的轮廓系数
    plt.figure(figsize=(10, 6))
    plt.plot(range(startN, endN), silhouette_scores, 'go-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Analysis')
    plt.show()


def determine_clusters_number_agglomerative(data, startN, endN):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial.distance import cdist
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import silhouette_score

    silhouette_scores = []  # 轮廓系数

    for i in range(startN, endN):
        clf = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
        clf.fit(data)
        lables = clf.labels_
        # 调用sklearn.metrics中的silhouette_score函数，计算轮廓系数
        silhouette_scores.append(silhouette_score(data, lables))

    # 2. 轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.plot(range(startN, endN), silhouette_scores, 'go-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Analysis')
    plt.show()


def plt_AP_result_point(data, n_clusters: int):
    df_normalized_data = data
    # K-means聚类
    kms = KMeans(n_clusters=n_clusters, init='k-means++')
    data_fig = kms.fit(df_normalized_data)  # 模型拟合
    centers = kms.cluster_centers_  # 计算聚类中心
    labs = kms.labels_  # 为数据打标签
    df_labels = pd.DataFrame(labs)  # 将标签存放为DataFrame
    df_labels.to_csv(f'datalabels_{n_clusters}clusters.csv')  # 输出数据标签

    # 将聚类结果为 0，1,..., n_clusters-1 的数据筛选出来 并打上标签
    df_labels_data = df_normalized_data.copy()
    df_labels_data['label'] = labs
    df_labels_data.to_csv(f'data_labeled_{n_clusters}clusters.csv')  # 输出带有标签的数据

    # 输出最终聚类中心
    df_centers = pd.DataFrame(centers, columns=[f'PC{i + 1}' for i in range(centers.shape[1])])
    df_centers.to_csv(f'data_final_center_{n_clusters}clusters.csv')

    # 假设 df_normalized_data 是已经预处理并归一化的数据集
    # 假设它包含 'centreOfMassX', 'centreOfMassY', 'centreOfMassZ' 列

    # 从数据集中提取三维坐标
    x = df_normalized_data['centreOfMassX']
    y = df_normalized_data['centreOfMassY']
    z = df_normalized_data['centreOfMassZ']

    # 如果存在聚类标签，可以使用它们来着色点
    # 假设聚类标签列名为 'label'
    labels = df_labels_data['label']

    # 创建三维散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维散点图
    scatter = ax.scatter(x, y, z, c=labels, marker='x', cmap='viridis', s=20)  # s 参数控制点的大小

    # 绘制聚类中心
    for i, center in enumerate(centers):
        # 根据聚类中心的索引找到对应的颜色
        center_label = labels[i]  # 获取聚类中心对应的标签
        center_color = scatter.cmap(scatter.norm(center_label))  # 根据标签获取颜色
        ax.scatter(center[1], center[2], center[3],
                   color=center_color,
                   marker='o',
                   s=100,
                   label=f'Center {i + 1}')

    # 设置图表标题和坐标轴标签
    ax.set_title('3D Scatter Plot of Centres of Mass of KMeans')
    ax.set_xlabel('Centre of Mass X')
    ax.set_ylabel('Centre of Mass Y')
    ax.set_zlabel('Centre of Mass Z')

    # 设置视角，旋转45度
    ax.view_init(elev=20, azim=45)

    # 添加图例
    ax.legend()
    # 添加颜色条
    plt.colorbar(scatter, ax=ax)

    # 显示图表
    plt.show()


def visualize_clusters(data, labels, title='Cluster Visualization'):
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)

    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    # 使用 set 来获取唯一标签
    unique_labels = set(labels)

    # 计算类别数量
    n_clusters = len(unique_labels)
    # 假设labels是一个包含聚类标签的数组，且聚类数量为9
    for i, label in enumerate(labels):
        if label < 0:  # 跳过噪声点或异常值
            continue
        color = plt.cm.get_cmap('tab20')(label / (n_clusters - 1))  # 使用tab20颜色映射，为每个簇分配不同的颜色
        plt.scatter(reduced_data[i - 1, 0], reduced_data[i - 1, 1], c=color, label=f'Cluster {label + 1}')  # 根据标签绘制散点

    plt.title('Cluster Visualization with 9 Clusters')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()


def main():
    data = PreprocessDataForAP()
    new_data = cluster_train_pre(data)
    plt_AP_result_point(new_data, 11)
    AP_train(new_data,11)
    # df_labels = pd.read_csv('ap_result.csv', header=None)
    # labels = df_labels[0].values
    # visualize_clusters(data,labels)


if __name__ == '__main__':
    main()
