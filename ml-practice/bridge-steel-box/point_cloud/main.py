import open3d as o3d
import numpy as np
from sklearn.preprocessing import StandardScaler


def cluster_ply():
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud("dataset/C3-JD-24_1.ply")

    # 提取点云数据
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    features = np.concatenate((points_scaled,normals), axis=1)

    n_clusters = 10
    from sklearn.cluster import KMeans
    kmean_clf = KMeans(n_clusters=n_clusters, random_state=0)
    kmean_clf.fit(features)
    labels = kmean_clf.labels_

    # from sklearn.cluster import AgglomerativeClustering
    # agg_clf = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    # agg_clf.fit(features)
    # labels=agg_clf.labels_

    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D

    # 可视化聚类结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_clusters):
        ax.scatter(points[labels==i, 0], points[labels==i, 1], points[labels==i, 2], label=f'Cluster {i+1}')
    ax.legend()
    plt.show()


def cluster_voxel():
    # 读取 STEP 模型
    model = o3d.io.read_triangle_mesh("dataset/tekla-2019-C3-JD-31-DB-252-modifyName-freecad.stl")

    # 将 STEP 模型转换为体素模型
    voxel_size = 10
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(model, voxel_size)

    # 将体素模型进行内部体素化
    # seed = (0, 0, 0)
    # mask = voxel_grid.crop(voxel_grid.get_axis_aligned_bounding_box())
    # filled_voxel_grid = o3d.geometry.VoxelGrid.create_from_voxel_grid_with_local_search(
    #     voxel_grid, mask, seed
    # )
    #
    # # 可视化内部体素化结果
    # o3d.visualization.draw_geometries([filled_voxel_grid])
    o3d.visualization.draw_geometries([voxel_grid])



if __name__ == "__main__":
    # cluster_ply()
    cluster_voxel()
