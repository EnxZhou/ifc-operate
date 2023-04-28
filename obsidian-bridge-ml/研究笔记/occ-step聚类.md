## AgglomerativeClustering

只针对如下特征，进行聚类模型训练
- centreOfMassX
- centreOfMassY
- centreOfMassZ
- maxFaceMass
- faceMassAverage
- faceMassVariance
- edgeCount

针对龙门桥-61进行训练，算法采用AgglomerativeClustering
AgglomerativeClustering(n_clusters=10, linkage='ward')

训练后的结果，发现可以完全把内部板IB(Internal board)聚类成一个类别

对沙埕湾C3-JD-24模型也进行了同样的训练，效果一般

## AffinityPropagation

特征少的时候，聚类数都是很大，基本没法用
特征多的时候，聚类结果不稳定，同样的参数，可能有多种聚类数结果
