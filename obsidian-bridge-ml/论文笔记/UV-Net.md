We introduce UV-Net, a novel neural network architecture and representation designed to operate directly on Boundary representation (B-rep) data from 3D CAD models. The B-rep format is widely used in the design, simulation and manufacturing industries to enable sophisticated and precise CAD modeling operations. However, B-rep data presents some unique challenges when used with modern machine learning due to the complexity of the data structure and its support for both continuous non-Euclidean geometric entities and discrete topological entities. In this paper, we ropose a unified representation for B-rep data that exploits the U and V parameter domain of curves and surfaces to model geometry, and an adjacency graph to explicitly model topology. This leads to a unique and efficient network architecture, UV-Net, that couples image and graph convolutional neural networks in a compute and memory-efficient manner. To aid in future research we present a synthetic labelled B-rep dataset, SolidLetters, derived from human designed fonts with variations in both geometry and topology. Finally we demonstrate that UV-Net can generalize to supervised and unsupervised tasks on five datasets, while outperforming alternate 3D shape representations such as point clouds, voxels, and meshes.

我们介绍了UV-Net，一种新型的神经网络结构和表示方法，旨在直接操作来自三维CAD模型的边界表示（B-rep）数据。B-rep格式被广泛用于设计、模拟和制造行业，以实现复杂和精确的CAD建模操作。然而，B-rep数据在用于现代机器学习时提出了一些独特的挑战 由于数据结构的复杂性以及它对连续的非欧几里得几何实体和离散的拓扑实体的支持。在本文中，我们为B-rep数据提出了一个统一的表示方法，利用曲线和曲面的U和V 曲线和曲面的参数域来建立几何模型、 和一个邻接图来明确地模拟拓扑结构。这 导致了一个独特而高效的网络架构，UV-Net、 它结合了图像和图形卷积神经网络 以一种计算和记忆效率高的方式结合在一起。为了帮助未来的 为了帮助未来的研究，我们提出了一个合成的标记B-rep数据集SolidLetters，该数据集来自于人类设计的字体，在几何和拓扑结构上都有变化。在几何形状和拓扑结构上都有变化。最后，我们证明了 UV-Net可以在五个数据集上通用于有监督和无监督的任务。在五个数据集上的监督和非监督任务，同时优于其他的三维形状 表示，如点云、体素和网格。

通过www.DeepL.com/Translator（免费版）翻译

Jayaraman P K, Sanghi A, Lambourne J G, et al. Uv-net: Learning from boundary representations[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 11703-11712.

## 涉及五个数据集
### Machining feature dataset

Zhang Z, Jaiswal P, Rai R. Featurenet: Machining feature recognition based on 3d convolution neural network[J]. Computer-Aided Design, 2018, 101: 12-22.
该数据集采用Soildworks API创建，
https://github.com/zibozzb/FeatureNet
https://github.com/madlabub/Machining-feature-dataset

### MFCAD dataset

[[开源项目/MFCAD]]

### FabWave dataset

Angrish A, Craver B, Starly B. “FabSearch”: A 3D CAD Model-Based Search Engine for Sourcing Manufacturing Services[J]. Journal of Computing and Information Science in Engineering, 2019, 19(4).

### ABC dataset

Koch S, Matveev A, Jiang Z, et al. Abc: A big cad model dataset for geometric deep learning[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 9601-9611.

https://deep-geometry.github.io/abc-dataset/

### SolidLetters dataset

本篇论文作者自己的数据集

#FR 