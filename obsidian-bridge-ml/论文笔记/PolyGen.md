PolyGen: An Autoregressive Generative Model of 3D Meshes

#多边形网格

多边形网格是三维几何的高效表示方法，在计算机图形学、机器人技术和游戏开发中具有重要作用。现有的基于学习的方法避开了处理三维网格的挑战，改用其他更符合神经网络架构和训练方法的对象表示方法。我们提出了一种直接对网格进行建模的方法，使用基于Transformer的架构逐个预测网格的顶点和面。我们的模型可以以一系列输入作为条件，包括物体类别、体素和图像，并且由于模型是概率性的，所以可以产生在模棱两可情况下捕捉不确定性的样品。我们展示了该模型能够产生高质量、可用的网格，并且为网格建模任务建立了对数似然度基准。

Nash C, Ganin Y, Eslami S M A, et al. Polygen: An autoregressive generative model of 3d meshes[C]//International conference on machine learning. PMLR, 2020: 7220-7229.