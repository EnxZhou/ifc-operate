# occ-graph
主要是根据图结构数据，进行一些尝试

图结构数据是通过occ提取模型之间关系，形成的数据

可以将solid之间是否相邻作为一个图节点之间是否存在边的判断依据

也可以将face之间是否相邻作为一个图节点之间是否存在边的判断依据


# 节点-边数据转化为graphml格式
可以将带有任意特征参数的节点数据，都转化为图结构数据

带有多少特征，完全依赖读取的xlsx文件有多少列
