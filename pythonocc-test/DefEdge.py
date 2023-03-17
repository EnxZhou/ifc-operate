import vectors
# vector代表三维空间中的坐标

# Points 点的集合
class PointsDef:
    def __init__(self, *vectors):
        self.vectors = list(vectors)


# Arrow 方向
class ArrowDef:
    def __init__(self, tip, tail=(0, 0, 0)):
        # 头
        self.tip = tip
        # 尾
        self.tail = tail
        # 角度
        self.arrow = vectors.subtract(self.tip, self.tail)


# Edge 边
class EdgeDef:
    def __init__(self, start_point, end_point):
        self.arrow = vectors.subtract(end_point, start_point)
        self.start_point = start_point
        self.end_point = end_point

    # 边长
    def length(self):
        self.length = vectors.distance(self.start_point, self.end_point)
        return self.length

    def directX(self):
        return self.arrow[0]

    def directY(self):
        return self.arrow[1]

    def directZ(self):
        return self.arrow[2]