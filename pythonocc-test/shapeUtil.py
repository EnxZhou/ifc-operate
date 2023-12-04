from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape


def isShapeAdjacent(shape1, shape2, tolerance):
    """determine whether two shapes are adjacent to each other
        _shape1: shape NO.1
        _shape2: shape NO.2
        tolerance: the tolerance of the distance, when the distance is less than this tolerance,
         determine that the two shapes are adjacent to each other
    """
    dist = BRepExtrema_DistShapeShape(shape1, shape2)
    # 似乎在python中设置多线程没有效果
    # dist.SetMultiThread(True)
    dist.Perform()
    if dist.Value() < tolerance:
        return True
    else:
        return False


def distOfShape(shape1, shape2):
    dist = BRepExtrema_DistShapeShape(shape1, shape2)
    # 似乎在python中设置多线程没有效果
    # dist.SetMultiThread(True)
    dist.Perform()
    return dist.Value()
