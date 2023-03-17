from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Edge
from OCC.Extend.TopologyUtils import is_face, is_edge, TopologyExplorer
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties


# 边的特征
class EdgeFeature:
    isStraight = True
    length = 0.0
    centreX = 0.0
    centreY = 0.0
    centreZ = 0.0

    # 边的起始点暂时还没取到
    # startX = 0.0
    # startY = 0.0
    # startZ = 0.0
    # endX = 0.0
    # endY = 0.0
    # endZ = 0.0
    # directX = 0.0
    # directY = 0.0
    # directZ = 0.0
    def __init__(self):
        self.length=0.0
        self.centreX = 0.0
        self.centreY = 0.0
        self.centreZ = 0.0

    def __str__(self):
        return f"EdgeFeature({self.isStraight}," \
               f" {self.length}," \
               f" ({self.centreX}," \
               f" {self.centreY}," \
               f" {self.centreZ}))"


def get_edge_props(_edge):
    edgeFeatureReturn = EdgeFeature()
    if not isinstance(_edge, TopoDS_Edge):
        return edgeFeatureReturn
    else:
        props = GProp_GProps()
        brepgprop_LinearProperties(_edge, props)
        edgeFeatureReturn.length = props.Mass()
        centreOfMass = props.CentreOfMass()
        edgeFeatureReturn.centreX = centreOfMass.X()
        edgeFeatureReturn.centreY = centreOfMass.Y()
        edgeFeatureReturn.centreZ = centreOfMass.Z()

    return edgeFeatureReturn


# 获取面的所有边
def get_edges_from_face(_face: TopoDS_Face):
    t = TopologyExplorer(_face)
    edgeList = []
    for edge in t.edges():
        edgeList.append(edge)
    return edgeList
