from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCC.Core.gp import gp_Pnt,gp_Dir
from OCC.Display.OCCViewer import rgb_color
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopoDS import TopoDS_Shape, topods_Face, topods_Vertex, TopoDS_Face
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_C0,
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
)

import numpy as np
from OCC.Extend.TopologyUtils import is_face, is_edge, TopologyExplorer
from OCC.Extend.DataExchange import read_step_file, read_step_file_with_names_colors
from OCC.Core.BRep import BRep_Tool

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


# 以上是通过点定义边，通过子级定义父级，但解析模型要反着来
# ---------------------------------------------------------------
class EdgeFeature:
    isStraight = True
    length = 0.0
    startX = 0.0
    startY = 0.0
    startZ = 0.0
    endX = 0.0
    endY = 0.0
    endZ = 0.0
    directX = 0.0
    directY = 0.0
    directZ = 0.0


class FaceFeature:
    kind = ""
    mass = 0.0
    centreOfMassX = 0.0
    centreOfMassY = 0.0
    centreOfMassZ = 0.0
    axisLocationX = 0.0
    axisLocationY = 0.0
    axisLocationZ = 0.0
    axisDirectX = 0.0
    axisDirectY = 0.0
    axisDirectZ = 0.0
    # 周长
    perimeter = 0.0
    maxEdgeFeature = EdgeFeature
    minEdgeFeature = EdgeFeature
    edgeLengthAverage = 0.0
    # 边长方差
    edgeLengthVariance = 0.0


def print_shape_mass(_shape):
    system = GProp_GProps()
    brepgprop_LinearProperties(_shape, system)
    print("shape linear mass: ", round(system.Mass(), 2))
    brepgprop_SurfaceProperties(_shape, system)
    print("shape surface mass: ", round(system.Mass(), 2))
    brepgprop_VolumeProperties(_shape, system)
    print("shape volume mass: ", round(system.Mass(), 2))
    centreOfMass = system.CentreOfMass()
    cx = round(centreOfMass.X(), 2)
    cy = round(centreOfMass.Y(), 2)
    cz = round(centreOfMass.Z(), 2)
    print("shape volume centre of mass: ", cx, cy, cz)


# 获取体的所有面
def get_face_from_solid(_solid):
    t = TopologyExplorer(_solid)
    faceList = []
    for face in t.faces():
        faceList.append(face)
    return faceList


# 获取面的属性
def get_face_props(_face: TopoDS_Face):
    props = GProp_GProps()
    brepgprop_SurfaceProperties(_face, props)

    faceFeatureReturn = FaceFeature

    faceFeatureReturn.mass=props.Mass()
    centreOfMass = props.CentreOfMass()
    faceFeatureReturn.centreOfMassX=centreOfMass.X()
    faceFeatureReturn.centreOfMassY=centreOfMass.Y()
    faceFeatureReturn.centreOfMassZ=centreOfMass.Z()

    # 获取面的类型、法线坐标、法线方向
    kind_location_axis=get_face_kind_location_axis(_face)
    faceFeatureReturn.kind = kind_location_axis[0]
    location = kind_location_axis[1]
    if location == gp_Pnt:
        faceFeatureReturn.axisLocationX = location.X()
        faceFeatureReturn.axisLocationY = location.Y()
        faceFeatureReturn.axisLocationZ = location.Z()
    axis = kind_location_axis[2]
    if axis == gp_Dir:
        faceFeatureReturn.axisDirectX = axis.X()
        faceFeatureReturn.axisDirectY = axis.Y()
        faceFeatureReturn.axisDirectZ = axis.Z()

    return faceFeatureReturn


def get_face_kind_location_axis(topods_face):
    """returns True if the TopoDS_Face is a planar surface"""
    if not isinstance(topods_face, TopoDS_Face):
        return "Not a face", None, None
    surf = BRepAdaptor_Surface(topods_face, True)
    surf_type = surf.GetType()
    if surf_type == GeomAbs_Plane:
        kind = "Plane"
        # look for the properties of the plane
        # first get the related gp_Pln
        gp_pln = surf.Plane()
        location = gp_pln.Location()  # a point of the plane
        normal = gp_pln.Axis().Direction()  # the plane normal
        tuple_to_return = (kind, location, normal)
    elif surf_type == GeomAbs_Cylinder:
        kind = "Cylinder"
        # look for the properties of the cylinder
        # first get the related gp_Cyl
        gp_cyl = surf.Cylinder()
        location = gp_cyl.Location()  # a point of the axis
        axis = gp_cyl.Axis().Direction()  # the cylinder axis
        # then export location and normal to the console output
        tuple_to_return = (kind, location, axis)
    elif surf_type == GeomAbs_Cone:
        kind = "Cone"
        tuple_to_return = (kind, None, None)
    elif surf_type == GeomAbs_Sphere:
        kind = "Sphere"
        tuple_to_return = (kind, None, None)
    elif surf_type == GeomAbs_Torus:
        kind = "Torus"
        tuple_to_return = (kind, None, None)
    elif surf_type == GeomAbs_BezierSurface:
        kind = "Bezier"
        tuple_to_return = (kind, None, None)
    elif surf_type == GeomAbs_BSplineSurface:
        kind = "BSpline"
        tuple_to_return = (kind, None, None)
    elif surf_type == GeomAbs_SurfaceOfRevolution:
        kind = "Revolution"
        tuple_to_return = (kind, None, None)
    elif surf_type == GeomAbs_SurfaceOfExtrusion:
        kind = "Extrusion"
        tuple_to_return = (kind, None, None)
    elif surf_type == GeomAbs_OffsetSurface:
        kind = "Offset"
        tuple_to_return = (kind, None, None)
    elif surf_type == GeomAbs_OtherSurface:
        kind = "Other"
        tuple_to_return = (kind, None, None)
    else:
        tuple_to_return = ("Unknown", None, None)

    return tuple_to_return


# 获取面的所有边
def get_edge_from_face(_face: TopoDS_Face):
    t = TopologyExplorer(_face)
    edgeList = []
    for edge in t.faces():
        edgeList.append(edge)
    return edgeList


def shape_faces_surface(the_shape):
    """Compute the surface of each face of a shape"""
    # then loop over faces
    t = TopologyExplorer(the_shape)
    props = GProp_GProps()
    faceList = []
    faceCount = t.number_of_faces()
    print("face count: ", faceCount)
    for face in t.faces():
        brepgprop_SurfaceProperties(face, props)
        # 面的面积
        face_surf = props.Mass()
        print("face mass: ", face_surf)
        faceList.append(round(face_surf, 2))
    return faceList


def shape_edges_length(the_shape):
    t = TopologyExplorer(the_shape)
    props = GProp_GProps()
    edgeList = []
    edgeCount = t.number_of_edges()
    print("edge count: ", edgeCount)
    for edge in t.edges():
        brepgprop_LinearProperties(edge, props)
        edge_len = props.Mass()
        edgeList.append(round(edge_len, 2))
    return edgeList


def step_model_explore():
    stpshp = read_step_file('segment-1.step')
    stp_names_colors = read_step_file_with_names_colors('segment-1.step')
    for shp in stp_names_colors:
        name, _ = stp_names_colors[shp]
        print("name:", name)

    # step文件生成的shape
    stpshp2 = TopologyExplorer(stpshp)
    # 获取文件中的体
    solidList = stpshp2.solids()
    idx = 0
    for solid in solidList:
        idx = idx + 1
        print("index: ", idx)
        print_shape_mass(solid)
        # print_face_of_shape(solid)
        faceList = shape_faces_surface(solid)
        edgeList = shape_edges_length(solid)
        print()

    print("shape count:", idx)


# 体的全部顶点
def shape_points(the_shape):
    t = TopologyExplorer(the_shape)
    pointCount = t.number_of_vertices()
    print("point count: ", pointCount)
    anVertexExplorer = TopExp_Explorer(the_shape, TopAbs_VERTEX)
    vertex = []
    while anVertexExplorer.More():
        anVertex = topods_Vertex(anVertexExplorer.Current())
        aPnt = BRep_Tool.Pnt(anVertex)

        vertex.append(aPnt)
        anVertexExplorer.Next()

    pnts = []
    for v in vertex:
        coordinate = (v.X(), v.Y(), v.Z())
        if coordinate not in pnts:
            pnts.append(coordinate)

    return pnts


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stpshp = read_step_file('segment-1.step')

    # step文件生成的shape
    stpshp2 = TopologyExplorer(stpshp)
    # 获取文件中的体
    solidList = stpshp2.solids()
    for solid in solidList:
        faceList=get_face_from_solid(solid)
        faceProps=get_face_props(faceList[1])
        print(faceProps.kind,faceProps.centreOfMassX,faceProps.axisLocationX)