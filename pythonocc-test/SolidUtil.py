from OCC.Core import BRepBuilderAPI, BRepGProp, TopoDS

from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_LinearProperties
import FaceUtil
from OCC.Extend.DataExchange import read_step_file, read_step_file_with_names_colors
from OCC.Extend.TopologyUtils import is_face, is_edge, TopologyExplorer


# 体特征
class SolidFeature:
    mass = 0.0
    centreOfMassX = 0.0
    centreOfMassY = 0.0
    centreOfMassZ = 0.0
    # 表面积
    surfaceArea = 0.0
    # 最大面特征
    maxFaceFeature = FaceUtil.FaceFeature
    # 最小面特征
    minFaceFeature = FaceUtil.FaceFeature
    # 面积平均值
    faceMassAverage = 0.0
    # 面积方差
    faceMassVariance = 0.0
    # 边数量
    edgeCount = 0
    # 边长总和
    edgeLenSum = 0.0


def get_solids_from_step(file_path):
    stpshp = read_step_file(file_path)

    # step文件生成的shape
    stpshp2 = TopologyExplorer(stpshp)
    # 获取文件中的体
    solidList = stpshp2.solids()
    return solidList


def get_solid_feature(_solid):
    props = GProp_GProps()
    # 创建计算几何属性对象
    brepgprop_VolumeProperties(_solid, props)
    # 计算重量和重心
    mass = props.Mass()
    centre_of_mass = props.CentreOfMass()
    # 获取表面积
    face_list = FaceUtil.get_faces_from_solid(_solid)
    surface_area = 0.0

    # 创建最大面和最小面的变量
    max_face = None
    max_area = 0.0
    min_face = None
    min_area = float('inf')
    face_masses = []

    # 遍历 Solid 中的每个面，获取面积最大和最小的面
    for face in face_list:
        area = FaceUtil.get_face_props(face).mass
        face_masses.append(area)
        if area > max_area:
            max_area = area
            max_face = face
        if area < min_area:
            min_area = area
            min_face = face

    # 计算面积平均值和方差
    surface_area = sum(face_masses)
    face_mass_average = sum(face_masses) / len(face_masses)
    face_mass_variance = sum([(m - face_mass_average) ** 2 for m in face_masses]) / len(face_masses)

    # 创建 SolidFeature 对象
    feature = SolidFeature()
    feature.mass = mass
    feature.centreOfMassX = centre_of_mass.X()
    feature.centreOfMassY = centre_of_mass.Y()
    feature.centreOfMassZ = centre_of_mass.Z()
    feature.surfaceArea = surface_area
    feature.maxFaceFeature = FaceUtil.get_face_props(max_face)
    feature.minFaceFeature = FaceUtil.get_face_props(min_face)
    feature.faceMassAverage = face_mass_average
    feature.faceMassVariance = face_mass_variance
    feature.edgeLenSum=get_solid_edge_len_sum(_solid)
    feature.edgeCount=get_solid_edge_count(_solid)

    return feature


# 获取边长总和
def get_solid_edge_len_sum(_solid):
    t = TopologyExplorer(_solid)
    props = GProp_GProps()
    edgeLenSum = 0.0
    for edge in t.edges():
        brepgprop_LinearProperties(edge, props)
        edge_len = props.Mass()
        edgeLenSum += edge_len
    return edgeLenSum


# 获取边数量
def get_solid_edge_count(_solid):
    t = TopologyExplorer(_solid)
    edgeCount = t.number_of_edges()
    return edgeCount
