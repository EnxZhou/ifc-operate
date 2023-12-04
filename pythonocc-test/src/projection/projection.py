# coding=utf-8
import numpy as np
from OCC.Core import GeomAPI, GeomConvert
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakePolygon, \
    BRepBuilderAPI_MakeFace
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepProj import BRepProj_Projection
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRep import BRep_Tool_Surface, BRep_Tool_Pnt
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods, TopoDS_Shape, TopoDS_Face, topods_Face, TopoDS_Vertex
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec, gp_Dir
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GeomLProp import GeomLProp_SurfaceTool
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.AIS import AIS_Shaded
from OCC.Display.SimpleGui import init_display
from OCC.Extend.DataExchange import read_step_file_with_names_colors
from OCC.Extend.TopologyUtils import TopologyExplorer
from scipy.spatial import ConvexHull

from FaceUtil import get_face_kind_location_axis


def get_face_feature(_face: TopoDS_Face):
    props = GProp_GProps()
    brepgprop_SurfaceProperties(_face, props)

    # 面积
    face_mass = props.Mass()
    centreOfMass = props.CentreOfMass()
    # 重心坐标
    face_centreOfMassX = centreOfMass.X()
    face_centreOfMassY = centreOfMass.Y()
    face_centreOfMassZ = centreOfMass.Z()

    # 获取面的类型、法线坐标、法线方向
    kind_location_axis = get_face_kind_location_axis(_face)
    location = kind_location_axis[1]
    if not isinstance(location, gp_Pnt):
        raise ValueError
    axis = kind_location_axis[2]
    if not isinstance(axis, gp_Dir):
        raise ValueError

    return face_mass, location, axis


def get_max_face_from_solid(_solid: TopoDS_Shape):
    t = TopologyExplorer(_solid)
    max_face = None
    max_location = None
    max_axis = None
    max_mass = 0.0
    index = 0
    for face in t.faces():
        index = index + 1
        mass, location, axis = get_face_feature(face)
        if mass > max_mass:
            max_mass = mass
            max_face = face
            max_location = location
            max_axis = location

    return max_face, max_location, max_axis


def visualize_shapes_with_colors(shapes):
    display, start_display, _, _ = init_display()

    for i, shape in enumerate(shapes):
        r = float(i) / len(shapes)
        g = 0.2  # 调整绿色通道的值
        b = 1.0 - float(i) / len(shapes)
        print(f"r:{r},g:{g},b:{b}")
        color = Quantity_Color(r,g,b, Quantity_TOC_RGB)
        display.DisplayShape(shape, color=color)

    display.FitAll()
    start_display()


def read_step_file_with_colors2(step_filename):
    stp_name_colors = read_step_file_with_names_colors(step_filename)
    solid_list = []
    for shp in stp_name_colors:
        solid_list.append(shp)
    return solid_list

def main():
    # 初始化显示
    display, start_display, add_menu, add_function_to_menu = init_display()

    file_name = "../../data/step/scw-C3-JD-31-DB-252-freecad.step"
    stp_name_colors = read_step_file_with_names_colors(file_name)
    solid_list = []
    solid_name_list = []
    face_list = []
    for shp in stp_name_colors:
        solid_list.append(shp)
        solid_name_list.append(stp_name_colors[shp][0])

    # 创建一个主要形状和一个要投影的形状
    main_shape = solid_list[0]
    shape_to_project = solid_list[1]

    # 获取主要形状的表面
    main_surface, location, axis = get_max_face_from_solid(main_shape)
    # 获取TopoDS_Face的几何表面
    face_surface = BRep_Tool_Surface(main_surface)  # 这将返回一个Handle_Geom_Surface对象

    # 转换为Geom_Surface
    proj_points = []  # 存储投影点的列表
    se = TopologyExplorer(shape_to_project)

    # visualize_shapes_with_colors(solid_list)
    display.DisplayShape(main_surface)
    display.DisplayShape(shape_to_project)
    coplanar_points=[]
    # 遍历实体上的点，计算投影点
    for vertex in se.vertices():
        point = BRep_Tool_Pnt(vertex)
        # 创建一个点到平面的投影
        proj_point = GeomAPI.GeomAPI_ProjectPointOnSurf(point, face_surface)
        projected_vertex = proj_point.NearestPoint()
        proj_points.append(projected_vertex)

        # 获取投影点的坐标
        x = projected_vertex.X()
        y = projected_vertex.Y()
        z = projected_vertex.Z()
        coplanar_points.append((x,y,z))
        display.DisplayShape(projected_vertex)

    # 将共面点坐标整理为二维数组
    points_array = np.array(coplanar_points)[:,:2]

    # 使用基于平面的凸包算法计算共面点的凸包
    hull = ConvexHull(points_array)

    # 创建一个构建块来构建投影面
    polygon_builder = BRepBuilderAPI_MakePolygon()
    for projected_coords in proj_points:
        polygon_builder.Add(projected_coords)

    # 创建投影面
    projected_face = BRepBuilderAPI_MakeFace(polygon_builder.Wire()).Face()

    # 显示投影面
    display.DisplayShape(projected_face)

    # 显示绘图
    display.FitAll()
    start_display()


if __name__ == '__main__':
    main()

