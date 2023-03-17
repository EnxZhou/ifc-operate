from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape, topods_Vertex
from OCC.Extend.DataExchange import read_step_file, read_step_file_with_names_colors
from OCC.Extend.TopologyUtils import TopologyExplorer

import FaceUtil
import SolidUtil


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


def step_model_explore2():
    solidList = SolidUtil.get_solids_from_step('segment-1.step')
    for solid in solidList:
        solidProps = SolidUtil.get_solid_feature(solid)
        print("solid mass:", solidProps.mass)
        print("solid centre mass:", solidProps.centreOfMassX, solidProps.centreOfMassY, solidProps.centreOfMassZ)
        print("solid max face prop:", solidProps.maxFaceFeature)
        print("solid max face max edge props:", solidProps.maxFaceFeature.maxEdgeFeature)

        faceList = FaceUtil.get_faces_from_solid(solid)
        print("-------------------")
        # for face in faceList:
        #     faceProps = FaceUtil.get_face_props(face)
        #     print(faceProps.kind, "axis direct:", (faceProps.axisDirectX, faceProps.axisDirectY, faceProps.axisDirectZ))
        #     print(faceProps.kind, "axis location:",
        #           (faceProps.axisLocationX, faceProps.axisLocationY, faceProps.axisLocationZ))


def get_name_from_step(file_name):
    stp_name_colors = read_step_file_with_names_colors(file_name)
    nameList = []
    index = 1
    for shp in stp_name_colors:
        solidProps = SolidUtil.get_solid_feature(shp)
        name, _ = stp_name_colors[shp]
        nameList.append(name)
        print("index: ", index,"name: ",name, "solid mass:", solidProps.mass)
        index=index+1

    return nameList


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # step_model_explore2()
    nameList = get_name_from_step('data/step/scw-C3-JD-31-DB-252-ifcConvert.stp')
    print("name_list: ", len(nameList))
