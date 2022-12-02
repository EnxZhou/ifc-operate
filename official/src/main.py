#!/bin/python
import ifcopenshell
import ifcopenshell.geom
# from ifcopenshell import geom
import multiprocessing
import os

import parseIFC

def VertStd(n: float):
    return round(n, 3)


class Coord(object):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "x: %s, y: %s, z: %s" % (self.x, self.y, self.z)


# Vertex 顶点
class Vertex(object):
    def __init__(self, x: float, y: float, z: float):
        self.point = Coord(x, y, z)

    def get_vertex(self):
        return self.point


# Edge 边
class Edge(object):
    def __init__(self, start: Coord, end: Coord):
        self.points = [start, end]

    def get_edge(self):
        return self.points


# Face 面
class Face(object):
    def __init__(self, one: Coord, two: Coord, three: Coord):
        self.points = [one, two, three]

    def get_face(self):
        return self.points


class Solid(object):
    def __init__(self):
        pass


def indexList2VertexList(vertex_list: list, index_list: list):
    res = []
    for i in range(0, len(index_list)):
        res.append(vertex_list[i])
    return res


class IFCElement(object):
    def __init__(self):
        self.guid = ""
        self.name = ""
        self.IFCVertices = []
        self.IFCEdges = []
        self.IFCFaces = []

    def vert_from_ifc(self, verts):
        for i in range(0, len(verts), 3):
            self.IFCVertices.append([VertStd(verts[i]), VertStd(verts[i + 1]), VertStd(verts[i + 2])])

    def edge_from_ifc(self, edges):
        for i in range(0, len(edges), 2):
            self.IFCEdges.append([edges[i], edges[i + 1]])

    def face_from_ifc(self, faces):
        for i in range(0, len(faces), 3):
            self.IFCFaces.append([faces[i], faces[i + 1], faces[i + 2]])

    def set_guid(self, guid):
        self.guid = guid

    def set_name(self, name):
        self.name = name


class IFCFile(object):
    def __init__(self):
        self.name = ""
        self.path = ""
        self.file = ifcopenshell.file

    def open(self, path: str):
        try:
            ifcFile = ifcopenshell.open(path)
            self.path = path
            self.file = ifcFile
            self.name = os.path.basename(path)
        except IOError:
            print(ifcopenshell.get_log())
        else:
            print("open ifc success")

    def get_elements(self):
        settings = ifcopenshell.geom.settings
        iterator = ifcopenshell.geom.iterator(settings, self.file, multiprocessing.cpu_count())
        elements = []
        if iterator.initialize():
            while True:
                cur_element = IFCElement()
                shape = iterator.get()
                elements.append(self.file.by_guid(shape.guid))
        return elements


def loadDataSet():
    try:
        ifc_file = ifcopenshell.open('../example/segment-1.ifc')
    except:
        print(ifcopenshell.get_log())
    else:
        settings = ifcopenshell.geom.settings()
        iterator = ifcopenshell.geom.iterator(settings, ifc_file, multiprocessing.cpu_count())
        grouped_elements = []
        if iterator.initialize():
            while True:
                curElement = IFCElement()
                shape = iterator.get()
                element = ifc_file.by_guid(shape.guid)
                curElement.set_guid(shape.guid)
                # X Y Z of vertices in flattened list e.g. [v1x, v1y, v1z, v2x, v2y, v2z, ...]
                verts = shape.geometry.verts
                curElement.vert_from_ifc(verts)
                edges = shape.geometry.edges
                # Indices of vertices per triangle face e.g. [f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...]
                faces = shape.geometry.faces
                # Material names and colour style information that are relevant to this shape
                # materials = shape.geometry.materials
                # Indices of material applied per triangle face e.g. [f1m, f2m, ...]
                # material_ids = shape.geometry.material_ids

                grouped_elements.append(curElement)
                # Since the lists are flattened, you may prefer to group them per face like so depending on your geometry kernel
                grouped_verts = []
                grouped_edges = []
                grouped_faces = []
                for i in range(0, len(verts), 3):
                    grouped_verts.append([VertStd(verts[i]), VertStd(verts[i + 1]), VertStd(verts[i + 2])])
                for i in range(0, len(edges), 2):
                    grouped_edges.append([edges[i], edges[i + 1]])
                for i in range(0, len(faces), 3):
                    grouped_faces.append([faces[i], faces[i + 1], faces[i + 2]])
                if not iterator.next():
                    break
        print(grouped_elements[0].IFCVertices)
    return grouped_verts, grouped_edges, grouped_faces



if __name__ == '__main__':
    verts, edges, faces = loadDataSet()
    # ifc_file=IFCFile.open('../example/segment-1.ifc')
