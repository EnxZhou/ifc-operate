#!/bin/python
import ifcopenshell
import ifcopenshell.geom
import multiprocessing

def VertStd(n: float):
    return round(n, 3)

# vert 顶点
class vert(object):
    def __init__(self,index:int, x: float, y: float, z: float):
        self.index = index
        self.x = x
        self.y = y
        self.z = z

    def getVert(self):
        return self.x, self.y, self.z


# edge 边
class edge(object):
    def getEdge(self):
        return ((self.start.x, self.start.y, self.start.z), (self.end.x, self.end.y, self.end.z))

    def genFromIndexVertList(self, vert_list: list, start_point_by_index: int, end_point_by_index: int):
        startPointByFloat = indexList2VertList(vert_list,start_point_by_index)
        endPointByFloat = indexList2VertList(vert_list,end_point_by_index),
        self.start = vert(start_point_by_index,startPointByFloat[0], startPointByFloat[1], startPointByFloat[2])
        self.end = vert(end_point_by_index,endPointByFloat[0], endPointByFloat[1], endPointByFloat[2])

def indexList2VertList(vert_list:list,index_list:list):
    res=[]
    for i in index_list:
        res.append(vert_list[i])
    return res


class TestClass:
    def test_indexList2VertList(self):
        vert_list=[[-0.0, 0.0, -0.001], [-0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.06, 0.0, -0.001]]
        a=indexList2VertList(vert_list,[0,1,2])
        assert a==[[-0.0, 0.0, -0.001], [-0.0, 0.0, 0.0], [0.06, 0.0, 0.0]]


class IFCElement(object):
    def __init__(self):
        self.guid = ""
        self.name = ""
        self.groupedVerts = []
        self.groupedEdges = []
        self.groupedFaces = []

    def vertFromIFC(self, verts):
        for i in range(0, len(verts), 3):
            self.groupedVerts.append([VertStd(verts[i]), VertStd(verts[i + 1]), VertStd(verts[i + 2])])

    def edgeFromIFC(self, edges):
        for i in range(0, len(edges), 2):
            self.groupedEdges.append([edges[i], edges[i + 1]])

    def setGuid(self, guid):
        self.guid = guid

    def setName(self, name):
        self.name = name


def loadDataSet():
    try:
        ifc_file = ifcopenshell.open('./example/segment-1.ifc')
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
                curElement.setGuid(shape.guid)
                # X Y Z of vertices in flattened list e.g. [v1x, v1y, v1z, v2x, v2y, v2z, ...]
                verts = shape.geometry.verts
                curElement.vertFromIFC(verts)
                edges = shape.geometry.edges
                # Indices of vertices per triangle face e.g. [f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...]
                faces = shape.geometry.faces
                # Material names and colour style information that are relevant to this shape
                materials = shape.geometry.materials
                # Indices of material applied per triangle face e.g. [f1m, f2m, ...]
                material_ids = shape.geometry.material_ids

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
                print(element)
                if not iterator.next():
                    break
        print(grouped_elements[0].groupedVerts)
    return grouped_verts, grouped_edges, grouped_faces


if __name__ == '__main__':
    # verts, edges, faces = loadDataSet()
    vert_list=[[-0.0, 0.0, -0.001], [-0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.06, 0.0, -0.001], [0.06, 0.2, 0.0], [0.06, 0.2, -0.001], [0.09, 0.2, 0.0], [0.09, 0.2, -0.001], [0.03, 0.35, 0.0], [0.03, 0.35, -0.001], [-0.03, 0.2, 0.0], [-0.03, 0.2, -0.001], [-0.0, 0.2, 0.0], [-0.0, 0.2, -0.001]]
    a=edge()
    a.genFromIndexVertList(vert_list,1,3)
    print(a.getEdge())