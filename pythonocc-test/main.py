from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCC.Core.gp import gp_Pnt
from OCC.Display.OCCViewer import rgb_color
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.BRep import BRep_Builder
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
from OCC.Extend.TopologyUtils import is_face, is_edge, TopologyExplorer
from OCC.Extend.DataExchange import read_step_file, read_step_file_with_names_colors


def get_face(_shape):
    """return the faces from `_shape`

    :param _shape: TopoDS_Shape, or a subclass like TopoDS_Solid
    :return: a list of faces found in `_shape`
    """
    topExp = TopExp_Explorer()
    topExp.Init(_shape, TopAbs_FACE)
    _faces = []

    while topExp.More():
        fc = topods_Face(topExp.Current())
        _faces.append(fc)
        topExp.Next()

    return _faces


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


def get_vert(_shape):
    topExp = TopExp_Explorer()
    topExp.Init(_shape, TopAbs_VERTEX)
    _verts = []

    while topExp.More():
        ve = topods_Vertex(topExp.Current())
        _verts.append(ve)
        topExp.Next()

    return _verts


# https://github.com/tpaviot/pythonocc-demos/blob/master/examples/core_shape_properties.py
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
        face_surf = props.Mass()
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


def measure_shape_mass_center_of_gravity(shape):
    """Returns the shape center of gravity
    Returns a gp_Pnt if requested (set as_Pnt to True)
    or a list of 3 coordinates, by default."""
    inertia_props = GProp_GProps()
    if is_edge(shape):
        brepgprop_LinearProperties(shape, inertia_props)
        mass_property = "Length"
    elif is_face(shape):
        brepgprop_SurfaceProperties(shape, inertia_props)
        mass_property = "Area"
    else:
        brepgprop_VolumeProperties(shape, inertia_props)
        mass_property = "Volume"
    cog = inertia_props.CentreOfMass()
    mass = inertia_props.Mass()
    return cog, mass, mass_property


def recognize_face(a_face):
    """Takes a TopoDS shape and tries to identify its nature
    whether it is a plane a cylinder a torus etc.
    if a plane, returns the normal
    if a cylinder, returns the radius
    """
    surf = BRepAdaptor_Surface(a_face, True)
    surf_type = surf.GetType()
    if surf_type == GeomAbs_Plane:
        print("--> plane")
        # look for the properties of the plane
        # first get the related gp_Pln
        gp_pln = surf.Plane()
        location = gp_pln.Location()  # a point of the plane
        normal = gp_pln.Axis().Direction()  # the plane normal
        # then export location and normal to the console output
        print(
            "--> Location (global coordinates)",
            location.X(),
            location.Y(),
            location.Z(),
        )
        print("--> Normal (global coordinates)", normal.X(), normal.Y(), normal.Z())
    elif surf_type == GeomAbs_Cylinder:
        print("--> cylinder")
        # look for the properties of the cylinder
        # first get the related gp_Cyl
        gp_cyl = surf.Cylinder()
        location = gp_cyl.Location()  # a point of the axis
        axis = gp_cyl.Axis().Direction()  # the cylinder axis
        # then export location and normal to the console output
        print(
            "--> Location (global coordinates)",
            location.X(),
            location.Y(),
            location.Z(),
        )
        print("--> Axis (global coordinates)", axis.X(), axis.Y(), axis.Z())
    else:
        # TODO there are plenty other type that can be checked
        # see documentation for the BRepAdaptor class
        # https://www.opencascade.com/doc/occt-6.9.1/refman/html/class_b_rep_adaptor___surface.html
        print("not implemented")


# https://github.com/tpaviot/pythonocc-core/blob/master/src/Extend/ShapeFactory.py
def recognize_face2(topods_face):
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


def print_face_property(tuple_of_prep):
    print("-->", tuple_of_prep[0])
    location = tuple_of_prep[1]
    print(
        "--> Location (global coordinates)",
        location.X(),
        location.Y(),
        location.Z(),
    )
    normal = tuple_of_prep[2]
    print("--> Normal (global coordinates)", normal.X(), normal.Y(), normal.Z())


def print_face_of_shape(_shape):
    read_face = get_face(_shape)
    for f in read_face:
        res = recognize_face2(f)
        print_face_property(res)


def step_model_explore():
    stpshp = read_step_file('segment-1.step')
    stp_names_colors = read_step_file_with_names_colors('segment-1.step')
    for shp in stp_names_colors:
        name, _ = stp_names_colors[shp]
        print("name:", name)

    stpshp2 = TopologyExplorer(stpshp)
    solidList=stpshp2.solids()
    idx = 0
    for solid in solidList:
        idx = idx + 1
        print("index: ",idx)
        print_shape_mass(solid)
        # print_face_of_shape(solid)
        faceList = shape_faces_surface(solid)
        edgeList = shape_edges_length(solid)
        print()

    print("shape count:", idx)


def brep_model_explore():
    read_shape = TopoDS_Shape()
    builder = BRep_Builder()
    breptools_Read(read_shape, 'segment-1.brep', builder)
    print_shape_mass(read_shape)

    faceList = shape_faces_surface(read_shape)
    edgeList = shape_edges_length(read_shape)
    print("face list:", faceList)
    print("edge list:", edgeList)
    print_face_of_shape(read_shape)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    step_model_explore()
    # display, start_display, add_menu, add_function_to_menu = init_display()
    # display.DisplayShape(read_face, update=True)
    # start_display()
