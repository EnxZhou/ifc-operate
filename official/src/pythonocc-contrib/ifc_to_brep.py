from __future__ import print_function

try:
    import ifcopenshell
    import ifcopenshell.geom.occ_utils as occ_utils
except ImportError:
    print("This example requires ifcopenshell for python. Please go to  http://ifcopenshell.org/python.html")
from OCC.Display.SimpleGui import init_display

import ifc_metadata

if __name__ == "__main__":
    # viewer settings
    settings = ifcopenshell.geom.settings()
    # settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)

    # Open the IFC file using IfcOpenShell
    filepath = "models/Wall.ifc"
    print("Opening IFC file %s ..." % filepath, end="")
    ifc_file = ifcopenshell.open(filepath)
    print("file opened.")
    # The geometric elements in an IFC file are the IfcProduct elements.
    # So these are opened and displayed.
    products = ifc_file.by_type("IfcProduct")
    metadata = ifc_metadata.metadata_dictionary(ifc_file)

    # First filter products to display
    # just keep the ones with a 3d representation
    products_to_display = []
    for product in products:
        if (product.is_a("IfcOpeningElement") or
             product.is_a("IfcSite") or product.is_a("IfcAnnotation")):
                continue
        if product.Representation is not None:
            shape = ifcopenshell.geom.create_shape(settings, product).geometry
            shape_gpXYZ = shape.Location().Transformation().TranslationPart()
            print(shape_gpXYZ.X(), shape_gpXYZ.Y(), shape_gpXYZ.Z())
            products_to_display.append(product)
    print("Products to display: %i" % len(products_to_display))
    # For every product a shape is created if the shape has a Representation.
    print("Traverse data with associated 3d geometry")
    idx = 0
    product_shapes = []
    brep_shapes = []
    for product in products_to_display:
        # display current product
        shape = ifcopenshell.geom.create_shape(settings, product).geometry
        product_shapes.append((product, shape))
        idx += 1
        print("\r[%i%%]Product: %s" % (int(idx*100/len(products_to_display)), product))
        print(metadata[product])

    # print(brep_shapes)
    # with open("models/brep_data","w") as file:
    #     file.write(brep_shapes[0])

