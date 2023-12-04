import ifcopenshell as ifc

# 加载IFC文件
ifc_file_path = 'example/Wall.ifc'
model = ifc.open(ifc_file_path)


# 获取体对象的几何信息
def get_body_geometry(ifc_object):
    if hasattr(ifc_object, 'Representation'):
        representation = ifc_object.Representation

        for item in representation.Items:
            if isinstance(item, ifc.entity_instance):
                vertices = item.geometry.verts
                return vertices

    return None

# 计算体到体的最短距离
def calculate_body_to_body_distance(body1, body2):
    vertices1 = get_body_geometry(body1)
    vertices2 = get_body_geometry(body2)

    if vertices1 is None or vertices2 is None:
        return None

    # 使用scipy库中的cdist函数计算两组点之间的距离
    distances = cdist(vertices1, vertices2)

    # 返回最短距离
    min_distance = np.min(distances)
    return min_distance

# 遍历IFC文件中的对象并计算体到体的最短距离
for ifc_object1 in model.by_type('IfcProduct'):
    for ifc_object2 in model.by_type('IfcProduct'):
        if ifc_object1 != ifc_object2 and ifc_object1.is_a('IfcSpace') and ifc_object2.is_a('IfcSpace'):
            distance = calculate_body_to_body_distance(ifc_object1, ifc_object2)
            if distance is not None:
                print(f"Distance between {ifc_object1.Name} and {ifc_object2.Name}: {distance}")
