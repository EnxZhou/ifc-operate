# coding=utf-8

# 这个文件的作用：
# 虽然一个桥梁项目中，节段数量很多，但是大多数节段都是相似的，几乎一模一样。
# 所以一个桥梁项目只能拿一个节段作为训练数据，这样训练数据就大大减少了，
# 一个节段也就能分解成十几个板单元，三个桥梁项目，也就50个板单元模型。
# 训练数据少，很容易出现过拟合，于是需要增加训练集数据，模型摄动是一个好方法。
# 这个文件就是将模型进行旋转、平移、缩放等操作

import math
import os

from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_AsIs
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_LinearProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Quaternion, gp_Pnt, gp_Ax1, gp_Dir
from OCC.Display.SimpleGui import init_display
from OCC.Extend.DataExchange import read_step_file, read_step_file_with_names_colors
from OCC.Core.BRepTools import breptools_Write
from OCC.Core.STEPControl import STEPControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone

from collections import namedtuple


# Point3D = namedtuple('Point3D', ['x', 'y', 'z'])
class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"{int(self.x)}{int(self.y)}{int(self.z)}"


def translate_shape(shape, x, y, z):
    vector_translation = gp_Vec(x, y, z)
    translation = gp_Trsf()
    translation.SetTranslation(vector_translation)
    return BRepBuilderAPI_Transform(shape, translation).Shape()


def rotate_shape(shape, angle_degrees, axis: Point3D, point: Point3D):
    point_rotation = gp_Pnt(point.x, point.y, point.z)
    axis_rotation = gp_Ax1(point_rotation, gp_Dir(axis.x, axis.y, axis.z))  # Use gp_Ax1 to define rotation axis
    rotation = gp_Trsf()
    rotation.SetRotation(axis_rotation, angle_degrees)  # Use the axis_rotation in the SetRotation function
    return BRepBuilderAPI_Transform(shape, rotation).Shape()


def scale_shape(shape, scaling_factor, point: Point3D):
    point_scaling = gp_Pnt(point.x, point.y, point.z)
    scaling = gp_Trsf()
    scaling.SetScale(point_scaling, scaling_factor)
    return BRepBuilderAPI_Transform(shape, scaling).Shape()


def display_shapes(*shapes):
    display, start_display, _, _ = init_display()
    for shape in shapes:
        display.DisplayShape(shape, update=True)
    start_display()


def calculate_model_centre(shape):
    props = GProp_GProps()
    # 创建计算几何属性对象
    brepgprop_VolumeProperties(shape, props)
    # 计算重心
    centre_of_mass = props.CentreOfMass()
    return Point3D(x=centre_of_mass.X(), y=centre_of_mass.Y(), z=centre_of_mass.Z())


def save_shape(shape, file_path, file_format="STEP"):
    if file_format.upper() == "STEP":
        step_writer = STEPControl_Writer()
        step_writer.Transfer(shape, STEPControl_AsIs)
        status = step_writer.Write(file_path)
        if status == IFSelect_RetDone:
            print(f"Shape saved as STEP file: {file_path}")
        else:
            print("Failed to save as STEP file.")
    else:
        breptools_Write(shape, file_path)
        print(f"Shape saved as BREP file: {file_path}")


def degrees_to_radians(degrees):
    radians = degrees * (math.pi / 180)
    return radians


# Helper function to generate different combinations of rotations and scaling factors
def generate_transformations():
    rotations = [-15, 15]  # Degrees to rotate
    rotation_axises = [Point3D(x=1, y=0, z=0), Point3D(x=0, y=1, z=0), Point3D(x=0, y=0, z=1)]
    # rotations = [10]  # Degrees to rotate
    # rotation_axises = [Point3D(x=1, y=0, z=0)]
    scaling_factors = [0.8, 1.2]  # Scaling factors

    transformations = []

    for rotation in rotations:
        for rotation_axis in rotation_axises:
            for scaling_factor in scaling_factors:
                transformations.append((rotation, rotation_axis, scaling_factor))

    return transformations


# Function to perform batch transformations
def batch_transformations_and_save(shape, save_name: str, centre_of_mass, save_path: str):
    """
    Perform batch transformations on a shape and save the transformed shapes.

    Parameters:
        - shape (TopoDS_Shape): The shape to be transformed.
        - centre_of_mass (Point3D): The centre of mass of the shape.
        - save_path (str): The path to save the transformed shapes.

    Returns:
        - list of str: The list of filenames of the saved transformed shapes.
    """
    transformed_shapes_filenames = []

    # Generate different combinations of rotations and scaling factors
    transformations = generate_transformations()

    # Apply transformations and save the transformed shapes
    for idx, (rotation, rotation_axis, scaling_factor) in enumerate(transformations):
        # Rotate the shape
        rotated_shape = rotate_shape(shape, degrees_to_radians(rotation),
                                     rotation_axis, centre_of_mass)

        # Scale the rotated shape
        scaled_shape = scale_shape(rotated_shape, scaling_factor, centre_of_mass)

        # Generate filename based on the parameters
        filename = f"{save_name}_{idx}_rotation_{rotation}_axis_{rotation_axis}_scaling_{scaling_factor}.step"
        full_save_path = os.path.join(save_path, filename)

        # Save the transformed shape to a STEP file
        save_shape(scaled_shape, full_save_path, "STEP")

        transformed_shapes_filenames.append(filename)

    return transformed_shapes_filenames


def try1():
    # 读取STEP格式文件
    shape = read_step_file("../../data/step/Z4-SS9-1-v2_1.step")

    centre_of_mass = calculate_model_centre(shape)
    # 平移、旋转、缩放
    translated_shape = translate_shape(shape, 10000.0, 1.0, 1.0)
    rotated_shape = rotate_shape(shape, degrees_to_radians(1), Point3D(x=0, y=0, z=1), centre_of_mass)
    scaled_shape = scale_shape(shape, 2, centre_of_mass)

    # 显示原始模型和变换后的模型
    # display_shapes(shape, translated_shape, rotated_shape, scaled_shape)
    # display_shapes(shape, translated_shape, scaled_shape)
    # display_shapes(shape, rotated_shape)
    # save_shape(rotated_shape, "../../data/step/Z4-SS9-1-v2_1_ro.step", "STEP")


if __name__ == "__main__":
    # 读取STEP格式文件
    # file_name = "C3-JD-27_v2.1_1"
    # file_name = "61-v2.1_1"
    file_name = "X11-SE1-v2.1_1"
    shape = read_step_file("../../data/step/"+file_name+".step")
    centre_of_mass = calculate_model_centre(shape)
    batch_transformations_and_save(shape, file_name, centre_of_mass, "../../data/step/")
