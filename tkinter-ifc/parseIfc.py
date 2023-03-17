#!/bin/python

import ifcopenshell as ifc


class IfcFile:
    file = ifc.file
    instances = []

    def __init__(self, path):
        self.file = ifc.open(path)

    def get_instance_by_type(self, filter_type):
        self.instances = self.file.by_type(filter_type)

    def update_guid_name(self):
        self.get_instance_by_type("IfcBuildingElement")
        for inst in self.instances:
            inst.Name = inst.Name + "_" + inst.Tag

    def save_file(self, new_file_name):
        self.file.write(new_file_name)