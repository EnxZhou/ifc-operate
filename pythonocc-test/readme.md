## 模型转换

### Tekla模型->IFC
Tekla可以直接导出ifc文件

### IFC->B-rep
FreeCAD可以打开ifc文件
必须选择具体的模型，再导出成brep文件
如果直接选择project导出，缺少具体的几何模型

### IFC->STEP
step文件和IFC格式比较相似（IFC格式是根据STEP格式演进的）
可以直接用FreeCAD打开模型，导出为STEP格式文件
step文件的优点，几乎保留了ifc文件的全部，包括零件名称
好像还是没法直接将零件名称和shape关联起来,
但可以直接读取step文件中的零件名称及颜色，顺序和FreeCAD的文件顺序是一致的，
可以判断一下，如果顺序和获取shape也一致，就解决了零件名称和shape关联的问题

## 模型特征提取

### shape的特征
包括shape的体积、表面积、周长
`system = GProp_GProps()
    brepgprop_LinearProperties(_shape, system)
    print("shape linear mass: ", system.Mass())
    brepgprop_SurfaceProperties(_shape, system)
    print("shape surface mass: ", system.Mass())
    brepgprop_VolumeProperties(_shape, system)
    print("shape volume mass: ", system.Mass())
    centreOfMass = system.CentreOfMass()
    print("shape volume centre of mass: ", centreOfMass.X(), centreOfMass.Y(), centreOfMass.Z())`

但获取的周长，是实际周长的2倍

### face的特征
包括面积、重心、法线方向
`t = TopologyExplorer(the_shape)
    for face in t.faces():`
重点是用TopologyExplorer遍历shape的各个子级（面、线）
进而获取面、线的各项数据