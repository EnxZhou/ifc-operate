# 获取IFC格式中的元素几何数据

## 直接调用ifcopenshell
现在能通过ifcopenshell，获取ifc文件中的所有element

再通过ifcopenshell.geom获取这个element的所有顶点（vertex），边（edge），面（face）

但这获取的数据有些问题：
1. ifcopenshell获取的顶点、线、面数据，为了压缩数据，线与面是用顶点的序列号来表示的
2. face是通过三个点来表示的，如果是一个矩形，就分成两个面来表示

如果按照这种方式解析几何数据，一个六面体，会包含12个face，很难体现这个soild是一个六面体。
因此需要更好的几何数据表示方法

## 图形的技术栈
### B-rep格式

B-rep是一种三维图形模型，
[B-rep模型与mesh模型简单介绍](https://ntopology.com/blog/understanding-the-basics-of-b-reps-and-implicits/)

以下将B-rep模型简称为brep格式

如果采用brep格式，就需要将ifc转换为brep格式

### OpenCascade
OpenCascade是一个开源的图形内核，其就是用B-rep模型来表达图形

[pythonocc-core](https://github.com/tpaviot/pythonocc-core)
是封装了OpenCascade内核的Python开发工具

pythonocc的作者贡献了ifc的解析工具，
[github](https://github.com/tpaviot/pythonocc-contrib/blob/master/IFCViewer/ifc_viewer.py)

### FreeCAD
FreeCAD就是用的OpenCascade内核,支持打开ifc格式，
但打开的ifc格式，转换为brep格式，有点问题

FreeCAD也可以支持python开发，也有相似的转换方式，
[github](https://github.com/CyrilWaechter/pythoncvc.net/tree/master/IfcOpenShellSamples)

[read geom as mesh](https://pythoncvc.net/?p=822)

[read geom as brep](https://pythoncvc.net/?p=839)


#### 失败的办法
FreeCAD 支持的python版本：3.8.6

FreeCAD 安装后总是有问题，无法正常使用。

>import FreeCAD
 
总是提示 "no module named FreeCAD"
[解决方案](https://academy.ifcopenshell.org/posts/Importing%20Part%20module%20in%20ifcOpenShell-python/)

安装上文的解决方案，导入FreeCAD可以，
>import Part

还是不行，也尝试过用更高版本的FreeCAD，也尝试过不用conda，直接安装Python3.8.6，结果均失败

在FreeCAD自己的Python控制台，发现有个问题，直接 import Part， 
也是失败的，必须先import FreeCAD，再import Part，[网上有说这是FreeCAD的bug](https://youtrack.jetbrains.com/issue/PY-12575/Cannot-access-FreeCAD-library-from-PyCharm)。

#### 可行的办法
可以在FreeCAD中直接加载编辑好的python文件，
创建一个空的宏（直接录制，关闭）
编辑宏，粘贴python脚本，即可执行

FreeCAD 可以直接打开ifc格式文件，然后选择模型层级，
直接导出为brep格式文件（不能选择project层级，不然brep就是空的）。

即使是直接在FreeCAD中执行转换脚本，
也无法直接将ifc文件直接转换为brep文件，
应该就是因为Tekla导出的ifc文件，加了一层project，
导致ifcopenshell（FreeCAD内置了该module）无法正常处理ifc文件


### pythonocc
是可以打开ifc文件