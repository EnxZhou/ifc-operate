利用pythonocc解析step模型之间是否相邻
从而生成图结构，每个solid和face都作为一个图的节点
相邻的两个solid或face之间，添加一条边

# 2023.8.4
将模型中，零件是否相邻形成边
先对node进行了分类，分成main和sub，
main表示主零件
sub表示次零件
按照一下条件进行归类：
sub node与其最近的main node，标为同一类
同为main node，direct X相同的标为同一类
如果所有最近main node都属于同一个group，直接属于这个类
如果某group占了最近main node多数（50%），也认为是同一类

# 2023.8.18
之前的sub node与main node归类工作，有了更好的办法
因为solid feature中包含max face的参数，需要该面的法线参数
max_face_axis_location_x
max_face_axis_location_y
max_face_axis_location_z
max_face_axis_direct_x
max_face_axis_direct_y
max_face_axis_direct_z
根据这六个参数，就可以确定一个三维面（但这个面没有边界，无限大）
再提取sub node的重心坐标
centre_of_mass_x
centre_of_mass_y
centre_of_mass_z

遍历判断每个sub node距离每个main node的max face的距离
选出若干最近的main node（这个选取方案有待优化，到底选容差多少的）
将这些最近的main node组成nearest main node
再判断nearest main node中的每个main node与sub node在graph中的距离
距离最近的，就是sub node所属的main node

选择nearest main node的容差，可能需要一种项目，一个特定参数
厦门二东-60mm
沙埕湾-20mm