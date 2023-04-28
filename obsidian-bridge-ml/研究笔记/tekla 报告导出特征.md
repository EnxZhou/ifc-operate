# 特征提取


所有特征如下：

| 字段名称       | 数据类型    | 描述       |
|------------|---------|----------|
| guid       | string  | 唯一标识     |
| part_pos   | string  | 零件编号     |
| name       | string  | 名称       |
| weight     | numeric | 重量       |
| area_gross | numeric | 毛面积      |
| area       | numeric | 面积       |
| area_net   | numeric | 净面积      |
| a_net_xy   | numeric | 净面积XY    |
| a_net_xz   | numeric | 净面积XZ    |
| a_net_yz   | numeric | 净面积YZ    |
| cog_x      | numeric | 质心X坐标    |
| cog_y      | numeric | 质心Y坐标    |
| cog_z      | numeric | 质心Z坐标    |
| box_minx   | numeric | 包围盒最小X坐标 |
| box_miny   | numeric | 包围盒最小Y坐标 |
| box_minz   | numeric | 包围盒最小Z坐标 |
| box_maxx   | numeric | 包围盒最大X坐标 |
| box_maxy   | numeric | 包围盒最大Y坐标 |
| box_maxz   | numeric | 包围盒最大Z坐标 |
| area_plan  | numeric | 平面面积     |

初步将name作为分类的label
但csv中有汉字会比较难处理，统一采用utf-8格式会比较容易处理
于是将name转化为class
| name    | 计数   | class |
|---------|------|-------|
| Unnamed | 48   | DJZP  |
| BEAM    | 10   | XXX   |
| PLATE   | 476  | PLATE |
| 边腹板     | 180  | BFP   |
| 底板      | 1390 | DBP   |
| 吊耳      | 1264 | ZRP   |
| 顶板      | 2520 | TBP   |
| 横隔板     | 3374 | HGP   |
| 箭头      | 220  | KZ    |
| 锚箱      | 408  | MXP   |
| 散件      | 2850 | SJ    |
| 现场匹配件   | 1100 | LJP   |
| 纵腹板     | 1020 | ZGP   |

因为和坐标相关的特征，是和项目模型所设置的绝对坐标相关，因此尝试去除所有坐标相关特征，如下：

| 字段名称       | 数据类型    | 描述       |
|------------|---------|----------|
| cog_x      | numeric | 质心X坐标    |
| cog_y      | numeric | 质心Y坐标    |
| cog_z      | numeric | 质心Z坐标    |
| box_minx   | numeric | 包围盒最小X坐标 |
| box_miny   | numeric | 包围盒最小Y坐标 |
| box_minz   | numeric | 包围盒最小Z坐标 |
| box_maxx   | numeric | 包围盒最大X坐标 |
| box_maxy   | numeric | 包围盒最大Y坐标 |
| box_maxz   | numeric | 包围盒最大Z坐标 |

也就是说，只根据模型的重量、面积、投影面积进行训练，结果还可以，
说明对于同一个项目，仅依靠面积、重量、投影面积，就可以训练出极好的模型
对于多个项目模型，待续。。。