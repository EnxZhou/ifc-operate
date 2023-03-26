# 特征分类

 这里特征只针对几何特征，特征可以分为三类， 体的特征、面的特征、边的特征
 后续可以考虑零件的点特征，比如最高点，最低点，但估计效果一般，暂时不考虑

## 边的特征

边长暂时都假定是直线

| 字段名称       | 数据类型    | 描述       |
|------------|---------|----------|
| isStraight | boolean | 是否为直线    |
| length     | float   | 直线长度     |
| centreX    | float   | 直线中心点X坐标 |
| centreY    | float   | 直线中心点Y坐标 |
| centreZ    | float   | 直线中心点Z坐标 |

目前isStraight没什么用处，因为基本上用tekla导出的模型，都是以直代曲，所有所有边都是直线

```python
class EdgeFeature:  
    isStraight = True  
    length = 0.0  
    centreX = 0.0  
    centreY = 0.0  
    centreZ = 0.0
    
```

  

## 面的特征


| 字段名称               | 数据类型   | 描述    |
|--------------------|--------|-------|
| kind               | string | 面种类   |
| mass               | float  | 面积    |
| centreOfMassX      | float  | 质心X坐标 |
| centreOfMassY      | float  | 质心Y坐标 |
| centreOfMassZ      | float  | 质心Z坐标 |
| axisLocationX      | float  | 轴心X坐标 |
| axisLocationY      | float  | 轴心Y坐标 |
| axisLocationZ      | float  | 轴心Z坐标 |
| axisDirectX        | float  | 轴向X坐标 |
| axisDirectY        | float  | 轴向Y坐标 |
| axisDirectZ        | float  | 轴向Z坐标 |
| perimeter          | float  | 周长    |
| maxEdgeFeature     | object | 最长边特征 |
| minEdgeFeature     | object | 最短边特征 |
| edgeLengthAverage  | float  | 边长平均值 |
| edgeLengthVariance | float  | 边长方差  |

```python
class FaceFeature:  
    kind = ""  
    mass = 0.0  
    centreOfMassX = 0.0  
    centreOfMassY = 0.0  
    centreOfMassZ = 0.0  
    axisLocationX = 0.0  
    axisLocationY = 0.0  
    axisLocationZ = 0.0  
    axisDirectX = 0.0  
    axisDirectY = 0.0  
    axisDirectZ = 0.0  
    # 周长  
    perimeter = 0.0  
    # 最长边特征  
    maxEdgeFeature = EdgeUtil.EdgeFeature  
    # 最短边特征  
    minEdgeFeature = EdgeUtil.EdgeFeature  
    edgeLengthAverage = 0.0  
    # 边长方差  
    edgeLengthVariance = 0.0
```
  

## 体的特征

| 字段名称             | 数据类型   | 描述    |
|------------------|--------|-------|
| mass             | float  | 质量    |
| centreOfMassX    | float  | 质心X坐标 |
| centreOfMassY    | float  | 质心Y坐标 |
| centreOfMassZ    | float  | 质心Z坐标 |
| surfaceArea      | float  | 表面积   |
| maxFaceFeature   | object | 最大面特征 |
| minFaceFeature   | object | 最小面特征 |
| faceMassAverage  | float  | 面积平均值 |
| faceMassVariance | float  | 面积方差  |
| edgeCount        | int    | 边数量   |
| edgeLenSum       | float  | 边长总和  |

```python
class SolidFeature:  
    mass = 0.0  
    centreOfMassX = 0.0  
    centreOfMassY = 0.0  
    centreOfMassZ = 0.0  
    # 表面积  
    surfaceArea = 0.0  
    # 最大面特征  
    maxFaceFeature = FaceUtil.FaceFeature  
    # 最小面特征  
    minFaceFeature = FaceUtil.FaceFeature  
    # 面积平均值  
    faceMassAverage = 0.0  
    # 面积方差  
    faceMassVariance = 0.0  
    # 边数量  
    edgeCount = 0  
    # 边长总和  
    edgeLenSum = 0.0
```

# 解析执行

  

## 解析时间

以沙埕湾大桥一个节段模型(C3-JD-25)为例

共有零件1613个

i5-6500，单核，解析时间180.4秒

i7-8700，单核，解析时间123.52秒