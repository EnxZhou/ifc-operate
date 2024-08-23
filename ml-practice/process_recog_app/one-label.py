import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 读取数据
file_path = 'ximen-63-VM42-67.xlsx'
data = pd.read_excel(file_path)

# 数据预处理
# 将“零部件.工艺路线”列转换为单一标签
le = LabelEncoder()
data['零部件.工艺路线'] = le.fit_transform(data['零部件.工艺路线'])

# 提取特征
features = data[['零部件.图号', '零部件.中文名称', '零部件.材质', '零部件.规格', '零部件.重量', '零部件.外形尺寸-长度',
                   '零部件.外形尺寸-高度', '零部件.外形尺寸-宽度']].copy()
# 首先，确保'零部件.图号'列是字符串类型
features['零部件.图号'] = features['零部件.图号'].astype(str)

# 使用str.split()分割字符串，然后使用str.get()获取分割后列表的最后一个元素
features['零部件.图号'] = features['零部件.图号'].str.split('-').str.get(-1)

features['零部件.图号'] = features['零部件.图号'].astype('category').cat.codes
features['零部件.中文名称'] = features['零部件.中文名称'].astype('category').cat.codes
features['零部件.材质'] = features['零部件.材质'].astype('category').cat.codes
features['零部件.规格'] = features['零部件.规格'].astype('category').cat.codes

X = features.fillna(0)  # 填充缺失值
y = data['零部件.工艺路线']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建和训练模型
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# 预测和评估
y_pred = classifier.predict(X_test)

# 显示分类报告
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 生成分类报告
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

# 输出为Markdown表格格式
markdown_output = "# 分类模型评估报告\n\n"
markdown_output += "## 分类报告\n\n"
markdown_output += "| 类别      | 精确度 | 召回率 | F1分数 | 支持样本数 |\n"
markdown_output += "|-----------|--------|--------|--------|-----------|\n"

for label, metrics in report.items():
    if label == 'accuracy':
        continue
    markdown_output += f"| {label} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | {metrics['f1-score']:.2f} | {metrics['support']} |\n"

# 添加总体准确率
markdown_output += "\n## 总体准确率\n\n"
markdown_output += f"**总体准确率**: {report['accuracy']:.2f}\n"

# 输出Markdown内容
print(markdown_output)