import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder

# train_file_path = 'ximen-63-VM42.xlsx'
# test_file_path = 'ximen-63-VM43-67.xlsx'
train_file_path = 'ximen-63-VM42-43-zex_revise.xlsx'
test_file_path = 'ximen-63-VM44-67-zex_revise.xlsx'

# mlb = MultiLabelBinarizer(classes=processList)
mlb = MultiLabelBinarizer()
scaler = StandardScaler()
le = LabelEncoder()


def load_data(file_path):
    return pd.read_excel(file_path)


def preprocess_data(df):
    # 数据预处理
    # 将“零部件.工艺路线”列转换为多标签格式
    # df['零部件.工艺路线'] = df['零部件.工艺路线'].apply(lambda x: x.split('、'))
    # y = mlb.fit_transform(df['零部件.工艺路线'])

    # 将“零部件.工艺路线”列转换为单一标签
    df['零部件.工艺路线'] = le.fit_transform(df['零部件.工艺路线'])
    y = df['零部件.工艺路线']

    # 提取特征
    features = df[
        ['零部件.图号', '零部件.中文名称', '零部件.材质', '零部件.规格', '零部件.重量', '零部件.外形尺寸-长度',
         '零部件.外形尺寸-高度', '零部件.外形尺寸-宽度']].copy()

    # 首先，确保'零部件.图号'列是字符串类型
    features['零部件.图号'] = features['零部件.图号'].astype(str)

    # 使用str.split()分割字符串，然后使用str.get()获取分割后列表的最后一个元素
    features['零部件.图号'] = features['零部件.图号'].str.split('-').str.get(-1)

    features['零部件.图号'] = features['零部件.图号'].astype('category').cat.codes
    features['零部件.中文名称'] = features['零部件.中文名称'].astype('category').cat.codes
    features['零部件.材质'] = features['零部件.材质'].astype('category').cat.codes
    features['零部件.规格'] = features['零部件.规格'].astype('category').cat.codes

    # 归一化处理
    numeric_features = features[['零部件.重量', '零部件.外形尺寸-长度',
                                 '零部件.外形尺寸-高度', '零部件.外形尺寸-宽度']]
    features[['零部件.重量', '零部件.外形尺寸-长度',
              '零部件.外形尺寸-高度', '零部件.外形尺寸-宽度']] = scaler.fit_transform(numeric_features)
    X = features.fillna(0)  # 填充缺失值

    return X, y


def train_model(X_train, y_train, model_path='model.pkl'):
    # rf = RandomForestClassifier()
    # multi_target_model = MultiOutputClassifier(rf)
    # multi_target_model.fit(X_train, y_train)
    # joblib.dump(multi_target_model, model_path)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, model_path)


def predict(model_path, X_test):
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    return predictions


# 输出预测结果
def show_report(y_pred, y_test):
    # 显示分类报告
    # print(classification_report(y_test, y_pred, target_names=mlb.classes_))
    # 生成分类报告
    # report = classification_report(y_test, y_pred, target_names=mlb.classes_, output_dict=True)
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

    # 输出Markdown内容
    print(markdown_output)


def save_pred_file(y_pred, test_data):
    # 使用inverse_transform将预测结果转换为原始标签
    # y_pred_labels = mlb.inverse_transform(y_pred)

    # 使用列表推导式和 join 方法将每一行的标签索引转换为用“、”分隔的字符串
    # y_pred_labels_str = ['、'.join(labels) for
    #                      labels in y_pred_labels]

    y_pred_labels = le.inverse_transform(y_pred)
    y_pred_labels_str = y_pred_labels
    # 将预测的标签转换为DataFrame
    df_predictions = pd.DataFrame(y_pred_labels_str)

    # 将预测结果与测试数据合并
    merged_data = pd.concat([test_data, df_predictions], axis=1)

    # 保存为新的Excel文件
    output_file_path = 'merged_test_results.xlsx'
    merged_data.to_excel(output_file_path, index=False)

    print(f'预测结果已合并并保存到 {output_file_path}')


# 分类操作
def classify():
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)
    train_len = len(train_data)
    test_len = len(test_data)

    combined_data = pd.concat([train_data, test_data])

    # 数据预处理
    features, y = preprocess_data(combined_data)

    # 分割训练集和测试集
    X_train = features[:train_len]
    y_train = y[:train_len]
    X_test = features[train_len:train_len + test_len]
    y_test = y[train_len:train_len + test_len]

    # 训练模型
    train_model(X_train, y_train)

    # 进行预测
    y_pred = predict('model.pkl', X_test)
    save_pred_file(y_pred, test_data)
    # show_report(y_pred, y_test)


if __name__ == '__main__':
    classify()
