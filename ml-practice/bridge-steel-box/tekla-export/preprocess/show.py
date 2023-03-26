
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def drop_unique_column(data):
    from sklearn.preprocessing import LabelEncoder
    object_feature = list(data.select_dtypes(include=['object']).columns)
    unique_column = data[object_feature].nunique()
    unique_column.drop('class', inplace=True)
    # guid作为唯一标识，不能作为特征
    unique_column.drop('guid', inplace=True)
    # assembly_no节段号，只为方便筛选，也不作为特征
    unique_column.drop('assembly_no', inplace=True)
    # part_pos零件号，等同于class，也不作为特征
    unique_column.drop('part_pos', inplace=True)
    unique_column = unique_column.index
    print("unique column: ", unique_column)
    for col in unique_column:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    data.to_csv("tmp_train.csv")


def object_col(data):
    object_feature = list(data.select_dtypes(include=['object']).columns)
    object_column_name = []
    unique_value = []
    for col in object_feature:
        object_column_name.append(col)
        unique_value.append(data[col].nunique())
    # print("object_column", object_column_name)
    # print("numerical_column", numerical_column_name)
    df = pd.DataFrame()
    df['col_name'] = object_column_name
    df['value'] = unique_value
    df = df.sort_values('value', ascending=False)
    print(df)


def distplot(serial_df):
    plt.figure(figsize=(10, 5))
    f = sns.FacetGrid(serial_df, col='variable', col_wrap=2, sharex=False, sharey=False)  # 生成画布，最多三列，不共享x、y轴
    f.map(sns.distplot, "value")
    plt.show()


def numerical_cor(data):
    cor = data.corr()
    sns.set_theme(style="white")
    plt.figure(figsize=(16, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(cor, cmap=cmap, annot=True, linewidths=0.2,
                cbar_kws={"shrink": 0.5}, linecolor="white", fmt=".1g")
    plt.show()


def boxplot(serial_df):
    plt.figure(figsize=(16, 8))
    f_box = sns.FacetGrid(serial_df, col='variable', col_wrap=5, sharex=False, sharey=False)
    f_box.map(sns.boxplot, "value")
    plt.show()