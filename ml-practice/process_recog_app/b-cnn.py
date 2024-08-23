import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 读取数据
file_path = 'ximen-63-VM42-67.xlsx'
data = pd.read_excel(file_path)

# 数据预处理
# 将“零部件.工艺路线”列转换为单一标签
le = LabelEncoder()
data['零部件.工艺路线'] = le.fit_transform(data['零部件.工艺路线'])

# 提取特征
features = data[['零部件.材质', '零部件.规格', '零部件.重量', '零部件.外形尺寸-长度',
                  '零部件.外形尺寸-高度', '零部件.外形尺寸-宽度']].copy()
features['零部件.材质'] = features['零部件.材质'].astype('category').cat.codes
features['零部件.规格'] = features['零部件.规格'].astype('category').cat.codes

X = features.fillna(0).values  # 填充缺失值并转换为数组
y = data['零部件.工艺路线'].values  # 标签

# 转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 调整特征形状以适应CNN输入
X_tensor = X_tensor.unsqueeze(1)  # 变为 (batch_size, channels, sequence_length)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义B-CNN模型
class B_CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(B_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (input_size // 4), 128)  # Adjust size based on pool and conv layers
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型
input_size = X_tensor.size(2)  # Sequence length after reshaping
num_classes = len(le.classes_)
model = B_CNN(input_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

train_model(model, train_loader, criterion, optimizer)

# 预测和评估
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

y_true, y_pred = evaluate_model(model, test_loader)

# 生成分类报告
report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)

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