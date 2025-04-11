# README

## 📦 项目结构

```
Graphene_app/
├── main.py                  # 程序入口
├── main_window.py           # 主窗口，包含多个 Tab
├── tabs/                    # GUI 逻辑分模块
│   ├── data_tab.py          # 数据采集界面
│   ├── train_tab.py         # 模型训练界面
│   └── predict_tab.py       # 层数预测界面
├── logic/                   # 核心功能逻辑
│   ├── data_collector.py    # 数据采集与特征构造
│   ├── trainer.py           # 模型训练与保存
│   └── predictor.py         # 模型加载与预测
├── models/                  # 保存模型的子目录
└──  data/                    # 自动保存的采集数据 CSV
```

------

## 🛠️ 安装依赖

- PySide6
- numpy
- pandas
- opencv-python
- scikit-learn

------

## 🚀 使用说明

### ✅ 启动程序

```bash
python main.py
```

------

## 🔧 模块功能说明

### 1. 数据采集模块（"数据采集" Tab）

- 支持图像缩放、拖动、点击选点
- 每组数据为两个点（样本 + 衬底）
- 自动提取 24 维特征（RGB、HSV、比值、差值）
- 支持撤销、清空、保存为 CSV（默认保存在 `data/` 文件夹）

### 2. 模型训练模块（"模型训练" Tab）

- 支持：
  - 自动加载 `data/` 中所有 CSV
  - 或手动多选文件训练
- 模型包括：SVM + RF + VotingClassifier
- 使用：
  - RFE 特征选择
  - GridSearchCV 自动调参
- 模型保存：用户命名版本号，自动保存至 `models/版本名/`
- 输出：准确率 + 分类报告 + 混淆矩阵 + 最佳参数

### 3. 层数预测模块（"层数预测" Tab）

- 加载图像，选点
- 加载模型（从 `models/` 中选）
- 多组点支持预测（Soft Voting + 众数融合）
- 实时预测展示
- 支持撤销、清除、图像缩放

------

## 📁 模型目录结构（每个版本）

```
models/版本名/
├── model.pkl               # VotingClassifier 模型
├── scaler.pkl              # 训练用 StandardScaler
├── rfe.pkl                 # 特征选择器（RFE）
├── label_encoder.pkl       # LabelEncoder
├── features.pkl            # 所有特征名
├── selected_features.pkl   # RFE 选中的 5 个特征
```

------

## 🧪 示例使用流程

1. 进入“数据采集”Tab，加载图像，点击记录多个样本点 → 保存 CSV
2. 进入“模型训练”Tab，选择数据 → 训练 → 输入版本名保存模型
3. 进入“层数预测”Tab，加载图像、加载模型 → 点击两点进行预测

