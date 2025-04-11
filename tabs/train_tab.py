from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTextEdit,
    QFileDialog, QLabel, QInputDialog
)
import os
from glob import glob
from logic.trainer import GrapheneTrainer

class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        self.trainer = GrapheneTrainer()
        self.csv_paths = []

        # 布局
        self.layout = QVBoxLayout(self)

        self.status_label = QLabel("请导入训练数据")
        self.layout.addWidget(self.status_label)

        # 加载数据按钮
        self.btn_load_all = QPushButton("加载 data/ 中所有 CSV")
        self.btn_load_select = QPushButton("手动选择 CSV 文件")
        self.layout.addWidget(self.btn_load_all)
        self.layout.addWidget(self.btn_load_select)

        # 训练 & 保存模型按钮
        self.btn_train = QPushButton("开始训练模型")
        self.btn_save = QPushButton("保存模型")
        self.layout.addWidget(self.btn_train)
        self.layout.addWidget(self.btn_save)

        # 报告输出
        self.layout.addWidget(QLabel("训练评估报告："))
        self.text_report = QTextEdit()
        self.text_report.setReadOnly(True)
        self.layout.addWidget(self.text_report)

        # 绑定
        self.btn_load_all.clicked.connect(self.load_all_data)
        self.btn_load_select.clicked.connect(self.load_selected_data)
        self.btn_train.clicked.connect(self.train_model)
        self.btn_save.clicked.connect(self.save_model)

    def set_status(self, text):
        self.status_label.setText(text)

    def load_all_data(self):
        csvs = glob("data/*.csv")
        if not csvs:
            self.set_status("data/ 中没有找到任何 CSV 文件")
            return
        ok = self.trainer.load_data(csvs)
        if ok:
            self.csv_paths = csvs
            self.set_status(f"已加载 {len(csvs)} 个文件中的数据")
        else:
            self.set_status("加载失败，请检查文件格式")

    def load_selected_data(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择一个或多个 CSV 文件", "", "CSV 文件 (*.csv)"
        )
        if not files:
            self.set_status("未选择任何文件")
            return
        ok = self.trainer.load_data(files)
        if ok:
            self.csv_paths = files
            self.set_status(f"已加载 {len(files)} 个文件中的数据")
        else:
            self.set_status("加载失败，请检查文件格式")

    def train_model(self):
        if not self.csv_paths:
            self.set_status("请先加载数据")
            return

        self.set_status("训练中，请稍候...")
        self.trainer.train()
        self.set_status("训练完成")
        self.text_report.setText(self.trainer.get_report())

    def save_model(self):
        if self.trainer.model is None:
            self.set_status("还未训练模型，无法保存")
            return

        version, ok = QInputDialog.getText(self, "输入模型版本", "请输入模型版本名：")
        if not ok or not version.strip():
            self.set_status("保存已取消")
            return

        path = os.path.join("models", version.strip())
        self.trainer.save_all(path)
        self.set_status(f"模型已保存到 models/{version.strip()}")
