from PySide6.QtWidgets import QMainWindow, QTabWidget
from tabs.data_tab import DataTab
from tabs.train_tab import TrainTab
from tabs.predict_tab import PredictTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("石墨烯层数识别程序")
        self.setGeometry(100, 100, 1000, 700)

        # 创建选项卡
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 添加功能页面（Tab）
        self.tabs.addTab(DataTab(), "数据采集")
        self.tabs.addTab(TrainTab(), "模型训练")
        self.tabs.addTab(PredictTab(), "预测")
