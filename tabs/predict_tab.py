from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFileDialog,
    QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QTextEdit, QComboBox
)
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QPen, QColor
from PySide6.QtCore import Qt, QPointF, QEvent
import numpy as np
import cv2
import os

from logic.predictor import GraphenePredictor

class PredictTab(QWidget):
    def __init__(self):
        super().__init__()
        self.predictor = GraphenePredictor()
        self.cv_img = None
        self.point_buffer = []
        self.point_items = []
        self.point_index = 0
        self.drag_start = None

        self.layout = QVBoxLayout(self)

        # 控件栏
        control_bar = QHBoxLayout()
        self.btn_refresh_models = QPushButton("刷新模型列表")
        self.model_selector = QComboBox()
        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_img = QPushButton("加载图像")
        self.btn_predict = QPushButton("重新预测")
        self.btn_clear = QPushButton("清除所有点")
        self.btn_undo = QPushButton("撤销上一个点")
        self.status = QLabel("状态：")

        control_bar.addWidget(self.btn_refresh_models)
        control_bar.addWidget(QLabel("模型："))
        control_bar.addWidget(self.model_selector)
        control_bar.addWidget(self.btn_load_model)
        control_bar.addWidget(self.btn_load_img)
        control_bar.addWidget(self.btn_undo)
        control_bar.addWidget(self.btn_clear)
        control_bar.addWidget(self.btn_predict)
        control_bar.addWidget(self.status)
        self.layout.addLayout(control_bar)

        # 预测结果（大字）
        self.result_summary = QLabel("暂无预测")
        self.result_summary.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.layout.addWidget(self.result_summary)

        # 图像显示
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.layout.addWidget(self.view)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.viewport().installEventFilter(self)

        # 文本预测结果
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(QLabel("所有预测结果："))
        self.layout.addWidget(self.result_text)

        # 信号绑定
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_img.clicked.connect(self.load_image)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_undo.clicked.connect(self.undo_point)
        self.btn_predict.clicked.connect(self.run_prediction)
        self.btn_refresh_models.clicked.connect(self.refresh_model_list)

        self.refresh_model_list()

    def set_status(self, text): self.status.setText(text)

    def refresh_model_list(self):
        self.model_selector.clear()
        os.makedirs("models", exist_ok=True)
        for d in os.listdir("models"):
            if os.path.isdir(os.path.join("models", d)):
                self.model_selector.addItem(d)

    def load_model(self):
        selected = self.model_selector.currentText()
        path = os.path.join("models", selected)
        if self.predictor.load_model(path):
            self.set_status(f"模型已加载：{selected}")
        else:
            self.set_status("模型加载失败")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.bmp)")
        if not path: return

        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = self.cv_img.shape
        qimg = QImage(self.cv_img.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.clear_all()  # ✅ 修复：先清除点、文本、模型状态
        self.scene.clear()

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)

        scale_x = self.view.viewport().width() / w
        scale_y = self.view.viewport().height() / h
        scale = min(scale_x, scale_y)
        self.view.resetTransform()
        self.view.scale(scale, scale)
        self.view.centerOn(self.pixmap_item)

        self.set_status("图像加载成功")

    def eventFilter(self, source, event):
        if source is self.view.viewport():
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    return self.handle_click(event)
                elif event.button() == Qt.RightButton:
                    self.drag_start = event.pos()
                    return True
            elif event.type() == QEvent.MouseMove and self.drag_start:
                delta = event.pos() - self.drag_start
                self.drag_start = event.pos()
                self.view.horizontalScrollBar().setValue(self.view.horizontalScrollBar().value() - delta.x())
                self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().value() - delta.y())
                return True
            elif event.type() == QEvent.MouseButtonRelease:
                self.drag_start = None
                return True
            elif event.type() == QEvent.Wheel:
                factor = 1.25 if event.angleDelta().y() > 0 else 0.8
                self.view.scale(factor, factor)
                return True

        return super().eventFilter(source, event)

    def handle_click(self, event: QMouseEvent):
        if self.cv_img is None: return True
        pos = self.view.mapToScene(event.pos())
        x, y = int(pos.x()), int(pos.y())
        h, w, _ = self.cv_img.shape
        if not (0 <= x < w and 0 <= y < h): return True

        rgb = self.cv_img[y, x]
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        h_norm, s_norm, v_norm = hsv.astype(float) / [179.0, 255.0, 255.0]

        self.point_buffer.append((rgb, (h_norm, s_norm, v_norm)))
        self.point_index += 1
        pen = QPen(QColor("red") if self.point_index % 2 == 1 else QColor("blue"))
        pen.setWidth(8)
        dot = self.scene.addEllipse(x - 4, y - 4, 8, 8, pen)
        label = self.scene.addText(str(self.point_index))
        label.setPos(x + 8, y)
        label.setDefaultTextColor(QColor("white"))
        label.setScale(1.3)

        self.point_items += [dot, label]

        if len(self.point_buffer) == 2:
            (rgb1, hsv1), (rgb2, hsv2) = self.point_buffer
            self.predictor.add_point_pair(rgb1, hsv1, rgb2, hsv2)
            self.point_buffer = []
        return True

    def run_prediction(self):
        labels, summary = self.predictor.predict_all()
        self.result_text.setText(summary)
        lines = summary.splitlines()
        self.result_summary.setText(lines[2] if len(lines) > 2 else "预测失败")

    def clear_all(self):
        self.point_buffer.clear()
        self.predictor.reset()
        self.result_text.clear()
        self.result_summary.setText("暂无预测")
        for item in self.point_items:
            self.scene.removeItem(item)
        self.point_items.clear()
        self.point_index = 0
        self.set_status("已清除所有点")

    def undo_point(self):
        if self.point_buffer:
            self.point_buffer.pop()
            for _ in range(2):
                if self.point_items:
                    self.scene.removeItem(self.point_items.pop())
            self.point_index -= 1
        elif self.predictor.prediction_data:
            self.predictor.prediction_data.pop()
            for _ in range(4):
                if self.point_items:
                    self.scene.removeItem(self.point_items.pop())
            self.point_index -= 2
            self.run_prediction()
        self.set_status("已撤销上一个点或点对")