import csv
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFileDialog,
    QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QInputDialog
)
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent, QPen, QColor, QPainter
from PySide6.QtCore import Qt, QPointF, QEvent
from logic.data_collector import GrapheneDataCollectorCore
from datetime import datetime

class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.core = GrapheneDataCollectorCore()

        # UI 布局
        self.layout = QVBoxLayout(self)

        # 按钮区
        self.button_bar = QHBoxLayout()
        self.layout.addLayout(self.button_bar)

        self.load_btn = QPushButton("打开图像")
        self.undo_btn = QPushButton("撤销上一个点")
        self.save_btn = QPushButton("保存数据为 CSV")
        self.status_label = QLabel("准备就绪")

        self.button_bar.addWidget(self.load_btn)
        self.button_bar.addWidget(self.undo_btn)
        self.button_bar.addWidget(self.save_btn)
        self.button_bar.addWidget(self.status_label)

        # 图像显示区
        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.layout.addWidget(self.view)

        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.pixmap_item = None

        self.scale = 1.0
        self.point_items = []

        # 信号绑定
        self.load_btn.clicked.connect(self.load_image)
        self.undo_btn.clicked.connect(self.undo_point)
        self.save_btn.clicked.connect(self.save_data)

        self.view.viewport().installEventFilter(self)
        self.drag_start = None  # 拖动起点

        self.point_index = 0 

    def set_status(self, text):
        self.status_label.setText(text)

    def load_image(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if not img_path:
            return

        count, ok = QInputDialog.getInt(self, "图像层数", "请输入该图像的石墨烯层数：")
        if not ok:
            self.set_status("取消加载图像。")
            return

        success = self.core.load_image(img_path, count)
        if not success:
            self.set_status("图像加载失败。")
            return

        self.scene.clear()
        self.point_items.clear()

        img = self.core.get_image()
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.set_status("图像加载成功，点击图像采样。")
        # 计算缩放因子，让图像适应视图大小
        view_size = self.view.viewport().size()
        scale_x = view_size.width() / self.pixmap_item.pixmap().width()
        scale_y = view_size.height() / self.pixmap_item.pixmap().height()
        scale = min(scale_x, scale_y)

        self.view.resetTransform()
        self.view.scale(scale, scale)

        # 居中显示图像
        self.view.centerOn(self.pixmap_item)

    def undo_point(self):
        if not self.point_items:
            return
        self.core.undo_last_point()
        # 删除最后两个图形元素（点和编号）
        for _ in range(2):
            item = self.point_items.pop()
            self.scene.removeItem(item)
        self.set_status("撤销上一个点。")

    from PySide6.QtWidgets import QFileDialog


    def save_data(self):
        data = self.core.get_data()
        if not data:
            self.set_status("当前没有记录任何数据，无法保存。")
            return

        # 默认路径 data/ + 时间戳
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join("data", f"graphene_data_{timestamp}.csv")

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存数据为 CSV", default_path, "CSV 文件 (*.csv)"
        )
        if not save_path:
            self.set_status("用户取消保存")
            return

        field_order = [
            "R1", "G1", "B1", "H1", "S1", "V1",
            "R2", "G2", "B2", "H2", "S2", "V2",
            "ratio_R", "ratio_G", "ratio_B", "ratio_H", "ratio_S", "ratio_V",
            "diff_R", "diff_G", "diff_B", "diff_H", "diff_S", "diff_V",
            "layer_count"
        ]

        try:
            with open(save_path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=field_order)
                writer.writeheader()
                writer.writerows(data)
            self.set_status(f"数据已保存到：{os.path.basename(save_path)}")
        except Exception as e:
            self.set_status(f"保存失败：{e}")

    def eventFilter(self, source, event):
        if source is self.view.viewport():
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    return self.handle_click(event)
                elif event.button() == Qt.RightButton:
                    self.drag_start = event.pos()
                    return True

            elif event.type() == QEvent.MouseMove:
                if self.drag_start:
                    delta = event.pos() - self.drag_start
                    self.drag_start = event.pos()
                    self.view.horizontalScrollBar().setValue(self.view.horizontalScrollBar().value() - delta.x())
                    self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().value() - delta.y())
                    return True

            elif event.type() == QEvent.MouseButtonRelease:
                self.drag_start = None
                return True

            elif event.type() == QEvent.Wheel:
                return self.handle_zoom(event)
        return super().eventFilter(source, event)


    def handle_click(self, event: QMouseEvent):
        scene_pos = self.view.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())

        if not self.core.add_point(x, y):
            self.set_status("点击无效。")
            return True

        color = "red" if len(self.core.get_points()) % 2 == 1 else "blue"
        pen = QPen(QColor(color))
        pen.setWidth(8)  # 更粗的边框
        dot = self.scene.addEllipse(x - 4, y - 4, 8, 8, pen)  # 更大的点
        self.point_items.append(dot)

        label = self.scene.addText(str(len(self.core.get_points())))
        label.setPos(QPointF(x + 8, y))
        label.setDefaultTextColor(QColor("white"))
        label.setScale(1.5)  # 放大文字
        self.point_items.append(label)

        if len(self.core.get_points()) % 2 == 0:
            self.set_status(f"已记录第 {len(self.core.get_data())} 组数据。")
        return True


    def handle_zoom(self, event: QWheelEvent):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.view.scale(factor, factor)
        return True
