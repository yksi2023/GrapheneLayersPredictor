import cv2
import numpy as np
import csv

class GrapheneDataCollectorCore:
    def __init__(self):
        self.cv_img = None
        self.layer_count = None
        self.points = []   
        self.data = []     

    def load_image(self, path: str, layer_count: int) -> bool:
        """加载图像并设置层数，成功返回True"""
        self.points.clear()
        self.data.clear()
        self.layer_count = layer_count

        try:
            with open(path, 'rb') as f:
                img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return False
            self.cv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return True
        except Exception as e:
            print(f"加载图像失败: {e}")
            return False

    def get_image(self):
        return self.cv_img

    def get_points(self):
        return self.points

    def undo_last_point(self):
        if self.points:
            self.points.pop()

    def add_point(self, x: int, y: int):
        if self.cv_img is None:
            return False

        h_img, w_img, _ = self.cv_img.shape
        if not (0 <= x < w_img and 0 <= y < h_img):
            return False

        rgb = self.cv_img[y, x]
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv.astype(float) / [179.0, 255.0, 255.0]  # 归一化

        self.points.append(((rgb, (h, s, v)), (x, y)))

        if len(self.points) % 2 == 0:
            (rgb1, (h1, s1, v1)), _ = self.points[-2]
            (rgb2, (h2, s2, v2)), _ = self.points[-1]

            r1, g1, b1 = map(int, rgb1)
            r2, g2, b2 = map(int, rgb2)


            def safe_div(a, b):
                return float(a) / b if b != 0 else 0.0

            row = {
                "R1": r1, "G1": g1, "B1": b1,
                "H1": h1, "S1": s1, "V1": v1,
                "R2": r2, "G2": g2, "B2": b2,
                "H2": h2, "S2": s2, "V2": v2,
                "ratio_R": safe_div(r1, r2),
                "ratio_G": safe_div(g1, g2),
                "ratio_B": safe_div(b1, b2),
                "ratio_H": safe_div(h1, h2),
                "ratio_S": safe_div(s1, s2),
                "ratio_V": safe_div(v1, v2),
                "diff_R": r1 - r2,
                "diff_G": g1 - g2,
                "diff_B": b1 - b2,
                "diff_H": h1 - h2,
                "diff_S": s1 - s2,
                "diff_V": v1 - v2,
                "layer_count": self.layer_count
            }

            self.data.append(row)

        return True


    def get_data(self):
        return self.data

    def export_to_csv(self, path: str):
        if not self.data:
            return False

        with open(path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "R1", "G1", "B1", "H1", "S1", "V1",
                "R2", "G2", "B2", "H2", "S2", "V2",
                "ratio_R", "ratio_G", "ratio_B", "ratio_H", "ratio_S", "ratio_V",
                "layer_count"
            ])
            writer.writerows(self.data)

        return True
