import numpy as np
import pickle
import os
import pandas as pd
from collections import Counter

class GraphenePredictor:
    def __init__(self):
        self.scaler = None
        self.rfe = None
        self.model = None
        self.label_encoder = None
        self.feature_names = []
        self.prediction_data = []  # 每组为 [(rgb1, hsv1), (rgb2, hsv2)]

    def load_model(self, folder_path: str = "models") -> bool:
        try:
            with open(os.path.join(folder_path, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(folder_path, "rfe.pkl"), "rb") as f:
                self.rfe = pickle.load(f)
            with open(os.path.join(folder_path, "model.pkl"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(folder_path, "label_encoder.pkl"), "rb") as f:
                self.label_encoder = pickle.load(f)
            with open(os.path.join(folder_path, "features.pkl"), "rb") as f:
                self.feature_names = pickle.load(f)
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def reset(self):
        self.prediction_data.clear()

    def add_point_pair(self, rgb1, hsv1, rgb2, hsv2):
        self.prediction_data.append((rgb1, hsv1, rgb2, hsv2))

    def _construct_feature_row(self, rgb1, hsv1, rgb2, hsv2):
        r1, g1, b1 = map(int, rgb1)
        h1, s1, v1 = hsv1
        r2, g2, b2 = map(int, rgb2)
        h2, s2, v2 = hsv2

        def safe_div(a, b):
            return float(a) / b if b != 0 else 0.0

        features = {
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
            "diff_R": abs(r1 - r2),
            "diff_G": abs(g1 - g2),
            "diff_B": abs(b1 - b2),
            "diff_H": abs(h1 - h2),
            "diff_S": abs(s1 - s2),
            "diff_V": abs(v1 - v2),
        }
        return pd.DataFrame([features], columns=self.feature_names)

    def predict_all(self):
        if not self.prediction_data:
            return [], "没有点对可预测。"

        rows = [self._construct_feature_row(*pair) for pair in self.prediction_data]
        X = pd.concat(rows, ignore_index=True)
        X_scaled = self.scaler.transform(X)
        X_selected = self.rfe.transform(X_scaled)

        y_pred = self.model.predict(X_selected)
        labels = self.label_encoder.inverse_transform(y_pred)

        proba = self.model.predict_proba(X_selected)
        mean_proba = proba.mean(axis=0)
        soft_vote_index = np.argmax(mean_proba)
        soft_vote_label = self.label_encoder.inverse_transform([soft_vote_index])[0]
        soft_vote_prob = mean_proba[soft_vote_index]

        count = Counter(labels)
        majority_label, majority_votes = count.most_common(1)[0]

        clean_labels = [int(l) for l in labels]

        summary = (
            f"每组预测结果：{clean_labels}\n"
            f"众数预测结果：{majority_label}（{majority_votes} 票）\n"
            f"Soft Voting 预测：{soft_vote_label}（平均置信度：{soft_vote_prob:.2f}）"
        )

        return list(labels), summary