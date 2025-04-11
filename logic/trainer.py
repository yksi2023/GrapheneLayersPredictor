import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import logging

class GrapheneTrainer:
    def __init__(self):
        self.df = None
        self.scaler = None
        self.rfe = None
        self.label_encoder = None
        self.model = None
        self.original_features = []
        self.selected_features = []
        self.report_text = ""

    def load_data(self, paths: list[str]):
        dfs = []
        for path in paths:
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                print(f"读取失败: {path}, 错误: {e}")

        if not dfs:
            return False

        self.df = pd.concat(dfs, ignore_index=True)

        # 构建 24 个特征
        self.df['diff_R'] = (self.df['R1'] - self.df['R2']).abs()
        self.df['diff_G'] = (self.df['G1'] - self.df['G2']).abs()
        self.df['diff_B'] = (self.df['B1'] - self.df['B2']).abs()
        self.df['diff_H'] = (self.df['H1'] - self.df['H2']).abs()
        self.df['diff_S'] = (self.df['S1'] - self.df['S2']).abs()
        self.df['diff_V'] = (self.df['V1'] - self.df['V2']).abs()

        def safe_div(a, b):
            return a / b.replace(0, np.nan)

        self.df['ratio_R'] = safe_div(self.df['R1'], self.df['R2']).fillna(0)
        self.df['ratio_G'] = safe_div(self.df['G1'], self.df['G2']).fillna(0)
        self.df['ratio_B'] = safe_div(self.df['B1'], self.df['B2']).fillna(0)
        self.df['ratio_H'] = safe_div(self.df['H1'], self.df['H2']).fillna(0)
        self.df['ratio_S'] = safe_div(self.df['S1'], self.df['S2']).fillna(0)
        self.df['ratio_V'] = safe_div(self.df['V1'], self.df['V2']).fillna(0)

        self.original_features = [
            'R1', 'G1', 'B1', 'H1', 'S1', 'V1',
            'R2', 'G2', 'B2', 'H2', 'S2', 'V2',
            'ratio_R', 'ratio_G', 'ratio_B', 'ratio_H', 'ratio_S', 'ratio_V',
            'diff_R', 'diff_G', 'diff_B', 'diff_H', 'diff_S', 'diff_V'
        ]

        return True

    def train(self):
        X = self.df[self.original_features]
        y_raw = self.df['layer_count']

        # 标签编码
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw)

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 特征选择
        svc_linear = SVC(kernel='linear', C=1.0, random_state=42)
        self.rfe = RFE(estimator=svc_linear, n_features_to_select=5)
        self.rfe.fit(X_scaled, y)
        self.selected_features = np.array(self.original_features)[self.rfe.support_]
        X_selected = self.rfe.transform(X_scaled)

        # 模型调参 SVM
        pipeline_svm = Pipeline([
            ('svm', SVC(probability=True, random_state=42))
        ])
        param_grid_svm = {
            'svm__kernel': ['rbf', 'linear'],
            'svm__C': [0.1, 1.0, 10.0],
            'svm__gamma': ['scale', 'auto']
        }
        grid_svm = GridSearchCV(pipeline_svm, param_grid_svm,
                                cv=5, scoring='accuracy', n_jobs=-1, refit=True)
        grid_svm.fit(X_selected, y)

        # 随机森林调参
        pipeline_rf = Pipeline([
            ('rf', RandomForestClassifier(random_state=42))
        ])
        param_grid_rf = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [None, 10],
            'rf__min_samples_split': [2, 4]
        }
        grid_rf = GridSearchCV(pipeline_rf, param_grid_rf,
                               cv=5, scoring='accuracy', n_jobs=-1, refit=True)
        grid_rf.fit(X_selected, y)

        # 集成模型
        self.model = VotingClassifier(
            estimators=[
                ('svm', grid_svm.best_estimator_),
                ('rf', grid_rf.best_estimator_)
            ],
            voting='soft'
        )
        self.model.fit(X_selected, y)

        # 评估
        y_pred = self.model.predict(X_selected)

        # 反编码：将 0/1/2 → 真实层数（如 4/5/6）
        y_true_labels = self.label_encoder.inverse_transform(y)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        # 获取真实标签顺序（用于固定顺序显示）
        label_names = list(self.label_encoder.classes_)

        acc = accuracy_score(y_true_labels, y_pred_labels)
        report = classification_report(y_true_labels, y_pred_labels, labels=label_names)
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_names)


        self.report_text = (
        f"训练完成\n\n"
        f"准确率：{acc * 100:.2f}%\n\n"
        f"最佳 SVM 参数：{grid_svm.best_params_}\n"
        f"最佳随机森林参数：{grid_rf.best_params_}\n\n"
        f"使用特征（RFE 选出）：{self.selected_features.tolist()}\n\n"
        f"分类报告：\n{report}\n"
        f"混淆矩阵：\n{cm}"
        )


    def save_all(self, folder_path: str = "models"):
        os.makedirs(folder_path, exist_ok=True)

        with open(os.path.join(folder_path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(folder_path, "rfe.pkl"), "wb") as f:
            pickle.dump(self.rfe, f)

        with open(os.path.join(folder_path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        with open(os.path.join(folder_path, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(os.path.join(folder_path, "features.pkl"), "wb") as f:
            pickle.dump(self.original_features, f)

        with open(os.path.join(folder_path, "selected_features.pkl"), "wb") as f:
            pickle.dump(self.selected_features.tolist(), f)

        return True

    def get_report(self):
        return self.report_text
