import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV

# === 特徴量抽出 ===
def extract_features(img_path, size=(800,600)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    features = hog(img, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    return img.flatten()

# === one-hot 変換 ===
def to_one_hot(class_id, num_classes=3):
    one_hot = np.zeros(num_classes, dtype=int)
    one_hot[class_id] = 1
    return one_hot

def svm_with_gridsearch(X_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": [0.001, 0.01, 0.1, 1],
        "kernel": ["rbf"]
    }
    grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    return grid.best_estimator_

# === メイン関数 ===
def evaluate_models(csv_file, image_dir):
    # CSV読み込み
    df = pd.read_csv(csv_file)

    # 正解ラベル (0=Zebra_Left, 1=Zebra_Center, 2=Zebra_Right)
    y_onehot = df[["Zebra_Left","Zebra_Center","Zebra_Right"]].values
    y_class = np.argmax(y_onehot, axis=1)

    # 特徴量抽出
    X, valid_files, valid_y = [], [], []
    for idx, row in df.iterrows():
        path = os.path.join(image_dir, row["image_name"])
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            valid_files.append(row["image_name"])
            valid_y.append(y_class[idx])

    X = np.array(X)
    y = np.array(valid_y)

    # データ分割
    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        X, y, valid_files, test_size=0.3, random_state=42, stratify=y
    )

    # モデル群
    models = {
        "SVM": svm_with_gridsearch(X_train, y_train),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # === one-hot 出力形式 (テストデータのみ) ===
        predictions = []
        for fname, pred in zip(f_test, y_pred):
            one_hot = to_one_hot(pred, 3)
            predictions.append([fname] + one_hot.tolist())
        
        # 精度計算
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=["Zebra_Left","Zebra_Center","Zebra_Right"], output_dict=True
        )

        results[name] = {
            "predictions": predictions,  # [["4.jpg",1,0,0], ...]
            "accuracy": acc,
            "report": report
        }

    return results

if __name__ == '__main__':
    # === 使用例 ===
    results = evaluate_models('./zebra_direction.csv', './PTL/ours')
    print(results["SVM"]["accuracy"])
    print(results["k-NN"]["accuracy"])
    print(results["Logistic Regression"]["accuracy"])
