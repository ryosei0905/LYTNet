import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

# パスを指定
csv_file = "./Annotations/training_file.csv"
img_dir = "./PTL/480-480/train"

# CSVを読み込み
df = pd.read_csv(csv_file)

# 確認したい枚数
num_samples = 5  

# ランダムにサンプルを選ぶ
indices = random.sample(range(len(df)), num_samples)

for idx in indices:
    row = df.iloc[idx]
    img_name = row[0]  # ファイル名
    label = row[1]    # クラス
    x1, y1, x2, y2 = row[2:6]  # 座標

    # 座標を4032x3024基準 → 480基準に変換
    x1 = x1 * (480 / 4032)
    x2 = x2 * (480 / 4032)
    y1 = y1 * (480 / 3024)
    y2 = y2 * (480 / 3024)

    # 画像を開く
    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path)

    # matplotlibで表示
    plt.imshow(image)
    plt.plot([x1, x2], [y1, y2], color="red", linewidth=2)  # ラインを引く
    plt.title(f"File: {img_name}, Class: {label}")
    plt.show()
