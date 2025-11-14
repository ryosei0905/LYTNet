import cv2
import os
import glob

# 入力フォルダと出力フォルダを指定
INPUT_DIR = "./lytnet/PTL/train"       # 元の画像が入っているフォルダ
OUTPUT_DIR = "./lytnet/PTL_c/train"     # 縮小画像を保存するフォルダ

# 出力フォルダがなければ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 対象とする画像拡張子
extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]



# 画像をすべて読み込み
for ext in extensions:
    for filepath in glob.glob(os.path.join(INPUT_DIR, ext)):
        # 画像読み込み
        img = cv2.imread(filepath)
        if img is None:
            print(f"読み込めませんでした: {filepath}")
            continue
        
        h, w = img.shape[:2]

        # 正方形480pxの中心座標
        center_y, center_x = h // 2, w // 2
        
        # cropsize
        half = 480 // 2 

        top    = max(center_y - half, 0)
        bottom = min(center_y + half, h)
        left   = max(center_x - half, 0)
        right  = min(center_x + half, w)
        
        #crop
        cropped = img[top:bottom, left:right]

        # ファイル名を取得して保存
        filename = os.path.basename(filepath)
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, cropped)
        print(f"保存しました: {save_path}")
