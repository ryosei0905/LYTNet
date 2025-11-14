import os
import sys
import xml.etree.ElementTree as ET
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# === 入出力設定 ===
INPUT  = './annotations.xml'
OUTPUT = './annotations.csv'

if not os.path.exists(INPUT):
    logging.error(f"入力XMLファイルが見つかりません: {INPUT}")
    sys.exit(1)

try:
    tree = ET.parse(INPUT)
    root = tree.getroot()
except ET.ParseError as e:
    logging.error("XML解析エラー: %s", e)
    sys.exit(1)

rows = []

# --- 画像単位で走査 ---
for img in root.findall("image"):
    image_name = img.get("name", "")

    zenra_left   = 1 if img.find(".//*[@label='Zebra_Left']")   is not None else 0
    zenra_center = 1 if img.find(".//*[@label='Zebra_Center']") is not None else 0
    zenra_right  = 1 if img.find(".//*[@label='Zebra_Right']")  is not None else 0

    # 3つの合計が0のときは追加しない
    if zenra_left + zenra_center + zenra_right > 0:
        rows.append([image_name, zenra_left, zenra_center, zenra_right])

# --- CSV出力 ---
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "Zebra_Left", "Zebra_Center", "Zebra_Right"])
    writer.writerows(rows)

logging.info("CSV出力完了: %s", OUTPUT)
