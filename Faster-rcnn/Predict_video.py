import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from model import get_model
import os
import json
import csv

def predict_video_to_csv():
    # --- 1. パラメータとパスの設定 ---
    OUTPUT_DIR = 'outputs'
    
    # ↓↓↓ 予測したい動画のパスをここに指定してください ↓↓↓
    VIDEO_PATH = 'data/videos/.mp4'  # 例: 'data/videos/test.mp4'

    # 入力ビデオファイル名から出力ファイル名を生成
    video_filename = os.path.basename(VIDEO_PATH)
    video_name_without_ext = os.path.splitext(video_filename)[0]
    
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, f'predicted_{video_filename}')
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, f'pillar_widths_{video_name_without_ext}.csv')
    
    MODEL_PATH = os.path.join(OUTPUT_DIR, 'fasterrcnn_model.pth')
    CLASS_DICT_PATH = os.path.join(OUTPUT_DIR, 'class_dict.json')
    
    DETECTION_THRESHOLD = 0.6  # 検出の閾値（必要に応じて調整）

    # --- 2. モデルとクラス情報のロード ---
    try:
        with open(CLASS_DICT_PATH, 'r') as f:
            class_dict = json.load(f)
    except FileNotFoundError:
        print(f"エラー: クラス辞書が見つかりません: {CLASS_DICT_PATH}")
        print("先に train.py を実行してモデルとクラス辞書を作成してください。")
        return

    CLASS_NAMES = {int(v): k for k, v in class_dict.items()}
    NUM_CLASSES = len(CLASS_NAMES)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model = get_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # --- 3. 動画の準備とデータの一時保存用リスト ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません: {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    # 全フレームの結果を一時的に保存するリスト
    all_frames_results = []
    
    frame_count = 0
    transform = T.Compose([T.ToTensor()])
    print("\n動画の処理を開始します... (全フレームの解析後にCSVを作成します)")

    # --- 4. フレームごとの処理ループ ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)
        
        # このフレームで検出されたpillarの幅を保存するリスト
        current_frame_widths = []

        for i in range(len(prediction[0]['scores'])):
            score = prediction[0]['scores'][i].item()

            if score > DETECTION_THRESHOLD:
                label_id = prediction[0]['labels'][i].item()
                class_name = CLASS_NAMES.get(label_id, 'Unknown')

                # 'pillar'クラスの場合のみ処理
                if class_name == 'pillar':
                    box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    current_frame_widths.append(width)
                    
                    # 映像への描画は通常通り行う
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    text = f"{class_name}: {score:.2f}"
                    cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # フレーム番号と、このフレームで見つかったpillarの幅リストを保存
        all_frames_results.append([frame_count] + current_frame_widths)

        video_writer.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  {frame_count} フレーム処理完了...")

    print(f"\n全 {frame_count} フレームの解析が完了しました。")

    # --- 5. CSVファイルへの書き出し ---
    print(f"CSVファイルを作成中: {OUTPUT_CSV_PATH}")

    # 最大の列数を決定（frame列 + 最も多くpillarが検出された数）
    max_pillars = 0
    if all_frames_results:
        max_pillars = max(len(row) for row in all_frames_results) - 1

    # ヘッダーを作成
    header = ['frame'] + [f'pillar_{i+1}_width' for i in range(max_pillars)]

    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header) # ヘッダーを書き込み
        csv_writer.writerows(all_frames_results) # 全データを一括で書き込み

    # --- 6. 終了処理 ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    print("\nすべての処理が完了しました。")
    print(f"予測結果ビデオ: {OUTPUT_VIDEO_PATH}")
    print(f"Pillar幅データCSV: {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    if not os.path.exists('data/videos'):
        os.makedirs('data/videos')
        print("`data/videos` フォルダを作成しました。ここに動画ファイルを入れてください。")
        
    predict_video_to_csv()