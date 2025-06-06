import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from model import get_model
import os
import json
import csv
import functools # Import functools for cmp_to_key

def predict_video_to_csv():
    # --- 1. パラメータとパスの設定 ---
    OUTPUT_DIR = 'outputs'
    
    # ↓↓↓ 予測したい動画のパスをここに指定してください ↓↓↓
    VIDEO_PATH = 'data/videos/your_video.mp4'  # 例: 'data/videos/test.mp4'

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

    # --- 3. 動画とCSVファイルの準備 ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません: {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    all_frames_results_for_csv = []
    
    frame_count = 0
    transform = T.Compose([T.ToTensor()])
    print("\n動画の処理を開始します... (全フレームの解析後にCSVを作成します)")

    fixed_y_tolerance = None  # 最初のフレームのピラーの高さに基づいて設定される
    first_pillar_height_determined = False

    # --- Custom comparison function for sorting pillars ---
    def compare_pillars(det1, det2, y_tolerance):
        box1 = det1['box']
        box2 = det2['box']
        
        ymin1 = box1[1]
        ymin2 = box2[1]
        
        # y座標の差が許容範囲内であれば、同じ行とみなす
        if abs(ymin1 - ymin2) <= y_tolerance:
            # 同じ行ならx座標でソート (左から右へ)
            return box1[0] - box2[0]
        else:
            # 異なる行ならy座標でソート (上から下へ)
            return ymin1 - ymin2

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
        
        detected_pillars_info = []

        for i in range(len(prediction[0]['scores'])):
            score = prediction[0]['scores'][i].item()

            if score > DETECTION_THRESHOLD:
                label_id = prediction[0]['labels'][i].item()
                class_name = CLASS_NAMES.get(label_id, 'Unknown')
                
                if class_name == 'pillar':
                    box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
                    detected_pillars_info.append({
                        'box': box,
                        'score': score, 
                        'class_name': class_name
                    })
        
        # --- 最初のフレームでのpillarの高さに基づいて許容範囲を設定 ---
        if not first_pillar_height_determined and detected_pillars_info:
            # 最初のフレームで検出された最初のpillarの情報を取得
            first_pillar_box = detected_pillars_info[0]['box']
            reference_pillar_height = first_pillar_box[3] - first_pillar_box[1] # ymax - ymin
            
            if reference_pillar_height > 0:
                fixed_y_tolerance = reference_pillar_height / 3.0
                print(f"基準となるピラーの高さ: {reference_pillar_height}, Y方向の許容範囲: {fixed_y_tolerance:.2f} ピクセル (フレーム {frame_count} で決定)")
            else:
                # 最初のピラーの高さが0または負の場合のフォールバック
                fixed_y_tolerance = frame_height / 15.0 # 例: フレームの高さの1/15
                print(f"警告: 最初のフレームのピラーの高さが0以下でした。デフォルトの許容範囲を使用します: {fixed_y_tolerance:.2f} ピクセル")
            first_pillar_height_determined = True
        
        # まだ許容範囲が設定されていない場合（最初のフレームにpillarがなかったなど）はデフォルト値を使用
        current_tolerance_to_use = fixed_y_tolerance if fixed_y_tolerance is not None else frame_height / 15.0
        if fixed_y_tolerance is None and frame_count == 0 and not detected_pillars_info:
            print(f"最初のフレームにピラーが検出されなかったため、デフォルトのY許容範囲 ({current_tolerance_to_use:.2f} ピクセル) を使用します。")


        # 検出されたpillarを位置に基づいてソート
        if detected_pillars_info:
            detected_pillars_info.sort(key=functools.cmp_to_key(
                lambda item1, item2: compare_pillars(item1, item2, current_tolerance_to_use)
            ))
        
        current_frame_widths_for_csv = []
        pillar_display_count = 0 

        for det_info in detected_pillars_info:
            pillar_display_count += 1
            box = det_info['box']
            class_name = det_info['class_name']
            
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            current_frame_widths_for_csv.append(width)
            
            text = f"{class_name} {pillar_display_count}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        all_frames_results_for_csv.append([frame_count] + current_frame_widths_for_csv)
        video_writer.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0: # 30フレームごとに進捗を表示
            print(f"  {frame_count} フレーム処理完了...")

    print(f"\n全 {frame_count} フレームの解析が完了しました。")

    # --- 5. CSVファイルへの書き出し ---
    print(f"CSVファイルを作成中: {OUTPUT_CSV_PATH}")
    max_pillars = 0
    if all_frames_results_for_csv:
        max_pillars = max(len(row) for row in all_frames_results_for_csv) - 1

    header = ['frame'] + [f'pillar_{i+1}_width' for i in range(max_pillars)]

    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(all_frames_results_for_csv)

    # --- 6. 終了処理 ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    print("\nすべての処理が完了しました。")
    print(f"予測結果ビデオ: {OUTPUT_VIDEO_PATH}")
    print(f"Pillar幅データCSV: {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    # フォルダの準備（なければ作成）
    if not os.path.exists('data/videos'):
        os.makedirs('data/videos')
        print("`data/videos` フォルダを作成しました。ここに動画ファイルを入れてください。")
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    predict_video_to_csv()