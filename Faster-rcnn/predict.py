import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from model import get_model
import os
import json

def predict():
    # --- パラメータ設定 ---
    OUTPUT_DIR = 'outputs'
    # ↓↓↓ 予測したい画像のパスをここに指定してください ↓↓↓
    IMAGE_PATH = 'data/images/Trim_Trim_000002.jpg'
    
    MODEL_PATH = os.path.join(OUTPUT_DIR, 'fasterrcnn_model.pth')
    CLASS_DICT_PATH = os.path.join(OUTPUT_DIR, 'class_dict.json')
    
    # --- クラス情報のロード ---
    try:
        with open(CLASS_DICT_PATH, 'r') as f:
            class_dict = json.load(f)
    except FileNotFoundError:
        print(f"エラー: クラス辞書が見つかりません: {CLASS_DICT_PATH}")
        print("先に train.py を実行してモデルとクラス辞書を作成してください。")
        return

    # IDからクラス名への逆引き辞書を作成
    CLASS_NAMES = {int(v): k for k, v in class_dict.items()}
    NUM_CLASSES = len(CLASS_NAMES)

    # --- モデルのロード ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # --- 画像の読み込みと前処理 ---
    img = Image.open(IMAGE_PATH).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # --- 推論の実行 ---
    print("Running prediction...")
    with torch.no_grad():
        prediction = model(img_tensor)

    # --- 結果の描画 ---
    threshold = 0.5 # このスコア以上の物体のみ表示
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # PillowからOpenCV形式(BGR)へ変換

    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        if score > threshold:
            box = box.cpu().numpy().astype(int)
            label_id = label.item()
            class_name = CLASS_NAMES.get(label_id, 'Unknown')
            
            # バウンディングボックスを描画 (緑色)
            cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # ラベルとスコアを描画
            text = f"{class_name}: {score:.2f}"
            cv2.putText(img_cv, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- 結果の保存 ---
    output_filename = 'predicted_' + os.path.basename(IMAGE_PATH)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, img_cv)
    print(f"Prediction saved to {output_path}")

    # ローカル環境で実行している場合、結果を画面に表示することもできます
    # cv2.imshow('Prediction', img_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    predict()