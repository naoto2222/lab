import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from model import get_model
from dataset import CustomDataset
import os
import json
import numpy as np

def continue_training():
    # --- 1. パラメータ設定 ---
    DATA_DIR = 'data'
    OUTPUT_DIR = 'outputs'
    
    # ↓↓↓ 追加学習のベースとなるモデルのパスを指定 ↓↓↓
    LOAD_MODEL_PATH = os.path.join(OUTPUT_DIR, 'fasterrcnn_model_best.pth')

    # ↓↓↓ 追加学習で実行するエポック数 ↓↓↓
    NUM_EPOCHS = 10  # 例えば、さらに10エポック学習する

    # ↓↓↓ 追加学習時の学習率。通常は元の学習率より小さくする ↓↓↓
    LEARNING_RATE = 0.0005 # 元が0.005だったので、1/10に設定
    
    BATCH_SIZE = 2
    VALIDATION_SPLIT = 0.2
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 2. デバイス設定 ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- 3. データセットの準備と分割 ---
    transforms = T.Compose([T.ToTensor()])
    dataset = CustomDataset(root=DATA_DIR, transforms=transforms)
    
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
    
    print(f"Training data size: {len(dataset_train)}")
    print(f"Validation data size: {len(dataset_val)}")

    num_classes = len(dataset.class_names)
    print(f"Found {num_classes} classes: {dataset.class_dict}")
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # --- 4. モデルの準備と重みの読み込み ---
    model = get_model(num_classes=num_classes)
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ ここで、既存のモデルの重みを読み込む ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if os.path.exists(LOAD_MODEL_PATH):
        print(f"Loading weights from existing model: {LOAD_MODEL_PATH}")
        model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
    else:
        print(f"ERROR: Model file not found at {LOAD_MODEL_PATH}. Cannot continue training.")
        print("Please check the path or run train.py first to create a model.")
        return # モデルがない場合は終了
    
    model.to(device)

    # --- 5. オプティマイザとスケジューラ ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- 6. 学習と検証ループ ---
    # 現在の最良lossをファイルから推定するか、無限大から開始する
    # ここでは簡単のため、無限大から開始。これにより、読み込んだモデルより性能が良ければすぐに保存される
    best_val_loss = float('inf') 
    print(f"\nContinuing training for {NUM_EPOCHS} more epochs...")

    for epoch in range(NUM_EPOCHS):
        # --- 訓練フェーズ ---
        model.train()
        train_loss_total = 0
        print(f"\n--- Additional Epoch {epoch+1}/{NUM_EPOCHS} ---")
        for i, (images, targets) in enumerate(data_loader_train):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                print(f"WARNING: NaN loss detected. Skipping batch.")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss_total += losses.item()
        
        avg_train_loss = train_loss_total / len(data_loader_train)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # --- 検証フェーズ ---
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for images, targets in data_loader_val:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss_total += losses.item()
        
        avg_val_loss = val_loss_total / len(data_loader_val)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        # --- 最良モデルの保存 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # これまでの最高のモデルを上書き保存
            model_save_path = os.path.join(OUTPUT_DIR, 'fasterrcnn_model_best.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"✨ New best model saved to {model_save_path} with validation loss: {best_val_loss:.4f}")

        lr_scheduler.step()

    print("\nAdditional training finished.")
    print(f"Best validation loss during this session: {best_val_loss:.4f}")
    print("The best model is saved as 'fasterrcnn_model_best.pth'")

if __name__ == '__main__':
    continue_training()