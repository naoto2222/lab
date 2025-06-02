import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import get_model
from dataset import CustomDataset
import os
import json

def main():
    # --- パラメータ設定 ---
    DATA_DIR = 'data'
    OUTPUT_DIR = 'outputs'
    NUM_EPOCHS = 10
    BATCH_SIZE = 2 # メモリに応じて調整してください
    LEARNING_RATE = 0.005
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- デバイス設定 ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- データセットとデータローダー ---
    transforms = T.Compose([T.ToTensor()])
    
    dataset = CustomDataset(root=DATA_DIR, transforms=transforms)
    
    num_classes = len(dataset.class_names)
    class_dict = dataset.class_dict
    print(f"Found {num_classes} classes: {class_dict}")

    with open(os.path.join(OUTPUT_DIR, 'class_dict.json'), 'w') as f:
        json.dump(class_dict, f, indent=4)
    print(f"Class dictionary saved to {os.path.join(OUTPUT_DIR, 'class_dict.json')}")

    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # --- モデル、オプティマイザ ---
    model = get_model(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- 学習ループ ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                print(f"WARNING: NaN loss detected at iteration {i+1}. Skipping batch.")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}")

        epoch_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
        lr_scheduler.step()

    # --- モデルの保存 ---
    model_save_path = os.path.join(OUTPUT_DIR, 'fasterrcnn_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining finished. Model saved to {model_save_path}")

if __name__ == '__main__':
    main()