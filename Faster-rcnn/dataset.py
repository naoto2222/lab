import torch
import os
import json
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    """
    LabelMe形式のJSONファイルに対応したカスタムデータセット。
    - 各画像に1つのJSONファイルが対応。
    - アノテーションフォルダ内の全JSONをスキャンしてクラスを自動検出。
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_dir = os.path.join(root, 'images')
        self.annotation_dir = os.path.join(root, 'annotations')
        
        # すべてのアノテーションファイル（.json）のパスを取得
        self.annotation_paths = [os.path.join(self.annotation_dir, f) 
                                 for f in sorted(os.listdir(self.annotation_dir)) if f.endswith('.json')]
        
        # すべてのJSONファイルを一度スキャンして、ユニークなクラス名のセットを作成
        self.class_names = self._get_all_class_names()
        
        # クラス名をIDにマッピングする辞書を作成 (背景=0は予約済、物体クラスは1から)
        self.class_dict = {name: i + 1 for i, name in enumerate(self.class_names)}
        
    def _get_all_class_names(self):
        class_names = set()
        for ann_path in self.annotation_paths:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            # data['shapes'] が存在し、かつ空でないことを確認
            if 'shapes' in data and data['shapes']:
                for shape in data['shapes']:
                    class_names.add(shape['label'])
        return sorted(list(class_names))

    def __getitem__(self, idx):
        # アノテーションファイルのパスを取得し、JSONを読み込む
        ann_path = self.annotation_paths[idx]
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        # 画像の読み込み
        image_path = os.path.join(self.image_dir, ann_data['imagePath'])
        img = Image.open(image_path).convert("RGB")
        
        boxes = []
        labels = []
        
        # アノテーション情報（shapes）をループ
        if 'shapes' in ann_data and ann_data['shapes']:
            for shape in ann_data['shapes']:
                if shape['shape_type'] == 'rectangle':
                    # ラベル名をIDに変換
                    label_name = shape['label']
                    labels.append(self.class_dict[label_name])
                    
                    # バウンディングボックスの座標を取得
                    points = shape['points']
                    xmin = min(points[0][0], points[1][0])
                    ymin = min(points[0][1], points[1][1])
                    xmax = max(points[0][0], points[1][0])
                    ymax = max(points[0][1], points[1][1])
                    boxes.append([xmin, ymin, xmax, ymax])

        # Tensorに変換
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.tensor(0, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.annotation_paths)