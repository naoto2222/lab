import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    """
    事前学習済みのFaster R-CNNモデルをロードし、分類器をカスタムクラス数に合わせて変更する。
    """
    # ImageNetで事前学習済みのモデルをロード
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # 分類器の入力特徴量数を取得
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 事前学習済みのヘッドを新しいものに置き換え
    # num_classesには背景クラスを含めず、物体クラスの数だけを指定する。
    # (内部で背景クラスの+1が自動的に行われる)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model