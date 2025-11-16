from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path

def setup_environment():
    """設定訓練環境"""
    print("🚀 設定訓練環境...")
    
    # 確保 YAML 配置文件存在
    yaml_path = r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\diabetic_retinopathy.yaml"
    if not os.path.exists(yaml_path):
        print("❌ YAML 配置文件不存在，請先建立")
        return None
    
    # 檢查資料集結構
    base_path = Path(r"dataset\IDRiD\A. Segmentation\IDRiD_yolo")
    required_folders = ['images/train', 'images/val', 'labels/train', 'labels/val']
    
    for folder in required_folders:
        folder_path = base_path / folder
        if not folder_path.exists():
            print(f"❌ 缺少資料夾: {folder_path}")
            return None
        else:
            file_count = len(list(folder_path.glob('*')))
            print(f"✅ {folder}: {file_count} 個檔案")
    
    return yaml_path

def analyze_dataset():
    """分析資料集"""
    print("\n📊 分析資料集...")
    
    base_path = Path(r"dataset\IDRiD\A. Segmentation\IDRiD_yolo")
    
    for split in ['train', 'val']:
        images_dir = base_path / 'images' / split
        labels_dir = base_path / 'labels' / split
        
        # 統計圖片和標註
        image_files = list(images_dir.glob('*.*'))
        label_files = list(labels_dir.glob('*.txt'))
        
        # 計算有標註的圖片數量
        labeled_count = 0
        total_objects = 0
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    labeled_count += 1
                    total_objects += len(lines)
        
        print(f"  {split.upper()} 集:")
        print(f"    圖片: {len(image_files)} 個")
        print(f"    標註: {len(label_files)} 個")
        print(f"    有標註的圖片: {labeled_count} 個")
        print(f"    總物件數量: {total_objects} 個")
        
        if labeled_count > 0:
            print(f"    平均每圖物件: {total_objects/labeled_count:.2f} 個")

def train_diabetic_retinopathy_model():
    """訓練糖尿病視網膜病變檢測模型"""
    
    print("\n🎯 開始訓練糖尿病視網膜病變檢測模型...")
    
    # 設定環境
    yaml_path = setup_environment()
    if not yaml_path:
        return
    
    # 分析資料集
    analyze_dataset()
    
    try:
        # 加載模型
        print("\n📦 加載 YOLOv12s 模型...")
        model = YOLO('yolov12s.pt')  # 可以選擇 yolov8s.pt, yolov8m.pt 等
        
        # 訓練參數
        train_args = {
            'data': yaml_path,
            'epochs': 100,
            'imgsz': 640,
            'batch': 8,           # 視網膜圖片較大，使用較小的批次
            'patience': 20,       # 早停耐心值
            'save': True,
            'device': 'cpu',      # 使用 CPU
            'project': 'runs/detect',
            'name': 'diabetic_retinopathy_v1',
            'exist_ok': True,     # 允許覆蓋現有實驗
            'verbose': True,      # 顯示詳細輸出
        }
        
        print("🚀 開始訓練...")
        print("這可能需要一些時間，請耐心等待...")
        
        # 開始訓練
        results = model.train(**train_args)
        
        print("✅ 訓練完成！")
        
        # 顯示訓練結果
        if hasattr(results, 'results_dict'):
            print("\n📈 訓練結果:")
            for key, value in results.results_dict.items():
                print(f"  {key}: {value:.4f}")
        
        return model, results
        
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")
        return None, None

def validate_model(model):
    """驗證訓練好的模型"""
    if model is None:
        return
    
    print("\n🔍 驗證模型...")
    
    try:
        # 在驗證集上評估模型
        metrics = model.val()
        
        print("✅ 驗證完成！")
        print(f"📊 mAP50: {metrics.box.map50:.4f}")
        print(f"📊 mAP50-95: {metrics.box.map:.4f}")
        print(f"📊 精確度: {metrics.box.precision:.4f}")
        print(f"📊 召回率: {metrics.box.recall:.4f}")
        
    except Exception as e:
        print(f"❌ 驗證過程中發生錯誤: {e}")

def main():
    """主函數"""
    print("=" * 60)
    print("🩺 糖尿病視網膜病變檢測模型訓練")
    print("=" * 60)
    
    # 訓練模型
    model, results = train_diabetic_retinopathy_model()
    
    # 驗證模型
    if model:
        validate_model(model)
        
        print("\n🎉 訓練流程完成！")
        print("📁 訓練結果保存在: runs/detect/diabetic_retinopathy_v1")
        print("💡 您可以使用訓練好的模型進行預測:")
        print("   results = model('path/to/image.jpg')")
    
    print("\n" + "=" * 60)

# 執行主函數
if __name__ == "__main__":
    main()