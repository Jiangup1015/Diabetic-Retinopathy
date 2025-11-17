import os
from pathlib import Path

def check_dataset_structure(base_path):
    """檢查 YOLO 資料集結構"""
    base = Path(base_path)
    
    print("📁 檢查資料集結構...")
    
    # 檢查必要的資料夾
    required_folders = [
        'images/train',
        'images/val', 
        'labels/train',
        'labels/val'
    ]
    
    for folder in required_folders:
        folder_path = base / folder
        if folder_path.exists():
            file_count = len(list(folder_path.glob('*')))
            print(f"✅ {folder}: {file_count} 個檔案")
        else:
            print(f"❌ {folder}: 不存在")
    
    # 檢查圖片和標註對應關係
    print("\n🔍 檢查圖片和標註對應關係...")
    
    for split in ['train', 'val']:
        images_dir = base / 'images' / split
        labels_dir = base / 'labels' / split
        
        if images_dir.exists() and labels_dir.exists():
            image_files = {f.stem for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg']}
            label_files = {f.stem for f in labels_dir.glob('*.txt')}
            
            common_files = image_files & label_files
            only_images = image_files - label_files
            only_labels = label_files - image_files
            
            print(f"\n{split.upper()} 集:")
            print(f"  ✅ 圖片和標註都有的: {len(common_files)}")
            print(f"  ⚠️ 只有圖片的: {len(only_images)}")
            print(f"  ⚠️ 只有標註的: {len(only_labels)}")
            
            if only_images:
                print(f"    只有圖片的檔案: {list(only_images)[:5]}...")
            if only_labels:
                print(f"    只有標註的檔案: {list(only_labels)[:5]}...")

# 使用範例
check_dataset_structure(r"dataset\IDRiD\A. Segmentation\IDRiD_yolo")