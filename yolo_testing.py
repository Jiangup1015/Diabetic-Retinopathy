import cv2
import random
import os
def visualize_annotations(image_dir, label_dir, num_samples=5):
    """隨機可視化幾個樣本檢查標註是否正確"""
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]
    
    if len(image_files) == 0:
        print("❌ 沒有找到圖片檔案")
        return
    
    # 隨機選擇幾個樣本
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"\n🎨 可視化 {len(samples)} 個隨機樣本...")
    
    for sample in samples:
        # 圖片路徑
        img_path = os.path.join(image_dir, sample)
        
        # 對應的標註路徑
        label_path = os.path.join(label_dir, os.path.splitext(sample)[0] + '.txt')
        
        # 讀取圖片
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 無法讀取圖片: {img_path}")
            continue
        
        # 讀取標註
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                annotations = f.readlines()
            
            # 繪製邊界框
            for ann in annotations:
                parts = ann.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    
                    # 轉換為像素座標
                    h, w = img.shape[:2]
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # 繪製邊界框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'Class {int(class_id)}', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            print(f"✅ {sample}: {len(annotations)} 個標註")
            
            # 顯示圖片
            cv2.imshow(f'Annotation: {sample}', img)
            cv2.waitKey(2000)  # 顯示2秒
            cv2.destroyAllWindows()
        else:
            print(f"⚠️  {sample}: 沒有對應的標註檔案")

# 可視化訓練集和驗證集
print("訓練集樣本:")
visualize_annotations(
    r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\images\train",
    r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\labels\train"
)

print("\n驗證集樣本:")
visualize_annotations(
    r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\images\val",
    r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\labels\val"
)