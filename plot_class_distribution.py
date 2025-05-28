import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def plot_class_distribution(train_dir):
    # Lấy danh sách các class từ thư mục train
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # Đếm số lượng ảnh trong mỗi class
    class_counts = {}
    for class_name in classes:
        class_path = os.path.join(train_dir, class_name)
        num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = num_images
    
    # Sắp xếp các class theo số lượng ảnh giảm dần
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_names = [x[0] for x in sorted_classes]
    counts = [x[1] for x in sorted_classes]
    
    # Tạo figure với kích thước lớn hơn
    plt.figure(figsize=(15, 8))
    
    # Vẽ bar plot
    bars = plt.bar(range(len(class_names)), counts)
    
    # Thêm số liệu lên trên mỗi bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Cấu hình đồ thị
    plt.title('Distribution of Images Across Classes in Training Set', fontsize=14, pad=20)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Thêm grid để dễ đọc
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Điều chỉnh layout để tránh cắt nhãn
    plt.tight_layout()
    
    # Lưu đồ thị
    plt.savefig('class_distribution_after_augmentation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # In thông tin chi tiết
    print("\nClass Distribution Details:")
    print("-" * 50)
    print(f"Total number of classes: {len(classes)}")
    print(f"Total number of images: {sum(counts)}")
    print("\nNumber of images per class:")
    for class_name, count in sorted_classes:
        print(f"{class_name}: {count} images")
    
    # Tính toán và in các thống kê
    print("\nStatistics:")
    print(f"Maximum images in a class: {max(counts)}")
    print(f"Minimum images in a class: {min(counts)}")
    print(f"Average images per class: {np.mean(counts):.2f}")
    print(f"Standard deviation: {np.std(counts):.2f}")

if __name__ == "__main__":
    train_dir = "./GastroVision/train"
    plot_class_distribution(train_dir) 