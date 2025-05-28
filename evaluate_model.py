import torch
from torchvision import models, transforms, datasets
from torch import nn
from torch.utils.data import DataLoader
from Test import test_net
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./checkpoints/C_15_8.pth"  # <== Thay bằng file bạn muốn test
test_data_path = "./GastroVision/test"
batch_size = 8
n_classes = 20

print(f"🔍 Loading model from: {model_path}")
model = models.resnet50(weights=None)
n_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(n_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, n_classes),
    nn.LogSoftmax(dim=1)
)
model.to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4762, 0.3054, 0.2368], [0.3345, 0.2407, 0.2164])
])

test_dataset = datasets.ImageFolder(test_data_path, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
print(f"🧪 Loaded {len(test_dataset)} test images.")

criterion = nn.NLLLoss()

with torch.no_grad():
    test_loss, test_metrics, test_num_steps = test_net(model, test_loader, device, criterion)

print("\n📊 KẾT QUẢ TEST:")
print(f"Test Loss: {test_loss:.4f}")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")
print("\n✅ Đã vẽ confusion matrix (lưu ở conf.png)")
