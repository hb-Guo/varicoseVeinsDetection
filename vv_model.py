import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

from model import resnet34, resnet50
from lightweight_seg import BackgroundRemovalTransform


class Cutout:
    """
    随机遮挡图像的部分区域，强迫模型关注更多特征
    """

    def __init__(self, n_holes=1, length=40):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomErase:
    """
    随机擦除，模拟遮挡效果
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if np.random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)

                img[0, x1 : x1 + h, y1 : y1 + w] = np.random.uniform(0, 1)
                img[1, x1 : x1 + h, y1 : y1 + w] = np.random.uniform(0, 1)
                img[2, x1 : x1 + h, y1 : y1 + w] = np.random.uniform(0, 1)

                return img

        return img


# 使用示例
from torchvision import transforms

# 增强的数据预处理管道
transform_with_cutout = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Cutout(n_holes=3, length=20),  # 随机遮挡3个区域
        RandomErase(probability=0.5),  # 50%概率随机擦除
    ]
)


# 1. 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 2. 注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


# 3. 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道维度的最大值和平均值
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # 拼接并生成空间注意力图
        attention = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        return x * attention


# 4. 改进的CNN模型（带注意力机制）
class AttentionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(AttentionCNN, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attention1 = AttentionBlock(32)
        self.spatial_att1 = SpatialAttention()

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention2 = AttentionBlock(64)
        self.spatial_att2 = SpatialAttention()

        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attention3 = AttentionBlock(128)
        self.spatial_att3 = SpatialAttention()

        # 第四个卷积块（增加深度）
        self.conv4 = nn.Conv2d(128, 256, kernel_size=9, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积块1 + 通道注意力 + 空间注意力
        x1 = x
        x = self.relu(self.bn1(self.conv1(x)))  # (4,3,128,128)
        x = self.attention1(x)
        # x = self.spatial_att1(x)
        x = self.pool(x)  # 128 -> 64

        # 卷积块2 + 通道注意力 + 空间注意力
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        # x = self.spatial_att2(x)
        x = self.pool(x)  # 64 -> 32

        # 卷积块3 + 通道注意力 + 空间注意力
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)
        x = self.spatial_att3(x)
        x = self.pool(x)  # 32 -> 16

        # 卷积块4
        # x = self.relu(self.bn4(self.conv4(x)))
        # x = self.pool(x)  # 16 -> 8

        # 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 5. 数据加载函数
def load_data(data_dir):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    image_paths.append(os.path.join(class_path, img_name))
                    labels.append(class_to_idx[class_name])

    return image_paths, labels, class_names


# 6. 训练函数（增加注意力图可视化）
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    train_sampler,
):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0.0
    save_path = "./resNet34.pth"

    for epoch in tqdm(range(num_epochs)):
        train_sampler.set_epoch(epoch)
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            correct = torch.tensor(correct, device=device)
            total = torch.tensor(total, device=device)
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        acc = 0.0
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct = torch.tensor(correct, device=device)
                total = torch.tensor(total, device=device)
                dist.all_reduce(correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                acc += torch.eq(predicted, labels).sum().item()
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            if dist.get_rank() == 0:
                torch.save(model.module.state_dict(), "resnet34_ddp.pth")
                # torch.save(model.state_dict(), save_path)
        if dist.get_rank() == 0:
            print(f"Epoch [{epoch}/{num_epochs}]")
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("-" * 50)

    return train_losses, val_losses, train_accs, val_accs


# 7. 绘制训练曲线
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label="Train Accuracy")
    ax2.plot(val_accs, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


class ResizeWithPad:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w))

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        img = F.pad(
            img, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        )
        return img


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# 8. 主程序
def main():
    # 设置参数
    data_dir = "./data"
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0001

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 增强的数据预处理（针对腿部特征）
    # transform_train = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(10),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    #     transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])

    # transform_val = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])

    transform_train = transforms.Compose(
        [
            ResizeWithPad(320),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transform_val = transforms.Compose(
        [
            ResizeWithPad(320),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 加载数据
    image_paths, labels, class_names = load_data(data_dir)
    print(f"总样本数: {len(image_paths)}")
    print(f"类别: {class_names}")

    # 划分训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 创建数据集
    train_dataset = ImageDataset(train_paths, train_labels, transform_train)
    val_dataset = ImageDataset(val_paths, val_labels, transform_val)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=16, sampler=val_sampler, num_workers=4, pin_memory=True
    )
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    net = resnet34()
    model_weight_path = "./resnet34-pre.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location="cpu"))
    in_channel = net.fc.in_features

    net.fc = nn.Linear(in_channel, len(class_names))
    net = net.to(device)

    net = DDP(net, device_ids=[local_rank])
    # 创建注意力增强模型
    # model = AttentionCNN(num_classes=len(class_names)).to(device)

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 学习率调度器
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=10, verbose=True
    # )

    # 训练模型
    print("开始训练带注意力机制的模型...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        net,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        num_epochs,
        device,
        train_sampler,
    )

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 保存模型
    # torch.save(model.state_dict(), './model/attention_cnn_classifier.pth')
    print("模型已保存为 attention_cnn_classifier.pth")


if __name__ == "__main__":
    main()
