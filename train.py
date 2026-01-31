import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from sklearn.metrics import recall_score

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

from model import resnet34, resnet50, resnet101, resnet18


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
                    w, h = Image.open(image_paths[-1]).size
                    if w < 512 or h < 512:
                        image_paths.pop()
                        continue
                    labels.append(class_to_idx[class_name])

    return image_paths, labels, class_names

from timm.data import Mixup

# 配置 Mixup 和 Cutmix
mixup_fn = Mixup(
        mixup_alpha=0.2,       # Mixup 强度（建议 0.8）
        cutmix_alpha=1.0,      # Cutmix 强度（建议 1.0）
        prob=0.5,              # 每次 batch 有多少概率触发 Mixup/Cutmix
        switch_prob=0.5,       # 在 Mixup 和 Cutmix 之间切换的概率
        mode='batch',          # 以 batch 为单位进行变换
        label_smoothing=0.1,   # 标签平滑
        num_classes=6          # 你的分类数
)

def set_seed(seed: int = 42, deterministic: bool = True):
    import os
    import random
    import numpy as np
    import torch

    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 环境变量（hash & CUDA）
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuDNN
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # PyTorch >= 1.8
        torch.use_deterministic_algorithms(True)

        # CUDA 11+
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    else:
        # 更快，但不可复现
        torch.backends.cudnn.benchmark = True


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
    best_recall = 0.0
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

            ordinal_targets = label_to_ordinal(labels)

            optimizer.zero_grad()
            # samples, targets = mixup_fn(images, labels)
            outputs = model(images)
            # loss = criterion(outputs, labels)

            weights = torch.tensor([1.0, 1.0, 3.0, 1.0, 1.0]).to(device) # 给中间神经元 3 倍权重
            # 手动计算 BCE Loss 并加权
            loss_matrix = F.binary_cross_entropy_with_logits(outputs, ordinal_targets, reduction='none')
            loss = (loss_matrix * weights).mean()

            # loss = criterion(outputs, ordinal_targets)
            # print(loss)
            loss.backward()
            optimizer.step()
            # optimizer.first_step(zero_grad=True)

            # outputs = model(images)
            # criterion(outputs, labels).backward()
            # optimizer.second_step(zero_grad=True)

            probs = torch.sigmoid(outputs) # 转换为概率
            # 统计大于 0.5 的个数。例如有 2 个大于 0.5，说明是 C3 (索引为2)
            predicted = (probs > 0.5).sum(dim=1)
            # print("predicted: ", predicted.cpu().numpy())
            # print("label: ", labels.cpu().numpy())

            running_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                ordinal_targets = label_to_ordinal(labels)

                outputs = model(images)
                loss = criterion(outputs, ordinal_targets)

                probs = torch.sigmoid(outputs) # 转换为概率
                # 统计大于 0.5 的个数。例如有 2 个大于 0.5，说明是 C3 (索引为2)
                predicted = (probs > 0.5).sum(dim=1)
                # print("predicted: ", predicted.cpu().numpy())
                # print("label: ", labels.cpu().numpy())
                val_loss += loss.item()
                # _, predicted = torch.max(outputs.data, 1)

                acc += torch.eq(predicted, labels).sum().item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        # macro recall
        # val_recall = recall_score(all_labels, all_preds, average='macro') * 100

        if val_acc > best_acc:
            best_acc = val_acc
            if dist.get_rank() == 0:
                print("最佳模型已更新，保存中...")
                torch.save(model.module.state_dict(), "aug1_new_resnet50_ddp.pth")
                # torch.save(model.state_dict(), save_path)
        # elif val_recall > best_recall:
        #     best_recall = val_recall
        #     if dist.get_rank() == 0:
        #         print("最佳模型已更新，保存中...")
        #         torch.save(model.module.state_dict(), "new_resnet50_ddp.pth")
        #         # torch.save(model.state_dict(), save_path)

        if dist.get_rank() == 0:
            print(f"Epoch [{epoch}/{num_epochs}]")
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("-" * 50)

    return train_losses, val_losses, train_accs, val_accs


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
        img = TF.resize(img, (new_h, new_w))

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        img = TF.pad(
            img, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        )
        # img = TF.pad(
        #     img,
        #     (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
        #     fill=tuple(int(x * 255) for x in [0.485, 0.456, 0.406]),
        # )
        return img


def setup_ddp():
    """
    启动并行训练
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def freeze_all_except_fc(net):
    for name, param in net.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_layer4_and_fc(net):
    for name, param in net.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(net):
    for param in net.parameters():
        param.requires_grad = True


def run_one_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    train_sampler,
    stage_name="",
):
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        correct, total, running_loss = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        val_acc = 100 * correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            if dist.get_rank() == 0:
                print("最佳模型已更新，保存中...")
                torch.save(
                    model.module.state_dict(), stage_name + "new_resnet50_ddp.pth"
                )
                # torch.save(model.state_dict(), save_path)
        if dist.get_rank() == 0:
            print(f"Epoch [{epoch}/{num_epochs}]")
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("-" * 50)
        if dist.get_rank() == 0:
            print(
                f"[{stage_name}] Epoch {epoch+1}/{num_epochs} "
                f"Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%"
            )


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        logits: [B, C]
        targets: [B]  (类别索引, 0 ~ C-1)
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # 取出真实类别对应的概率 p_t
        targets = targets.view(-1, 1)
        pt = probs.gather(1, targets).squeeze(1)
        log_pt = log_probs.gather(1, targets).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets.squeeze(1))
            loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -((1 - pt) ** self.gamma) * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to original position
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self):
        raise NotImplementedError("SAM requires first_step and second_step")

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2).to(device))
        if len(norms) == 0:
            return torch.tensor(0.0).to(device)
        return torch.norm(torch.stack(norms), p=2)


class SEBlock(nn.Module):
    """SE注意力块 - 最常用于分类任务"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, channels)
        batch_size, channels = x.size()

        # Squeeze: 全局平均池化
        y = x.view(batch_size, channels, 1)
        y = self.squeeze(y).view(batch_size, channels)

        # Excitation: 学习通道权重
        y = self.excitation(y)

        # Scale: 重标定
        return x * y


class ClassifierWithSE(nn.Module):
    def __init__(self, in_channel, num_classes, dropout_prob=0.5):
        super(ClassifierWithSE, self).__init__()
        self.se = SEBlock(in_channel, reduction_ratio=16)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob), nn.Linear(in_channel, num_classes)
        )

    def forward(self, x):
        x = self.se(x)
        x = self.classifier(x)
        return x

import torch.nn.functional as F

def label_to_ordinal(labels, num_classes=6):
    # labels: [batch_size]
    # return: [batch_size, num_classes - 1]
    batch_size = labels.size(0)
    ordinal_labels = torch.zeros(batch_size, num_classes - 1).to(labels.device)
    for i in range(batch_size):
        ordinal_labels[i, :labels[i]] = 1
    return ordinal_labels

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, s=30.0):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # 缩放因子，非常重要
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 1. 对输入特征进行归一化
        x_norm = F.normalize(x, p=2, dim=1)
        # 2. 对权重进行归一化
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # 3. 计算余弦相似度并缩放
        # 余弦值范围在 [-1, 1]，如果不乘 s，Softmax 后的概率分布会太“平”，导致不收敛
        logits = torch.mm(x_norm, w_norm.t())
        return self.s * logits

class VaricoseNet(nn.Module):
    def __init__(self, backbone, num_classes=6):
        super().__init__()
        self.backbone = backbone            # DINO ViT
        # print(backbone)
        
        # self.backbone = resnet34()
        # state = torch.load("./resnet34-pre.pth", map_location="cpu", weights_only=True)
        # self.backbone.load_state_dict(state, strict=False)
        # self.classifier = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(1000, 5))# 0.5效果最好

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True

        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        embed_dim = 768

        # self.se_module = SEBlock(embed_dim * 2) 

        self.norm = nn.LayerNorm(embed_dim * 2) 
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim *2, 256)  ,   # 先降维，减少参数
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.7),                  # 0.5效果好
            nn.Linear(256, 5)
            # CosineLinear(256, num_classes, s=15)
        )

    def forward(self, x):

        features = self.backbone.get_intermediate_layers(x, n=1)[0]
        
        cls_token = features[:, 0]              # [B, 768]
        patch_tokens = features[:, 1:]          # [B, N, 768]

        # 2. 全局平均池化 (GAP)
        gap = torch.mean(patch_tokens, dim=1)   # [B, 768]

        # 3. 拼接 CLS 和 GAP
        combined = torch.cat([cls_token, gap], dim=1) # [B, 1536]

        # 4. 分类
        combined = self.norm(combined)



        # features = self.backbone(x)

        out = self.classifier(combined)
        return out

device = torch.device("cuda")



def main():
    set_seed(42,deterministic=True)
    train_data_dir = "/home/zengwanxin/openworld/vvdata/aug1/"
    val_data_dir = "./output/val"
    batch_size = 16
    num_epochs = 200
    learning_rate = 0.0001

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")



    transform_train = transforms.Compose(
        [
            ResizeWithPad(512),
            transforms.ToTensor(),
            # transforms.CenterCrop((int(0.8 * 384), int(0.8 * 384))),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transform_val = transforms.Compose(
        [
            ResizeWithPad(512),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 加载数据
    train_image_paths, train_labels, train_class_names = load_data(train_data_dir)
    val_image_paths, val_labels, val_class_names = load_data(val_data_dir)
    print(f"训练样本数: {len(train_image_paths)}, 验证样本数: {len(val_image_paths)}")
    print(f"类别: {train_class_names}")

    # 划分训练集和验证集
    # train_paths, val_paths, train_labels, val_labels = train_test_split(
    #     image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    # )

    train_dataset = ImageDataset(train_image_paths, train_labels, transform_train)
    val_dataset = ImageDataset(val_image_paths, val_labels, transform_val)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler, num_workers=4, pin_memory=True
    )
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # net = resnet34(use_cbam=False)
    # model_weight_path = "./resnet34-pre.pth"
    # state = torch.load(model_weight_path, map_location="cpu")
    # net.load_state_dict(state, strict=False)
    # net.load_state_dict(
    #     torch.load(model_weight_path, map_location="cpu", weights_only=True)
    # )
    # in_channel = net.fc.in_features

    # dropout_prob = (
    #     0.3  # 这是一个常用的强度，如果依然过拟合可以设为 0.5，如果欠拟合设为 0.3
    # )
    # net.fc = nn.Sequential(
    #     nn.Dropout(p=dropout_prob), nn.Linear(in_channel, len(train_class_names))
    # )

    # net.fc = ClassifierWithSE(
    #     in_channel=in_channel,
    #     num_classes=len(train_class_names),
    #     dropout_prob=dropout_prob,
    # )
    net = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    model = VaricoseNet(net, num_classes=6)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    print(model)

    # loss_function = FocalLoss(
    #     alpha=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gamma=2  # 按你CEAP各级样本数反比设
    # )
    # weights = torch.tensor([1.5, 1.0, 2.5, 1.2, 3.0,1.0]).cuda()
    # loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)


    loss_function = nn.BCEWithLogitsLoss()
    params = [p for p in model.parameters() if p.requires_grad]

    base_optimizer = torch.optim.AdamW
    # optimizer = SAM(params, base_optimizer, rho=0.05, lr=1e-4, weight_decay=5e-4)

    optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-4)


    # 训练模型
    print("开始...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model,
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


if __name__ == "__main__":
    main()
    # eval()
