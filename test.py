import torch
import torch.nn as nn
from model import resnet34, resnet50,resnet18
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as F



class ResizeWithPad:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, img):
        import torchvision.transforms.functional as F
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

class_names = ["C1", "C2", "C3", "C4", "C5", "C6"]


def load_test_data(test_dir):
    image_paths = []
    image_names = []
    labels_list = []

    for img_name in os.listdir(test_dir):
        # print(test_dir.split("/")[-1])
        if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_paths.append(os.path.join(test_dir, img_name))
            image_names.append(img_name)
            basename = os.path.basename(img_name)
            label = str(test_dir.split("/")[-1])
            # print("标签是：", label)
            labels_list.append(label)

    return image_paths, image_names, labels_list

import torch.nn.functional as F

def label_to_ordinal(labels, num_classes=6):
    # labels: [batch_size]
    # return: [batch_size, num_classes - 1]
    batch_size = labels.size(0)
    ordinal_labels = torch.zeros(batch_size, num_classes - 1).to(labels.device)
    for i in range(batch_size):
        ordinal_labels[i, :labels[i]] = 1
    return ordinal_labels

# 3. 核心修改：针对序数回归的推理函数
def test_model(model, test_image_paths, class_names, transform, device):
    model.eval()
    results = []

    with torch.no_grad():
        for img_path in test_image_paths:
            img_raw = Image.open(img_path).convert("RGB")
            img_tensor = transform(img_raw).unsqueeze(0).to(device)

            logits = model(img_tensor)
            probs = torch.sigmoid(logits) # 获取 5 个神经元的概率

            # --- 序数回归逻辑 ---
            # 统计大于 0.5 的神经元个数。
            # 0个 > 0.5 => C1; 1个 > 0.5 => C2 ... 5个 > 0.5 => C6
            pred_idx = (probs > 0.5).sum(dim=1).item()
            pred_class = class_names[pred_idx]

            # 置信度计算：在序数回归中，通常取最接近阈值的那个概率的明确度
            # 简单处理：取 5 个神经元概率的平均值作为参考，或者显示具体每个开关的概率
            confidence = probs.mean().item()

            ground_truth = os.path.basename(os.path.dirname(img_path)) # 获取父目录名作为真值
            flag = (str(ground_truth) == str(pred_class))

            results.append({
                "image": os.path.basename(img_path),
                "pred_class": pred_class,
                "confidence": confidence,
                "probs": probs.cpu().numpy().tolist(), # 记录原始概率列表
                "flag": flag,
            })
    return results

# def test_model(model, test_image_paths, class_names, transform, device, labels):
#     """
#     自动化测试函数（推理）
#     """
#     model.eval()
#
#     results = []
#
#     with torch.no_grad():
#         for img_path in test_image_paths:
#             # print("正在测试图片：",img_path.split("/")[-2])
#             img = Image.open(img_path).convert("RGB")
#             # img1 = img.convert("RGB")
#             img = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]
#             # print("正在测试图片：", os.path.basename(img_path))
#
#             # outputs = model(img)
#
#             # ground_truth = img_path.split("/")[-2]
#             # ordinal_targets = label_to_ordinal(int(ground_truth[-1])-1)
#
#             outputs = model(img)
#             # loss = criterion(outputs, ordinal_targets)
#
#             probs = torch.sigmoid(outputs) # 转换为概率
#             # # 统计大于 0.5 的个数。例如有 2 个大于 0.5，说明是 C3 (索引为2)
#             # predicted = (probs > 0.5).sum(dim=1)
#             #
#             # probs = torch.softmax(outputs, dim=1)
#             # # print("{}判断概率分布{}".format(img_path, probs.cpu().numpy()))
#             conf, pred = torch.max(probs, dim=1)
#
#             pred_class = class_names[pred.item()]
#             # print("Predicted Class: {}".format(pred.item()+1))
#             ground_truth = img_path.split("/")[-2]
#             # print("Ground Truth: {}, Predicted: {}".format(ground_truth, pred.item()+1))
#             confidence = conf.item()
#             flag = True if str(ground_truth) == str(class_names[pred.item()]) else False
#             # if confidence<0.6:
#             #     flag = "uncertain"
#             results.append(
#                 {
#                     "image": os.path.basename(img_path),
#                     "pred_class": pred_class,
#                     "confidence": confidence,
#                     "flag": flag,
#                 }
#             )
#
#     return results

# class VaricoseNet(nn.Module):
#     def __init__(self, backbone, num_classes=6):
#         super().__init__()
#         self.backbone = backbone            # DINO ViT
#         self.classifier = nn.Sequential(nn.Dropout(p=0.99), nn.Linear(768, num_classes))

#     def forward(self, x):
#         feat = self.backbone(x)             # [B, 768]
#         out = self.classifier(feat)         # [B, num_classes]
#         return out

# class VaricoseNet(nn.Module):
#     def __init__(self, backbone, num_classes=6):
#         super().__init__()
#         self.backbone = backbone            # DINO ViT
#         # self.resnet = resnet34()
#         # state = torch.load("./resnet34-pre.pth", map_location="cpu", weights_only=True)
#         # self.resnet.load_state_dict(state, strict=False)
#         self.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(768, 6))# 0.5效果最好

#     def forward(self, x):
#         feat = self.backbone(x)             # [B, 768]
#         # feat_res = self.resnet(x)          # [B, 512]
#         # print(feat.shape,"              ",feat_res.shape)
#         # import time
#         # time.sleep(5)
#         # feat = torch.concatenate([feat, feat_res], dim=1)
        
#         out = self.classifier(feat)         # [B, num_classes]
#         return out

# class VaricoseNet(nn.Module):
#     def __init__(self, backbone, num_classes=6):
#         super().__init__()
#         self.backbone = backbone            # DINO ViT
#         print(backbone)
#         # self.resnet = resnet34()
#         # state = torch.load("./resnet34-pre.pth", map_location="cpu", weights_only=True)
#         # self.resnet.load_state_dict(state, strict=False)
#         # self.classifier = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(768, 6))# 0.5效果最好
#         for param in self.backbone.parameters():
#             param.requires_grad = False
        
#         for param in self.backbone.blocks[-1].parameters():
#             param.requires_grad = True

        
#         if hasattr(self.backbone, 'norm'):
#             for param in self.backbone.norm.parameters():
#                 param.requires_grad = True

#         embed_dim = 768
        
#         self.norm = nn.LayerNorm(embed_dim * 2) 
        
#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim * 2, 256),      # 先降维，减少参数
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(p=0.6),                  # 0.5效果好
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         # feat = self.backbone(x)             # [B, 768]
#         # feat_res = self.resnet(x)          # [B, 512]
#         # # print(feat.shape,"              ",feat_res.shape)
#         # # import time
#         # # time.sleep(5)
#         # feat = torch.concatenate([feat, feat_res], dim=1)
#         # out = self.classifier(feat)         # [B, num_classes]

#         # 1. 提取最后一层所有 token
#         # 注意：DINOv2 官方模型 forward 会返回 [B, 768] (即 CLS)
#         # 如果要获取所有 patch token，通常需要调用 model.get_intermediate_layers
        
#         # 假设你使用的是标准的 dinov2 结构，通过特定接口获取 patch tokens
#         # 这里以获取最后一层特征图为例: [B, N+1, 768]
#         features = self.backbone.get_intermediate_layers(x, n=1)[0] 
        
#         cls_token = features[:, 0]              # [B, 768]
#         patch_tokens = features[:, 1:]          # [B, N, 768]
        
#         # 2. 全局平均池化 (GAP)
#         gap = torch.mean(patch_tokens, dim=1)   # [B, 768]
        
#         # 3. 拼接 CLS 和 GAP
#         combined = torch.cat([cls_token, gap], dim=1) # [B, 1536]
        
#         # 4. 分类
#         combined = self.norm(combined)
#         out = self.classifier(combined)
#         return out

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

# class VaricoseNet(nn.Module):
#     def __init__(self, backbone, num_classes=6):
#         super().__init__()
#         self.backbone = backbone            # DINO ViT
#         print(backbone)
#         # self.resnet = resnet34()
#         # state = torch.load("./resnet34-pre.pth", map_location="cpu", weights_only=True)
#         # self.resnet.load_state_dict(state, strict=False)
#         # self.classifier = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(768, 6))# 0.5效果最好
#         for param in self.backbone.parameters():
#             param.requires_grad = False
        
#         for param in self.backbone.blocks[-1].parameters():
#             param.requires_grad = True
        
#         for param in self.backbone.blocks[-2].parameters():
#             param.requires_grad = True

    
#         if hasattr(self.backbone, 'norm'):
#             for param in self.backbone.norm.parameters():
#                 param.requires_grad = True

#         embed_dim = 768
        
#         self.norm = nn.LayerNorm(embed_dim * 2) 
        
#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim * 2, 256),      # 先降维，减少参数
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(p=0.7),                  # 0.5效果好
#             nn.Linear(256, num_classes)
#             # CosineLinear(256, num_classes, s=15) 
#         )

#     def forward(self, x):
#         # feat = self.backbone(x)             # [B, 768]
#         # feat_res = self.resnet(x)          # [B, 512]
#         # # print(feat.shape,"              ",feat_res.shape)
#         # # import time
#         # # time.sleep(5)
#         # feat = torch.concatenate([feat, feat_res], dim=1)
#         # out = self.classifier(feat)         # [B, num_classes]

#         # 1. 提取最后一层所有 token
#         # 注意：DINOv2 官方模型 forward 会返回 [B, 768] (即 CLS)
#         # 如果要获取所有 patch token，通常需要调用 model.get_intermediate_layers
        
#         # 假设你使用的是标准的 dinov2 结构，通过特定接口获取 patch tokens
#         # 这里以获取最后一层特征图为例: [B, N+1, 768]
#         features = self.backbone.get_intermediate_layers(x, n=1)[0] 
        
#         cls_token = features[:, 0]              # [B, 768]
#         patch_tokens = features[:, 1:]          # [B, N, 768]
        
#         # 2. 全局平均池化 (GAP)
#         gap = torch.mean(patch_tokens, dim=1)   # [B, 768]
        
#         # 3. 拼接 CLS 和 GAP
#         combined = torch.cat([cls_token, gap], dim=1) # [B, 1536]
        
#         # 4. 分类
#         combined = self.norm(combined)
#         out = self.classifier(combined)
#         return out

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

net = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
model = VaricoseNet(net, num_classes=5)
# model = nn.Sequential(
#         net,
#         nn.Linear(768, 6)
#     )

# state = torch.load("new_resnet50_ddp.pth", map_location="cpu")
# net.load_state_dict(state, strict=True)
# model.eval()
# net = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
# net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# model = VaricoseNet(net, num_classes=6)

state = torch.load("aug1_new_resnet50_ddp.pth", map_location="cpu")
model.load_state_dict(state, strict=True)

model = model.to(device)
model.eval()
# net = resnet34()
# backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")

# state = torch.load("./new_resnet50_ddp.pth", map_location="cpu")
# backbone.load_state_dict(state["backbone"])
# # net = torch.load("facebookresearch/dino:main", "dino_vitb16")
# # net = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
# net = backbone
# # net.load_state_dict(
# #     torch.load(model_weight_path, map_location="cpu", weights_only=True)
# # )
# model = VaricoseNet(net, num_classes=6)
# model_weight_path = "./new_resnet50_ddp.pth"
# in_channel = net.fc.in_features

# dropout_prob = 0.4  # 这是一个常用的强度，如果依然过拟合可以设为 0.5，如果欠拟合设为 0.3
# net.fc = nn.Sequential(
#     nn.Dropout(p=0.4),nn.Linear(in_channel, 6)
# )
# # net.fc = ClassifierWithSE(
# #         in_channel=in_channel,
# #         num_classes=6,
# #         dropout_prob=dropout_prob,
# #     )
# state = torch.load(model_weight_path, map_location="cpu", weights_only=True)
# net.load_state_dict(state, strict=False)
# # net.load_state_dict(
# #     torch.load(model_weight_path, map_location="cpu", weights_only=True)
# # )

# net = net.to(device)


transform_val = transforms.Compose(
    [
        ResizeWithPad(512),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
data_dir = "./data"


total = 0
correct_total = 0
num_uncertain =0
for class_name in class_names:
    test_dir = "./output/val/" + class_name  # 你自己的测试集路径
    test_image_paths, _, labels_list = load_test_data(test_dir)

    test_results = test_model(
        model=model,
        test_image_paths=test_image_paths,
        class_names=class_names,
        transform=transform_val,
        device=device,
        # labels=labels_list,
    )
    correct_count = sum(1 for res in test_results if res["flag"]==True)
    len_count = len(test_results)
    num_uncertain += sum(1 for res in test_results if res["flag"]=="uncertain")



    for res in test_results:
        if res["flag"] == "uncertain":
            len_count -= 1
        print(
            f"Image: {res['image']:<20} "
            f"→ Pred: {res['pred_class']} "
            f"(conf={res['confidence']:.4f})"
            f" → {res['flag']}"
        )

    print(
        f"{class_name}类准确率: {correct_count} / {len(test_results)} = {correct_count / len(test_results):.4f}"
    )
    correct_total += correct_count
    total += len(test_results)

print(f"\n总体准确率: {correct_total}/{total} = {correct_total / total:.4f}")
print(f"处于分类边界的图片数: {num_uncertain}")
