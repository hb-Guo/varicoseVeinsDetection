# # import cv2
# # import os
# # import numpy as np

# # def crop_leg_region(img):
# #     h, w = img.shape[:2]

# #     # 1. 转灰度 + 高斯模糊
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     blur = cv2.GaussianBlur(gray, (7,7), 0)

# #     # 2. 自适应阈值（比肤色鲁棒）
# #     thresh = cv2.adaptiveThreshold(
# #         blur, 255,
# #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #         cv2.THRESH_BINARY_INV,
# #         31, 5
# #     )

# #     # 3. 形态学处理
# #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 25))
# #     mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# #     # 4. 找轮廓
# #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if len(contours) == 0:
# #         return img

# #     # 5. 选“最像腿”的轮廓（高度最大 & 纵横比大）
# #     best = None
# #     best_score = 0
# #     for cnt in contours:
# #         x, y, w_box, h_box = cv2.boundingRect(cnt)
# #         area = cv2.contourArea(cnt)
# #         ratio = h_box / (w_box + 1e-5)

# #         # 腿应该是又高又大的
# #         score = area * ratio
# #         if score > best_score:
# #             best_score = score
# #             best = (x, y, w_box, h_box)

# #     x, y, w_box, h_box = best

# #     # 稍微放大一点
# #     pad = 20
# #     x = max(0, x - pad)
# #     y = max(0, y - pad)
# #     w_box = min(img.shape[1] - x, w_box + 2*pad)
# #     h_box = min(img.shape[0] - y, h_box + 2*pad)

# #     cropped = img[y:y+h_box, x:x+w_box]
# #     return cropped


# # def batch_crop(input_dir, output_dir):
# #     os.makedirs(output_dir, exist_ok=True)

# #     for name in os.listdir(input_dir):
# #         if not name.lower().endswith(('.jpg','.png','.jpeg')):
# #             continue
# #         img = cv2.imread(os.path.join(input_dir, name))
# #         cropped = crop_leg_region(img)
# #         cv2.imwrite(os.path.join(output_dir, name), cropped)

# #     print("裁剪完成:", output_dir)

# # if __name__ == "__main__":
# #     input_dir = "./avi_data/test/C2"      # 原始图片目录
# #     output_dir = "./avi_data/test_cai"    # 裁剪后保存目录
# #     batch_crop(input_dir, output_dir)


# # import cv2
# # import numpy as np

# # img = cv2.imread("./avi_data/test/C1/1_2.jpg")
# # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # # 放宽肤色范围（更鲁棒）
# # lower = np.array([0, 20, 40], dtype=np.uint8)
# # upper = np.array([35, 255, 255], dtype=np.uint8)

# # mask = cv2.inRange(hsv, lower, upper)

# # cv2.imshow("original", img)
# # cv2.imshow("mask", mask)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# import os
# import cv2
# import numpy as np
# from tqdm import tqdm


# def grabcut_auto(img):
#     """
#     自动前景分割：假设腿在画面中部
#     """
#     h, w = img.shape[:2]
#     mask = np.zeros((h, w), np.uint8)

#     # 假设腿在中间区域
#     rect = (
#         int(w * 0.1),
#         int(h * 0.05),
#         int(w * 0.8),
#         int(h * 0.9)
#     )

#     bgdModel = np.zeros((1, 65), np.float64)
#     fgdModel = np.zeros((1, 65), np.float64)

#     cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

#     # 前景 = 1，背景 = 0
#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
#     return mask2 * 255


# def crop_by_mask(img, mask):
#     """
#     根据分割mask裁剪ROI
#     """
#     ys, xs = np.where(mask == 255)
#     if len(xs) == 0 or len(ys) == 0:
#         # 如果失败，直接返回原图
#         return img

#     y1, y2 = ys.min(), ys.max()
#     x1, x2 = xs.min(), xs.max()
#     return img[y1:y2, x1:x2]

# def apply_mask(img, mask):
#     """
#     img: BGR image
#     mask: 0/255
#     """
#     result = img.copy()
#     result[mask == 0] = (0, 0, 0)   # 背景变黑
#     return result

# def process_folder(input_root, output_root):
#     os.makedirs(output_root, exist_ok=True)

#     classes = sorted(os.listdir(input_root))

#     for cls in classes:
#         input_dir = os.path.join(input_root, cls)
#         output_dir = os.path.join(output_root, cls)
#         os.makedirs(output_dir, exist_ok=True)

#         if not os.path.isdir(input_dir):
#             continue

#         imgs = [f for f in os.listdir(input_dir)
#                 if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

#         print(f"处理类别 {cls}, 共 {len(imgs)} 张")

#         for name in tqdm(imgs):
#             path = os.path.join(input_dir, name)
#             img = cv2.imread(path)

#             if img is None:
#                 print(f"读取失败: {path}")
#                 continue

#             # GrabCut分割
#             mask = grabcut_auto(img)

#             # 裁剪
#             crop = crop_by_mask(img, mask)

#             # 保存
#             save_path = os.path.join(output_dir, name)
#             masked = apply_mask(img, mask)
#             cv2.imwrite(save_path, masked)
#             # mask_path = os.path.join(output_dir, name.replace('.jpg', '_mask.jpg'))
#             # cv2.imwrite(mask_path, mask)


# if __name__ == "__main__":
#     input_root = "./avi_data/train"
#     output_root = "./avi_data/train_crop"

#     process_folder(input_root, output_root)
#     print("全部裁剪完成，输出目录:", output_root)


import os
import cv2
import numpy as np
from tqdm import tqdm


def grabcut_fast(img, small_size=640):
    """
    先缩小，再GrabCut，大幅加速
    """
    h, w = img.shape[:2]
    scale = small_size / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)))

    mask = np.zeros(small.shape[:2], np.uint8)
    rect = (
        int(small.shape[1] * 0.1),
        int(small.shape[0] * 0.05),
        int(small.shape[1] * 0.8),
        int(small.shape[0] * 0.9),
    )

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(small, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8") * 255

    # 放大回原图尺寸
    mask_full = cv2.resize(mask2, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_full

def refine_mask_for_two_legs(mask):
    """
    mask: 0/255
    目标：断开两腿之间细小连接，去掉中间背景
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25))

    # 1. 腐蚀：切断细连接（中间地板/墙壁）
    eroded = cv2.erode(mask, kernel, iterations=1)

    # 2. 膨胀：恢复腿的主体
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # 3. 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    # stats: [label, x, y, w, h, area]
    # 选面积最大的两个区域（两条腿）
    areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景
    if len(areas) < 2:
        return mask  # 不是双腿，直接返回

    idx = np.argsort(areas)[-2:] + 1  # 最大的两个连通域索引

    new_mask = np.zeros_like(mask)
    for i in idx:
        new_mask[labels == i] = 255

    return new_mask


def apply_mask(img, mask):
    """
    img: BGR image
    mask: 0/255
    """
    result = img.copy()
    result[mask == 0] = (255, 255, 255)  # 背景变白
    return result


def crop_by_mask(img, mask):
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return img

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    pad = 20
    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(img.shape[0], y2 + pad)
    x2 = min(img.shape[1], x2 + pad)

    return img[y1:y2, x1:x2]


def process_folder(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    classes = sorted(os.listdir(input_root))

    for cls in classes:
        input_dir = os.path.join(input_root, cls)
        output_dir = os.path.join(output_root, cls)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(input_dir):
            continue

        imgs = [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
        ]

        print(f"处理类别 {cls}, 共 {len(imgs)} 张")

        for name in tqdm(imgs):
            path = os.path.join(input_dir, name)
            img = cv2.imread(path)
            if img is None:
                continue

            # 快速 GrabCut
            mask = grabcut_fast(img)
            mask = refine_mask_for_two_legs(mask)
            # 真正裁剪
            crop = crop_by_mask(img, mask)
            masked = apply_mask(img, mask)
            
            save_path = os.path.join(output_dir, name)
            cv2.imwrite(save_path, masked)
            # cv2.imwrite(save_path, crop)


if __name__ == "__main__":
    input_root = "./avi_data/val"
    output_root = "./avi_data/val_crop"
    process_folder(input_root, output_root)
    print("全部完成:", output_root)
