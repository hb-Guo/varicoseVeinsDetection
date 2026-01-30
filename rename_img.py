import os
import shutil


def rename_images(source_dir, target_dir):
    """
    将图像重命名并组织到新的目录结构中

    参数:
        source_dir: 原始数据集路径，应包含多个类别文件夹
        target_dir: 目标路径，将创建重命名后的数据集
    """
    # 创建目标目录
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取所有类别文件夹
    class_folders = [f for f in os.listdir(source_dir)
                     if os.path.isdir(os.path.join(source_dir, f))]

    print(f"找到 {len(class_folders)} 个类别")

    # 遍历每个类别文件夹
    for class_idx, class_name in enumerate(sorted(class_folders), 1):
        source_class_path = os.path.join(source_dir, class_name)
        target_class_path = os.path.join(target_dir, class_name)

        # 创建目标类别文件夹
        if not os.path.exists(target_class_path):
            os.makedirs(target_class_path)

        # 获取该类别下的所有图片
        images = [f for f in os.listdir(source_class_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        print(f"类别 {class_idx} ({class_name}): {len(images)} 张图片")

        # 重命名并复制图片
        for img_idx, img_name in enumerate(sorted(images), 1):
            source_img_path = os.path.join(source_class_path, img_name)

            # 获取文件扩展名
            _, ext = os.path.splitext(img_name)

            # 新文件名格式: {类别编号}_{图片编号}.ext
            new_img_name = f"{class_idx}_{img_idx}{ext}"
            target_img_path = os.path.join(target_class_path, new_img_name)

            # 复制文件
            shutil.copy2(source_img_path, target_img_path)

        print(f"  已完成重命名")

    print(f"\n数据预处理完成！新数据集保存在: {target_dir}")


if __name__ == "__main__":
    # 设置路径
    source_directory = "./output/test"  # 修改为你的原始数据集路径
    target_directory = "./avi_data/test_name"  # 处理后的数据集路径
    rename_images(source_directory, target_directory)