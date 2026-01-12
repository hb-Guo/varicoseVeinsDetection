"""
通过多种数据增强技术扩充数据集
将每张图片生成N个变体，大幅增加数据量
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from tqdm import tqdm
import random

class DataAugmentor:
    """
    数据增强器 - 生成多样化的图像变体
    """
    
    def __init__(self, augmentations_per_image=20):
        self.augmentations_per_image = augmentations_per_image
    
    def random_rotation(self, image, max_angle=30):
        """随机旋转"""
        angle = random.uniform(-max_angle, max_angle)
        return image.rotate(angle, fillcolor=(255, 255, 255))
    
    def random_flip(self, image):
        """随机翻转"""
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.7:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image
    
    def random_brightness(self, image):
        """随机亮度调整"""
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def random_contrast(self, image):
        """随机对比度调整"""
        factor = random.uniform(1.3, 2)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def random_saturation(self, image):
        """随机饱和度调整"""
        factor = random.uniform(0.5, 2)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def random_sharpness(self, image):
        """随机锐度调整"""
        factor = random.uniform(0.5, 2.0)
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def random_blur(self, image):
        """随机模糊"""
        if random.random() > 0.5:
            radius = random.uniform(0, 2)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image
    
    def random_crop_and_resize(self, image, crop_ratio=0.8):
        """随机裁剪并调整大小"""
        w, h = image.size
        crop_w = int(w * random.uniform(crop_ratio, 1.0))
        crop_h = int(h * random.uniform(crop_ratio, 1.0))
        
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        
        image = image.crop((left, top, left + crop_w, top + crop_h))
        image = image.resize((w, h), Image.LANCZOS)
        
        return image
    
    def random_perspective(self, image):
        """随机透视变换"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # 定义源点和目标点
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # 随机扰动
        offset = int(w * 0.1)
        dst_points = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)]
        ])
        
        # 透视变换
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(img_array, matrix, (w, h), 
                                     borderValue=(255, 255, 255))
        
        return Image.fromarray(result)
    
    def random_noise(self, image, noise_level=0.02):
        """添加随机噪声"""
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # 高斯噪声
        noise = np.random.randn(*img_array.shape) * noise_level
        noisy = np.clip(img_array + noise, 0, 1)
        
        return Image.fromarray((noisy * 255).astype(np.uint8))
    
    def random_elastic_transform(self, image, alpha=30, sigma=5):
        """弹性变形"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # 生成随机位移场
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        
        # 创建网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y = np.clip(y + dy, 0, h - 1).astype(np.float32)
        
        # 应用变形
        result = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, 
                          borderValue=(255, 255, 255))
        
        return Image.fromarray(result)
    
    def augment_image(self, image):
        """
        对单张图像应用随机增强组合
        """
        # 随机选择3-5种增强方法
        augmentation_methods = [
            self.random_rotation,
            self.random_flip,
            # self.random_brightness,
            # self.random_contrast,
            # self.random_saturation,
            self.random_sharpness,
            # self.random_blur,
            self.random_crop_and_resize,
            self.random_perspective,
            self.random_noise,
            # self.random_elastic_transform,
        ]
        
        # 随机选择增强方法
        num_augmentations = random.randint(3, 6)
        # num_augmentations = 10
        selected_methods = random.sample(augmentation_methods, num_augmentations)
        
        # 应用增强
        augmented = image.copy()
        for method in selected_methods:
            try:
                augmented = method(augmented)
            except Exception as e:
                print(f"增强失败: {method.__name__}, {e}")
                continue
        
        return augmented
    
    def augment_dataset(self, input_dir, output_dir, keep_original=True):
        """
        批量增强整个数据集
        
        Args:
            input_dir: 原始数据集路径
            output_dir: 输出路径
            keep_original: 是否保留原始图像
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        class_names = sorted(os.listdir(input_dir))
        
        total_original = 0
        total_generated = 0
        
        for class_name in class_names:
            class_input_path = os.path.join(input_dir, class_name)
            class_output_path = os.path.join(output_dir, class_name)
            
            if not os.path.isdir(class_input_path):
                continue
            
            os.makedirs(class_output_path, exist_ok=True)
            
            # 获取所有图片
            image_files = [f for f in os.listdir(class_input_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            print(f"\n处理类别: {class_name} ({len(image_files)} 张原始图片)")
            
            for img_name in tqdm(image_files):
                input_path = os.path.join(class_input_path, img_name)
                base_name = os.path.splitext(img_name)[0]
                ext = os.path.splitext(img_name)[1]
                
                try:
                    # 读取原始图像
                    image = Image.open(input_path)
                    
                    # 保存原始图像
                    if keep_original:
                        output_path = os.path.join(class_output_path, 
                                                  f"{base_name}_original{ext}")
                        image1= image.convert('RGB')
                        image1.save(output_path)
                        total_original += 1
                    
                    # 生成增强图像
                    for i in range(self.augmentations_per_image):
                        augmented = self.augment_image(image).convert('RGB')
                        output_path = os.path.join(class_output_path, 
                                                  f"{base_name}_aug_{i+1}{ext}")
                        augmented.save(output_path)
                        total_generated += 1
                    
                except Exception as e:
                    print(f"\n处理失败 {img_name}: {e}")
                    continue
            
            class_total = len(os.listdir(class_output_path))
            print(f"类别 {class_name} 完成: {class_total} 张图片")
        
        print(f"\n数据扩充完成！")
        print(f"原始图片: {total_original}")
        print(f"生成图片: {total_generated}")
        print(f"总计: {total_original + total_generated}")
        print(f"输出目录: {output_dir}")


def main():
    """
    使用示例
    """
    # 配置参数
    input_dir = "./avi_data/train_crop"  # 原始数据集
    output_dir = "./avi_data/train_aug"  # 扩充后的数据集
    augmentations_per_image = 10  # 每张图片生成20个变体
    
    # 创建增强器
    augmentor = DataAugmentor(augmentations_per_image=augmentations_per_image)
    
    # 扩充数据集
    augmentor.augment_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        keep_original=True  # 保留原始图像
    )
    
    print("\n现在可以使用扩充后的数据集训练了！")
    print(f"将训练脚本中的 data_dir 改为: {output_dir}")


if __name__ == '__main__':
    main()