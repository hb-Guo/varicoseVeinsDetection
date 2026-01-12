"""
静脉曲张腿部图像抠图工具
使用多种方法进行腿部区域分割和背景去除
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class LegSegmentation:
    """腿部图像分割类"""

    def __init__(self):
        self.methods = {
            "grabcut": self.grabcut_segmentation,
            "skin_color": self.skin_color_segmentation,
            "edge_based": self.edge_based_segmentation,
            "combined": self.combined_segmentation,
        }

    def load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def skin_color_segmentation(self, image: np.ndarray) -> np.ndarray:
        """基于肤色的分割方法"""
        # 转换到YCrCb色彩空间（对肤色检测更有效）
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        # 定义肤色范围（可根据实际情况调整）
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        # 创建肤色掩码
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # 高斯模糊平滑边缘
        skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0)

        return skin_mask

    def grabcut_segmentation(
        self, image: np.ndarray, rect: Optional[Tuple] = None
    ) -> np.ndarray:
        """使用GrabCut算法进行分割"""
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 如果没有提供矩形，自动估计前景区域
        if rect is None:
            h, w = image.shape[:2]
            # 假设腿部在图像中心区域
            rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))

        # 应用GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # 创建最终掩码（0和2为背景，1和3为前景）
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        # 形态学操作优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

        return mask2 * 255

    def edge_based_segmentation(self, image: np.ndarray) -> np.ndarray:
        """基于边缘检测的分割"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        # 膨胀操作连接边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # 找到轮廓
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 创建掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 填充最大的轮廓
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        return mask

    def combined_segmentation(self, image: np.ndarray) -> np.ndarray:
        """组合多种方法的分割"""
        # 获取各种方法的掩码
        skin_mask = self.skin_color_segmentation(image)
        edge_mask = self.edge_based_segmentation(image)

        # 归一化
        skin_mask = skin_mask / 255.0
        edge_mask = edge_mask / 255.0

        # 加权组合
        combined = skin_mask * 0.7 + edge_mask * 0.3

        # 二值化
        _, combined_mask = cv2.threshold(
            (combined * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY
        )

        # 形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        return combined_mask

    def apply_mask(
        self, image: np.ndarray, mask: np.ndarray, background: str = "transparent"
    ) -> np.ndarray:
        """应用掩码到图像"""
        # 确保掩码是二值的
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if background == "transparent":
            # 创建带alpha通道的图像
            b, g, r = cv2.split(image)
            rgba = cv2.merge([b, g, r, mask])
            return rgba
        elif background == "white":
            # 白色背景
            result = image.copy()
            result[mask == 0] = [255, 255, 255]
            return result
        elif background == "black":
            # 黑色背景
            result = image.copy()
            result[mask == 0] = [0, 0, 0]
            return result
        else:
            # 保持原始背景模糊
            blurred_bg = cv2.GaussianBlur(image, (21, 21), 0)
            mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
            result = (image * mask_3ch + blurred_bg * (1 - mask_3ch)).astype(np.uint8)
            return result

    def process_image(
        self,
        image_path: str,
        method: str = "combined",
        background: str = "transparent",
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """处理单张图像"""
        # 加载图像
        image = self.load_image(image_path)

        # 选择分割方法
        if method not in self.methods:
            raise ValueError(
                f"未知方法: {method}. 可用方法: {list(self.methods.keys())}"
            )

        # 执行分割
        mask = self.methods[method](image)

        # 应用掩码
        result = self.apply_mask(image, mask, background)

        # 保存结果
        if output_path:
            if background == "transparent":
                # 保存为PNG以支持透明度
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(output_path, result_bgr)
            else:
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, result_bgr)
            print(f"结果已保存到: {output_path}")

        return result, mask

    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        method: str = "combined",
        background: str = "transparent",
    ):
        """批量处理图像"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 支持的图像格式
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        print(f"找到 {len(image_files)} 张图像")

        # 处理每张图像
        for i, img_file in enumerate(image_files, 1):
            print(f"处理 {i}/{len(image_files)}: {img_file.name}")
            try:
                # 设置输出文件名
                if background == "transparent":
                    output_file = output_path / f"{img_file.stem}_segmented.png"
                else:
                    output_file = (
                        output_path / f"{img_file.stem}_segmented{img_file.suffix}"
                    )

                # 处理图像
                self.process_image(str(img_file), method, background, str(output_file))
            except Exception as e:
                print(f"处理 {img_file.name} 时出错: {str(e)}")

        print(f"\n批量处理完成！结果保存在: {output_dir}")

    def visualize_results(self, image_path: str, method: str = "combined"):
        """可视化分割结果"""
        image = self.load_image(image_path)

        # 尝试所有方法
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"腿部分割结果对比 - {Path(image_path).name}", fontsize=16)

        # 显示原图
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("原始图像")
        axes[0, 0].axis("off")

        # 显示各种方法的结果
        methods_to_show = ["skin_color", "grabcut", "edge_based", "combined"]
        for idx, method_name in enumerate(methods_to_show, 1):
            row = idx // 3
            col = idx % 3

            try:
                mask = self.methods[method_name](image)
                result = self.apply_mask(image, mask, "white")

                axes[row, col].imshow(result)
                axes[row, col].set_title(f"{method_name}")
                axes[row, col].axis("off")
            except Exception as e:
                axes[row, col].text(
                    0.5, 0.5, f"Error: {str(e)}", ha="center", va="center"
                )
                axes[row, col].axis("off")

        # 隐藏多余的子图
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建分割器实例
    segmenter = LegSegmentation()

    result, mask = segmenter.process_image(
        "./avi_data/train/C3/3_8.jpg",
        method="combined",  # 推荐使用combined方法
        background="transparent",  # 透明背景
        output_path="output.png",
    )

    # 示例1: 处理单张图像
    # print("=" * 60)
    # print("示例1: 处理单张图像")
    # print("=" * 60)

    # # 修改为您的图像路径
    # input_image = "leg_image.jpg"
    # output_image = "leg_segmented.png"

    # # 可选方法: 'skin_color', 'grabcut', 'edge_based', 'combined'
    # # 可选背景: 'transparent', 'white', 'black', 'blur'

    # # result, mask = segmenter.process_image(
    # #     input_image,
    # #     method='combined',
    # #     background='transparent',
    # #     output_path=output_image
    # # )

    # # 示例2: 批量处理
    # print("\n" + "=" * 60)
    # print("示例2: 批量处理图像")
    # print("=" * 60)

    # input_dir = "input_images"    # 输入文件夹
    # output_dir = "output_images"  # 输出文件夹

    # # segmenter.batch_process(
    # #     input_dir,
    # #     output_dir,
    # #     method='combined',
    # #     background='transparent'
    # # )

    # # 示例3: 可视化对比不同方法
    # print("\n" + "=" * 60)
    # print("示例3: 可视化分割结果")
    # print("=" * 60)

    # # segmenter.visualize_results(input_image)

    # print("\n使用说明:")
    # print("1. 取消注释上述示例代码")
    # print("2. 修改图像路径为实际路径")
    # print("3. 运行脚本进行处理")
    # print("\n可选参数:")
    # print("- method: 'skin_color', 'grabcut', 'edge_based', 'combined'")
    # print("- background: 'transparent', 'white', 'black', 'blur'")
