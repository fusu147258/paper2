import cv2
import numpy as np
import os


def generate_residual_map(img1, img2, boost=10):
    """
    计算两张图片的残差图
    :param img1: 原始图片
    :param img2: 处理后的图片
    :param boost: 增强系数。微小扰动肉眼难辨，乘以这个系数可以让残差更明显
    """
    # 确保尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 计算绝对差值 |img1 - img2|
    # 使用 absdiff 避免负数溢出
    diff = cv2.absdiff(img1, img2)

    # 增强残差的可视化效果
    # 如果差值很小（如 1 或 2），增强后会更容易观察
    diff_boosted = np.clip(diff.astype(np.float32) * boost, 0, 255).astype(np.uint8)

    return diff_boosted


def batch_residual_analysis(folder_clean, folder_processed, folder_output, boost_factor=10):
    # 规范化路径
    path1 = os.path.normpath(folder_clean)
    path2 = os.path.normpath(folder_processed)
    out_path = os.path.normpath(folder_output)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 以原始文件夹的文件列表为准
    files = [f for f in os.listdir(path1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in files:
        file1 = os.path.join(path1, filename)
        file2 = os.path.join(path2, filename)

        if not os.path.exists(file2):
            print(f"跳过：在处理后的文件夹中找不到对应文件 {filename}")
            continue

        # 兼容中文路径读取
        img1 = cv2.imdecode(np.fromfile(file1, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imdecode(np.fromfile(file2, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        if img1 is None or img2 is None:
            continue

        # 如果有 Alpha 通道，只取前三通道进行比较
        if img1.shape[-1] == 4: img1 = img1[:, :, :3]
        if img2.shape[-1] == 4: img2 = img2[:, :, :3]

        # 生成残差图
        residual = generate_residual_map(img1, img2, boost=boost_factor)

        # 保存结果
        save_path = os.path.join(out_path, f"res_{filename}")
        is_success, buffer = cv2.imencode('.png', residual)
        if is_success:
            buffer.tofile(save_path)
            print(f"残差图已生成: res_{filename}")


if __name__ == "__main__":
    # --- 路径设置 ---
    clean_dir = r'D:\代码\grad-cam-pytorch-master\clean'
    processed_dir = r'D:\代码\grad-cam-pytorch-master\overlay_results'
    output_dir = r'D:\代码\grad-cam-pytorch-master\cancha\blend'

    # --- 增强系数 ---
    # 如果你的扰动非常淡（alpha很小），把这个值调大，比如 20 或 50
    boost = 10

    batch_residual_analysis(clean_dir, processed_dir, output_dir, boost)
    print(f"\n分析完成！残差图保存在: {output_dir}")