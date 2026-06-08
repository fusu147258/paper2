import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter


def apply_fiba_low_freq_perturbation(image, intensity=0.08, low_pass_sigma=3.0, noise_seed=42):
   
    has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False
    img_bgr = image[:, :, :3] if has_alpha else image

    img_float = img_bgr.astype(np.float32) / 255.0  # 归一化到 0-1 范围
    h, w, c = img_float.shape

    perturbed_channels = []

    for i in range(c):
        channel = img_float[:, :, i]

       
        np.random.seed(noise_seed + i)  # 每个通道用不同种子，确保噪声多样性
        noise = np.random.randn(h, w).astype(np.float32)


        low_freq_noise = gaussian_filter(noise, sigma=low_pass_sigma, mode='reflect')

        perturbed_channel = channel + low_freq_noise * intensity

        perturbed_channels.append(perturbed_channel)

    result_float = cv2.merge(perturbed_channels)
    result_bgr = np.clip(result_float * 255.0, 0, 255).astype(np.uint8)

    if has_alpha:
        return cv2.merge([result_bgr, image[:, :, 3]])
    return result_bgr


def batch_process(input_path, output_path):
    abs_input = os.path.normpath(input_path)
    abs_output = os.path.normpath(output_path)

    if not os.path.exists(abs_input):
        print(f"错误：找不到输入文件夹 -> {abs_input}")
        return

    if not os.path.exists(abs_output):
        os.makedirs(abs_output)

    files = [f for f in os.listdir(abs_input) if f.lower().endswith('.png')]

    if not files:
        print(f"提示：在 {abs_input} 文件夹中没有找到 PNG 图片。")
        return

    for filename in files:
        file_path = os.path.join(abs_input, filename)

        # 兼容中文路径的读取
        img_data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"无法解码图片: {filename}")
            continue

  
        processed = apply_fiba_low_freq_perturbation(img, intensity=0.4, low_pass_sigma=4.0)

    
        save_path = os.path.join(abs_output, filename)
        ext = os.path.splitext(filename)[1]
        result, nparray = cv2.imencode(ext, processed)
        if result:
            nparray.tofile(save_path)
            print(f"FIBA 处理成功: {filename}")


if __name__ == "__main__":
    # 请确保这些路径指向你实际的文件夹
    input_dir = r'D:\代码\grad-cam-pytorch-master\clean'
    output_dir = r'D:\代码\grad-cam-pytorch-master\clean_fiba_attack'

    batch_process(input_dir, output_dir)
    print(f"\n--- FIBA 完成 ---")
