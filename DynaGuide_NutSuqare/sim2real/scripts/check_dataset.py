import h5py
import cv2
import numpy as np
import os

def inspect_dataset(hdf5_path, save_dir="dataset_inspect"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在读取数据集: {hdf5_path}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 获取第一个 demo
            demos = list(f["data"].keys())
            # 过滤出 "demo_" 开头的键
            demo_keys = [k for k in demos if k.startswith("demo_")]
            if not demo_keys:
                print("未找到 demo 数据！")
                return
            
            first_demo = demo_keys[0]
            if "demo_0" in demo_keys:
                first_demo = "demo_0"
                
            print(f"\n========== 提取 {first_demo} 的第一帧 ==========")
            
            # 1. 提取并打印四元数
            quat_path = f"data/{first_demo}/obs/robot0_eef_quat"
            if quat_path in f:
                eef_quat = f[quat_path][0]
                print(f"[四元数] {quat_path} 形状: {eef_quat.shape}")
                print(f"[四元数] 具体数值: {eef_quat}")
            else:
                print(f"[警告] 数据集中不存在 {quat_path}")

            # 2. 查找并保存所有图像观测
            obs_group = f[f"data/{first_demo}/obs"]
            obs_keys = list(obs_group.keys())
            
            image_keys = [k for k in obs_keys if "image" in k]
            if not image_keys:
                print("[警告] 在 obs 中未找到包含 'image' 的键名。当前的 keys:", obs_keys)
            
            for img_key in image_keys:
                img_data = obs_group[img_key][0]
                print(f"[图像] {img_key} 形状: {img_data.shape}, 数据类型: {img_data.dtype}")
                
                # 假设图像格式为 (H, W, 3) 且为 RGB
                if len(img_data.shape) == 3 and img_data.shape[-1] == 3:
                    # Robomimic 数据集通常存储为 uint8 RGB，OpenCV 需要 BGR
                    if img_data.dtype == np.uint8:
                        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    else:
                        # 如果是 float 类型 (0.0~1.0)，先转换
                        img_bgr = cv2.cvtColor((img_data * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    save_path = os.path.join(save_dir, f"{first_demo}_{img_key}.png")
                    cv2.imwrite(save_path, img_bgr)
                    print(f"      -> 成功保存图片至: {save_path}")
                else:
                    print(f"      -> 无法保存 {img_key}，形状不符合常规图像格式")
                    
    except Exception as e:
        print(f"读取数据集时发生错误: {e}")

if __name__ == "__main__":
    # 请将这里的路径替换为你用于训练的真实 .hdf5 数据集路径
    dataset_path = "/home/smlr_g2/smlr_g2_WS25/square_d0_dr0_two_views.hdf5"
    inspect_dataset(dataset_path)