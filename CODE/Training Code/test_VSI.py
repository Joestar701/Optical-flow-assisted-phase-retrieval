import os
import re
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Vibphase_train import RAFTVibPhase  # ✅ 你的训练模型类

# ===============================
# 基本设置
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 模型路径和测试文件夹
MODEL_PATH = r"checkpoints_pi\raft_epoch_025.pth"
DATA_DIR = r"E:\Project\VSI\Study\step_SIM_V0.1"

# ===============================
# 工具函数
# ===============================
def numeric_sort_key(filename):
    """按文件名中的数字排序"""
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else 0

def load_gray(path):
    """加载灰度图 -> Tensor [1,1,H,W]"""
    img = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    img = (img - img.mean()) / (img.std() + 1e-6)  # 与训练时一致的归一化
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor

def build_model():
    model = RAFTVibPhase(iters=12, corr_radius=4).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"✅ 模型加载完成: {MODEL_PATH}")
    return model

@torch.no_grad()
def test_vibphase_sequence(data_dir):
    imgs = sorted(
        [f for f in os.listdir(data_dir) if f.lower().endswith(".bmp")],
        key=numeric_sort_key
    )
    if len(imgs) < 2:
        raise ValueError("图像数量不足")

    pairs = [(imgs[i], imgs[i+1]) for i in range(len(imgs)-1)]
    model = build_model()

    vib_list = []

    for idx, (imgA, imgB) in enumerate(pairs, 1):
        img1 = load_gray(os.path.join(data_dir, imgA))
        img2 = load_gray(os.path.join(data_dir, imgB))
        pair = torch.cat([img1, img2], dim=1)  # [1,2,H,W]

        flow_pred, vib_pred = model(pair)
        vib_value = vib_pred.item()
        vib_list.append(vib_value)

        print(f"[{idx:04d}] {imgA} -> {imgB}: vibphase = {vib_value:.6f} rad")

    # ===============================
    # 保存 vibphase（不累计）
    # ===============================
    save_path = os.path.join(data_dir, "infphase.txt")

    indices = np.arange(1, len(vib_list) + 1).reshape(-1, 1)
    vib_array = np.array(vib_list, dtype=np.float32).reshape(-1, 1)
    data_to_save = np.hstack([indices, vib_array])

    np.savetxt(save_path, data_to_save, fmt=["%d", "%.6f"])
    print(f"✅ 已保存 vibphase 到: {save_path}")
    # ===============================
    # 绘制 vibphase - pi/2
    # ===============================
    vib_array = np.array(vib_list, dtype=np.float32)
    vib_minus_pi2 = vib_array - np.pi / 2

    plt.figure(figsize=(10, 4))
    plt.plot(vib_minus_pi2, marker='o', linewidth=1)
    plt.axhline(0, color='r', linestyle='--', label='0')
    plt.xlabel('Frame Pair Index')
    plt.ylabel('vibphase - π/2 (rad)')
    plt.title('Predicted Vibphase Deviation from π/2')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return vib_list

if __name__ == "__main__":
    vib_list = test_vibphase_sequence(DATA_DIR)
    print("\n✅ 推理完成！共处理帧数:", len(vib_list))