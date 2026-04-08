# 0. Git clone
git clone org-16943930@github.com:facebookresearch/vip.git
git clone git@github.com:zcczhang/UVD.git

# 1. Create environment
conda activate base
conda create -n uvd python=3.9
conda activate uvd

# 2. Downgrade packaging tools (important for compatibility)
pip install "pip<24.1"
pip install "setuptools==65.5.0"
pip install "wheel<0.38"

# 3. Install VIP (dependency)
cd ./vip
pip install -e .

# 4. Pre-install MuJoCo (avoid incompatibility with newer versions)
pip install "mujoco==2.3.3"

# 5. Install UVD
cd ../UVD
pip install -e . --no-build-isolation

# 6. (Optional) install ipython for convenience
pip install ipython

import torch
import uvd
import matplotlib.pyplot as plt

# Extract subgoals from a video
subgoals = uvd.get_uvd_subgoals(
    "/path/to/video.mp4",  # video file path
    preprocessor_name="vip",  # one of ["vip", "r3m", "liv", "clip", "vc1", "dinov2"]
    device="cuda" if torch.cuda.is_available() else "cpu",
    return_indices=False,  # set True to return only subgoal indices
)

# subgoals: (N, H, W, 3)
N = subgoals.shape[0]

fig, axes = plt.subplots(1, N, figsize=(4 * N, 4))

# Handle the case when N == 1
if N == 1:
    axes = [axes]

for i in range(N):
    axes[i].imshow(subgoals[i])
    axes[i].axis("off")
    axes[i].set_title(f"{i}")

plt.tight_layout()
plt.show()
