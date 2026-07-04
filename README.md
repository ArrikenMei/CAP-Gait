# CAP-Gait
Official implementation of **CAP-Gait: A Lightweight Condition-Adaptive Pose Modulation Network for Gait Recognition**.

CAP-Gait is a lightweight multimodal gait recognition model. It uses silhouettes as the dominant identity representation and uses 2D pose sequences as structural motion conditions. The pose branch does not act as an independent identity branch. Instead, it modulates early silhouette features through channel recalibration and spatial gating.

> Note: in this repository, the model class and config folder still use the experimental name `CAR_Gait` / `configs/cargait`. In the paper and this README, the method is referred to as **CAP-Gait**.

## Highlights

- Lightweight deployable backbone: about **4.7M** parameters after removing dataset-specific classification heads and training-only auxiliary modules.
- Silhouette-dominant and pose-guided asymmetric fusion.
- Condition-Adaptive Fusion (CAF) for channel modulation and spatial gating.
- Supports CCPG, CASIA-B, SUSTech1K, and Gait3D in the OpenGait framework.
- Uses standard gait retrieval evaluation with Euclidean distance.

- ## Environment

The code is based on OpenGait and PyTorch.

Recommended environment:

```text
Python >= 3.8
PyTorch >= 1.10
torchvision
pyyaml
tensorboard
opencv-python
tqdm
kornia
einops
numpy
```

Install common dependencies:

```bash
pip install torch torchvision
pip install tqdm pyyaml tensorboard opencv-python kornia einops numpy
```

If you use Conda:

```bash
conda create -n capgait python=3.8
conda activate capgait
pip install torch torchvision
pip install tqdm pyyaml tensorboard opencv-python kornia einops numpy
```

## Data Format

CAP-Gait requires both pose and silhouette sequences. Each sequence folder should contain:

```text
ID/condition/view/
|-- 0_pose.pkl
`-- 1_sil.pkl
```

Expected data:

- `0_pose.pkl`: 2D pose sequence, usually COCO-style keypoints with shape `[T, V, C]`, where `V=17` and `C=2` or `3`. If confidence is available, use `[x, y, confidence]`.
- `1_sil.pkl`: silhouette sequence with shape `[T, H, W]`.

For most CAP-Gait configs, set:

```yaml
data_cfg:
  data_in_use:
    - true   # 0_pose.pkl
    - true   # 1_sil.pkl
```

SUSTech1K uses the original OpenGait multi-file indexing convention, so its `data_in_use` is dataset-specific. See `configs/cargait/cargait_SUSTech1K_4070ti.yaml`.
