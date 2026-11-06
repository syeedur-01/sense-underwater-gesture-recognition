# WaterNet Pretrained Weights — Source Documentation

## Implementation used

**PyTorch community port: `tnwei/waternet`**  
GitHub: <https://github.com/tnwei/waternet>

Weights are loaded automatically via `torch.hub` — **no manual download needed**.  
They are cached to `~/.cache/torch/hub/checkpoints/` on first use.

```python
preprocess, postprocess, model = torch.hub.load('tnwei/waternet', 'waternet')
```

> **No weight file is needed in `weights/waternet/`.**  
> This directory is kept for documentation purposes only.

## Original paper

Li, C., Guo, C., Ren, W., Cong, R., Hou, J., Kwong, S., & Tao, D. (2019).  
*An Underwater Image Enhancement Benchmark Dataset and Beyond.*  
IEEE Transactions on Image Processing (TIP), 29, 4376–4389.  
DOI: [10.1109/TIP.2019.2955241](https://doi.org/10.1109/TIP.2019.2955241)

Original code (TensorFlow 1.x, **not used here**):  
<https://github.com/Li-Chongyi/Water-Net_Code>

## Architecture note

WaterNet uses a gated fusion network with three colour-processed inputs:
White Balance (WB), Histogram Equalization (HE), and Gamma Correction (GC).
The `tnwei` port faithfully reimplements this in PyTorch and trains on the
UIEB dataset (890 reference pairs).

```
Inputs : RGB image (HxWx3 uint8) → 4 derived tensors (RGB, WB, HE, GC)
Output : Enhanced image (HxWx3 uint8)
```

## Citation

```bibtex
@article{li2019underwater,
  title   = {An Underwater Image Enhancement Benchmark Dataset and Beyond},
  author  = {Li, Chongyi and Guo, Chunle and Ren, Wenqi and Cong, Runmin
             and Hou, Junhui and Kwong, Sam and Tao, Dacheng},
  journal = {IEEE Transactions on Image Processing},
  year    = {2020},
  volume  = {29},
  pages   = {4376--4389},
  doi     = {10.1109/TIP.2019.2955241}
}
```
