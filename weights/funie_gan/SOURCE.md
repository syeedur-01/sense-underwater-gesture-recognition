# FUnIE-GAN Pretrained Weights — Source Documentation

## Weight file

`weights/funie_gan/generator.pth`

## Source

- **Paper:** Islam, M. J., Xia, Y., & Sattar, J. (2020).
  *Fast Underwater Image Enhancement for Improved Visual Perception.*
  IEEE Robotics and Automation Letters (RA-L), 5(2), 3227–3234.
  DOI: [10.1109/LRA.2020.2974710](https://doi.org/10.1109/LRA.2020.2974710)

- **Official GitHub repository:**
  <https://github.com/xahidbuffon/FUnIE-GAN>

- **Weight download (generator — Paired model):**
  From the repository release / Google Drive link listed in the README:
  `saved_models/Generator.pth`
  Mirror: <https://drive.google.com/drive/folders/1UuNPv1Rj2jeSN10kFY9OmP-mVFiOKF8t>

## Conversion note

The released checkpoint (`Generator.pth`) was saved with `torch.save(model.state_dict(), ...)`.
Rename it to `generator.pth` and place it in this directory.

## Architecture

FUnIE-GAN Generator — a 5-block encoder–decoder (U-Net style) with instance normalization,
input resolution flexible (internally pads to multiples of 32), output: RGB image same size as input.

```
Input shape : (B, 3, H, W)   — any resolution; pads internally
Output shape: (B, 3, H, W)   — same as input, pixel values in [−1, 1] (tanh)
```

Post-processing: scale output from [−1, 1] → [0, 255] and clamp.

## Citation

```bibtex
@article{islam2020fast,
  title   = {Fast Underwater Image Enhancement for Improved Visual Perception},
  author  = {Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2020},
  volume  = {5},
  number  = {2},
  pages   = {3227--3234},
  doi     = {10.1109/LRA.2020.2974710}
}
```

## Licence

MIT (as of the original repository). Verify before redistribution.

