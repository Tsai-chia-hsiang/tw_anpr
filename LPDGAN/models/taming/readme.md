Reference: [OCR-VQGAN](https://github.com/joanrod/ocr-vqgan.git) ([WACV2023](https://arxiv.org/abs/2210.11248)), we use it as the perceptual loss for training LPDGAN

- please go to [craft_mlt_25k.pth](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) (from [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)) to download the The OCR detector model used in OCR Perceptual loss
  - Put it at [./modules/autoencoder/ocr_perceptual/](./modules/autoencoder/ocr_perceptual/)
