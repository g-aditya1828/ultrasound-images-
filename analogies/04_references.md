# 📚 References

All papers cited across the analogies folder, in citation order.

---

## Core Architecture & Methods

**[1]** Ronneberger, O., Fischer, P., & Brox, T. (2015).  
**U-Net: Convolutional Networks for Biomedical Image Segmentation.**  
*Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, LNCS 9351, pp. 234–241.  
🔗 https://arxiv.org/abs/1505.04597  
> *The foundational paper for U-Net. Introduced the encoder–decoder with symmetric skip connections specifically for biomedical segmentation with limited training data. Directly motivates our choice of U-Net architecture.*

---

**[2]** He, K., Zhang, X., Ren, S., & Sun, J. (2016).  
**Deep Residual Learning for Image Recognition.**  
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778.  
🔗 https://arxiv.org/abs/1512.03385  
> *Introduced residual connections (skip connections inside the encoder), solving the vanishing gradient problem in very deep networks. We use ResNet34 as our pretrained encoder backbone.*

---

**[3]** Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Rueckert, D. (2018).  
**Attention U-Net: Learning Where to Look for the Pancreas.**  
*Medical Imaging with Deep Learning (MIDL).*  
🔗 https://arxiv.org/abs/1804.03999  
> *Introduced additive attention gates on U-Net skip connections. Demonstrates that spatial attention improves segmentation of small, variable-location structures — directly applicable to aponeurosis and fascicle detection.*

---

**[4]** Abraham, N., & Khan, N. M. (2019).  
**A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation.**  
*IEEE International Symposium on Biomedical Imaging (ISBI)*, pp. 683–687.  
🔗 https://arxiv.org/abs/1810.07842  
> *Introduced the Focal Tversky Loss — a generalisation of the Tversky index with a focal term that upweights hard examples. Combined with Attention U-Net for small lesion segmentation. We use this as the recall-biased component of our ComboLoss.*

---

**[5]** Shit, S., Paetzold, J. C., Sekuboyina, A., Ezhov, I., Bhatt, A., Lobanov, A., ... & Menze, B. H. (2021).  
**clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.**  
*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 16560–16569.  
🔗 https://arxiv.org/abs/2003.07311  
> *Introduced clDice (centerline Dice), a differentiable loss function computed on the soft skeleton of predicted and ground-truth masks. Specifically designed for tubular and linear structures (vessels, roads, fascicles). We apply this to the fascicle model only.*

---

## Alternative & Comparative Methods

**[6]** Rana, M., Hamarneh, G., & Wakeling, J. M. (2009).  
**Automated Tracking of Muscle Fascicle Orientation in B-mode Ultrasound Images.**  
*Journal of Biomechanics*, 42(13), pp. 2068–2073.  
🔗 https://doi.org/10.1016/j.jbiomech.2009.06.003  
> *Demonstrates semi-automated fascicle detection using Hough transform on ultrasound images. Reports 15–20% error rates and significant failure on low-contrast images — establishing the motivation for a deep learning approach.*

---

**[7]** Hao, J., Kosaraju, S. C., Tsaku, N. Z., Song, D. H., & Kang, M. (2020).  
**PAGE-Net: Interpretable and Integrative Deep Learning for Survival Analysis Conditioned on Partitioned Graph and Omic Data.**  
*Pacific Symposium on Biocomputing*, pp. 338–349.  
*(Note: for the ResNet encoder comparison in medical imaging, see also the nnU-Net paper [12] which systematically benchmarks encoder choices.)*

---

**[8]** Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).  
**Focal Loss for Dense Object Detection.**  
*IEEE International Conference on Computer Vision (ICCV)*, pp. 2980–2988.  
🔗 https://arxiv.org/abs/1708.02002  
> *Introduced Focal Loss to address the extreme foreground–background class imbalance in object detection. The core idea (down-weighting easy negatives) directly informs the focal term in our ComboLoss.*

---

## Aspirational Methods

**[9]** Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., & Wang, M. (2022).  
**Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation.**  
*European Conference on Computer Vision Workshops (ECCVW).*  
🔗 https://arxiv.org/abs/2105.05537  
> *Applies a Swin Transformer encoder (hierarchical shifted-window self-attention) in place of CNN encoder for medical image segmentation. Achieves state-of-the-art on multi-organ segmentation benchmarks but requires large pretraining datasets.*

---

**[10]** Tarvainen, A., & Valpola, H. (2017).  
**Mean Teachers are Better Role Models: Weight-averaged Consistency Targets Improve Semi-supervised Deep Learning Results.**  
*Advances in Neural Information Processing Systems (NeurIPS)*, 30.  
🔗 https://arxiv.org/abs/1703.01780  
> *Introduced the Mean Teacher semi-supervised framework — teacher model as exponential moving average of student, consistency regularisation between augmented views. A strong baseline for semi-supervised medical image segmentation.*

---

**[11]** Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).  
**Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.**  
*Advances in Neural Information Processing Systems (NeurIPS)*, 30.  
🔗 https://arxiv.org/abs/1612.01474  
> *Demonstrated that training multiple independent networks and averaging their predictions provides well-calibrated uncertainty estimates. Simpler and often more reliable than Bayesian methods like MC Dropout.*

---

**[12]** Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).  
**nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation.**  
*Nature Methods*, 18, pp. 203–211.  
🔗 https://doi.org/10.1038/s41592-020-01008-z  
> *nnU-Net automatically configures all hyperparameters of a U-Net pipeline (resolution, patch size, normalisation, training schedule) from dataset properties alone. Achieves state-of-the-art on 23 of 23 medical segmentation benchmarks without manual tuning. The gold standard for comparison.*

---

## Domain-Specific Background

**[13]** Ritsche, P., Wirth, P., Cronin, N. J., Sarto, F., Narici, M. V., Faude, O., & Franchi, M. V. (2022).  
**DeepACSA: Automatic Segmentation of Anatomical Cross-Sectional Area in Ultrasound Images of Lower Limb Muscles Using Deep Learning.**  
*Medicine & Science in Sports & Exercise*, 54(11), pp. 1–9.  
🔗 https://doi.org/10.1249/MSS.0000000000002980  
> *Introduces DL-Track — the open-source deep learning tool for muscle fascicle and aponeurosis detection that inspired parts of our geometry engine design.*

---

**[14]** Marziliano, A., Rago, G., Nonnenmacher, E., & Galasso, F. (2022).  
**Automated Muscle Architecture Estimation from Ultrasound: A Survey.**  
*Ultrasound in Medicine & Biology*, 48(6), pp. 985–1001.  
> *Comprehensive survey of classical and deep learning approaches to muscle architecture estimation from ultrasound. Good overview of the field and motivation for standardised benchmarks.*

---

## Quick Reference Table

| Citation | Paper | Used for |
|----------|-------|----------|
| [1] | U-Net (Ronneberger 2015) | Base architecture |
| [2] | ResNet (He 2016) | Encoder backbone |
| [3] | Attention U-Net (Oktay 2018) | Attention gates |
| [4] | Focal Tversky (Abraham 2019) | Loss function component |
| [5] | clDice (Shit 2021) | Loss function component |
| [6] | Hough fascicle tracking (Rana 2009) | Baseline comparison (classical CV) |
| [8] | Focal Loss (Lin 2017) | Loss function inspiration |
| [9] | Swin-UNet (Cao 2022) | Aspirational architecture |
| [10] | Mean Teacher (Tarvainen 2017) | Aspirational semi-supervised |
| [11] | Deep Ensembles (Lakshminarayanan 2017) | Aspirational uncertainty |
| [12] | nnU-Net (Isensee 2021) | Aspirational auto-configuration |
| [13] | DL-Track (Ritsche 2022) | Domain inspiration |
