# AIRL — Internship Coding Assignment

**Author:** *(Your Name Here)*  
**Repo contents:** `q1.ipynb`, `q2.ipynb`, `README.md`

---

## How to Run (Google Colab)

1. Open the repository in **Google Colab**  
   - Go to <https://colab.research.google.com>  
   - Click **File → Open notebook → GitHub**, then paste your repo URL  
   - Or upload `q1.ipynb` / `q2.ipynb` manually  

2. In Colab, go to **Runtime → Change runtime type → GPU** (e.g., T4 / P100 / A100)  
   Then choose **Runtime → Run all**  

3. Notes  
   - `q1.ipynb` trains a Vision Transformer on CIFAR-10 using PyTorch  
   - `q2.ipynb` performs text-prompted segmentation with SAM 2 and includes installation cells  

---

## Q1 — Vision Transformer (ViT) on CIFAR-10

### Goal
Implement a ViT that patchifies CIFAR-10 images, adds learnable positional embeddings, prepends a CLS token, stacks Transformer encoder blocks (MHSA + MLP + residual + norm), and classifies from the CLS token.

### Model Configuration

| Parameter | Value |
|------------|--------|
| Dataset | CIFAR-10 (32×32 RGB) |
| Patch size | 4 |
| Embedding dimension | 256 |
| Depth / blocks | 6 |
| Heads | 8 |
| MLP ratio | 4.0 |
| Dropout | 0.1 |
| Batch size | 128 |
| Epochs | 65 |
| Optimizer | AdamW (weight decay = 0.05) |
| LR scheduler | Cosine LR with warm-up |
| Learning rate | 3 × 10⁻⁴ |
| Loss | Cross-Entropy + label smoothing 0.1 |
| Augmentations | RandomCrop (32, padding 4), Horizontal Flip, RandAugment |
| Seed | 42 |

### Results

| Model / Config | CIFAR-10 Test Accuracy (%) |
|----------------|----------------------------|
| ViT (patch = 4, embed = 256, depth = 6, heads = 8) | **85.66 %** |

**Notes**
- Achieved 85.66 % average test accuracy on the CIFAR-10 test set.  
- Best model checkpoint chosen via validation accuracy tracking.  
- Mixed-precision training used for speed and memory efficiency.

---

## Q2 — Text-Driven Image Segmentation with SAM 2

### Pipeline
1. **Install dependencies** — `segment-anything`, `torch`, `opencv-python`, `matplotlib`  
2. **Load SAM 2 checkpoint** (`sam_vit_h`) and build predictor  
3. **Accept text prompt** from user (e.g., “dog”)  
4. **Seed generation:** current version uses a simple center-point seed  
   ```python
   points = np.array([[w // 2, h // 2]])
   labels = np.array([1])
   mask = predictor.predict(point_coords=points, point_labels=labels)[0]
   ```
5. **Display result:** overlay predicted mask on image  
6. *(Optional)* run the **video propagation** example to create `output_masked_video.mp4`

### Limitations
- Text prompts are accepted but not grounded automatically — the code uses a center-point heuristic instead of a text-to-region model.  
- Accuracy drops for off-center or multiple similar objects.  

### Possible Improvements
- Integrate **GroundingDINO**, **CLIPSeg**, or **GLIP** to map text → region seeds / boxes.  
- Add interactive point or box selection for improved localization.  
- Use mask propagation for smoother video segmentation.

---

## Bonus Analysis (Concise)

Experiments on CIFAR-10 show that using patch size = 4, embed dim = 256, and depth = 6 offers the best trade-off between accuracy and computation.  
Increasing depth to 8 gave only marginal gains (< 1 %) but required more careful learning-rate tuning to prevent overfitting.  
RandAugment improved generalization by roughly 2 % over simple crop-and-flip augmentation, and the AdamW + cosine scheduler combination was more stable than SGD with fixed LR.

---

## Reproducibility Checklist
✅ Use GPU runtime in Colab  
✅ Run `q1.ipynb` and `q2.ipynb` top-to-bottom successfully  
✅ Repo contains only `q1.ipynb`, `q2.ipynb`, and `README.md`  
✅ Report **85.66 % CIFAR-10 Accuracy** and repo link in the Google Form  
