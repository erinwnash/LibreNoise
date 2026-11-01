# Clean-Room Functional Specification: 3D Collaborative Spatiotemporal Shrinkage
**Document ID:** `SPEC-CR-001`  
**Scientific Basis:** Maggioni, Boracchi, Foi, Egiazarian (2012), *"Video Denoising, Deblurring, and Highly Displaced PCA Reconstruction"* (IEEE Transactions on Image Processing, Vol. 21, No. 11, pp. 4452-4469).  
**Notice to Implementers:** Implement code solely from the mathematical formulations and algorithms defined herein. **Do not look at, execute, or reference non-commercial code (TAU MATLAB/Python repositories).**

---

## 1. Problem Formulation

Let $z(x, t)$ denote the noisy observation of a video frame at spatial coordinates $x = (u, v) \in \mathbb{R}^2$ and temporal index $t \in \mathbb{Z}$:
$$z(x, t) = y(x, t) + \eta(x, t)$$
where $y(x, t)$ is the true uncorrupted signal and $\eta(x, t) \sim \mathcal{N}(0, \sigma^2(y(x, t)))$ is signal-dependent sensor noise. After variance-stabilizing transformation (Generalized Anscombe Transform), the noise is approximately additive white Gaussian noise with unit variance:
$$\tilde{z}(x, t) \approx \tilde{y}(x, t) + \tilde{\eta}(x, t), \quad \tilde{\eta} \sim \mathcal{N}(0, 1)$$

---

## 2. Processing Pipeline Overview

The collaborative filtering algorithm operates in two consecutive passes over sliding reference patches:
1. **Pass 1: Hard-Thresholding Initial Estimate ($\hat{y}^{\text{basic}}$)**
   - Grouping: Find similar patches in spatial neighborhoods across temporal trajectory $\mathcal{T}_t$.
   - 3D Transform: Apply 2D spatial orthogonal transform followed by 1D temporal transform along grouping axis.
   - Shrinkage: Hard thresholding of 3D spectrum using noise variance threshold $\lambda_{3D} \sigma$.
   - Inverse Transform & Aggregation: Inverse 3D transform and weighted spatial accumulation.
2. **Pass 2: Empirical Wiener Filtering Final Estimate ($\hat{y}^{\text{final}}$)**
   - Grouping: Match patches on the *basic estimate* $\hat{y}^{\text{basic}}$ (cleaner distance metric).
   - Co-transformation: Transform both basic estimate patches and raw noisy patches.
   - Shrinkage: Apply empirical Wiener attenuation factors computed from the energy spectrum of the basic estimate.
   - Aggregation: Adaptive weighted accumulation onto the output canvas.

---

## 3. Mathematical Details

### 3.1 Patch Definition & Grouping
Let $P_t(x)$ be a square patch of size $N_1 \times N_1$ centered at coordinate $x$ in frame $t$.
Given a reference patch $P_t(x_R)$, the distance between $P_t(x_R)$ and a candidate patch $P_{t+k}(x_C)$ located within search neighborhood $S_{x_R}$ is:
$$d(P_t(x_R), P_{t+k}(x_C)) = \frac{1}{N_1^2} \| P_t(x_R) - P_{t+k}(x_C) \|_2^2$$

- **Motion-Compensated Search**: The search center at frame $t+k$ is displaced by motion vector $v_k(x_R)$:
  $$x_C \in x_R + v_k(x_R) + [-W, W]^2$$
- **Residual Trust Gating**: If $d(P_t(x_R), P_{t+k}(x_C)) > \tau_{\text{trust}} \cdot 2\sigma^2$, the candidate is rejected (temporal trust = 0) to avoid ghosting across occlusions.
- **Group Formation**: The top $N_2$ candidate patches having $d \le \tau_{\text{match}}$ are stacked into a 3D tensor $\mathbf{Z}_{x_R} \in \mathbb{R}^{N_1 \times N_1 \times K}$ where $K \le N_2$.

### 3.2 3D Spatiotemporal Transform
The separable 3D transform is defined as:
$$\mathbf{\Theta} = \mathcal{T}_{\text{temp}} \left( \mathcal{T}_{\text{spat}} \left( \mathbf{Z}_{x_R} \right) \right)$$
- Spatial Transform $\mathcal{T}_{\text{spat}}$: 2D Biorthogonal Wavelet or 2D Discrete Cosine Transform (DCT).
- Temporal Transform $\mathcal{T}_{\text{temp}}$: 1D Haar Transform or 1D DCT along the group axis $K$.

### 3.3 Pass 1: Hard Thresholding
$$\mathbf{\Theta}_{\text{hard}}(u, v, k) = \begin{cases} 
\mathbf{\Theta}(u, v, k) & \text{if } |\mathbf{\Theta}(u, v, k)| \ge \lambda_{\text{hard}} \cdot \sigma \\
0 & \text{otherwise}
\end{cases}$$
The inverse transform yields patch estimates:
$$\hat{\mathbf{Y}}_{x_R}^{\text{basic}} = \mathcal{T}_{3D}^{-1} \left( \mathbf{\Theta}_{\text{hard}} \right)$$
The aggregation weight for this 3D group is proportional to the inverse of non-zero coefficients:
$$w_{\text{basic}}(x_R) = \frac{1}{\sigma^2 \cdot \max(1, N_{\text{non-zero}}(\mathbf{\Theta}_{\text{hard}}))}$$

### 3.4 Pass 2: Empirical Wiener Filtering
Form 3D groups using coordinates determined from $\hat{y}^{\text{basic}}$.
Transform both basic estimate group $\mathbf{\Phi}^{\text{basic}}$ and noisy group $\mathbf{\Phi}^{\text{noisy}}$:
$$W_{\text{wiener}}(u, v, k) = \frac{|\mathbf{\Phi}^{\text{basic}}(u, v, k)|^2}{|\mathbf{\Phi}^{\text{basic}}(u, v, k)|^2 + \sigma^2}$$
$$\mathbf{\Phi}_{\text{filt}}(u, v, k) = W_{\text{wiener}}(u, v, k) \cdot \mathbf{\Phi}^{\text{noisy}}(u, v, k)$$
Inverse transform yields:
$$\hat{\mathbf{Y}}_{x_R}^{\text{final}} = \mathcal{T}_{3D}^{-1} \left( \mathbf{\Phi}_{\text{filt}} \right)$$
The final aggregation weight is:
$$w_{\text{final}}(x_R) = \frac{1}{\sigma^2 \cdot \sum_{u, v, k} W_{\text{wiener}}(u, v, k)^2}$$

### 3.5 Seamless Fallback to 2D Spatial
If no temporal candidates meet the residual trust criterion, $K = 1$. The 1D temporal transform degenerates to an identity operation, and the filter gracefully collapses into a pure 2D spatial collaborative filter without switching modes or introducing branch artifacts.

---

## 4. Engineering & Implementation Requirements
- **Data Types**: Internal representation is single-precision floating point (`float`, IEEE 754).
- **SIMD Layout**: Organize patch buffers to allow 8-wide AVX2 / 4-wide NEON vectorization for patch differences and DCT butterfly passes.
- **Tiling**: Frames must be processed in independent spatial tiles with overlap margins $\ge N_1$ to guarantee zero boundary artifacts and bounded working memory.
- **Determinism**: Aggregation order must be deterministic regardless of CPU thread count.
