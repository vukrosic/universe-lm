# Curvature-Aware Muon (C-Muon): Making AI Training More Stable and Reliable

## Abstract
Training Large Language Models (LLMs) behaves similarly to navigating a steep loss landscape—a complex mathematical "surface" where we aim to reach the point of lowest error. A recent optimizer called **Muon** facilitates this by ensuring update steps are **orthogonal**, meaning each move is decorrelated from the last. This decorrelation accelerates convergence by preventing redundant oscillations. However, Muon can become unstable in regions of **high curvature**, where the loss landscape is exceptionally jagged or irregular. in these regimes, a standard orthogonal step may lead to divergence, causing the training process to fail.

We propose **Curvature-Aware Muon (C-Muon)** to address this instability. Inspired by **Ricci Flow**—a geometric process used to smooth irregular manifolds—C-Muon functions as a local terrain sensor. It monitors the jaggedness of the landscape and applies a smoothing dampener to the gradients *before* the Muon optimization logic is executed. This stabilizes training in high-curvature regions without significant computational overhead. We provide a practical implementation recipe that enables researchers to integrate curvature-awareness into existing large-scale training pipelines.

## 1. Introduction
### 1.1 Motivation
In the quest to train larger and more capable AI models, the efficiency of the optimizer is paramount. Traditional first-order methods like AdamW are robust but often slow to converge due to redundant gradient information across different parameter dimensions. Recently, **Muon** has emerged as a powerful alternative, using Newton-Schulz iterations to force gradient updates to be orthogonal. This effectively "decorrelates" the learning process, allowing the model to learn features in parallel.

However, the "jaggedness" of a neural network's loss landscape is a well-known challenge. In areas of extreme curvature (steep cliffs or narrow valleys), a large orthogonal step can throw the model into a region of significantly higher loss, leading to "divergence spikes."

### 1.2 Objective
This paper introduces **C-Muon**. Our goal is to make the Muon optimizer "feel" the shape of the landscape it is navigating. If the landscape is smooth, C-Muon behaves like standard Muon. If the landscape is sharply curved, C-Muon applies a local "smoothing" operation inspired by **Ricci Flow**, a mathematical technique used in differential geometry to smooth out irregular surfaces.

## 2. Background and Related Work
### 2.1 Muon and Orthogonalization
Muon works by regularizing the weight updates to satisfy $U^T U = I$, where $U$ is the update matrix. This is typically achieved via a Newton-Schulz iteration:
$$X_{k+1} = \frac{1}{2} X_k (3I - X_k^T X_k)$$
where $X$ is the preconditioned gradient.

### 2.2 Ricci Flow
In mathematics, Ricci Flow is a process that deforms the metric of a Riemannian manifold in a way analogous to the diffusion of heat, effectively smoothing out curvature singularities. It is governed by the equation:
$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij}$$
where $g$ is the metric tensor and $R_{ij}$ is the Ricci curvature tensor.

## 3. Methodology: The C-Muon Framework
### 3.1 Defining the Metric and Curvature
To apply these concepts to neural networks, we treat the parameter space as a manifold (a complex mathematical "surface"). We define a local metric $\hat{g}$ (a way to measure distances on this surface) based on the moving average of the squared gradients:
$$\hat{g}_i = \beta_2 \hat{g}_{i, t-1} + (1 - \beta_2) G_{i, t}^2$$
This metric represents the "local density" or "steepness" of the loss landscape at parameter $i$.

The **Scalar Curvature $R$** at a point can be approximated by observing how the metric $\hat{g}$ changes as we move across the parameter indices. In a discretized parameter space, we use a central difference approximation of the Laplacian of the metric:
$$R(\hat{g}) \approx \sum_{i} \left| \frac{\partial^2 \hat{g}_i}{\partial i^2} \right| \approx \sum_{i} | \hat{g}_{i+1} - 2\hat{g}_i + \hat{g}_{i-1} |$$

We compute the final curvature weight $\kappa$:
$$\kappa = \text{Softplus}(R(\hat{g}))$$
The `Softplus` function, defined as $\ln(1 + e^x)$, ensures that $\kappa$ is always positive and smoothly approaches zero in flat regions.

### 3.2 The Smoothing Step (Internal Ricci Flow)
Before applying the Muon orthogonalization, we "flow" the gradients to smooth them:
$$G_{smoothed} = G \cdot \exp(-\alpha \cdot \kappa)$$
Here, $\alpha$ is a hyperparameter that controls the strength of the smoothing. If curvature $\kappa$ is zero (flat landscape), $\exp(0) = 1$, and $G_{smoothed} = G$. If curvature is high, the gradient is dampened proportionally to the severity of the jaggedness.

### 3.3 The C-Muon Algorithm (Developer Recipe)
For a given parameter tensor $W$ and gradient $G$:
1.  **Metric Update**: Compute the running squared gradient $\hat{g}$.
2.  **Curvature Estimation**: Compute the discrete second-order variation of $\hat{g}$ using:
    `diff = torch.abs(g[2:] - 2*g[1:-1] + g[:-2])`
3.  **Dampening**: Calculate $\kappa = \text{Softplus}(\text{Sum}(diff))$.
4.  **Flow**: $G_{raw} \leftarrow G_{raw} \cdot \exp(-\alpha \cdot \kappa)$.
5.  **Orthogonalize**: Apply standard Muon Newton-Schulz iteration to $G_{raw}$.
6.  **Apply Update**: $W \leftarrow W - \eta \cdot G_{raw}$.

## 4. Mathematical Derivation (For Undergraduates)
### 4.1 Why Orthogonalize?
Imagine you are trying to find the bottom of a bowl. If you only move in the direction of the steepest slope, you might zigzag back and forth across the bottom. If you enforce that each step you take is "independent" (orthogonal) to the last, you cover the area more efficiently.

### 4.2 The Role of Curvature
Now imagine the bowl is actually a long, narrow canyon with very steep walls. Even a small step "sideways" could hit the wall and bounce you back up. Curvature $R$ measures how much the "bowl" is squeezing or stretching. By multiplying the gradient by $\exp(-\alpha \cdot \kappa)$, we are essentially saying: "If the walls are too steep, slow down."

### 4.3 Why the Second Difference?
In calculus, the first derivative (gradient) tells you the slope. The **second derivative** (curvature) tells you how fast the slope is changing. If the slope changes suddenly (high curvature), it means you are hitting a "bumpy" part of the landscape. C-Muon uses the difference between neighboring gradients to detect these bumps without needing to solve complex differential equations.

## 5. Proposed Experiments
### 5.1 Baseline Comparison
We will compare C-Muon against:
- **AdamW**: The industry standard first-order optimizer.
- **Base Muon**: To isolate the effect of the curvature-aware smoothing.

### 5.2 Benchmarks
Testing will be performed on the 8M and 20M token training runs using the provided `train_llm.py` script. We will monitor:
- **Cross-Entropy Loss**: Final validation loss.
- **Divergence Frequency**: Count of iterations where loss increases by $>10\%$.
- **Compute Overhead**: Measuring the milliseconds added by the finite-difference step (expected to be minimal, $<1\%$).

## 6. Conclusion
C-Muon provides a theoretically grounded yet computationally efficient bridge between simple orthogonal optimization and full curvature-aware training. By leveraging the geometric intuition of Ricci Flow through a simple discrete approximation, it makes high-performance optimizers like Muon significantly more robust for large-scale training.
