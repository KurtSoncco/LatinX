# ELF Replication and Baseline Experiment Notes

This document summarizes the experiment we are implementing based on the paper:

**Lightweight Online Adaption for Time Series Foundation Model Forecasts**  
ArXiv link: https://arxiv.org/pdf/2502.12920

The goal is to reproduce the paper’s core idea in a minimal, transparent setup:
a frozen time series foundation model (FM) combined with a lightweight online
adaptation module. We start with Chronos as the FM because it is easy to run on
CPU and exposes a clean inference API.

---

## 1. Motivation and big picture

Time series foundation models have shown strong zero-shot forecasting ability.
However, when applied to a new dataset or a non-stationary stream, they often
suffer from systematic bias or calibration errors.

Fully fine-tuning a large FM online is usually infeasible due to:
- high computational cost
- risk of catastrophic forgetting
- slow adaptation to distribution shifts

The paper proposes a lightweight adapter that can be trained online in closed
form, while keeping the FM frozen. This adapter is designed to be fast, stable,
and cheap to update.

---

## 2. Overview of the ELF method

The proposed method is called **ELF** and has two components:

### 2.1 ELF Forecaster
A linear model trained in the frequency domain that maps a context window to a
forecast horizon. It is trained with ridge regression and updated online using
a Woodbury identity.

### 2.2 ELF Weighter
A weighting mechanism that decides how much each training window should
contribute to the update, based on signals such as FM uncertainty or recent
forecast errors.

In our current work, we only implement the **ELF Forecaster**. The weighter will
be added later.

---

## 3. Why operate in the frequency domain?

Time series often exhibit strong periodic and seasonal structure. In the
frequency domain:
- low frequencies capture trend and seasonality
- high frequencies capture fast changes and noise
- linear regression becomes expressive with few parameters

The paper applies a frequency cropping rule controlled by a parameter `alpha`:
- keep very low frequencies
- keep very high frequencies
- discard most mid-range frequencies

This reduces dimensionality and improves numerical conditioning.

---

## 4. Forecasting formulation

Let:
- L be the context length
- H be the prediction horizon

Steps:
1. Transform the context window `x` of length L using FFT.
2. Keep a subset of frequency bins according to `alpha`, producing `d_x`
   complex features.
3. Transform the target window `y` of length H using RFFT.
4. Keep the first `d_y` low-frequency bins.
5. Learn a complex-valued linear map `W` from context frequencies to target
   frequencies.
6. Zero-pad the predicted spectrum and apply inverse RFFT to recover the
   forecast in the time domain.

---

## 5. Closed-form online learning

The ELF forecaster is trained via ridge regression:

W = (X* X + λI)⁻¹ X* Y

where:
- X is the design matrix in frequency space
- Y is the target spectrum matrix
- X* denotes conjugate transpose
- λ is a ridge regularization coefficient

To support online updates efficiently, we maintain:
- A_inv = (X* X + λI)⁻¹
- XTY = X* Y

When a batch of new windows arrives:
- update XTY by accumulation
- update A_inv using a Woodbury identity
- recompute W = A_inv @ XTY

---

## 6. Baseline handling and numerical stability

Several implementation details are critical for stability:
- FFT, RFFT, and IRFFT use orthonormal scaling (`norm="ortho"`).
- A consistent per-window baseline is used for both context and target.
  We default to subtracting the last context value.
- A small jitter term is added to the Woodbury system.
- Ridge coefficient λ is kept reasonably large in early experiments.

---

## 7. Experimental setup

We begin with a synthetic dataset to validate correctness:
- sine wave plus noise
- cosine wave plus noise

Windowing configuration:
- context length L = 256
- horizon H = 24
- stride = 8

Each dataset item contains:
- context window
- target window
- series identifier and time index

---

## 8. Foundation model: Chronos

Chronos is used as the frozen FM because:
- it runs efficiently on CPU
- it supports probabilistic forecasting
- it integrates cleanly with PyTorch

For each context window:
- multiple forecast samples are drawn
- the median forecast is computed
- uncertainty bands are extracted for visualization

---

## 9. Comparison experiment

For a selected window, we plot:
- historical context
- true future target
- Chronos median forecast with uncertainty band
- ELF FFT forecast

This visualization is used for qualitative debugging and sanity checks.

---

## 10. Current evaluation goals

At this stage, we focus on:
- stability of ELF predictions
- correct scale and offset
- reasonable slope and trend behavior
- correct window alignment

We also log inference time, update time, and parameter norms.

---

## 11. Datasets used in the paper

The paper evaluates on:
- ETT datasets (ETTm1, ETTm2, ETTh1, ETTh2)
- Weather
- Traffic
- ECL
- Solar
- US Weather

Additional per-second datasets are evaluated in the appendix.

---

## 12. Dataset choice for replication

For initial replication, **ETTm1** is preferred because:
- it is distributed as a simple CSV
- it is widely used in forecasting literature
- loaders and preprocessing are well understood

---

## 13. Next steps

Planned extensions:
1. Switch from synthetic data to ETTm1.
2. Implement rolling online evaluation.
3. Add ELF Weighter and uncertainty-based weighting.
4. Reproduce paper metrics.
5. Study robustness under distribution shifts.

---

## 14. Reproducibility checklist

- Fix random seeds.
- Log all hyperparameters.
- Disable shuffling during debugging.
- Record window indices used for plots.
- Record Chronos model ID and sampling settings.

---

## 15. Summary

This experiment isolates the paper’s main insight:
a simple frequency-domain adapter can efficiently and stably adapt a frozen
foundation model for time series forecasting.

ArXiv reference:
https://arxiv.org/pdf/2502.12920
