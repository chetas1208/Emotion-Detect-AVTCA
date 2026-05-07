# AVT-CA Code Fixes — Change Report

**Branch:** `yuvraj/code-fix`  
**Baseline:** `yuvraj/code-comments-review`  
**Date:** May 2026  
**Author:** Yuvraj Gupta

---

## Executive Summary

The AVT-CA model is a research system that watches a video and listens to audio at the same time to recognize a person's emotion (angry, happy, sad, etc.). It was implemented based on a published research paper. After a detailed review, **8 bugs were found where the code did not match the design described in the paper**. This document explains each bug, what was wrong, what was fixed, and why the fix matters.

---

## Background: How the System Works (Plain English)

Imagine two specialists reviewing a video clip of someone speaking:

- **Specialist A (Audio)** listens only to the sound — tone of voice, pitch, rhythm.
- **Specialist B (Video)** watches only the face — expressions, eye movement, lip shape.

They independently analyze the clip, then briefly **consult each other** ("does what you heard match what I saw?"), continue their own analysis, consult again at the end, and then together make a final emotion prediction.

The paper describes exactly this process. The bugs below are places where the code skipped steps, mixed up the order, or made one specialist share a notepad with the other instead of having their own.

---

## Changes Made

---

### Fix 1 — The Final Consultation Step Was Completely Missing
**File:** `models/multimodalcnn.py`  
**Severity:** Critical

**What was supposed to happen:**  
After both specialists fully analyzed the clip on their own, they were designed to do one final exchange — Audio asks Video a final question, Video asks Audio a final question. Each updates their understanding before making the emotion prediction. This is the last "are we aligned?" step.

**What the code was doing:**  
This final exchange had zero lines of code. No module was defined for it. Both specialists finished their individual analysis, then were immediately merged together with no final consultation. The paper's own diagram clearly shows this step.

**The fix:**  
Two new attention modules were created — one where Audio consults Video, and one where Video consults Audio. These are now called at the correct point in the forward pass, just before the final prediction.

```python
# Before (missing entirely):
audio_pooled = x_audio_attention.mean([-1])   # jumped straight to pooling

# After (final cross-consultation added):
x_audio_final  = self.audioCrossAttention( xk=x_visual_ca, xq=x_audio_ca)
x_visual_final = self.visualCrossAttention(xk=x_audio_ca,  xq=x_visual_ca)
audio_pooled   = x_audio_final.max(dim=1).values
```

**Why it matters:**  
Without this step, the two branches never confirmed their understanding with each other before making a decision. The system was missing an entire architectural layer described in the paper, meaning results produced before this fix cannot be compared to the paper's reported accuracy numbers.

---

### Fix 2 — The "Number of Heads" Setting Was Silently Ignored
**File:** `models/multimodalcnn.py`  
**Severity:** Critical

**What was supposed to happen:**  
The paper specifically tests different configurations of attention "heads" — a tuning parameter that controls how many independent perspectives each specialist uses when consulting the other. The command-line flag `--num_heads` lets users reproduce the paper's experiments.

**What the code was doing:**  
The user's setting was received correctly as a function argument, then immediately overwritten on the very next line:

```python
def __init__(self, ..., num_heads=1):   # user's setting arrives here
    ...
    num_heads = 8                        # silently overwritten — user's value is gone
```

Every single run, regardless of what `--num_heads` was set to, used 8 heads.

**The fix:**  
Deleted the one offending line (`num_heads = 8`). The parameter now flows through correctly.

**Why it matters:**  
The paper's ablation study (comparing 1-head vs 4-head models) could not be reproduced from this code. Every experiment was secretly running an 8-head model. This also means any published results from this codebase may not correspond to the configurations the paper claims to have tested.

---

### Fix 3 — The Attention Step in the Video Branch Ran in the Wrong Order
**File:** `models/multimodalcnn.py`  
**Severity:** Critical

**What was supposed to happen:**  
The video processing pipeline is: (1) **look at the raw feature map and decide what's important** (attention), then (2) **process what's important deeply** (convolution). The paper's diagram shows attention first, then deep processing.

**What the code was doing:**  
The order was reversed:

```python
# Before (wrong order — process first, then decide what was important):
x = self.modulator(self.stage2(x)) + self.local(x)
#   ^attention on output    ^deep process happens first
```

The deep convolutional stage was running on unguided raw features, and the attention module was then looking at already-processed output — doing a fundamentally different job than intended.

**The fix:**  
Swapped the order so attention runs first:

```python
# After (correct order — decide first, then process deeply):
x = self.stage2(self.modulator(x)) + self.local(x)
```

**Why it matters:**  
Attention is designed to highlight what matters *before* expensive computation. Running it after means the deep processing had no guidance. Additionally, the model loads pretrained weights from a separately trained face recognition model (EfficientFace). Those weights were trained with the correct order — loading them into an inverted pipeline means the pretrained knowledge transfers incorrectly.

---

### Fix 4 — One Shared Attention Module Was Used for Both Audio and Video
**File:** `models/multimodalcnn.py`  
**Severity:** Critical

**What was supposed to happen:**  
The diagram shows two completely separate self-attention blocks — one dedicated to Audio, one dedicated to Video. Each specialist has their own private notepad that specializes over time in their own type of signal.

**What the code was doing:**  
One single module (`self.finalAttention`) was created and then called twice — once for Audio, once for Video:

```python
# Before (one module, used for both):
self.finalAttention = MultiheadAttention(e_dim, num_heads)
...
x_audio_attention,  _ = self.finalAttention(x_audio,  x_audio,  x_audio)
x_visual_attention, _ = self.finalAttention(x_visual, x_visual, x_visual)
```

During training, the same weights were simultaneously being pulled toward audio patterns and video patterns — two opposite directions. They couldn't specialize in either.

**The fix:**  
Created two separate, independent modules:

```python
# After (two dedicated modules):
self.audioAttention  = MultiheadAttention(e_dim, num_heads)
self.visualAttention = MultiheadAttention(e_dim, num_heads)
```

**Why it matters:**  
Audio attention needs to learn: "which moments in the sound carry emotion?" Video attention needs to learn: "which frames in the video carry emotion?" These are different problems requiring different learned weights. The model now has twice the attention capacity the paper intended, and each module can fully specialize.

---

### Fix 5 — Audio Was Processed With 1D Convolution; Diagram Shows 2D
**File:** `models/multimodalcnn.py`  
**Severity:** Medium

**What was supposed to happen:**  
The paper's diagram shows "3×3 CONV" for the audio branch — a two-dimensional convolution that scans across both frequency and time simultaneously. Audio is represented as a 2D grid: 10 frequency bands on one axis, ~172 time frames on the other.

**What the code was doing:**  
Conv1D was used, treating the 10 frequency bands as fixed channels and only scanning along the time axis. It could only ask "how does this sound change over time?" — never "how do these frequency bands relate to each other?"

**The fix:**  
The first two convolutional stages now use Conv2D, treating the audio as a proper 2D spectrogram image. The frequency dimension is collapsed into the channel dimension afterward, maintaining compatibility with the rest of the pipeline:

```python
def conv2d_block_audio(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )
```

**Why it matters:**  
Relationships between frequency bands carry critical information about voice pitch, tone, and emotional state. A rising pitch over time is a 2D pattern; a nasal quality involves specific frequency band relationships. Conv1D completely ignores this information. Conv2D can detect both time-based and frequency-based patterns.

---

### Fix 6 — Mean Pooling Used Instead of Max Pooling
**File:** `models/multimodalcnn.py`  
**Severity:** Medium

**What was supposed to happen:**  
The paper's diagram shows MAXPOOLING before the final prediction. Max pooling picks the strongest signal from across the entire clip — the single peak moment of emotion.

**What the code was doing:**  
Mean pooling was used everywhere, averaging all time steps equally:

```python
# Before:
audio_pooled = x_audio_attention.mean([-1])   # average of all frames
video_pooled = x_visual_attention.mean([-1])
```

**The fix:**

```python
# After:
audio_pooled = x_audio_final.max(dim=1).values   # peak emotional signal
video_pooled = x_visual_final.max(dim=1).values
```

**Why it matters:**  
A person may look neutral for most of a clip and show one strong moment of anger or fear at a single instant. Max pooling captures that peak. Mean pooling dilutes it by averaging it with all the neutral frames. For emotion recognition specifically, peak moments carry far more diagnostic information than the average state across the whole clip.

---

### Fix 7 — Audio Had 168 Time Steps, Video Had 15 — A Severe Mismatch
**File:** `models/multimodalcnn.py`  
**Severity:** Medium

**What was supposed to happen:**  
When the Audio and Video specialists consult each other (cross-attention), both should be operating at a comparable scale — a similar number of time steps — so the consultation is precise and meaningful.

**What the code was doing:**  
The audio pooling used `MaxPool1d(kernel=2, stride=1)`, which only reduces the sequence by 1 frame per layer. After 4 audio processing blocks:

```
Audio enters cross-attention:  168 time steps
Video enters cross-attention:   15 time steps
```

Audio had 168 "questions" to ask but Video only had 15 "answers" available. Each audio moment was forced to pick from only 15 video frames — an 11× mismatch.

**The fix:**  
Changed `MaxPool1d(2, 1)` to `MaxPool1d(2, 2)` — stride 2 instead of stride 1 — so each pooling layer halves the sequence length:

```
Audio after fix:  ~10–12 time steps  (comparable to video's 15)
```

**Why it matters:**  
Attention with a massive length mismatch becomes imprecise and noisy. With 168 audio positions attending to 15 video positions, the attention weights spread too thin to meaningfully align specific audio moments to specific video frames. Bringing both branches to a similar scale makes the cross-attention substantially more focused.

---

### Fix 8 — `ia` Fusion Was Discarding the Useful Part of Attention
**File:** `models/multimodalcnn.py`  
**Severity:** Medium

**What was supposed to happen:**  
When cross-attention runs between Audio and Video, it produces two things: (1) a new enriched version of the query incorporating what it learned from the other branch, and (2) a weight matrix showing where attention focused. The enriched representation is the useful output.

**What the code was doing:**  
The code was throwing away the enriched representation and keeping the weight matrix:

```python
# Before (wrong values kept):
_, h_av = self.av1(proj_x_v, proj_x_a)   # enriched output discarded
_, h_va = self.va1(proj_x_a, proj_x_v)   # enriched output discarded

x_audio  = h_va * x_audio    # raw weights used as volume knobs
x_visual = h_av * x_visual
```

The original features weren't replaced with anything new — they were just scaled up or down by raw attention scores.

**The fix:**

```python
# After (correct values kept):
h_av, _ = self.av1(proj_x_v, proj_x_a)   # enriched output kept
h_va, _ = self.va1(proj_x_a, proj_x_v)

x_audio  = x_audio  + h_av   # residual update with new information
x_visual = x_visual + h_va
```

**Why it matters:**  
Standard cross-attention should give Audio a brand new enriched version of itself informed by what Video saw. What was implemented was an unconventional gating mechanism — a volume control that never actually injected any new information from the other branch. The results from `ia` fusion mode were not comparable to any standard cross-attention implementation, and were inconsistent with how `it` fusion (the other mode) worked.

---

## Additional Improvements

Beyond the 8 architectural bugs, several diagnostic and quality improvements were made:

### Better Training Visibility
- **Gradient norm logging** added to every training step. This lets you immediately detect vanishing gradients (norm near 0, meaning the model has stopped learning) or exploding gradients (norm very large, meaning training is unstable).
- **GPU memory usage** printed during training so you know if you're close to running out of memory before a crash occurs.
- **Input tensor shapes** printed on the first batch of the first epoch to catch data pipeline mismatches immediately.
- **Per-class accuracy** printed after every validation epoch. Instead of only knowing "overall accuracy was 72%", you now see which emotions the model is good at and which it struggles with (e.g., "confused vs. neutral", "happy vs. calm").

### Device Compatibility
- Training now automatically detects Apple Silicon GPUs (MPS) in addition to NVIDIA GPUs (CUDA). This allows development on a Mac before running on the GPU server.

### Code Quality
- Noisy per-batch print statements in validation removed (previously printed one full line for every single batch, flooding the console).
- A misleading `assert` message fixed — previously calling `assert` with `print()` as the error message would always evaluate to truthy and never raise an error.
- Modulator input channel count corrected from 116 to 29 to match the actual tensor shape at the point it's called.

---

## Summary Table

<!-- HTML table: pipe tables cannot set column widths; Issue/Impact stay readable while Where stays a narrow code column. -->
<table>
<colgroup>
  <col style="width:3%;" />
  <col style="width:30%;" />
  <col style="width:14%;" />
  <col style="width:9%;" />
  <col style="width:44%;" />
</colgroup>
<thead>
<tr>
  <th align="right">#</th>
  <th align="left">Issue</th>
  <th align="left">Where</th>
  <th align="left">Severity</th>
  <th align="left">Impact</th>
</tr>
</thead>
<tbody>
<tr>
  <td align="right">1</td>
  <td>Final cross-attention block completely absent</td>
  <td><code>forward_feature_3</code></td>
  <td>Critical</td>
  <td>Architecture incomplete; results not reproducible</td>
</tr>
<tr>
  <td align="right">2</td>
  <td><code>--num_heads</code> CLI flag silently ignored</td>
  <td><code>MultiModalCNN.__init__</code></td>
  <td>Critical</td>
  <td>Ablation study unreproducible; all runs use 8 heads</td>
</tr>
<tr>
  <td align="right">3</td>
  <td>Attention runs after deep processing instead of before</td>
  <td><code>forward_features</code></td>
  <td>Critical</td>
  <td>Pretrained weights load incorrectly; wrong computation order</td>
</tr>
<tr>
  <td align="right">4</td>
  <td>One shared self-attention for both audio and video</td>
  <td><code>__init__</code> + <code>forward_feature_3</code></td>
  <td>Critical</td>
  <td>Half the attention capacity; no specialization</td>
</tr>
<tr>
  <td align="right">5</td>
  <td>Conv1D instead of Conv2D for audio</td>
  <td><code>conv1d_block_audio</code></td>
  <td>Medium</td>
  <td>Frequency-band relationships in audio ignored</td>
</tr>
<tr>
  <td align="right">6</td>
  <td>Mean pooling instead of max pooling</td>
  <td><code>forward_feature_3</code>, <code>forward_feature_2</code>, <code>forward_lt</code></td>
  <td>Medium</td>
  <td>Peak emotional signals diluted by averaging</td>
</tr>
<tr>
  <td align="right">7</td>
  <td>168 vs 15 time-step mismatch at cross-attention</td>
  <td><code>conv1d_block_audio</code></td>
  <td>Medium</td>
  <td>Cross-attention imprecise; 11× length imbalance</td>
</tr>
<tr>
  <td align="right">8</td>
  <td><code>ia</code> fusion keeps weight matrix, discards attended output</td>
  <td><code>forward_feature_2</code></td>
  <td>Medium</td>
  <td>No new information injected; unconventional undocumented gating</td>
</tr>
</tbody>
</table>

All 8 issues have been corrected. The codebase now implements the architecture as described in the paper.
