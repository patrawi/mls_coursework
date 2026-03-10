# HW1-ASR Triton Track: Complete Prioritised Action Plan

> **Course:** INFR11269 — Machine Learning Systems (MSc AI, University of Edinburgh)
> **Task:** Build a GPU-accelerated inference engine for GLM-ASR (1.5B parameter speech recognition model) using Triton
> **Track:** Triton (recommended for H200 GPU compatibility on teaching cluster)

---

## The Big Picture

You are implementing 10 Triton GPU kernels across 3 files (`layers.py`, `attention.py`, `rope.py`) that power a real speech-to-text model. The work breaks into two stages:

1. **Make it work** — implement all TODO kernels so the model produces correct transcriptions
2. **Make it fast** — apply 3 required optimizations to beat the baseline

Your **baseline** to beat is `glm_asr_triton_example/` (the unoptimized Triton reference). PyTorch (`glm_asr_scratch/`) is the aspirational target — the TA showed it's possible to reach 1.19x faster than PyTorch with comprehensive optimization.

### Understanding the Baseline and the Goal

This is critical to understand — there are **three reference points**, not two:

**1. The Baseline You Must Beat: `glm_asr_triton_example/`**

This is an unoptimized Triton implementation. It uses `torch.einsum` for attention (materialising the full O(n²) matrix), has no kernel fusion, and uses basic tile sizes. From the TA's earlier demo session, the cuTile equivalent (V1) ran at ~3000ms — similar to NumPy-level performance. The lecturer in the homework session confirmed: "the Triton example and cuTile example — you can regard as the baseline. You want to beat them."

Your template starts here. Once you fill in the TODOs and add the three required optimisations, your code must run faster than this baseline. This is the minimum bar.

**2. The PyTorch Reference: `glm_asr_scratch/`**

This is a pure PyTorch implementation showing the model architecture with explicit loops, no optimisations, and clear comments. Its purpose is **mathematical reference only** — it shows you the exact formulas (RMSNorm, attention, RoPE, SwiGLU) so you know what to implement. The lecturer said: "you can see how the mathematical equation has been implemented using basic Pytorch operator, so that you are in a very good position knowing exactly what functions you need to call."

However, PyTorch with `torch.compile` is actually highly optimised under the hood (it generates Triton kernels internally via torch inductor). From the TA's week 2 demo: "PyTorch represents a decade of engineering by a billion-dollar company. You shouldn't regard it as the baseline. Regard it as something you want to reach."

**3. The Ultimate Goal: Beat PyTorch**

The TA demonstrated that with ~2000 lines of optimised cuTile code (V10), he achieved **1.19x faster than PyTorch** — 0.585ms per layer vs PyTorch's 0.697ms. The full V1→V10 journey gave an 8.6x total speedup. The lecturer in the homework session reinforced this: "if you really want to write this homework into your resume, implementing 10 different techniques to beat a highly optimised kernel already written by Meta or OpenAI engineers can be a very good achievement to show to the recruiter."

You're NOT expected to reach V10. But the lecturer made clear that the depth of your optimisation directly affects your mark: "the different ways you really push the kernel performance to the very optimal can really reflect your thinking in the report."

**In summary:**

| Level | What | Your Goal |
|-------|------|-----------|
| **Baseline** (must beat) | `glm_asr_triton_example/` — unoptimised Triton with einsum attention, no fusion | Apply 3 required optimisations to go faster |
| **PyTorch** (aspirational) | `glm_asr_scratch/` with torch.compile — decade of Meta engineering | Match or approach with FlashAttention + fusion + tuning |
| **Beyond PyTorch** (stretch) | TA's V10 — fused QKV, custom RoPE, dedicated decode attention | 10+ techniques, ~2000 lines, 1.19x faster than PyTorch |

The key insight from the homework lecture: "using Triton and tuning the tile size, you should be able to optimise the performance compared to the basic implementation. If you can fuse the entire kernel, you can get even further performance." And for the ultimate version: "you can fuse all operations into a single Triton kernel without writing the result back — the result will be only written back once you finish the last layer."

### The ASR Pipeline You're Powering

```
Audio (WAV)
  → Mel Spectrogram (128 bins)
  → Conv Subsampler (4x downsample)
  → Audio Encoder (32 layers)       ← LayerNorm, GELU, Linear, Attention, RoPE
  → Projector (pool 4 frames, MLP)  ← GELU, Linear
  → Text Decoder (28 layers)        ← RMSNorm, SiLU, Linear, Attention, RoPE
  → Text Output
```

### Three Required Optimizations (Graded)

1. **Tile/block size tuning** — try at least 2–3 configs of BLOCK_M/N/K, num_warps, num_stages
2. **Kernel fusion (at least 1)** — combine 2+ operations into a single kernel
3. **FlashAttention-style attention** — streaming softmax, O(1) memory, blockwise QK^T

---

## Phase 0: Environment Setup [Day 1]

**Goal:** Get the teaching cluster working and verify the baseline runs.

### Steps

1. SSH into the teaching cluster (configure VS Code Remote-SSH with `mlp.inf.ed.ac.uk`)

2. Clone the course repo and set up the Triton environment:
   ```bash
   git clone <repo-url> && cd edin-mls-26-spring
   source utils/setup-triton.sh
   ```

3. Request a GPU node (do NOT run on login node):
   ```bash
   srun -p Teaching -w saxa --gres gpu:1 --mem=16G --pty bash
   ```

4. Verify the baseline example works:
   ```bash
   cd hw1-asr
   ./benchmark.sh glm_asr_triton_example
   ```

5. Expected output:
   ```
   Transcription: Concord returned to its place amidst the tents.
   Accuracy: 100.0%
   Status: PASS
   ```

**If this fails, STOP. Fix the environment before doing anything else.** Common issues: CUDA version mismatch, insufficient memory (add `--mem=16G` to srun).

### Alternative GPU Access

If the teaching cluster is unavailable:
- **RunPod**: ~$5–10 free credits per account (~10 hours GPU time)
- **Lightning.ai**: 15 hours free A100
- Your own NVIDIA GPU (Ampere+ / sm70+)

---

## Phase 1: Triton Tutorials — Learn the Language [Days 1–3]

**Goal:** Build Triton fluency before touching the homework. Each tutorial is 15–20 minutes and teaches patterns you'll use directly in the homework kernels.

### Lesson 1: Vector Add (15 min)

```bash
cd triton-tutorial/1-vectoradd && python vectoradd.py
```

**What you learn:** `@triton.jit`, `tl.program_id`, `tl.arange`, `tl.load`/`tl.store`, masking with `mask = offs < N`

**Maps to homework:** Every single kernel uses this load → compute → store pattern.

### Lesson 2: Execution Model (15 min)

```bash
cd triton-tutorial/2-execution-model && python grid_2d.py
```

**What you learn:** 2D grids with `tl.program_id(0)` and `tl.program_id(1)`, mapping program coordinates to data positions, stride-based pointer arithmetic.

**Maps to homework:** Attention kernels use 2D grids — `pid_bh` (batch×heads) and `pid_q` (query position). The linear kernel also uses a 2D grid over output tile rows and columns.

### Lesson 3: Data Model (15 min)

```bash
cd triton-tutorial/3-data-model && python datatype.py
```

**What you learn:** FP16 vs FP32, explicit casting with `.to(tl.float32)`, when lower precision helps.

**Maps to homework:** You can speed up inference by loading FP16 and accumulating in FP32. The `linear_kernel_tf32` uses TF32 tensor cores. ASR transcription doesn't need full FP32 precision.

### Lesson 4: Transpose (20 min)

```bash
cd triton-tutorial/4-transpose && python grid_2d.py
```

**What you learn:** 2D tile loading, `tl.trans()`, swapping store coordinates, memory coalescing.

**Maps to homework:** The attention score computation needs Q @ K^T — understanding how to load and transpose 2D tiles is essential for this.

### Lesson 5: Secret Notes (10 min)

Read `triton-tutorial/5-secret-notes/README.md` carefully. Key takeaways:

- `tl.constexpr` is required for block shapes and loop bounds
- Always use masks for out-of-bounds access
- Cast explicitly: `x_fp32 = x.to(tl.float32)` for accumulation
- Strides are in elements, not bytes
- `num_warps` controls parallelism inside a program; `num_stages` controls memory pipelining

**Maps to homework:** These are the gotchas that cause silent bugs. Every kernel needs correct masking and type handling.

### Summary: What You Should Know After Phase 1

- How to write a `@triton.jit` kernel and launch it with a grid
- How `tl.load` / `tl.store` work with pointer arithmetic and masks
- How 2D grids map programs to tile positions
- How to use strides to address multi-dimensional tensors
- The importance of explicit type casting for numerical accuracy

---

## Phase 2: Study the Example Code [Day 3–4]

**Goal:** Understand exactly what code you need to write by studying the working reference.

### Steps

1. Read the TODO comments in your template files:
   ```
   glm_asr_triton_template/layers.py
   glm_asr_triton_template/attention.py
   glm_asr_triton_template/rope.py
   ```

2. Diff the template against the example to see the exact code that fills each TODO:
   ```bash
   diff glm_asr_triton_template/layers.py glm_asr_triton_example/layers.py
   diff glm_asr_triton_template/attention.py glm_asr_triton_example/attention.py
   diff glm_asr_triton_template/rope.py glm_asr_triton_example/rope.py
   ```

3. Read `hw1-asr/GUIDE.md` sections 2 and 5 — they explain the model architecture and which kernel goes where:

   | Kernel | Audio Encoder | Projector | Text Decoder |
   |--------|:---:|:---:|:---:|
   | `layernorm` | ✓ | | |
   | `rmsnorm` | | | ✓ |
   | `gelu` | ✓ | ✓ | |
   | `silu` | | | ✓ |
   | `linear_kernel_tf32` | ✓ | ✓ | ✓ |
   | `attention_scores` | ✓ | | ✓ |
   | `softmax_inplace` | ✓ | | ✓ |
   | `attention_output` | ✓ | | ✓ |
   | `compute_freqs` (RoPE) | ✓ | | ✓ |
   | `softmax` (standalone) | | | ✓ |

4. Optionally read `model.py` to see how your kernels are assembled into the full pipeline.

---

## Phase 3: Implement the TODO Kernels [Days 4–10]

**Goal:** Get all 10 kernels producing correct output. Follow this order strictly — each phase builds on the previous one.

### Phase 3a: Element-wise Operations

**File:** `layers.py`
**Kernels:** `silu_kernel`, `gelu_kernel`
**Pattern:** Load a block of data → apply math → store result

These are the simplest kernels. They teach you the fundamental Triton pattern.

**Formulas:**
```
SiLU:  output = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
GELU:  output = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

**Test:**
```bash
cd glm_asr_triton_template && python layers.py
```

### Phase 3b: Reduction Operations

**File:** `layers.py`
**Kernels:** `softmax_kernel`, `rmsnorm_kernel`, `layernorm_kernel`
**Pattern:** Load a row → compute statistics (max, sum, mean) → normalize → store

These add `tl.max`, `tl.sum` and row-wise processing. Key insight: always use FP32 for accumulation even if inputs are FP16.

**Formulas:**
```
Softmax:   exp(x - max(x)) / sum(exp(x - max(x)))
RMSNorm:   x / sqrt(mean(x²) + eps) * weight
LayerNorm: (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

**Test after each kernel:**
```bash
python layers.py
```

### Phase 3c: Tiled Matrix Multiplication

**File:** `layers.py`
**Kernel:** `linear_kernel_tf32`
**Pattern:** 2D grid of output tiles, inner loop over K dimension with `tl.dot`

This is the hardest layer kernel. It computes `A @ B` by tiling: each program computes one BLOCK_M × BLOCK_N output tile by iterating over BLOCK_K chunks of the shared dimension.

**Key code pattern:**
```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(A_ptr + ...)   # (BLOCK_M, BLOCK_K)
    b = tl.load(B_ptr + ...)   # (BLOCK_K, BLOCK_N)
    acc += tl.dot(a, b)
tl.store(C_ptr + ..., acc)
```

**Test:**
```bash
python layers.py
```

### Phase 3d: Attention Kernels

**File:** `attention.py`
**Kernels:** `attention_scores_kernel`, `softmax_inplace_kernel`, `attention_output_kernel`

These implement standard (non-flash) attention as three separate kernels:
1. Compute scores: `Q @ K^T * scale`
2. Apply softmax to scores
3. Compute output: `softmax(scores) @ V`

Study `glm_asr_triton_example/attention.py` carefully — the example shows the grid layout (batch_heads × seq_q) and stride calculations.

**Test:**
```bash
python attention.py
```

### Phase 3e: Rotary Position Embedding

**File:** `rope.py`
**Kernel:** `compute_freqs_kernel`

Computes cos/sin frequency tables for position encoding. The rotation formula is:
```
[x1, x2] → [x1*cos - x2*sin, x2*cos + x1*sin]
```

**Test:**
```bash
python rope.py
```

### Phase 3 Checkpoint: Full Pipeline Test

Once all kernels pass individual tests, run the end-to-end benchmark:
```bash
cd hw1-asr
./benchmark.sh glm_asr_triton_template
```

You must see:
```
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

**If accuracy is wrong:** Use the debugging strategies from GUIDE.md section 8:
- Check for NaN propagation layer by layer
- Test individual kernels with known inputs (e.g., all-ones → RMSNorm should return the weight vector)
- Compare your output shapes against the example

---

## Phase 4: Triton Tutorials 6–7 — Learn Optimization [Days 8–10]

**Goal:** Learn the optimization techniques before applying them.

### Lesson 6: Performance Tuning (30 min)

```bash
cd triton-tutorial/6-performance-tuning && python autotune_benchmark.py
```

**What you learn:** How `triton.autotune` works, how different block sizes affect performance, the trade-off between occupancy (small blocks → more parallelism) and efficiency (large blocks → better memory coalescing but more register pressure).

**Maps to homework:** Optimization 1 — you need to show you tried at least 2–3 configurations. This tutorial gives you the exact pattern:
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["N_CTX"],
)
```

### Lesson 7: Attention (30 min)

```bash
cd triton-tutorial/7-attention && python attention.py
```

**What you learn:** The tiled attention pattern — load a Q block, loop over K/V blocks, compute `tl.dot(q, tl.trans(k))`, apply `tl.exp`, accumulate with `tl.dot(scores, v)`.

**Maps to homework:** This is the stepping stone to FlashAttention. The tutorial uses `tl.exp` without softmax normalization. To get FlashAttention, you add the online softmax trick (running max + running sum) on top of this exact loop structure.

---

## Phase 5: The Three Required Optimizations [Days 10–18]

**Goal:** Beat the baseline. This is where marks come from.

### Optimization 1: FlashAttention (Do This First — Biggest Win, ~3.9x)

**What:** Replace the three separate attention kernels with a single fused kernel that never materializes the full O(n²) attention matrix.

**The algorithm:**
```
For each block of Q (BLOCK_M rows):
    Initialize: m_i = -inf, l_i = 0, acc = 0
    For each block of K,V (BLOCK_N rows):
        scores = Q_block @ K_block^T * scale      # (BLOCK_M, BLOCK_N)
        m_new = max(m_i, row_max(scores))          # update running max
        p = exp(scores - m_new)                     # stable softmax numerator
        l_i = exp(m_i - m_new) * l_i + row_sum(p)  # update running sum
        acc = exp(m_i - m_new) * acc + p @ V_block  # rescale and accumulate
        m_i = m_new
    output = acc / l_i                              # final normalization
```

**Study resources (in recommended order):**
1. FlashAttention paper Algorithm 1: https://arxiv.org/abs/2205.14135
2. Alex Dremov's blog (implements from scratch in Triton): https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/
3. Triton official tutorial `06-fused-attention.py`: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
4. Katherine Olowookere's 3-part series (Feb 2026): https://medium.com/@katherineolowookere/how-i-wrote-flashattention-2-from-scratch-in-custom-triton-kernels-885cac1da357

**Why do this first:** The TA's roadmap showed V2 (FlashAttention) gave 3.9x speedup — by far the largest single optimization. Everything else is incremental on top.

### Optimization 2: Kernel Fusion (At Least One Fused Kernel)

**What:** Combine two or more operations that are currently separate kernels into one kernel, eliminating intermediate memory round-trips.

**Best fusion candidates in the codebase:**

**Option A — Fused SwiGLU (easiest, already scaffolded):**
The MLP does: `gate = silu(x @ W_gate)`, `up = x @ W_up`, `output = gate * up`. Fusing the element-wise part (silu + multiply) into one kernel avoids storing and reloading intermediate tensors.
- Set `MLP.FUSED = True` in `layers.py` to enable the fused path
- Study the `swiglu_fused_kernel` in the template

**Option B — Fused RMSNorm + Residual:**
A transformer block typically does `x = x + residual` then `x = rmsnorm(x)`. Fusing these into one kernel saves one full memory read/write cycle.

**Option C — Fused RoPE:**
The rotary embedding applies sin/cos rotations to Q and K. Fusing this avoids extra load/store cycles.

**Study resources:**
- Liger-Kernel (LinkedIn): https://github.com/linkedin/Liger-Kernel — production Triton implementations of fused RMSNorm, RoPE, SwiGLU
- bassrehab/triton-kernels: https://github.com/bassrehab/triton-kernels — standalone fused kernels with roofline analysis

### Optimization 3: Tile/Block Size Tuning

**What:** Use `triton.autotune` to sweep different configurations and pick the best for the H200.

**What to tune:**
- `BLOCK_M`, `BLOCK_N`, `BLOCK_K` — tile dimensions for matmul and attention
- `num_warps` — parallelism inside each program (typically 4 or 8)
- `num_stages` — memory pipelining depth (typically 2 or 3)

**Approach:**
1. Add `@triton.autotune` decorators with 2–3 configs to your hottest kernels (attention, linear)
2. Run `./benchmark_detailed.sh glm_asr_triton_template` to see per-operator timing
3. Focus tuning on the slowest operators (typically linear/matmul dominates)
4. Document which configs you tried and which won — graders check this

**Good starting configs for H200:**
```python
configs=[
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
]
```

**Do this last** because it polishes everything you've already built.

---

## Phase 6: Profile and Verify [Days 18–20]

### Run Detailed Benchmarks

```bash
# Your implementation
./benchmark_detailed.sh glm_asr_triton_template

# Baseline to compare against
./benchmark_detailed.sh glm_asr_triton_example
```

This shows per-operator timing. Focus on:
- Is your total time faster than the baseline?
- Which operator is the bottleneck? (usually linear/attention)
- Did FlashAttention reduce the attention time significantly?

### Generate Nsight Profile (Optional but Recommended)

```bash
./benchmark_detailed.sh glm_asr_triton_template --nsys
```

The TA strongly recommended Nsight — it shows L1 cache hit rates, kernel execution time, and data loading costs. This is how you find your next optimization opportunity.

### Correctness Check

Always verify after any optimization:
```bash
./benchmark.sh glm_asr_triton_template
# Must still show: Accuracy: 100.0%, Status: PASS
```

---

## Phase 7: Stretch Goals (If Time Permits)

These are optional but can push you past PyTorch performance:

- **FP16 computation with FP32 accumulation** — load data in half precision, accumulate in full precision, write back in half. Significant bandwidth savings.
- **KV Cache** — for the decode phase, cache K and V per layer so each new token only computes its own attention (O(n) instead of O(n²) per step).
- **Fused QKV projection** — instead of three separate linear layers for Q, K, V, do one matmul on the concatenated weight matrix and split the output. Reduces kernel launch overhead 3→1.
- **Pre-allocated buffers** — avoid dynamic memory allocation during generation.

The TA's roadmap showed the full V1→V10 journey achieved 8.6x total speedup:

| Version | Optimization | Speedup |
|---------|-------------|---------|
| V1 | Naive (CuPy, O(n²) attention) | 1.0x |
| V2 | FlashAttention | 3.9x |
| V3 | cuBLAS for linear layers | 2.6x more |
| V4 | TF32 Tensor Cores | 1.2x more |
| V5 | Fused SwiGLU | ~1.02x |
| V6 | Pre-allocated buffers | ~1.02x |
| V8 | KV Cache | 1.25x |
| V10 | Comprehensive (fused QKV, RoPE, decode attention) | 1.38x |
| **Total** | **V1 → V10** | **~8.6x** |

---

## Key Resources

### Must-Read

| Resource | Purpose |
|----------|---------|
| `hw1-asr/GUIDE.md` | Detailed walkthrough, kernel formulas, debugging tips |
| `glm_asr_triton_example/` | Working reference code — diff against template |
| `triton-tutorial/` lessons 1–7 | Triton language fundamentals |
| FlashAttention paper (arxiv 2205.14135) | Algorithm 1 for your attention optimization |

### Recommended Reading

| Resource | Purpose |
|----------|---------|
| Triton official fused attention tutorial | Production FlashAttention v2 in Triton |
| Alex Dremov's FlashAttention blog | Step-by-step Triton implementation |
| Liger-Kernel (github.com/linkedin/Liger-Kernel) | Fused RMSNorm, RoPE, SwiGLU in Triton |
| RoPE paper (arxiv 2104.09864) | Understanding rotary position embeddings |
| FlashAttention-2 paper (arxiv 2307.08691) | Improved parallelism and work partitioning |

### Papers (From Course GUIDE)

| Paper | Relevance |
|-------|-----------|
| "Attention Is All You Need" (Vaswani et al., 2017) | The attention formula you're implementing |
| "RoFormer" (Su et al., 2021) | The RoPE algorithm for rope.py |
| "FlashAttention-2" (Dao, 2023) | The attention optimization algorithm |

---

## Important Details from Lectures

### Assessment Structure (from Homework Announcement)

The homework has TWO parts:
1. **Programming** (start now) — implement and optimise kernels, beat the baseline
2. **Report** (template shared ~3 weeks after announcement) — describe your optimisation journey, submitted to OpenReview as a conference-style paper. You will also peer-review another group's report.

The marking scheme will be shared later. The depth of your optimisations is directly reflected in the marking — more techniques = stronger report = higher marks.

### No Automatic Code Checker

There is no autograder. Your classmate-reviewers will verify your code. This means:
- You have freedom to refactor/rename/fuse beyond the TODO structure
- You MUST provide clear documentation so reviewers can reproduce your results
- Learning to write reproducible documentation is an explicit learning objective

### AI Coding Agents Are Explicitly Allowed

The lecturer said: "You can use AI for coding tools. This is not forbidden at all." Use ChatGPT, Claude, Copilot, etc. But he warned: agents sometimes believe they've reached optimal performance when they haven't. You need GPU architecture knowledge to push further. "Don't believe too much about what the agent told you — use it as a companion, not a teacher."

### Time Estimates

- Strong programming background / PhD: **2–3 hours**
- New to GPU programming: **~30 hours**
- AI agents significantly help with implementation speed

### Start Early — GPU Contention

There are 58 H200s + other GPUs for ~200+ students across multiple courses. The lecturer explicitly warned: "Don't wait until the last 2 weeks. Then there's nothing I can really help you." GPU queues get congested near deadlines.

### Key Insight from Triton Lecture: Why Tiles Matter

Memory coalescing can waste up to **97% of bandwidth** if done wrong. When you write scalar code, the compiler can't predict your access pattern. Tile-based programming enforces that all elements within a tile are consecutive, which lets the compiler prefetch, merge transactions, and optimise automatically. This is the fundamental reason Triton exists and why every kernel must access memory in contiguous patterns.

### Flash Attention — Simplest Mental Model (from Lecturer)

The naive approach: load Q,K → compute intermediate matrices S,P → write S,P to HBM → reload S,P → compute output. Flash Attention: keep S,P in SRAM and never write them to HBM. "As long as we are not running out of SRAM memory, you can avoid the HBM round-trip entirely."

### The "Ultimate" Fusion (Aspirational, Not Required)

Beyond fusing two operations, the lecturer described fusing an entire model layer into one kernel: all intermediate outputs stay in SRAM, only the final result is written to HBM. This is what the TA achieved in V10. Not required for the homework, but demonstrates how far you can push.

---

## Rules & Constraints

- **Must use Triton only** — no PyTorch operators inside kernels (no `torch.matmul` as a substitute)
- **May use examples as reference** — diff and learn from `glm_asr_triton_example/`
- **May refactor and fuse kernels** — not limited to filling existing TODOs
- **Do NOT modify:** `model.py`, `weight_loader.py`, `conv.py`
- **Do NOT run code on the login node** — always use `srun` or `sbatch`
- **Memory:** Request `--mem=16G` to avoid OOM during model weight loading

---

## Quick Command Reference

```bash
# Environment setup
source utils/setup-triton.sh

# Request GPU on teaching cluster
srun -p Teaching -w saxa --gres gpu:1 --mem=16G --pty bash

# Test baseline (should PASS)
./benchmark.sh glm_asr_triton_example

# Test your implementation
./benchmark.sh glm_asr_triton_template

# Detailed profiling
./benchmark_detailed.sh glm_asr_triton_template

# Compare against baseline
./benchmark_detailed.sh glm_asr_triton_example

# Unit tests (from inside template folder)
cd glm_asr_triton_template
python layers.py      # Test all layer kernels
python attention.py   # Test attention kernels
python rope.py        # Test RoPE kernels

# Nsight profiling
./benchmark_detailed.sh glm_asr_triton_template --nsys

# Interactive demo
streamlit run demo.py
```
