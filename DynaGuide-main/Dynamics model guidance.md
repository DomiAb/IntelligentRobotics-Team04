# How DynaGuide uses the dynamics model

What the code really does with the guidance conditions, what alpha is and who sets it, and how much the blocks matter.
Line numbers refer to the code as of 2026-11-10.

## 1. From guidance conditions to a pull on the actions

**Once, before the rollouts** (`calculate_classifier_guidance`, `core/dynaguide.py:34`):

- Each demo in `pos_examples` contributes **one** embedding, from its **last frame**:
  - the `third_person` image and the proprio;
  - the last action repeated over the 16-step chunk (padding `repeat`).
- `model.state_action_embedding(state, action)` turns these into the vectors `e_1 … e_K`.
- With `--max_examples n`, only the first `n` are used.

**At every denoising step of every action chunk** (`guidance(states, actions)`, `core/dynaguide.py:75`):

1. The dynamics model predicts the embedding `p(a)` of the scene **after** executing the current noisy action chunk `a`, from the current observation.
2. It computes the distance to every condition: `d_k = ‖e_k − p(a)‖₂`.
3. It computes a score and its gradient with respect to the actions:

   ```python
   latent_prob_distance = torch.logsumexp(-norm_pairwise_dist / alpha, dim = 0)   # core/dynaguide.py:87
   gradient = scale * torch.autograd.grad(latent_prob_distance, actions)[0]
   ```

4. With `neg_examples`, the same term is computed for the negative conditions and subtracted.

**Where the gradient goes** (`_get_action_trajectory_guidance`, `robomimic/robomimic/algo/diffusion_policy.py:574`):

- It shifts the noise prediction of the diffusion policy:

  ```python
  noise_pred = noise_pred - (1 - alphas_cumprod[k]).sqrt() * scaled_grad
  ```

- This repeats `ss` times per denoising step ("stochastic sampling", `--ss`, default 4).
- Of the resulting chunk, 14 actions are executed.

> `alphas_cumprod` belongs to the noise scheduler of the diffusion policy. It is **not** the alpha of the guidance.

## 2. The pull goes toward the nearest conditions, not their average

The gradient of the score is a weighted sum of one pull per condition:

```
∇_a score = (1/alpha) · Σ_k  w_k · (−∇_a d_k)        with  w_k = softmax_k(−d_k / alpha)
```

- Each `−∇_a d_k` points toward condition `k`.
- The weight `w_k` drops **exponentially** with the distance: a condition that is `Δ` farther away counts `e^(Δ/alpha)` times less.
- The result is a soft nearest-neighbour, or a kernel density with width alpha:
  - **small alpha:** almost only the nearest condition pulls;
  - **large alpha:** the weights become more even. This is still not a pull toward the mean embedding: it is the average of the *directions* toward each condition.
- The robot is never pulled toward a blended "average drawer_open". It is pulled toward the drawer_open conditions that most resemble its own predicted outcome.

**What this means for building conditions:**

- **Coverage matters, not averaging.** For any situation a rollout can be in, some condition of the target behavior should be close. Then that condition dominates and pulls in the right direction.
- A condition that is far from every situation reached in a rollout has almost no effect. It only costs time to embed.

## 3. Alpha

### What it is

The temperature, or kernel width, of the soft-minimum above. The code calls it "softness of guidance consideration". The docstring in `dynaguide_scripts/run_dynaguide.py:177` says it corresponds to **sigma in the paper**. Its unit is the L2 distance in the embedding space: alpha 30 means a condition 30 farther away weighs `e ≈ 2.7` times less.

**Alpha also scales the strength.** The authors left the factor `alpha *` in front of the logsumexp commented out (`core/dynaguide.py:87`), so the gradient carries a factor `1/alpha`:

- the effective strength is roughly `scale / alpha`;
- raising alpha both softens the choice of conditions **and** weakens the guidance;
- when you change alpha, consider changing `scale` with it.

### Who defines it

Alpha is **not learned** and **not stored** in the dynamics model checkpoint or in the experiment config (`env_setup`, `pos_examples`, …). The person starting the run sets it:

| Where | Value | Set by |
|---|---|---|
| `--alpha` in `dynaguide_scripts/run_dynaguide.py` | default **20** if left out | authors (code default) |
| `paper_experiments/*.sh` | chosen by hand per experiment: mostly **30**; 40 for some (e.g. drawer); 15 and 20 rarely | authors |
| `paper_experiments/calvin_ablations.sh` | ablation over 1, 2, 5, 10, 20, 30, 40, 50 | authors |
| our plans (`harvest/articulated.json`, `experiments/plan_first_articulated_*.json`) | plan-wide `"alpha"`, each entry can override it (copied from the paper scripts: drawer 40, door_right 15, others 30) | us; `rollout_tasks.py` passes it on as `--alpha` |

`DG_README.md` uses `--alpha 30` in its examples.

## 4. Measurement: how much do the blocks move the embeddings?

**Data:** `datasets/guidance_conditions/recombined_debug` and `recombined_debug_no_blocks`:
- 8 behaviors × 8 scenes, each with and without blocks;
- several robot poses per file;
- 528 conditions in all.

**Method:**
- Every condition was embedded with `dynamics_model_5900.pth`, exactly as `calculate_classifier_guidance` does it (on the CPU).
- Pairs of conditions that differ in only one thing were compared by L2 distance.

| What differs between the two conditions | pairs | median | 10 % | 90 % |
|---|---|---|---|---|
| only the blocks (same robot, same scene, with vs without blocks)¹ | 264 | 172 | 0 | 215 |
| only one joint plus the light (drawer, door, switch, green light) | 924 | 291 | 159 | 425 |
| another robot pose of the same behavior, same scene | 456 | 240 | 177 | 306 |
| another behavior, same scene | 1596 | 448 | 307 | 498 |

¹ 28 conditions have distance 0 because their scene shows no blocks anyway (blocks hidden or out of view). Without them, the median is **175** (10 %: 124, 90 %: 217).

**Soft-nearest weights.** Each with-blocks condition was used as a query against the other conditions of its behavior:

| alpha | mean weight of the nearest condition | nearest is the same robot pose with blocks in another scene |
|---|---|---|
| 15 | 0.86 | 63 % |
| 30 | 0.65 | 63 % |
| 40 | 0.51 | 63 % |

### Findings

1. **The blocks are a large part of the embedding.** A blocks-only change (about 175) is about 40 % of the gap between two different behaviors (about 450), and in the same range as another robot pose (about 240).
2. **The blocks decide which condition guides.** At alpha 30, a block mismatch of 175 makes a condition about `e^(175/30) ≈ 340` times weaker. If the blocks of a rollout match one condition better than the others, that condition dominates, whatever its robot pose.
3. **The nearest condition dominates at the alphas used.** It carries 51 % to 86 % of the weight. The guidance is close to nearest-neighbour, not an average.

## 5. Consequences for the blocks

- **Randomizing the blocks in the conditions helps, for coverage.** Whatever the blocks in a rollout look like, a condition of the target behavior with similar blocks should exist. Otherwise the block mismatch, not the task, decides which condition pulls.
- **Full coverage is expensive.** Three blocks, each with position and rotation, on several surfaces make a large space. With many conditions, alpha must stay small enough that a near one still dominates.
- **Cheaper alternative: match the blocks of the rollout.**
  - The start scene of a rollout, blocks included, is known after the reset.
  - `dynaguide_scripts/recombine_conditions.py` already places a frozen robot and joints into a given scene, so it can also give every condition the blocks of that start scene.
  - The block mismatch is then close to zero, and the conditions differ only in what the task changes.
  - Cost: rendering and embedding the conditions again for each rollout, a few seconds.
  - Limit: blocks the robot moves during the rollout drift away from the conditions again.
- **First, check the running experiments.** The `blocks`, `no_blocks` and `both` variants of `plan_first_articulated_1/2` compare the current with-blocks and no-blocks condition sets on the same start scenes. That is the first evidence of whether the block mismatch matters for success, not just for distances.
