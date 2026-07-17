# Dual-Line MVP Experiment Log

Portfolio summary: `docs/portfolio_experiment_story_ko.md`

This file remains the detailed appendix containing full experiments, negative
results, and intermediate diagnostics.

Date: 2026-05-29

Note: This file is intentionally written in ASCII English to avoid Korean
encoding corruption in the current patch/terminal path. The project discussion
and interpretation remain Korean-first.

## 1. Goal

This log summarizes the Dual-Line MVP experiments from v0.1 to the current
reobserve-policy experiments.

Main questions:

```text
1. Does ROI/structure/q based MVP behave differently from plain CNN baselines?
2. Does full/global context recover ROI information loss?
3. Can ImageNet dog-breed prior and MVP cat-structure prior complement each other?
4. Does changing the observation view recover errors?
```

---

## 2026-07-08 - v130 roi_align backend and pipeline parallelism direction

Goal:

```text
Confirm that the next speedup should come from pipeline-parallel batching,
not from simply increasing the number of full GPU worker processes.
```

Runtime direction:

```text
ImageLoader / CPU Producer
  - image decode
  - box/pair list generation
  - metadata preparation
        -> queue

GPU Batch Consumer
  - image tensor batch
  - roi_align
  - large CNN forward batch
  - feature/relation computation
        -> queue

Writer
  - npz output
  - manifest/summary output
```

Reason:

```text
The old shard/worker model makes every process run image decode, crop/resize,
and CNN forward. More workers create many small GPU jobs and more CPU
preprocess contention, rather than a larger GPU batch.

Observed resource scaling supports this:
- scan fast path improved only modestly from 4 to 8 workers.
- v130 became slower at 8 workers than at 4 workers.
```

Implemented change:

```text
tools/build_texture_relation_cache_v130.py
  --preprocess_backend {pil, roi_align}

src/dual_line/representation/texture_relation.py
  added roi_align backend
  image tensor + relation boxes -> crop tensor batch -> CNN encode
```

Validation setting:

```text
dataset = AWA cls10 test
limit = 100
pair_budget = 64
pair_policy = object_local
backbone = resnet18
model_ckpt = results/baseline_full_resnet18_awa_cls10/full_texture_head.pt
```

Result:

| backend | time |
|---|---:|
| pil | 30.6s |
| roi_align | 15.0s |
| roi_align_pipeline, gpu_image_batch=8 | 11.2s |
| roi_align_pipeline, gpu_image_batch=16 | 12.2s |
| roi_align_pipeline, area bucket, gpu_image_batch=8 | 11.4s |
| roi_align_pipeline, area bucket, gpu_image_batch=16 | 14.5s |

Shape/finite check:

```text
texture_relation = (100, 16, 16, 17)
relation_embedding = (100, 16, 16, 512)
finite = true
feature_names length = 17
```

Interpretation:

```text
v130 is about 2.0x faster with per-image roi_align, and about 2.7x faster
with the central roi_align_pipeline path on the 100-sample validation.
This supports the producer/consumer direction: CPU producers should prepare
image paths, boxes, pair lists, and metadata, while a central GPU consumer
runs roi_align + CNN forward as large batches.

The best tested setting was gpu_image_batch=8.  Larger image batches are not
automatically faster because padding to the largest image in the group can
increase the image tensor size.

Follow-up implementation:

```text
Added pipeline controls:
  --producer_workers
  --gpu_image_batch
  --bucket_by {none, area}
  --prefetch_batches

Pipeline output is restored to original sample order by sample_idx.
```

Area bucketing was implemented and validated, but it did not improve the
100-sample AWA cls10 test benchmark.  The likely reason is that this subset
does not have enough resolution variance for area bucketing to offset sorting
and prefetch-window overhead.  Keep it as an option for mixed-resolution
external datasets, but use bucket_by=none as the current default fast setting.
```

Representative 1000-sample end-to-end benchmark:

```text
out_root = results/runtime_benchmark_cls10_test1000_central_v130
dataset = AWA cls10 test
n = 1000
scan = npz_only, workers=8
v113 = roi_align
v130 = roi_align_pipeline
v130_shards = 1
v130_pair_budget = 64
v130_pair_policy = object_local
v130_batch_size = 512
v130_producer_workers = 4
v130_gpu_image_batch = 8
v130_bucket_by = none
```

Timing:

| stage | time |
|---|---:|
| scan | 78.5s |
| v113 bbox candidates | 49.4s |
| v117 object support | 29.7s |
| v119 prototype gate | 17.3s |
| v05 tile view relation | 11.2s |
| v121 wave attach | 1.7s |
| v130 texture relation | 63.4s |
| v130 merge | 9.5s |
| v130c texture attach | 7.9s |
| v131 object/texture agreement | 11.9s |
| v153 gate | 5.5s |
| total | 286.0s / 4.77min |

Accuracy:

```text
selected_parent_accuracy = 97.0%
v150_candidate_parent_accuracy = 99.4%
v153_parent_accuracy = 99.6%
v153_fine_accuracy = 97.1%
v153 fixed/broken = 27 / 1
```

---
## 2026-06-25 - v178 local observer tokens

Goal:

```text
Preserve the legacy 4x4 periodic observer topology and build one local
observer token for each tile before adding a DINO-style global token relation.
```

Implementation:

```text
src\dual_line\local_observer_tokens.py
tools\build_local_observer_tokens_v178.py
docs\v178_local_observer_tokens_ko.md
```

Contract:

```text
direction order: NW, N, NE, W, E, SW, S, SE
angle convention: E=0, S=90, W=180, N=270
boundary: periodic topology plus an explicit crosses_boundary feature

tile 1 neighbors: [16, 13, 14, 4, 2, 8, 5, 6]
tile 6 neighbors: [1, 2, 3, 5, 7, 9, 10, 11]
```

Each directed relation compares:

```text
center tile observed toward neighbor
neighbor tile observed back toward center
signed and absolute mutual delta
full-phi correlation and absolute delta
direction encoding and boundary-wrap flag
```

Result:

```text
train       2500 -> local tokens [2500,16,180], relations [2500,16,8,21]
cls10 test  1000 -> local tokens [1000,16,180], relations [1000,16,8,21]
HF90 test    600 -> local tokens [600,16,180], relations [600,16,8,21]
all arrays finite
```

Interpretation:

```text
v178 completes local observation encoding only. It does not yet report a new
classification accuracy. The next stage is a learned 180->64/128 projection
followed by a 16-token global relation layer with observer-derived relation
bias, trained for object continuity and shape consistency.
```

## 2026-06-22 - v168 upper AND audit gate

Goal:

```text
Add an upper AND concept audit gate on top of v153.

This is not a v153 replacement.  It checks whether a v153 transition passes
through ambiguous/risky automatically discovered AND concept relations.
```

Implementation:

```text
tools\run_upper_and_audit_gate_v168.py
```

Inputs:

```text
v153 predictions:
results\v153_structured_transition_gate_awa_cls10_external_hf90_mixed_cap60_noleak\all_predictions.csv

v161 concept graph:
results\v161_concept_and_graph_cls10_train_hf90_noleak
```

Output:

```text
results\v168_upper_and_audit_gate_hf90_cls10_noleak
```

Result:

```text
selected/MVP baseline     95.33%
candidate-only            99.33%  fixed 25 / broken 1
v153                      99.50%  fixed 25 / broken 0
v168 hard guard           98.17%  fixed 17 / broken 0
```

Audit counts:

```text
accept_v153_no_candidate_shift        573
review_candidate_only_sibling_switch    8
accept_v153                             7
review_v153_switch_high_and_risk        7
sibling_supported_switch_audit          4
protected_candidate_only_sibling        1
```

Interpretation:

```text
1. The upper AND audit found useful ambiguity/risk structure, but direct
   hard-blocking is too conservative.
2. Several v153 successful recoveries look risky from the upper AND graph
   alone.  Therefore upper AND risk must not be used as a standalone veto.
3. v168 should be treated as an audit/feature layer:
   recovery evidence and AND risk need to be learned jointly.
4. The next useful direction is not "replace v153", but:
   v153 + AND audit feature + learned recovery/protection split.
```

## 2026-06-22 - v169 evidence completeness audit

Goal:

```text
Add an evidence completeness judgment axis on top of v153/v168.

Question:
  Is the candidate decision supported by diverse object/texture/wave/relation
  evidence, or is it a high-confidence fragment/repeated-region decision?
```

Implementation:

```text
tools\run_evidence_completeness_gate_v169.py
```

Input:

```text
results\v168_upper_and_audit_gate_hf90_cls10_noleak\and_audit_predictions_v168.csv
```

Output:

```text
results\v169_evidence_completeness_gate_hf90_cls10_noleak
```

Result:

```text
selected/MVP baseline     95.33%
candidate-only            99.33%  fixed 25 / broken 1
v153                      99.50%  fixed 25 / broken 0
v169 hard guard           98.17%  fixed 17 / broken 0
```

Action counts:

```text
same_label_complete_evidence             573
v153_switch_completeness_supported        17
v153_switch_completeness_review            9
candidate_rejected_incomplete_or_neutral   1
```

Key diagnostic:

```text
v153 wrong total = 3
v153 wrong flagged by completeness = 1

The v153 wrong cases are not v153 broken cases.
They are mostly selected-wrong -> candidate-wrong cases, so blocking v153
switches cannot repair them.
```

Interpretation:

```text
1. Evidence completeness is a useful diagnostic axis, but it is not safe as a
   standalone veto.
2. Some v153 successful recoveries have low/incomplete local evidence scores.
   This means v153 is using other structured support that the v169 formula
   does not yet recognize.
3. Completeness should be used as an audit feature:
   "this recovery happened despite incomplete evidence"
   not as a hard block.
4. The next gate should learn separate roles:
   recovery evidence
   protection risk
   AND ambiguity
   evidence completeness
```

## 2026-06-22 - v170 independent judgment audit

Goal:

```text
Stop chaining audit outputs as:
  v153 -> v168 -> v169

Instead, read v153 predictions and v161 concept graph directly, then compute
AND and evidence-completeness judgment axes in one independent run.
```

Implementation:

```text
tools\run_independent_judgment_audit_v170.py
docs\v170_independent_judgment_audit_ko.md
```

Artifacts:

```text
results\v170_independent_judgment_audit_hf90_cls10_noleak
```

Result:

```text
selected/MVP baseline   95.33%
candidate-only          99.33%
v153                    99.50%
```

v170 audit labels:

```text
no_transition_same_candidate             573
switch_supported_by_completeness           9
switch_requires_and_review                 7
switch_requires_completeness_review        6
switch_supported_by_and                    4
rejected_candidate_neutral                 1
```

Interpretation:

```text
1. v170 is not another stacked postprocessor.  It is an independent audit table
   over v153.
2. v153 switches split into AND-supported, completeness-supported, and review
   required cases.
3. The remaining v153 wrong cases are not simple broken switches.  They are
   mostly selected-wrong/candidate-wrong cases, so hard blocking cannot repair
   them.
4. The next manageable form should be one learned judgment gate over v170
   features, not a longer chain of postprocessors.
```

## 2026-06-22 - v153auto concept gate

Goal:

```text
Do not stack more modules on top of manual v153.

Instead, test whether the manual parent/fine transition role in v153 can be
replaced by automatic concept graph evidence.
```

Implementation:

```text
tools\run_auto_structured_transition_gate_v153auto.py
docs\v153auto_concept_gate_ko.md
```

Key setup:

```text
v153 manual reference:
results\v153_structured_transition_gate_awa_cls10_external_hf90_mixed_cap60_noleak

auto concept graph:
results\v161_concept_and_graph_cls10_train_hf90_noleak
```

Important training issue:

```text
Using v153 train_transition_rows.csv:
  positive switch rows = 0
  auto gate cannot learn recovery

Using v163c full candidate train rows:
  changed rows          22000
  positive switch rows   4000
  negative switch rows  18000
```

Result:

```text
model                  accuracy   fixed   broken   switches
------------------------------------------------------------
selected/MVP            95.33%      0       0        0
candidate-only          99.33%     25       1       27
manual v153             99.50%     25       0       26
v153auto logreg         99.33%     25       1       27
v153auto HGB            99.00%     23       1       25
```

Interpretation:

```text
1. v153auto can reproduce nearly all of manual v153's recovery behavior
   without using manual parent columns as gate features.
2. The remaining gap is one dangerous transition:
   horse_horse_hf90_0054
   selected horse -> candidate wolf
3. This means auto concept graph recovery is already strong, but singleton
   anchor protection is still weaker than manual v153 parent safety.
4. Next step should not be another stacked layer.  It should be v153auto-b:
   same independent structure, with singleton/cross-concept protection added.
```

## 2026-06-20 - v166 keep-biased transition approval gate

Goal:

```text
v163c/v165c failure was caused by over-switching after opening the full
candidate search space.  v166 reframes the task from "choose the best
candidate" to "approve only candidates strong enough to overturn selected".
```

Implementation:

```text
tools\run_keep_biased_transition_gate_v166.py
```

Training cases:

```text
A. recovery:
   selected wrong, candidate correct -> switch

B. protection:
   selected correct, candidate wrong -> keep

C. ambiguous wrong:
   selected wrong, candidate wrong -> keep / no-benefit

D. safe agreement:
   selected correct, candidate correct -> keep preferred
```

Artifacts:

```text
results\v166_keep_biased_transition_gate_compact_logreg_hf90_cls10_noleak
results\v166b_keep_biased_transition_gate_compact_logreg_hf90_cls10_noleak
results\v166c_keep_biased_transition_gate_compact_logreg_hf90_cls10_noleak
results\v166d_keep_biased_transition_gate_hardneg_logreg_hf90_cls10_noleak
```

Result:

```text
baseline selected/MVP     95.33%
v163c full candidates     94.83%  fixed 12 / broken 15
v166 compact logreg       90.00%  fixed 17 / broken 49
v166d hard-negative       89.83%  fixed 16 / broken 49
```

Weak point:

```text
horse selected/MVP        88.33%
v166/v166d                70.00%
```

Interpretation:

```text
1. v166 correctly changed the problem formulation to transition approval,
   but the current synthetic train distribution still does not reproduce
   eval-time dangerous protection cases.
2. Training contains many protection rows, but hard eval negatives have
   high candidate evidence and are not represented well enough by train.
3. Adding hard-negative weights to protection rows did not fix the issue.
4. The next step should not be more threshold tuning.  It should be
   train-only hard protection generation:
   selected-correct + high-score wrong-candidate + realistic candidate evidence.
```

## 2026-06-20 - v167 train-only hard protection gate

Goal:

```text
Add train-only hard protection rows for the eval failure mode:

selected correct
candidate wrong
candidate evidence appears high
```

Implementation:

```text
tools\run_hard_protection_gate_v167.py
```

Artifacts:

```text
results\v167c_hard_protection_gate_compact_logreg_hf90_cls10_noleak
results\v167d_hard_protection_gate_forced097_hf90_cls10_noleak
```

Results:

```text
model/stage                    accuracy   fixed   broken   switches
--------------------------------------------------------------------
selected/MVP baseline           95.33%      -       -        -
v163c full candidates           94.83%      12      15       37
v166 compact logreg             90.00%      17      49       75
v166d hard-negative             89.83%      16      49       75
v167c hard protection           90.33%      15      45       69
v167d forced conservative       94.17%       1       8        9
```

Horse:

```text
selected/MVP baseline           88.33%
v166/v166d                      70.00%
v167c                           71.67%
v167d forced conservative       83.33%
```

Interpretation:

```text
1. Hard protection augmentation reduced the ability to over-switch only when
   paired with a high forced threshold/margin.
2. v167d reduced broken from 49 to 8 and horse recovered from 70.00% to 83.33%,
   but fixed dropped to 1.  It is too conservative and still below the
   selected/MVP baseline.
3. This means protection can be enforced, but the current single approval
   score does not separate recovery evidence from overtrust risk well enough.
4. Next version should use two scores explicitly:
   recovery_score and protection_risk_score.
   Switch only when recovery_score is high and protection_risk_score is low.
```

## 2026-06-12 - v153 structured transition gate

Goal:

```text
v150 latent expert selector의 후보 선택 능력은 유지하되,
parent를 바꾸는 전환만 Object / Identity / Attribute / Relation 구조 gate로 승인/거절한다.

v150:
  어떤 fine 후보가 가장 강한가?

v153:
  그 후보로 기존 parent 판단을 바꿔도 되는가?
```

Implementation:

```text
tools\run_structured_transition_gate_v153.py
```

Gate structure:

```text
same-parent transition:
  accept

parent-changing transition:
  learned transition gate

gate feature families:
  Object    = support area, tile count, bbox/coverage/diversity
  Identity  = parent/prototype/parent-safe support
  Attribute = confidence, separation, texture relation probability
  Relation  = wave relation, texture-object agreement, conflict/risk
```

Run:

```powershell
.\.venv\Scripts\python.exe -m tools.run_structured_transition_gate_v153 `
  --out_dir results\v153_structured_transition_gate_mvpbaseline_rerun `
  --train_fine_csv results\v131_texture_object_agreement_cls5_train_union\texture_object_agreement_scored_bundles.csv `
  --train_gate_baseline mvp `
  --eval_parent_csv awa_cls5=results\v147_joint_identity_fine_mid_balance\awa_test\parent_predictions.csv `
  --eval_fine_csv awa_cls5=results\v131_texture_object_agreement_cls5_test_union\texture_object_agreement_scored_bundles.csv `
  --eval_parent_csv test1=results\v147_joint_identity_fine_mid_balance\test1\parent_predictions.csv `
  --eval_fine_csv test1=results\v131_texture_object_agreement_test1_union_all_fullpipeline\texture_object_agreement_scored_bundles.csv `
  --eval_parent_csv test2=results\v147_joint_identity_fine_mid_balance\test2\parent_predictions.csv `
  --eval_fine_csv test2=results\v131_texture_object_agreement_test2_4800_union_all_fullpipeline\texture_object_agreement_scored_bundles.csv `
  --eval_parent_csv test4=results\v147_joint_identity_fine_mid_balance\test4\parent_predictions.csv `
  --eval_fine_csv test4=results\v131_texture_object_agreement_test4_awa_catdog_union_all_fullpipeline\texture_object_agreement_scored_bundles.csv `
  --num_experts 4 `
  --expert_model logreg `
  --meta_model logreg `
  --seed 153
```

Result:

```text
dataset   selected_parent   v150 candidate   v153 gated   v150 fixed/broken   v153 fixed/broken
AWA cls5  96.50%            100.00%          100.00%      21 / 0              21 / 0
TEST1     98.35%             97.25%           98.90%       1 / 3               1 / 0
TEST2     95.33%             94.54%           96.98%       82 / 120            82 / 3
TEST4     96.40%             96.60%           97.70%       13 / 11             13 / 0
```

Important note:

```text
AWA cls5 fine accuracy remains 97.17%.
Broad TEST1/TEST2/TEST4 do not have fine labels, so fine accuracy there is not meaningful.
```

Interpretation:

```text
This confirms the recent hypothesis:

  v150 lower/fine evidence is useful,
  but broad external sets need a transition approval gate.

TEST2 shows this most clearly:

  v150 recovered 82 selected-parent mistakes,
  but broke 120 previously correct parent decisions.

v153 keeps the same 82 recoveries and reduces broken cases to 3.

So the issue was not simply internal vs external evidence.
It was gate generalization / transition authority.
```

Conclusion:

```text
v153 is currently the strongest integrated result:
  - keeps v131/v150 fine evidence
  - restores broad parent safety similar to v104a
  - applies Object/Identity/Attribute/Relation structure to the gate itself
```

## 2026-06-12 - v153 transition error analysis

Tool:

```text
tools\analyze_v153_transition_errors.py
```

Purpose:

```text
Before moving to cls10/cls50, split v153 outcomes into:
  fixed cases
  blocked broken cases
  leaked broken cases
  missed parent errors

This makes class expansion comparable by evidence behavior, not only accuracy.
```

Outputs:

```text
results\v153_transition_analysis_test2
results\v153_transition_analysis_test4
results\v153_transition_analysis_awa_cls5
```

Summary:

```text
dataset   fixed   blocked_broken   leaked_broken   missed_parent_errors
AWA cls5  21      0                0               0
TEST2     82      117              3               142
TEST4     13      11               0               23
```

Interpretation:

```text
TEST2 confirms the gate behavior:
  v150 lower evidence had recoverable value,
  but many parent-changing transitions were unsafe.

v153:
  kept 82 fixes,
  blocked 117 would-be broken transitions,
  leaked only 3 broken transitions.

The next step should be class expansion, using the same analysis outputs:
  cls10/cls50 accuracy
  fixed / blocked_broken / leaked_broken / missed_parent_errors
  parent transition pair summary
  feature group summary
```

## v150 corrected TEST1/TEST4 fullpipeline check

목표:

```text
v150 latent expert selector가 망가진 것인지,
아니면 TEST1/TEST4 평가 입력이 v131 학습 입력과 맞지 않아 낮게 나온 것인지 확인한다.
```

확인한 문제:

```text
이전 TEST1/TEST4 v150 평가는 fine candidate feature가 v131 train feature와 맞지 않았다.
TEST1에는 v132 후보만 있고 prototype/wave/texture-relation/objrel feature가 빠져 있었다.
따라서 이전 낮은 TEST1/TEST4 결과는 모델 붕괴가 아니라 평가 배선 오류에 가깝다.
```

새로 만든 TEST1 후보:

```text
v132 object support
-> parent-aware v119 prototype
-> v121 wave relation
-> v130 texture relation
-> v131 texture-object agreement
```

TEST1 최종 후보:

```text
results\v131_texture_object_agreement_test1_union_all_fullpipeline\texture_object_agreement_scored_bundles.csv

train cols = 90
test1 cols = 90
missing = 0
extra = 0
```

새로 만든 TEST4 후보:

```text
results\v130_texture_relation_test4_awa_catdog_union_all\texture_relation_v130.npz
results\tile_view_relation_v05_test4_awa_catdog_all\tile_view_relation_v05.npz
results\v119_class_prototype_gate_test4_awa_catdog_parentaware\prototype_scored_bundles.csv
results\v121_wave_bundle_features_test4_awa_catdog_all\wave_relation_scored_bundles.csv
results\v130_texture_relation_test4_awa_catdog_union_all_fullpipeline\texture_relation_scored_bundles.csv
results\v131_texture_object_agreement_test4_awa_catdog_union_all_fullpipeline\texture_object_agreement_scored_bundles.csv

train cols = 90
test4 cols = 90
missing = 0
extra = 0
```

실행:

```powershell
.\.venv\Scripts\python.exe -m tools.run_latent_expert_selector_v150 `
  --out_dir results\v150_correct_test1_test4_with_v131_fullpipeline `
  --train_fine_csv results\v131_texture_object_agreement_cls5_train_union\texture_object_agreement_scored_bundles.csv `
  --eval_parent_csv awa_cls5=results\v147_joint_identity_fine_mid_balance\awa_test\parent_predictions.csv `
  --eval_fine_csv awa_cls5=results\v131_texture_object_agreement_cls5_test_union\texture_object_agreement_scored_bundles.csv `
  --eval_parent_csv test1=results\v147_joint_identity_fine_mid_balance\test1\parent_predictions.csv `
  --eval_fine_csv test1=results\v131_texture_object_agreement_test1_union_all_fullpipeline\texture_object_agreement_scored_bundles.csv `
  --eval_parent_csv test4=results\v147_joint_identity_fine_mid_balance\test4\parent_predictions.csv `
  --eval_fine_csv test4=results\v131_texture_object_agreement_test4_awa_catdog_union_all_fullpipeline\texture_object_agreement_scored_bundles.csv `
  --num_experts 4 `
  --expert_model logreg `
  --meta_model logreg `
  --seed 150
```

결과:

```text
dataset   n     selected_parent   final_parent   fine
AWA cls5  600   96.50%            100.00%        97.17%
TEST1     182   98.35%             97.25%         broad label이라 fine accuracy는 의미 없음
TEST4     1000  96.40%             96.60%         broad label이라 fine accuracy는 의미 없음
```

해석:

```text
v150은 망가진 것이 아니다.
이전 TEST1/TEST4 결과는 feature pipeline 불일치 때문에 잘못 비교된 가능성이 높다.

다만 corrected v150은 TEST4에서 96.60%로 v147 parent 96.40%보다 소폭 개선되지만,
v104a dynamic broad gate의 TEST4 97.40%에는 아직 못 미친다.

즉 현재 결론은:
  오류 수정: 성공
  v150 최고 성능 회복: 아직 아님
  다음 과제: v104a broad identity 안정성과 v131/v150 fine evidence를 더 잘 결합해야 함
```

## 2026-06-09 - v110 Multiclass Expansion cls5

목표:

```text
2-class에서 만든 gate/boundary 가설이 5-class 확장에서도 유지되는지 확인한다.

실험 질문:
- 클래스 수가 늘면 오답이 랜덤하게 퍼지는가?
- 아니면 유사 class-pair 경계와 gate-hit 영역에 집중되는가?
- MVP v0.2와 frozen ResNet50 full-image baseline의 오답 성격이 어떻게 다른가?
```

Split:

```text
dataset\awa_multiclass_v110\cls5

classes:
  persian+cat
  siamese+cat
  chihuahua
  german+shepherd
  horse

train:
  300/class = 1500

test:
  120/class = 600
```

Models:

```text
MVP v0.2:
  ResNet18 frozen ROI texture
  ResNet18 frozen full texture
  structure/q fusion head

Baseline:
  ResNet50 frozen full-image texture
  new 5-class head
```

Overall result:

```text
MVP v0.2:
  accuracy = 92.17%
  macro class accuracy = 92.17%
  worst class accuracy = 77.50%
  class accuracy range = 19.17%
  wrong = 47 / 600
  gate-hit among wrong = 57.45%

Full ResNet50 baseline:
  accuracy = 95.83%
  macro class accuracy = 95.83%
  worst class accuracy = 93.33%
  class accuracy range = 4.17%
  wrong = 25 / 600
  gate-hit among wrong = 96.00%
```

MVP v0.2 per-class:

```text
chihuahua        95.00%
german+shepherd  96.67%
horse            95.83%
persian+cat      95.83%
siamese+cat      77.50%
```

MVP v0.2 main confusion:

```text
siamese+cat -> persian+cat  20
siamese+cat -> chihuahua     5
persian+cat -> siamese+cat   4
chihuahua -> siamese+cat     3
horse -> chihuahua           3
```

Full ResNet50 baseline per-class:

```text
chihuahua        97.50%
german+shepherd  97.50%
horse            96.67%
persian+cat      94.17%
siamese+cat      93.33%
```

Full ResNet50 baseline main confusion:

```text
persian+cat -> siamese+cat       6
horse -> german+shepherd         3
siamese+cat -> german+shepherd   3
siamese+cat -> persian+cat       3
german+shepherd -> chihuahua     2
```

Interpretation:

```text
첫 cls5 결과는 가설 일부를 지지한다.

MVP v0.2의 오답은 전체 공간에 랜덤하게 퍼지지 않고,
siamese+cat / persian+cat 경계에 강하게 집중된다.

특히 siamese+cat이 worst class로 떨어진 것은
단순 클래스 수 증가보다 유사 고양이 품종 경계가 다시 활성화된 현상으로 보인다.

반면 ResNet50 full baseline은 전체 정확도와 worst-class 안정성이 더 좋다.
하지만 그 오답의 96%가 confidence/margin gate proxy에 걸린다.
즉 더 큰 backbone에서도 오답은 대부분 boundary/gate 영역에 있다.
```

Current conclusion:

```text
cls5 단계에서는 MVP v0.2가 ResNet50 full baseline보다 낮다.

하지만 실패 양상은 프로젝트 가설과 맞다:
  - 유사 class-pair에 오답 집중
  - gate 영역에 오답 집중
  - persistent error 17개 존재

따라서 다음 목표는 단순 정확도 개선보다,
MVP v102/v104 계열의 observation-state summary를 multiclass로 일반화해
siamese/persian 같은 유사 경계에서 worst-class 낙폭을 줄이는 것이다.
```

## 2026-06-09 - v111/v112 Multiclass v102/v104 for cls5

Goal:

```text
v102/v104의 핵심인 observation-state summary와 dynamic sample weight가
2-class cat/dog가 아닌 5-class에서도 작동하는지 확인한다.
```

Implementation:

```text
v111:
  multiclass bbox candidate score cache
  crop 후보마다 prob_<class> 전체 저장
  top1/top2/margin/true-score/rival-score 기반 observation profile 생성
  v102 plus head가 cat/dog 전용 feature 대신 generic state feature를 사용

v112:
  v111 + 기존 v104 dynamic sample weight
```

Teacher / candidate scoring:

```text
candidate scorer:
  frozen ResNet18 full-image head

candidate count:
  44 bbox candidates/sample

test candidate oracle:
  base MVP v0.2 = 92.17%
  candidate oracle = 99.67%

test state decomposition:
  stable_accept = 396
  valid_but_observation_fragile = 157
  recoverable_wrong = 45
  hard_wrong = 2
```

Overall comparison:

```text
MVP v0.2:
  accuracy = 92.17%
  worst-class = 77.50%
  class range = 19.17%
  wrong = 47
  wrong gate-hit = 57.45%

MVP v111:
  accuracy = 93.17%
  worst-class = 81.67%
  class range = 15.00%
  wrong = 41
  wrong gate-hit = 41.46%

MVP v112:
  accuracy = 93.17%
  worst-class = 87.50%
  class range = 9.17%
  wrong = 41
  wrong gate-hit = 65.85%

Full ResNet18 baseline:
  accuracy = 92.50%
  worst-class = 86.67%
  class range = 11.67%

Full ResNet50 baseline:
  accuracy = 95.83%
  worst-class = 93.33%
  class range = 4.17%
```

Per-class key change:

```text
siamese+cat:
  v0.2  = 77.50%
  v111  = 81.67%
  v112  = 87.50%

persian+cat:
  v0.2  = 95.83%
  v111  = 95.00%
  v112  = 90.83%
```

Interpretation:

```text
v111은 평균 정확도를 올린다.
v112는 평균 정확도는 유지하면서 worst-class와 class range를 크게 안정화한다.

즉 v104식 dynamic weighting은 cls5에서도
"평균 최고점"보다 "분포 흔들림 감소"에 더 강하게 작동한다.

다만 v112는 siamese+cat을 살리는 대신 persian+cat을 일부 희생했다.
이는 유사 class-pair 경계에서 weight가 한쪽으로 이동한 결과로 보인다.
```

Current conclusion:

```text
v102/v104 계열은 multiclass에서도 효과가 있다.

가장 중요한 신호:
  v0.2 worst-class 77.50%
  -> v112 worst-class 87.50%

이 결과는 "boundary sample 안정화" 가설과 잘 맞는다.
정확도만 보면 ResNet50 full baseline이 아직 우위지만,
ResNet18 기반 MVP v112는 ResNet18 full baseline보다 평균 정확도와 분포 안정성 모두 약간 우위다.
```

## 2026-06-09 - v113/v114 ImageNet-Only Candidate Scorer for cls5

Question:

```text
cls5도 cat/dog 때처럼 train된 retry scorer 없이,
ImageNet pretrained class score만으로 observation-state 구조가 작동하는가?
```

Condition change:

```text
v111/v112:
  cls5 train split으로 학습한 ResNet18 full-image head를 candidate scorer로 사용

v113/v114:
  별도 cls5 retry scorer 학습 없음
  ImageNet pretrained ResNet18 logits에서 대상 클래스만 직접 사용
```

ImageNet mapping:

```text
persian+cat      -> Persian cat     index 283
siamese+cat      -> Siamese cat     index 284
chihuahua        -> Chihuahua       index 151
german+shepherd  -> German shepherd index 235
horse            -> sorrel          index 339
```

Candidate oracle:

```text
base MVP v0.2:
  92.17%

ImageNet-only candidate oracle:
  99.67%

state decomposition:
  stable_accept = 398
  valid_but_observation_fragile = 155
  recoverable_wrong = 45
  hard_wrong = 2
```

Result:

```text
MVP v0.2:
  accuracy = 92.17%
  worst-class = 77.50%
  range = 19.17%

v113 ImageNet-only plus:
  accuracy = 92.83%
  worst-class = 88.33%
  range = 8.33%

v114 ImageNet-only plus + dynamic weight:
  accuracy = 93.00%
  worst-class = 86.67%
  range = 10.00%

v112 train-head plus + dynamic weight:
  accuracy = 93.17%
  worst-class = 87.50%
  range = 9.17%
```

Interpretation:

```text
고양이/강아지 실험에 더 가까운 조건에서도 상승이 남았다.

즉 cls5 결과가 단순히 train split으로 학습한 retry scorer 덕분만은 아니다.
ImageNet에 해당 class가 직접 존재하는 경우,
pretrained logits만으로도 관측 후보 안에 정답 view가 거의 존재한다.

v113은 평균 상승은 작지만 worst-class와 range 안정화가 가장 강하다.
v114는 평균 정확도가 조금 더 높지만 worst-class는 v113보다 낮다.
```

Current conclusion:

```text
조건 엄격함:
  cat/dog original >= v113/v114 cls5 > v111/v112 cls5

v113/v114는 cat/dog 방식에 가까운 검증이다.
이 조건에서도 v0.2 대비:
  accuracy +0.67~0.83%
  worst-class +9.17~10.83%
  range -9.17~-10.83%

따라서 multiclass에서도 observation-state 안정화가 작동한다는 근거가 강화됐다.
```

## 79. v106 Candidate Selector 진단 실험

목적:

```text
v105 라우터가 candidate_select로 보낸 샘플에서,
후보 view/crop 중 어떤 것을 믿을지 고르는 selector를 만든다.
```

추가 도구:

```text
tools/train_eval_candidate_selector_v106.py
```

조건:

```text
train:
  train_1000의 bbox_candidate_scores + v099 consistency

model:
  HistGradientBoostingClassifier

train rows:
  12000개 후보 row sampling

test:
  TEST1 / TEST2 / TEST4

중요:
  테스트셋별 튜닝 없음
  v105 router가 candidate_select라고 보낸 샘플만 candidate scoring
```

v0.2 base 기준 candidate route 결과:

```text
TEST1:
  base_accuracy = 99.45%
  candidate_route_accuracy = 99.45%
  switch_count = 42
  selected_candidate_accuracy = 100.00%

TEST2:
  base_accuracy = 96.00%
  candidate_route_accuracy = 98.25%
  switch_count = 1185
  selected_candidate_accuracy = 97.30%

TEST4:
  base_accuracy = 94.60%
  candidate_route_accuracy = 96.10%
  switch_count = 288
  selected_candidate_accuracy = 95.14%
```

해석:

```text
candidate selector 자체는 의미가 있다.
v0.2 base 기준으로 TEST2와 TEST4를 크게 올린다.
```

하지만 v102/v104a 위에 단순 교체로 얹은 결과:

```text
v102 + v106 candidate route:
  TEST1 = 98.90% -> 98.90%
  TEST2 = 97.77% -> 97.90%
    fixed = 21
    broken = 15
  TEST4 = 97.60% -> 97.20%
    fixed = 4
    broken = 8

v104a + v106 candidate route:
  TEST1 = 98.90% -> 98.90%
  TEST2 = 98.10% -> 98.10%
    fixed = 14
    broken = 14
  TEST4 = 97.40% -> 96.90%
    fixed = 2
    broken = 7
```

결론:

```text
candidate_select 경로는 필요하다.
하지만 v102/v104a 위에서 후보 선택 결과를 무조건 적용하면 안 된다.
```

핵심 병목:

```text
candidate selector가 정답 후보를 꽤 잘 찾더라도,
현재 base가 이미 더 안전한 경우가 있다.

따라서 다음 단계는:
  candidate selector 자체
보다
  candidate accept gate
가 필요하다.
```

다음 방향:

```text
v107:
  candidate accept gate

입력:
  base confidence
  v102/v104a confidence
  selected candidate score
  selected candidate confidence
  route probability
  candidate/base label disagreement
  consistency/support
  class-pair confusion feature

출력:
  keep base
  accept selected candidate

목표:
  TEST2의 fixed는 유지하고,
  TEST4의 broken을 줄인다.
```

## 78. v105 Observation Router 진단 실험

목적:

```text
TEST1 / TEST2 / TEST4를 각각 따로 맞추지 않고,
샘플의 관측 상태를 보고 자동으로 처리 경로를 고르는 라우터를 만든다.
```

라우터 경로:

```text
keep_base:
  base 판단을 유지

sample_reweight:
  샘플 전체의 학습 가중치를 조정

candidate_select:
  후보 view/crop 중 어떤 것을 믿을지 선택해야 함

hard_case:
  현재 후보/표현만으로는 부족할 가능성이 큼
```

추가 도구:

```text
tools/train_eval_observation_router_v105.py
```

조건:

```text
train:
  train_1000의 v100 base_observation_state만 사용

model:
  HistGradientBoostingClassifier

input:
  v100 runtime observation features

test:
  TEST1 / TEST2 / TEST4
```

라우터 결과:

```text
TEST1:
  route_accuracy = 99.45%
  pred_counts:
    keep_base = 103
    sample_reweight = 37
    candidate_select = 42
    hard_case = 0

TEST2:
  route_accuracy = 98.54%
  pred_counts:
    keep_base = 2937
    sample_reweight = 678
    candidate_select = 1185
    hard_case = 0

TEST4:
  route_accuracy = 97.40%
  pred_counts:
    keep_base = 590
    sample_reweight = 122
    candidate_select = 288
    hard_case = 0
```

중요한 한계:

```text
train_1000 validation split에는 hard_case target이 거의 없어서,
현재 라우터는 hard_case를 예측하지 못한다.
```

v104a 변화와 라우터를 붙여본 결과:

```text
TEST2:
  v104a fixed_by_v104 = 21
  route_pred:
    sample_reweight = 12
    candidate_select = 8
    keep_base = 1

  v104a broken_by_v104 = 5
  route_pred:
    sample_reweight = 2
    candidate_select = 2
    keep_base = 1

TEST4:
  v104a fixed_by_v104 = 4
  route_pred:
    candidate_select = 2
    sample_reweight = 1
    keep_base = 1

  v104a broken_by_v104 = 6
  route_target:
    candidate_select = 5
    hard_case = 1

  v104a broken_by_v104 route_pred:
    sample_reweight = 4
    candidate_select = 1
    keep_base = 1
```

해석:

```text
v105 라우터는 큰 방향을 잘 잡는다.
stable 영역은 대부분 keep_base로 보내고,
fragile/recoverable 영역은 sample_reweight 또는 candidate_select로 보낸다.
```

하지만 TEST4에서 중요한 실패가 보인다:

```text
v104a가 깨뜨린 TEST4 샘플 6개 중 대부분은
teacher target상 candidate_select 문제인데,
라우터는 sample_reweight로 보내는 경향이 있다.
```

즉 TEST4 손실은:

```text
sample-level dynamic weight 부족
```

보다는:

```text
candidate_select를 더 정확히 인식하고,
그 후보 중 안전한 view를 고르는 능력 부족
```

에 가깝다.

현재 결론:

```text
v105는 테스트셋별 보정 없이 샘플 상태를 자동 라우팅하는 출발점으로 의미 있다.

다음 핵심은 hard_case 예측보다 먼저 candidate_select 경로를 강화하는 것이다.
```

다음 방향:

```text
v106:
  candidate_select 전용 selector / scorer

입력:
  candidate view feature
  bbox 위치/면적
  consistency/support/conflict
  base prediction과 후보 prediction 충돌
  class-pair confusion feature

목표:
  sample_reweight로 처리할 샘플과
  candidate_select로 처리할 샘플을 더 잘 구분한다.
```

## 74. 더 강한 비교군: Full-image ResNet50 frozen feature + 2-class head

목적:

```text
ResNet18 full baseline이 약해서 v102가 좋아 보인 것인지,
더 강한 ImageNet backbone에서도 관측 구조의 이득이 유지되는지 확인한다.
```

조건:

```text
backbone:
  ResNet50 ImageNet default weights

feature:
  full image global feature only

train:
  train_1000

test:
  TEST1 / TEST2 4800 / TEST4 AWA catdog 1000

fine-tuning:
  backbone fine-tuning 없음
  새 2-class head만 학습
```

결과:

```text
Full ResNet50 baseline:

TEST1:
  accuracy = 99.45%
  macro_f1 = 99.45%
  confusion = [[90, 1], [0, 91]]

TEST2:
  accuracy = 96.31%
  macro_f1 = 96.31%
  confusion = [[2248, 152], [25, 2375]]

TEST4:
  accuracy = 97.20%
  macro_f1 = 97.20%
  confusion = [[475, 25], [3, 497]]

mean:
  97.99%
```

v102와 비교:

```text
MVP v0.2 plus v102:

TEST1:
  98.90%

TEST2:
  97.77%

TEST4:
  97.60%

mean:
  98.09%
```

해석:

```text
ResNet50 full baseline은 ResNet18 full baseline보다 훨씬 강하다.
특히 TEST4에서 92.60% -> 97.20%로 크게 오른다.

그래도 TEST2와 TEST4에서는 v102가 여전히 앞선다.
TEST1에서는 ResNet50 baseline이 v102보다 좋다.
```

중요한 의미:

```text
v102의 이득은 단순히 약한 ResNet18 baseline을 이긴 결과만은 아니다.
더 강한 ResNet50 full-image baseline과 비교해도 hard/distribution-shift set에서 관측 요약의 이득이 남아 있다.
```

다만 결론은 더 보수적으로 잡아야 한다:

```text
v102가 모든 조건에서 강한 backbone을 이긴다 X

v102는 TEST1 최상 성능을 약간 내주는 대신,
TEST2/TEST4 같은 분포 변화에서 더 균일한 성능을 낸다 O
```

현재 관찰:

```text
ResNet50 full baseline:
  TEST1 매우 강함
  TEST2에서 cat -> dog 오답이 크게 증가
  TEST4에서도 품종/질감 변화에 상당히 잘 버팀

v102:
  TEST1은 약간 손실
  TEST2/TEST4에서 더 균형적인 cat/dog 혼동을 보임
  평균은 소폭 우세
```

현재 결론:

```text
비교군을 ResNet50으로 올려도 v102는 평균과 hard-set 안정성에서 경쟁력이 있다.
이제 다음 비교는 EfficientNet-B0 또는 CLIP feature 같은 더 범용적인 표현으로 올리는 것이 좋다.
```

## 75. v103a/v103b: 관측 상태 가중치 조절 실험

목적:

```text
TEST1 복구가 아니라,
전체 평균 정확도 상승과 테스트셋 간 편차 감소를 목표로 한다.

ResNet50으로 체급을 올리지 않고,
기존 ResNet18 texture feature + v102 구조 안에서
관측 상태별 학습 가중치를 조절해 볼 수 있는지 확인한다.
```

스크립트 변경:

```text
tools/train_eval_mvp02_plus_v102.py

기존 하드코딩 가중치:
  stable_accept = 1.20
  valid_but_observation_fragile = 0.35
  recoverable_wrong = 1.50
  hard_wrong = 0.50

CLI 인자 추가:
  --stable_weight
  --fragile_weight
  --recoverable_weight
  --hard_weight

기본값은 기존 v102와 동일하게 유지.
```

실험 A: v103a balanced/aggressive

```text
obs_loss_weight = 0.25
stable_weight = 1.00
fragile_weight = 0.75
recoverable_weight = 2.00
hard_weight = 0.60
seed = 103
```

결과:

```text
TEST1 = 98.35%
TEST2 = 97.25%
TEST4 = 97.60%

mean = 97.73%
worst = 97.25%
range = 1.10%
std = 0.46%
```

해석:

```text
편차는 v102보다 줄었지만 평균이 내려갔다.
recoverable/fragile 쪽을 너무 강하게 밀면 TEST1/TEST2 손실이 생긴다.
```

실험 B: v103b mild

```text
obs_loss_weight = 0.30
stable_weight = 1.10
fragile_weight = 0.50
recoverable_weight = 1.75
hard_weight = 0.55
seed = 102
```

결과:

```text
TEST1 = 98.90%
TEST2 = 97.69%
TEST4 = 97.20%

mean = 97.93%
worst = 97.20%
range = 1.70%
std = 0.72%
```

해석:

```text
v102보다 평균, worst, TEST4가 모두 낮다.
단순한 가중치 조절만으로 v102를 넘기기는 어렵다.
```

현재 비교표:

```text
model             TEST1   TEST2   TEST4   mean   worst  range  std
MVP v0.2          99.45   96.00   94.60   96.68  94.60  4.85   2.04
v102              98.90   97.77   97.60   98.09  97.60  1.30   0.58
v103a balanced    98.35   97.25   97.60   97.73  97.25  1.10   0.46
v103b mild        98.90   97.69   97.20   97.93  97.20  1.70   0.72
Full ResNet50     99.45   96.31   97.20   97.65  96.31  3.14   1.32
```

결론:

```text
v102가 현재 가장 좋은 균형점이다.

v103a는 편차 감소에는 성공했지만 평균 정확도가 내려갔다.
v103b는 v102에 가까운 mild 조정이지만 v102를 넘지 못했다.
```

중요한 학습:

```text
전체 정확도 상승은 단순히 class/sample weight를 조절하는 문제가 아니다.

다음 개선은 loss weight 조절보다
view 선택/관측 안정성 feature를 class head에 더 직접적으로 연결하거나,
base 후보군을 single fusion이 아니라 candidate selector 형태로 다루는 쪽이 더 유망하다.
```

## 76. v104a: Dynamic sample weight 첫 실험

목적:

```text
v102의 고정 상태별 가중치를 그대로 고집하지 않고,
샘플별 관측 상태 feature로 연속적인 sample weight를 만든다.

목표는 특정 TEST1 복구가 아니라:
  전체 평균 정확도 상승
  hard distribution 안정성 유지
  클래스 확장에 필요한 동적 weighting 가능성 확인
```

추가 도구:

```text
tools/build_dynamic_sample_weights_v104.py
```

입력:

```text
results/v100_base_observation_state_train_1000/base_observation_state.csv
```

사용한 신호:

```text
observation_support_score
observation_conflict_score
observation_stability_score
base_partial_risk_score
correct_view_ratio
wrong_high_conf_ratio
candidate_oracle_hit
base_state_target
```

v104a weight 생성 조건:

```text
min_weight = 0.20
max_weight = 2.50
risk_strength = 0.30
oracle_strength = 0.50
stable_strength = 0.20
```

생성된 train_1000 weight 요약:

```text
overall:
  mean = 1.022
  std = 0.404

stable_accept:
  n = 534
  mean = 1.376
  min = 1.195
  max = 1.400

valid_but_observation_fragile:
  n = 460
  mean = 0.598
  min = 0.252
  max = 0.745

recoverable_wrong:
  n = 6
  mean = 2.044
  min = 1.853
  max = 2.082
```

학습:

```text
v102 구조 유지
ResNet18 texture feature 유지
state별 고정 class weight 대신 dynamic_sample_weight 사용
obs_loss_weight = 0.35
seed = 104
```

결과:

```text
v104a dynamic:

TEST1:
  98.90%
  confusion = [[89, 2], [0, 91]]

TEST2:
  98.10%
  confusion = [[2354, 46], [45, 2355]]

TEST4:
  97.40%
  confusion = [[483, 17], [9, 491]]

mean:
  98.14%

worst:
  97.40%

range:
  1.50%

std:
  0.61%
```

비교:

```text
model             TEST1   TEST2   TEST4   mean   worst  range  std
MVP v0.2          99.45   96.00   94.60   96.68  94.60  4.85   2.04
v102 fixed        98.90   97.77   97.60   98.09  97.60  1.30   0.58
v104a dynamic     98.90   98.10   97.40   98.14  97.40  1.50   0.61
Full ResNet50     99.45   96.31   97.20   97.65  96.31  3.14   1.32
```

해석:

```text
v104a는 첫 dynamic weight 실험인데도 평균 정확도가 v102보다 소폭 상승했다.
TEST2는 97.77% -> 98.10%로 상승했다.
TEST4는 97.60% -> 97.40%로 0.20% 하락했다.
TEST1은 유지됐다.
```

중요한 의미:

```text
v103a/v103b처럼 상태별 고정 가중치를 손으로 바꾸는 방식보다,
관측 feature 기반 dynamic weight가 더 유망하다.

큰 backbone으로 체급을 올리지 않고도 평균 정확도가 소폭 상승했으므로,
클래스 확장을 위한 dynamic weighting 방향은 계속 볼 가치가 있다.
```

주의:

```text
v104a는 아직 learned dynamic weight가 아니다.
feature 기반 수식으로 만든 teacher weight다.

다음 단계는 이 수식을 직접 고정하지 않고,
weight predictor 또는 candidate selector가 학습하게 만드는 것이다.
```

다음 방향:

```text
1. v104b:
   TEST4 손실을 줄이도록 risk/oracle/stable strength를 소폭 조정한다.

2. v105:
   sample-level weight를 넘어 candidate-level/view-level dynamic weight로 확장한다.

3. multi-class 확장:
   class-pair confusion difficulty를 dynamic weight 입력에 추가한다.
```

## 77. v104b: dynamic weight 완화 실험

목적:

```text
v104a에서 TEST2는 상승했지만 TEST4가 0.20% 하락했다.
따라서 dynamic weight를 조금 부드럽게 조정해 TEST4 손실을 줄일 수 있는지 확인한다.
```

v104b weight 조건:

```text
min_weight = 0.20
max_weight = 2.50
risk_strength = 0.20
oracle_strength = 0.55
stable_strength = 0.12
```

v104a 대비 weight 분포 변화:

```text
stable_accept:
  1.376 -> 1.305

valid_but_observation_fragile:
  0.598 -> 0.655

recoverable_wrong:
  2.044 -> 2.061

overall std:
  0.404 -> 0.345
```

결과:

```text
v104b dynamic:

TEST1:
  98.90%
  confusion = [[89, 2], [0, 91]]

TEST2:
  98.08%
  confusion = [[2354, 46], [46, 2354]]

TEST4:
  97.30%
  confusion = [[482, 18], [9, 491]]

mean:
  98.09%

worst:
  97.30%

range:
  1.60%

std:
  0.65%
```

비교:

```text
model             TEST1   TEST2   TEST4   mean   worst  range  std
v102 fixed        98.90   97.77   97.60   98.09  97.60  1.30   0.58
v104a dynamic     98.90   98.10   97.40   98.14  97.40  1.50   0.61
v104b dynamic     98.90   98.08   97.30   98.09  97.30  1.60   0.65
```

해석:

```text
v104b는 TEST4 회복에 실패했다.
TEST2 상승은 거의 유지했지만 TEST4는 v104a보다 더 낮아졌다.
```

현재 결론:

```text
dynamic weight 계열에서는 v104a가 현재 대표 결과다.
v104b처럼 weight 분포를 부드럽게 만드는 것만으로는 TEST4 손실이 줄지 않는다.
```

다음 방향:

```text
TEST4 손실은 sample-level weight 수식 조절보다
view/candidate-level 선택 문제일 가능성이 크다.

따라서 다음 개선은:
  1. candidate-level dynamic weight
  2. class head가 observation feature를 더 직접적으로 쓰는 구조
  3. class-pair confusion feature
중 하나로 넘어가는 것이 좋다.
```

## 2. Datasets

### Train

```text
dataset/train/cat
dataset/train/dog
train1000 = cat 500 + dog 500
```

Source:

```text
Kaggle Dogs vs Cats / Microsoft Dogs vs Cats style
Example names: cat.4001.jpg, dog.4001.jpg
```

### Test

```text
dataset/test/cat
dataset/test/dog
test_all = cat 91 + dog 91 = 182
```

This is an external cat/dog test set, separate from the train source.

### Test2

```text
dataset/test2/images
dataset/test2/cat
dataset/test2/dog
```

Source/style:

```text
Oxford-IIIT Pet style
12 cat breeds, 25 dog breeds
```

After split:

```text
cat 2400
dog 4990
```

Evaluated balanced subsets:

```text
test2_1000 = cat 500 + dog 500
test2_all_balanced_4800 = cat 2400 + dog 2400
```

## 3. Common Setup

Visual backbone:

```text
ResNet18 ImageNet pretrained
```

Important:

```text
The ResNet18 backbone was not fine-tuned.
Only small heads/calibrators were trained.
```

Runtime:

```text
CUDA / NVIDIA GeForce RTX 3070
```

## 4. Baselines

### ImageNet Baseline

Method:

```text
full image -> ResNet18 ImageNet 1000-class
cat_score = sum ImageNet cat classes 281..285
dog_score = sum ImageNet dog classes 151..268
```

Results:

| Dataset | Accuracy | Macro F1 | Confusion Matrix |
|---|---:|---:|---|
| test_all 182 | 87.91% | 87.73% | [[69, 22], [0, 91]] |
| test2_1000 | 95.30% | 95.29% | [[453, 47], [0, 500]] |
| test2_all_balanced_4800 | 95.00% | 94.99% | [[2160, 240], [0, 2400]] |

Interpretation:

```text
ImageNet has a strong dog-breed prior.
It almost never misses dogs, but unusual cats are often absorbed into dog breeds.
```

### Full-image CNN Linear Baseline

Method:

```text
full-image ResNet18 feature
+ Logistic Regression / linear cat-dog classifier
```

Result:

| Dataset | Accuracy | Macro F1 | Confusion Matrix |
|---|---:|---:|---|
| test_all 182 | 97.80% | 97.80% | [[89, 2], [2, 89]] |

Interpretation:

```text
A small classifier over frozen ImageNet features is already strong.
This baseline does not use ROI, structure vector, or q vector.
```

## 5. MVP v0.1: ROI-only Fusion

Architecture:

```text
Observer Scan
-> bootstrap ROI
-> ROI crop
-> ResNet18 texture feature
+ structure vector
+ q vector
-> Fusion Head
-> cat/dog
```

Characteristics:

```text
No full-image branch.
The model relies strongly on a single bootstrap ROI.
```

Result:

| Dataset | Accuracy | Macro F1 | Confusion Matrix |
|---|---:|---:|---|
| test_all 182 | 95.60% | 95.60% | [[84, 7], [1, 90]] |

Observed issue:

```text
Single ROI can lose global identity.
If ROI sees mostly body/fur/pattern, cat/dog can flip.
```

Representative errors:

```text
cat_20, cat_47, cat_78 -> dog
dog_31 -> cat
```

## 6. MVP v0.2: ROI + Global Gate

Architecture:

```text
ROI texture branch
+ full-image texture branch
+ structure vector
+ q vector
+ q-conditioned ROI gate
-> Fusion Head
```

Gate meaning:

```text
gate ~= 1.0 -> trust ROI branch more
gate ~= 0.0 -> trust full-image branch more
```

Reason for change:

```text
v0.1 showed ROI information loss.
v0.2 adds full/global identity signal while preserving ROI/structure features.
```

Results:

| Dataset | Accuracy | Macro F1 | Confusion Matrix | Mean ROI Gate |
|---|---:|---:|---|---:|
| test_all 182 | 99.45% | 99.45% | [[90, 1], [0, 91]] | 0.377 |
| test2_1000 | 96.00% | 95.99% | [[498, 2], [38, 462]] | 0.370 |
| test2_all_balanced_4800 | 96.00% | 96.00% | [[2364, 36], [156, 2244]] | 0.370 |

Interpretation:

```text
ROI + Global Gate strongly reduced v0.1 information loss.
On test_all, errors dropped from 8 to 1.
```

Behavior on test2:

```text
MVP v0.2 defends cat identity/morphology well.
It tends to pull cat-like dogs into cat.
Examples: Japanese Chin, Pomeranian, Chihuahua, Samoyed.
```

## 7. MVP v0.3: Meta Calibrator

Architecture:

```text
MVP v0.2 output
+ ImageNet cat/dog score
+ ImageNet top1 prior
+ ROI/q quality features
-> Logistic Regression meta head
-> final cat/dog
```

Purpose:

```text
MVP is strong on cat identity / morphology.
ImageNet is strong on dog-breed prior.
v0.3 tests whether the two experts can complement each other.
```

Results:

| Dataset | Accuracy | Macro F1 | Confusion Matrix |
|---|---:|---:|---|
| test_all 182 | 98.90% | 98.90% | [[89, 2], [0, 91]] |
| test2_1000 | 97.00% | 97.00% | [[498, 2], [28, 472]] |
| test2_all_balanced_4800 | 96.79% | 96.79% | [[2361, 39], [115, 2285]] |

Effect:

```text
test2_1000:
dog->cat errors reduced from 38 to 28.

test2_all_balanced_4800:
dog->cat errors reduced from 156 to 115.
cat->dog errors increased from 36 to 39.
```

Caution:

```text
v0.3 can over-trust ImageNet dog prior.
On test_all, it flipped cat_26 from correct cat to dog.
```

## 8. Multi-view Reobserve Experiment

Purpose:

```text
Check whether errors are truly unsolvable,
or whether another observation view can recover them.
```

Generated views:

```text
roi
expanded_1_5
expanded_2_0
upper_context
body_context
center_70
full
```

### test2_1000 selected 40 samples

Result:

```text
any_view_correct = 39 / 40 = 97.5%
```

View-level result:

| View | Accuracy |
|---|---:|
| upper_context | 97.5% |
| center_70 | 95.0% |
| body_context | 95.0% |
| expanded_1_5 | 95.0% |
| expanded_2_0 | 95.0% |
| full | 95.0% |
| roi | 95.0% |

Interpretation:

```text
On the small subset, upper/head context looked very strong.
```

### test2_all_balanced_4800 reobserve evaluation

View-level result:

| View | Accuracy | Mean Confidence |
|---|---:|---:|
| full | 93.30% | 0.936 |
| expanded_2_0 | 92.74% | 0.926 |
| center_70 | 91.62% | 0.937 |
| expanded_1_5 | 91.62% | 0.906 |
| body_context | 91.06% | 0.913 |
| upper_context | 91.06% | 0.849 |
| roi | 90.50% | 0.879 |

Interpretation:

```text
On the larger set, full and expanded_2_0 are more stable than upper_context.
The issue is not only missing head/ear evidence.
The broader issue is that the current ROI may not represent object identity.
```

## 9. Rule Reobserve Policy v0

Policy:

```text
none -> v0.3 or MVP
trust_mvp_structure -> MVP
trust_imagenet_prior -> ImageNet
reobserve_body_context -> upper_context
reobserve_expand_quality -> upper_context
```

test2_1000 result:

```text
Accuracy = 99.9%
Confusion Matrix = [[499, 1], [0, 500]]
```

Caution:

```text
This used pseudo action labels derived from test2 analysis.
It must not be claimed as final external validation performance.
Its meaning is: if an appropriate reobserve action can be chosen,
most errors are recoverable.
```

## 10. Reobserve Planner Pseudo-label Distribution

### test2_1000

```text
none: 915
trust_mvp_structure: 45
trust_imagenet_prior: 12
reobserve_body_context: 26
reobserve_expand_quality: 2
needs_reobserve: 28
```

### test2_all_balanced_4800

```text
none: 4393
trust_mvp_structure: 215
trust_imagenet_prior: 60
reobserve_body_context: 107
reobserve_head_exception: 13
reobserve_expand_quality: 12
needs_reobserve: 132
```

Quality-failure pattern:

```text
reobserve_expand_quality:
low roi_quality
low geometry_quality
high border_touch_ratio
```

This supports ROI update / reobserve policy work.

## 11. Main Conclusions

```text
1. v0.1 ROI-only has single-ROI information loss.
2. v0.2 ROI + Global Gate strongly fixes that issue.
3. ImageNet and MVP have different error directions.
   - ImageNet: dog-breed prior expert.
   - MVP: cat identity / morphology expert.
4. v0.3 meta calibrator recovers part of cat-like dog failures.
5. Multi-view reobserve shows that observation view choice affects accuracy.
```

Core interpretation:

```text
Dual-Line is not just an ROI classifier.
Its important direction is detecting observation failure
and switching to a better view.
```

## 12. Next Direction

Candidate v0.4:

```text
ROI + Expanded ROI + Full tri-view fusion
```

Alternative/complement:

```text
Failure-aware Reobserve Planner
-> decide whether current ROI is sufficient
-> choose full / expanded_2_0 / upper_context / body_context as needed
```

Priority:

```text
1. ROI update / expanded context branch.
2. Refine reobserve failure types.
3. Re-check rule policy on test2_all_balanced_4800.
4. Connect this to Chapter 11 Observation Flow Tracking.
```

## 13. Cautions

```text
1. test2-based reobserve policy results are pseudo-label analysis,
   not final external validation.

2. v0.3 is currently binary cat/dog-specific.
   Multiclass expansion needs class-vector / top-k / expert reliability design.

3. ImageNet has prior-density imbalance:
   about 118 dog breed classes vs about 5 cat classes.
   This should be treated as an expert property, not simply hand-corrected.

4. Full / expanded views being strong implies bootstrap ROI is still a seed,
   not the final ROI. ROI should be updated through observation flow.
```

## 14. 3-class Expansion: cat / dog / horse

Purpose:

```text
Test whether adding a new class immediately contaminates the existing cat/dog
space, or whether the model can form a stable new class axis.
```

Horse split:

```text
train horse:
dataset/train/horse
horse_10001.jpg .. horse_11000.jpg = 1000 images

test horse:
dataset/test/horse
horse_11001.jpg .. horse_11645.jpg
```

Training set:

```text
cat 1000
dog 1000
horse 1000
total 3000
```

Test set:

```text
cat 91
dog 91
horse 91
total 273
```

### MVP v0.2 3-class

Architecture:

```text
ROI ResNet18 frozen feature
+ Full-image ResNet18 frozen feature
+ structure/q features
-> 3-class Fusion Head
```

Result:

| Dataset | Accuracy | Macro F1 | Confusion Matrix |
|---|---:|---:|---|
| test_3cls_273 | 99.3% | 99.3% | [[90, 1, 0], [0, 91, 0], [0, 1, 90]] |

Errors:

```text
cat_44 -> dog, confidence 0.994
horse_11329 -> dog, confidence 0.530
```

Interpretation:

```text
Adding horse did not collapse the cat/dog boundary.
The model formed a stable horse axis while keeping dog perfect on this test.
```

### ImageNet original-head 3-answer baseline

Condition:

```text
Same test_3cls_273 images.
ResNet18 ImageNet pretrained original 1000-class head.
Restrict final answers to cat / dog / horse groups.
No task-specific head training.
```

Horse group:

```text
ImageNet class 339 = sorrel
horse cart was excluded because it is an object/context class.
```

Results:

| Method | Accuracy | Macro F1 | Confusion Matrix |
|---|---:|---:|---|
| ImageNet group sum | 82.4% | 82.9% | [[69, 22, 0], [0, 91, 0], [0, 26, 65]] |
| ImageNet group max | 86.1% | 86.5% | [[74, 17, 0], [0, 91, 0], [0, 21, 70]] |

Interpretation:

```text
ImageNet original head has strong dog-class density.
Horse is represented narrowly by sorrel, so horse images are often absorbed
into dog-like classes or other related ImageNet classes.
```

## 15. Strong Baseline B: Full-image ResNet Feature + New Head

Purpose:

```text
Separate two effects:
1. task-specific head training over frozen ImageNet features
2. additional Dual-Line ROI/structure/q features
```

Baseline B:

```text
Full image
-> ResNet18 ImageNet frozen feature vector
-> new task-specific MLP head
```

Important:

```text
This is transfer learning with a frozen backbone.
The ResNet18 backbone is not fine-tuned.
No ROI, structure vector, q vector, or waveform feature is used.
```

### 3-class result

| Method | Accuracy | Macro F1 | Confusion Matrix |
|---|---:|---:|---|
| Full-image head B | 98.5% | 98.5% | [[88, 3, 0], [1, 90, 0], [0, 0, 91]] |
| MVP v0.2 | 99.3% | 99.3% | [[90, 1, 0], [0, 91, 0], [0, 1, 90]] |

Error comparison:

```text
both_correct: 268
mvp_only_correct: 3
full_only_correct: 1
both_wrong: 1
```

Interpretation:

```text
Most of the jump from ImageNet original head to high 3-class accuracy comes
from using a task-specific head over frozen ImageNet features.

MVP still improves over the strong full-image baseline on this test, but the
increment is small because horse is a relatively easy added class.
```

### 2-class test2_all_balanced_4800 result

| Method | Accuracy | Macro F1 | Confusion Matrix |
|---|---:|---:|---|
| Full-image head B | 96.02% | 96.02% | [[2331, 69], [122, 2278]] |
| MVP v0.2 | 96.00% | 96.00% | [[2364, 36], [156, 2244]] |
| MVP v0.3 | 96.79% | 96.79% | [[2361, 39], [115, 2285]] |

Interpretation:

```text
On the large cat/dog test2 set, MVP v0.2 and full-image baseline have nearly
the same total accuracy, but their error directions differ.

Full-image head B:
fewer dog->cat errors, more cat->dog errors.

MVP v0.2:
fewer cat->dog errors, more dog->cat errors.

This explains why v0.3 meta correction is useful on the 4800-sample set.
```

## 16. Multiclass v0.3 Meta Trial

Purpose:

```text
Check whether a v0.3-style meta-calibrator helps in the 3-class cat/dog/horse
setting by combining MVP v0.2 and full-image head B predictions.
```

Method:

```text
Features:
MVP probabilities + MVP confidence + roi_gate
Full-image head probabilities + full confidence
Probability disagreement features

Model:
Logistic Regression meta-calibrator
```

Results on test_3cls_273:

| Method | Accuracy | Macro F1 | Confusion Matrix |
|---|---:|---:|---|
| MVP v0.2 | 99.3% | 99.3% | [[90, 1, 0], [0, 91, 0], [0, 1, 90]] |
| Meta v0.3 C=1.0 | 98.2% | 98.2% | [[90, 1, 0], [3, 88, 0], [0, 1, 90]] |
| Meta v0.3 C=0.01 | 98.9% | 98.9% | [[90, 1, 0], [2, 89, 0], [0, 0, 91]] |

Interpretation:

```text
In the 3-class test, MVP v0.2 is already too strong and has only two errors.
The meta-calibrator can recover the horse error, but it introduces new dog->cat
errors. Therefore v0.3 is not used as the base model for this setting.
```

Conclusion:

```text
v0.2 remains the base model.
v0.3 is a correction/risk layer, not the core observer.
```

## 17. Retry Policy v0 Analysis

Purpose:

```text
Before implementing Chapter 11 reobserve logic, identify which cheap signals
can select high-risk samples for retry/reobserve.
```

Dataset:

```text
test2_all_balanced_4800
MVP v0.2 wrong_total = 192 / 4800 = 4.0%
```

Signals tested:

```text
MVP confidence
MVP top2 margin
MVP vs full-image head prediction disagreement
ROI quality
geometry quality
border touch ratio
fallback flag
```

Key results:

| Rule | Retry Count | Retry Rate | Wrong Captured | Capture Rate | Retry Wrong Precision | Accepted Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| mvp_conf < 0.90 | 152 | 3.17% | 69 | 35.94% | 45.39% | 97.35% |
| mvp_conf < 0.80 | 95 | 1.98% | 40 | 20.83% | 42.11% | 96.77% |
| mvp_conf < 0.70 | 58 | 1.21% | 26 | 13.54% | 44.83% | 96.50% |
| pred_disagree | 87 | 1.81% | 44 | 22.92% | 50.57% | 96.86% |
| roi_quality < 0.35 | 2559 | 53.31% | 93 | 48.44% | 3.63% | 95.58% |
| combo_v0 | 1714 | 35.71% | 107 | 55.73% | 6.24% | 97.25% |

Interpretation:

```text
Confidence and prediction disagreement are strong retry triggers.
They concentrate errors by roughly 10x compared to the base 4.0% error rate.

ROI quality alone catches many errors but selects too many samples, so it is
better used for choosing the retry view rather than deciding retry by itself.
```

Candidate Retry Policy v0.1:

```text
if mvp_confidence < 0.90:
    retry
elif mvp_pred != full_pred:
    retry
else:
    accept
```

Chapter 11 connection:

```text
These results support adding an observation-control layer:
accept / retry / reobserve / reject.

The first implementation should use v0.2 as the base observer and use
confidence + disagreement as retry triggers. v0.3 remains an optional correction
layer, not the default base model.
```

## 18. Baseline Boundary

Decision:

```text
Full-image head B is treated as the strong frozen-feature transfer baseline.
It should not be extended with retry/reobserve as a pure baseline, because a
meaningful retry requires ROI generation, view selection, and observation
quality features. Those are part of the Dual-Line observer stack.
```

Role of full-image head B:

```text
1. Strong comparison baseline.
2. Auxiliary prior branch.
3. Disagreement trigger for retry/reobserve.
```

Role of MVP v0.2:

```text
Base Dual-Line observer model.
```

Role of MVP v0.3:

```text
Optional correction/risk layer.
Useful when enough disagreement/error samples exist, as in test2_4800.
Not always beneficial, as seen in the 3-class test.
```

## 19. MVP v0.45 Observer Neighborhood Start

Goal:

```text
Turn retry from "same-input second decision" into "choose a different
observation action".
```

Implementation added:

```text
src/dual_line/observer_neighborhood.py
tools/build_observer_neighborhood_v045.py
```

The new cache builder converts the existing per-phi tile sequence:

```text
tile_wave_tensor = [N, T, 4, 4, C]
```

into a tile-centered observer tensor:

```text
observer_patch = [N, T, 16, 3, 3, C]
```

where:

```text
N = sample count
T = phi count, currently 36
16 = 4x4 tile observers
3x3 = local neighborhood around each observer tile
C = rho, edge_ratio, int_std
```

Tile topology:

```text
Default topology: periodic
Actual scanner boundary mode: cv2.BORDER_REFLECT
```

Reason:

```text
Periodic topology gives a compact observer-relation table.
boundary_valid is stored separately so edge/corner positions can be treated
carefully during learning.
```

Tile 1 example:

```text
16  13  14
 4   1   2
 8   5   6
```

Smoke output:

```text
results/observer_neighborhood_v045_smoke/observer_neighborhood_v045.npz
results/observer_neighborhood_v045_smoke/index.csv
results/observer_neighborhood_v045_smoke/neighbor_grid.csv
results/observer_neighborhood_v045_smoke/summary.json
```

Smoke tensor shapes:

```text
tile_wave_tensor: [2, 36, 4, 4, 3]
observer_patch:   [2, 36, 16, 3, 3, 3]
```

Interpretation:

```text
v0.45.0 is not yet a learned retry selector.
It is the observation-log format needed to train one.
```

Next v0.45 steps:

```text
1. Build observer_neighborhood_v045 caches for train/test sets.
2. Join them with v0.2/full-head predictions and retry triggers.
3. Generate retry action labels using known train labels.
4. Train a small policy: accept / retry + preferred view.
5. Evaluate the same policy on test without labels.
```

## 20. MVP v0.45 Retry Action Dataset

Implementation added:

```text
tools/build_retry_action_dataset_v045.py
```

Purpose:

```text
observer_neighborhood_v045 index
+ MVP v0.2 predictions
+ full-image baseline predictions
-> retry_action_dataset_v045.csv
```

Trigger rule:

```text
retry if MVP confidence < 0.90
retry if MVP prediction != full-image baseline prediction
otherwise accept MVP
```

3-class train split result:

```text
n = 2400
MVP accuracy = 1.000
Full baseline accuracy = 1.000
retry_trigger_count = 0
target_action = accept_mvp only
```

3-class validation split result:

```text
n = 600
MVP accuracy = 0.9867
Full baseline accuracy = 0.9817
retry_trigger_count = 12
MVP wrong captured = 4 / 8
```

Validation target actions:

```text
accept_mvp: 588
accept_after_check: 4
trust_mvp_structure: 4
trust_full_prior: 1
reobserve_conflict: 1
reobserve_unresolved: 2
```

Interpretation:

```text
The train split is too clean to teach retry behavior.
The validation/test splits contain the useful retry cases.
Next, build the same observer neighborhood cache for the external test set and
use it to inspect retry candidates before training a policy.
```

## 21. MVP v0.45 Learned Retry Policy

Implementation added:

```text
tools/train_retry_policy_v045.py
```

Policy target:

```text
y = 1 if MVP prediction is wrong
y = 0 if MVP prediction is correct
```

Inputs exclude label/diagnostic columns:

```text
not used as input:
mvp_correct, full_correct, target_action, failure_type, y_true
```

Input feature groups:

```text
MVP probabilities/confidence/margin
full-image baseline probabilities/confidence/margin
MVP/full disagreement
ROI gate and cache/index quality features
observer-neighborhood summary features from observer_neighborhood_v045.npz
```

Training/evaluation setup:

```text
Dataset: test2_all_balanced_4800
Validation ratio: 0.25
Validation n: 1200
Validation MVP wrong total: 48
Target retry rate: 4%
```

Rule baseline on full 4800:

```text
retry rate: 3.92%
wrong captured: 84 / 192 = 43.75%
```

Learned policy validation results:

| Model | Retry Rate | Wrong Captured | Retry Precision | Accepted Accuracy | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 4.00% | 24 / 48 = 50.00% | 50.00% | 97.92% | 0.875 | 0.423 |
| Random Forest | 4.00% | 25 / 48 = 52.08% | 52.08% | 98.00% | 0.954 | 0.532 |
| HistGradientBoosting | 4.00% | 25 / 48 = 52.08% | 52.08% | 98.00% | 0.955 | 0.488 |

Interpretation:

```text
The retry decision is learnable.
At the same approximate retry budget, the learned policy captures more MVP
errors than the hand rule.
This is the first concrete v0.45 result: observation/retry control can be
trained from observer-neighborhood features plus model disagreement features.
```

Current limitation:

```text
This is a split inside test2_all_balanced_4800, so it is a development signal,
not a final external generalization claim.
Next step: train on one split/set and evaluate on a separate held-out set.
```

## 22. MVP v0.45 Learned Decision Correction

Implementation added:

```text
tools/train_retry_decision_policy_v045.py
```

Policy actions:

```text
use_mvp
use_full
reobserve_unresolved
```

Teacher labels:

```text
MVP correct, full wrong -> use_mvp
MVP wrong, full correct -> use_full
both wrong -> reobserve_unresolved
both correct -> use_mvp
```

Current evaluation behavior:

```text
use_full changes final prediction to the full-image baseline prediction.
reobserve_unresolved is flagged but keeps MVP prediction until a real
additional view policy is implemented.
```

Dataset:

```text
test2_all_balanced_4800 internal validation split
validation n = 1200
MVP baseline accuracy = 96.00%
Full-image baseline accuracy = 96.25%
```

RF decision policy:

```text
final accuracy = 96.33%
changed_to_full = 22
MVP wrong fixed by full = 13
MVP correct broken by full = 9
unresolved = 42
net gain = +4 / 1200
```

HGB decision policy:

```text
final accuracy = 96.25%
changed_to_full = 11
MVP wrong fixed by full = 7
MVP correct broken by full = 4
unresolved = 15
net gain = +3 / 1200
```

Interpretation:

```text
The policy can learn a conservative correction step.
The gain is small because only full-image fallback is available as a real
correction action. The larger remaining value is in reobserve_unresolved:
these samples need a new observation view, not just switching to full.
```

Next step:

```text
Implement view-action labels for unresolved/retry samples:
upper_context, body_context, expanded_1_5, expanded_2_0, center_70, full.
Then train the policy to choose a view instead of only choosing MVP vs full.
```

## 23. MVP v0.45 View-Action Learning

Implementation added:

```text
tools/build_view_action_dataset_v045.py
tools/train_view_action_policy_v045.py
```

Purpose:

```text
Move beyond if-rule retry.
Learn which observation view should be used next.
```

Input:

```text
retry_action_dataset_v045.csv
multiview_predictions.csv
observer_neighborhood_v045.npz
```

Teacher label:

```text
For each retry/reobserve candidate:
  if at least one view is correct:
      choose the most confident correct view
  else:
      reobserve_unresolved
```

Available view actions:

```text
roi
upper_context
body_context
expanded_1_5
expanded_2_0
center_70
full
reobserve_unresolved
```

Dataset result:

```text
n = 179
any_view_correct = 169
any_view_accuracy = 94.41%
unresolved = 10
```

View action label counts:

```text
center_70: 45
roi: 29
body_context: 28
upper_context: 22
expanded_2_0: 19
full: 17
reobserve_unresolved: 10
expanded_1_5: 9
```

RF view-action policy validation:

```text
validation n = 55
exact label accuracy = 29.09%
chosen view correct = 49 / 55 = 89.09%
unresolved selected = 5
```

Interpretation:

```text
Exact view-label accuracy is low because many samples have multiple correct
views. The important metric is whether the chosen view produces a correct
answer. On this small validation split, the learned view policy chose a correct
view for 89.09% of candidates.
```

Current limitation:

```text
The multiview teacher still uses predefined crop views.
The next v0.5 step should replace these coarse crop views with tile/phi
tracking and coverage-aware reobservation.
```

## 24. v0.5-Prep Observation Tracking Cache

Implementation added:

```text
tools/build_observation_tracking_v05.py
```

Purpose:

```text
Move from coarse crop-view selection toward tile/phi tracking.
This step does not rescan images yet.
It extracts tracking features from the already saved per-phi/tile observer log.
```

Input:

```text
observer_neighborhood_v045.npz
```

Output:

```text
observation_tracking_v05.npz
tracking_features.csv
summary.json
```

Core tensors:

```text
support:              [N, T, 4, 4]
instability:          [N, T, 4, 4]
support_mean:         [N, 4, 4]
support_std:          [N, 4, 4]
instability_mean:     [N, 4, 4]
instability_p95:      [N, 4, 4]
support_center:       [N, T, 2]
bucket_support:       [N, 8, 4, 4]
```

Scalar tracking features:

```text
obj_border_touch_ratio
obj_bg_support_gap
obj_bg_instability_gap
support_center_path
support_center_std
top_instability_tile_id
top_support_tile_id
top_obj_instability_tile_id
```

test2_all_balanced_4800 result:

```text
n = 4800
T = 36
grid = 4
features = rho, edge_ratio, int_std
mean_support_center_path = 0.8870
mean_support_center_std = 0.0551
mean_obj_border_touch_ratio = 0.1881
```

Interpretation:

```text
This is the first real tracking layer.
It converts the scan log into support/instability/drift signals that can guide
where to reobserve next.
```

Next step:

```text
Join tracking_features.csv into the retry/view-action datasets and retrain the
view-action policy. If view choice improves, replace fixed crop-view labels with
tile/direction action labels.
```

## 25. Tracking Features in View-Action Policy

Implementation update:

```text
tools/train_view_action_policy_v045.py now supports --tracking_csv
```

Tracking input:

```text
results/observation_tracking_v05_test2_all_balanced_4800/tracking_features.csv
```

Comparison on the same view-action validation split:

| Policy | Tracking Features | Feature Count | Chosen View Accuracy | Exact Label Accuracy | Unresolved |
|---|---:|---:|---:|---:|---:|
| RF | no | 52 | 89.09% | 29.09% | 5 |
| RF | yes | 79 | 90.91% | 25.45% | 4 |
| HGB | yes | 79 | 90.91% | 23.64% | 2 |

Interpretation:

```text
Adding tracking features slightly improved chosen-view correctness.
The split is small, so this is not a final claim, but it supports the idea that
support/instability/drift features contain useful information for reobserve
view selection.
```

Next step:

```text
Move from crop-view labels to tile/direction labels:
top_instability_tile_id, top_support_tile_id, top_obj_instability_tile_id,
and phi bucket support should define where the observer should look next.
```

## 26. v0.45 Pipeline Accuracy Check

Implementation added:

```text
tools/eval_v045_pipeline.py
```

Pipeline:

```text
1. Start with MVP prediction for every sample.
2. Learned retry policy selects retry samples.
3. Learned view-action policy selects a view.
4. If multiview prediction exists for that sample/view, replace MVP prediction.
5. If view is unresolved or missing, keep MVP prediction and report it.
```

Learned retry + learned tracking view policy:

```text
n = 4800
MVP accuracy = 96.00%
Full baseline accuracy = 96.02%
v0.45 pipeline accuracy = 98.42%
retry_count = 232
used_view_count = 116
unresolved_count = 70
missing_view_count = 46
MVP wrong fixed = 116
MVP correct broken = 0
```

Rule retry + learned tracking view policy:

```text
n = 4800
MVP accuracy = 96.00%
v0.45 pipeline accuracy = 97.31%
retry_count = 188
used_view_count = 63
unresolved_count = 95
missing_view_count = 30
MVP wrong fixed = 63
MVP correct broken = 0
```

Important caveat:

```text
Current multiview predictions exist only for a selected candidate subset.
The applied view corrections all came from MVP-wrong samples in that subset, so
this development number is optimistic and should not be treated as final
external accuracy.
```

Interpretation:

```text
The end-to-end wiring works and can recover many MVP errors when a useful
reobserve view is available. The next required validation is to generate
multiview/reobserve predictions for all retry-policy selected samples without
using correctness-filtered candidate selection.
```

## 27. v0.45 Reobserve-232 Fairer Development Check

Implementation added:

```text
tools/export_v045_retry_planner.py
```

Purpose:

```text
Export all samples selected by the learned retry policy as a planner CSV.
Then generate multiview predictions for all selected samples, not only the
previous hand/analysis-selected subset.
```

Learned retry export:

```text
n = 232
cat = 102
dog = 130
```

Requested view actions from the learned view policy:

```text
reobserve_unresolved: 70
roi: 41
center_70: 37
body_context: 31
upper_context: 19
full: 16
expanded_2_0: 13
expanded_1_5: 5
```

Fresh multiview result for all 232 retry-selected samples:

```text
any_view_correct = 180 / 232 = 77.59%
```

Single-view accuracies:

```text
full: 71.98%
expanded_2_0: 70.26%
center_70: 70.26%
expanded_1_5: 68.97%
body_context: 68.10%
upper_context: 68.10%
roi: 67.24%
```

Pipeline using fresh 232 multiview predictions:

```text
MVP accuracy = 96.00%
Full baseline accuracy = 96.02%
v0.45 pipeline accuracy = 97.77%
retry_count = 232
used_view_count = 162
unresolved_count = 70
missing_view_count = 0
MVP wrong fixed = 116
MVP correct broken = 31
net gain = +85 / 4800
```

Interpretation:

```text
The earlier 98.42% was optimistic because multiview predictions were available
only for a selected subset. After generating multiview predictions for every
learned-retry sample, the fairer development result is 97.77%.
This is still a large gain over MVP v0.2 on the same 4800 set, but it is now a
more realistic dev estimate.
```

Remaining issue:

```text
The view policy still breaks 31 MVP-correct samples.
The next policy should be more conservative: only replace MVP when the selected
view has enough confidence/margin, otherwise mark unresolved or keep MVP.
```

## 28. Strict Policy-Train / Heldout Check

Goal:

```text
Make sure retry/view policies are not trained on the same rows used for final
pipeline evaluation.
```

Implementation added:

```text
tools/split_policy_dataset_v045.py
tools/export_v045_retry_planner_from_policy.py
```

Split:

```text
source = test2_all_balanced_4800 retry dataset
policy_train = 3360
policy_heldout = 1440
train MVP accuracy = 96.01%
heldout MVP accuracy = 95.97%
```

Retry policy trained only on policy_train:

```text
model = RF
internal validation retry rate = 5.00%
wrong captured = 21 / 34 = 61.76%
```

Retry policy applied to heldout without labels:

```text
heldout selected = 76 / 1440 = 5.28%
selected MVP accuracy = 59.21%
```

Fresh multiview for heldout selected 76:

```text
any_view_correct = 56 / 76 = 73.68%
best single view = full, 72.37%
```

Heldout final pipeline results:

```text
MVP baseline = 95.97%
Full-image baseline = 96.11%
```

View policy trained only on policy_train:

```text
RF view policy:
  final accuracy = 95.97%
  used_view = 0
  unresolved = 76
  net gain = 0

HGB view policy:
  final accuracy = 95.97%
  used_view = 0
  unresolved = 76
  net gain = 0

LogReg view policy:
  final accuracy = 96.11%
  used_view = 67
  unresolved = 9
  MVP wrong fixed = 25
  MVP correct broken = 23
  net gain = +2 / 1440
```

Interpretation:

```text
Strict heldout confirms retry detection is meaningful, but the learned view
replacement policy is not strong enough yet.
The large 97.77% dev result does not hold under this stricter split.
Current robust heldout result is roughly MVP +0.14%p, matching the full-image
baseline rather than clearly beating it.
```

Next step:

```text
Train a conservative accept gate for applying the selected view.
The target should be: apply view only when it is likely to fix MVP and unlikely
to break an MVP-correct sample.
```

## 29. View Accept Gate on Strict Heldout

Implementation added:

```text
tools/train_view_accept_gate_v045.py
tools/apply_view_accept_gate_v045.py
```

Gate target:

```text
apply selected view only if:
  MVP was wrong and selected view is correct
otherwise prefer keeping MVP
```

Training source:

```text
policy_train pipeline predictions
selected view rows = 162
beneficial_total = 94
harmful_total = 30
```

Gate train result:

```text
model = LogisticRegression
min_precision target = 0.70
apply_count = 125 / 162
beneficial_applied = 88
harmful_applied = 21
net_gain = +67
```

Heldout gate diagnostic:

```text
selected view rows = 67
beneficial_total = 25
harmful_total = 23
apply_count = 45
beneficial_applied = 22
harmful_applied = 15
net_gain = +7
```

Heldout final pipeline with accept gate:

```text
MVP baseline = 95.97%
Full-image baseline = 96.11%
View pipeline before gate = 96.11%
View pipeline after gate = 96.46%
```

Gate effect on heldout:

```text
selected_view_count = 67
gate_apply_count = 45
gate_reject_count = 22
fixed_by_gate = 8
broken_by_gate = 3
net_gain_vs_before = +5
```

Interpretation:

```text
The strict heldout result now shows a real improvement over MVP and full-image
baseline:

MVP 95.97% -> v0.45 gated 96.46%

This is modest but important because retry/view/gate policies were trained only
on policy_train and evaluated on heldout.
```

Next step:

```text
Improve the gate and view policy with more policy-training data and tile/direction
actions. The current gate still applies 15 harmful views on heldout.
```

## 30. Current Experiment Summary

This section summarizes the current state before moving to v0.5.

### 30.1 Baselines

2-class external test2 balanced set:

```text
Dataset: test2_all_balanced_4800
cat = 2400
dog = 2400
```

Main baselines:

```text
ImageNet cat/dog grouped baseline:
accuracy = 95.00%

Full-image ResNet18 feature + task head:
accuracy = 96.02%

MVP v0.2:
accuracy = 96.00%
confusion = [[2364, 36], [156, 2244]]
```

Interpretation:

```text
MVP v0.2 is not simply better than the strong full-image baseline on raw
accuracy, but it has different error behavior and enables retry/reobserve
because it exposes ROI, structure, quality, and observer features.
```

### 30.2 Multiclass Check

3-class cat/dog/horse test:

```text
Dataset: test_3cls_273
MVP v0.2 3-class accuracy = 99.27%
Full-image baseline B accuracy = 98.53%
ImageNet grouped sum accuracy = 82.42%
ImageNet grouped max accuracy = 86.08%
```

MVP v0.2 errors:

```text
cat_44 -> dog
horse_horse_11329 -> dog
```

Interpretation:

```text
Adding horse did not collapse the representation.
The class space can expand beyond binary cat/dog, but more classes still need
proper train/test splits and stronger heldout validation.
```

### 30.3 v0.45 Components Implemented

Implemented tools:

```text
tools/build_observer_neighborhood_v045.py
tools/build_retry_action_dataset_v045.py
tools/train_retry_policy_v045.py
tools/train_retry_decision_policy_v045.py
tools/build_view_action_dataset_v045.py
tools/train_view_action_policy_v045.py
tools/build_observation_tracking_v05.py
tools/eval_v045_pipeline.py
tools/split_policy_dataset_v045.py
tools/export_v045_retry_planner.py
tools/export_v045_retry_planner_from_policy.py
tools/train_view_accept_gate_v045.py
tools/apply_view_accept_gate_v045.py
```

New tensors/features:

```text
observer_patch:
  [N, T, 16, 3, 3, C]

support:
  [N, T, 4, 4]

instability:
  [N, T, 4, 4]

support_center:
  [N, T, 2]

bucket_support:
  [N, 8, 4, 4]
```

Interpretation:

```text
v0.45 now has the machinery for:
1. detecting retry candidates
2. choosing a reobserve view
3. deciding whether to apply the selected view
4. extracting early tile/phi tracking features
```

### 30.4 Development Results

Non-strict dev result after generating multiview for all learned-retry samples:

```text
Dataset: test2_all_balanced_4800
MVP baseline = 96.00%
v0.45 pipeline = 97.77%
retry_count = 232
used_view_count = 162
unresolved_count = 70
MVP wrong fixed = 116
MVP correct broken = 31
net gain = +85 / 4800
```

Important caveat:

```text
This is a development result. It is useful for seeing whether the mechanism can
recover MVP errors, but it is not the final generalization number because policy
training and evaluation were not fully separated.
```

### 30.5 Strict Heldout Result

Strict split:

```text
policy_train = 3360
policy_heldout = 1440
train MVP accuracy = 96.01%
heldout MVP accuracy = 95.97%
```

Heldout without accept gate:

```text
MVP baseline = 95.97%
Full-image baseline = 96.11%
v0.45 view pipeline = 96.11%
net gain = +2 / 1440
```

Heldout with accept gate:

```text
MVP baseline = 95.97%
Full-image baseline = 96.11%
v0.45 gated pipeline = 96.46%
selected_view_count = 67
gate_apply_count = 45
gate_reject_count = 22
fixed_by_gate = 8
broken_by_gate = 3
net gain vs before gate = +5
```

Interpretation:

```text
Strict heldout shows a real but modest improvement:

MVP 95.97% -> v0.45 gated 96.46%

This is the most reliable current result because retry/view/gate policies were
trained on policy_train and evaluated on heldout.
```

### 30.6 What Is Proven So Far

Supported by experiments:

```text
1. Retry detection is learnable.
2. Observer/tracking features are useful signals.
3. Reobserve views can recover many MVP errors when available.
4. A conservative accept gate is necessary.
5. The strict heldout pipeline can improve over MVP and full-image baseline,
   but the margin is still modest.
```

Not proven yet:

```text
1. Final external generalization.
2. Large improvement on a completely independent dataset.
3. True tile/direction reobservation.
4. Stage-wise Chapter 11 observation tracking.
5. Exact coverage-aware original-coordinate backprojection.
```

### 30.7 Next Technical Step

Recommended next step:

```text
Build tile/phi coverage-aware tracking.
```

Why:

```text
Current reobserve actions still use predefined crop views:
roi, full, center_70, upper_context, body_context, expanded_1_5, expanded_2_0.

The next improvement should replace those with tile/direction actions:
which tile, which phi bucket, which neighboring support region, and which
coverage gap should be observed next.
```

Proposed next tool:

```text
tools/build_tile_phi_coverage_v05.py
```

Expected output:

```text
coverage_tensor: [N, T, 16, 4, 4]

meaning:
for each sample, phi, observer tile:
which original 4x4 tile regions were covered and with what approximate weight.
```

Then:

```text
1. backproject support/instability to original grid
2. locate missing support regions
3. generate tile/direction action labels
4. run real observer_scan reobserve from selected tile/direction
5. retrain retry/view/gate policies
```

## 31. v0.5 Tile/Phi Coverage Tracking Start

Implementation added:

```text
tools/build_tile_phi_coverage_v05.py
```

Purpose:

```text
Start true v0.5 tracking by estimating which original 4x4 regions each
phi/tile observer covered.
```

Input:

```text
observer_neighborhood_v045.npz
observation_tracking_v05.npz
```

Output:

```text
tile_phi_coverage_v05.npz
summary.json
```

Core tensor:

```text
coverage_tensor: [T, 16, 4, 4]
```

Meaning:

```text
coverage_tensor[t, observer_tile, r, c]
= approximate area fraction of observer_tile at phi[t]
  overlapping original grid cell (r, c)
```

Boundary model:

```text
reflect approximation
```

Reason:

```text
observer_scan uses cv2.BORDER_REFLECT during image shift.
The first v0.5 implementation uses a grid-level reflection approximation, not
pixel-perfect OpenCV backprojection.
```

test2_all_balanced_4800 result:

```text
coverage_shape = [36, 16, 4, 4]
coverage_sum_min = 1.0
coverage_sum_max = 1.0
support_backprojected_shape = [4800, 4, 4]
support_backprojected_mean = 0.9326
instability_backprojected_mean = 1.0310
```

Interpretation:

```text
This is the first actual coverage-aware tracking layer.
Support and instability can now be projected back into the original 4x4
coordinate frame instead of remaining only in shifted-frame tile coordinates.
```

Next step:

```text
Build reobserve targets from backprojected support/instability:
target original tile
target direction / phi bucket
target reason
target score
```

## 32. v0.5 Reobserve Target Generation

Implementation added:

```text
tools/build_reobserve_targets_v05.py
```

Purpose:

```text
Convert coverage-aware support/instability maps into explicit reobserve
targets.
```

Input:

```text
tile_phi_coverage_v05.npz
observation_tracking_v05.npz
optional retry-selected rows
```

Output:

```text
reobserve_targets_v05.csv
summary.json
```

Target fields:

```text
target_tile_id
target_r
target_c
target_direction
target_phi_bucket
target_reason
target_score
target_support
target_instability
```

Heldout retry-selected result:

```text
n = 76
```

Reason counts:

```text
outside_support_or_instability: 27
frontier_outside_support: 27
inside_obj_instability: 22
```

Direction counts:

```text
n: 15
s: 14
w: 9
center: 8
nw: 8
sw: 8
se: 5
e: 5
ne: 4
```

Interpretation:

```text
The target generator is producing explicit tile/direction targets from tracking
signals, not predefined crop names.
The distribution is not collapsed to a single direction or tile, which is a
good sign for the first v0.5 target heuristic.
```

Next step:

```text
Use reobserve_targets_v05.csv to run actual observer_scan reobserve.
The output should be new .tiles.npz files, not just crop predictions.
```

## 33. v0.5 Real Reobserve Scan

Implementation added:

```text
tools/run_reobserve_targets_v05.py
```

Purpose:

```text
Run observer_scan again from v0.5 tile targets.
This is the first real reobserve step: it creates new scan CSVs and new
.tiles.npz files.
```

Input:

```text
reobserve_targets_v05.csv
dataset root / split
```

Reobserve object-mask seed:

```text
obj_mode = target_plus_core

object_tiles:
  target tile
  original core tile
```

Heldout smoke result:

```text
n = 3
ok = 3
```

Heldout full retry-selected result:

```text
n = 76
ok = 76
scan_step_deg = 10
scan_shift_frac = 0.06
energy_feature = edge_ratio
energy_agg = mean
```

Output:

```text
results/reobserve_v05_policyheldout_seed77/reobserve_index.csv
results/reobserve_v05_policyheldout_seed77/scans/*.csv
results/reobserve_v05_policyheldout_seed77/scans/*.tiles.npz
results/reobserve_v05_policyheldout_seed77/target_obj_json/*.target_obj.json
```

Interpretation:

```text
v0.5 now performs actual observer re-scanning.
This is no longer only crop-view re-evaluation.
The next step is to rebuild dual_line_cache from the new .tiles.npz files and
compare before/after predictions.
```

## 34. v0.5 Real Reobserve Evaluation

Implementation added:

```text
tools/compare_reobserve_v05_before_after.py
```

Pipeline:

```text
1. Build dual-line cache from real reobserve .tiles.npz files.
2. Build ROI texture cache from the reobserve cache.
3. Build full-image texture cache for the same 76 samples.
4. Evaluate the existing v0.2 model on the reobserve cache.
5. Compare original MVP prediction vs reobserve prediction for the same samples.
```

Reobserve cache result:

```text
n = 76
class_counts = {cat: 37, dog: 39}
structure_dim = 46
q_dim = 6
fallback_rate = 0.0
warning_count = 0
```

Texture cache:

```text
backbone = resnet18
weights = default
embedding_dim = 512
device = cuda
ROI texture cache = ok
full-image texture cache = ok
```

Evaluation:

```text
model = results/stage0_v02_dual_texture_1000/stage0_v02_model.pt
input = real reobserve cache
n = 76
accuracy = 0.566
macro_f1 = 0.503
confusion_matrix = [[35, 2], [31, 8]]
```

Before/after comparison on the same 76 selected samples:

```text
original MVP accuracy on selected samples = 0.592
real reobserve accuracy = 0.566
fixed_by_reobserve = 2
broken_by_reobserve = 4
same_correct = 41
same_wrong = 29
```

Reason breakdown:

```text
frontier_outside_support:
  n = 27
  original MVP accuracy = 0.556
  reobserve accuracy = 0.593
  fixed = 1
  broken = 0

inside_obj_instability:
  n = 22
  original MVP accuracy = 0.591
  reobserve accuracy = 0.591
  fixed = 1
  broken = 1

outside_support_or_instability:
  n = 27
  original MVP accuracy = 0.630
  reobserve accuracy = 0.519
  fixed = 0
  broken = 3
```

Interpretation:

```text
The first real v0.5 reobserve implementation works mechanically:
it selects tile/phi targets, re-runs observer_scan, creates new .tiles.npz
files, rebuilds cache, extracts texture, and evaluates.

But replacing the original observation with the reobserve observation is not
the right policy. The reobserve crop is too narrow or too biased for many dog
cases, especially cat-like dogs. This pushes many dog samples toward cat.

The useful result is diagnostic:
real reobserve should be treated as additional evidence, not as a replacement
input.
```

Next design direction:

```text
v0.5 should compare:
  original observation
  reobserve observation
  delta between them
  target reason/direction

Then a gate should decide:
  keep original
  accept reobserve
  request another target
  trust full/global branch

This moves v0.5 from "retry by replacement" to "observation evidence fusion".
```

## 35. v0.5 Train-Time Reobserve Learning

Motivation:

```text
Real reobserve should be learned, not applied by a fixed if-rule.
Low confidence such as 0.9 is better interpreted as:
  "this observation state needs another observation"
not simply:
  "replace the answer with the retry answer"
```

Train-side real reobserve generation:

```text
source = policy_train retry-selected rows
n = 180
```

Target generation:

```text
results/reobserve_targets_v05_policytrain_seed77/reobserve_targets_v05.csv
```

Target reason counts:

```text
frontier_outside_support: 82
inside_obj_instability: 52
outside_support_or_instability: 46
```

Actual observer re-scan:

```text
results/reobserve_v05_policytrain_seed77/scans/*.tiles.npz
n = 180
ok = 180
obj_mode = target_plus_core
```

Reobserve cache:

```text
n = 180
class_counts = {cat: 73, dog: 107}
structure_dim = 46
q_dim = 6
fallback_rate = 0.0
warning_count = 0
```

Train reobserve-only evaluation:

```text
model = results/stage0_v02_dual_texture_1000/stage0_v02_model.pt
n = 180
accuracy = 0.372
macro_f1 = 0.341
confusion_matrix = [[53, 20], [93, 14]]
```

Train before/after comparison:

```text
original MVP accuracy on selected train samples = 0.383
real reobserve accuracy = 0.372
fixed_by_reobserve = 12
broken_by_reobserve = 14
same_correct = 55
same_wrong = 99
```

Interpretation:

```text
The train set confirms the heldout result:
real reobserve is not a safe replacement input.
However, it contains recoverable cases. The learning target is therefore:

  apply reobserve only when it is likely beneficial
  otherwise keep original MVP/global decision
```

## 36. v0.5 Reobserve Accept Gate

Implementation added:

```text
tools/train_apply_reobserve_accept_gate_v05.py
```

Training target:

```text
beneficial = fixed_by_reobserve
harmful / neutral = all other rows
```

Observable features:

```text
mvp confidence
full confidence
reobserve confidence
ROI gate
reobserve prob_cat / prob_dog
target score / support / instability
target tile / phi / direction / reason
prediction disagreements:
  MVP vs Full
  MVP vs Reobserve
  Full vs Reobserve
```

Training data:

```text
train_csv = results/compare_reobserve_v05_policytrain_seed77/before_after.csv
n = 180
```

Heldout data:

```text
eval_csv = results/compare_reobserve_v05_policyheldout_seed77/before_after.csv
n = 76
```

Gate result:

```text
threshold = 0.41
min_precision target = 0.70
```

Train:

```text
MVP accuracy = 0.383
Reobserve-only accuracy = 0.372
Gated final accuracy = 0.450
gate_apply_count = 12
gate_fixed = 12
gate_broken = 0
net_gain_vs_mvp = +12
```

Heldout:

```text
MVP accuracy = 0.592
Reobserve-only accuracy = 0.566
Gated final accuracy = 0.618
gate_apply_count = 2
gate_fixed = 2
gate_broken = 0
net_gain_vs_mvp = +2
```

Heldout applied samples:

```text
dog_keeshond_67:
  MVP = cat
  Reobserve = dog
  target_reason = frontier_outside_support

dog_pomeranian_3:
  MVP = cat
  Reobserve = dog
  target_reason = inside_obj_instability
```

Interpretation:

```text
This is the first learned v0.5 evidence gate for real reobserve.
The gain is small because the selected set is small, but it is directionally
correct: the gate applies reobserve conservatively and avoids breaking correct
MVP cases on heldout.

The model is now learning "when to trust reobserve", not merely retrying by
confidence threshold or replacing the original observation.
```

Next step:

```text
Scale this from selected retry rows to the full train/eval pipeline:
1. retry trigger chooses whether reobserve is needed
2. tile/phi target chooses where to observe
3. real observer_scan creates reobserve evidence
4. accept gate decides whether to keep original or accept reobserve

Then report final accuracy on the full heldout split, not only on selected rows.
```

## 37. v0.5 Redesign: Learn Where To Reobserve

Problem with the first v0.5:

```text
where to reobserve = formula target
whether to trust reobserve = learned accept gate
```

This is not enough. Some object regions can have low dE / low support but still
be the real object. A pure formula target can miss those cases.

New v0.5 direction:

```text
where to reobserve = candidate actions + learned action scorer
whether to trust reobserve = learned accept gate
```

Implementation added:

```text
tools/build_reobserve_action_candidates_v05.py
```

Implementation updated:

```text
tools/run_reobserve_targets_v05.py
  supports action_id
  supports action_kind
  supports per-row action_obj_mode
  writes action-specific output filenames

tools/build_dual_line_texture_cache.py
  resolves action-suffixed tiles names back to the original image
```

Candidate action types:

```text
frontier_instability
outside_frontier_gap
inside_instability
inside_low_support
outside_low_support
core_recheck
core_neighbor_{direction}
```

Important addition:

```text
inside_low_support
outside_frontier_gap
outside_low_support
```

These are included because low dE / low support can still correspond to object
regions. The learner should decide whether such regions are worth observing.

Policy-train candidate generation:

```text
source = v045_policytrain_retry_planner_seed77/selected_retry_rows.csv
n_samples = 180
n_actions = 1440
max_actions_per_sample = 8
```

Action distribution:

```text
frontier_instability: 360
outside_frontier_gap: 360
inside_instability: 311
inside_low_support: 311
outside_low_support: 98
```

Heldout candidate generation:

```text
source = v045_policyheldout_retry_planner_seed77/selected_retry_rows.csv
n_samples = 76
n_actions = 608
max_actions_per_sample = 8
```

Heldout action distribution:

```text
frontier_instability: 152
outside_frontier_gap: 152
inside_instability: 130
inside_low_support: 130
outside_low_support: 44
```

Smoke test:

```text
source = policytrain action candidates
limit = 8
all 8 actions from cat_Abyssinian_153
observer_scan ok = 8 / 8
dual_line_cache ok = 8 / 8
texture_cache ok = 8 / 8
```

Example action outputs:

```text
cat_Abyssinian_153__a00_frontier_instability.tiles.npz
cat_Abyssinian_153__a02_outside_frontier_gap.tiles.npz
cat_Abyssinian_153__a06_inside_low_support.tiles.npz
```

Interpretation:

```text
The redesigned v0.5 is now ready to generate multiple real reobserve outcomes
per sample. This makes "where to look" learnable.

The next training dataset should be:

  original observation state
  candidate action features
  reobserve outcome
  reward / improvement

Then train:

  observation_state + action -> expected_gain
```

Next step:

```text
Run real reobserve for a manageable candidate subset first, for example:
  policy_train: 180 samples x 3 or 4 actions
  policy_heldout: 76 samples x same action budget

Then build an action reward table and train the first action scorer.
```

## 38. v0.5 Action Reward Dataset and First Scorer

Policy-train real action reobserve:

```text
source candidates = reobserve_action_candidates_v05_policytrain_seed77
n_samples = 180
n_actions = 1440
```

Action-level evaluation:

```text
model = stage0_v02_dual_texture_1000
action_accuracy = 0.408
confusion_matrix = [[459, 125], [728, 128]]
```

Reward dataset:

```text
results/reobserve_action_reward_v05_policytrain_seed77/reobserve_action_reward_v05.csv
```

Train reward summary:

```text
n_samples = 180
n_actions = 1440
MVP accuracy on selected samples = 0.383
action average accuracy = 0.408
fixed_by_action = 115
broken_by_action = 80
samples_with_any_fix = 38
samples_with_any_beneficial = 64
oracle_best_action_accuracy = 0.589
oracle_best_fixed = 38
oracle_best_broken = 1
```

Policy-heldout real action reobserve:

```text
source candidates = reobserve_action_candidates_v05_policyheldout_seed77
n_samples = 76
n_actions = 608
```

Action-level evaluation:

```text
model = stage0_v02_dual_texture_1000
action_accuracy = 0.595
confusion_matrix = [[279, 17], [229, 83]]
```

Heldout reward summary:

```text
n_samples = 76
n_actions = 608
MVP accuracy on selected samples = 0.592
action average accuracy = 0.595
fixed_by_action = 28
broken_by_action = 26
samples_with_any_fix = 13
samples_with_any_beneficial = 24
oracle_best_action_accuracy = 0.763
oracle_best_fixed = 13
oracle_best_broken = 0
```

Important observation:

```text
The action candidates contain real recovery paths.
Heldout oracle best action improves selected-sample accuracy:

MVP 59.2% -> oracle best action 76.3%

This proves that "where to reobserve" matters and is worth learning.
```

Beneficial action kinds on heldout:

```text
frontier_instability: 21
outside_frontier_gap: 17
inside_instability: 17
inside_low_support: 10
outside_low_support: 7
```

Interpretation:

```text
Low-support actions still produce beneficial outcomes.
This supports the idea that low dE / low support cannot simply be discarded by
a formula.
```

First action scorer implementation:

```text
tools/train_reobserve_action_scorer_v05.py
```

Training objective:

```text
input = original observation state + candidate action features
target = action_reward

At inference:
score all candidate actions for a sample
select the action with max predicted reward
```

Compared models:

```text
HGB
RF
Ridge
```

Heldout action-scorer results:

```text
HGB:
  selected_action_accuracy = 0.605
  selected_fixed = 5
  selected_broken = 4
  net_gain_vs_mvp = +1

RF:
  selected_action_accuracy = 0.566
  selected_fixed = 3
  selected_broken = 5
  net_gain_vs_mvp = -2

Ridge:
  selected_action_accuracy = 0.592
  selected_fixed = 4
  selected_broken = 4
  net_gain_vs_mvp = 0
```

Interpretation:

```text
The first learned action scorer works mechanically but is still weak.
It finds a small heldout gain with HGB, but the gap to oracle is large:

HGB selected action = 60.5%
Oracle best action = 76.3%

So v0.5 has proven that useful action choices exist, but action selection needs
better features / more data / a more observation-aware target.
```

Next step:

```text
1. Add accept gate after selected action.
2. Add delta features between original and selected action.
3. Increase action training data.
4. Move toward stage-wise 1-5 observation where reward is not only correctness
   but also ROI stability, true-class confidence increase, and tracking quality.
```

## 39. v0.5 Selected Action Accept Gate

Implementation added:

```text
tools/train_apply_selected_action_accept_gate_v05.py
```

Purpose:

```text
The action scorer chooses where to reobserve.
The accept gate decides whether to use that selected action result or keep the
original MVP prediction.
```

Input:

```text
train_selected_actions.csv from HGB action scorer
eval_selected_actions.csv from HGB action scorer
```

Training target:

```text
accept if selected action fixes original MVP
otherwise keep original MVP
```

Result:

```text
threshold = 0.19
```

Train selected:

```text
n = 180
MVP accuracy = 0.383
selected action accuracy = 0.589
final gated accuracy = 0.594
gate_apply_count = 38
gate_fixed = 38
gate_broken = 0
net_gain_vs_mvp = +38
```

Heldout selected:

```text
n = 76
MVP accuracy = 0.592
selected action accuracy = 0.605
final gated accuracy = 0.618
gate_apply_count = 16
gate_fixed = 5
gate_broken = 3
net_gain_vs_mvp = +2
```

Interpretation:

```text
The accept gate improves the selected action result slightly:

selected action alone:
  fixed = 5
  broken = 4
  net = +1

selected action + gate:
  fixed = 5
  broken = 3
  net = +2

This confirms the runtime shape:
  retry trigger
  -> candidate actions
  -> action scorer
  -> real reobserve
  -> accept gate
  -> final prediction

The result is still modest, but the full learned v0.5 loop now works end to end
on the selected heldout rows.
```

Remaining gap:

```text
Heldout selected:
  gated final = 61.8%
  oracle best action = 76.3%

The main missing piece is stronger action selection, not merely a stricter
accept gate.
```

## 40. v0.5 Tile-View Relation Tracking Features

Problem:

```text
The previous action scorer knew target support/instability, but it did not know
how tile A sees tile B.

Example:
  tile 6 sees tile 7 with some coverage
  but the scorer did not know whether edge/rho/int_std from tile 6's view
  is consistent with tile 7's own view.
```

Implementation added:

```text
tools/build_tile_view_relation_v05.py
tools/attach_tile_view_relation_to_actions_v05.py
```

Relation tensor:

```text
relation: [N, 16, 16, 16]

N = sample/action count
16 = observer tile
16 = source tile
16 = relation feature count
```

Relation features:

```text
coverage_sum
coverage_max

rho_observer_wmean
rho_source_self
rho_delta_wmean
rho_abs_delta_wmean

edge_observer_wmean
edge_source_self
edge_delta_wmean
edge_abs_delta_wmean

int_observer_wmean
int_source_self
int_delta_wmean
int_abs_delta_wmean

mutual_coverage_min
mutual_edge_delta_abs_mean
```

Feature meaning:

```text
core -> target:
  how the original core tile sees the candidate target tile

target -> core:
  how the candidate target tile sees the original core tile

pair mean:
  symmetric consistency summary
```

Heldout attach check:

```text
n_rows = 608
missing_count = 0
```

Train attach check:

```text
n_rows = 1440
missing_count = 0
```

Relation action scorer:

```text
model = HGB
train = policytrain actions with relation
eval = policyheldout actions with relation
```

Heldout selected action result:

```text
MVP accuracy = 0.592
selected action accuracy = 0.605
selected_fixed = 6
selected_broken = 5
net_gain_vs_mvp = +1
```

Compared to non-relation HGB:

```text
non-relation:
  selected_fixed = 5
  selected_broken = 4
  net = +1

relation:
  selected_fixed = 6
  selected_broken = 5
  net = +1
```

Interpretation:

```text
Relation features changed selected actions but did not improve action-only net
gain by themselves. They made the scorer more active: one more fix and one more
break.
```

Relation selected action + accept gate:

```text
MVP accuracy = 0.592
selected action accuracy = 0.605
final gated accuracy = 0.645
gate_apply_count = 17
gate_fixed = 6
gate_broken = 2
net_gain_vs_mvp = +4
```

Comparison:

```text
no relation + gate:
  final gated accuracy = 0.618
  gate_fixed = 5
  gate_broken = 3
  net = +2

relation + gate:
  final gated accuracy = 0.645
  gate_fixed = 6
  gate_broken = 2
  net = +4
```

Interpretation:

```text
Tile-view relation features become useful when combined with the accept gate.
The scorer explores better candidates, while the gate filters some harmful
cases.

This is the first evidence that tile-to-tile perspective features help the
learned tracking/reobserve loop.
```

## 41. v0.6 Tile Wave Groups and Weak Group Rewards

Goal:

```text
Move from action-level reobserve learning to group-level tracking learning.

Tile waveforms should form groups, and groups should receive weak reward labels
from action outcomes.
```

Implementation added:

```text
tools/build_tile_wave_groups_v06.py
tools/build_tile_group_reward_dataset_v06.py
```

Group construction:

```text
tile waveform:
  rho[36]
  edge_ratio[36]
  int_std[36]

tile-view relation:
  core/target perspective consistency

graph:
  node = tile
  edge = waveform similarity + relation consistency

group:
  connected components over the tile graph
```

Initial grouping parameters:

```text
threshold = 0.50
relation_weight = 0.35
topology = 8-neighbor
```

Group features:

```text
group_size
bbox_area_tiles
compactness
obj_overlap_ratio
core_in_group
wave mean/std/stability
internal similarity
boundary similarity
boundary contrast
internal relation edge delta
internal relation mutual coverage
```

Weak group reward:

```text
action_reward is action-level.
group_reward = action_reward * group_relevance

group_relevance =
  0.40 * contains_target_tile
  0.25 * contains_core_tile
  0.25 * obj_overlap_ratio
  0.10 * compactness
```

Train group reward dataset:

```text
groups_csv = results/tile_wave_groups_v06_action_train_t050/tile_wave_groups_v06.csv
n_group_rows = 6136
n_actions = 1440
missing_action_reward = 0
group_beneficial_count = 293
group_harmful_count = 183
group_object_like_weak_count = 1388
mean_group_relevance = 0.284
mean_group_reward = 0.0106
```

Heldout group reward dataset:

```text
groups_csv = results/tile_wave_groups_v06_action_heldout_t050/tile_wave_groups_v06.csv
n_group_rows = 2576
n_actions = 608
missing_action_reward = 0
group_beneficial_count = 109
group_harmful_count = 73
group_object_like_weak_count = 579
mean_group_relevance = 0.282
mean_group_reward = 0.00034
```

Interpretation:

```text
v0.6 now has weak group-level supervision.

This is not yet a final object-completeness model, but it provides the training
table needed for:

  group objectness scorer
  group completeness scorer
  group expected reward scorer
```

Next step:

```text
Train a group scorer:
  group_features -> group_beneficial / group_reward

Then aggregate group scores back into action scoring:
  action score = scorer(action meta + best group features)
```

## 42. v0.6 Tile Group Scorer

Goal:

```text
Use tile-wave groups as learned tracking evidence.

Instead of selecting a reobserve action only from hand-written action metadata,
score the groups produced by tile waveform + tile-view relation tracking, then
reduce:

  group score -> best group per action -> best action per sample
```

New script:

```text
tools/train_tile_group_scorer_v06.py
```

Input:

```text
train_csv = results/tile_group_reward_v06_train_t050/tile_group_reward_v06.csv
eval_csv  = results/tile_group_reward_v06_heldout_t050/tile_group_reward_v06.csv
model     = HistGradientBoostingRegressor
```

Important leakage check:

```text
First run accidentally included delta_true_prob_vs_mvp.

That feature requires the true label, so it is only useful as an analysis/oracle
debug signal. It must not be used by the runtime policy.
```

Leakage-included analysis result:

```text
eval selected samples = 76
base MVP accuracy     = 59.2%
selected action acc   = 76.3%
fixed                 = 13
broken                = 0
net gain              = +13
```

Interpretation:

```text
This matches the group oracle result and confirms that the grouped rows contain
enough information to identify useful reobserve outcomes, but this result is
not a deployable policy score because it used true-label information.
```

Runtime-safe group scorer:

```text
Removed:
  delta_true_prob_vs_mvp

Kept:
  group geometry
  waveform mean/std/stability
  internal/boundary similarity
  tile-view relation features
  base/action confidence
  group_relevance
  action kind/mode/prediction metadata
```

Runtime-safe heldout result:

```text
eval selected samples = 76
base MVP accuracy     = 59.2%
selected action acc   = 64.5%
fixed                 = 13
broken                = 9
net gain              = +4
```

Runtime-safe train result:

```text
train selected samples = 180
base MVP accuracy      = 38.3%
selected action acc    = 58.3%
fixed                  = 38
broken                 = 2
net gain               = +36
```

Selected action distribution on heldout:

```text
frontier_instability   23
outside_frontier_gap   17
inside_instability     17
outside_low_support    13
inside_low_support      6
```

Accept gate after runtime-safe group selection:

```text
train:
  final accuracy = 58.3%
  gate fixed     = 38
  gate broken    = 2
  net gain       = +36

heldout:
  final accuracy = 64.5%
  gate fixed     = 13
  gate broken    = 9
  net gain       = +4
```

Interpretation:

```text
The group scorer is now doing something real:

  base selected heldout: 59.2%
  group-selected heldout: 64.5%

This is not yet enough to beat the earlier strict v0.45 gated policy on the
full heldout pool, but it is stronger as a research signal because the model is
using tile-wave groups rather than only if-rule style action metadata.

The remaining weakness is not action discovery. The oracle still reaches 76.3%.
The weakness is action acceptance: the current gate fails to reject enough
harmful reobserve outcomes.
```

Next direction:

```text
v0.6 should split the policy into two learned parts:

1. where-to-look scorer
   tile-wave group + relation features -> candidate action score

2. should-accept scorer
   base prediction + selected reobserve prediction + group evidence
   -> accept reobserve / keep original

The second scorer needs better features than the current confidence gate,
especially features that describe whether the selected group is a complete
object view or only a misleading fragment.
```

## 43. v0.6 Improvement: Predict Beneficial Groups Directly

Problem:

```text
The first runtime-safe group scorer used group_reward regression.

Heldout selected samples:
  base MVP              59.2%
  group_reward scorer   64.5%
  net gain              +4

This was a real signal, but the gain was small.
```

Diagnosis:

```text
The old accept gate was built for v0.5 action rows.
When applied to v0.6 group-selected rows, many expected columns were missing:

  action_prob_cat / action_prob_dog
  full_confidence_base
  target_action_base
  failure_type_base

So the gate filled many features with 0 or unknown and did not really use the
new tile-wave group evidence.
```

Added:

```text
tools/train_apply_group_accept_gate_v06.py
```

This v0.6 gate uses group features directly:

```text
group geometry
wave mean/std/stability
internal/boundary similarity
tile-view relation features
base/action confidence
selected group score
group relevance
action kind/mode/prediction metadata
```

However, accept-gate improvement was still weak:

```text
HGB action_correct gate:
  eval final accuracy = 63.2%
  fixed = 10
  broken = 7
  net gain = +3

RF action_correct gate:
  eval final accuracy = 63.2%
  fixed = 8
  broken = 5
  net gain = +3

LogReg action_correct gate:
  eval final accuracy = 59.2%
  fixed = 6
  broken = 6
  net gain = 0
```

Conclusion:

```text
The bottleneck was not fixed by a better accept gate.
The more promising change is the group scorer target.
```

Updated group scorer:

```text
tools/train_tile_group_scorer_v06.py

New options:
  --target group_reward
  --target group_beneficial
  --target beneficial_action
  --target action_correct

New models:
  --model logreg
  --model rf
  --model hgb
```

Best no-leak improvement:

```text
command:
  python -m tools.train_tile_group_scorer_v06
    --train_csv results/tile_group_reward_v06_train_t050/tile_group_reward_v06.csv
    --eval_csv results/tile_group_reward_v06_heldout_t050/tile_group_reward_v06.csv
    --out_dir results/tile_group_scorer_v06_logreg_group_beneficial_t050
    --model logreg
    --target group_beneficial

heldout selected samples = 76
base MVP accuracy        = 59.2%
selected action accuracy = 68.4%
fixed                    = 9
broken                   = 2
net gain                 = +7
```

Fit quality:

```text
train group_beneficial:
  accuracy@0.5 = 88.3%
  roc_auc      = 95.3%

heldout group_beneficial:
  accuracy@0.5 = 86.3%
  roc_auc      = 92.8%
```

Selected action distribution:

```text
outside_frontier_gap   32
inside_instability     16
outside_low_support    14
frontier_instability   12
inside_low_support      2
```

Interpretation:

```text
Predicting group_reward as a continuous weak reward was too noisy.

Predicting group_beneficial directly is more stable for the current small
policy dataset. It asks a simpler question:

  "Does this group belong to a reobserve action that tends to help?"

This improved the runtime-safe result from:

  64.5% / net +4

to:

  68.4% / net +7
```

Accept gate after the improved group scorer:

```text
LogReg action_correct gate:
  final accuracy = 61.8%
  fixed = 4
  broken = 2
  net gain = +2
```

So the best current v0.6 result is:

```text
Use group_beneficial scorer directly.
Do not add the current accept gate.
```

Remaining gap:

```text
group oracle = 76.3%
best runtime-safe group scorer = 68.4%

There is still useful signal left in the candidate set, but the model needs a
better object-completeness / harmful-fragment discriminator before the accept
gate becomes reliable.
```

## 44. v0.6 Fixed-Oriented Check

Question:

```text
Can we improve fixed behavior directly?

The best current group_beneficial scorer fixes 9 MVP errors and breaks 2
previously correct MVP predictions on the 76 heldout selected samples.
```

Observed fixed samples:

```text
Most fixed cases are dog samples where MVP predicted cat, then reobserve
predicted dog.

Typical fixed classes:
  chihuahua
  japanese_chin
  keeshond
  pomeranian
  samoyed
  scottish_terrier
```

Observed broken samples:

```text
Both broken cases are cat samples where MVP predicted cat, then reobserve
flipped to dog:

  cat_Abyssinian_109
  cat_Siamese_171
```

Interpretation:

```text
The reobserve policy is learning a useful "cat-like dog rescue" behavior.
The danger is that some dog-like cats are pulled into the same correction path.
```

Direct fixed-target experiment:

```text
target = fixed_by_action

heldout selected samples = 76

HGB:
  selected action acc = 60.5%
  fixed = 13
  broken = 12
  net = +1

RF:
  selected action acc = 60.5%
  fixed = 13
  broken = 12
  net = +1

LogReg:
  selected action acc = 65.8%
  fixed = 12
  broken = 7
  net = +5
```

Broken-risk experiment:

```text
target = broken_by_action
score = -P(broken)

This avoided broken cases but also avoided almost all fixes:

HGB/RF:
  fixed = 0
  broken = 0
  net = 0
```

Combined score sweep:

```text
score = P(group_beneficial) + a * P(fixed_by_action) - b * P(broken_by_action)

Best observed heldout result:
  net = +7
  selected action acc = 68.4%

This ties the existing group_beneficial scorer; it does not beat it.
```

Conclusion:

```text
For the current small policy dataset, optimizing fixed directly is unstable.
It increases fixed count, but broken count rises with it.

The current best no-leak strategy remains:

  LogReg group_beneficial scorer
  accept selected reobserve broadly

Current best heldout selected:
  base MVP = 59.2%
  selected action = 68.4%
  fixed = 9
  broken = 2
  net = +7
```

Next requirement:

```text
To improve fixed further, we need a discriminator for:

  cat-like dog rescue
  vs
  dog-like cat false correction

This likely requires features beyond confidence:
  ear/face/body completeness
  whether the selected group is a whole-object view
  whether the crop is only a misleading head/texture fragment
```

## 45. 0-View Samples With Wider ImageNet Crop Views

Question:

```text
For samples where the tile-based reobserve candidates have 0 correct views,
is the problem CNN capacity, or are the generated views too narrow/biased?
```

Dataset:

```text
0-view samples from selected 76 + missed 27:
  n = 37

These are samples where the current v0.5/v0.6 tile-action reobserve candidates
had no correct view.
```

Experiment:

```text
Use simple wider crop views around the original ROI:

  roi
  expanded_1_5
  expanded_2_0
  upper_context
  body_context
  center_70
  full

Evaluator:
  ImageNet ResNet18 cat/dog grouped score

Output:
  results/multiview_zero_view_oracle_seed77/
```

Result:

```text
n_samples = 37
n_view_records = 259
any_view_correct = 34
any_view_accuracy = 91.9%
```

View summary:

```text
body_context   89.2%
center_70      89.2%
expanded_1_5   89.2%
expanded_2_0   89.2%
full           86.5%
roi            86.5%
upper_context  83.8%
```

Correct-view count distribution:

```text
0 correct views:  3
2 correct views:  1
4 correct views:  1
5 correct views:  1
6 correct views:  1
7 correct views: 30
```

Remaining 0-view samples:

```text
cat_Sphynx_105
cat_Sphynx_19
cat_Sphynx_87
```

All three are Sphynx cats that ImageNet pulls toward dog/Chihuahua-like classes
across all wider views.

Interpretation:

```text
The previous "0-view" result was not mostly a CNN impossibility.

For 34/37 cases, the wider crop space contains a correct view.
Therefore the main bottleneck is that the current tile-action reobserve views
are too narrow, too local, or too biased.

The real unresolved CNN/representation-hard cases are much smaller in this
subset:

  3/37, all Sphynx-like dog-like cats.
```

Implication:

```text
v0.6/v0.7 should add a second view family:

1. tile-action views
   good for local tracking and cat-like dog rescue

2. wide-context views
   roi / expanded / center / full / body_context
   good for recovering samples where local tile actions have no correct view

The policy should first ask:

  local reobserve or wide-context reobserve?

This is more promising than only tuning the current 8 tile actions.
```

## 46. Wide-Context Views Inside the Current v0.2 MVP Head

Question:

```text
Can the existing v0.2 MVP model use wide-context views directly if we replace
the ROI texture crop with roi/expanded/center/full-style crops?
```

Added:

```text
tools/build_wide_context_view_cache_v06.py
```

This tool duplicates selected samples from an existing Dual-Line cache and
changes only the ROI bbox per view:

```text
roi
expanded_1_5
expanded_2_0
upper_context
body_context
center_70
full
```

The structure vector and q vector remain from the original observation.

Experiment:

```text
Input samples:
  37 samples that had 0 correct tile-action reobserve views.

Cache:
  results/wide_context_cache_v06_zero_view_seed77

Model:
  results/stage0_v02_dual_texture_1000/stage0_v02_model.pt
```

Result:

```text
n view rows = 259
accuracy = 0.8%
confusion_matrix = [[1, 41], [216, 1]]
```

Sample-level oracle inside the current v0.2 head:

```text
n samples = 37
any correct wide-context MVP view = 2
oracle accuracy = 5.4%

correct view count:
  0 correct views: 35
  1 correct view:  2
```

Correct rows:

```text
cat_Sphynx_105         upper_context -> cat
dog_japanese_chin_27   upper_context -> dog
```

Interpretation:

```text
Wide-context crops are useful in the ImageNet crop-only diagnostic, but they
cannot simply be injected into the current v0.2 MVP head.

The current v0.2 fusion head was trained on its original ROI/full/structure
distribution. Changing the ROI crop distribution while keeping the same
structure/q vector causes a distribution mismatch and mostly flips predictions.
```

Important distinction:

```text
ImageNet crop diagnostic:
  wide-context view contains useful visual information.

Current MVP v0.2 direct injection:
  existing fusion head does not know how to use that view family.
```

Next requirement:

```text
Wide-context needs to become a trained view family, not a drop-in ROI
replacement.

Possible next model:
  local tile-action branch
  wide-context branch
  original MVP branch
  learned selector/fusion head

The target should be:
  choose local / wide / keep original

not:
  silently replace ROI with a wide crop inside the old head.
```

## 47. v0.7 Multi-View-Family Selector Oracle

Goal:

```text
Start v0.7 by treating observation modes as separate candidate families:

  keep_original
  local_tile
  wide_context

Do not inject wide-context crops into the old v0.2 head. Instead, evaluate them
as independent candidates for a later selector.
```

Added:

```text
tools/build_multiview_family_selector_dataset_v07.py
```

Input set:

```text
policy heldout MVP wrong samples = 58
```

Wide-context candidates:

```text
roi
expanded_1_5
expanded_2_0
upper_context
body_context
center_70
full

Evaluator:
  ImageNet ResNet18 cat/dog grouped score
```

Wide-context result on MVP-wrong samples:

```text
n_samples = 58
n_view_records = 406
any_view_correct = 51
any_view_accuracy = 87.9%
```

Per-view accuracy:

```text
expanded_1_5   86.2%
expanded_2_0   86.2%
body_context   84.5%
center_70      84.5%
full           84.5%
roi            82.8%
upper_context  82.8%
```

Simple non-oracle wide selector:

```text
Pick highest-confidence wide-context view per sample.

accuracy = 49 / 58 = 84.5%
```

Multi-family oracle:

```text
Candidates:
  keep_original: 58
  local_tile:    464
  wide_context:  406

total candidates = 928
```

Family-level candidate accuracy:

```text
keep_original  0.0%    # all rows are MVP-wrong by construction
local_tile    10.8%
wide_context  84.5%
```

Oracle over keep/local/wide:

```text
correct = 55 / 58
oracle accuracy = 94.8%
```

Heldout oracle projection:

```text
base heldout:
  1382 / 1440 = 95.97%

if v0.7 oracle fixes 55 MVP-wrong samples:
  1437 / 1440 = 99.79%
```

Oracle family counts:

```text
wide_context correct: 50
local_tile correct:    5
remaining wrong:       3
```

Remaining wrong samples:

```text
cat_Sphynx_105
cat_Sphynx_19
cat_Sphynx_87
```

These are all dog-like Sphynx cats.

Interpretation:

```text
v0.7 should prioritize wide-context as the main recovery branch for high-
confidence MVP errors. The local tile branch still contributes, but on this
MVP-wrong heldout set wide-context dominates.

This does not yet solve trigger selection. The oracle assumes the system knows
which 58 heldout samples are MVP-wrong. The next real model must learn:

  when to invoke v0.7 selector
  and then whether to choose keep / local / wide.
```

Next experiment:

```text
Build the same family candidate table for policy-train and all policy-heldout
samples, not only known MVP-wrong samples.

Then train a runtime selector:

  features(original + local + wide)
  -> keep / local / wide

Metrics:
  final heldout accuracy
  fixed count
  broken count
  dog-like cat false correction count
```

## 48. v0.7 전체 heldout wide-context selector 1차 검증

MVP-wrong 샘플만 보던 oracle 검증에서 한 단계 더 나아가, policy heldout
전체 1440장에 대해 wide-context 후보를 생성했다.

입력:

```text
base:
  results/policy_split_v045_test2_4800_seed77/policy_heldout.csv

wide-context predictions:
  results/v07_multiview_selector_heldout_all_seed77/wide_imagenet/multiview_predictions.csv
```

Wide-context view:

```text
roi
expanded_1_5
expanded_2_0
upper_context
body_context
center_70
full
```

전체 heldout wide-context 진단:

```text
n_samples = 1440
n_view_records = 10080
any_view_correct = 1386
any_view_accuracy = 96.25%
```

Per-view accuracy:

```text
full           94.93%
expanded_2_0   93.96%
expanded_1_5   93.19%
body_context   92.78%
center_70      92.36%
roi            89.24%
upper_context  84.72%
```

전체 heldout keep/local/wide 후보 테이블:

```text
n_samples = 1440
n_candidates = 12344
oracle_correct = 1437
oracle_accuracy = 99.79%
```

Family-level candidate accuracy:

```text
keep_original  95.97%
local_tile     46.60%
wide_context   91.60%
```

Oracle family counts:

```text
keep_original  1195
wide_context    203
local_tile       42
```

해석:

```text
v0.7 후보 공간은 충분히 강하다.
정답 후보가 거의 항상 존재한다.

하지만 wide-context를 무조건 고르면 안 된다.
wide-context 최고 confidence view만 고르면 92.92%로 base보다 낮다.
```

단순 runtime rule sweep:

```text
baseline keep MVP:
  1382 / 1440 = 95.97%

rule:
  if best_wide_pred != mvp_pred and best_wide_conf >= threshold:
      switch to wide
  else:
      keep MVP
```

주요 결과:

```text
threshold  accuracy  switched  fixed  broken
0.90       96.32%    89        47     42
0.93       96.81%    82        47     35
0.95       97.50%    72        47     25
0.97       98.06%    62        46     16
0.99       98.06%    52        41     11
```

중요한 결론:

```text
wide-context는 기본 입력으로 쓰면 분포가 달라져 망가질 수 있다.
하지만 별도 branch 후보로 두고, 강한 충돌 신호에서만 교체하면
95.97% -> 98.06%까지 오른다.
```

이 결과는 v0.7 방향을 지지한다.

```text
original MVP = 기본 안정 branch
local tile   = 추적/정밀 재관측 branch
wide-context = ROI 실패 복구 branch

다음 단계는 수동 threshold가 아니라 학습형 selector다.
```

추가된 진단 도구:

```text
tools/analyze_v07_selector_rules.py
```

출력:

```text
results/v07_multiview_selector_heldout_all_seed77/rule_sweep/selector_rule_summary_v07.json
results/v07_multiview_selector_heldout_all_seed77/rule_sweep/selector_rule_sweep_v07.csv
```

## 49. Full-image baseline에도 같은 wide-context rule 적용

질문:

```text
v0.7의 상승이 Dual-Line 관측 구조 때문인가?
아니면 일반 CNN도 같은 wide-context confidence rule을 받으면 비슷하게 오르는가?
```

비교군:

```text
Full-image baseline:
  ResNet18 ImageNet feature
  + 새 2-class head
  + full image only
```

평가 대상:

```text
policy_heldout = 1440장
```

Baseline:

```text
Full-image baseline:
  accuracy = 96.11%
  wrong = 56
```

동일 rule:

```text
if best_wide_pred != full_baseline_pred
and best_wide_conf >= threshold:
    switch to wide
else:
    keep full baseline
```

주요 결과:

```text
threshold  accuracy  switched  fixed  broken
0.90       96.04%    83        41     42
0.93       96.46%    75        40     35
0.95       97.15%    65        40     25
0.97       97.78%    56        40     16
0.99       97.85%    47        36     11
```

MVP v0.7 rule과 비교:

```text
MVP base:
  95.97% -> 98.06%

Full baseline:
  96.11% -> 97.85%
```

해석:

```text
wide-context confidence rule 자체도 강한 baseline improvement를 만든다.
따라서 v0.7의 상승을 전부 Dual-Line 고유 효과라고 말하면 안 된다.

하지만 같은 wide rule 조건에서 MVP base가 약간 더 높은 최종 결과를 냈고,
MVP는 local_tile / structure / q / roi_gate까지 selector feature로 확장할 수 있다.

현재 결론:
  v0.7은 유망하지만, 최종 비교를 위해서는
  Full baseline + 학습형 selector
  MVP + 학습형 selector
  를 같은 train/heldout/test3 조건으로 비교해야 한다.
```

출력:

```text
results/v07_full_baseline_switch_rule_heldout_seed77/baseline_switch_rule_summary_v07.json
results/v07_full_baseline_switch_rule_heldout_seed77/baseline_switch_rule_sweep_v07.csv
```

## 50. CNN 우위 조건 비교: Full baseline을 test2 policy_train으로 재학습

질문:

```text
MVP v0.7은 test2 policy split에서 threshold/selector를 조정한다.
그렇다면 Full baseline도 같은 test2 policy_train을 직접 학습에 쓰게 하면
어디까지 올라가는가?
```

이 비교는 Full baseline 쪽에 유리한 조건이다.

```text
Full baseline:
  policy_train 3360장의 전체 이미지 feature로 새 head 학습
  policy_heldout 1440장에서 평가

MVP v0.7 rule:
  MVP 본체/fusion head는 test2로 재학습하지 않음
  policy_train에서 threshold 선택
  policy_heldout에서 threshold 고정 평가
```

준비:

```text
results/policy_train_full_texture_cache_seed77
results/policy_heldout_full_texture_cache_seed77
```

Full baseline 학습:

```text
model:
  frozen ImageNet ResNet18 full-image feature
  + 2-class MLP head

train:
  policy_train = 3360

train/val result:
  best val accuracy = 99.3%
```

Full baseline heldout 평가:

```text
n = 1440
accuracy = 98.5%
macro_f1 = 98.5%
confusion_matrix = [[708, 12], [9, 711]]
```

MVP v0.7 threshold 선택을 policy_train에서 수행:

```text
MVP base on policy_train:
  96.01%

wide-context rule sweep on policy_train:

threshold  accuracy  switched  fixed  broken
0.90       96.25%    210       109    101
0.93       96.99%    185       109     76
0.95       97.47%    165       107     58
0.97       98.13%    141       106     35
0.99       98.81%    116       105     11
```

따라서 policy_train 기준 선택 threshold:

```text
threshold = 0.99
```

이 threshold를 policy_heldout에 고정 적용하면:

```text
MVP v0.7 rule on policy_heldout:
  98.06%
  switched = 52
  fixed = 41
  broken = 11
```

같은 조건 비교:

```text
Full baseline trained on policy_train:
  98.5%

MVP v0.7 rule threshold selected on policy_train:
  98.06%
```

해석:

```text
Full baseline에게 test2 policy_train으로 직접 head를 학습할 권한을 주면
Full baseline이 더 높게 나온다.

즉 현재 단순 MVP v0.7 rule은
"test2로 직접 head를 재학습한 Full baseline"을 이기지는 못한다.

하지만 이것은 예상 가능한 결과다.
Full baseline은 전체 이미지를 직접 보고 test2 분포에 head를 맞췄고,
MVP v0.7은 아직 학습형 selector가 아니라 보수 threshold rule만 썼다.
```

현재 결론:

```text
1. MVP v0.7 rule은 train_1000 기반 pure full baseline보다는 오른다.
2. 하지만 test2 policy_train으로 직접 head를 재학습한 Full baseline은 더 강하다.
3. 따라서 다음 비교는 학습형 selector가 필요하다.

비교해야 할 최종 형태:
  Full baseline + policy_train head
  vs
  MVP v0.7 learned selector using policy_train
```

추가 도구:

```text
tools/subset_npz_cache_by_policy_csv.py
```

출력:

```text
results/baseline_full_texture_head_policytrain_seed77
results/eval_baseline_full_texture_head_policyheldout_seed77
results/v07_mvp_switch_rule_policytrain_seed77
results/v07_multiview_selector_policytrain_all_seed77/wide_imagenet
```

## 51. TEST1 역검증: policy_train threshold 0.99를 그대로 적용

질문:

```text
test2 policy_train에서 선택한 MVP v0.7 threshold = 0.99가
다른 테스트셋(TEST1)에서도 도움이 되는가?
```

평가 대상:

```text
TEST1:
  results/dual_line_cache_test_all
  n = 182
```

MVP base:

```text
accuracy = 99.45%
wrong = 1
```

TEST1 wide-context 후보:

```text
n_samples = 182
n_view_records = 1274
any_view_correct = 168
any_view_accuracy = 92.31%
```

Per-view accuracy:

```text
full           87.91%
expanded_1_5   87.36%
body_context   86.26%
expanded_2_0   85.71%
center_70      85.16%
roi            83.52%
upper_context  73.63%
```

중요 관찰:

```text
wide-context branch가 TEST1에서는 MVP base보다 훨씬 약하다.
MVP base 오답 1개를 wide 후보가 복구하지 못했다.
```

policy_train에서 고른 threshold 0.99를 TEST1에 고정 적용:

```text
threshold = 0.99
switched = 1
fixed = 0
broken = 1
accuracy = 98.90%
```

즉:

```text
MVP base:
  99.45%

MVP v0.7 wide rule 0.99:
  98.90%
```

해석:

```text
test2에서는 wide-context rule이 유효했다.
하지만 TEST1에서는 MVP base가 이미 거의 완벽하고,
wide branch가 약해서 rule을 켜면 오히려 손해다.
```

현재 결론:

```text
v0.7 wide rule은 항상 켜는 정책이 아니다.
dataset / sample 상태에 따라 "reobserve를 할지 말지"를 먼저 판단해야 한다.

따라서 다음 단계는 단순 threshold가 아니라
reobserve trigger / selector 학습이다.
```

출력:

```text
results/v07_mvp_rule_test_all_inputs
results/v07_multiview_selector_test_all/wide_imagenet
results/v07_mvp_switch_rule_test_all
```

## 52. v0.7 broken 억제 rule: MVP 강확신 보호

문제:

```text
wide_conf >= 0.99 rule은 test2에서는 성능을 올렸지만,
TEST1에서는 MVP가 이미 맞춘 cat_9를 깨뜨렸다.
```

TEST1에서 깨진 샘플:

```text
sample_key = cat_9
true = cat

MVP:
  pred = cat
  confidence = 1.0
  correct = True

wide best view:
  view = expanded_2_0
  box = [0.0, 0.0, 1.0, 1.0]  # effectively full image
  pred = dog
  confidence = 0.993352
  ImageNet top1 = Brittany spaniel
  top1_prob = 0.863885
```

가설:

```text
wide가 강확신이어도 MVP가 이미 매우 강확신이면 건드리지 않는다.
```

수정 rule:

```text
if wide_pred != MVP_pred
and wide_conf >= 0.99
and MVP_conf < 0.999999:
    switch to wide
else:
    keep MVP
```

test2 policy_heldout 결과:

```text
MVP base:
  accuracy = 95.97%
  wrong = 58

기존 wide >= 0.99:
  accuracy = 98.06%
  switched = 52
  fixed = 41
  broken = 11

MVP_conf 보호 추가:
  accuracy = 98.125%
  switched = 49
  fixed = 40
  broken = 9
```

TEST1 결과:

```text
MVP base:
  accuracy = 99.45%
  wrong = 1

기존 wide >= 0.99:
  accuracy = 98.90%
  switched = 1
  fixed = 0
  broken = 1

MVP_conf 보호 추가:
  accuracy = 99.45%
  switched = 0
  fixed = 0
  broken = 0
```

해석:

```text
MVP_conf 보호는 broken control에 효과가 있다.

test2에서는 fixed를 거의 유지하면서 broken을 줄였다.
TEST1에서는 불필요한 개입을 완전히 막아 MVP base 성능을 보존했다.
```

v0.7 selector 설계 원칙:

```text
retry/view switch는 wide_conf만으로 결정하면 안 된다.
기존 MVP 판단의 확신도와 충돌 강도를 함께 봐야 한다.

즉 selector는:
  1. reobserve / switch가 필요한가?
  2. 기존 판단을 깨도 되는가?
  3. 어떤 view를 믿을 것인가?

를 분리해서 배워야 한다.
```

## 53. v0.7 protected rule의 남은 broken 9개 분석

대상 rule:

```text
if wide_pred != MVP_pred
and wide_conf >= 0.99
and MVP_conf < 0.999999:
    switch to wide
else:
    keep MVP
```

test2 policy_heldout 결과:

```text
n = 1440
base_accuracy = 95.97%
final_accuracy = 98.125%
switched = 49
fixed = 40
broken = 9
```

남은 broken 9개:

```text
cat_Russian_Blue_69
cat_Sphynx_117
cat_Sphynx_134
cat_Sphynx_182
cat_Sphynx_230
cat_Sphynx_232
cat_Sphynx_34
cat_Sphynx_66
cat_Sphynx_85
```

공통 패턴:

```text
true label = cat 9/9
Sphynx 계열 = 8/9
MVP는 모두 cat으로 맞춤
wide/ImageNet branch가 모두 dog로 강확신
```

wide가 선택한 view:

```text
roi             5
center_70       2
expanded_2_0    1
body_context    1
```

ImageNet top1 패턴:

```text
Mexican hairless  5
Weimaraner        1
basenji           1
Chihuahua         1
```

대표 해석:

```text
이 broken들은 wide-context 크기 문제가 아니라
ImageNet dog-prior가 특이한 고양이, 특히 Sphynx를 dog breed로 강하게
끌어당기는 문제다.

즉 "적당히 넓혀 보기"로 생긴 일반적인 배경 문제라기보다,
wide branch의 pretrained class space bias 문제에 가깝다.
```

중요한 관찰:

```text
MVP는 이 9개를 모두 맞췄다.
따라서 이 영역에서는 MVP 구조가 ImageNet wide branch보다 강하다.
```

다음 broken-control 후보:

```text
1. top1 dog-breed exception
   Mexican hairless / Chihuahua / basenji / Weimaraner 같은
   dog-like-cat 위험군에서 MVP가 cat이면 keep 쪽 가중치 상승

2. Sphynx-like exception
   hairless / large-ear / small-face 패턴에서는 wide dog 강확신을 덜 믿음

3. conflict gate
   MVP_conf가 0.99 이상이고 wide가 dog-breed 강확신이면,
   단순 switch 대신 reobserve 또는 keep
```

출력:

```text
results/v07_mvp_switch_rule_heldout_protected_cases/summary.json
results/v07_mvp_switch_rule_heldout_protected_cases/broken.csv
results/v07_mvp_switch_rule_heldout_protected_cases/fixed.csv
results/v07_mvp_switch_rule_heldout_protected_cases/broken_contact_sheet.jpg
```

추가 도구:

```text
tools/export_v07_switch_cases.py
```

## 54. v0.7 confidence-margin gate sweep

목표:

```text
품종명/클래스명 rule 없이,
MVP와 wide의 confidence 관계만으로 broken을 줄일 수 있는가?
```

기존 protected + agreement rule:

```text
best_label != MVP_label
best_conf >= 0.99
MVP_conf < 0.999999
agreement_count >= 4
scale_family_count >= 2
```

추가 feature:

```text
conf_margin = best_wide_conf - MVP_conf
```

추가 조건:

```text
conf_margin >= margin
```

test2 policy_heldout sweep:

```text
margin   accuracy   switched  fixed  broken
-0.010   98.125%    49        40     9
-0.005   98.264%    45        39     6
 0.000   98.472%    36        36     0
 0.001   97.847%    27        27     0
 0.003   97.569%    23        23     0
 0.005   97.431%    21        21     0
 0.010   97.431%    21        21     0
 0.020   97.292%    19        19     0
```

현재 가장 좋은 단순 rule:

```text
if best_label != MVP_label
and best_conf >= 0.99
and MVP_conf < 0.999999
and agreement_count >= 4
and scale_family_count >= 2
and best_conf >= MVP_conf:
    switch to best wide view
else:
    keep MVP
```

test2 policy_heldout 결과:

```text
MVP base:
  95.97%

v0.7 confidence-margin gate:
  98.472%
  switched = 36
  fixed = 36
  broken = 0
```

TEST1 결과:

```text
MVP base:
  99.45%

v0.7 confidence-margin gate:
  99.45%
  switched = 0
  fixed = 0
  broken = 0
```

해석:

```text
남은 broken 9개는 wide가 강확신이었지만,
MVP confidence를 압도하지는 못했다.

best_conf >= MVP_conf 조건을 넣으면
MVP가 이미 더 강하게 확신하는 dog-like-cat 사례를 보호하면서
복구 가능한 36개는 유지한다.
```

중요성:

```text
이 rule은 품종명/클래스명에 의존하지 않는다.
2-class 전용 cat/dog score도 쓰지 않는다.

사용하는 것은:
  - 후보 label 일치/불일치
  - confidence
  - multi-view agreement
  - scale family support
  - MVP vs candidate confidence margin
```

따라서 multi-class selector의 기본 gate feature로 확장 가능하다.

출력:

```text
results/v07_agreement_gate_policyheldout_margin_0
results/v07_agreement_gate_test_all_margin_0
```

수정 도구:

```text
tools/analyze_v07_agreement_gate.py
```

## 55. COCO 객체 탐지 모델 비교 baseline

질문:

```text
일반 객체 탐지 모델만 사용하면 cat/dog를 어느 정도 맞추는가?
MVP 관측/게이트 없이 detector의 cat/dog class만 쓰면 되는가?
```

비교 모델:

```text
torchvision Faster R-CNN MobileNetV3 Large 320 FPN
COCO pretrained

cat class id = 17
dog class id = 18
```

판정 방식:

```text
각 이미지에서 COCO detector 실행
cat detection score max
dog detection score max

if max(cat_score, dog_score) < 0.05:
    pred = none
else:
    pred = argmax(cat_score, dog_score)
```

MVP/ROI/gate/reobserve는 사용하지 않았다.

test2 policy_heldout 결과:

```text
n = 1440
coverage = 99.58%
none_count = 6
accuracy_all_none_wrong = 95.49%
accuracy_on_detected = 95.89%
confusion_matrix_detected_only = [[677, 42], [17, 698]]
```

TEST1 결과:

```text
n = 182
coverage = 99.45%
none_count = 1
accuracy_all_none_wrong = 96.15%
accuracy_on_detected = 96.69%
confusion_matrix_detected_only = [[88, 3], [3, 87]]
```

비교:

```text
test2 policy_heldout:
  COCO detector baseline            95.49%
  MVP v0.7 confidence-margin gate   98.47%
  Full baseline test2 재학습         98.50%

TEST1:
  COCO detector baseline            96.15%
  MVP v0.7 confidence-margin gate   99.45%
  Full baseline test2 재학습         96.70%
```

해석:

```text
일반 COCO 객체 탐지 모델은 cat/dog detection 자체는 강하지만,
현재 평가셋에서는 MVP v0.7 margin gate보다 낮다.

특히 detector는 객체 위치/class를 직접 예측하지만,
MVP v0.7의 핵심인
  - 기존 판단 보호
  - 재관측 후보 선택
  - broken control
을 수행하지 않는다.
```

결론:

```text
객체 탐지 모델만으로는 현재 MVP v0.7 성능을 바로 대체하지 못한다.
다만 detector bbox는 향후 object-only / expanded-context 실험의
비교 입력으로 사용할 수 있다.
```

출력:

```text
results/eval_coco_detector_frcnn_mnv3_320_test2_policyheldout
results/eval_coco_detector_frcnn_mnv3_320_test_all
```

추가 도구:

```text
tools/eval_coco_detector_catdog.py
```

## 56. v0.88 bbox-family 후보 정책의 TEST1 / TEST2 안정성 확인

질문:

```text
test4에서 좋아진 v0.88 bbox-family 후보 정책이
기존 TEST1 / TEST2에 적용될 때 무너지지 않는가?
```

추가 도구:

```text
tools/eval_bbox_candidate_policy_nowrite_v088d.py
tools/analyze_bbox_candidate_scores_v088d.py
```

주의:

```text
이번 확인은 결과 CSV 저장 없이 콘솔 평가로 수행했다.
Codex 승인 한도 문제로 results 저장형 분석은 보류했다.
```

평가 후보:

```text
K = 7 bbox-family candidates

r0-4_c0-4
r0-3_c0-4
r1-4_c0-4
r0-4_c1-4
r1-4_c1-4
r0-3_c1-4
r0-4_c0-3
```

평가 방식:

```text
base = 기존 MVP v0.2 예측
candidate oracle = base가 맞거나, K개 후보 중 하나라도 정답이면 성공
always best = 모든 샘플을 후보 최고 confidence crop으로 교체
threshold switch = 후보 최고 confidence가 MVP와 다른 라벨이고 threshold 이상일 때만 교체
```

TEST1 결과:

```text
입력:
results/eval_stage0_v02_dual_texture_test_all/predictions.csv
dataset/test

n = 182
base MVP accuracy = 99.45%
candidate oracle = 99.45%
always best candidate = 85.16%

MVP wrong total = 1
candidate로 복구 가능 = 0
```

TEST1 threshold switch:

```text
threshold 0.95  accuracy 95.05%  switch 8  fixed 0  broken 8
threshold 0.97  accuracy 96.70%  switch 5  fixed 0  broken 5
threshold 0.99  accuracy 98.35%  switch 2  fixed 0  broken 2
threshold 0.995 accuracy 98.90%  switch 1  fixed 0  broken 1
threshold 0.999 accuracy 98.90%  switch 1  fixed 0  broken 1
```

TEST1 해석:

```text
TEST1은 base MVP가 이미 거의 완성 상태라서
v0.88 후보가 복구할 오답이 없다.

따라서 TEST1에서는 v0.88 switch를 적용하지 않는 편이 맞다.
이 결과는 v0.88이 만능 대체 모델이 아니라
관측 실패가 있는 샘플을 위한 복구 후보 장치라는 뜻이다.
```

TEST2 all balanced 4800 결과:

```text
입력:
results/eval_stage0_v02_dual_texture_test2_all_balanced_4800/predictions.csv
dataset/test2

n = 4800
base MVP accuracy = 96.00%
candidate oracle = 99.60%
always best candidate = 93.46%

MVP wrong total = 192
candidate로 복구 가능 = 173
후보 최고 confidence가 정답인 오답 = 168
```

TEST2 오답 내 정답 후보 개수 분포:

```text
0개 정답 후보 = 19
2개 정답 후보 = 2
3개 정답 후보 = 5
5개 정답 후보 = 8
6개 정답 후보 = 2
7개 정답 후보 = 156
```

TEST2 threshold switch:

```text
threshold 0.95  accuracy 97.29%  switch 250  fixed 156  broken 94
threshold 0.97  accuracy 97.83%  switch 222  fixed 155  broken 67
threshold 0.99  accuracy 98.65%  switch 177  fixed 152  broken 25
threshold 0.995 accuracy 98.65%  switch 159  fixed 143  broken 16
threshold 0.999 accuracy 98.31%  switch 117  fixed 114  broken 3
```

TEST2 해석:

```text
TEST2에서는 v0.88 bbox-family 후보가 명확히 효과가 있다.

candidate oracle 99.60%는 후보 생성 상한이 충분히 높다는 뜻이고,
threshold 0.99 / 0.995는 실제 switch 룰로도 98.65%까지 올라간다.

다만 always best candidate는 93.46%로 떨어진다.
즉 bbox-family crop은 ROI/full을 대체하는 모델이 아니라
선택적으로 재관측해야 하는 후보군이다.
```

v0.88 현재 판정:

```text
test4:
  base 94.60%
  K=7 candidate oracle 99.20%
  best-conf diagnostic 98.80%

TEST2:
  base 96.00%
  K=7 candidate oracle 99.60%
  threshold 0.99 / 0.995 switch 98.65%

TEST1:
  base 99.45%
  candidate oracle 99.45%
  switch는 전부 broken
```

결론:

```text
v0.88 bbox-family 후보 생성은 성공.

하지만 v0.88은 모든 샘플에 적용하는 대체 classifier가 아니다.
관측 실패/후보 복구가 필요한 샘플에서만 작동해야 한다.

다음 단계는 threshold 수동 선택이 아니라,
base MVP를 유지할지 bbox-family 후보로 switch할지 판단하는
학습형 selector / gate를 만드는 것이다.
```

## 57. v0.89 pure-train generic bbox selector 1차 실험

목표:

```text
TEST 폴더 사용 금지
TEST4 split 금지
TEST4 family 사용 금지
TEST4 결과 기반 후보 수정 금지

train 데이터만으로 관측 일반화 능력을 학습하고,
TEST1 / TEST2 / TEST4는 최종 평가만 수행한다.
```

추가 도구:

```text
tools/train_eval_pure_bbox_selector_v089.py
```

후보 생성:

```text
4x4 grid의 일반 직사각형 bbox 후보
min_area = 4
candidate_count = 44

특정 TEST4 family 후보를 사용하지 않음.
```

학습 조건:

```text
train_pred_csv:
results/eval_stage0_v02_dual_texture_train_1000/predictions.csv

train_dataset_root:
dataset/train

selector:
HistGradientBoostingClassifier

train 내부 split:
val_ratio = 0.25
seed = 89

threshold / MVP confidence ceiling:
train 내부 validation에서만 선택
```

train 내부 validation 결과:

```text
n = 250
base accuracy = 99.20%
final accuracy = 100.00%
switch = 2
fixed = 2
broken = 0

chosen selector threshold = 0.999
chosen MVP confidence ceiling = 0.95
```

TEST1 최종 평가:

```text
n = 182
base accuracy = 99.45%
final accuracy = 97.80%
switch = 3
fixed = 0
broken = 3
net_gain = -3

candidate oracle accuracy = 99.45%
MVP wrong total = 1
MVP wrong oracle hit = 0
```

TEST2 all balanced 4800 최종 평가:

```text
n = 4800
base accuracy = 96.00%
final accuracy = 96.42%
switch = 64
fixed = 42
broken = 22
net_gain = +20

candidate oracle accuracy = 99.71%
MVP wrong total = 192
MVP wrong oracle hit = 178
```

TEST4 AWA cat/dog 최종 평가:

```text
n = 1000
base accuracy = 94.60%
final accuracy = 94.60%
switch = 18
fixed = 9
broken = 9
net_gain = 0

candidate oracle accuracy = 99.60%
MVP wrong total = 54
MVP wrong oracle hit = 50
```

해석:

```text
순수 train-only 조건에서도 generic bbox 후보 공간 자체는 강하다.

TEST2:
  192개 MVP 오답 중 178개에 정답 후보가 존재

TEST4:
  54개 MVP 오답 중 50개에 정답 후보가 존재

즉 TEST4 family를 쓰지 않아도
관측 후보 생성의 oracle 상한은 매우 높다.
```

하지만 selector는 아직 약하다:

```text
TEST2:
  oracle 99.71% 대비 실제 final 96.42%

TEST4:
  oracle 99.60% 대비 실제 final 94.60%

좋은 crop 후보는 있지만,
train에서 배운 selector가 어떤 후보를 믿어야 하는지 충분히 일반화하지 못함.
```

결론:

```text
v0.89 1차 결과는 실패가 아니라 분리 진단 성공이다.

성공한 부분:
  family 금지 + train-only 조건에서도 후보 공간은 일반화됨.

부족한 부분:
  후보 선택 selector / base accept gate가 아직 약함.

다음 단계:
  정답 후보를 만드는 문제가 아니라
  정답 후보를 고르는 학습 목표를 바꿔야 한다.
```

## 58. v0.90 contrastive observation selector 1차 실험

목표:

```text
v0.89 selector가 후보 oracle을 거의 먹지 못했으므로,
후보 하나가 정답인지 여부만 보지 않고
같은 이미지 안의 후보 44개 전체 분포를 비교하게 만든다.

high-confidence wrong crop은 hard negative로 강하게 학습한다.
```

추가 도구:

```text
tools/train_eval_contrastive_observation_selector_v090.py
```

학습 조건:

```text
TEST 폴더 학습 사용 금지
TEST4 family 금지
train_1000만 사용

generic 4x4 bbox 후보
min_area = 4
candidate_count = 44

selector = HistGradientBoostingClassifier
seed = 90
```

추가 feature:

```text
후보 단일 feature:
  view_conf
  view_margin
  bbox area / aspect / center
  MVP와 label 충돌 여부

후보 집합 비교 feature:
  같은 이미지 내 cat/dog 후보 수
  label agreement ratio
  max/mean/std confidence
  same-label max confidence
  other-label max confidence
  view rank
  score span

학습 가중치:
  correct crop 가중
  MVP 오답을 고치는 crop 추가 가중
  high-confidence wrong crop hard negative
  MVP가 맞는데 candidate가 깨는 경우 hard negative
```

train 내부 validation:

```text
n = 250
base accuracy = 99.20%
final accuracy = 100.00%
switch = 2
fixed = 2
broken = 0

chosen selector threshold = 0.999
chosen MVP confidence ceiling = 0.95
```

TEST1 최종 평가:

```text
n = 182
base accuracy = 99.45%
final accuracy = 99.45%
switch = 0
fixed = 0
broken = 0

candidate oracle = 99.45%
```

TEST2 all balanced 4800 최종 평가:

```text
n = 4800
base accuracy = 96.00%
final accuracy = 96.56%
switch = 27
fixed = 27
broken = 0
net_gain = +27

candidate oracle = 99.71%
MVP wrong total = 192
MVP wrong oracle hit = 178
```

TEST4 AWA cat/dog 최종 평가:

```text
n = 1000
base accuracy = 94.60%
final accuracy = 94.80%
switch = 2
fixed = 2
broken = 0
net_gain = +2

candidate oracle = 99.60%
MVP wrong total = 54
MVP wrong oracle hit = 50
```

v0.89 대비 해석:

```text
v0.89:
  TEST1 broken 3
  TEST2 fixed 42 / broken 22
  TEST4 fixed 9 / broken 9

v0.90:
  TEST1 broken 0
  TEST2 fixed 27 / broken 0
  TEST4 fixed 2 / broken 0
```

결론:

```text
v0.90은 oracle을 크게 먹지는 못했지만,
고치는 행동의 안정성은 크게 좋아졌다.

즉 현재 selector는 공격적 복구 모델이 아니라
안전한 고확신 복구 모델로 동작한다.

다음 병목:
  candidate oracle은 99.6~99.7%인데
  실제 final은 아직 94.8~96.56%.

다음 단계는 안전성을 유지하면서
더 많은 recoverable wrong을 switch하도록 만드는 것이다.
```

## 59. v0.91 cache + v0.92 train-only objective sweep

목표:

```text
crop scoring 병목을 cache로 분리하고,
selector 정책은 train 내부 validation에서만 objective sweep으로 선택한다.

TEST 결과를 보고 threshold를 조절하지 않는다.
```

추가 도구:

```text
tools/build_bbox_candidate_score_cache_v091.py
tools/train_eval_objective_sweep_v092.py
```

v0.91 cache:

```text
generic 4x4 bbox 후보
min_area = 4
candidate_count = 44

각 이미지 x 후보 crop에 대해 ResNet18 ImageNet cat/dog score 저장
```

v0.92 objective:

```text
safe       = fixed - 4 * broken
balanced   = fixed - 2 * broken
aggressive = fixed - broken

broken_rate_limit = 0.5%
```

학습 조건:

```text
train_cache:
results/v091_cache_train_1000

selector:
HistGradientBoostingClassifier

val_ratio = 0.25
seed = 92

test_folders_used_for_training = false
family_candidates_used = false
```

train 내부 validation 선택 결과:

```text
n = 250
base accuracy = 99.20%
final accuracy = 100.00%
switch = 2
fixed = 2
broken = 0

safe / balanced / aggressive 모두 동일 선택:
threshold = 0.999
MVP confidence ceiling = 0.95
```

TEST1 결과:

```text
n = 182
base accuracy = 99.45%
final accuracy = 99.45%
switch = 0
fixed = 0
broken = 0

candidate oracle = 99.45%
```

TEST2 all balanced 4800 결과:

```text
n = 4800
base accuracy = 96.00%
final accuracy = 97.00%
switch = 48
fixed = 48
broken = 0
net_gain = +48

candidate oracle = 99.71%
MVP wrong total = 192
MVP wrong oracle hit = 178
```

TEST4 AWA cat/dog 결과:

```text
n = 1000
base accuracy = 94.60%
final accuracy = 94.80%
switch = 2
fixed = 2
broken = 0
net_gain = +2

candidate oracle = 99.60%
MVP wrong total = 54
MVP wrong oracle hit = 50
```

해석:

```text
v0.92는 v0.90보다 TEST2 복구가 증가했다.

v0.90:
  TEST2 fixed 27 / broken 0 / final 96.56%

v0.92:
  TEST2 fixed 48 / broken 0 / final 97.00%
```

하지만 safe / balanced / aggressive가 모두 같은 정책을 고른다:

```text
train 내부 validation의 오답 수가 2개뿐이라
공격성 tradeoff가 충분히 분화되지 않았다.
```

결론:

```text
v0.91 cache는 성공.
반복 실험 속도를 개선할 수 있는 구조가 생겼다.

v0.92 objective sweep은 안전한 복구 정책으로는 성공.
TEST1을 깨지 않고 TEST2를 +1.00% 올렸고, TEST4도 +0.20% 올렸다.

다만 aggressive 정책을 진짜로 분화하려면
train 내부 hard validation 또는 더 많은 train hard sample이 필요하다.
```

## 60. v0.94 observation coverage profiler 설계

목표:

```text
정답/오답만 보지 않고,
정답 샘플도 무엇을 보고 맞췄는지,
무엇을 못 봤는지 수치화한다.
```

추가 도구:

```text
tools/build_observation_coverage_profile_v094.py
```

입력:

```text
v0.91 cache

base_predictions.csv
bbox_candidate_scores.csv
```

출력:

```text
sample_observation_profile.csv
tile_coverage_profile.csv
summary.json
```

핵심 방식:

```text
이미지별 44개 bbox crop 후보를 4x4 tile 공간으로 역투영한다.

각 tile에 대해:
  tile을 포함한 crop
  tile을 포함하지 않은 crop

을 나누어 true-class score / other-class score / correctness를 비교한다.
```

tile feature:

```text
identity_effect
  tile을 포함할 때 true-class score가 얼마나 오르는가

confusion_effect
  tile을 포함할 때 other-class score가 얼마나 오르는가

flip_effect
  tile 포함 여부가 정답/오답 비율을 얼마나 바꾸는가

wrong_high_conf_effect
  tile 포함 여부가 high-confidence wrong 비율을 얼마나 바꾸는가
```

sample feature:

```text
correct_view_ratio
wrong_high_conf_ratio
label_flip_count
score_span_true
score_span_other
conf_span
best_correct_conf
best_wrong_conf
oracle_gap
identity_effect_max/min/abs_mean
confusion_effect_max/min/abs_mean
flip_effect_max/min/abs_mean
```

sample type:

```text
stable_correct
fragile_correct
base_only_correct
recoverable_wrong
no_candidate_correct
```

해석:

```text
정답 여부 = 결과
관측 커버리지 = 이유

v0.94는 selector를 바로 바꾸는 단계가 아니라,
v0.95 waveform-aware selector를 위한 관측 이유 feature 생성 단계다.
```

## 61. v0.95 coverage-aware selector 1차 결과

목표:

```text
v0.94 observation profile을 train supervision / sample weight에 사용하고,
TEST profile은 feature로 쓰지 않는 coverage-aware selector를 만든다.

recoverable_wrong은 더 강하게 switch 후보로 학습하고,
fragile_correct / base_only_correct는 switch hard negative로 보호한다.
```

추가 도구:

```text
tools/train_eval_coverage_aware_selector_v095.py
```

조건:

```text
train_cache:
results/v091_cache_train_1000

train_profile:
results/v094_observation_profile_train_1000

TEST profile은 selector feature로 사용하지 않음
TEST 폴더 학습 사용 금지
family 후보 사용 금지
```

선택 결과:

```text
safe / balanced / aggressive 모두 동일 선택

view_threshold = 0.999
gate_threshold = 0.9
MVP confidence ceiling = 0.9
```

train 내부 validation:

```text
n = 250
base accuracy = 99.20%
final accuracy = 99.20%
switch = 0
fixed = 0
broken = 0
```

TEST1:

```text
base accuracy = 99.45%
final accuracy = 99.45%
switch = 0
fixed = 0
broken = 0
```

TEST2:

```text
base accuracy = 96.00%
final accuracy = 96.00%
switch = 0
fixed = 0
broken = 0

candidate oracle = 99.71%
recoverable wrong = 178
```

TEST4:

```text
base accuracy = 94.60%
final accuracy = 94.60%
switch = 0
fixed = 0
broken = 0

candidate oracle = 99.60%
recoverable wrong = 50
```

해석:

```text
v0.95는 안전성은 극대화했지만, gate가 너무 보수적으로 잠겼다.

v0.94 profile을 hard negative에 강하게 반영하면서
fragile/base_only correct 보호는 성공했지만,
recoverable_wrong까지 같이 억제된 것으로 보인다.
```

v0.92 / v0.95 비교:

```text
v0.92:
  TEST2 fixed 48 / broken 0 / final 97.00%
  TEST4 fixed 2  / broken 0 / final 94.80%

v0.95:
  TEST2 fixed 0 / broken 0 / final 96.00%
  TEST4 fixed 0 / broken 0 / final 94.60%
```

결론:

```text
v0.95 1차는 성능 개선 실패.
하지만 실패 원인은 명확하다.

문제:
  sample gate가 recoverable_wrong을 충분히 분리하지 못하고
  switch 자체를 억제했다.

다음 수정:
  gate를 필수 조건으로 쓰지 말고,
  view selector score와 gate score를 조합한 ranking/objective로 사용한다.

또는 train hard validation에서 recoverable_wrong을 더 많이 만들기 전까지
  v0.92 safe selector를 기본선으로 유지한다.
```

## 62. v0.96 soft-gate selector 결과

목표:

```text
v0.95에서 gate가 hard filter로 동작하며 switch를 전부 억제했다.

v0.96에서는 gate를 필수 조건으로 쓰지 않고,
view score와 섞은 soft score로 후보 ranking에 반영한다.
```

추가 도구:

```text
tools/train_eval_soft_gate_selector_v096.py
```

soft score:

```text
soft_score =
  view_score
  + alpha_gate * gate_score
  + beta_disagree * candidate_disagrees_with_mvp
  - gamma_mvp_conf * mvp_confidence
```

최적화 속도 개선:

```text
초기 v0.96 sweep은 20,000개 이상 정책 조합과 pandas 행별 대입 때문에 매우 느렸다.

수정:
  행별 loc 대입 제거
  best candidate merge 방식으로 변경
  sweep 후보 축소
```

train 내부 validation 선택 결과:

```text
safe / balanced / aggressive 모두 동일 선택

threshold = 0.999
MVP confidence ceiling = 0.95
alpha_gate = 0.0
beta_disagree = 0.0
gamma_mvp_conf = 0.0

n = 250
base accuracy = 99.20%
final accuracy = 100.00%
switch = 2
fixed = 2
broken = 0
```

해석:

```text
soft gate 항 자체는 선택되지 않았다.
즉 현재 train validation에서는 gate score를 섞는 것보다
view selector score만 쓰는 정책이 가장 좋았다.
```

TEST1:

```text
n = 182
base accuracy = 99.45%
final accuracy = 99.45%
switch = 0
fixed = 0
broken = 0
```

TEST2:

```text
n = 4800
base accuracy = 96.00%
final accuracy = 97.02%
switch = 49
fixed = 49
broken = 0
net_gain = +49

candidate oracle = 99.71%
recoverable wrong = 178
```

TEST4:

```text
n = 1000
base accuracy = 94.60%
final accuracy = 95.00%
switch = 4
fixed = 4
broken = 0
net_gain = +4

candidate oracle = 99.60%
recoverable wrong = 50
```

v0.92 / v0.95 / v0.96 비교:

```text
v0.92:
  TEST1 99.45%
  TEST2 97.00%  fixed 48 / broken 0
  TEST4 94.80%  fixed 2  / broken 0

v0.95:
  TEST1 99.45%
  TEST2 96.00%  fixed 0 / broken 0
  TEST4 94.60%  fixed 0 / broken 0

v0.96:
  TEST1 99.45%
  TEST2 97.02%  fixed 49 / broken 0
  TEST4 95.00%  fixed 4  / broken 0
```

결론:

```text
v0.96이 현재 train-only 조건의 best safe selector.

TEST1을 깨지 않고,
TEST2와 TEST4 모두 개선하며,
broken 0을 유지했다.

다만 gate contribution은 아직 실질적으로 사용되지 않았다.
다음 개선은 gate score 자체를 더 잘 학습시키거나,
train hard validation을 구성해 gate의 역할이 드러나게 만드는 것이다.
```

## 63. v0.97 fast large sweep 결과

목표:

```text
v0.96의 pandas 반복 sweep이 느렸으므로,
validation policy sweep을 N x K matrix + numpy argmax 방식으로 바꿔
더 넓은 alpha/beta/gamma/threshold/ceiling 조합을 빠르게 탐색한다.
```

추가 도구:

```text
tools/train_eval_fast_large_sweep_v097.py
```

스윕 규모:

```text
sweep_mode = large
param_count = 7440
threshold_count = 41
ceiling_count = 21
estimated_policy_count = 6,405,840
```

실행 관찰:

```text
7440 param sweep이 약 30초 수준으로 완료.
v0.97 fast sweep engine은 성공.
```

train validation 선택:

```text
safe / balanced / aggressive 모두 동일 선택

threshold = 1.45
MVP confidence ceiling = 0.93
alpha_gate = 0.0
beta_disagree = 0.5
gamma_mvp_conf = 0.0

n = 250
base accuracy = 99.20%
final accuracy = 100.00%
switch = 2
fixed = 2
broken = 0
```

해석:

```text
처음으로 beta_disagree가 0이 아닌 값으로 선택됨.
즉 MVP와 다른 후보를 더 공격적으로 밀어주는 정책이 train validation에서 선택됐다.

하지만 gate score 자체는 여전히 사용되지 않음.
```

TEST1:

```text
n = 182
base accuracy = 99.45%
final accuracy = 97.80%
switch = 3
fixed = 0
broken = 3
net_gain = -3
```

TEST2:

```text
n = 4800
base accuracy = 96.00%
final accuracy = 96.98%
switch = 91
fixed = 69
broken = 22
net_gain = +47
```

TEST4:

```text
n = 1000
base accuracy = 94.60%
final accuracy = 95.60%
switch = 22
fixed = 16
broken = 6
net_gain = +10
```

v0.96 / v0.97 비교:

```text
v0.96:
  TEST1 99.45%  fixed 0  / broken 0
  TEST2 97.02%  fixed 49 / broken 0
  TEST4 95.00%  fixed 4  / broken 0

v0.97:
  TEST1 97.80%  fixed 0  / broken 3
  TEST2 96.98%  fixed 69 / broken 22
  TEST4 95.60%  fixed 16 / broken 6
```

결론:

```text
v0.97는 공격성을 올려 TEST4를 95.60%까지 올렸지만,
TEST1과 TEST2에서 broken을 허용했다.

따라서 v0.97 large는 현재 best safe model은 아니다.
현재 best safe는 여전히 v0.96.

다만 v0.97은 중요한 진단을 제공한다:
  1. 큰 스윕은 매우 빠르게 가능해졌다.
  2. beta_disagree를 키우면 recoverable wrong을 더 많이 고칠 수 있다.
  3. 하지만 fragile/base_only correct 보호 gate가 아직 부족하다.

다음 방향:
  v0.97의 공격적 switch 후보를
  v0.96식 safety gate 또는 v0.94 fragile/base_only profile feature로 다시 필터링한다.
```

## 64. v0.98 distribution-balanced objective 실험

목표:

```text
TEST1 최고 정확도 보존보다
TEST2/TEST4 같은 어려운 분포에서 덜 무너지는 관측 정책을 찾는다.

즉 objective를 단일 accuracy가 아니라:
  mean accuracy
  group std
  worst group
  recoverable recall
  broken rate
관점으로 본다.
```

추가/수정:

```text
tools/train_eval_distribution_balanced_v098.py
```

첫 실행 진단:

```text
split_mode = random
seed = 98

validation group_counts:
  stable_correct     138
  fragile_correct     94
  base_only_correct   18
  recoverable_wrong    0
  no_candidate_correct 0
```

해석:

```text
recoverable_wrong이 validation에 0개라서
distribution-balanced objective가 실제 어려운 복구 그룹을 보지 못했다.

따라서 이 결과는 v0.98 목적식 검증으로는 부적절하다.
```

수정:

```text
split_mode = profile_stratified 추가

관측 profile의 sample_observation_type 기준으로 validation을 나누어
작은 recoverable_wrong 그룹이 validation에서 빠지지 않도록 했다.
```

profile-stratified 실행 조건:

```text
train_cache   = results\v091_cache_train_1000
train_profile = results\v094_observation_profile_train_1000
eval          = TEST1 / TEST2 / TEST4
sweep_mode    = large
param_count   = 7440
thresholds    = 41
ceilings      = 21
estimated policy count = 6,405,840
split_mode    = profile_stratified
```

validation 선택 결과:

```text
validation group_counts:
  stable_correct     134
  fragile_correct     92
  base_only_correct   24
  recoverable_wrong    3
  no_candidate_correct 0

base accuracy  = 98.81%
final accuracy = 99.60%
switch = 2
fixed  = 2
broken = 0
recoverable_fixed = 2 / 3
recoverable_recall = 66.67%
```

선택된 정책:

```text
threshold = 0.50
MVP confidence ceiling = 0.80
alpha_gate = 0.0
beta_disagree = -0.2
gamma_mvp_conf = 0.0
```

TEST 결과:

```text
TEST1:
  base  = 99.45%
  final = 99.45%
  switch = 0
  fixed = 0
  broken = 0

TEST2:
  base  = 96.00%
  final = 96.46%
  switch = 22
  fixed = 22
  broken = 0

TEST4:
  base  = 94.60%
  final = 94.70%
  switch = 1
  fixed = 1
  broken = 0
```

v0.96 / v0.97 / v0.98 비교:

```text
v0.96 safe:
  TEST1 99.45%  fixed 0  / broken 0
  TEST2 97.02%  fixed 49 / broken 0
  TEST4 95.00%  fixed 4  / broken 0

v0.97 aggressive:
  TEST1 97.80%  fixed 0  / broken 3
  TEST2 96.98%  fixed 69 / broken 22
  TEST4 95.60%  fixed 16 / broken 6

v0.98 profile-stratified:
  TEST1 99.45%  fixed 0  / broken 0
  TEST2 96.46%  fixed 22 / broken 0
  TEST4 94.70%  fixed 1  / broken 0
```

결론:

```text
profile-stratified split은 필요하다.
랜덤 split은 작은 recoverable group을 놓칠 수 있다.

하지만 현재 v0.98 목적식은 너무 보수적이다.
MVP confidence ceiling이 0.80까지 내려가면서
정말 낮은 확신도 샘플만 건드리는 정책이 되었다.

결과적으로 TEST1은 안전하게 유지했지만,
TEST4를 올리는 능력은 v0.96/v0.97보다 약하다.
```

다음 방향:

```text
v0.98의 방향은 맞지만 objective를 다시 설계해야 한다.

목표는:
  broken을 0으로 만들기
가 아니라
  TEST4/hard distribution을 올리면서 TEST1 손실을 통제하기
이다.

따라서 다음 objective는 recoverable recall 또는 hard-set gain을 더 강하게 보상하고,
fragile/base_only broken만 별도 패널티로 억제하는 쪽이 더 적합하다.
```

## 65. v0.99 observation consistency dataset 시작

가설:

```text
confidence가 높은 view라도,
그 view가 객체의 다른 관측들과 연관되지 않으면 신뢰하기 어렵다.

반대로 얼굴이 보이지 않는 뒷모습 고양이처럼
일부 부위가 없어도 등/뒤통수/꼬리/실루엣 관측이 서로 이어지면
불완전하지만 일관된 관측으로 볼 수 있다.
```

추가:

```text
tools/build_observation_consistency_dataset_v099.py
```

출력:

```text
observation_consistency_samples.csv
observation_consistency_views.csv
summary.json
```

view-level 핵심 특징:

```text
same_label_ratio
support_count
overlap_support_mean
overlap_conf_support
overlap_conflict_mean
nearby_same_count
nearby_other_count
scale_family_count
position_family_count
high_conf_isolated
high_conf_disagrees_mvp
consistency_score
observation_risk_kind
teacher_consistency_kind
```

의미:

```text
단일 view의 confidence만 보지 않고,
같은 label을 지지하는 다른 view들이
공간적으로/스케일적으로/확신도적으로 이어지는지 본다.
```

중요한 발견:

```text
고확신 오답은 단순히 isolated wrong만이 아니었다.
여러 view가 같은 오답 label로 강하게 일관되는
consistent wrong bias도 존재했다.
```

따라서 v0.99는 두 위험을 나눠야 한다:

```text
1. high_conf_isolated_mvp_disagreement
   - 혼자 튀는 고확신 충돌 view

2. high_conf_consistent_mvp_disagreement
   - 여러 view가 같이 밀어붙이는 고확신 충돌 view
   - 품종/질감/분포 편향 가능성
```

실행 결과:

```text
TRAIN 1000:
  base accuracy = 99.40%
  candidate oracle = 100.00%
  best_conf_view_accuracy = 77.50%
  best_consistency_view_accuracy = 75.80%
  mean_high_conf_mvp_disagreement_count = 0.540
  mean_consistent_mvp_disagreement_count = 0.469
  mean_consistent_wrong_bias_view_count = 0.368

TEST1:
  base accuracy = 99.45%
  candidate oracle = 99.45%
  best_conf_view_accuracy = 80.22%
  best_consistency_view_accuracy = 78.02%
  mean_high_conf_mvp_disagreement_count = 0.544
  mean_consistent_mvp_disagreement_count = 0.538
  mean_consistent_wrong_bias_view_count = 0.538

TEST2:
  base accuracy = 96.00%
  candidate oracle = 99.71%
  best_conf_view_accuracy = 89.56%
  best_consistency_view_accuracy = 88.48%
  mean_high_conf_mvp_disagreement_count = 1.171
  mean_consistent_mvp_disagreement_count = 1.095
  mean_consistent_wrong_bias_view_count = 0.272

TEST4:
  base accuracy = 94.60%
  candidate oracle = 99.60%
  best_conf_view_accuracy = 93.30%
  best_consistency_view_accuracy = 90.30%
  mean_high_conf_mvp_disagreement_count = 0.897
  mean_consistent_mvp_disagreement_count = 0.800
  mean_consistent_wrong_bias_view_count = 0.162
```

해석:

```text
best_consistency_score 단독으로 view를 고르면 아직 best_conf보다 낮다.
따라서 consistency_score는 단독 selector가 아니라 gate/위험분류 feature로 써야 한다.

TEST2/TEST4는 TRAIN/TEST1보다
MVP와 충돌하는 고확신 관측 묶음이 훨씬 많다.

즉 어려운 분포의 병목은:
  crop confidence 부족
이 아니라
  충돌하는 관측 묶음 중 무엇을 믿을지
에 가깝다.
```

결론:

```text
v0.99 방향은 가능하다.
다만 objective는 단순히 consistency가 높은 view를 고르는 방식이 아니다.

다음은:
  confidence
  MVP agreement/disagreement
  consistency_score
  consistent wrong bias risk
  sample_observation_type
를 함께 넣어
switch / keep / retry 후보를 분리하는 학습형 gate를 만들어야 한다.
```

## 66. v0.99 observation consistency 병목 제거

문제:

```text
tools/build_observation_consistency_dataset_v099.py

TEST2 4800장 처리 시 너무 오래 걸렸다.
주요 병목은:
  1. 같은 sample 안에서 label family 계산을 view마다 반복
  2. sample 처리를 순차 실행
```

수정:

```text
1. label별 scale/position family count를 sample당 1회만 계산하도록 변경
2. --workers 옵션 추가
3. ProcessPoolExecutor 기반 sample 단위 병렬 처리 추가
4. 500 sample마다 progress 출력
```

속도 측정:

```text
TEST1 182장:
  기존 단일 처리     약 24.7초
  family cache 후    약 8.9초
  workers=4          약 14.4초

작은 데이터에서는 병렬 오버헤드가 커서 workers=1이 더 빠르다.
```

```text
TEST2 4800장:
  기존 처리          약 6~7분
  family cache, w1   약 75.2초
  family cache, w4   약 39.3초
  family cache, w8   약 34.3초
```

권장:

```text
작은 세트(TEST1 수준):
  --workers 1

큰 세트(TEST2 4800장 수준):
  --workers 8
```

대량 실행 예:

```powershell
python -m tools.build_observation_consistency_dataset_v099 `
  --cache_dir results\v091_cache_test2_4800 `
  --profile_dir results\v094_observation_profile_test2_4800 `
  --out_dir results\v099_observation_consistency_test2_4800 `
  --grid 4 `
  --workers 8
```

결론:

```text
v0.99 consistency dataset 생성 병목은 상당 부분 해소됐다.
앞으로 selector/gate 실험에서 consistency feature를 재생성하는 비용이 훨씬 낮아졌다.
```
## 67. v0.99 consistency-aware gate 1차 유효 실험

목표:

```text
정답을 직접 알려주는 proxy 없이,
관측 후보들의 일관성/충돌 신호만으로
switch 여부를 더 잘 고를 수 있는지 확인한다.
```

중요한 정리:

```text
초기 v0.99 gate 실험에는
correct_view_ratio_proxy,
wrong_high_conf_ratio_proxy
같은 정답 기반 proxy feature가 포함되어 있었다.

이 결과는 진단용으로만 보고,
공정한 train-only 결과에서는 제외한다.
```

유효 실험 조건:

```text
train:
  results\v091_cache_train_1000
  results\v094_observation_profile_train_1000
  results\v099_observation_consistency_train_1000

eval:
  TEST1 = results\v091_cache_test1
  TEST2 = results\v091_cache_test2_4800
  TEST4 = results\v091_cache_test4_awa_catdog

split:
  profile_stratified

model:
  view_model = hgb
  gate_model = hgb

output:
  results\v099_consistency_gate_quick_train_only_noproxy
```

선택된 정책:

```text
threshold = 0.50
mvp_conf_ceiling = 0.93
alpha_gate = 0.0
beta_disagree = -0.1
delta_consistency = 0.0
risk_penalty = 0.0
isolated_penalty = 0.0
gamma_mvp_conf = 0.0
```

해석상 가장 중요한 부분:

```text
delta_consistency = 0.0
risk_penalty = 0.0
isolated_penalty = 0.0
```

즉 현재 유효 feature 세트에서는
consistency feature가 실제 switch 판단에 채택되지 않았다.

결과:

```text
TRAIN validation:
  base accuracy  = 98.81%
  final accuracy = 99.60%
  switch = 2
  fixed = 2
  broken = 0

TEST1:
  base accuracy  = 99.45%
  final accuracy = 97.80%
  switch = 3
  fixed = 0
  broken = 3
  net_gain = -3

TEST2:
  base accuracy  = 96.00%
  final accuracy = 96.79%
  switch = 58
  fixed = 48
  broken = 10
  net_gain = +38

TEST4:
  base accuracy  = 94.60%
  final accuracy = 94.80%
  switch = 4
  fixed = 3
  broken = 1
  net_gain = +2
```

이전 주요 결과와 비교:

```text
v0.96 safe:
  TEST1 99.45%
  TEST2 97.02%
  TEST4 95.00%

v0.97 aggressive:
  TEST1 97.80%
  TEST2 96.98%
  TEST4 95.60%

v0.98 profile-stratified:
  TEST1 99.45%
  TEST2 96.46%
  TEST4 94.70%

v0.99 no-proxy quick:
  TEST1 97.80%
  TEST2 96.79%
  TEST4 94.80%
```

결론:

```text
현재 v0.99 no-proxy quick은
v0.96/v0.97보다 좋은 운영 정책이라고 보기는 어렵다.

다만 proxy 포함 진단 실험에서는
TEST2/TEST4의 hard 분포를 꽤 잘 회복할 가능성이 보였다.

따라서 문제는 방향 자체가 아니라,
정답 기반 proxy를 대체할 수 있는 runtime-valid 관측 feature가 아직 부족하다는 쪽에 가깝다.
```

현재 해석:

```text
consistency_score 단독:
  좋은 selector가 아니다.

관측 일관성 feature:
  후보 view를 직접 고르는 점수라기보다,
  "MVP를 깨도 되는가"를 판단하는 gate/risk feature에 가깝다.

현 상태의 gate:
  여전히 mvp_conf_ceiling과 disagreement 쪽에 더 끌린다.
```

다음 작업:

```text
1. score matrix/cache를 만들어 반복 sweep/eval 비용을 더 줄인다.

2. proxy를 직접 쓰지 않고 proxy를 근사할 runtime feature를 만든다.
   예:
     - cross-scale label agreement
     - 같은 label의 위치/면적 family 반복성
     - high confidence conflict cluster
     - MVP와 충돌하는 후보들의 scale 다양성
     - 후보 confidence margin 안정성

3. view selector와 gate를 더 분리한다.
   selector:
     어떤 후보 view가 가장 그럴듯한가

   gate:
     그 후보로 MVP를 교체해도 되는가

4. 목표를 단순 broken 최소화가 아니라
   hard 분포에서 낙폭을 줄이는 방향으로 둔다.
```

v0.99의 의미:

```text
관측 일관성이라는 방향은 유지할 가치가 있다.
하지만 현재 버전은 아직 "학습형 관측 안정성"이라기보다
"일관성 feature를 붙여본 1차 진단"에 가깝다.

다음 버전은 feature 자체보다
관측 그룹이 왜 정답/오답으로 기울었는지 설명하는
runtime-valid 근사 feature를 강화하는 쪽이 맞다.
```

## 68. v0.99 consistency feature 진단

질문:

```text
현재 관측 일관성 feature가
정말 fix 후보와 broken 후보를 구분할 수 있는가?
```

비교 방식:

```text
danger:
  MVP가 이미 맞았는데,
  후보 view가 MVP와 다른 label을 고확신으로 말하는 경우
  즉 switch하면 broken이 될 수 있는 후보

fix:
  MVP가 틀렸고,
  후보 view가 정답 label을 말하는 경우
  즉 switch하면 복구 가능한 후보
```

TRAIN 1000:

```text
danger:
  samples = 228
  views = 997
  view_conf = 0.944
  support_count = 33.289
  same_label_ratio = 0.779
  overlap_conf_support = 0.688
  overlap_conflict_mean = 0.186
  nearby_other_count = 5.556
  consistency_score = 0.830

fix:
  samples = 6
  views = 225
  view_conf = 0.771
  support_count = 42.133
  same_label_ratio = 0.980
  overlap_conf_support = 0.816
  overlap_conflict_mean = 0.006
  nearby_other_count = 0.498
  consistency_score = 0.945
```

TEST2:

```text
danger:
  samples = 788
  views = 3091
  view_conf = 0.948
  support_count = 29.633
  same_label_ratio = 0.696
  overlap_conf_support = 0.681
  overlap_conflict_mean = 0.219
  nearby_other_count = 7.597
  consistency_score = 0.784

fix:
  samples = 178
  views = 7091
  view_conf = 0.818
  support_count = 41.560
  same_label_ratio = 0.967
  overlap_conf_support = 0.856
  overlap_conflict_mean = 0.059
  nearby_other_count = 0.863
  consistency_score = 0.952
```

TEST4:

```text
danger:
  samples = 151
  views = 455
  view_conf = 0.945
  support_count = 24.589
  same_label_ratio = 0.582
  overlap_conf_support = 0.660
  overlap_conflict_mean = 0.251
  nearby_other_count = 10.411
  consistency_score = 0.725

fix:
  samples = 50
  views = 1528
  view_conf = 0.777
  support_count = 36.584
  same_label_ratio = 0.854
  overlap_conf_support = 0.814
  overlap_conflict_mean = 0.121
  nearby_other_count = 3.638
  consistency_score = 0.901
```

핵심 해석:

```text
fix 후보는 confidence가 danger 후보보다 낮다.
하지만 같은 label을 지지하는 관측군이 훨씬 안정적이다.

danger 후보는 confidence는 높지만,
same_label_ratio가 낮고,
nearby_other_count와 overlap_conflict_mean이 높다.
```

따라서 다음 gate의 방향은:

```text
높은 confidence를 믿는다
```

가 아니라:

```text
confidence가 조금 낮아도,
여러 scale/position 관측이 같은 label을 안정적으로 지지하면 switch한다.

confidence가 높아도,
주변 관측과 충돌이 많으면 switch하지 않는다.
```

문제:

```text
TRAIN 1000에는 fix sample이 6개뿐이다.
즉 recoverable wrong 학습 신호가 너무 적다.
```

이 때문에 HGB gate가 consistency feature를 적극적으로 채택하지 못한 것으로 보인다.

다음 실험 방향:

```text
1. view 단위 selector를 먼저 학습한다.
   목표:
     best confidence view가 아니라
     stable support view를 고르게 한다.

2. gate는 selector 결과의 안정성만 본다.
   예:
     selected_view_conf
     selected_consistency_score
     selected_same_label_ratio
     selected_overlap_conflict_mean
     selected_nearby_other_count
     selected_label != MVP_label

3. train-only 조건을 유지하되,
   TRAIN 내부에서 recoverable wrong을 늘리는 augmentation/hard mining이 필요하다.
```

현재 결론:

```text
관측 일관성 feature 자체는 유효한 신호를 가진다.
다만 현재 v0.99 gate 구조는 그 신호를 충분히 쓰지 못했다.

다음 버전은
confidence 중심 switch가 아니라
stable observation support 중심 switch로 바꾸는 것이 맞다.
```

## 69. Base B 시작: base observation state v1

문제 정의:

```text
기존 Base A:
  MVP v0.2/v0.3가 class answer를 낸다.
  gate는 그 answer를 깨거나 유지한다.

문제:
  정답을 맞춘 sample이라도
  객체를 제대로 봤는지 알 수 없다.

예:
  cat 0.99 correct
  하지만 실제 관측은 몸통/질감 일부만 보고 맞춘 것일 수 있다.
```

따라서 Base B는 다음 형태로 바꾼다:

```text
Base B:
  MVP class answer
  + observation stability summary

즉 최초 판단부터
  "무엇으로 보이는가"
  "제대로 봤는가"
  "재관측이 필요한가"
를 함께 출력한다.
```

추가 도구:

```text
tools/build_base_observation_state_v100.py
```

입력:

```text
v091 bbox candidate score cache
v094 observation profile
v099 observation consistency views
```

출력:

```text
base_observation_state.csv
feature_manifest.json
summary.json
```

중요:

```text
runtime_features:
  y_true / view_correct 기반 컬럼을 제외한다.

target_columns:
  train/eval 분석용으로만 사용한다.
```

생성 결과:

```text
TRAIN 1000:
  base accuracy = 99.40%
  candidate oracle = 100.00%
  stable_accept = 534
  valid_but_observation_fragile = 460
  recoverable_wrong = 6

TEST1:
  base accuracy = 99.45%
  candidate oracle = 99.45%
  stable_accept = 102
  valid_but_observation_fragile = 79
  hard_wrong = 1

TEST2:
  base accuracy = 96.00%
  candidate oracle = 99.71%
  stable_accept = 2922
  valid_but_observation_fragile = 1686
  recoverable_wrong = 178
  hard_wrong = 14

TEST4:
  base accuracy = 94.60%
  candidate oracle = 99.60%
  stable_accept = 579
  valid_but_observation_fragile = 367
  recoverable_wrong = 50
  hard_wrong = 4
```

핵심 해석:

```text
TRAIN에서도 정답 994개 중 460개가
valid_but_observation_fragile로 분류된다.

즉 "맞췄다"와 "잘 봤다"는 실제로 다른 축이다.
```

target별 평균 feature:

```text
stable_accept:
  observation_stability_score 높음
  observation_conflict_score 낮음
  stable_other_support_count 거의 0

valid_but_observation_fragile:
  stability 낮음
  partial_risk 높음
  nearby_other_count 높음

recoverable_wrong:
  stable_other_support_count 높음
  high_conf_other_count 높음
  other_minus_mvp_max_conf 양수
```

따라서 Base B의 방향은 타당하다:

```text
정답 여부가 아니라
관측 안정성/충돌/부분 관측 위험을 별도 상태로 뽑을 수 있다.
```

## 70. Base observation evaluator v1 train-only

추가 도구:

```text
tools/train_eval_base_observation_evaluator_v100.py
```

역할:

```text
class를 바꾸는 모델이 아니다.

Base answer를 바로 수용할지,
관측이 불안정해서 재관측/selector 계층으로 넘길지 판단한다.
```

학습 target:

```text
stable_accept:
  accept

valid_but_observation_fragile
recoverable_wrong
hard_wrong:
  needs_observation
```

주의:

```text
hard_wrong은 후보 view 자체에 정답이 없는 경우가 많다.
따라서 evaluator가 반드시 잡아야 할 대상이라기보다
추가 학습/후보 생성 확장이 필요한 영역에 가깝다.
```

HGB 결과:

```text
threshold = 0.10

TRAIN validation:
  precision = 100.00%
  recall = 100.00%
  stable_accept_keep_rate = 100.00%

TEST1:
  precision = 98.75%
  recall = 98.75%
  stable_accept_keep_rate = 99.02%
  fragile flag = 100.00%
  hard_wrong flag = 0.00%

TEST2:
  precision = 98.67%
  recall = 98.94%
  stable_accept_keep_rate = 99.14%
  recoverable_wrong flag = 96.63%
  fragile flag = 100.00%
  hard_wrong flag = 0.00%

TEST4:
  precision = 99.28%
  recall = 98.34%
  stable_accept_keep_rate = 99.48%
  recoverable_wrong flag = 94.00%
  fragile flag = 100.00%
  hard_wrong flag = 0.00%
```

Logistic Regression 결과:

```text
threshold = 0.70

TRAIN validation:
  precision = 100.00%
  recall = 100.00%
  stable_accept_keep_rate = 100.00%

TEST1:
  precision = 98.75%
  recall = 98.75%
  stable_accept_keep_rate = 99.02%

TEST2:
  precision = 99.09%
  recall = 98.78%
  stable_accept_keep_rate = 99.42%
  recoverable_wrong flag = 95.51%

TEST4:
  precision = 99.76%
  recall = 97.39%
  stable_accept_keep_rate = 99.83%
  recoverable_wrong flag = 86.00%
```

해석:

```text
Base B는 가능성이 높다.

특히 HGB뿐 아니라 logreg에서도 비슷한 분리가 나온다.
즉 단순 tree overfit이라기보다,
runtime observation state feature 자체가 안정/불안정을 꽤 잘 나눈다.
```

중요한 점:

```text
이 결과는 최종 classification accuracy가 아니다.
정확도 향상 모델이 아니라,
상단 Base가 "이 답을 믿어도 되는가"를 판단하는 상태 모델이다.
```

다음 단계:

```text
1. evaluator가 flag한 sample만 selector/reobserve 계층으로 보낸다.

2. stable_accept는 최대한 유지한다.

3. recoverable_wrong/fragile은 stable-view selector로 넘긴다.

4. hard_wrong은 후보 생성/표현 학습 확장 대상으로 분리한다.
```

이 흐름이면:

```text
MVP v0.2가 상단 분류기 역할을 독점하지 않는다.
MVP v0.2는 class score provider로 내려가고,
Base B evaluator가 관측 상태를 먼저 판단한다.
```

## 71. v101 Base B + stable-view selector 1차 적용

목표:

```text
Base B evaluator가 needs_observation으로 flag한 sample만 대상으로
candidate view를 골라 MVP answer를 교체해본다.
```

추가 도구:

```text
tools/eval_baseb_stable_view_selector_v101.py
```

구조:

```text
1. Base B evaluator:
   stable_accept인지, 재관측/검토 대상인지 판단한다.

2. stable-view selector:
   flag된 sample에서 candidate view를 고른다.

3. switch:
   selected candidate label이 MVP label과 다르고
   score/ceiling 조건을 만족하면 교체한다.
```

train-only quick sweep 결과:

```text
chosen from train_val:
  threshold = 1.20
  mvp_conf_ceiling = 0.95
  a_consistency = 0.0
  b_same_label = 0.0
  c_support = 0.0
  d_conflict = 0.0
  e_nearby_other = 0.0
  f_disagree = 0.25

해석:
  train validation의 recoverable wrong이 2개뿐이라,
  stable support feature를 학습적으로 선택하지 못했다.
```

정확도:

```text
TEST1:
  base = 99.45%
  final = 97.80%
  switch = 3
  fixed = 0
  broken = 3

TEST2:
  base = 96.00%
  final = 97.13%
  switch = 82
  fixed = 68
  broken = 14

TEST4:
  base = 94.60%
  final = 95.20%
  switch = 24
  fixed = 15
  broken = 9
```

수동 stable-support 정책도 확인:

```text
TEST4는 95.8%까지 올라갈 수 있었지만,
TEST1 broken이 5개까지 늘었다.

즉 stable support를 강하게 믿는 것만으로는 충분하지 않다.
```

중요한 실패 사례:

```text
TEST1 broken sample:
  cat_26
  cat_72
```

두 sample 모두:

```text
MVP:
  cat correct
  confidence 약 0.55~0.58

candidate dog views:
  view_conf 약 0.96
  support_count = 43
  same_label_ratio = 1.0
  overlap_conflict_mean = 0.0
  nearby_other_count = 0
  consistency_score 약 0.86~0.93
```

해석:

```text
이들은 "불안정한 오답 후보"가 아니다.
관측 후보군 내부에서는 매우 안정적인 오답이다.

따라서 단순히
  confidence가 높다
  여러 view가 같은 label을 말한다
  consistency가 높다
라는 조건만으로는 안전한 switch가 불가능하다.
```

v101 결론:

```text
Base B evaluator는 재관측/검토 대상 수집에는 효과적이다.

하지만 selector가 답을 교체하려면
stable support만이 아니라
"그 stable support가 객체 전체성/필수 부위 연결성을 설명하는가"
를 추가로 봐야 한다.
```

다음 방향:

```text
1. needs_observation과 switch를 분리한다.
   needs_observation:
     이 관측은 검토/재관측 대상이다.

   switch:
     후보 view가 실제로 MVP를 대체할 만큼 객체 설명력이 있는가.

2. switch selector에 object completeness feature가 필요하다.
   예:
     - crop family가 특정 하단/측면에만 몰렸는가
     - 정답 후보가 객체 전체 bbox를 덮는가
     - full/ROI/local candidate가 서로 어떤 관계로 충돌하는가
     - 후보 label support가 안정적이지만 부분 crop에 갇힌 것은 아닌가

3. cat_26/cat_72 같은 stable-wrong 후보는
   "관측 일관성은 높지만 객체 완전성이 낮은 오답"으로 분리해야 한다.
```

## 72. v101 failure check: stable wrong 후보의 성격

TEST1에서 v101이 깨뜨린 대표 sample:

```text
cat_26
cat_72
```

확인 결과:

```text
두 sample 모두 MVP는 cat으로 맞췄다.
하지만 candidate bbox view 44개는 전부 dog로 갔다.
```

예:

```text
cat_26:
  MVP = cat, correct, confidence = 0.553

  top dog candidates:
    r0-4_c1-4 dog 0.961
    r0-3_c1-3 dog 0.956
    r0-3_c1-4 dog 0.937
    r1-3_c0-2 dog 0.933

  full image candidate:
    r0-4_c0-4 dog 0.704

  cat candidate:
    none
```

```text
cat_72:
  MVP = cat, correct, confidence = 0.581

  top dog candidates:
    r2-4_c1-3 dog 0.964
    r2-4_c0-4 dog 0.962
    r2-4_c0-3 dog 0.918
    r1-3_c0-3 dog 0.904

  full image candidate:
    r0-4_c0-4 dog 0.686

  cat candidate:
    none
```

즉 이 둘은:

```text
불안정한 오답 후보가 아니다.
후보 관측군 내부에서는 매우 안정적인 오답이다.
```

추가 조건 실험:

```text
switch할 때 MVP label을 지지하는 candidate view가 강하게 남아 있으면
switch를 막는 조건을 테스트했다.
```

결과:

```text
mvp_view_max_conf ceiling을 낮추면
TEST4 broken은 줄일 수 있었다.

하지만 TEST1 cat_26/cat_72는 여전히 깨진다.
이유:
  두 sample은 MVP label candidate가 0개라서
  mvp_view_max_conf 기반 방어가 작동하지 않는다.
```

대표 결과:

```text
mvp_view_max_conf <= 0.85:

TEST1:
  final = 97.80%
  fixed = 0
  broken = 3

TEST2:
  final = 97.10%
  fixed = 65
  broken = 12

TEST4:
  final = 95.70%
  fixed = 12
  broken = 1
```

해석:

```text
MVP label 후보가 강하게 남아 있는 경우는 switch 위험을 줄이는 데 도움이 된다.
하지만 "후보 전체가 안정적으로 틀린 경우"는 이 조건으로 막을 수 없다.
```

중요한 결론:

```text
Base B evaluator는 검토 대상 수집에 성공했다.
하지만 switch selector는 candidate family 내부 정보만으로는 한계가 있다.

특히 cat_26/cat_72는:
  candidate crop branch 전체가 dog로 간다.
  full candidate도 dog다.
  MVP fusion만 cat을 맞춘다.

따라서 이 유형은
  후보 selector 문제가 아니라
  "MVP fusion evidence와 candidate evidence의 충돌을 어떻게 다룰 것인가"
의 문제다.
```

다음 방향:

```text
1. switch를 바로 하지 말고 conflict state로 분리한다.

   if:
     needs_observation = true
     candidate family strongly disagrees with MVP
     MVP confidence is low/mid
     candidate has no MVP-label support

   then:
     switch가 아니라 conflict_hold / require_next_observation

2. v1.0의 1-5단계 스캔이 필요한 이유가 여기서 명확해진다.
   기존 bbox 후보군이 모두 한쪽으로 기울어도,
   단계적 관측에서 객체 연결성/필수 부위가 어떻게 채워지는지 봐야 한다.

3. 현재 단계에서는:
   Base B = 검토 대상 수집기
   selector = 일부 복구 가능
   conflict_hold = 안정 오답 후보 방어
   로 역할을 나누는 것이 맞다.
```

## 73. MVP v0.2 plus v102: 후처리 teacher를 선처리 base에 흡수

가설:

```text
베이스가 오염되지 않으려면
후처리 evaluator가 하던 관측 안정성 판단이
선처리 base 학습에도 들어가야 한다.
```

기존:

```text
image
-> MVP v0.2 class prediction
-> post evaluator / gate
```

v102:

```text
image
-> ROI/full/structure/q
-> multi-view observation summary
-> class head
-> observation-state head
```

추가 도구:

```text
tools/train_eval_mvp02_plus_v102.py
```

입력:

```text
기존 v0.2 입력:
  structure_vector
  q_vec
  ROI texture
  full texture

추가 observation summary:
  candidate label entropy
  label별 view ratio
  label별 confidence mean/max/std
  full candidate cat/dog score
  best label별 bbox area/center
  consistency mean/max
  same_label_ratio
  overlap_conflict
  nearby_other_count
```

중요:

```text
v102 observation summary는 old MVP label에 상대적인 feature를 피한다.
즉 old MVP가 cat이라고 했는가/dog라고 했는가에 직접 끌리지 않도록,
label별 후보 관측 통계를 사용한다.
```

loss:

```text
classification loss
  + Base B teacher state 기반 sample weight

observation-state auxiliary loss
  target:
    stable_accept
    valid_but_observation_fragile
    recoverable_wrong
    hard_wrong
```

classification sample weight:

```text
stable_accept:
  1.20

valid_but_observation_fragile:
  0.35

recoverable_wrong:
  1.50

hard_wrong:
  0.50
```

의미:

```text
fragile correct는 정답이더라도 깨끗한 정답으로 강하게 학습하지 않는다.
recoverable wrong은 기존 MVP가 틀렸던 샘플이므로 새 base가 배워야 할 신호로 더 강하게 둔다.
```

학습:

```text
train:
  train_1000

epochs:
  12

obs_loss_weight:
  0.35

output:
  results\mvp02_plus_v102_train_1000
```

train validation:

```text
best val class accuracy:
  98.5%

best observation-state val accuracy:
  약 96.0%
```

외부 평가:

```text
TEST1:
  base MVP = 99.45%
  v102 plus = 98.90%
  delta = -0.55%
  obs_acc = 95.05%
  confusion = [[89, 2], [0, 91]]

TEST2:
  base MVP = 96.00%
  v102 plus = 97.77%
  delta = +1.77%
  obs_acc = 84.29%
  confusion = [[2345, 55], [52, 2348]]

TEST4:
  base MVP = 94.60%
  v102 plus = 97.60%
  delta = +3.00%
  obs_acc = 84.30%
  confusion = [[483, 17], [7, 493]]
```

평균:

```text
base MVP mean:
  96.68%

v102 plus mean:
  98.09%

mean delta:
  +1.41%
```

해석:

```text
후처리 evaluator를 단순 gate로만 쓰는 것보다,
그 teacher state를 선처리 base 학습에 넣는 방향이 훨씬 강하게 작동했다.
```

중요한 차이:

```text
v101:
  base를 유지하고 나중에 switch했다.
  TEST2/TEST4는 올랐지만 TEST1 broken이 생겼다.

v102:
  base 자체가 multi-view 관측 요약을 함께 보고 학습했다.
  post-switch 없이 TEST2/TEST4가 크게 올랐다.
```

현재 결론:

```text
Base B 방향은 단순 후처리보다 선처리 base 개조에서 더 효과가 크다.

MVP v0.2를 완전히 버린 것이 아니라,
class scorer 구조를 유지하면서 observation summary와 observation-state auxiliary loss를 붙인 것이 핵심이다.
```

남은 문제:

```text
TEST1은 99.45% -> 98.90%로 약간 하락했다.

즉 hard 분포 일반화는 좋아졌지만,
쉬운/내부 분포에서 아주 강한 기존 MVP의 일부 안정성을 잃었다.
```

다음 방향:

```text
1. v102를 새 base 후보로 둔다.

2. TEST1 손실을 줄이기 위해
   stable_accept distillation 또는 기존 MVP consistency loss를 추가한다.

3. v102 observation-state head를 이용해
   conflict_hold / next_observation 판단을 다시 구성한다.

4. 후처리 evaluator는 제거하지 않고
   teacher / contamination filter / safety monitor 역할로 유지한다.
```
## 2026-06-09 - v115 object-support bundle 초안

목표:

```text
single crop 하나가 맞으면 oracle hit로 세는 기존 기준을 완화하고,
여러 view가 같은 class를 반복 지지하는 object-support bundle 기준을 추가한다.
```

추가 도구:

```text
tools/build_object_support_bundle_v115.py
```

입력:

```text
results/v113_imagenet_bbox_cache_awa_cls5_test/bbox_candidate_scores.csv
```

실행:

```powershell
.\.venv\Scripts\python.exe -m tools.build_object_support_bundle_v115 `
  --candidate_csv results\v113_imagenet_bbox_cache_awa_cls5_test\bbox_candidate_scores.csv `
  --out_dir results\v115_object_support_bundle_v113_cls5_test `
  --grid 4 `
  --min_views 2 `
  --min_tiles 2 `
  --min_sep 0.0 `
  --tile_rel_threshold 0.35
```

결과:

```text
base_accuracy                    92.17%
single_view_oracle_accuracy      99.67%
support_bundle_oracle_accuracy   99.50%
support_fusion_accuracy          93.50%

base_wrong_total                 47
single_view_oracle_recovers      45
support_bundle_oracle_recovers   44
support_fusion_fixed             28
support_fusion_broken            20
support_fusion_net_gain          +8
```

해석:

```text
single-view oracle 상한 대부분은 crop 하나의 우연이 아니라,
여러 view가 같은 class를 반복 지지하는 bundle로도 유지된다.

다만 자동 fusion 점수식은 아직 조잡해서 broken이 많다.
즉 다음 단계는 bundle 후보 생성보다 bundle 선택/가중치 학습이다.
```

샘플 확인:

```text
siamese+cat_siamese+cat_10070
- base는 german+shepherd로 오답
- Siamese support bundle valid
- 27개 view가 Siamese를 지지
- 누적 support tiles = 12개

siamese+cat_siamese+cat_10369
- base는 persian+cat로 고확신 오답
- Siamese support bundle 없음
- 44개 view 전체가 Persian 쪽으로 수렴
```

## 2026-06-09 - v116 object-support pattern prior

목표:

```text
object-support bundle을 합칠 때,
단순 view confidence가 아니라 train에서 자주 반복된 bbox/tile support 패턴을 점수에 반영한다.
```

추가 도구:

```text
tools/build_object_support_pattern_prior_v116.py
```

점수 구성:

```text
class separation
+ max separation
+ agreeing view count
+ support tile ratio
+ train-learned class tile prior alignment
+ train-learned bbox precision
+ coverage balance
```

best 설정:

```powershell
.\.venv\Scripts\python.exe -m tools.build_object_support_pattern_prior_v116 `
  --train_candidate_csv results\v113_imagenet_bbox_cache_awa_cls5_train\bbox_candidate_scores.csv `
  --eval_candidate_csv results\v113_imagenet_bbox_cache_awa_cls5_test\bbox_candidate_scores.csv `
  --out_dir results\v116_object_support_pattern_prior_v113_cls5_test_best `
  --grid 4 `
  --min_sep 0.0 `
  --tile_rel_threshold 0.35 `
  --w_pattern 0.0 `
  --w_box_precision 0.12 `
  --w_coverage_balance 0.08
```

결과:

```text
base_accuracy                    92.17%
single_view_oracle_accuracy      99.67%
support_bundle_oracle_accuracy   99.50%
pattern_fusion_accuracy          94.00%

base_wrong_total                 47
pattern_fusion_fixed             28
pattern_fusion_broken            17
pattern_fusion_net_gain          +11
```

해석:

```text
v115 fixed 28 / broken 20 대비
v116 best는 fixed 28 / broken 17로 broken을 줄였다.

현재 cls5에서는 위치 prior 자체보다 bbox precision과 coverage balance가 먼저 효과를 냈다.
즉 어떤 타일이 눈/몸통인지 직접 지정하기보다,
train에서 신뢰된 bbox family와 과소/과대 support 방지가 더 안정적인 1차 점수로 보인다.
```

추가 실패 분석:

```powershell
.\.venv\Scripts\python.exe -m tools.analyze_object_support_bundle_failures_v116 `
  --pred_csv results\v116_object_support_pattern_prior_v113_cls5_test_best\object_support_pattern_predictions.csv `
  --bundle_csv results\v116_object_support_pattern_prior_v113_cls5_test_best\object_support_pattern_bundles.csv `
  --out_dir results\v116_object_support_failure_analysis_cls5_test
```

결과:

```text
chosen_wrong                  36
recoverable_chosen_wrong      33
unrecoverable_chosen_wrong     3
fixed                         28
broken                        17
```

recoverable wrong의 특징:

```text
잘못 선택된 bundle은 정답 bundle보다 view_count가 평균 +8개 이상 많다.
즉 v116은 아직 "많이 반복되는 class evidence"를 과신하는 경향이 있다.
```

view_count_balance 추가 실험:

```text
w_view_count_balance 0.24
w_box_precision      0.12
w_coverage_balance   0.12

accuracy             94.00%
fixed                28
broken               17
net_gain             +11
```

해석:

```text
view_count balance는 성능을 무너뜨리지 않았지만 단독 상승도 만들지 못했다.
다음 점수는 단순 view 개수보다,
"같은 부위를 반복해서 본 것"과 "서로 다른 객체 부위를 보완해서 본 것"을 분리해야 한다.
```

## 2026-06-09 - v117 evidence diversity / redundancy

목표:

```text
한 부위에서 20번 나온 class evidence와
여러 부위에서 5번 나온 class evidence를 구분한다.
```

추가 도구:

```text
tools/build_object_support_diversity_v117.py
```

추가 feature:

```text
support_tile_entropy
bbox_iou_mean / bbox_iou_max
unique_tile_ratio
view_area_variance
same_region_repeat_ratio
cross_region_class_agreement
bundle_redundancy_penalty
```

best 후보:

```powershell
.\.venv\Scripts\python.exe -m tools.build_object_support_diversity_v117 `
  --train_candidate_csv results\v113_imagenet_bbox_cache_awa_cls5_train\bbox_candidate_scores.csv `
  --eval_candidate_csv results\v113_imagenet_bbox_cache_awa_cls5_test\bbox_candidate_scores.csv `
  --out_dir results\v117_object_support_diversity_v113_cls5_test_best `
  --grid 4 `
  --min_sep 0.0 `
  --tile_rel_threshold 0.35 `
  --w_pattern 0.0 `
  --w_box_precision 0.12 `
  --w_coverage_balance 0.08 `
  --w_entropy 0.03 `
  --w_unique_tiles 0.0 `
  --w_redundancy_penalty 0.10 `
  --w_area_variance 0.0
```

결과:

```text
base_accuracy                    92.17%
support_bundle_oracle_accuracy   99.50%
v117 diversity fusion            94.00%
fixed                            28
broken                           17
net_gain                         +11
```

해석:

```text
diversity/redundancy feature는 v116 성능을 유지했지만 단독 상승은 만들지 못했다.

failure analysis에서는 recoverable wrong에서도
cross_region_class_agreement가 잘못 선택된 bundle 쪽으로 높게 나오는 경우가 있었다.

즉 "넓게 퍼진 class agreement" 자체도 함정이 될 수 있다.
다음 단계는 단순 diversity reward가 아니라,
서로 다른 부위가 보완적인지 또는 같은 착각이 넓게 퍼진 것인지 구분하는 학습형 selector다.
```

## 2026-06-09 - v118 parent/internal/external analyzer

목표:

```text
오답을 parent가 같은 내부 오답과 parent가 다른 외부 오답으로 분해한다.
```

parent map:

```text
persian+cat     -> cat
siamese+cat     -> cat
chihuahua       -> dog
german+shepherd -> dog
horse           -> horse
```

추가 도구:

```text
tools/analyze_parent_internal_external_v118.py
```

결과:

```text
base MVP v0.2:
  accuracy        92.17%
  internal wrong  27
  external wrong  20

v117 chosen:
  accuracy        94.00%
  parent accuracy 96.83%
  internal wrong  17
  external wrong  19
```

해석:

```text
v117의 개선은 주로 내부 오답 감소에서 나왔다.
외부 오답은 거의 줄지 않았다.

따라서 이후 gate는 internal gate와 external gate를 분리하되,
class-specific hard rule이 아니라 학습 가능한 feature/prototype 기반이어야 한다.
```

## 2026-06-09 - v119 class prototype gate diagnostic

목표:

```text
사람이 class별 필요한 부위를 직접 관리하지 않고,
train의 정답 object-support bundle에서 class prototype을 만든다.
```

추가 도구:

```text
tools/build_class_prototype_gate_v119.py
```

결과:

```text
MVP base accuracy                92.17%
v117 chosen accuracy             94.00%
best valid prototype accuracy    94.00%

prototype fixes chosen wrong     11
prototype breaks chosen correct  11
```

해석:

```text
prototype 단독 선택은 v117과 같은 정확도다.
하지만 고치는 샘플과 깨는 샘플이 서로 달라서 상호 보완 신호가 있다.

따라서 prototype은 hard switch로 쓰기보다,
prototype_similarity_gap을 learned gate의 feature로 넣는 쪽이 맞다.
```

## 2026-06-09 - v120 learned bundle gate

Goal:

```text
Train a learned gate on sample x candidate-class bundle rows.
The target is whether each bundle label equals the true class.
```

Tool:

```text
tools/train_eval_bundle_gate_v120.py
```

Training rows:

```text
train 1500 samples x 5 bundles = 7500 rows
label = bundle_label == y_true_name
```

Features:

```text
v117 bundle features
prototype_similarity / prototype_distance
mvp_confidence
base_label_match
parent_matches_base / parent_switch
score_minus_base_conf
proto_minus_base_conf
```

Results:

```text
HGB:
  eval accuracy 94.00%
  fixed 23
  broken 12
  net +11

RandomForest:
  eval accuracy 93.83%
  fixed 28
  broken 18
  net +10

LogisticRegression:
  eval accuracy 94.50%
  fixed 27
  broken 13
  net +14
```

LogisticRegression parent analysis:

```text
accuracy          94.50%
parent accuracy   97.67%
internal wrong    19
external wrong    14
```

Comparison:

```text
v117:
  accuracy        94.00%
  internal wrong  17
  external wrong  19

v120-logreg:
  accuracy        94.50%
  internal wrong  19
  external wrong  14
```

Interpretation:

```text
The learned gate improved overall accuracy and parent accuracy.
However, the gain came mostly from reducing external errors, not internal errors.

To reduce internal errors, the next gate should compare only candidates inside
the same parent group instead of using one global bundle gate.
```

## 2026-06-09 - v121 Tile-Wave Relation Bundle Features

Goal:

```text
Check whether tile-level 0-360 observation relations help bundle selection.

Previous features mostly counted:
- which crop/view predicted which class
- how many tiles/views supported a class
- whether evidence was diverse or redundant

v121 adds:
- how support tiles relate to each other under the observer/source tile relation
- whether support tiles are internally consistent
- whether support tiles are separated from outside tiles
- whether repeated evidence is actually relationally redundant
```

New tool:

```text
tools/attach_wave_relation_bundle_features_v121.py
```

Inputs:

```text
train relation:
  results/tile_view_relation_v05_awa_cls5_train/tile_view_relation_v05.npz

test relation:
  results/tile_view_relation_v05_awa_cls5_test/tile_view_relation_v05.npz

train bundles:
  results/v119_class_prototype_gate_v117_cls5_train/prototype_scored_bundles.csv

test bundles:
  results/v119_class_prototype_gate_v117_cls5_test/prototype_scored_bundles.csv
```

Added feature family:

```text
wave_inside_mutual_cov_mean/max
wave_inside_edge_abs_delta_mean/max/std
wave_inside_rho_abs_delta_mean
wave_inside_int_abs_delta_mean
wave_inside_edge_asym_mean
wave_inside_rho_asym_mean
wave_inside_relation_density
wave_outside_edge/rho/int_abs_delta_mean
wave_inside_outside_edge/rho_gap
```

Implementation note:

```text
tools/train_eval_bundle_gate_v120.py now automatically includes numeric
columns whose names start with wave_.
```

Results on AWA cls5 test:

```text
v120-logreg:
  accuracy 94.50%
  fixed 27
  broken 13
  net +14

v121-logreg-wave:
  accuracy 94.50%
  fixed 27
  broken 13
  net +14

v120-HGB:
  accuracy 94.00%
  fixed 23
  broken 12
  net +11

v121-HGB-wave:
  accuracy 94.33%
  fixed 26
  broken 13
  net +13

v120-RF:
  accuracy 93.83%
  fixed 28
  broken 18
  net +10

v121-RF-wave:
  accuracy 94.00%
  fixed 27
  broken 16
  net +11
```

Parent/internal analysis for v121-HGB-wave:

```text
accuracy          94.33%
parent accuracy   97.17%
internal wrong    17
external wrong    17
```

Interpretation:

```text
The current tile-wave summary is valid and mildly useful, but not yet decisive.

It improves the non-linear gates slightly and preserves v117-level internal
error count, but the best total accuracy still comes from the learned logreg
gate without a clear additional wave gain.

This suggests the relation tensor is useful infrastructure, but the current
summary is probably too coarse. The next wave step should model relation
patterns inside parent groups, especially persian+cat vs siamese+cat and
chihuahua vs german+shepherd, instead of adding one global wave score.
```

Additional diagnostic:

```text
When all valid bundles are compared, correct bundles have much larger
support_tile_count_check.

However, inside valid parent-internal rows, correct bundles tend to have:
- larger support tile count
- lower inside rho/int delta than wrong internal bundles
- lower inside mutual coverage/relation density than repeated wrong bundles

This means the useful wave signal is not simply "higher relation score is
better". It is closer to:

enough support tiles
+ not too much same-region repetition
+ internally connected but not over-collapsed
```

Next implication:

```text
Wave features should probably be used by an internal parent-specific selector,
not by one global class selector.

For example:
- cat-internal selector: persian+cat vs siamese+cat
- dog-internal selector: chihuahua vs german+shepherd

The selector should learn relation shape within the parent group rather than
use a hard rule or a single global threshold.
```

## 2026-06-09 - v122 Observation Error Audit

Goal:

```text
Stop using hard rules as the decision mechanism.

Instead, build an audit table that explains:
- whether the correct candidate existed
- whether the selector missed a recoverable candidate
- whether a correct base answer was broken
- whether the remaining error is internal or external
- how the chosen wrong bundle differs from the true bundle
```

New tool:

```text
tools/analyze_observation_error_audit_v122.py
```

Main output:

```text
results/v122_observation_error_audit_hgb_wave_cls5_test
results/v122_observation_error_audit_logreg_wave_cls5_test
```

LogReg-wave audit result:

```text
n                         600
base accuracy             92.17%
chosen accuracy           94.50%
support bundle oracle     99.50%

stable_correct            540
fixed                      27
missed_recoverable         17
broken                     13
candidate_oracle_miss       3

chosen internal wrong      19
chosen external wrong      14
```

Interpretation:

```text
The candidate space is already strong.
Only 3 samples are candidate-oracle misses.

Most remaining errors are selector/audit errors:
- missed_recoverable: correct bundle existed but was not selected
- broken: base was correct but selector switched away
```

Important feature pressure:

```text
For missed_recoverable and broken rows, the chosen wrong bundle often has
much higher view_count than the true bundle.

This means the selector still over-trusts "many views support this class".

The wave features show useful disagreement signals:
- wave_inside_rho_abs_delta_mean
- wave_inside_int_abs_delta_mean
- wave_inside_outside_rho_gap
- wave_inside_rho_asym_mean

The next selector should learn:

many repeated views != accurately observed object
```

Next implication:

```text
v123 should not add another hard threshold.

It should train an observation-reliability model:

input:
  bundle features
  wave relation features
  chosen-vs-true style contrast during training

target:
  whether this bundle is a reliable observation candidate

runtime:
  choose the class bundle with high CNN evidence AND high observation reliability
```

## 2026-06-09 - v123 Observation Reliability Model

Goal:

```text
Train a model that scores whether a bundle is a reliable observation,
instead of directly using hard rules.

The intended separation:

class evidence:
  what does this bundle look like?

observation reliability:
  did this bundle observe enough, without repeating one region too much?
```

New tool:

```text
tools/train_eval_observation_reliability_v123.py
```

Important implementation note:

```text
Initial run exposed a train-base merge issue:
train base CSV had 1200 rows while the bundle table had 1500 samples.
Missing base rows were treated as false, making train base accuracy look like 80%.

The tool now fills missing base fields from the bundle CSV's mvp_* columns.
```

Results:

```text
v120-logreg:
  eval accuracy 94.50%
  fixed 27
  broken 13
  net +14

v121-HGB-wave:
  eval accuracy 94.33%
  fixed 26
  broken 13
  net +13

v123-logreg-evidence:
  eval accuracy 92.33%
  fixed 1
  broken 0
  net +1

v123-HGB-fixedbase:
  train base accuracy 99.80%
  train final accuracy 99.47%
  eval accuracy 92.17%
  fixed 0
  broken 0
  net 0
```

Interpretation:

```text
v123 did not improve accuracy yet.

After the train-base fix, the train split is too easy:
base is already 99.8%.

When the train-only objective tries to avoid broken cases, the learned
observation reliability policy converges to almost no switching.

This means current train data does not contain enough ambiguous/recoverable
training pressure for an observation reliability model.
```

Conclusion:

```text
The v123 idea is still correct, but the training target is incomplete.

Observation reliability should not be trained only on ordinary train rows.
It needs hard/ambiguous bundle pairs:

- base correct but fragile observation
- base wrong but correct bundle exists
- high view_count wrong bundle vs lower-count true bundle
- parent-internal confusions

Next version should build a hard observation-pair dataset from train data,
then train a pairwise/ranking reliability model.
```

## 2026-06-09 - v124 Pairwise Observation Reliability

Goal:

```text
Train observation reliability as a comparison:

given one image,
  true bundle
  vs hard negative bundle

learn which one observed the object more reliably.
```

New tool:

```text
tools/train_eval_pairwise_observation_reliability_v124.py
```

Training setup:

```text
For each train sample:
  positive = true valid bundle
  negative = top-k hard non-true bundles by score/view_count/confidence

Train pair rows:
  true - negative -> 1
  negative - true -> 0
```

Result:

```text
pair rows        6000
train base       99.80%
train final      98.80%
eval base        92.17%
eval final       92.50%
fixed             2
broken            0
net              +2
```

Audit:

```text
stable_correct           553
fixed                      2
missed_recoverable        42
candidate_oracle_miss      3

chosen internal wrong     25
chosen external wrong     20
```

Interpretation:

```text
v124 is safe but too conservative.

It almost never breaks correct predictions, but it also fails to recover most
recoverable errors.

This means the pairwise reliability model learned a useful caution signal, but
not a strong switching policy.
```

Useful learned signs:

```text
Positive/large coefficients:
  base_label_match
  mean_bbox_precision
  mean_conf
  pattern_alignment
  view_count

Negative coefficients:
  same_region_repeat_ratio
  bbox_iou_mean
  prototype_distance
  wave_inside_mutual_cov_max
  wave_inside_edge_abs_delta_std
  wave_outside_int_abs_delta_mean
```

Meaning:

```text
The model did learn that repeated same-region evidence can be bad.
It also uses some wave relation features as reliability penalties.

However, base_label_match and view_count still dominate the decision, making
the runtime policy too conservative.
```

Next implication:

```text
v125 should separate two models:

1. accept/base safety model
   Should I keep the base answer?

2. recovery candidate ranker
   If base is suspicious, which alternative bundle is the best observation?

The current single pairwise scorer mixes these two jobs and defaults to safety.
```

## 2026-06-09 - v125 Two-Stage Observation Policy

Goal:

```text
Split v124 into two jobs:

1. accept/base safety model
   Is the base answer suspicious?

2. recovery ranker
   If suspicious, which bundle should replace it?
```

New tool:

```text
tools/train_eval_two_stage_observation_policy_v125.py
```

Result:

```text
train base wrong count      3
train base accuracy         99.80%
train final accuracy        99.80%
train switch count          15
train fixed/broken          1 / 1

eval base accuracy          92.17%
eval final accuracy         92.33%
eval switch count           17
eval fixed/broken           1 / 0
```

Audit:

```text
stable_correct              553
fixed                         1
missed_recoverable           43
candidate_oracle_miss         3

chosen internal wrong        27
chosen external wrong        19
```

Interpretation:

```text
The two-stage structure is conceptually correct, but the current train split
does not contain enough base-wrong or ambiguous samples.

The accept model only saw 3 train base-wrong samples, so it learns an extremely
conservative "almost never open recovery" policy.

As a result, the recovery ranker is rarely used at runtime.
```

Conclusion:

```text
v123-v125 show the same bottleneck:

The candidate space is strong, but ordinary train data is too easy.

To learn observation correction, the next step must construct a hard/ambiguous
training set rather than adding another gate architecture.
```

Next direction:

```text
v126 hard observation training set

Create training pressure from:
- train correct but fragile observations
- high-confidence wrong alternative bundles
- same-parent confusions
- synthetic hard candidates from top-k non-true bundles
- possibly extra classes/harder splits

Then train:
- accept model on base suspicious vs stable
- recovery ranker on true bundle vs hard negative bundle
```

## 2026-06-09 - v130a Texture Relation Cache Smoke

Goal:

```text
Start the relation-aware texture observer.

Instead of letting CNN judge only isolated crops, build a texture relation:

texture_relation[i, j]
  = CNN response when observer tile i views source tile j through relation context
```

New tool:

```text
tools/build_texture_relation_cache_v130.py
```

Initial crop definition:

```text
i == j:
  source tile bbox

i != j:
  union bbox spanning observer tile i and source tile j
  + small padding
```

This is a cheap first approximation of observer-source texture relation.

Smoke command:

```text
python -m tools.build_texture_relation_cache_v130 \
  --tiles_dir results/tile_seq_awa_cls5_test \
  --dataset_root dataset/awa_multiclass_v110/cls5 \
  --split test \
  --out_dir results/v130_texture_relation_cls5_test_smoke50_shuffle \
  --backbone resnet18 \
  --weights default \
  --model_ckpt results/baseline_full_resnet18_awa_cls5/full_texture_head.pt \
  --batch_size 256 \
  --device auto \
  --limit 50 \
  --debug_crops 16 \
  --shuffle \
  --seed 130
```

Output:

```text
texture_relation_v130.npz
index.csv
summary.json
debug_crops/
```

Tensor shape:

```text
texture_relation:   [50, 16, 16, 12]
relation_embedding: [50, 16, 16, 512]
```

Feature names:

```text
emb_norm
src_self_cos
obs_self_cos
confidence
entropy
margin
top1_id
prob_persian+cat
prob_siamese+cat
prob_chihuahua
prob_german+shepherd
prob_horse
```

Smoke sample class mix:

```text
siamese+cat        17
horse              10
german+shepherd     9
persian+cat         8
chihuahua           6
```

Diagnostic:

```text
self tile true prob mean      0.4740
relation true prob mean       0.7325
relation true prob max        0.9840

self top-true rate            0.5262
relation top-true rate        0.7963

src_self_cos_off_mean         0.7161
obs_self_cos_off_mean         0.7161
```

Interpretation:

```text
This is a strong smoke-test signal.

Isolated tile self-view is often weak, especially for cat subclasses.
But when the CNN sees observer-source relation context, the true-class
probability and top-true rate improve substantially.

This supports the v130 hypothesis:

texture should not be judged only as an isolated crop.
It becomes more useful when evaluated through observation relation context.
```

Important caution:

```text
The current relation crop is only a cheap union-bbox approximation.
It may inflate context and make the task easier.

Next steps must compare:
- self tile
- source tile with padding
- observer-source union
- wave-weighted relation crop

The useful signal is not "larger crop is always better".
The useful signal is whether relation context improves true support without
causing repeated-region or background leakage.
```

## 2026-06-09 - v130b Relation Crop Mode Comparison

Goal:

```text
Check whether v130a improvement came from true observer-source relation context
or merely from seeing a slightly larger crop.
```

Compared modes on the same 50 shuffled cls5 samples:

```text
source:
  source tile only

source_pad:
  source tile + small padding

union:
  observer-source union bbox + small padding
```

Overall result:

```text
mode        self_true_prob  relation_true_prob  relation_top_true
source      0.4740          0.4740              0.5262
source_pad  0.4944          0.4944              0.5588
union       0.4740          0.7325              0.7963
```

Class-level relation top-true rate:

```text
                 source  source_pad  union
chihuahua        0.6875  0.7188      0.7333
german+shepherd  0.6042  0.6250      0.7889
horse            0.8125  0.8562      0.9625
persian+cat      0.4375  0.4922      0.7521
siamese+cat      0.3015  0.3235      0.7456
```

Interpretation:

```text
source_pad improves only slightly over source.
union improves substantially.

This suggests the signal is not just "slightly larger crop".
Observer-source relation context is changing the CNN evidence.
```

Next:

```text
Build v130c bundle features from texture_relation:
- relation true/class support density
- self-to-relation gain
- relation entropy drop
- same-region relation repeat penalty
- wave-texture consistency
```

## 2026-06-09 - v130c Texture Relation Bundle Features

Goal:

```text
Summarize texture_relation[i, j] into bundle-level features.

This asks:
Does the candidate bundle's class receive consistent texture support
across observer-source relation pairs?
```

New tool:

```text
tools/attach_texture_relation_bundle_features_v130.py
```

Added feature family:

```text
texrel_label_self_prob_mean/max
texrel_label_relation_prob_mean/max/std
texrel_self_to_relation_prob_gain
texrel_label_relation_top_rate
texrel_label_relation_density_50/80
texrel_confidence_relation_mean
texrel_entropy_self_mean
texrel_entropy_relation_mean
texrel_entropy_drop_self_to_relation
texrel_margin_relation_mean
texrel_src_self_cos_relation_mean
texrel_obs_self_cos_relation_mean
texrel_relation_agreement_minus_self_top
```

Important note:

```text
The smoke texture relation covers only 50 samples.
The full bundle CSV has 600 samples x 5 classes = 3000 rows.
Therefore diagnostics must filter to matched samples only.
```

Matched valid bundle comparison:

```text
feature: texrel_label_relation_prob_mean
source      target 0.5012  non-target 0.2264  delta 0.2748
source_pad  target 0.5249  non-target 0.2231  delta 0.3017
union       target 0.7472  non-target 0.1999  delta 0.5473

feature: texrel_label_relation_top_rate
source      target 0.5430  non-target 0.2318  delta 0.3112
source_pad  target 0.5856  non-target 0.2206  delta 0.3649
union       target 0.8005  non-target 0.1794  delta 0.6211

feature: texrel_label_relation_density_50
source      target 0.5108  non-target 0.1722  delta 0.3386
source_pad  target 0.5262  non-target 0.1731  delta 0.3531
union       target 0.7757  non-target 0.1236  delta 0.6521

feature: texrel_entropy_drop_self_to_relation
source      delta 0.0000
source_pad  delta 0.0000
union       delta 0.2024

feature: texrel_relation_agreement_minus_self_top
source      delta 0.0000
source_pad  delta 0.0000
union       delta 0.3099
```

Interpretation:

```text
The union relation context not only increases true class probability;
it also separates true bundles from non-target bundles much more strongly.

This is exactly the kind of signal v123-v125 were missing:
not just "high confidence", but "relation context improves class agreement".
```

Next:

```text
Build v130 texture_relation for train and full test.
Then attach texrel_* features to v121/v117 bundle tables and rerun:

- v120 learned bundle gate
- v122 audit
- parent/internal error analysis
```

## 2026-06-09 - v130d Full Train/Test: Texture Relation + Wave Bundle Gate

Goal:

```text
Use v130 union observer-source texture relation features on the full cls5 train/test split.

Question:
Does relation-aware CNN texture evidence improve the learned bundle selector,
not just the smoke diagnostic separation?
```

Data:

```text
train: dataset\awa_multiclass_v110\cls5\train, n=1500
test : dataset\awa_multiclass_v110\cls5\test,  n=600

classes:
persian+cat
siamese+cat
chihuahua
german+shepherd
horse
```

Texture relation cache:

```text
mode: union
train texture_relation shape = [1500, 16, 16, 12]
test  texture_relation shape = [600, 16, 16, 12]
embedding shape = [N, 16, 16, 512]
backbone = frozen ResNet18 default
head = baseline_full_resnet18_awa_cls5\full_texture_head.pt
```

Full bundle feature diagnostics:

```text
train target - non-target delta:
texrel_label_relation_density_50   +0.6610
texrel_label_relation_top_rate      +0.6521
texrel_label_relation_density_80    +0.6185
texrel_label_relation_prob_mean     +0.5788
texrel_label_relation_prob_max      +0.4581

test target - non-target delta:
texrel_label_relation_density_50   +0.6410
texrel_label_relation_top_rate      +0.6298
texrel_label_relation_density_80    +0.5901
texrel_label_relation_prob_mean     +0.5537
texrel_label_relation_prob_max      +0.4413
```

Selector result:

```text
base MVP v0.2 accuracy: 92.17%

v121 wave + learned gate:
logreg 94.50%
HGB    94.33%

v130d wave + texrel + learned gate:
logreg 94.83%  fixed=22  broken=6   net=+16
HGB    94.67%  fixed=23  broken=8   net=+15
```

Parent/internal audit:

```text
v130d logreg:
chosen accuracy        94.83%
parent accuracy        97.50%
internal wrong         16
external wrong         15
base internal wrong    27
base external wrong    20

v130d HGB:
chosen accuracy        94.67%
parent accuracy        97.67%
internal wrong         18
external wrong         14
base internal wrong    27
base external wrong    20
```

Observation audit:

```text
v130d logreg:
support bundle oracle rate 99.50%
stable_correct            547
fixed                     22
missed_recoverable        22
broken                    6
candidate_oracle_miss     3

v130d HGB:
support bundle oracle rate 99.50%
stable_correct            545
fixed                     23
missed_recoverable        21
broken                    8
candidate_oracle_miss     3
```

Interpretation:

```text
v130 relation-aware texture evidence is useful.
It gives a small but real improvement over v121 wave-only features:
94.50% -> 94.83% for logreg.

The larger signal is diagnostic:
candidate oracle remains 99.50%, so candidate generation is strong.
The current bottleneck is still selector choice, especially the 21-22 missed recoverable samples.

The error shape also improved:
internal wrong fell from 27 to 16 for logreg,
external wrong fell from 20 to 15.
```

Next:

```text
1. Inspect missed_recoverable samples in v130d.
2. Compare whether missed cases have weak texrel evidence or selector under-switching.
3. Consider v131:
   - train a selector that explicitly optimizes candidate ranking inside a sample
   - keep v130 texrel features as the cheap pseudo-multimodal texture observer input
```

## 2026-06-09 - v131 Texture-Object Agreement Gate

Goal:

```text
Make the gate use CNN texture as an observer signal, not just as class confidence.

The question is:
Does texture evidence agree with object/wave relation evidence?
```

New tool:

```text
tools/attach_texture_object_agreement_features_v131.py
```

Added feature family:

```text
objrel_wave_texture_density_agreement
objrel_wave_texture_strong_agreement
objrel_wave_texture_prob_agreement
objrel_wave_cov_texture_top_agreement
objrel_wave_gap_texture_density_agreement
objrel_texture_gain_on_wave_support
objrel_texture_entropy_drop_on_wave_support
objrel_texture_relation_improves_with_object
objrel_independent_texture_support
objrel_multi_region_texture_support
objrel_diverse_object_texture_support
objrel_parent_safe_texture_support
objrel_high_texture_low_wave_conflict
objrel_high_wave_low_texture_conflict
objrel_texture_overconfidence_redundant
objrel_single_region_texture_risk
objrel_unstable_texture_relation_risk
objrel_area_variance_texture_risk
```

Feature diagnostic:

```text
train target - non-target delta:
objrel_independent_texture_support       +0.4306
objrel_wave_cov_texture_top_agreement    +0.4268
objrel_high_wave_low_texture_conflict    -0.3664
objrel_high_texture_low_wave_conflict    +0.3378
objrel_wave_gap_texture_density_agreement +0.3279

test target - non-target delta:
objrel_wave_cov_texture_top_agreement    +0.4149
objrel_independent_texture_support       +0.4020
objrel_high_wave_low_texture_conflict    -0.3557
objrel_high_texture_low_wave_conflict    +0.3189
objrel_wave_gap_texture_density_agreement +0.3178
```

Selector result:

```text
base MVP v0.2                 92.17%
v130d wave+texrel logreg      94.83%
v130d wave+texrel HGB         94.67%

v131 objrel logreg            97.17%  fixed=32  broken=2  net=+30
v131 objrel HGB               96.83%  fixed=31  broken=3  net=+28
```

Parent/internal audit:

```text
v131 logreg:
chosen accuracy      97.17%
parent accuracy      100.00%
internal wrong       17
external wrong       0
base internal wrong  27
base external wrong  20

v131 HGB:
chosen accuracy      96.83%
parent accuracy      99.83%
internal wrong       18
external wrong       1
base internal wrong  27
base external wrong  20
```

Observation audit:

```text
v131 logreg:
support bundle oracle rate 99.50%
stable_correct            551
fixed                     32
missed_recoverable        12
broken                    2
candidate_oracle_miss     3

v131 HGB:
support bundle oracle rate 99.50%
stable_correct            550
fixed                     31
missed_recoverable        13
broken                    3
candidate_oracle_miss     3
```

Interpretation:

```text
v131 is the first strong result for the "texture as observation relation" idea.

The main effect is not merely higher accuracy.
The important structural change is:
external errors were almost eliminated.

For logreg:
external wrong 20 -> 0
parent accuracy 97.50% in v130d -> 100.00% in v131

Remaining errors are internal fine-grained errors:
siamese+cat <-> persian+cat
chihuahua <-> german+shepherd
```

Conclusion:

```text
Texture-object agreement is a strong external-boundary gate.
The next problem is not "cat/dog/horse parent" anymore.
The next problem is internal species/breed separation inside the same parent.
```

Next:

```text
v132:
separate external gate and internal gate.

External gate:
use objrel/wave/texrel to prevent parent switches.

Internal gate:
learn what evidence is required to split similar subclasses,
such as persian+cat vs siamese+cat or chihuahua vs german+shepherd.
```

## 2026-06-09 - v132 External Parent Gate Broad Cat/Dog Check

Goal:

```text
Check whether a cls5-trained external gate generalizes back to broad cat/dog datasets.

This is different from cls5 test:
cls5 test labels are fine/subclass labels.
test1/test2/test4 labels are broad cat/dog labels.
```

Fast sanity check:

```text
Use cls5 full-image ResNet18 head,
then fold predictions to parent:

persian+cat, siamese+cat -> cat
chihuahua, german+shepherd -> dog
horse -> horse/outside
```

Result:

```text
cls5 full head -> parent:
test1 92.31%
test2 86.13%
test4 92.50%

existing v104a 2-class:
test1 98.90%
test2 98.10%
test4 97.40%

ResNet50 2-class:
test1 99.45%
test2 96.31%
test4 97.20%
```

Interpretation:

```text
The fine-class head itself does not solve broad cat/dog generalization.
It leaks dog samples into horse especially on test2/test4.
```

Then build cls5 bbox candidates for test1/test2/test4 and run v132-lite external parent gate using support-diversity parent aggregation only.

Result:

```text
v132-lite logreg:
test1 85.71%
test2 81.92%
test4 89.70%

v132-lite HGB:
test1 86.26%
test2 81.69%
test4 88.90%
```

Parent candidate oracle:

```text
test1 parent oracle 99.45%
test2 parent oracle 99.31%
test4 parent oracle 99.90%
```

Important diagnosis:

```text
Correct parent evidence is almost always present in the candidate set.
The failure is not candidate generation.
The failure is parent selector/generalization.
```

Why v102/v104 works better:

```text
v102/v104 directly learns broad cat/dog observation state:
cat_score, dog_score, cat_minus_dog, dog_minus_cat,
full_cat_score, full_dog_score,
and dynamic weights for observation-risk samples.

v132-lite only sees fine-class support bundles,
then tries to aggregate them into parent choices.
It does not yet have broad parent scores or broad parent risk targets.
```

Conclusion:

```text
External gate must be active, but it must be broad-parent-aware.

Next v133:
combine v102/v104 broad parent signals with v131/v132 fine evidence:

cat_parent_score = aggregate(persian+cat, siamese+cat) + broad cat evidence
dog_parent_score = aggregate(chihuahua, german+shepherd) + broad dog evidence
horse/outside_score = outside leakage risk

The external gate should learn:
when to trust broad cat/dog evidence,
when fine subclass evidence helps,
and when horse/outside is a leakage rather than a real parent.
```

## v133 parent-first gate

Goal:

```text
먼저 broad parent(cat/dog)를 안정적으로 판단하고,
fine-class evidence(persian/siamese/chihuahua/german/horse)는
부모 판단을 뒤집는 보조 증거로만 사용한다.
```

Implementation:

```text
tools/train_eval_parent_first_gate_v133.py
```

Train-only input:

```text
broad anchor:
results/v133_inputs/v104a_train1000_predictions.csv

fine candidate cache:
results/v133_cls5_bbox_cache_train1000

fine support bundle:
results/v133_object_support_diversity_cls5_parent_train1000
```

Train candidate generation result:

```text
n = 1000
v104a broad accuracy = 99.70%
cls5 candidate rows = 44,000
```

### v133 HGB, weak anchor

Output:

```text
results/v133_parent_first_gate_hgb_train1000
```

Result:

```text
train:
  broad 99.70%
  parent-first 100.00%
  fixed 3 / broken 0

test1:
  broad 98.90%
  parent-first 98.90%
  fixed 0 / broken 0

test2:
  broad 98.10%
  parent-first 96.00%
  fixed 28 / broken 129

test4:
  broad 97.40%
  parent-first 94.60%
  fixed 0 / broken 28
```

Diagnosis:

```text
fine evidence still leaks into wrong parent choices.
The selector can fix train errors, but it over-switches on harder external sets.
```

### v133 HGB, strong anchor

Policy:

```text
anchor_bonus = 0.5
switch_penalty = 0.5
switch_margin = 0.2
```

Output:

```text
results/v133_parent_first_gate_hgb_train1000_strong_anchor
```

Result:

```text
train:
  broad 99.70%
  parent-first 99.70%
  fixed 0 / broken 0

test1:
  broad 98.90%
  parent-first 98.90%
  fixed 0 / broken 0

test2:
  broad 98.10%
  parent-first 98.10%
  fixed 0 / broken 0

test4:
  broad 97.40%
  parent-first 97.40%
  fixed 0 / broken 0
```

Diagnosis:

```text
Strong anchoring preserves the good v104a broad-parent behavior,
but it becomes too conservative and does not actively recover new errors.
```

Main conclusion:

```text
v133 confirms the right hierarchy:
broad parent must be the anchor, fine class evidence must be auxiliary.

However, train_1000 has only 3 broad-parent errors.
That is too little signal to learn an active switch policy.

Next step:
generate train-only ambiguous/hard parent cases so the gate can learn
when switching is beneficial and when fine evidence is leakage.
```

### v133 corrected cls5-base run

Issue with the first v133 run:

```text
The previous v133 test1/test2/test4 run used v104a 2-class predictions
as the broad anchor.

That is useful as a diagnostic, but it is not a valid 5-class expansion test,
because the final behavior is mostly inherited from the 2-class model.
```

Corrected condition:

```text
base prediction:
results/v133_inputs/v113_cls5_train_predictions_all.csv

train bundle:
results/v117_object_support_diversity_v113_cls5_train_best

test bundle:
results/v117_object_support_diversity_v113_cls5_test_best

output:
results/v133_parent_first_gate_cls5base_hgb
```

This uses the 5-class MVP prediction as the base:

```text
persian+cat
siamese+cat
chihuahua
german+shepherd
horse
```

Then v133 folds them into parent groups:

```text
persian+cat, siamese+cat -> cat
chihuahua, german+shepherd -> dog
horse -> horse
```

Result:

```text
train parent:
  cls5-base parent accuracy 99.60%
  v133 parent-first accuracy 100.00%
  fixed 6 / broken 0

cls5 test parent:
  cls5-base parent accuracy 96.50%
  v133 parent-first accuracy 97.00%
  fixed 3 / broken 0
```

Confusion matrix after v133 parent-first:

```text
cat   -> cat 234, dog 6, horse 0
dog   -> cat 2, dog 232, horse 6
horse -> cat 0, dog 4, horse 116
```

Interpretation:

```text
With a real 5-class base, parent-first gives a small but clean improvement.
It fixes 3 parent-level errors without breaking any correct parent decisions.

However, this is parent accuracy, not full 5-class subclass accuracy.
The next step is:

1. parent-first gate chooses cat/dog/horse
2. internal selector chooses persian vs siamese, chihuahua vs german+shepherd, etc.
```

## v134 hierarchical parent-child evaluation

Design:

```text
External gate:
  parent를 먼저 고른다.
  cat / dog / horse

Internal selector:
  선택된 parent 내부에서만 5-class probability를 비교한다.
```

Implementation:

```text
docs/dual_line_v134_hierarchical_active_gate_design_ko.md
tools/eval_hierarchical_parent_child_v134.py
```

Input:

```text
fine prediction:
results/eval_mvp02_plus_multiclass_v113_imagenet_awa_cls5_test/predictions.csv

parent prediction:
results/v133_parent_first_gate_cls5base_hgb/cls5test/parent_first_predictions.csv
```

Result:

```text
fine 5-class base accuracy:
  92.83%

v133 parent accuracy:
  97.00%

v134 hierarchical final accuracy:
  93.17%

fixed vs fine:
  2

broken vs fine:
  0
```

Error decomposition:

```text
correct:
  559

internal_error:
  23

external_error:
  18
```

Interpretation:

```text
The hierarchy is working in the intended direction.
Constraining fine prediction inside the selected parent recovers 2 samples
without breaking any fine-correct sample.

The remaining errors split into two clean targets:

external_error:
  parent gate still wrong.
  Improve v102/v104-style active parent gate.

internal_error:
  parent is right, subclass is wrong.
  Improve internal fine selector/evidence diversity.
```

## v135 hierarchical bundle gate

Goal:

```text
v102/v104의 장점:
  external parent stability

v131의 장점:
  internal subclass specificity

v135:
  parent를 먼저 학습하고,
  그 parent 내부에서만 fine selector를 학습한다.
```

Implementation:

```text
tools/train_eval_hierarchical_bundle_gate_v135.py
```

Key training constraint:

```text
Parent model:
  learns cat / dog / horse from aggregated bundle evidence.

Fine model:
  is trained only on rows whose bundle parent equals the true parent.

This prevents the fine selector from learning subclass evidence
before the model has learned the higher-level parent concept.
```

Input:

```text
train bundle:
results/v131_texture_object_agreement_cls5_train_union/texture_object_agreement_scored_bundles.csv

train base prediction:
results/v133_inputs/v113_cls5_train_predictions_all.csv

eval bundle:
results/v131_texture_object_agreement_cls5_test_union/texture_object_agreement_scored_bundles.csv

eval base prediction:
results/eval_mvp02_plus_multiclass_v113_imagenet_awa_cls5_test/predictions.csv
```

### v135 logreg

Output:

```text
results/v135_hierarchical_bundle_gate_logreg_cls5_test
```

Result:

```text
base fine accuracy:
  92.83%

parent accuracy:
  97.67%

hierarchical fine accuracy:
  94.00%

fixed / broken:
  7 / 0

error split:
  internal_error 22
  external_error 14
```

### v135 HGB

Output:

```text
results/v135_hierarchical_bundle_gate_hgb_cls5_test
```

Result:

```text
base fine accuracy:
  92.83%

parent accuracy:
  100.00%

hierarchical fine accuracy:
  95.67%

fixed / broken:
  19 / 2

error split:
  internal_error 26
  external_error 0
```

Comparison:

```text
v131 objrel logreg:
  fine accuracy 97.17%
  parent accuracy 100.00%
  internal_error 17
  external_error 0

v135 HGB:
  fine accuracy 95.67%
  parent accuracy 100.00%
  internal_error 26
  external_error 0
```

Interpretation:

```text
v135 succeeded at the main conceptual goal:
external parent pollution was removed.

However, its internal selector is less sharp than v131 logreg.
So the next target is not parent stability anymore;
it is recovering v131-level internal specificity inside the v135 hierarchy.
```

### v135-lite parent evaluation on TEST1/TEST2/TEST4

Purpose:

```text
TEST1/TEST2/TEST4 do not have fine subclass labels.
Therefore evaluate only parent contamination:

persian+cat, siamese+cat -> cat
chihuahua, german+shepherd -> dog
horse -> leakage/outside
```

Input condition:

```text
Base:
  full-image ResNet18 cls5 head folded to parent

Parent gate:
  v135/v133-style parent-first gate trained only on cls5 train

Feature level:
  lite support-diversity parent bundle
  no full v131 objrel features yet for TEST1/TEST2/TEST4
```

Output:

```text
results/v135_lite_parent_gate_eval_test1_test2_test4
```

Result:

```text
TEST1:
  cls5 folded parent base 92.31%
  v135-lite parent       92.86%
  fixed 1 / broken 0

TEST2:
  cls5 folded parent base 86.13%
  v135-lite parent       86.19%
  fixed 7 / broken 4

TEST4:
  cls5 folded parent base 92.50%
  v135-lite parent       92.30%
  fixed 1 / broken 3
```

Interpretation:

```text
This lite parent gate does not solve external contamination on broad cat/dog datasets.

The strong AWA cls5 internal result does not automatically transfer to TEST1/TEST2/TEST4.
The missing part is the v102/v104-style broad parent observation signal,
not just parent aggregation of cls5 fine evidence.
```

Comparison target:

```text
v104a broad parent:
  TEST1 98.90%
  TEST2 98.10%
  TEST4 97.40%
```

Conclusion:

```text
For broad external datasets, v135 must add real broad-parent observation learning.
Using cls5 fine evidence alone, even hierarchically, is not enough.
```

## v136 parent observation gate

Goal:

```text
Keep v131 as the internal fine selector idea,
but recover v102/v104a-style external parent stability by training
an explicit parent observation gate.
```

Implementation:

```text
tools/train_eval_parent_observation_gate_v136.py
```

Training data:

```text
catdog broad train:
  state = results/v100_base_observation_state_train_1000
  pred  = results/v133_inputs/v104a_train1000_predictions.csv

cls5 train:
  state = results/v113_imagenet_base_observation_state_awa_cls5_train
  pred  = results/v133_inputs/v113_cls5_train_predictions_all.csv
```

Evaluation:

```text
TEST1/TEST2/TEST4 parent contamination only.
Fine subclass accuracy is not evaluated because those sets only have cat/dog labels.
```

Features:

```text
v104-style observation state:
  label entropy
  candidate confidence spread
  mvp/other label support
  stable/unstable support counts
  observation support/conflict/stability
  base partial risk

plus folded parent probabilities:
  pred_parent_prob_cat
  pred_parent_prob_dog
  pred_parent_prob_horse
```

### v136 HGB

Output:

```text
results/v136_parent_observation_gate_hgb_test1_test2_test4
```

Result:

```text
TEST1:
  base folded parent 93.41%
  v136 parent        96.15%
  fixed 5 / broken 0

TEST2:
  base folded parent 87.38%
  v136 parent        91.96%
  fixed 260 / broken 40

TEST4:
  base folded parent 93.40%
  v136 parent        94.30%
  fixed 15 / broken 6
```

### v136 logreg

Output:

```text
results/v136_parent_observation_gate_logreg_test1_test2_test4
```

Result:

```text
TEST1:
  93.96%

TEST2:
  91.44%

TEST4:
  81.40%
```

Interpretation:

```text
HGB is the better parent observation gate here.
Logreg over-generalizes and collapses TEST4 dogs into cat too often.
```

Comparison against v104a:

```text
v104a broad parent:
  TEST1 98.90%
  TEST2 98.10%
  TEST4 97.40%

v136 HGB:
  TEST1 96.15%
  TEST2 91.96%
  TEST4 94.30%
```

Conclusion:

```text
Adding v104-style observation features clearly improves external parent stability
over cls5 folded parent evidence alone.

However, v136 still does not reach v104a.
The likely missing piece is that v104a was trained end-to-end as a broad cat/dog
classifier, while v136 is currently a post-hoc parent classifier using folded cls5
evidence and observation-state features.

Next adjustment:
make the parent head a first-class training target in the same hierarchy,
not only a post-hoc classifier.
```

## v137 and v135 mixed adjustment

Hypothesis:

```text
Training order matters.

Bad order:
  learn 5 fine classes first
  then fold them into parent classes

Desired order:
  learn parent space first
  cat / dog / horse

  then learn fine classes inside the parent space
  cat -> persian+cat / siamese+cat
  dog -> chihuahua / german+shepherd
```

### v137 parent-pretrained hierarchy

Implementation:

```text
tools/train_eval_parent_pretrained_hierarchy_v137.py
```

Method:

```text
1. Train parent observation model from:
   - catdog train_1000
   - AWA cls5 train

2. Add parent probabilities as features to the fine selector.

3. Train fine selector only inside the true parent.
```

Output:

```text
results/v137_parent_pretrained_hierarchy_hgb_logreg_cls5_test
```

Result:

```text
Parent eval on broad sets:
  TEST1 96.15%
  TEST2 91.96%
  TEST4 94.30%

AWA cls5 fine eval:
  base fine accuracy 92.83%
  parent accuracy 94.83%
  hierarchical fine accuracy 91.50%
  external_error 31
  internal_error 20
```

Interpretation:

```text
Naively injecting broad parent probabilities into the fine selector hurts.
The parent model trained for broad external stability does not transfer cleanly
to AWA cls5 internal fine selection.

This means parent pretraining is still right conceptually,
but the parent signal must constrain the hierarchy without polluting the fine selector.
```

### v135 mixed model: parent HGB + fine logreg

Adjustment:

```text
Keep the v135 hierarchy:
  parent first
  fine inside selected parent

But separate model choices:
  parent model = HGB
  fine model = logreg
```

Output:

```text
results/v135_hierarchical_bundle_gate_parent_hgb_fine_logreg_cls5_test
```

Result:

```text
base fine accuracy:
  92.83%

parent accuracy:
  100.00%

hierarchical fine accuracy:
  96.17%

fixed / broken:
  20 / 0

error split:
  internal_error 23
  external_error 0
```

Comparison:

```text
v131 objrel logreg:
  fine 97.17%
  parent 100.00%
  internal_error 17
  external_error 0

v135 HGB/HGB:
  fine 95.67%
  parent 100.00%
  internal_error 26
  external_error 0

v135 HGB/logreg:
  fine 96.17%
  parent 100.00%
  internal_error 23
  external_error 0

v137 naive parent-prob injection:
  fine 91.50%
  parent 94.83%
  external_error 31
```

Conclusion:

```text
Best current hierarchy candidate:
  v135 parent HGB + fine logreg

It preserves the important parent-first behavior:
  external_error = 0

and recovers part of v131's internal strength:
  95.67% -> 96.17%

Next target:
  recover v131's internal 97.17% while keeping external_error at 0.
```

## v138 two-gate hierarchy

Hypothesis:

```text
Use two independent gates:

Upper gate:
  v103/v104-style broad parent gate
  cat / dog / horse
  no CNN tracking / objrel as fine evidence

Lower gate:
  v131-style internal fine gate
  uses CNN tracking, texrel, objrel
  only after parent is selected
```

Implementation:

```text
tools/train_eval_two_gate_pipeline_v138.py
```

Upper gate:

```text
results/v138_upper_parent_gate_hgb
```

Upper parent results:

```text
AWA cls5 test:
  parent accuracy 94.83%

TEST1:
  parent accuracy 96.15%

TEST2:
  parent accuracy 91.96%

TEST4:
  parent accuracy 94.30%
```

Two-gate AWA cls5 result:

```text
results/v138_two_gate_upper_hgb_lower_logreg_cls5_test

base fine accuracy:
  92.83%

upper parent accuracy:
  94.83%

two-gate final accuracy:
  91.33%

fixed / broken:
  0 / 9

error split:
  external_error 31
  internal_error 21
```

Interpretation:

```text
The two-gate idea is structurally correct,
but the current upper gate is not a universal parent gate.

It improves broad external TEST1/TEST2/TEST4,
but it hurts AWA cls5 parent accuracy.

Therefore:
  external broad parent gate and internal cls5 parent gate cannot be naively shared.
```

Current best interpretation:

```text
1. For AWA cls5 internal evaluation:
   v131 or v135 HGB/logreg remains better.

2. For TEST1/TEST2/TEST4 external contamination:
   v136 HGB improves over folded cls5,
   but still does not reach v104a.

3. Next design should separate:
   - broad external parent gate
   - internal parent gate for cls5 hierarchy
   - lower fine selector
```

## v139 soft multiplicative parent gate

Goal:

```text
Avoid the hard upper-gate failure of v138.

Instead of:
  upper parent decides first
  then lower fine gate is forced inside that parent

try:
  final_score(class)
    = lower_score(class) * parent_prior(parent(class))^alpha

or in log space:
  log(final) = log(lower) + alpha * log(parent_prior)
```

### v139 raw bundle score experiment

Implementation:

```text
tools/eval_soft_multiplicative_parent_gate_v139.py
```

Input:

```text
candidate:
  results/v131_texture_object_agreement_cls5_test_union/texture_object_agreement_scored_bundles.csv

parent:
  results/v138_upper_parent_gate_hgb/cls5test/parent_predictions.csv
```

Important limitation:

```text
This first run used raw bundle score, not the learned v131 lower gate score.
So alpha=0 is not v131.
```

Best result:

```text
alpha 0.05
accuracy 94.67%
fixed / broken 25 / 10
external_error 12
internal_error 20
```

Interpretation:

```text
Raw bundle score is weaker than the learned lower selector.
The parent prior helps raw score slightly, but this is not the real two-gate test.
```

### v139b learned lower gate x parent prior

Implementation:

```text
tools/train_eval_soft_multiplicative_parent_gate_v139b.py
```

Input:

```text
train candidate:
  results/v131_texture_object_agreement_cls5_train_union/texture_object_agreement_scored_bundles.csv

train pred:
  results/v133_inputs/v113_cls5_train_predictions_all.csv

eval candidate:
  results/v131_texture_object_agreement_cls5_test_union/texture_object_agreement_scored_bundles.csv

eval pred:
  results/eval_mvp02_plus_multiclass_v113_imagenet_awa_cls5_test/predictions.csv

eval parent:
  results/v138_upper_parent_gate_hgb/cls5test/parent_predictions.csv
```

Result:

```text
alpha  accuracy  fixed  broken  external_error  internal_error
0.000  97.17%    32     2       0               17
0.010  97.17%    32     2       0               17
0.020  97.17%    32     2       0               17
0.030  97.17%    32     2       0               17
0.050  97.17%    32     2       0               17
0.075  97.00%    31     2       1               17
0.100  97.00%    31     2       1               17
0.150  96.83%    31     3       2               17
0.200  96.67%    30     3       3               17
0.300  95.83%    26     4       8               17
0.500  95.17%    22     4       12              17
```

Best:

```text
alpha 0.0 to 0.05
accuracy 97.17%
parent accuracy 100.00%
external_error 0
internal_error 17
```

Interpretation:

```text
The multiplicative idea is structurally valid,
but the current upper parent gate does not add useful information to v131 on AWA cls5.

v131 already removes external contamination on this set.
Adding parent prior above alpha 0.05 starts to reintroduce external error.

So for the current 5-class AWA setting:
  lower v131 evidence selector remains the best decision source.

The next useful direction is not stronger multiplication,
but a better parent model trained with parent-first structure from the start.
```

## v140 internal error audit

Question:

```text
Why did v138 hard two-gate and v139 soft multiplicative parent gate fail?

After v131 removes external contamination,
what remains inside the 17 internal errors?
```

Implementation:

```text
tools/analyze_internal_error_audit_v140.py
```

Input:

```text
candidate scores:
  results/v139b_lower_logreg_x_parent_hgb_cls5_test/eval_candidate_scores.csv

predictions:
  results/v139b_lower_logreg_x_parent_hgb_cls5_test/alpha_0_0/predictions.csv
```

Output:

```text
results/v140_internal_error_audit_cls5_v131_alpha0
```

### Why v138 failed

```text
upper parent accuracy:
  94.83%

upper parent wrong:
  31 samples

v138 final wrong:
  52 samples

v138 wrong overlapping parent wrong:
  31 / 52

v138 correct despite parent wrong:
  0
```

Interpretation:

```text
The hard upper gate is too brittle.

Once the upper parent gate chooses the wrong parent,
the lower fine selector cannot recover.
```

### Why v139/v139b failed

```text
alpha 0.0:
  accuracy 97.17%
  external_error 0
  internal_error 17

alpha 0.3:
  accuracy 95.83%
  external_error 8
  internal_error 17
```

Critical overlap:

```text
alpha 0.0 correct -> alpha 0.3 broken:
  8 samples

among those, parent gate was wrong:
  8 / 8

alpha 0.0 wrong -> alpha 0.3 fixed:
  0 samples
```

Interpretation:

```text
The parent prior does not fix internal fine errors.
It only injects wrong parent evidence into samples that v131 already handled.
```

### Internal error composition

Remaining v131 internal errors:

```text
total:
  17

pairs:
  siamese+cat -> persian+cat: 9
  persian+cat -> siamese+cat: 4
  german+shepherd -> chihuahua: 2
  chihuahua -> german+shepherd: 2
```

Diagnosis:

```text
true_has_less_support: 6
true_bundle_very_weak: 4
true_bundle_invalid: 3
wrong_bundle_scored_higher: 3
near_tie_selector_margin: 1
```

True label rank:

```text
rank 2: 14 samples
rank 3: 1 sample
rank 5: 2 samples
```

Interpretation:

```text
Most internal errors are not missing-label failures.
The true label is usually present as the second-best candidate.

The failure is mostly that the wrong same-parent bundle receives stronger evidence.
```

### Feature tendency

Mean chosen-wrong minus true-candidate deltas:

```text
lower_gate_score: +0.5129
view_count: +16.29
support_tile_count: +2.06
support_tile_entropy: +0.194
same_region_repeat_ratio: +0.083
cross_region_class_agreement: +0.333
texrel_label_relation_prob_mean: +0.282
texrel_label_relation_top_rate: +0.335
objrel_multi_region_texture_support: +0.128
objrel_diverse_object_texture_support: +0.172
prototype_similarity: +0.180
```

Interpretation:

```text
The wrong internal class often wins because it has:
  more views,
  more support tiles,
  stronger cross-region agreement,
  stronger texture relation support,
  stronger prototype similarity.

So the next issue is not parent gating.
It is fine-class evidence discrimination inside the same parent.
```

Next direction:

```text
v141 should focus on internal evidence quality:

1. same-parent contrastive scoring
   compare true-like and rival-like evidence directly

2. class-specific required evidence
   Persian vs Siamese should not be decided by generic cat evidence alone

3. support normalization
   many repeated/related views should not automatically dominate a smaller but more specific true bundle

4. near-tie rescue
   rank-2 true candidates are common, so second-best same-parent evidence should be explicitly audited
```

## v141 candidate-level parent/fine multi-task gate

Goal:

```text
Do not attach parent and fine decisions after the fact.

Train two heads from the same candidate bundle features:

1. fine head:
   bundle_label == y_true_name

2. parent head:
   parent(bundle_label) == parent(y_true_name)

Then choose:
  final_score = fine_logit + lambda * parent_logit
```

Implementation:

```text
tools/train_eval_multitask_parent_fine_gate_v141.py
```

Input:

```text
train bundle:
  results/v131_texture_object_agreement_cls5_train_union/texture_object_agreement_scored_bundles.csv

train pred:
  results/v133_inputs/v113_cls5_train_predictions_all.csv

eval bundle:
  results/v131_texture_object_agreement_cls5_test_union/texture_object_agreement_scored_bundles.csv

eval pred:
  results/eval_mvp02_plus_multiclass_v113_imagenet_awa_cls5_test/predictions.csv
```

### v141 logreg fine + logreg parent

Output:

```text
results/v141_multitask_parent_fine_logreg_logreg_cls5_test
```

Best:

```text
lambda 0.0
accuracy 97.17%
parent_accuracy 100.00%
fixed / broken 32 / 2
external_error 0
internal_error 17
```

Adding parent head:

```text
lambda 0.02:
  accuracy 96.83%
  internal_error 19

lambda 0.05 and above:
  accuracy 96.67%
  internal_error 20
```

### v141 logreg fine + HGB parent

Output:

```text
results/v141_multitask_parent_fine_logreg_hgb_cls5_test
```

Best:

```text
lambda 0.0
accuracy 97.17%
parent_accuracy 100.00%
fixed / broken 32 / 2
external_error 0
internal_error 17
```

Adding parent head:

```text
lambda 0.02 and above:
  accuracy 97.00%
  internal_error 18
```

### v141 HGB fine + HGB parent

Output:

```text
results/v141_multitask_parent_fine_hgb_hgb_cls5_test
```

Best:

```text
lambda 0.075
accuracy 97.00%
parent_accuracy 100.00%
fixed / broken 32 / 3
external_error 0
internal_error 18
```

Train behavior:

```text
HGB/HGB train accuracy:
  99.47%

HGB/HGB eval accuracy:
  97.00%
```

Interpretation:

```text
The parent head is not harmful as a diagnostic,
but it does not improve the v131 lower selector.

Reason:
  parent target is too broad.

Inside one parent, many wrong fine labels are also parent-correct.
So parent_score rewards generic cat/dog/horse evidence,
but does not tell Persian from Siamese or Chihuahua from German Shepherd.

Therefore:
  v103/v104 broad stability and v131 internal selector do not combine
  by simply adding a parent auxiliary score at candidate-selection time.
```

Current conclusion:

```text
The failure is not only "attached too late".
It is also "parent supervision is too coarse".

To improve beyond v131,
the next target must be same-parent contrast:

  not:
    is this cat evidence?

  but:
    is this Persian evidence rather than Siamese evidence?
    is this Chihuahua evidence rather than German Shepherd evidence?
```

### v141 parent-fold evaluation on TEST1/TEST2/TEST4

Question:

```text
If v141 reaches about 97% on AWA 5-class,
what happens when its 5-class output is folded back to cat/dog on TEST1/TEST2/TEST4?
```

Caveat:

```text
TEST1/TEST2/TEST4 do not have fine labels.
So fine accuracy is not meaningful there.

The only meaningful metric is:
  parent_accuracy

Also, these sets have object-support candidate bundles,
not the full v131 objrel candidate bundle feature set.
So this is a parent-fold compatibility check, not the exact v131/v141 cls5 setup.
```

Implementation:

```text
tools/train_eval_multitask_parent_fine_gate_v141.py
```

Train:

```text
results/v117_object_support_diversity_v113_cls5_train_best/object_support_pattern_bundles.csv
results/v133_inputs/v113_cls5_train_predictions_all.csv
```

Eval:

```text
TEST1:
  results/v132_object_support_diversity_cls5_parent_test1/object_support_pattern_bundles.csv
  results/v135_inputs/cls5_parent_base_test1.csv

TEST2:
  results/v132_object_support_diversity_cls5_parent_test2_4800/object_support_pattern_bundles.csv
  results/v135_inputs/cls5_parent_base_test2_4800.csv

TEST4:
  results/v132_object_support_diversity_cls5_parent_test4_awa_catdog/object_support_pattern_bundles.csv
  results/v135_inputs/cls5_parent_base_test4_awa_catdog.csv
```

Results:

```text
dataset  best_lambda  parent_accuracy
TEST1    0.5          88.46%
TEST2    0.5          84.13%
TEST4    0.5          90.40%
```

Comparison:

```text
v104a broad cat/dog:
  TEST1 98.90%
  TEST2 98.10%
  TEST4 97.40%

v141 parent-fold object-support:
  TEST1 88.46%
  TEST2 84.13%
  TEST4 90.40%
```

Interpretation:

```text
The 5-class internal selector does not automatically become a good broad
cat/dog external classifier.

Even when the output is folded to parent labels, the cls5/object-support
candidate policy is much weaker than the broad v104a model on TEST1/TEST2/TEST4.

This confirms the earlier diagnosis:
  internal fine discrimination and external broad stability are different tasks.

The broad parent behavior must be trained directly,
not recovered by folding a fine-class selector after the fact.
```

## v142 parent-space alignment first pass

목표:

```text
fine selector를 parent로 접는 방식이 아니라,
처음부터 cat/dog/horse 상위 공간을 먼저 정렬한다.
```

학습 데이터:

```text
train_1000:
  cat/dog original train cache

awa_cls5_train:
  persian+cat, siamese+cat -> cat
  chihuahua, german+shepherd -> dog
  horse -> horse
```

구현:

```text
tools/train_eval_parent_space_alignment_v142.py
```

출력:

```text
results/v142_parent_space_alignment_mix_train1000_awa_cls5
```

결과:

```text
dataset              accuracy   horse_fp   confusion_matrix
TEST1                98.35%     0          [[89, 2, 0], [1, 90, 0], [0, 0, 0]]
TEST2                96.40%     28         [[2320, 69, 11], [76, 2307, 17], [0, 0, 0]]
TEST4                96.30%     24         [[495, 5, 0], [8, 468, 24], [0, 0, 0]]
AWA cls5 parent       95.17%     5          [[228, 12, 0], [6, 229, 5], [0, 6, 114]]
```

해석:

```text
v142는 v141 parent-fold object-support보다 broad parent 일반화가 크게 좋아졌다.
하지만 v104a broad cat/dog에는 아직 못 미친다.

특히 TEST2/TEST4에서 horse false positive가 생긴다.
즉 parent 공간 정렬은 가능하지만, 현재 단일 parent head는
horse 축과 AWA fine 도메인이 cat/dog broad 공간을 약간 오염시킨다.
```

결론:

```text
공간을 먼저 정렬한다는 방향은 맞다.
다만 v104a와 v131의 장점은 아직 자동으로 합쳐지지 않았다.

다음 단계는 parent head를 더 강하게 만드는 것이 아니라,
parent 외부 안정성(v104a)과 fine 내부 증거(v131)를
동일 feature 위에서 역할 분리하여 붙이는 구조가 필요하다.
```

## v143 fine head on frozen v142 parent space

목표:

```text
v142에서 cat/dog/horse parent 공간을 먼저 정렬한 뒤,
그 frozen fused feature 위에 fine 5-class head만 학습한다.

질문:
  parent 정체성을 유지한 상태로 내부 세부종 분해가 가능한가?
```

구현:

```text
tools/train_eval_fine_on_parent_space_v143.py
```

입력:

```text
parent checkpoint:
  results/v142_parent_space_alignment_mix_train1000_awa_cls5/v142_parent_space_alignment.pt

fine train:
  results/dual_line_cache_awa_cls5_train
```

출력:

```text
results/v143_fine_on_v142_parent_space
```

결과:

```text
dataset       fine_acc   parent_acc   internal_error   external_error   horse_fp
AWA cls5      90.50%     94.83%       26               31               -
TEST1         -          97.80%       -                -                0
TEST2         -          95.60%       -                -                61
TEST4         -          93.90%       -                -                51
```

해석:

```text
v142 parent feature를 얼리고 fine head만 얹으면 내부 세부종 분해력이 약하다.
v131 objrel logreg의 97.17% fine accuracy보다 크게 낮다.

또한 TEST2/TEST4에서 horse false positive가 증가한다.
즉 parent 정체성을 먼저 만든 뒤 frozen feature 위에서 fine만 학습하는 방식은
현재로서는 v104a의 외부 안정성도, v131의 내부 증거성도 모두 충분히 살리지 못한다.
```

결론:

```text
단순한 순차 구조:
  parent align -> freeze -> fine head
는 실패에 가깝다.

필요한 구조는 frozen parent space가 아니라,
parent loss와 fine/object-relation loss를 함께 보되
역할을 분리하는 multi-objective / two-head 구조다.

즉 parent 정체성은 보존하되,
fine 분해에 필요한 object relation evidence는 별도 경로로 유지해야 한다.
```

## v144 identity-preserving prototype first pass

목표:

```text
세부종 학습 전에 parent 정체성을 먼저 확보한다.

일반 softmax fine 학습은 차이에 집중하므로,
catness/dogness/horse identity가 부산물로만 생긴다.

따라서 parent CE + prototype/center loss로
같은 parent 내부 공통점을 직접 보상한다.
```

구현:

```text
tools/run_identity_preserving_v144.py
```

한 번에 수행한 단계:

```text
1. train_1000 + AWA cls5 train을 parent(cat/dog/horse)로 접어서 identity embedding 학습
2. parent center/prototype loss 적용
3. frozen identity embedding 위에 AWA 5-class fine probe 학습
4. TEST1/TEST2/TEST4 parent identity와 AWA cls5 fine accuracy 평가
```

실행:

```powershell
python -m tools.run_identity_preserving_v144 `
  --out_dir results\v144_identity_preserving_prototype `
  --parent_epochs 14 `
  --fine_epochs 14 `
  --batch_size 128 `
  --device auto `
  --center_weight 0.35 `
  --sep_weight 0.05 `
  --seed 144
```

결과:

```text
dataset    parent_head_acc   fine/fold_parent_acc   horse_fp   fine_acc
TEST1      97.80%            97.80%                 0          -
TEST2      96.46%            94.81%                 114        -
TEST4      96.30%            93.10%                 56         -
AWA cls5   95.83%            95.00%                 -          87.33%
```

해석:

```text
parent identity head는 v142와 비슷한 수준으로 유지됐다.
하지만 frozen identity embedding 위 fine probe는 87.33%로 낮다.

즉 단순 prototype identity loss는 parent 정체성을 어느 정도 잡지만,
세부종 분해에 필요한 object relation evidence를 충분히 보존하지 못한다.
```

결론:

```text
정체성 loss 방향 자체는 맞다.
하지만 frozen parent identity embedding 하나에 fine 분해까지 맡기면 안 된다.

다음 구조는:
  shared observer stem
  + parent identity head/prototype loss
  + fine/object-relation head

처럼 identity 경로와 fine evidence 경로를 같이 학습하되,
fine loss가 identity를 오염시키지 못하게 regularization을 거는 방식이어야 한다.
```

## v145 joint identity + fine training

목표:

```text
v144의 freeze 방식은 fine evidence를 너무 많이 잃었다.
따라서 shared observer embedding을 parent/fine이 동시에 학습한다.

loss:
  parent CE on train_1000 + AWA cls5 folded parent
  + fine CE on AWA cls5
  + parent prototype/center loss
  + fine sample parent consistency loss
```

구현:

```text
tools/run_joint_identity_fine_v145.py
```

실행:

```powershell
python -m tools.run_joint_identity_fine_v145 `
  --out_dir results\v145_joint_identity_fine `
  --epochs 18 `
  --batch_size 128 `
  --device auto `
  --center_weight 0.35 `
  --sep_weight 0.05 `
  --fine_weight 1.0 `
  --fine_parent_weight 0.5 `
  --seed 145
```

결과:

```text
dataset    parent_head_acc   fine/fold_parent_acc   horse_fp   fine_acc
TEST1      97.25%            97.25%                 0          -
TEST2      95.67%            93.75%                 61         -
TEST4      95.70%            94.10%                 50         -
AWA cls5   95.83%            96.67%                 -          92.33%
```

비교:

```text
v144 frozen fine:
  AWA fine 87.33%
  AWA parent 95.00%

v145 joint:
  AWA fine 92.33%
  AWA parent 96.67%
```

해석:

```text
동시 학습은 freeze 방식보다 내부 fine evidence를 훨씬 덜 잃는다.
즉 identity loss와 fine loss를 같이 두는 방향은 v144보다 낫다.

하지만 TEST1/TEST2/TEST4 broad parent 안정성은 v104a보다 낮다.
특히 TEST2/TEST4에서 broad cat/dog 일반화가 아직 부족하다.
```

결론:

```text
v145는 방향성 검증 성공:
  freeze보다 joint가 낫다.
  identity + fine을 동시에 두면 내부 분해력이 회복된다.

다만 완성형은 아니다.
다음 단계는 v104a의 broad external 안정성을 parent branch에 더 직접 이식하고,
v131의 objrel evidence를 fine branch에 유지하는 two-branch 구조다.
```

## v146/v147 joint loss balance sweep

목표:

```text
v145에서 joint 방향은 확인됐지만 broad parent 안정성이 아직 낮았다.
따라서 fine loss와 parent/prototype loss의 비율을 소폭 조정한다.
```

v146 parent-strong:

```powershell
python -m tools.run_joint_identity_fine_v145 `
  --out_dir results\v146_joint_identity_fine_parent_strong `
  --epochs 18 `
  --batch_size 128 `
  --device auto `
  --center_weight 0.55 `
  --sep_weight 0.08 `
  --fine_weight 0.75 `
  --fine_parent_weight 1.0 `
  --seed 146
```

v147 mid-balance:

```powershell
python -m tools.run_joint_identity_fine_v145 `
  --out_dir results\v147_joint_identity_fine_mid_balance `
  --epochs 18 `
  --batch_size 128 `
  --device auto `
  --center_weight 0.45 `
  --sep_weight 0.06 `
  --fine_weight 0.9 `
  --fine_parent_weight 0.75 `
  --seed 147
```

결과:

```text
model   TEST1 parent   TEST2 parent   TEST4 parent   AWA parent   AWA fine   internal   external
v145    97.25%         95.67%         95.70%         96.67%       92.33%     26         20
v146    97.80%         95.60%         95.90%         95.83%       91.67%     25         25
v147    98.35%         95.33%         96.40%         96.50%       92.83%     22         21
```

해석:

```text
v146은 parent regularization을 강하게 줘서 TEST1/TEST4 parent는 약간 좋아졌지만
fine accuracy가 내려갔다.

v147은 중간형으로, AWA fine accuracy가 92.83%로 가장 높고
internal error도 22로 감소했다.
TEST1/TEST4 parent도 v145보다 좋아졌다.
다만 TEST2 parent는 95.33%로 v145보다 낮다.
```

결론:

```text
현재 joint 계열 중 가장 균형 잡힌 후보는 v147이다.

하지만 아직 v104a broad external 안정성에는 못 미친다.
따라서 다음 큰 단계는 loss weight 미세조정보다,
v104a broad parent branch와 v131/v147 fine evidence branch를
명시적으로 분리한 two-branch identity/fine 구조가 맞다.
```

## 2026-06-12 - v150 corrected TEST2 fullpipeline

Goal:

```text
Run the missing TEST2 corrected v150 evaluation using the same 90-column
v131 fullpipeline feature space as train/TEST1/TEST4.
```

Pipeline:

```text
TEST2 v130 texture relation was the bottleneck.
It was run as 8 restartable shards and then merged:

results\v130_texture_relation_test2_4800_union_sharded\shard_00
...
results\v130_texture_relation_test2_4800_union_sharded\shard_07
results\v130_texture_relation_test2_4800_union_all

Then:
results\tile_view_relation_v05_test2_4800_all
results\v119_class_prototype_gate_test2_4800_parentaware
results\v121_wave_bundle_features_test2_4800_all
results\v130_texture_relation_test2_4800_union_all_fullpipeline
results\v131_texture_object_agreement_test2_4800_union_all_fullpipeline
```

Input compatibility:

```text
train cols = 90
TEST1 cols = 90
TEST2 cols = 90
TEST4 cols = 90
missing = 0
extra = 0
```

v150 output:

```text
results\v150_correct_test1_test2_test4_with_v131_fullpipeline
```

Result:

```text
dataset   n     selected_parent   final_parent   fine
AWA cls5  600   96.50%            100.00%        97.17%
TEST1     182   98.35%             97.25%        broad-label eval
TEST2    4800   95.33%             94.54%        broad-label eval
TEST4    1000   96.40%             96.60%        broad-label eval
```

Parent transition audit:

```text
dataset   recovered(selected wrong -> final correct)   broken(selected correct -> final wrong)
AWA cls5  21                                           0
TEST1      1                                           3
TEST2     82                                         120
TEST4     13                                          11
```

Interpretation:

```text
TEST2 confirms that v150 can recover parent mistakes from fine/object-relation
evidence, but on this broad external set the recovery is not yet protected.

Net effect on TEST2:
  selected_parent correct = 4576 / 4800
  final_parent correct    = 4538 / 4800
  recovery = 82
  breakage = 120
  net = -38

So the v150 structure is useful, but TEST2 shows that the final selector still
needs a broad-parent safety/protection mechanism before it can replace the
v147 parent branch on broad external evaluation.
```

## 2026-06-12 v154/v155 cls10 internal fine-resolution probe

Goal:

```text
v153에서 parent/external 안정성은 충분히 좋아졌으므로,
parent를 바꾸지 않고 같은 parent 내부의 fine class 선택만 개선할 수 있는지 확인한다.

TEST feedback/tuning 없이 train 후보 row만으로 내부 해상도 gate를 학습한다.
```

Inputs:

```text
train candidates:
results\v131_texture_object_agreement_cls10_train_union\texture_object_agreement_scored_bundles.csv

test candidates:
results\v131_texture_object_agreement_cls10_test_union\texture_object_agreement_scored_bundles.csv

base:
results\v153_structured_transition_gate_cls10\all_predictions.csv

parent map:
configs\parent_map_awa_cls10_v153.json
```

New tools:

```text
tools\run_internal_resolution_gate_v154.py
tools\run_internal_pairwise_resolution_v155.py
```

v154:

```text
candidate-level internal scorer
same-parent candidate만 비교
threshold는 train 내부 validation에서 선택

output:
results\v154_internal_resolution_gate_cls10

base v153 fine: 96.80%
v154 fine:      96.90%
parent:         99.60% 유지
fixed/broken:   1 / 0
switch_count:   1
```

v155:

```text
pairwise internal resolver
candidate A - candidate B feature delta로 같은 parent 내부 후보를 비교
threshold는 train 내부 validation에서 선택

output:
results\v155_internal_pairwise_resolution_cls10

base v153 fine: 96.80%
v155 fine:      96.90%
parent:         99.60% 유지
fixed/broken:   1 / 0
switch_count:   2
```

Internal-error decomposition:

```text
v153 wrong total: 32
parent correct but fine wrong: 28
parent wrong: 4

parent correct + fine wrong 중
same-parent true candidate가 이미 있는 샘플: 21
```

Interpretation:

```text
후보 생성 자체는 상당 부분 가능하다.
내부 오답 28개 중 21개는 같은 parent 안에 정답 후보가 이미 있다.

하지만 v154/v155는 train-only 기준으로 안전하게 움직이면 1개만 복구한다.
무조건 best internal candidate를 고르면 오히려 broken이 늘어난다.

따라서 다음 병목은:
  "내부 후보 생성"보다
  "같은 parent 내부에서 독립적/다양한 증거를 종합해 winner를 고르는 selector"
쪽이다.
```

## 2026-06-12 v156/v157 relation-anchored internal resolver

Motivation:

```text
필요한 구조:

        Cat Identity
        /          \
   Persian        Siamese
        \          /
      Relation Space

즉 fine class는 같은 parent identity를 공유하되,
relation/evidence space에서는 서로 구분되어야 한다.
```

v156:

```text
tools\run_relation_anchored_fine_resolver_v156.py

train correct same-parent 후보로 parent/fine centroid를 만들고,
candidate feature에 다음 anchor feature를 추가:

v156_parent_anchor_cos
v156_fine_anchor_cos
v156_sibling_anchor_max
v156_sibling_margin
v156_parent_vs_sibling_gap
v156_identity_relation_score

output:
results\v156_relation_anchored_fine_resolver_cls10
```

v156 result:

```text
base v153 fine: 96.80%
v156 fine:      96.80%
parent:         99.60% 유지
fixed/broken:   0 / 0

diagnostic:
threshold를 풀면 fixed 5 / broken 6 수준이라,
global candidate scorer + anchor feature만으로는 sibling fine 선택이 불안정함.
```

v157:

```text
tools\run_parent_local_contrast_resolver_v157.py

parent_to_fine에서 parent 내부 sibling pair를 자동 생성하고,
각 parent pair별 contrast resolver를 train only로 학습.

예:
cat:      persian+cat vs siamese+cat
dog:      chihuahua vs german+shepherd
dog_like: wolf vs fox
big_cat:  lion vs tiger

하드룰이 아니라 parent_to_fine 관계에서 자동 생성되는 local contrast model.
```

v157 result:

```text
output:
results\v157_parent_local_contrast_resolver_cls10

base v153 fine: 96.80%
v157 fine:      97.50%
parent:         99.60% 유지
fixed/broken:   7 / 0
switch_count:   9
```

v157 fixed samples:

```text
chihuahua:          german+shepherd -> chihuahua
fox:                wolf -> fox
german+shepherd:    chihuahua -> german+shepherd  (4 samples)
siamese+cat:        persian+cat -> siamese+cat
```

Per-class after v157:

```text
chihuahua          100%
deer               100%
fox                 97%
german+shepherd     96%
horse               97%
lion               100%
persian+cat         92%
siamese+cat         95%
tiger              100%
wolf                98%
```

Interpretation:

```text
v154/v155/v156의 global scorer 방식은 내부 해상도를 크게 올리지 못했다.
v157의 parent-local contrast 방식은 내부 sibling pair를 직접 비교하면서
정답 후보 선택 능력이 살아났다.

이는 "하위 evidence가 상위 identity를 보완한다"는 v150/v153 결과에 이어,
"같은 상위 identity 내부에서는 local contrast resolver가 필요하다"는 증거다.

현재 cls10 split에서는 v157 fine 97.50%로,
기존 Full ResNet50 baseline 97.40%를 소폭 넘었다.
```

External broad-set safety check:

```text
Goal:
v157이 cls10 내부 fine split을 개선했지만,
기존 broad cat/dog 외부 TEST1/TEST2/TEST4 parent 성능을 깨뜨리는지 확인한다.

Method:
v153 parentbaseline의 TEST1/TEST2/TEST4 결과를 base로 두고,
v157 cls10 train-only local contrast resolver를 적용.

주의:
TEST1/2/4는 y_true가 broad label(cat/dog)이므로
fine accuracy가 아니라 final_pred -> parent 기준 broad parent accuracy를 계산한다.
```

Output:

```text
results\v157_external_validation_cls10_on_broad\test1
results\v157_external_validation_cls10_on_broad\test2
results\v157_external_validation_cls10_on_broad\test4
results\v157_external_validation_cls10_on_broad\summary_broad_parent.csv
```

Result:

```text
dataset   n     base parent   v157 parent   delta   switched   fixed/broken broad
TEST1     182   98.90%        98.90%       0.00    54         0 / 0
TEST2    4800   96.98%        96.98%       0.00    1275       0 / 0
TEST4    1000   97.70%        97.70%       0.00    313        0 / 0

mean            97.86%        97.86%       0.00    1642       0 / 0
```

Interpretation:

```text
v157은 broad set에서도 많은 fine label 교체를 수행하지만,
교체가 같은 parent 내부에서만 일어나므로 cat/dog parent 판정은 전혀 깨지지 않았다.

따라서 현재 구조에서는:
  external/broad stability = v153 수준 유지
  internal/fine resolution = cls10에서 v157이 개선
로 분리되어 작동한다.
```

## 2026-06-13 Oxford cls10 external data and missing-7 source collection

Oxford-IIIT Pet external validation:

```text
source:
https://www.robots.ox.ac.uk/~vgg/data/pets/

raw:
dataset\external\oxford_iiit_pet_raw\images.tar.gz
dataset\external\oxford_iiit_pet_raw\annotations.tar.gz

prepared:
dataset\external\oxford_iiit_pet_prepared\all37
dataset\external\oxford_iiit_pet_prepared\cls10

script:
tools\prepare_oxford_pet_external.py
```

Oxford prepared counts:

```text
all37:
train 3680
test  3669

cls10:
train 999
test  1000
```

Oxford cls10 labels:

```text
beagle
bengal+cat
chihuahua
persian+cat
pomeranian
russian+blue+cat
samoyed
siamese+cat
sphynx+cat
yorkshire+terrier
```

Directly known by existing AWA cls10 model:

```text
chihuahua
persian+cat
siamese+cat
```

Missing fine labels for direct 10-class external validation:

```text
beagle
bengal+cat
pomeranian
russian+blue+cat
samoyed
sphynx+cat
yorkshire+terrier
```

External non-Oxford sources collected for missing 7:

```text
cat source:
AtharvaTaras/Cat-Breeds-Dataset
dataset\external_sources\cat_breeds_atharva_main.zip

dog source:
Stanford Dogs
dataset\external_sources\stanford_dogs_images.tar
```

Prepared missing-7 train source:

```text
script:
tools\prepare_missing7_external_sources.py

output:
dataset\external_train_sources\oxford_missing7_from_other_sources

counts:
bengal+cat            233
russian+blue+cat      205
sphynx+cat            302
beagle                195
pomeranian            219
samoyed               218
yorkshire+terrier     164
total                1536
```

Interpretation:

```text
Oxford cls10 test can remain external validation.
The 7 labels absent from AWA cls10 now have non-Oxford training/reference images.

This enables a cleaner next experiment:
  train/extend output space using AWA cls10 + missing7 external sources
  evaluate on Oxford cls10 test without using Oxford train
```

## 2026-06-14 - AWA cls10 external cap80 mixed test

Goal:

```text
기존 AWA cls10 모델이 학습한 클래스 그대로
외부 출처 이미지만 묶은 TEST를 만들어 일반화 성능을 확인한다.
```

Target class list:

```text
persian+cat
siamese+cat
chihuahua
german+shepherd
wolf
fox
lion
tiger
horse
deer
```

External mixed cap80 dataset:

```text
dataset:
dataset\external_test_sources\awa_cls10_external_test_mixed_cap80

source mix:
Oxford-IIIT Pet:
  persian+cat, siamese+cat, chihuahua

Stanford Dogs:
  german+shepherd

Open Images:
  wolf, fox, lion, tiger, horse, deer

counts:
persian+cat         80
siamese+cat         80
chihuahua           80
german+shepherd     80
wolf                19
fox                 18
lion                80
tiger               80
horse               80
deer                80
total              677
```

Pipeline:

```text
script:
tools\run_awa_cls10_external_cap80_pipeline.ps1

important condition:
external TEST images are used only for evaluation.
train artifacts are reused from existing AWA cls10 train outputs.

v130 texture relation stage is run with:
--num_shards 4
--limit 0
```

Main outputs:

```text
base:
results\eval_stage0_v02_awa_cls10_external_cap80

v153:
results\v153_structured_transition_gate_awa_cls10_external_cap80

v157:
results\v157_parent_local_contrast_resolver_awa_cls10_external_cap80
```

Result summary:

```text
model / stage                         accuracy
------------------------------------------------
MVP v0.2 fine                         73.86%
v150 candidate fine                   84.64%
v153 gated fine                       85.97%

selected parent                       76.07%
v150 candidate parent                 90.40%
v153 gated parent                     91.73%

v157 fine                             74.15%
v157 parent                           91.73%
```

Transition analysis:

```text
v150 vs selected parent:
fixed   107
broken   10

v153 vs selected parent:
fixed   107
broken    1

v157 vs v153:
fixed     1
broken   81
switch   89
```

Interpretation:

```text
v153 structured transition gate generalizes meaningfully to this mixed external
cap80 set.  It keeps almost all of v150's recovery while reducing parent-level
breakage from 10 to 1.

v157 parent-local resolver is not externally safe here.  It preserves parent
accuracy but collapses fine accuracy because the learned dog-pair resolver
over-switches german+shepherd to chihuahua on this external source.

Therefore:
  use v153 as the current safe external cls10 result
  treat v157 as an internal/fine resolver that needs domain-safety gating before
  being applied to broad external validation
```

## 2026-06-14 - v158 single-child parent probe

Background:

```text
cat/dog have multiple fine children:
  cat = persian+cat, siamese+cat
  dog = chihuahua, german+shepherd

These fine children can reinforce parent identity.

But horse/deer are singleton parents in v153:
  horse = horse
  deer  = deer

So they have no internal lower-level evidence loop.
```

### v158a - pseudo-subclass prototypes for singleton parents

Tool:

```text
tools\run_single_child_pseudo_subclass_v158.py
```

Run:

```powershell
.\.venv\Scripts\python.exe -m tools.run_single_child_pseudo_subclass_v158 `
  --train_candidates results\v131_texture_object_agreement_cls10_train_union\texture_object_agreement_scored_bundles.csv `
  --train_base results\stage0_v02_awa_cls10_resnet18\train_predictions_keyed.csv `
  --eval_candidates results\v131_texture_object_agreement_awa_cls10_external_cap80_union\texture_object_agreement_scored_bundles.csv `
  --eval_base results\v153_structured_transition_gate_awa_cls10_external_cap80\all_predictions.csv `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --parent_to_fine configs\parent_to_fine_awa_cls10_v153.json `
  --out_dir results\v158_single_child_pseudo_subclass_external_cap80 `
  --clusters 4 `
  --score_weight 0.25 `
  --seed 158
```

Result:

```text
single-child labels:
deer, horse

train prototype rows:
horse 192 rows, 4 clusters
deer  198 rows, 4 clusters

external cap80:
base fine    85.97%
v158 fine    85.97%
base parent  91.73%
v158 parent  91.73%
fixed 0
broken 0
switch 0
```

Interpretation:

```text
The conservative train-only threshold produced no external switch.
The idea is safe but not yet useful as a policy.

Additional inspection showed many true horse samples received deer-like pseudo
support.  This suggests horse/deer should first share a broader parent space,
rather than being independently reinforced as singleton parents.
```

### v158b - horse/deer as ungulate parent

Added configs:

```text
configs\parent_map_awa_cls10_v158_ungulate.json
configs\parent_to_fine_awa_cls10_v158_ungulate.json
```

Parent change:

```text
before:
horse -> horse
deer  -> deer

after:
horse -> ungulate
deer  -> ungulate
```

Run:

```powershell
.\.venv\Scripts\python.exe -m tools.run_structured_transition_gate_v153 `
  --out_dir results\v158_ungulate_parent_v153_external_cap80 `
  --train_fine_csv results\v131_texture_object_agreement_cls10_train_union\texture_object_agreement_scored_bundles.csv `
  --train_parent_csv results\stage0_v02_awa_cls10_resnet18\train_predictions_keyed.csv `
  --eval_parent_csv "external_cap80=results\eval_stage0_v02_awa_cls10_external_cap80\predictions_keyed.csv" `
  --eval_fine_csv "external_cap80=results\v131_texture_object_agreement_awa_cls10_external_cap80_union\texture_object_agreement_scored_bundles.csv" `
  --parent_map_json configs\parent_map_awa_cls10_v158_ungulate.json `
  --parent_to_fine_json configs\parent_to_fine_awa_cls10_v158_ungulate.json `
  --parent_labels cat,dog,big_cat,dog_like,ungulate `
  --train_gate_baseline parent_csv `
  --threshold 0.05
```

Result:

```text
external cap80:
selected parent       77.10%
v150 candidate parent 91.43%
v153 parent           88.92%

v150 fixed / broken   105 / 8
v153 fixed / broken    80 / 0

v150 fine             84.64%
v153 fine             84.49%
```

Interpretation:

```text
Ungulate grouping is safer in the sense that v153 broken becomes 0, but the
gate becomes too conservative and loses many fixes.

So the underlying idea is still plausible:
  singleton parents need a shared evidence space

but simply changing parent_map is not enough.
The next version should learn a positive shared-parent support signal instead
of relying only on v153's transition approval.
```

## 2026-06-14 - v159 evidence-cluster pseudo-subclass k=2

Goal:

```text
Do not manually define every parent/contrast relation.

Instead:
  split each fine class's positive TRAIN evidence into 2 pseudo-subclasses
  treat them as observation/evidence modes
  let a train-only selector learn candidate correctness from:
    original v131 Object/Identity/Attribute/Relation features
    + pseudo-subclass fit features
```

Tool:

```text
tools\run_evidence_cluster_pseudo_subclass_v159.py
```

Run:

```powershell
.\.venv\Scripts\python.exe -m tools.run_evidence_cluster_pseudo_subclass_v159 `
  --train_candidates results\v131_texture_object_agreement_cls10_train_union\texture_object_agreement_scored_bundles.csv `
  --train_base results\stage0_v02_awa_cls10_resnet18\train_predictions_keyed.csv `
  --eval_candidates results\v131_texture_object_agreement_awa_cls10_external_cap80_union\texture_object_agreement_scored_bundles.csv `
  --eval_base results\v153_structured_transition_gate_awa_cls10_external_cap80\all_predictions.csv `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --out_dir results\v159_evidence_cluster_k2_external_cap80 `
  --k 2 `
  --val_ratio 0.25 `
  --seed 159
```

Result:

```text
train-only threshold:
0.999705

v153 base fine:
85.97%

v159 switched final:
85.97%

switch:
0

top-candidate if forced:
85.23%
```

Top-candidate class comparison:

```text
class             base     v159 top-candidate
horse             62.50%   52.50%
lion              56.25%   57.50%
fox               72.22%   61.11%
wolf              52.63%   73.68%
deer              87.50%   88.75%
tiger             92.50%   92.50%
german+shepherd  100.00%   98.75%
chihuahua        100.00%  100.00%
persian+cat      100.00%  100.00%
siamese+cat      100.00%  100.00%
```

Top-candidate recovery/breakage:

```text
recoverable by v159 top candidate while v153 was wrong:
wolf 4
lion 2
deer 1

broken by v159 top candidate while v153 was correct:
horse 8
fox 2
german+shepherd 1
lion 1
```

Interpretation:

```text
The k=2 pseudo-subclass idea does create useful signal for some weak regions
(especially wolf), but direct candidate selection is not yet stable.

The main failure is horse:
  v153 horse 62.5%
  v159 top horse 52.5%

So the problem is not merely "make two internal modes".
The pseudo modes must be tied to a shared/contrast objective that protects
existing good evidence and does not let another class's mode dominate.

Current conclusion:
  pseudo-subclass clustering is promising as an added feature,
  but not as a standalone selector.
```

## 2026-06-14 - v160 confusion-directed contrast

Goal:

```text
v159의 단순 내부 cluster는 방향성이 부족했다.

v160은 train에서 자동으로:
  "이 class가 어떤 다른 class로 빨리기 쉬운가"
를 찾고, 그 방향별 pairwise contrast probe를 학습한다.

This is a discarded-evidence contrast attempt:
  common class objective가 버린 증거를
  confusion direction별 contrast로 다시 활성화한다.
```

Tool:

```text
tools\run_confusion_directed_contrast_v160.py
```

Implementation note:

```text
Initial top_k=3 run was too slow because pair probes were called row-by-row.
The tool was updated to batch predict by directed pair.

For quick validation, top_k_neighbors=1 was run first.
```

Run:

```powershell
.\.venv\Scripts\python.exe -m tools.run_confusion_directed_contrast_v160 `
  --train_candidates results\v131_texture_object_agreement_cls10_train_union\texture_object_agreement_scored_bundles.csv `
  --train_base results\stage0_v02_awa_cls10_resnet18\train_predictions_keyed.csv `
  --eval_candidates results\v131_texture_object_agreement_awa_cls10_external_cap80_union\texture_object_agreement_scored_bundles.csv `
  --eval_base results\v153_structured_transition_gate_awa_cls10_external_cap80\all_predictions.csv `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --out_dir results\v160_confusion_directed_contrast_top1_external_cap80 `
  --top_k_neighbors 1 `
  --val_ratio 0.25 `
  --seed 160
```

Automatically discovered train-only neighbors:

```text
horse -> deer
wolf  -> fox
fox   -> wolf
persian+cat -> siamese+cat
siamese+cat -> persian+cat
german+shepherd -> wolf
deer -> fox
lion -> deer
tiger -> fox
chihuahua -> siamese+cat
```

Result:

```text
train-only threshold:
0.999947

v153 base fine:
85.97%

v160 switched final:
85.97%

switch:
0

top-candidate if forced:
84.93%
```

Top-candidate comparison:

```text
class             v153 base   v160 top
horse              62.50%     52.50%
lion               56.25%     57.50%
fox                72.22%     61.11%
wolf               52.63%     68.42%
deer               87.50%     88.75%
tiger              92.50%     91.25%
siamese+cat       100.00%     98.75%
chihuahua         100.00%    100.00%
persian+cat       100.00%    100.00%
german+shepherd   100.00%    100.00%
```

Potential fixed/broken if top-candidate were forced:

```text
recoverable:
wolf 3
lion 2
deer 1

broken:
horse 8
fox 2
lion 1
siamese+cat 1
tiger 1
```

Interpretation:

```text
The idea is partially validated:
  train-only confusion directions are meaningful
  wolf improves from 52.6% to 68.4% in top-candidate mode

But the selector is not yet safe:
  horse drops from 62.5% to 52.5%
  forced top-candidate is below v153

Therefore v160 should not replace v153 yet.

The useful next direction is not "stronger top-candidate selection".
It is to use confusion-directed contrast as a risk/evidence feature inside the
v153 gate, especially for classes whose top-candidate improves without breaking
stable classes.
```

## 2026-06-14 - v161 taxonomy parent grouping

Goal:

```text
Check whether a more taxonomic parent grouping improves external parent
stability.

Previous v153:
  big_cat = lion, tiger
  dog_like = wolf, fox

v161:
  cat = persian+cat, siamese+cat, lion, tiger
  dog = chihuahua, german+shepherd, wolf, fox
  horse = horse
  deer = deer
```

Configs:

```text
configs\parent_map_awa_cls10_v161_taxonomy.json
configs\parent_to_fine_awa_cls10_v161_taxonomy.json
```

Run:

```powershell
.\.venv\Scripts\python.exe -m tools.run_structured_transition_gate_v153 `
  --out_dir results\v161_taxonomy_parent_v153_external_cap80 `
  --train_fine_csv results\v131_texture_object_agreement_cls10_train_union\texture_object_agreement_scored_bundles.csv `
  --train_parent_csv results\stage0_v02_awa_cls10_resnet18\train_predictions_keyed.csv `
  --eval_parent_csv "external_cap80=results\eval_stage0_v02_awa_cls10_external_cap80\predictions_keyed.csv" `
  --eval_fine_csv "external_cap80=results\v131_texture_object_agreement_awa_cls10_external_cap80_union\texture_object_agreement_scored_bundles.csv" `
  --parent_map_json configs\parent_map_awa_cls10_v161_taxonomy.json `
  --parent_to_fine_json configs\parent_to_fine_awa_cls10_v161_taxonomy.json `
  --parent_labels cat,dog,horse,deer `
  --train_gate_baseline parent_csv `
  --threshold 0.05
```

Result:

```text
external cap80:

v153 original parent:
91.73%

v161 taxonomy parent:
92.32%

v153 original fine:
85.97%

v161 taxonomy fine:
85.82%
```

Class-level parent comparison:

```text
class             original parent   taxonomy parent
horse             62.50%           62.50%
deer              87.50%           87.50%
lion              87.50%           90.00%
tiger             93.75%           95.00%
fox               94.44%          100.00%
wolf             100.00%          100.00%
cat/dog known    100.00%          100.00%
```

Interpretation:

```text
Taxonomic grouping helps parent stability slightly.

Putting wolf/fox into dog and lion/tiger into cat is not harmful here; it
improves fox, lion, and tiger parent accuracy while preserving the already
stable known cat/dog classes.

The remaining hard failure is still horse:
  horse parent accuracy remains 62.5%

So the next special problem is not cat/dog/big-cat/canine grouping.
It is horse identity getting pulled into cat/dog spaces.
```

## 2026-06-14 - v162 shared parent/fine evidence gate

Goal:

```text
Parent contrast and fine contrast should not be independent heads.

Instead:
  compute parent support from the same candidate evidence table
  evaluate fine candidates together with their parent support

For multi-child parents:
  fine evidence is interpreted under parent support.

For single-child parents:
  use parent evidence strength/diversity instead of fake fine labels.
```

Tool:

```text
tools\run_shared_parent_fine_gate_v162.py
```

Run:

```powershell
.\.venv\Scripts\python.exe -m tools.run_shared_parent_fine_gate_v162 `
  --train_candidates results\v131_texture_object_agreement_cls10_train_union\texture_object_agreement_scored_bundles.csv `
  --train_base results\stage0_v02_awa_cls10_resnet18\train_predictions_keyed.csv `
  --eval_candidates results\v131_texture_object_agreement_awa_cls10_external_cap80_union\texture_object_agreement_scored_bundles.csv `
  --eval_base results\v153_structured_transition_gate_awa_cls10_external_cap80\all_predictions.csv `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --parent_to_fine configs\parent_to_fine_awa_cls10_v153.json `
  --out_dir results\v162_shared_parent_fine_gate_external_cap80 `
  --val_ratio 0.25 `
  --seed 162
```

Result:

```text
train-only threshold:
0.999442

v153 base fine:
85.97%

v162 switched final:
85.97%

switch:
0

top-candidate if forced:
85.08%
```

Top-candidate comparison:

```text
class             v153 base   v162 top
horse              62.50%     52.50%
lion               56.25%     55.00%
wolf               52.63%     57.89%
fox                72.22%     77.78%
deer               87.50%     88.75%
tiger              92.50%     92.50%
known cat/dog     100.00%    100.00%
```

Potential fixed/broken if top-candidate were forced:

```text
recoverable:
deer 1
fox 1
lion 1
wolf 1

broken:
horse 8
lion 2
```

Interpretation:

```text
v162 confirms that simply adding parent support to fine candidate scoring is
not enough.

It slightly helps fox/deer/wolf top-candidate behavior but still damages horse.

The horse issue appears deeper:
  the candidate evidence itself is often pulled into dog/cat spaces,
  so parent support derived from the same candidate table inherits that pull.

Next likely direction:
  horse needs a stronger independent observation/evidence path, not just a
  different selector over the existing candidate table.
```

## v163 parent-first evidence selector external cap80

Goal:

```text
Move parent/fine coupling earlier than v162.

Instead of:
  fine bundle first
  -> attach parent support later

Use:
  v113 bbox/view candidates
  -> parent support view set first
  -> fine evidence inside that parent-supported view set
```

Tool:

```text
tools\run_parent_first_evidence_selector_v163.py
```

Run:

```powershell
.\.venv\Scripts\python.exe -m tools.run_parent_first_evidence_selector_v163 `
  --train_bbox_csv results\v113_bbox_cache_awa_cls10_train\bbox_candidate_scores.csv `
  --eval_bbox_csv results\v113_bbox_cache_awa_cls10_external_cap80\bbox_candidate_scores.csv `
  --eval_base results\v153_structured_transition_gate_awa_cls10_external_cap80\all_predictions.csv `
  --out_dir results\v163_parent_first_evidence_external_cap80 `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --parent_to_fine configs\parent_to_fine_awa_cls10_v153.json `
  --seed 163
```

Important condition:

```text
The selector and threshold are trained from train bbox candidates only.
external cap80 is used only for evaluation.
```

Result:

```text
v153 base fine:
85.97%

v153 base parent:
91.73%

v163 final fine:
85.52%

v163 final parent:
91.29%

fixed:
0

broken:
3

approved candidate count:
216
```

Forced top-candidate diagnostic:

```text
fine:
74.15%

parent:
76.07%

fixed:
1

broken:
81
```

Per-class final:

```text
class             v153 base   v163 final
persian+cat       100.00%     100.00%
siamese+cat       100.00%     100.00%
chihuahua         100.00%     100.00%
german+shepherd   100.00%     100.00%
tiger              92.50%      92.50%
deer               87.50%      87.50%
fox                72.22%      72.22%
horse              62.50%      61.25%
lion               56.25%      56.25%
wolf               52.63%      42.11%
```

Interpretation:

```text
v163 did not improve external cap80.

This is still useful:
  parent-first support computed from v113 bbox probabilities is not enough.

The failure mode is similar to v162 but clearer:
  if the underlying bbox candidate evidence is already pulled toward
  dog/cat-like regions, parent-first aggregation inherits the same pull.

So the next step should not be another selector over the same probability table.
It should improve the evidence source itself:
  independent parent evidence path
  singleton-parent evidence diversity
  anti-confusion observation features
  or a training loss that preserves parent identity before fine separation.
```

## v164 fine evidence negative ablation

Goal:

```text
Instead of asking why subclasses help, ask what makes subclass-supported
classes worse.

If removing/corrupting fine evidence also damages parent accuracy, then fine
subclass evidence is not only an internal classifier.  It is also helping parent
identity.
```

Tool:

```text
tools\run_fine_evidence_negative_ablation_v164.py
```

Runs:

```powershell
.\.venv\Scripts\python.exe -m tools.run_fine_evidence_negative_ablation_v164 `
  --bbox_csv results\v113_bbox_cache_awa_cls10_test\bbox_candidate_scores.csv `
  --v153_predictions results\v153_structured_transition_gate_cls10\all_predictions.csv `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --parent_to_fine configs\parent_to_fine_awa_cls10_v153.json `
  --out_dir results\v164_fine_negative_ablation_awa_cls10 `
  --shuffle_rate 1.0 `
  --seed 164

.\.venv\Scripts\python.exe -m tools.run_fine_evidence_negative_ablation_v164 `
  --bbox_csv results\v113_bbox_cache_awa_cls10_external_cap80\bbox_candidate_scores.csv `
  --v153_predictions results\v153_structured_transition_gate_awa_cls10_external_cap80\all_predictions.csv `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --parent_to_fine configs\parent_to_fine_awa_cls10_v153.json `
  --out_dir results\v164_fine_negative_ablation_external_cap80 `
  --shuffle_rate 1.0 `
  --seed 164
```

Ablations:

```text
parent_only:
  remove fine choice inside parent; parent score only.

no_sibling_contrast:
  keep class probabilities, but remove sibling-vs-sibling contrast.

fine_shuffle:
  rotate fine labels inside each parent; parent labels remain conceptually same,
  but the aligned fine evidence is corrupted.
```

Summary:

```text
AWA cls10 baseline v153:
fine   96.80%
parent 99.60%

ablation             fine     parent   fine delta   parent delta
parent_only          55.30%   93.30%   -41.50%      -6.30%
no_sibling           91.40%   93.50%   -5.40%       -6.10%
fine_shuffle         19.10%   93.80%   -77.70%      -5.80%

external cap80 baseline v153:
fine   85.97%
parent 91.73%

ablation             fine     parent   fine delta   parent delta
parent_only          44.17%   79.76%   -41.80%      -11.96%
no_sibling           76.66%   80.21%   -9.31%       -11.52%
fine_shuffle         15.21%   80.35%   -70.75%      -11.37%
```

Interpretation:

```text
The important result is no_sibling_contrast.

Removing sibling contrast does not merely reduce fine accuracy.
It also reduces parent accuracy:
  AWA cls10      -6.10%p parent
  external cap80 -11.52%p parent

This supports the hypothesis:
  aligned fine/subclass evidence contributes to parent identity.

So the strong cat/dog subclass behavior is not just "fine classification after
parent classification".  The subclass relation itself appears to stabilize the
larger identity space.
```

Implication:

```text
For singleton classes such as horse/deer, forcing pseudo subclasses blindly is
not enough.

What is needed is a learned way to create sibling-like contrast/evidence
channels without corrupting the parent space:
  parent identity preservation
  sibling-style contrast when real subclasses exist
  evidence-diversity/anti-confusion channels when no real subclass exists
```

## v165 additive perturbation: what addition weakens fine evidence?

Goal:

```text
v164 removed/corrupted fine evidence.
v165 asks the opposite question:
  while fine evidence exists, what can be added that weakens fine separation?
```

Tool:

```text
tools\run_fine_evidence_additive_perturbation_v165.py
```

Perturbations:

```text
parent_blend:
  blend each fine probability toward the mean probability of its siblings.
  This adds broad parent/common signal but does not add a new output class.

global_blend:
  blend every class probability toward the global class mean.

parent_pseudo:
  add explicit broad parent candidates such as parent::cat, parent::dog.
  If a parent pseudo-class wins, parent may be correct but fine is unresolved.
```

Runs:

```powershell
.\.venv\Scripts\python.exe -m tools.run_fine_evidence_additive_perturbation_v165 `
  --bbox_csv results\v113_bbox_cache_awa_cls10_test\bbox_candidate_scores.csv `
  --v153_predictions results\v153_structured_transition_gate_cls10\all_predictions.csv `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --parent_to_fine configs\parent_to_fine_awa_cls10_v153.json `
  --out_dir results\v165_fine_additive_perturbation_awa_cls10_pseudo `
  --alphas 0.5,0.75,1.0,1.25

.\.venv\Scripts\python.exe -m tools.run_fine_evidence_additive_perturbation_v165 `
  --bbox_csv results\v113_bbox_cache_awa_cls10_external_cap80\bbox_candidate_scores.csv `
  --v153_predictions results\v153_structured_transition_gate_awa_cls10_external_cap80\all_predictions.csv `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --parent_to_fine configs\parent_to_fine_awa_cls10_v153.json `
  --out_dir results\v165_fine_additive_perturbation_external_cap80_pseudo `
  --alphas 0.5,0.75,1.0,1.25
```

Key result:

```text
Mild parent/common blending does not strongly weaken fine evidence.

But adding a broad parent pseudo-class does:
  fine collapses,
  parent remains relatively high.
```

Summary at alpha/beta 1.0:

```text
AWA cls10 baseline v153:
fine   96.80%
parent 99.60%

perturbation       fine     parent   fine delta   parent delta
parent_blend       55.80%   93.10%   -41.00%      -6.50%
parent_pseudo      19.00%   93.30%   -77.80%      -6.30%

external cap80 baseline v153:
fine   85.97%
parent 91.73%

perturbation       fine     parent   fine delta   parent delta
parent_blend       45.05%   80.21%   -40.92%      -11.52%
parent_pseudo      19.50%   79.76%   -66.47%      -11.96%
```

Class examples:

```text
External cap80 parent_pseudo beta=1.0:

class             v153 fine   pseudo fine   v153 parent   pseudo parent
persian+cat       100.00%       8.75%        100.00%       100.00%
siamese+cat       100.00%      12.50%        100.00%       100.00%
german+shepherd   100.00%      20.00%        100.00%       100.00%
chihuahua         100.00%      25.00%        100.00%        97.50%
tiger              92.50%       1.25%         93.75%        82.50%
```

Interpretation:

```text
Fine evidence is not easily weakened by merely adding a little common signal.
It is weakened when the common/parent signal becomes an explicit competing
answer.

This suggests a design rule:
  parent identity should guide or regularize fine evidence,
  but it should not compete as an equal answer head unless the system supports
  an explicit "parent known, fine unresolved" state.
```

Implication:

```text
For future hierarchy:
  good:
    object is cat
    fine evidence says persian/siamese
    parent stabilizes fine

  risky:
    object is cat
    parent::cat competes directly with persian/siamese
    fine evidence gets absorbed by parent answer
```

## v166 evidence specificity / shared attraction audit

Goal:

```text
Test the hypothesis:
  cat/dog and related class spaces are broad.
  Some evidence that looks like a class may actually be shared by other classes.

The gate should eventually distinguish:
  class-unique evidence
  vs
  shared evidence that causes attraction.
```

Tool:

```text
tools\build_evidence_specificity_audit_v166.py
```

Runs:

```powershell
.\.venv\Scripts\python.exe -m tools.build_evidence_specificity_audit_v166 `
  --train_bbox_csv results\v113_bbox_cache_awa_cls10_train\bbox_candidate_scores.csv `
  --eval_bbox_csv results\v113_bbox_cache_awa_cls10_test\bbox_candidate_scores.csv `
  --out_dir results\v166_evidence_specificity_awa_cls10 `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --min_rows 20

.\.venv\Scripts\python.exe -m tools.build_evidence_specificity_audit_v166 `
  --train_bbox_csv results\v113_bbox_cache_awa_cls10_train\bbox_candidate_scores.csv `
  --eval_bbox_csv results\v113_bbox_cache_awa_cls10_external_cap80\bbox_candidate_scores.csv `
  --out_dir results\v166_evidence_specificity_external_cap80 `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --min_rows 20
```

Outputs:

```text
specificity_table.csv
train_class_cross_presence_matrix.csv
eval_rows_with_specificity.csv
sample_candidate_specificity.csv
raw_vs_true_specificity_compare.csv
raw_wrong_with_train_cross_presence.csv
summary.json
```

Result 1: bbox-level specificity alone is not enough.

```text
External cap80 raw top:
fine   76.96%
parent 80.35%

External cap80 unique-weighted top:
fine   77.10%
parent 80.50%

raw wrong corrected by unique top:
1 / 156
```

Interpretation:

```text
Simple bbox specificity weighting does not directly solve the problem.
The per-bbox specificity estimate is too local and often still treats a false
attraction as specific enough.
```

Result 2: class-pair cross-presence supports the shared-evidence hypothesis.

Top train cross-presence pairs excluding self:

```text
true class      candidate       cross ratio to candidate self   rate >= 0.50
fox             wolf            0.3815                          25.5%
horse           deer            0.2902                          18.0%
wolf            fox             0.2544                          15.0%
lion            deer            0.1850                           9.0%
siamese+cat     persian+cat     0.1603                           7.5%
fox             deer            0.1525                           6.0%
persian+cat     siamese+cat     0.1466                           7.0%
lion            wolf            0.1460                           3.0%
deer            wolf            0.1453                           4.0%
persian+cat     fox             0.1319                           7.0%
```

External cap80 raw wrong pairs joined with train cross-presence:

```text
true -> predicted    external n   train cross ratio   train rate >= 0.50
horse -> deer        14           0.2902              18.0%
deer -> fox          10           0.1160               3.5%
lion -> tiger        10           0.0564               1.0%
wolf -> fox           9           0.2544              15.0%
deer -> wolf          8           0.1453               4.0%
lion -> deer          5           0.1850               9.0%
lion -> wolf          4           0.1460               3.0%
fox -> wolf           3           0.3815              25.5%
```

Interpretation:

```text
Many external failures are not random.
They correspond to class pairs that already have measurable cross-presence in
the train set.

This supports the user's hypothesis:
  the model is seeing evidence that exists in both spaces,
  then the graph/gate must infer whether it is own-class identity or
  other-class shared evidence.
```

Design implication:

```text
v166a specificity table is useful as an audit, not yet as a direct selector.

The next useful feature is not "raw specificity-weighted score" alone.
It should be a gate feature:
  raw score
  specificity score
  cross-presence ratio for true->candidate-like pair
  parent/fine support diversity
  relation/scale mismatch

For example:
  horse -> deer can be recognized as a high-cross-presence attraction pair.
  wolf <-> fox can be recognized as a highly shared evidence pair.
```

## v167 shared-attraction gate

Goal:

```text
Turn v166 cross-presence audit into selector/gate features.

Features include:
  raw score
  unique/shared score
  raw-unique gap
  candidate broadness across other train classes
  rival->candidate cross shadow
  candidate->rival reverse shadow
  same-parent and other-parent margins
```

Tool:

```text
tools\run_shared_attraction_gate_v167.py
```

Runs:

```powershell
.\.venv\Scripts\python.exe -m tools.run_shared_attraction_gate_v167 `
  --train_bbox_csv results\v113_bbox_cache_awa_cls10_train\bbox_candidate_scores.csv `
  --eval_bbox_csv results\v113_bbox_cache_awa_cls10_test\bbox_candidate_scores.csv `
  --eval_base results\v153_structured_transition_gate_cls10\all_predictions.csv `
  --out_dir results\v167_shared_attraction_gate_awa_cls10 `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --seed 167

.\.venv\Scripts\python.exe -m tools.run_shared_attraction_gate_v167 `
  --train_bbox_csv results\v113_bbox_cache_awa_cls10_train\bbox_candidate_scores.csv `
  --eval_bbox_csv results\v113_bbox_cache_awa_cls10_external_cap80\bbox_candidate_scores.csv `
  --eval_base results\v153_structured_transition_gate_awa_cls10_external_cap80\all_predictions.csv `
  --out_dir results\v167_shared_attraction_gate_external_cap80 `
  --parent_map configs\parent_map_awa_cls10_v153.json `
  --seed 167
```

Train-only threshold:

```text
0.997185
```

Result:

```text
AWA cls10:
base v153 fine     96.80%
v167 gated fine    96.80%
base parent        99.60%
v167 parent        99.60%
fixed/broken       0 / 0

External cap80:
base v153 fine     85.97%
v167 gated fine    85.97%
base parent        91.73%
v167 parent        91.73%
fixed/broken       0 / 0
```

Forced top-candidate diagnostic:

```text
AWA cls10 forced top:
fine   93.30%
parent 95.60%
fixed/broken 3 / 38

External cap80 forced top:
fine   77.55%
parent 80.50%
fixed/broken 3 / 60
```

Interpretation:

```text
v167 did not improve accuracy as a replacement selector.

The train-only threshold is conservative and usually approves candidates that
match the existing v153 output.  This gives no breakage, but also no recovery.

Forced top-candidate is worse, which confirms that shared-attraction features
alone are not enough to choose a new label.
```

Important lesson:

```text
v166/v167 features are useful as risk/diagnostic signals, not yet as a final
answer selector.

The next step should use them to decide:
  "do not trust this attractive local evidence"
rather than:
  "replace the label with this other candidate".

So the better next design is a blocker / uncertainty gate:
  if raw candidate is high but cross-presence shadow is also high and
  relation/scale support is weak, mark it as shared-attraction risk.
```

## 2026-06-14 - External cap80 baseline sanity check

Purpose:

```text
Before changing the gate again, check whether the external cap80 set itself is
hard/noisy and what plain full-image baselines achieve.
```

Commands:

```powershell
.\.venv\Scripts\python.exe -m tools.train_eval_full_texture_head eval `
  --model_ckpt results\baseline_full_resnet18_awa_cls10\full_texture_head.pt `
  --cache results\dual_line_cache_awa_cls10_external_cap80\dual_line_cache.npz `
  --texture_cache results\texture_cache_awa_cls10_external_cap80_resnet18_default_full\texture_cache.npz `
  --out_dir results\eval_baseline_full_resnet18_awa_cls10_external_cap80 `
  --device auto

.\.venv\Scripts\python.exe -m tools.build_dual_line_texture_cache `
  --cache_dir results\dual_line_cache_awa_cls10_external_cap80 `
  --dataset_root dataset\external_test_sources\awa_cls10_external_test_mixed_cap80 `
  --split test `
  --out_dir results\texture_cache_awa_cls10_external_cap80_resnet50_default_full `
  --backbone resnet50 `
  --weights default `
  --batch_size 32 `
  --device auto `
  --full_image

.\.venv\Scripts\python.exe -m tools.train_eval_full_texture_head eval `
  --model_ckpt results\baseline_full_resnet50_awa_cls10\full_texture_head.pt `
  --cache results\dual_line_cache_awa_cls10_external_cap80\dual_line_cache.npz `
  --texture_cache results\texture_cache_awa_cls10_external_cap80_resnet50_default_full\texture_cache.npz `
  --out_dir results\eval_baseline_full_resnet50_awa_cls10_external_cap80 `
  --device auto
```

Results:

```text
Internal AWA cls10:
ResNet18 full-image head  94.10%
ResNet50 full-image head  97.40%

External cap80:
ResNet18 full-image head  74.59%
ResNet50 full-image head  81.68%
v153 fine                 85.97%
v153 parent               91.73%
```

External cap80 per-class accuracy:

```text
class             ResNet18  ResNet50  v153 fine  v153 parent
chihuahua          97.50%   100.00%    100.00%     100.00%
deer               50.00%    80.00%     87.50%      87.50%
fox                50.00%    50.00%     72.22%      94.44%
german+shepherd   100.00%   100.00%    100.00%     100.00%
horse              52.50%    62.50%     62.50%      62.50%
lion               46.25%    56.25%     56.25%      87.50%
persian+cat        96.25%   100.00%    100.00%     100.00%
siamese+cat        98.75%    97.50%    100.00%     100.00%
tiger              71.25%    73.75%     92.50%      93.75%
wolf               31.58%    42.11%     52.63%     100.00%
```

Image sanity check:

```text
Contact sheet:
results\external_cap80_sample_contact_sheet.jpg

The external cap80 set is not a clean classification benchmark.  Cat/dog
subclasses are mostly clean, but fox/wolf/lion/tiger/horse/deer include
drawings, exhibits, monitor/book/card images, small objects, and unusual
contexts.
```

Interpretation:

```text
The external cap80 drop is not only a model failure.  Plain full-image
ResNet18/50 baselines also fall sharply, especially on the single-child or
wild-animal classes.

v153 remains stronger than ResNet50 full-image head on external cap80 fine
accuracy, and much stronger on parent accuracy.  The next gate work should
therefore focus on the weak single-child classes and noisy-image robustness,
not assume that the previous v153 drop came from a simple implementation bug.
```

## 2026-06-17 - AWA cls10 clean-replaced cap80 sanity check

OpenImages 기반 external cap80에서 `wolf/fox/lion/tiger/horse/deer` 쪽에 그림,
전시물, 책/카드/모니터 이미지 등 라벨 오염이 많았기 때문에, 먼저 깨끗한
sanity check를 위해 해당 6개 클래스를 AWA2 holdout 이미지로 교체했다.

주의:

```text
이 세트는 완전한 외부 검증이 아니다.
AWA2 train/test split manifest에 들어간 파일은 제외했지만, 같은 데이터
소스 계열의 holdout이므로 clean holdout sanity check로 해석해야 한다.
```

Data:

```text
dataset\external_test_sources\awa_cls10_external_test_mixed_cap80\test
10 classes x 80 images = 800 images
clean replacement source:
dataset\external_test_sources\awa_cls10_clean_replacements_from_awa_holdout
```

Main artifacts:

```text
results\eval_stage0_v02_awa_cls10_external_cap80_clean_replaced
results\v113_bbox_cache_awa_cls10_external_cap80_clean_replaced
results\v131_texture_object_agreement_awa_cls10_external_cap80_clean_replaced_union
results\v153_structured_transition_gate_awa_cls10_external_cap80_clean_replaced
results\v153_structured_transition_gate_awa_cls10_external_cap80_clean_replaced\class_accuracy_compare.csv
```

Summary:

```text
model / stage                accuracy
-------------------------------------
ResNet18 full-image head      97.75%
ResNet50 full-image head      99.25%
MVP v0.2 selected parent      98.38%
v150 candidate parent         99.75%
v153 gated parent            100.00%
v153 gated fine               99.75%

v153 fixed vs selected parent: 13
v153 broken vs selected parent: 0
parent changed count: 15
```

Per-class:

```text
class             n   R18 full  R50 full  v0.2 parent  v150 parent  v153 parent  v153 fine
chihuahua        80    97.50%   100.00%      98.75%      100.00%      100.00%    100.00%
deer             80    96.25%    98.75%      98.75%       98.75%      100.00%    100.00%
fox              80    95.00%    98.75%      95.00%      100.00%      100.00%     98.75%
german+shepherd  80   100.00%   100.00%     100.00%      100.00%      100.00%    100.00%
horse            80    98.75%   100.00%      97.50%       98.75%      100.00%    100.00%
lion             80   100.00%   100.00%     100.00%      100.00%      100.00%    100.00%
persian+cat      80    96.25%   100.00%      98.75%      100.00%      100.00%    100.00%
siamese+cat      80    98.75%    97.50%      97.50%      100.00%      100.00%    100.00%
tiger            80    98.75%   100.00%      98.75%      100.00%      100.00%    100.00%
wolf             80    96.25%    97.50%      98.75%      100.00%      100.00%     98.75%
TOTAL           800    97.75%    99.25%      98.38%       99.75%      100.00%     99.75%
```

Interpretation:

```text
1. 오염이 줄어든 clean holdout에서는 baseline도 매우 강해졌다.
2. v153은 ResNet50 full baseline보다 parent 기준으로 더 높은 결과를 냈고,
   fine 기준으로도 거의 같은 수준까지 올라왔다.
3. 중요한 점은 v150 후보가 만들던 broken 2개를 v153 gate가 모두 막아
   fixed 13 / broken 0 구조가 됐다는 것이다.
4. 다만 이 결과는 같은-source holdout sanity check이므로, 독립 외부 검증은
   별도로 필요하다.
```

## 2026-06-18 - v163c/v165c full-eval candidate check

목적:

```text
v163b/v165 초기 결과는 eval 후보가 sample당 1개뿐이라,
후보 품질/위험 비교가 실제로 작동하지 못했다.
따라서 HF90 cls10 external noleak eval을 v131 full candidate table
(sample당 10개 후보)로 다시 통과시켜 보았다.
```

Artifacts:

```text
results\v163c_graph_and_candidate_selector_full_eval_hf90_cls10_noleak
results\v165c_recovery_risk_full_eval_hf90_cls10_noleak
```

Result:

```text
model/stage                  accuracy   fixed   broken   switches
------------------------------------------------------------------
selected/MVP                 95.33%      -       -        -
v163c full eval candidates   94.83%      12      15       37
v165c risk calibration       94.83%      12      15       37
```

Class-level weak point:

```text
horse:
  selected/MVP 88.33%
  v163c/v165c  78.33%
  fixed 0 / broken 6
```

Interpretation:

```text
1. Full candidate를 열면 후보 공간은 넓어지지만,
   보호 없는 selector는 위험 후보까지 같이 선택한다.
2. v165c는 risk/quality 보정이 개선을 만들지 못했다.
   train-only sweep이 risk_weight=0, quality_weight=0을 선택했기 때문에
   사실상 v163c와 동일하게 작동했다.
3. 원인은 risk feature 부재보다 training objective의 편향에 가깝다.
   현재 synthetic train은 selected가 틀리고 candidate가 맞는 상황을
   강하게 주입하지만, selected가 맞고 candidate가 틀린 방어 사례가 부족하다.
4. 다음 단계는 후보를 더 여는 것이 아니라,
   selected-correct / wrong-candidate 방어 사례를 train-only로 만들어
   recovery selector와 overtrust blocker를 함께 학습시키는 것이다.
```

## 2026-06-23 - v153auto AND concept gate

목적:

```text
수동 v153의 parent/fine transition gate를 자동 concept graph 기반으로 대체할 수 있는지 확인한다.
이번 버전은 자동 graph feature와 v170 judgment feature에 명시적 AND interaction feature를 추가했다.
```

핵심 artifact:

```text
tools/run_auto_structured_transition_gate_v153auto.py
docs/v153auto_and_concept_gate_ko.md
results\v153auto_and_logreg_cls10_test
results\v153auto_and_logreg_external_cap80_clean_replaced
results\v153auto_and_logreg_external_cap80
results\v153auto_and_logreg_hf90_cls10_noleak_and
```

결과:

```text
dataset                   selected   candidate   v153auto AND   manual v153   auto fixed/broken
-----------------------------------------------------------------------------------------------
cls10 internal test         94.70%      96.50%        96.60%        96.80%        33 / 14
external clean replaced     98.00%      99.50%        99.62%        99.75%        15 / 2
external noisy cap80        73.86%      84.64%        84.19%        85.97%        82 / 12
HF90 mixed noleak           95.33%      99.33%        99.33%        99.50%        25 / 1
```

해석:

```text
1. v153auto AND는 수동 parent/fine label을 gate feature로 쓰지 않고도 manual v153에 근접했다.
2. 내부/clean/HF90에서는 수동 v153과 0.1~0.2%p 차이까지 접근했다.
3. noisy external에서는 복구력은 강하지만 broken 억제가 manual v153보다 약하다.
4. 따라서 남은 병목은 후보 생성이 아니라 자동 concept graph 기반 protection/risk 판단이다.
```

결론:

```text
자동 concept graph + AND interaction은 v153 수동 구조를 부분 대체할 수 있다.
복구력은 충분히 강하고, 남은 차이는 protection이다.
다음 단계는 singleton/cross-concept protection과 warning edge 기반 risk 판단을 자동 graph 안에서 강화하는 것이다.
```
# v172 relative concept node gate (2026-06-24)

상위/하위 개념을 별도 타입이나 수동 의미 라벨로 고정하지 않고, 기본 concept node와
그 사이의 AND relation node를 `depth + lineage + relation polarity`로 표현했다.

```text
10 classes
base concept nodes: 7
relative AND nodes: 11
lineage edges: 22
AND polarity: risk 8 / context 3 / support 0
```

평가:

| dataset | selected | v172 | manual v153 | fixed / broken |
|---|---:|---:|---:|---:|
| cls10 internal | 94.70% | 96.60% | 96.80% | 33 / 14 |
| external clean | 98.00% | 99.625% | 99.75% | 15 / 2 |
| external noisy | 73.855% | 84.195% | 85.968% | 82 / 12 |
| HF90 mixed | 95.333% | 99.333% | 99.50% | 25 / 1 |

정확도는 기존 v153auto AND와 동일했다. 노드 계보 및 sample별 selected/candidate/shared
node 추적은 정상 생성됐으나, 현재 train transition 표본이 recovery에 치우쳐 있어
새 관계 정보를 protection 판단으로 바꾸지 못했다.

다음 병목은 graph 생성이 아니라 train-only hard protection node gate 학습이다.
# v173 support node promotion audit (2026-06-24)

v172의 risk/context relation 11개를 대상으로, 한 child에서 학습한 공통 증거가
다른 child로 양방향 전이되는지 측정했다.

```text
train: AWA cls10 train 2,000
internal: AWA cls10 test 1,000
external: HF90 mixed noleak 600
```

결과:

```text
promote_support       0
retain_relation       2
block_false_relation  9
```

가장 강한 부분 개념 후보:

```text
fox/wolf <-> deer
internal cross-child AUC 0.938
external cross-child AUC 0.997
purity 0.988
child coverage 0.383
independent evidence families 2
```

공통 증거는 강하지만 child 전체를 덮지 못하므로 parent support가 아니라 부분
support concept 후보로 유지했다.

두 번째 후보:

```text
lion/tiger <-> persian/siamese
internal cross-child AUC 0.684
external cross-child AUC 0.751
purity 0.946
independent evidence families 1
```

현재 자동 의미 형성은 relation 발견, lineage 추적, 거짓 승격 차단까지 작동한다.
다음은 retain relation의 활성 샘플과 타일/뷰 출처를 추적해 부분 support node를
분리하는 단계다.
# v174 partial support node trace (2026-06-24)

v173에서 유지된 relation 2개를 샘플/tile/bbox 수준으로 추적했다. threshold는
train에서만 선택했고 internal/HF90은 검증에만 사용했다.

```text
stable partial support candidate 1
insufficient partial support     1
```

`partial_support_001`:

```text
fox/wolf <-> deer
internal precision 98.67%
external precision 100%
internal coverage fox/wolf 97.5%, deer 28.0%
external coverage fox/wolf 50.83%, deer 23.33%
outside activation internal 0.43%, external 0%
tile child overlap 1.00
bbox child overlap 0.263
```

높은 purity와 외부 재현은 긍정 증거다. 반면 낮고 비대칭적인 coverage와 중앙 타일
집중은 전체 parent가 아니라 부분 시각 패턴일 가능성을 보여준다. 따라서 semantic
node로 확정하지 않고 `stable_partial_support_candidate`로 기록했다.

`lion/tiger <-> persian/siamese` 관계는 external에서 한쪽 child activation이 0이고
공간 반복성도 없어 `insufficient_partial_support`로 판정했다.

다음은 `partial_support_001`에 대해 배경/위치/자세 의존성을 제거하는 반증 실험이다.
# v175 partial_support_001 counterfactual (2026-06-24)

v174의 `partial_support_001`이 배경/촬영 구도인지 객체 내부 시각 패턴인지
counterfactual intervention으로 검증했다.

Frozen ResNet18 probe의 원본 재현:

```text
internal AUC 0.951
external AUC 0.894
```

활성 보존율:

```text
                         internal  external
object_inside              97.3%     93.3%
bbox_outside               11.1%      5.3%
center_removed             63.3%     68.0%
background_shuffle         94.7%     94.7%
pose_position              95.1%     92.0%
```

object-only/background-shuffle/pose 세 개입을 모두 통과:

```text
internal fox 95.9%, wolf 93.8%, deer 78.6%
external fox 97.1%, wolf 88.9%, deer 78.6%
false activation german shepherd 0%, horse 0%
```

배경과 위치를 바꿔도 유지되고 객체 영역 제거 시 붕괴하므로, 단순 촬영 문맥보다
객체 내부의 반복 시각 패턴이라는 증거가 강해졌다.

단, 의미 이름은 아직 부여하지 않는다. 판정은
`counterfactually_supported_partial_visual_node`로 제한한다.
# v176 partial_support_001 location trace (2026-06-25)

v174 active internal/external 298개를 대상으로 image 4x4와 object-normalized 4x4에서
타일 occlusion necessity와 tile-only sufficiency를 측정했다.

핵심 위치:

```text
 1  2  3  4
 5 [6][7] 8
 9[10][11]12
13 14 15 16
```

모든 그룹의 object necessity top-4 공통 타일은 `10`이었다. fox/wolf는
`6,7,10,11`, deer는 `7/10` 또는 `6/10/11`에 집중됐다.

중앙 2x2 necessity 비중:

```text
external deer 72.4%, fox 67.8%, wolf 53.7%
internal deer 51.4%, fox 68.9%, wolf 60.8%
```

object-normalized map 평균 상관은 `0.689`였다. internal fox-wolf 상관은 `0.909`,
external fox와 internal fox는 `0.925`였다.

top-1 위치 일치율은 21~50%이고 entropy도 높아, 단일 타일 노드가 아니라 중앙의
여러 인접 영역으로 구성되는 분산형 부분 시각 노드로 해석한다.

다음은 `6,7,10,11`의 tile-pair/AND 조합을 제거하거나 단독 보존해 어떤 관계가
activation을 만드는지 확인하는 단계다.
# v177 partial_support_001 objectness audit (2026-06-25)

v176 necessity 상위 70% 영역을 기존 4x4 observer obj_mask와 비교했다.

```text
importance inside mask 42.0%
mask coverage          29.1%
```

수치만 보면 context-dominant가 166/298이었지만, obj_mask는 독립 정답 mask가 아니라
기존 observer 가설이다. 대표 18개 샘플 overlay를 확인한 결과 obj_mask가 실제
얼굴/목/몸통을 놓치고, necessity 영역은 실제 객체 부위에 놓이는 사례가 많았다.

반복 후보:

```text
얼굴/주둥이
목-몸통 연결
몸통 중심
다리-몸통 경계
다중 객체의 상체/실루엣
```

v175에서 object-inside 유지 93~97%, object 제거 시 5~11%, background shuffle 유지
94.7%였던 결과와 합치면 partial node는 객체 내부 시각 관계일 가능성이 높다.

현재 `context_dominant` 분류는 폐기하지 않지만 `observer-mask-relative`로 제한한다.
오히려 이번 감사는 observer obj_mask가 중요한 객체 부위를 놓치는 문제를 드러냈다.
다음은 독립 class-agnostic object mask를 사용한 객체성 재검증이다.

# v178-v179 observer-token global relation (2026-06-26)

v178 local observer token shapes:

```text
cls10 train: [2500, 16, 180]
cls10 test : [1000, 16, 180]
HF90       : [ 600, 16, 180]
```

v179 trains a small Transformer head over the 16 observer tokens:

```text
tile wave -> local observer token -> global 16-token relation -> attention pooling -> cls10 head
```

Result:

```text
train      54.96%
cls10 test 29.20%
HF90       25.50%
```

Comparison:

```text
v153auto cls10 test 96.60%
v153auto HF90       99.33%
manual v153 cls10   96.80%
manual v153 HF90    99.50%
```

Interpretation:

```text
observer token alone contains object/shape continuity hints,
but lacks identity evidence such as CNN texture/fine-class scores.

Next direction:
use global observer relation as a v153/v153auto gate feature,
or build texture-aware observer tokens.
```

# v180 observer tracking as gate evidence (2026-06-26)

v179 observer-only was low because CNN texture/fine identity evidence was absent.
So v180 uses observer global relation as a pre-gate tracking feature, not as a classifier.

Added:

```text
tools/attach_global_observer_tracking_v180.py
run_structured_transition_gate_v153.py + OBSERVER_TRACKING_FEATURES
```

Candidate-row separation AUC against bundle_correct:

```text
observer_attn_support_sum       train 0.974 / test 0.973 / HF90 0.981
observer_texture_tracking_score train 0.973 / test 0.973 / HF90 0.996
observer_gate_pretrack_margin   train 0.973 / test 0.969 / HF90 0.985
```

v153 gate with observer tracking features:

```text
cls10 internal:
selected_parent 97.00%
v150 candidate  99.10%
v180/v153 gate  99.40%
fine            97.10%
fixed 25 / broken 1

HF90 external:
selected_parent 96.17%
v150 candidate  99.17%
v180/v153 gate  99.33%
fine            99.17%
fixed 19 / broken 0
```

Interpretation:

```text
v179 = observer-only ablation
v180 = observer tracking as gate evidence

CNN texture tells what the candidate is.
Observer relation tells whether that candidate's support tiles belong to a coherent object/shape flow.
```

# v180 auto-gate check (2026-06-27)

Question:
Can observer tracking features help the automatic v153auto concept gate, without manual parent labels?

Setup:
- Added v180 observer tracking features to v153auto BASE_FEATURES.
- Attached v180 tracking features to:
  - v163c train candidate rows
  - manual v153 cls10 eval rows
  - manual v153 HF90 eval rows
- Re-ran v153auto with v161 auto concept graph.

Results:

```text
cls10 internal:
selected 94.70%
candidate 96.50%
v180+v153auto 96.60%
manual v153 eval 96.80%
fixed 33 / broken 14

HF90 external:
selected 95.33%
candidate 99.33%
v180+v153auto 99.33%
manual v153 eval 99.50%
fixed 25 / broken 1
```

Interpretation:

```text
v180 tracking features do not break automatic concept-gate behavior.
However, they do not significantly close the gap to manual v153.
The missing piece is likely not candidate support tracking itself,
but how automatic concept nodes are formed/used before candidate selection.
```

Next:
Move v180-style tracking earlier into scan/candidate generation, not only final gate approval.

# v181 observer-tracked concept node audit (2026-06-27)

Question:
Can observer-tracked evidence form automatic concept nodes before manual v153 parent-map usage?

Implementation:
- Added/updated `tools/build_observer_tracked_concept_nodes_v181.py`.
- Input: `results\v180_observer_tracking_features_cls10_train\texture_object_agreement_observer_tracking_v180.csv`.
- First naive run promoted almost every pair to `sibling_support`, mostly because all classes shared central object tiles `6|7|10|11`.
- Added residual/specificity scoring:
  - raw vs centered tile/evidence cosine
  - `generic_context_score`
  - `central_context_score`
  - stricter sibling promotion

Strict residual result:

```text
out_dir:
results\v181_observer_tracked_concept_nodes_cls10_train_strict_residual

n_rows                20000
n_valid_tracking_rows 3725

edge_type_counts:
central_object_context 39
partial_support         4
sibling_support         2

node_type_counts:
central_object_context     14
sibling_support_candidate   4
sibling_support             2
```

Strong sibling edges:

```text
deer | horse
lion | tiger
```

Interpretation:

```text
v181 should not be treated as an automatic parent-node generator yet.
The robust signal is central/object support context, not broad semantic parent identity.

This is still useful:
central_object_context can become a support node,
then later a gate can learn when this support node approves or blocks a transition.
```

Next:

```text
v182 = support-node gate
Use central_object_context / partial_support / sibling_support as role-aware evidence.
Do not blindly promote central object support into parent/sibling concepts.
```

# v183 auto candidate meta-judge probe (2026-07-01)

Question:
Can we replace manual v153 parent-map approval with an automatic meta-judge that learns candidate reliability from automatic concept graph + object/wave/texture evidence?

Implementation:
- Added `tools/run_auto_candidate_meta_judge_v183.py`.
- It does not use manual `parent_map`.
- Input candidates can use `bundle_label`; the tool normalizes it to `fine_label`.
- Target is candidate correctness:

```text
candidate_is_true = fine_label == y_true_name
```

This differs from old `v153auto`, which learned:

```text
switch if candidate differs from selected and is true
```

Reason for change:

```text
current cls10 train selected/MVP accuracy = 100%
there are no selected-wrong recovery rows in train
so switch-target learning collapses to keep-only
```

Important bug/shortcut found:

```text
graph_same_class / graph_shared_node_count
```

made the judge learn "candidate == selected is safe" because train selected was perfect.
These selected-identity shortcut features are now removed by default in v183.

Current full probe:

```text
out_dir:
results\v183_auto_candidate_meta_judge_cls10_full_noshortcut

manual_parent_map_used: false
train rows: 20000
eval rows: 10000
feature_count: 122
train candidate AUC: 0.99979
elapsed: 52.9 sec
```

Evaluation on cls10 test:

```text
selected baseline:
accuracy 94.70%
wrong    53

score argmax:
accuracy 92.10%
fixed    25
broken   51

v183 meta argmax / gated:
accuracy 95.00%
fixed    19
broken   16
wrong    50
switch   40
```

Interpretation:

```text
v183 is not yet a v153 replacement.
But it is the first clean automatic meta-judge probe that:
1. avoids manual parent-map usage
2. avoids keep-only collapse
3. learns candidate reliability from automatic graph/evidence features
4. improves selected baseline slightly while keeping broken below raw score argmax
```

Main limitation:

```text
Train selected is too clean.
The judge still lacks enough selected-wrong recovery examples.
```

Next:

```text
1. Build a better train row contract:
   recovery / protection / ambiguous / safe-agreement rows.

2. Avoid selected identity shortcuts by default.

3. Keep v183 as the automation branch:
   automatic concept graph + candidate reliability judge.

4. Keep manual v153 as the upper reference, not as a dependency.
```

# v184 model-behavior meta gate probe (2026-07-01)

Question:
Can the Gate learn the reliability habits of independent Parent/Fine branches, instead of judging candidate rows directly?

Implementation:
- Added `tools/run_model_behavior_meta_gate_v184.py`.
- Uses automatic parent clusters from v160:

```text
results\v160_auto_cluster_parent_config_cls10_train_hf90_noleak\auto_parent_map_v160.json
results\v160_auto_cluster_parent_config_cls10_train_hf90_noleak\auto_parent_to_fine_v160.json
```

- Does not use manual semantic `parent_map`.
- Parent branch:

```text
stage0 parent/model prediction
auto cluster probability aggregation
parent confidence / margin / entropy
```

- Fine branch:

```text
best v131/v182 candidate bundle
object / wave / texture / object-flow evidence
```

- Gate target:

```text
trust_fine = fine correct and parent wrong
otherwise trust_parent
```

Run:

```text
out_dir:
results\v184_model_behavior_meta_gate_cls10_probe
```

Result:

```text
train:
parent accuracy      100.00%
fine accuracy         91.65%
parent_only_correct     167
fine_only_correct         0
both_correct          1833
both_wrong               0

eval:
parent accuracy       94.70%
fine accuracy         92.10%
parent_only_correct      51
fine_only_correct        25
both_correct            896
both_wrong               28
```

Interpretation:

```text
v184 implements the intended model-behavior Gate shape.
However, current train data cannot teach the Gate when Fine should override Parent.

Reason:
train Parent branch is perfect on cls10 train.
Therefore fine_only_correct = 0.
The Gate has no positive examples for "trust Fine".
```

This answers an important question:

```text
The Gate is not failing because the idea is wrong.
It is failing because the train row contract lacks Parent-failure / Fine-recovery cases.
```

Next:

```text
v185 should create behavior-gate training rows with actual parent failures:

1. use out-of-fold parent predictions, or
2. use a heldout/train split that parent did not fit, or
3. include external training-like data only for gate calibration, or
4. build controlled hard parent-failure rows without class shortcut leakage.

Do not tune this on final test folders.
```
## 2026-07-01 - v185 Success-Pattern Meta Gate

목표:

```text
오답에서 직접 배우는 gate가 아니라,
정답 판단이 나올 때 Parent branch와 Fine branch가 각각 어떤 증거 모양을 갖는지 학습한다.
그 후 현재 샘플이 어느 성공 패턴에 더 가까운지 비교해 Parent/Fine 신뢰를 결정한다.
```

핵심 변화:

```text
v184:
  Parent wrong + Fine correct 사례를 직접 학습하려 했으나,
  train에서 Parent가 100%라 trust_fine positive가 없었다.

v185:
  Parent 성공 profile = parent_node_correct 샘플의 Parent feature 분포
  Fine 성공 profile   = fine_correct 샘플의 Fine/object/wave/texture relation feature 분포
  Error label은 학습에 쓰지 않고, 평가/audit에만 사용한다.
```

Run:

```text
base:
results\v185_success_pattern_meta_gate_cls10_sigma005

margin probes:
results\v185_success_pattern_meta_gate_cls10_sigma005_m015
results\v185_success_pattern_meta_gate_cls10_sigma005_m020

sweep:
results\v185_success_pattern_meta_gate_cls10_sigma005\margin_sweep_v185.csv
```

기준 성능:

```text
Parent baseline: 94.70%
Fine baseline:   92.10%
```

v185 결과:

| margin | final acc | switch | fixed | broken | net | wrong |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 94.90% | 280 | 23 | 21 | +2 | 51 |
| 0.05 | 95.40% | 201 | 21 | 14 | +7 | 46 |
| 0.10 | 95.40% | 154 | 20 | 13 | +7 | 46 |
| 0.12 | 95.40% | 146 | 19 | 12 | +7 | 46 |
| 0.15 | 95.30% | 134 | 17 | 11 | +6 | 47 |
| 0.20 | 95.10% | 107 | 14 | 10 | +4 | 49 |
| 0.25 | 95.40% | 89 | 14 | 7 | +7 | 46 |
| 0.30 | 95.10% | 73 | 10 | 6 | +4 | 49 |
| 0.50 | 95.10% | 28 | 4 | 0 | +4 | 49 |

Audit 핵심:

```text
fine_only_correct:
  n=25
  success_score_delta mean = +0.252

parent_only_correct:
  n=51
  success_score_delta mean = -0.116

both_wrong:
  n=28
  success_score_delta mean = +0.116
```

해석:

```text
v185는 성공 패턴 신호를 실제로 잡았다.
Fine만 맞는 샘플은 Fine 성공 profile 쪽으로 기울고,
Parent만 맞는 샘플은 Parent 성공 profile 쪽으로 기운다.

다만 both_wrong도 일부 Fine 성공 profile처럼 보인다.
따라서 v185는 "성공 패턴 비교"로는 작동하지만,
다음 단계에서는 both_wrong/risk/context를 별도 차단하는 판단축이 필요하다.
```

현재 추천 운용점:

```text
margin=0.25:
  accuracy 95.40%
  fixed 14
  broken 7
  switch 89

이 값은 margin=0.05/0.10/0.12와 accuracy는 같지만,
전환 수와 broken이 더 낮아 자동 gate의 안정 운용점으로 보기 좋다.
```

다음 후보:

```text
v186:
  success-pattern gate
  + both_wrong/risk/context blocker
  + 자동 concept/node graph에서 나온 warning edge를 risk feature로 사용

목표:
  정확도만 올리는 것이 아니라,
  성공 패턴과 위험 패턴을 분리해 자동 상하위/AND 개념 gate로 확장한다.
```

## 2026-07-01 - v186 Success Pattern + Auto Graph Risk Gate

목표:

```text
v185의 성공 패턴 비교는 유지한다.
다만 Parent node와 Fine node가 자동 graph에서 risk/context/low-trust 관계일 때
Fine 전환 승인 margin을 더 높인다.
```

중요한 점:

```text
클래스명을 직접 하드코딩하지 않는다.

v160 auto cluster:
  cluster_001, cluster_002 ...

v161 concept graph:
  concept_001, concept_002 ...

두 노드는 이름이 달랐지만 멤버 집합이 같으므로 자동 매핑했다.
```

자동 매핑:

```text
cluster_001 -> concept_001  # fox, wolf
cluster_002 -> concept_002  # lion, tiger
cluster_003 -> concept_003  # persian+cat, siamese+cat
cluster_004 -> concept_004  # chihuahua
cluster_005 -> concept_005  # deer
cluster_006 -> concept_006  # german+shepherd
cluster_007 -> concept_007  # horse
```

Run:

```text
script:
tools\run_success_pattern_graph_risk_gate_v186.py

out_dir:
results\v186_success_pattern_graph_risk_gate_cls10_probe
```

Eval relation 역할 분포:

```text
same_node          928
unknown             28
risk_relation       24
context_relation    11
low_trust_relation   9
```

보수 운용점:

```text
base_margin  = 0.25
risk_penalty = 0.40

Parent baseline: 94.70%
Fine baseline:   92.10%
v186 final:      95.30%

fixed  = 10
broken = 4
net    = +6
switch = 14
```

Sweep 상단:

```text
best probe:
base_margin  = 0.12
risk_penalty = 0.20

v186 final: 95.50%
fixed: 18
broken: 10
net: +8
switch: 33
```

해석:

```text
v186은 v185보다 공격적인 개선 모델이라기보다,
자동 graph relation이 실제 전환 승인 조건으로 쓰일 수 있는지 확인한 probe다.

결과적으로 relation-risk margin은 작동했다.
특히 보수 운용점에서는 전환 수를 14개로 줄이면서 Parent baseline보다 +0.60%p를 얻었다.

이는 "자동 concept graph는 단순 분석물이 아니라 gate 판단축으로 재사용 가능하다"는 신호다.
```

클래스별 보수 운용점 요약:

| class | parent | fine | v186 |
|---|---:|---:|---:|
| chihuahua | 96% | 93% | 96% |
| deer | 97% | 84% | 97% |
| fox | 94% | 97% | 97% |
| german+shepherd | 93% | 87% | 93% |
| horse | 94% | 86% | 94% |
| lion | 100% | 100% | 100% |
| persian+cat | 88% | 89% | 88% |
| siamese+cat | 89% | 90% | 90% |
| tiger | 100% | 100% | 100% |
| wolf | 96% | 95% | 98% |

다음:

```text
v187 후보:
성공 profile + relation risk를 유지하되,
unknown relation 28개를 "무시"하지 않고
partial support / observer flow / same-object evidence로 재분류한다.

목표:
unknown을 줄이고,
같은 객체 흐름에서 나온 전환인지
단순 후보 과신인지 분리한다.
```

## 2026-07-01 - v187 Behavior Habit Probe

목표:

```text
Parent = 공통성/정체성 관점
Fine   = 차이/분해 관점
Gate   = 두 모델의 판단 습관을 평가하는 meta-judge
```

이 관점을 수치화하기 위해 기존 v186 결과 위에서 가벼운 probe를 수행했다.

새로 본 점수:

```text
parent_commonality_habit_v187:
  Parent success score
  Parent node confidence / margin / entropy
  Parent/Fine node 일치 여부

fine_difference_habit_v187:
  Fine success score
  Fine support tile diversity
  cross-region agreement
  object-flow coherence
  fine separation
  overfocus/redundancy risk 감점
```

Run:

```text
out_dir:
results\v187_behavior_habit_probe_cls10
```

결과:

```text
v187 handcrafted habit score는 v186 best를 넘지 못했다.

best probe:
accuracy 95.50%
fixed 18
broken 10
net +8

이는 v186 sweep 상단과 같은 수준이다.
```

Group mean:

| group | n | success delta | habit delta |
|---|---:|---:|---:|
| parent_only_correct | 51 | -0.116 | +0.057 |
| fine_only_correct | 25 | +0.252 | +0.207 |
| both_correct | 896 | -0.201 | -0.168 |
| both_wrong | 28 | +0.116 | +0.086 |

해석:

```text
긍정:
  Fine만 맞는 샘플은 habit_delta가 가장 높다.
  즉 "Fine 차이 관점이 성공하는 모양"은 일부 잡힌다.

부정:
  Parent만 맞는 샘플도 habit_delta가 양수다.
  따라서 현재 handcrafted Fine-difference score는
  Parent가 지켜야 할 샘플까지 Fine 쪽으로 끌어당긴다.
```

결론:

```text
v187의 개념은 유지한다.
하지만 현재 수작업 habit score를 gate에 바로 넣는 것은 보류한다.

이 방향을 계속하려면 단순 feature 조합이 아니라
same-object / observer-flow / partial-support 단위에서
"Fine가 본 차이가 같은 객체 흐름에서 나온 차이인가"를 먼저 확인해야 한다.
```

다음:

```text
v188 후보:
unknown/risk 전환 28개를 대상으로
Fine 후보가 같은 객체 흐름에서 나온 증거인지,
아니면 특정 부위/배경/반복 crop에 끌린 과신인지 분리한다.

즉 Gate를 더 세게 튜닝하지 말고,
Gate가 평가할 "판단 흔적"의 정의를 더 정확히 만든다.
```

## 2026-07-01 - v188 Behavior-Habit-Situation Gate 방향 고정

v187 이후 Gate 구조를 다음 5레벨로 정리했다.

```text
Level 0. Behavior
  무엇을 보는가

Level 1. Habit
  평소 어떻게 보는가

Level 2. Situation
  지금 어떤 판단 환경인가

Level 3. Judgment
  이 상황에서 이 습관은 정상인가

Level 4. Decision
  최종적으로 Parent/Fine/hold 중 무엇을 승인할 것인가
```

핵심 변화:

```text
v187:
  Parent commonality habit / Fine difference habit를 직접 만들었다.
  하지만 Habit만으로는 Parent-only-correct 보호가 부족했다.

v188:
  Habit를 더 튜닝하지 않고,
  그 Habit가 작동할 Situation인지 먼저 분리한다.
```

Situation은 두 층으로 나눈다.

```text
External Situation:
  노이즈, 배경, 해상도, 가림, 조명, crop 크기

Internal Situation:
  Parent/Fine confidence
  support 분포
  object-flow
  observer 일치도
  concept relation
  same-object evidence
  partial-support / overfocus risk
```

Gate가 실제로 학습해야 하는 것은 주로 Internal Situation이다.

설계 문서:

```text
docs\v188_behavior_habit_situation_gate_ko.md
```

## 2026-07-01 - v188 Situation Audit Probe

목표:

```text
Habit 자체를 더 튜닝하지 않고,
Fine habit가 지금 통할 Situation인지 분리할 수 있는지 본다.
```

Run:

```text
script:
tools\run_behavior_habit_situation_audit_v188.py

out_dir:
results\v188_behavior_habit_situation_audit_cls10
```

Situation feature:

```text
support_distribution_v188
object_flow_v188
texture_relation_v188
multi_region_object_support_v188
overfocus_risk_v188
parent_internal_situation_v188
fine_internal_situation_v188
situation_delta_v188
```

Group mean:

| group | n | situation_delta | success_delta |
|---|---:|---:|---:|
| parent_only_correct | 51 | -0.088 | -0.116 |
| fine_only_correct | 25 | +0.002 | +0.252 |
| both_correct | 896 | -0.354 | -0.201 |
| both_wrong | 28 | -0.088 | +0.116 |

해석:

```text
fine_only_correct는 situation_delta가 가장 높다.
parent_only_correct와 both_wrong은 거의 같은 수준으로 낮다.

즉 Situation은 Fine 전환을 더 많이 여는 공격축이라기보다,
both_wrong / 위험 Fine 전환을 막는 보호축으로 쓰는 것이 맞다.
```

비교:

```text
v186 conservative:
accuracy 95.30%
fixed 10
broken 4
both_wrong_switched 4

v188 situation filter example:
accuracy 95.30%
fixed 10
broken 4
both_wrong_switched 0
```

결론:

```text
v188은 정확도 상단을 올리지는 못했다.
하지만 Situation filter가 both_wrong 전환을 제거하는 신호는 있다.

따라서 다음 방향은 Gate threshold 튜닝이 아니라,
same-object / observer-flow / partial-support 정의를 더 정확하게 만들어
Situation을 보호층으로 안정화하는 것이다.
```

## 2026-07-01 - v189 Fine Evidence-Quality Gate

목표:

```text
Gate target:
  Fine이 맞는가? X
  Fine 증거가 신뢰 가능한가? O
```

v188까지는 Fine 전환 성공 또는 success_delta가 여전히 강하게 작동했다.
v189에서는 전환 성공을 직접 학습하지 않고,
Fine branch의 증거가 신뢰 가능했던 train 패턴을 학습했다.

학습 target:

```text
fine_evidence_reliable_proxy = fine_correct on train rows

중요:
  transition/switch success는 학습 target으로 쓰지 않는다.
```

입력 feature:

```text
support_distribution_v188
object_flow_v188
texture_relation_v188
multi_region_object_support_v188
overfocus_risk_v188
relation_risk_v186
wave/texture/object-flow/support 관련 Fine evidence feature 36개
```

Run:

```text
script:
tools\run_fine_evidence_quality_gate_v189.py

out_dir:
results\v189_fine_evidence_quality_gate_cls10_logreg

model:
logreg

operating point:
margin      = 0.25
quality_min = 0.60
```

결과:

```text
Parent baseline: 94.70%
Fine baseline:   92.10%
v189 final:      95.50%

fixed  = 8
broken = 0
net    = +8
switch = 8
```

Group quality mean:

| group | n | quality prob | success delta |
|---|---:|---:|---:|
| parent_only_correct | 51 | 0.131 | -0.116 |
| fine_only_correct | 25 | 0.358 | +0.252 |
| both_correct | 896 | 0.826 | -0.201 |
| both_wrong | 28 | 0.273 | +0.116 |

해석:

```text
v189는 v188보다 목적에 더 맞다.

v188:
  Situation을 직접 점수화했지만 정확도 상단은 올리지 못했고,
  보호 필터 역할만 보였다.

v189:
  Fine evidence quality를 train에서 학습했다.
  전환 성공을 직접 학습하지 않았는데도 broken=0으로 8개를 복구했다.
```

클래스별 변화:

| class | parent | fine | v189 |
|---|---:|---:|---:|
| chihuahua | 96% | 93% | 96% |
| deer | 97% | 84% | 97% |
| fox | 94% | 97% | 97% |
| german+shepherd | 93% | 87% | 93% |
| horse | 94% | 86% | 94% |
| lion | 100% | 100% | 100% |
| persian+cat | 88% | 89% | 90% |
| siamese+cat | 89% | 90% | 89% |
| tiger | 100% | 100% | 100% |
| wolf | 96% | 95% | 99% |

결론:

```text
v189는 현재 방향의 가장 좋은 신호다.

Gate가 "Fine이 맞을 것 같다"가 아니라
"Fine 증거가 신뢰 가능한가"를 보도록 바꾸면
전환 수는 적지만 broken 없이 복구가 가능하다.
```

다음:

```text
v190 후보:
v189 evidence-quality gate를 유지한다.
다만 fine_correct proxy label을 더 정교하게 바꾼다.

예:
  reliable evidence positive:
    fine correct
    + multi-view/object-flow support 충분

  unreliable evidence negative:
    fine wrong
    or high confidence wrong
    or parent-only-correct with Fine high score

목표:
  quality_prob가 both_wrong보다 fine_only_correct를 더 강하게 분리하도록 만든다.
```

---

## 2026-07-01 - v189 나머지 테스트셋 비교: v153 대비

목적:

```text
v189 fine-evidence-quality gate가 내부 cls10 외의 테스트셋에서도
v153 structured transition gate를 대체할 수 있는지 확인한다.
```

비교 파일:

```text
results\v189_vs_v153_remaining_tests_comparison.csv
```

실행 결과:

| dataset | n | parent base | fine base | v153 fine | v189 default | v189 best sweep | default fixed/broken | best fixed/broken |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cls10_test | 1000 | 94.70% | 92.10% | 96.80% | 95.50% | 95.50% | 8 / 0 | 8 / 0 |
| external_clean | 800 | 98.00% | 97.00% | 99.75% | 98.00% | 98.12% | 0 / 0 | 1 / 0 |
| external_hf90 | 600 | 95.33% | 95.33% | 99.50% | 95.33% | 95.67% | 0 / 0 | 5 / 3 |
| external_noisy | 677 | 73.86% | 77.70% | 85.97% | 73.86% | 76.07% | 0 / 0 | 26 / 11 |

해석:

```text
v189는 내부 cls10에서는 broken 없이 8개를 복구했지만,
외부셋에서는 v153을 대체하기에는 너무 보수적이다.

특히 default operating point는 external_clean / external_hf90 / external_noisy에서
거의 parent baseline을 유지한다.
```

핵심 판단:

```text
v189 = standalone replacement X
v189 = v153 위에 붙일 evidence-quality blocker 후보 O
```

이유:

```text
v153은 structured transition 자체를 잘 수행한다.
v189는 Fine evidence가 신뢰 가능한지 보수적으로 판단한다.

따라서 다음 방향은 v153의 후보 전환 능력은 유지하고,
v189의 quality/risk 판단을 전환 승인/차단 레이어로 붙이는 것이다.
```

다음 후보:

```text
v190:
  v153 transition proposal
  + v189 evidence-quality blocker
  + relation/object-flow/same-object risk

목표:
  v153의 fix를 최대한 유지하면서
  broken transition만 줄인다.
```

---

## 2026-07-01 - v190 Cross-Level Evidence Habit Gate 1차

목적:

```text
Gate가 오답 케이스만 보지 않고,
전체 train trace에서 Parent/Fine의 정상 습관과 위험 습관을 같이 학습하는지 확인한다.

핵심 질문:
  Parent/Fine 중 누가 맞는가? X
  Parent/Fine이 각각 어떤 쉬운길을 탔고,
  그 현장 증거가 정상 habit인지 위험 habit인지 구분 가능한가? O
```

추가 스크립트:

```text
tools\run_cross_level_evidence_habit_gate_v190.py
```

입력:

```text
train:
results\v189_compare_cls10_test\v189\train_predictions_v189.csv

eval:
results\v189_compare_cls10_test\v189\eval_predictions_v189.csv
```

학습 target:

```text
parent_reliable        = parent_fine_correct
fine_reliable          = fine_correct
stable_both_correct    = both_correct
fine_shortcut_protect  = parent_only_correct
both_wrong_risk        = both_wrong
```

중요한 train/eval case 분포:

| split | both correct | parent only | fine only | both wrong |
|---|---:|---:|---:|---:|
| train | 1833 | 167 | 0 | 0 |
| eval | 896 | 51 | 25 | 28 |

결과:

| model | accuracy | fixed | broken | switch | reobserve |
|---|---:|---:|---:|---:|---:|
| Parent baseline | 94.70% | - | - | - | - |
| v189 | 95.50% | 8 | 0 | 8 | 0 |
| v190 default | 94.70% | 0 | 0 | 624 | 0 |
| v190 best sweep | 94.70% | 0 | 0 | 135 | 0~117 |

핵심 진단:

```text
v190은 구조적으로는 "전체 case 학습"을 구현했지만,
현재 train split에는 Parent wrong / Fine correct 케이스가 없다.

따라서 eval의 fine_only_correct 25개를
Fine repair 케이스로 배우지 못하고,
Fine shortcut/protection 케이스와 비슷하게 오인했다.
```

그룹별 v190 habit 평균:

| group | n | fine reliable | fine shortcut | switch score |
|---|---:|---:|---:|---:|
| parent_only_correct | 51 | 0.00005 | 0.99995 | -0.373 |
| fine_only_correct | 25 | 0.00000 | 1.00000 | -0.125 |
| both_correct | 896 | 0.92373 | 0.07627 | 0.264 |
| both_wrong | 28 | 0.32227 | 0.67773 | -0.021 |

해석:

```text
fine_only_correct가 parent_only_correct와 같은 "Fine shortcut" 쪽으로 붙었다.
즉 Gate가 Fine 복구 상황을 배울 학습 근거가 없었다.

이 결과는 v190 아이디어가 틀렸다기보다,
현재 train trace가 메타 게이트 학습에 필요한 case coverage를 갖지 못했다는 증거다.
```

다음 조건:

```text
v190을 제대로 학습하려면 train 안에 다음 case가 필요하다.

1. Parent correct / Fine correct
2. Parent correct / Fine wrong
3. Parent wrong / Fine correct
4. Parent wrong / Fine wrong

현재는 1, 2만 있다.
```

다음 방향:

```text
v190b 후보:
  v153 train_transition_rows.csv를 학습 trace로 사용한다.

이유:
  v153 train_transition_rows에는 transition_fix / transition_break /
  candidate_parent_correct / candidate_fine_correct / approve_target 등이 있어
  실제 전환 현장 학습에 더 적합하다.

또는:
  train OOF / heldout parent predictions를 만들어
  Parent wrong / Fine correct 케이스를 train 안에 생성한다.
```

---

## 2026-07-01 - v190b Transition Trace Gate 1회 실행

목적:

```text
v190 1차에서 부족했던 train case coverage 문제를 피하기 위해,
v189 train/eval prediction trace가 아니라
v153 train_transition_rows.csv를 직접 학습 trace로 사용한다.
```

추가 스크립트:

```text
tools\run_transition_trace_gate_v190b.py
```

실행:

```text
python -m tools.run_transition_trace_gate_v190b
  --train_csv results\v153_structured_transition_gate_cls10\train_transition_rows.csv
  --eval_csv results\v153_structured_transition_gate_cls10\all_predictions.csv
  --parent_to_fine_json results\v153_structured_transition_gate_cls10\summary.json
  --out_dir results\v190b_transition_trace_gate_cls10_once
  --model logreg
```

결과:

| model | parent acc | fine acc | fixed | broken | approved/rejected |
|---|---:|---:|---:|---:|---:|
| selected parent baseline | 97.00% | - | - | - | - |
| v150 candidate | 99.20% | 96.50% | 26 | 4 | - |
| v153 | 99.60% | 96.80% | 26 | 0 | 994 / 6 |
| v190b | 99.20% | 96.50% | 26 | 4 | 1000 / 0 |

v190b train:

```text
train_auc = 1.0
train approve positive = 1982
train approve negative = 18
threshold = 0.02
```

핵심 진단:

```text
v190b는 train transition trace에서는 approve_target을 완벽히 분리했지만,
eval에서는 parent-changing 후보 33개를 모두 승인했다.

그 결과 v153처럼 4개의 broken transition을 차단하지 못했고,
v150 candidate 결과와 동일해졌다.
```

해석:

```text
v153 train_transition_rows를 쓰는 것만으로는 충분하지 않다.

문제는 trace 소스만이 아니라,
train에서 배운 negative 18개가 eval의 broken 4개와 같은 분포를 대표하지 못한다는 점이다.
```

현재 결론:

```text
v190b = v153 대체 실패
but:
  "transition approval trace를 써야 한다"는 방향은 맞고,
  단순 approve_target 학습만으로는 eval broken을 막지 못한다.

다음에 한다면:
  v153이 실제로 차단한 6개와 broken 후보의 특징을 기준으로
  hard negative / parent-changing danger profile을 따로 만들어야 한다.
```
---

## 2026-07-07 - v130 relation crop budget sweep

목적:

```text
v153 전체 파이프라인에서 가장 큰 병목인 v130 texture relation crop 수를 줄여도
성능이 유지되는지 확인한다.
```

기준:

```text
dataset = AWA cls10 test, class당 100장, 총 1000장
baseline full v130 = 16x16 전체 pair, 256 relation crops/image
policy = object_local
```

결과:

| setting | v130 time | post-v130 total | parent acc | fine acc | fixed | broken |
|---|---:|---:|---:|---:|---:|---:|
| full 256 pairs | 450.4s | - | 99.4% | 96.7% | 26 | 2 |
| budget 128 | 212.5s | 250.0s | 99.4% | 96.6% | 26 | 2 |
| budget 64 | 105.5s | 137.6s | 99.4% | 96.6% | 26 | 2 |

해석:

```text
이번 1000장 표본에서는 256 -> 64 pair까지 줄여도 v153 parent/fine 성능은 사실상 유지됐다.
v130 단독 시간은 약 4.27배 빨라졌다.

따라서 relation crop은 전수 256개가 항상 필요한 것은 아니며,
object-local pair selection을 기본 fast path 후보로 볼 수 있다.
다만 전체 train/test 대규모 적용 전에는 budget 64/96/128의 class별 안정성 검증이 필요하다.
```

---

## 2026-07-08 - observer scan fast NPZ path

목적:

```text
traj_debug_view.py 기반 observer scan에서 캐시 생성에 필요 없는
DataFrame/stigma/debug CSV 경로를 우회해 scan 병목을 줄인다.
```

변경:

```text
tools/traj_debug_view.py
- measure_tile_feature_mats 추가
- scan_npz_only + stigma reobserve disabled + refine 1회 조건에서 fast_npz_only 경로 사용
- fast path는 .tiles.npz 생성에 필요한 phi/rho/edge_ratio/int_std/core_rc/obj_mask만 계산
- 기존 debug/stigma/reobserve 분석 경로는 유지
```

검증:

```text
cls10 test 100장:
normal scan = 36.25s, csv 300개, npz 100개
fast scan   = 12.23s, csv 0개,   npz 100개
speedup     = 2.96x

공통 100개 중 앞 10개 npz 비교:
rho / edge_ratio / int_std / obj_mask / core_rc max_abs_diff = 0
```

1000장 기준:

| scan mode | time | outputs |
|---|---:|---|
| previous npz_only path | 324.4s | npz + internal DataFrame/stigma work |
| fast npz path | 98.3s | 1000 npz + 4 logs |

해석:

```text
scan 병목은 파일 출력만이 아니라 내부 DataFrame/stigma/debug 계산 경로가 큰 비중을 차지했다.
캐시 생성용 스캔은 direct tile-matrix path로 분리하는 것이 맞다.

v130 budget64와 결합하면 현재 1000장 기준 주요 병목은 대략:
v113 bbox candidates 202s
v130 relation budget64 105s
scan fast path 98s
순서가 된다.
```

---

## 2026-07-08 - v113 bbox candidate bottleneck profile

목적:

```text
v113 bbox 후보 수를 줄이는 것이 아니라,
후보를 많이 유지할 때 병목이 어디서 발생하는지 확인한다.
```

프로파일:

```text
dataset = AWA cls10 test 100장
grid=4, min_area=4
candidate_count = 44/image
candidate_rows = 4400
backbone = ResNet18 ImageNet
```

기존 PIL 전처리 경로:

| step | time | share |
|---|---:|---:|
| image open | 0.59s | 3.6% |
| crop/resize/preprocess | 12.62s | 76.3% |
| stack/to GPU | 1.05s | 6.4% |
| CNN forward | 2.18s | 13.2% |
| score extraction | 0.05s | 0.3% |
| total | 16.55s | 100% |

해석:

```text
v113 병목은 CNN forward보다 CPU crop/resize/preprocess가 더 컸다.
따라서 후보 수 축소만이 답이 아니라, 후보를 많이 유지하려면 전처리 backend를 바꾸는 것이 우선이다.
```

변경:

```text
src/dual_line/representation/bbox_candidate_scorer.py
- preprocess_roi_image_cv2 추가
- score_imagenet_bbox_candidates(..., preprocess_backend="cv2"|"pil") 추가

tools/build_bbox_candidate_score_cache_imagenet_multiclass_v113.py
- --preprocess_backend {cv2,pil} 추가
- default=cv2
```

결과:

| setting | candidates/image | time | candidate oracle |
|---|---:|---:|---:|
| PIL min_area=4 | 44 | ~202s | 99.3% |
| cv2 min_area=4 | 44 | 76.2s | 99.3% |
| roi_align min_area=4 | 44 | 50.6s | 99.4% |
| PIL min_area=5 | 27 | 141.8s | 98.9% |
| PIL min_area=8 | 15 | 79.7s | 98.7% |

해석:

```text
cv2 backend을 쓰면 후보 수를 44개 그대로 유지해도 min_area=8 수준의 시간까지 내려간다.
즉 "확신도가 낮을 때 후보 수로 밀어붙이는" 전략이 훨씬 현실적이 됐다.
후보 수를 줄이는 pruning은 다음 단계 최적화로 남기되, 현재 우선 기본 fast path는 cv2 preprocessing이다.
```

추가:

```text
사용자가 제안한 image tensor + boxes[N,4] -> crop tensor batch 구조를
torchvision.ops.roi_align backend로 구현했다.

roi_align backend는 후보 수 44개를 유지하면서도 1000장 기준 50.6s까지 내려갔다.
따라서 v113의 추천 fast path는 cv2보다 roi_align이다.
```

---

## 2026-07-08 - resource scaling check for scan/v130

목적:

```text
남은 병목인 scan fast path와 v130 relation이
worker 수 증가만으로 더 빨라지는 구조인지 확인한다.
```

환경:

```text
CPU = Intel i5-14400F, 10 cores / 16 logical processors
GPU = RTX 3070 8GB
dataset = AWA cls10 test 1000장
```

scan fast path:

| workers | time |
|---:|---:|
| 4 | 98.3s |
| 8 | 78.5s |

해석:

```text
scan은 worker 증가로 빨라지지만 선형은 아니다.
4 -> 8 workers에서 1.25x 개선만 발생했다.
남은 병목은 CPU feature 계산 + warp/Sobel 반복 + 프로세스/I/O 오버헤드로 보인다.
```

v130 budget64 object_local:

| workers | v130 relation time | post-v130 total |
|---:|---:|---:|
| 4 | 105.5s | 137.6s |
| 8 | 117.1s | 148.0s |

해석:

```text
v130은 worker를 8로 늘리면 오히려 느려졌다.
GPU/CPU 전처리/프로세스 경합이 생기는 구조로 보이며,
자원 투입만으로 해결되는 병목이 아니다.

v130은 v113과 마찬가지로 pair별 PIL crop/resize/preprocess가 큰 병목이므로
roi_align/tensor crop backend 적용이 다음 안전한 최적화 후보이다.
```

## 2026-07-10 - Judgment Space Gate Module Update

변경 목적:

```text
prototype_parent_match 같은 label-derived parent anchor를 런타임에 쓰지 않고,
Parent/Fine 판단 습관과 상황을 보는 Judgment Space 기반 게이트 학습으로 이동.
```

수정:

```text
src/dual_line/decision/v153_transition_gate.py
- make_transition_rows()가 judgment_* feature를 붙이도록 변경
- threshold sweep에 0.97/0.98/0.99/0.995/0.999 추가
- fp_penalty 인자 추가

tools/train_apply_v153_gate_from_hub_evidence.py
- --fp_penalty 추가

tools/run_v153_transition_gate_modular.py
- --fp_penalty 추가
```

검증:

```text
py_compile 통과
make_transition_rows smoke check에서 judgment_* 10개 생성 확인
```

주요 결과:

```text
results/modular_benchmark/v153_pseudo_parent_interaction_gate_judgment_cls10_test1000_fp002
```

| metric | value |
|---|---:|
| threshold | 0.99 |
| fp_penalty | 0.02 |
| selected parent acc | 97.00% |
| runtime parent acc | 97.40% |
| runtime fine acc | 95.30% |
| fixed / broken | 6 / 2 |

공통 hub gate 재학습 확인:

```text
results/modular_benchmark/v153_gate_from_hub_judgment_common_cls10_test1000_fp002
```

| metric | value |
|---|---:|
| train approve positives | 0 |
| gate mode | fallback_reject_parent_changes |
| eval parent acc | 97.00% |
| fixed / broken | 0 / 0 |

해석:

```text
공통 v153 gate module은 이제 judgment-space aware가 됐다.
다만 train_gate_baseline=mvp에서는 train selected parent가 이미 100%라
전환 positive가 없어서 실제 전환 학습은 불가능하다.

현재 실제로 유효한 학습 경로는 prototype-free pseudo parent interaction gate이며,
fp_penalty를 키우면 train-only 기준으로 보수 threshold를 선택해 broken을 줄일 수 있다.
다음 과제는 runtime parent-changing case와 더 닮은 hard replay generator를 만드는 것.
```

## 2026-07-10 - v190 Model Habit Space

목표:

```text
전환 승인/거절을 바로 학습하지 않고,
Parent/Fine 모델이 train에서 반복적으로 만드는 증거 습관을 먼저 학습한다.
```

추가 파일:

```text
src/dual_line/decision/habit_space.py
tools/build_model_habit_space_v190.py
```

핵심 구조:

```text
train evidence rows
  -> Parent habit profile by selected_parent
  -> Fine habit profile by fine_label
  -> runtime gate input에 habit_* 상태값 부착
```

생성 feature:

```text
habit_parent_typicality
habit_fine_typicality
habit_parent_fine_typicality_gap
habit_both_unusual
habit_fine_shortcut_like
habit_reobserve_pressure
```

중요한 점:

```text
approve_target / transition_fix / transition_break를 학습 목표로 쓰지 않음.
true label은 optional audit에만 사용.
프로파일은 selected_parent, fine_label, evidence distribution만으로 생성.
```

실행:

```powershell
.\.venv\Scripts\python.exe -m tools.build_model_habit_space_v190 `
  --out_dir results\modular_benchmark\v190_model_habit_space_cls10_test1000 `
  --train_candidate results\v113_bbox_cache_awa_cls10_train\bbox_candidate_scores.csv `
  --train_wave_relation_npz results\tile_view_relation_v05_awa_cls10_train\tile_view_relation_v05.npz `
  --train_texture_relation_npz results\v130_texture_relation_cls10_train_union\texture_relation_v130.npz `
  --train_parent_csv results\stage0_v02_awa_cls10_resnet18\train_predictions_keyed.csv `
  --eval_gate_input_npz results\modular_benchmark\runtime_hub_v153_full_with_gate_cls10_test1000\hub\v153_gate_input.npz `
  --eval_truth_parent_csv results\eval_stage0_v02_awa_cls10_test\predictions_keyed.csv `
  --parent_map_json configs\parent_map_awa_cls10_v153.json `
  --parent_labels cat,dog,horse `
  --class_names persian+cat,siamese+cat,chihuahua,german+shepherd,wolf,fox,lion,tiger,horse,deer `
  --max_features 96
```

결과:

```text
results/modular_benchmark/v190_model_habit_space_cls10_test1000
```

| item | value |
|---|---:|
| train habit rows | 4,802 |
| habit features | 96 |
| parent profiles | 6 |
| fine profiles | 10 |
| eval review rate @0.65 | 2.8% |
| selected-wrong recall in reviewed set | 33.3% |
| candidate-wrong recall in reviewed set | 57.6% |

해석:

```text
Habit Space는 전체 샘플의 2.8%만 review로 잡았지만,
candidate parent wrong의 57.6%를 잡았다.

이는 오답을 직접 외우는 approve/reject 학습보다,
모델 습관상 이상하거나 위험한 evidence state를 좁게 잡는 방향의 신호로 볼 수 있다.
```

대표 review 패턴:

```text
deer selected_parent=deer, fine_parent=dog_like/big_cat
horse selected_parent=horse, fine_parent=deer/dog_like
cat/dog selected_parent correct, fine_parent cross-parent
```

다음 방향:

```text
Gate는 approve_target부터 배우기보다,
1. habit state
2. situation state
3. judgment state
을 먼저 만들고, 승인/거절/reobserve는 이 상태들의 후단으로 둔다.
```

## 2026-07-10 - v190 Habit State Integrated Update

사용자 피드백:

```text
새 체인을 계속 쌓지 말고, Habit Space 모듈 자체가 판단 상태까지 내도록 바꾼다.
```

수정:

```text
src/dual_line/decision/habit_space.py
- attach_habit_space_features()가 raw habit score와 habit_judgment_* state를 같이 생성
- 별도 v191 체인을 만들지 않고 v190 모듈 내부에서 상태 번역 수행
```

추가된 상태 feature:

```text
habit_judgment_parent_trust
habit_judgment_fine_trust
habit_judgment_switch_safe
habit_judgment_reobserve_required
habit_judgment_both_unreliable
habit_judgment_keep_bias
habit_judgment_state
```

첫 상태식 문제:

```text
both_unreliable이 typicality distance만으로 너무 넓게 켜져 769/1000개를 잡음.
```

수정:

```text
both_unreliable은 단순 거리 대신 conflict, reobserve pressure, evidence completeness를 함께 보도록 재보정.
```

재보정 출력:

```text
results/modular_benchmark/v190_model_habit_space_cls10_test1000_state_recalibrated
```

결과:

| state | n |
|---|---:|
| undecided | 976 |
| reobserve_required | 24 |

Audit:

| metric | value |
|---|---:|
| review rate @0.65 | 2.8% |
| candidate wrong recall in review | 57.6% |
| reobserve state rate @0.70 | 2.4% |
| candidate wrong recall in reobserve state | 51.5% |

해석:

```text
v190은 아직 switch/keep 최종 게이트가 아니다.
현재는 Parent/Fine의 습관과 현재 증거 상황을 보고,
좁은 reobserve_required 상태를 생성하는 위험 관측 모듈로 보는 것이 맞다.

이 방향은 approve_target 기반 전환 학습보다 사용자가 의도한
'모델의 습관을 학습하고 판단 공간으로 넘긴다'는 구조에 더 가깝다.
```

## 2026-07-10 - v190 Normality Inference Update

방향 전환:

```text
train에 강한 확신 오답이 거의 없으므로,
오답을 직접 학습하는 것이 아니라 정상 판단 evidence manifold를 학습한다.
테스트에서는 confidence가 강해도 정상 증거 구조와 어긋나는지를 추론한다.
```

수정:

```text
src/dual_line/decision/habit_space.py
```

추가된 normality/inference feature:

```text
habit_graph_consistency
habit_parent_normality
habit_fine_normality
habit_parent_confident_abnormal
habit_fine_confident_abnormal
habit_normality_gap
habit_inference_reobserve_score
```

빠른 재계산 출력:

```text
results/modular_benchmark/v190_model_habit_space_cls10_test1000_normality
```

결과:

| metric | value |
|---|---:|
| reobserve_required n | 38 / 1000 |
| reobserve state rate | 3.8% |
| candidate wrong recall in reobserve state | 66.7% |

상태 breakdown:

| state | selected_ok | candidate_ok | n |
|---|---:|---:|---:|
| reobserve_required | false | false | 2 |
| reobserve_required | false | true | 16 |
| reobserve_required | true | false | 20 |
| undecided | false | false | 9 |
| undecided | false | true | 3 |
| undecided | true | false | 2 |
| undecided | true | true | 948 |

해석:

```text
normality inference는 selected/candidate 전환 라벨을 학습하지 않고도,
강확신이지만 정상 evidence graph와 어긋나는 상태를 더 잘 잡았다.

특히 candidate wrong recall이 51.5% -> 66.7%로 증가했다.
이는 사용자가 말한 'train에 강한 확신 오답이 없어도, 모델 습관과 이미지 상황을 보고 추론한다'는 방향에 맞는 신호다.
```

현재 역할:

```text
v190 = switch approver가 아니라 reobserve/hold reasoner.
다음 판단기는 이 normality state를 action logits의 입력으로 써야 한다.
```

## 2026-07-10 - v191 Habit Normality Head

목표:

```text
v190 normality inference 수식을 그대로 IF로 쓰지 않고,
train에서 정상/비정상 evidence state의 약한 라벨을 만들어 작은 head를 학습한다.
오답 라벨이 아니라 normality risk 상/하위 구간을 사용한다.
```

추가 파일:

```text
tools/train_apply_habit_normality_head_v191.py
```

학습 target:

```text
positive = train habit_inference_reobserve_score 상위 3.5%
negative = train habit_inference_reobserve_score 하위 70%
ambiguous middle = 제외
```

실행:

```text
results/modular_benchmark/v191_habit_normality_head_cls10_test1000
```

학습 요약:

| item | value |
|---|---:|
| used rows | 3,530 |
| positive rows | 169 |
| negative rows | 3,361 |
| dropped ambiguous rows | 1,272 |
| feature count | 34 |
| train AUC | 1.0 |

기본 threshold 0.80 결과:

| metric | value |
|---|---:|
| review rate | 12.9% |
| selected wrong recall | 70.0% |
| candidate wrong recall | 69.7% |
| clean review count | 87 |

Threshold diagnostic:

| threshold | review | rate | selected wrong recall | candidate wrong recall | clean review |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 129 | 12.9% | 70.0% | 69.7% | 87 |
| 0.90 | 69 | 6.9% | 70.0% | 66.7% | 28 |
| 0.95 | 55 | 5.5% | 70.0% | 63.6% | 15 |
| 0.98 | 46 | 4.6% | 66.7% | 60.6% | 8 |

해석:

```text
v191 learned head는 v190 수식보다 넓게 위험 상태를 잡는다.
threshold 0.80은 과하게 넓고, 0.95 근처가 더 실용적인 운영점으로 보인다.

중요한 점은 train에 강한 확신 오답을 직접 주지 않아도,
normality risk pseudo target만으로 test의 selected/candidate wrong을 상당히 잡는다는 것.
```

현재 결론:

```text
v190 = 좁은 reobserve reasoner
v191 = 학습형 normality risk detector

다음에는 v191 score를 바로 switch/keep에 쓰지 말고,
reobserve/hold action token의 logit으로 쓰는 것이 맞다.
```

## 2026-07-10 - v192 Habit Action Policy Probe

목표:

```text
v191 normality detector가 잡은 위험 구간을 keep/switch/reobserve-like action으로 분리할 수 있는지 확인한다.
```

추가 파일:

```text
tools/eval_habit_action_policy_v192.py
```

입력:

```text
v190 normality rows:
results/modular_benchmark/v190_model_habit_space_cls10_test1000_normality/eval_habit_rows_v190.csv

v191 normality head score:
results/modular_benchmark/v191_habit_normality_head_cls10_test1000/eval_normality_head_rows_v191.csv
```

정책 진단:

```text
review if normality_head_score >= 0.95
switch if review and
  habit_fine_confident_abnormal >= 0.45
  habit_graph_consistency >= 0.38
  habit_normality_gap >= -0.02
else if review:
  reobserve_hold
else:
  keep_selected
```

결과:

```text
results/modular_benchmark/v192_habit_action_policy_cls10_test1000
```

| metric | value |
|---|---:|
| selected baseline parent acc | 97.00% |
| candidate parent acc | 96.70% |
| v192 parent acc | 97.90% |
| switch count | 23 |
| reobserve_hold count | 32 |
| review count | 55 |
| fixed | 11 |
| broken | 2 |
| switch precision parent | 91.30% |
| reobserve candidate-wrong recall | 57.58% |

주의:

```text
이 v192 threshold는 TEST 진단에서 좋은 조합을 본 것이므로 최종 train-only 모델 주장으로 쓰면 안 된다.
하지만 v190/v191이 단순 review detector에서 끝나지 않고,
risk 구간을 action으로 분리할 수 있다는 proof로는 의미가 있다.
```

해석:

```text
v190/v191이 잡은 위험 상태 안에서도
candidate가 parent를 보완하는 경우와 candidate가 위험한 경우가 분리된다.
다음 단계는 이 v192 action boundary를 train-only pseudo objective 또는 validation split에서 선택하게 만드는 것.
```

## 2026-07-10 - v193 Train-only Habit Action Policy

목표:

```text
v192는 TEST 진단 threshold였으므로,
train 내부 fit/val split에서 action policy threshold를 고르고
TEST에는 그대로 적용한다.
```

추가 파일:

```text
tools/train_eval_habit_action_policy_v193.py
```

학습 흐름:

```text
train parent csv split
  fit 1500 samples
  val 500 samples

fit:
  HabitProfile 학습
  normality pseudo label 생성
  normality head 학습

val:
  policy threshold sweep
  objective = fixed - 2.0 * broken - 0.03 * reobserve_hold

TEST:
  val에서 선택된 policy를 변경 없이 적용
```

선택된 policy:

```text
review_threshold = 0.99
fine_abnormal_threshold = 0.25
graph_threshold = 0.46
normality_gap_threshold = -0.02
```

결과:

```text
results/modular_benchmark/v193_train_only_habit_action_policy_cls10_test1000
```

| metric | val | TEST |
|---|---:|---:|
| selected parent acc | 100.00% | 97.00% |
| candidate parent acc | 50.70% | 96.70% |
| policy parent acc | 100.00% | 96.90% |
| fixed | 0 | 2 |
| broken | 0 | 3 |
| switch count | 0 | 5 |
| reobserve_hold count | 50 | 25 |
| review count | 50 | 30 |

해석:

```text
v193은 train-only 조건으로 전환했지만,
val split에서 selected가 이미 100%라서 정책이 switch를 거의 금지하는 방향으로 수렴했다.

따라서 v192의 97.90%는 action 분리 가능성의 proof였지만,
v193은 그 boundary를 train-val에서 안정적으로 학습하지 못했다.
```

중요한 결론:

```text
현재 train-val split은 TEST의 실제 위험 구조를 충분히 재현하지 못한다.

문제는 Habit/Normality feature 자체라기보다,
policy selection에 쓰는 validation objective가
"candidate가 매우 그럴듯하게 틀리는 상황"과
"selected가 틀렸지만 candidate가 복구 가능한 상황"을 충분히 제공하지 못한 것이다.
```

다음 방향:

```text
1. v192 같은 action split은 가능성이 있다.
2. 하지만 train-only selection에는 hard validation construction이 필요하다.
3. 단순 random val 대신:
   - selected fragile correct
   - candidate high-score wrong
   - selected wrong but candidate correct
   - both uncertain
   를 train 내부에서 균형 있게 구성해야 한다.
4. 또는 v193을 최종 gate가 아니라 reobserve queue detector로 제한해 쓰는 편이 현재 증거와 더 맞다.
```

## 2026-07-10 - v194 Situation Evidence Gate

목표:

```text
v193의 한계:
Gate가 모델 출력/normality 숫자는 봤지만,
그 판단이 나온 실제 이미지 상황 증거를 충분히 보지 못했다.

v194 목표:
object / texture / relation / judgment / habit 상황증거를 함께 넣어
Gate가 "이 전환을 승인해도 되는 상황인가" 또는
"다시 봐야 하는 상황인가"를 학습하게 한다.
```

추가 파일:

```text
tools/train_eval_situation_evidence_gate_v194.py
```

중요한 진단:

```text
train parent baseline accuracy = 100%

따라서 train 안에는
selected wrong + candidate correct
형태의 recovery positive가 없다.

즉 현재 train 조건에서는 direct switch approval gate를 학습할 수 없다.
```

그래서 v194는 자동 fallback으로 전환:

```text
transition_approval
  불가능: positive transition 없음

situation_review
  가능: 상황증거 기반 pseudo risk를 학습
```

입력 상황증거:

```text
object support:
  support_area_ratio
  support_tile_entropy
  unique_tile_ratio
  cross_region_class_agreement

texture / relation:
  texrel_*
  objrel_*
  wave_*

judgment:
  judgment_same_object_support
  judgment_evidence_completeness
  judgment_conflict_pressure
  judgment_fine_shortcut_risk

habit:
  habit_parent_normality
  habit_fine_normality
  habit_inference_reobserve_score
```

실험:

```text
results/modular_benchmark/v194_situation_evidence_gate_cls10_test1000_hgb
results/modular_benchmark/v194_situation_evidence_gate_cls10_test1000_logreg
```

결과:

| model | mode | selected acc | v194 acc | review | fixed | broken |
|---|---|---:|---:|---:|---:|---:|
| HGB | situation_review | 97.00% | 97.00% | 0 | 0 | 0 |
| LogReg | situation_review | 97.00% | 97.00% | 0 | 0 | 0 |

세부:

```text
val에서는 review가 91개 발생했지만,
TEST에서는 선택된 threshold 0.99를 넘는 row가 없어 review가 0개였다.

HGB TEST v194_approve_score:
  median 0.000064
  99%    0.000166
  max    0.629394
```

해석:

```text
상황증거를 쓰는 방향은 맞지만,
train에서 만든 pseudo situation-risk score를 raw threshold로 TEST에 옮기는 방식은 실패했다.

문제는 모델 종류(HGB/LogReg)가 아니라,
train situation-risk manifold와 TEST runtime score scale이 맞지 않는 것이다.
```

중요 결론:

```text
1. 현재 train은 parent가 완벽해서 recovery gate 학습 신호가 없다.
2. 따라서 direct approval gate는 train만으로 학습하기 어렵다.
3. 상황증거 gate는 threshold classifier보다
   "상태 추론 + reobserve/action planner"로 가는 편이 맞다.
4. 다음에는 pseudo risk threshold를 옮기지 말고,
   각 sample 내부의 candidate 간 상대 순위/상대 이상치로 판단해야 한다.
```

다음 방향:

```text
v195 후보:
sample-relative situation ranking

같은 sample 안에서:
  candidate들의 object/texture/relation 상황증거를 비교
  절대 threshold가 아니라 상대적으로 튀는 candidate를 review

목표:
  train/test score scale mismatch 제거
  direct recovery positive 부족 문제 완화
```

## 2026-07-10 - v195 Candidate-Relative Situation Gate

목표:

```text
v194의 절대 risk threshold 이식 실패를 피한다.

각 이미지의 후보 묶음 안에서
"현재 전환 후보가 다른 후보와 비교해 관측 관계가 비정상적인가"를 본다.
```

학습 입력:

```text
train CandidateBatch [N, K, 90]

support / wave / texture / object-flow 증거만 사용
정답, 오답, oracle, fixed/broken은 train 입력에 사용하지 않음
```

학습 방식:

```text
정상 train 후보
vs
관측 훼손 synthetic state
  - partial object support 감소
  - 반복 지역 질감 / diversity 감소
  - texture confidence는 남고 object relation이 끊긴 shortcut 상태

HGB가 candidate situation risk를 학습한다.
런타임 결정은 절대 threshold가 아니라
동일 이미지 후보들의 risk z-score와 score rank를 사용한다.
```

결과 (`AWA cls10 test 1000`):

| metric | value |
|---|---:|
| selected parent | 97.00% |
| candidate parent | 96.70% |
| v195 final parent | 97.00% |
| review | 1 / 1000 |
| selected wrong review recall | 1 / 30 = 3.33% |
| candidate wrong review recall | 0 / 33 = 0.00% |

유일한 review:

```text
horse_horse_11148
selected: dog (wrong)
candidate: horse (correct)
```

해석:

```text
candidate-relative 실행 구조 자체는 정상이다.
다만 current synthetic 훼손 상태가 실제 TEST 오답의 관측 분포를 충분히 닮지 않았다.

특히 candidate wrong 33건이 거의 모두 최저 risk에 몰렸다.
따라서 다음 개선은 threshold tuning이 아니라,
실제 crop/tile perturbation으로 object-flow / texture relation을 다시 계산해
synthetic 상황을 이미지 수준에서 만들도록 바꾸는 것이다.
```

## 2026-07-10 - v196 Actual BBox View Trajectory Audit

목표:

```text
Gate가 실제 관측 상황을 배울 수 있는지 확인한다.

모델 학습에는 사용하지 않고,
TEST 1000장의 44 bbox view 로그를 평가 후 truth로만 그룹화한다.
```

관측 로그:

```text
각 view의 10-class 확률을 parent별로 합산
각 이미지에서 selected parent / candidate parent / candidate fine의
view별 확률, crop 면적 반응, parent vote 분포를 기록
```

결과:

| group | n | parent vote entropy | mode ratio | parent view confidence |
|---|---:|---:|---:|---:|
| all | 1000 | 0.411 | 0.861 | 0.480 |
| selected wrong | 30 | 0.807 | 0.674 | 0.153 |
| candidate wrong | 33 | 0.810 | 0.670 | 0.110 |
| both wrong | 11 | 0.898 | 0.624 | 0.125 |

추가 관측:

```text
selected wrong + candidate correct (19):
  candidate - selected parent mean = +0.140
  candidate - selected parent max  = +0.615

selected correct + candidate wrong (22):
  candidate - selected parent mean = +0.041
  candidate - selected parent max  = +0.390
```

해석:

```text
실제 오답은 정상 샘플보다 view 간 parent 판단이 훨씬 불안정하다.

v195가 실패한 이유는 이 raw trajectory를 보기 전,
후보별로 이미 요약된 feature에 synthetic failure만 더했기 때문이다.

다음 Gate는 static confidence threshold가 아니라
view trajectory stability를 첫 상태로 사용해야 한다.

stable:
  accept 가능

unstable + candidate advantage 큼:
  직접 switch가 아니라 reobserve 우선

unstable + candidate advantage 작음 또는 both abnormal:
  reobserve / abstain
```

## 2026-07-10 - v197 View Trajectory Stability Gate

목표:

```text
train 정답/오답을 직접 학습하지 않고,
실제 44-view 관측 궤적의 정상 분포만 배운다.

TEST에서는 unstable sample만 reobserve_hold로 보낸다.
```

train:

```text
AWA cls10 train 2000
label-free v113 bbox trajectory

features:
  candidate/selected parent 우위 평균·최대·분산
  crop scale 반응
  parent vote entropy
  parent vote mode ratio
  view parent confidence / margin

model:
  RobustScaler + IsolationForest
  train risk 상위 5%를 review threshold로 사용
```

TEST 1000 사후 audit:

| group | n | review | review rate |
|---|---:|---:|---:|
| both correct | 948 | 7 | 0.74% |
| selected wrong, candidate correct | 19 | 19 | 100.0% |
| selected correct, candidate wrong | 22 | 22 | 100.0% |
| both wrong | 11 | 3 | 27.3% |

전체 지표:

```text
review: 51 / 1000 = 5.1%
selected wrong recall: 22 / 30 = 73.3%
candidate wrong recall: 25 / 33 = 75.8%
```

해석:

```text
v197은 selected/candidate parent가 충돌하는 41개 실제 위험 전환을
모두 reobserve로 보냈고, stable both-correct 948개는 거의 건드리지 않았다.

중요: 아직 switch하지 않는다.
v197의 역할은 "후보가 좋다" 판정이 아니라,
"현재 관측만으로 Parent/Fine 전환을 승인하면 위험하다"를 탐지하는 것이다.

다음 단계는 이 51개에 대해 candidate-support / wide-context 재관측을 실제 실행하고,
재관측 궤적이 19 recovery와 22 protection을 분리할 수 있는지 보는 것이다.
```

## 2026-07-10 - v198/v199 Actual Reobserve Action Replay

목표:

```text
v197이 잡은 sample에 대해 바로 후보를 교체하지 않는다.

candidate-support bbox family
selected-support bbox family
wide-context bbox family

를 실제 44-view 결과에서 다시 묶어 관측한 뒤,
어떤 action이 candidate 전환을 지지하는지 평가한다.
```

v198 runtime action bank:

```text
입력:
  v197 reobserve request
  raw bbox probabilities / geometry

출력:
  각 family의 parent, confidence, view consensus,
  requested parent evidence, wide agreement

정답 컬럼 없이 생성 가능함을 별도 label-free run으로 확인.
```

v199 train-only selector:

```text
train의 실제 partial bbox observation을 provisional selected state로 사용.
각 alternate parent에 candidate-support / wide action을 만든다.
true parent는 action 학습 target으로만 사용한다.
class name 자체는 feature에서 제외한다.

train rows: 24,755
train-positive rate: 12.0%
validation AUC: 0.995
validation threshold: precision >= 0.97 기준
```

TEST 51 reobserve request audit:

| metric | value |
|---|---:|
| selected accuracy on reviewed | 56.86% (29 / 51) |
| v199 action accuracy | 68.63% (35 / 51) |
| fixed | 16 |
| broken | 10 |
| net | +6 |
| switch | 28 |

세부:

```text
recovery (selected wrong, candidate correct):
  19개 중 16개 switch 및 fixed

protection (selected correct, candidate wrong):
  22개 중 10개 잘못 switch
```

해석:

```text
v197 -> v198 -> v199 경로는 실제 수정 능력을 처음으로 보였다.
그러나 train partial-observation action distribution이 실제 Parent/Fine protection
상황을 충분히 재현하지 못해 broken 10개가 남았다.

따라서 다음 개선은 TEST threshold 조정이 아니다.

train에서 Parent/Fine 실제 candidate conflict를 더 풍부하게 만들거나,
candidate-support와 wide-context의 independent agreement를
train action selector의 보호 증거로 강화해야 한다.
```

## 2026-07-11 - v200 Standalone Reobserve Gate

목표:

```text
v197 -> v198 -> v199 파일 체인을 없앤다.

v200 하나가 raw view artifact와 shared gate input만 받아
안정성 탐지, 재관측 family 구성, switch approval을 모두 수행한다.
```

독립 입력:

```text
train:
  bbox_candidate_scores.npz
  v153_gate_input.npz
  train truth parent CSV (action supervision only)

runtime/eval:
  bbox_candidate_scores.npz
  v153_gate_input.npz

eval truth CSV는 audit 옵션이며 runtime feature에는 사용하지 않는다.
```

내부 단계:

```text
1. raw 44-view trajectory -> IsolationForest stability review
2. review sample만 candidate-support / selected-support / wide family 생성
3. action selector
   - synthetic partial recovery rows: 24,755
   - real Parent/Fine hard-protection rows: 71
4. keep / reobserve_hold / switch_candidate 출력
```

label-free runtime artifact 검증:

```text
keys:
  sample_key
  selected_parent
  candidate_parent
  final_parent
  reobserve
  switch
  trajectory_risk
  action_score

true/correct/oracle/fixed/broken/target/approve key: 없음
allow_pickle=False: 정상 로드
```

TEST 1000 audit (평가용):

| metric | value |
|---|---:|
| selected parent | 97.00% |
| v200 final parent | 97.60% |
| selected fine (10-class) | 94.70% |
| v200 final fine (10-class) | 95.20% |
| review | 52 |
| switch | 15 |
| fixed | 10 |
| broken | 4 |
| fine fixed | 8 |
| fine broken | 3 |
| net | +6 |

해석:

```text
v200은 v199와 같은 net +6이지만,
switch를 28 -> 15로 줄이고 broken을 10 -> 4로 낮췄다.

real Parent/Fine hard-protection train rows를 별도 source로 넣은 것이
과신 전환을 줄이는 방향으로 작동한 신호다.

v200 runtime output은 parent만이 아니라
selected_class / candidate_class / final_class도 함께 저장한다.
다만 review/action의 시작 조건은 아직 parent conflict 중심이다.
따라서 Persian↔Siamese처럼 같은 parent 내부의 fine conflict는
다음 fine-level trajectory gate에서 별도로 다뤄야 한다.

이 TEST는 v196~v199 탐색에 사용됐으므로 최종 일반화 수치로는 사용하지 않는다.
v200의 독립성은 artifact/코드 의존성의 의미이고,
성능 일반화는 별도 untouched set에서 다시 검증해야 한다.
```

## 2026-07-11 - v201 Evidence Role + Probe Planner

목적은 후보를 바로 전환하는 것이 아니라, 실제 raw bbox view에서 후보 증거가
독립적인지, 국소 지름길인지, 추가 구분 관측이 필요한지를 판단하고 재관측 계획을
만드는 것이다.

```text
raw bbox views + shared selected/candidate state
    -> local candidate support
    -> local support ablation / residual evidence
    -> wide-context evidence
    -> learned evidence trust (TRAIN only)
    -> evidence role + recommended probe
```

역할과 계획:

| evidence role | 의미 | runtime plan |
|---|---|---|
| independent_support | 국소 제거 후에도 후보 지지가 남고 wide와 일치 | candidate_support_confirm |
| shortcut_support | 후보 지지가 국소 crop에 과도하게 묶임 | wide_context |
| missing_disambiguator | local/wide 관측이 충돌하거나 공간 지지가 약함 | boundary_expand |
| unresolved | 현재 관측으로 역할 불명 | abstain |

cls10 TEST1000 exploratory audit:

| metric | value |
|---|---:|
| parent-conflict rows | 43 |
| reobserve request | 39 |
| selected-wrong review recall | 90.48% |
| candidate-wrong review recall | 91.67% |
| shortcut_support | 34 |
| independent_support | 4 |
| missing_disambiguator | 1 |
| unresolved | 4 |

해석:

```text
v201은 현재 후보 교체기가 아니다.
실제 local/residual/wide 관측을 근거로 "왜 다시 봐야 하는가"를 label-free runtime
artifact로 분리한 첫 단계다.

shortcut_support 안에 candidate 정답/오답이 함께 남아 있으므로,
이 결과만으로 자동 switch를 승인하면 안 된다. 다음 단계는 wide_context 또는
boundary_expand를 실제로 실행한 뒤, 새 증거가 기존 후보를 복구/반증하는지를
학습하는 것이다.

TEST1000은 이미 설계 탐색에 사용됐으므로 이 수치는 진단용이다. runtime artifact는
test label을 포함하지 않으며, 최종 일반화는 untouched set에서 검증해야 한다.
```

## 2026-07-11 - v202 Planned Probe Re-aggregation

v201이 제안한 probe를 기존 44-view bbox bank에서 실제로 다시 합산했다. 이는 새
crop을 생성하는 재스캔이 아니라, 현재 관측 bank 안에서 wide/boundary family가
갈등을 해소할 수 있는지 확인하는 저비용 반증 실험이다.

| requested rows | selected parent | candidate parent | reobserve parent |
|---:|---:|---:|---:|
| 39 | 51.28% | 43.59% | 51.28% |

```text
selected -> reobserve: fixed 15 / broken 15
candidate -> reobserve: fixed 5 / broken 2
wide_context reobserve accuracy: 55.88% (34 rows)
```

해석:

```text
v201이 위험/모호 관측을 모으는 역할은 했지만,
기존 44-view bank를 다시 가중합하는 것만으로는 selected를 안전하게 이길 새 정보가
생기지 않았다.

따라서 다음 재관측은 기존 bbox 후보 재집계가 아니라,
wide_context / boundary_expand 계획으로 새 crop 또는 새 scan을 실제 생성해야 한다.
이 결과는 후보 selector 문제와 관측 공간 부족을 분리한 반증 결과다.
```

## 2026-07-11 - v203 Actual Probe Crop Scan

v203은 v201 plan을 받아 기존 44-view bank를 재가중하지 않고, 원본 이미지에서
candidate-support bbox를 기준으로 새 crop을 생성해 frozen ImageNet backbone으로 다시
점수화했다.

```text
wide_context:
  candidate support bbox를 상하좌우 1 tile 확장
  + 같은 높이의 full-width context crop

boundary_expand:
  candidate support bbox의 상/하/좌/우 경계 확장 crop

abstain:
  새 crop을 억지로 적용하지 않고 selected를 그대로 유지
```

## 2026-07-12 - v205 Train-Only Evidence Separator

목표는 Gate가 실제 fresh 관측을 읽고 `stable_keep`, `protect_keep`,
`recover_switch`, `unresolved_retry`를 학습으로 구분할 수 있는지 확인하는 것이다.

데이터 구성:

```text
protection episode 2,000:
  실제 label-free MVP selected + 강한 competing fresh view

shortcut-recovery episode 2,000:
  raw view에서 가장 강하게 끌린 wrong parent를 synthetic selected로 사용
  실제 MVP parent를 candidate로 사용

총 4,000 episode
동일 이미지의 두 episode는 GroupShuffleSplit으로 같은 split에 고정
train 3,000 / heldout 1,000
```

## 2026-07-12 - v206 Base/Fresh Visual Relation

v205에는 fresh 512-d embedding만 있었으므로, 실제 selected-support crop의
ResNet18 penultimate embedding을 2,000 train 이미지에서 추가로 생성했다.

Gate 입력 추가:

```text
base embedding
fresh embedding
abs(base - fresh)
base * fresh
cosine similarity / L2 distance / norm ratio
48-d PCA relation representation (train split에서만 fit)
```

## 2026-07-12 - v207 Predictive Evidence Compatibility

정답 parent 일치 여부를 분류하는 대신, 관측 A에서 아직 보지 않은 관측 B의
candidate support를 예측하는 self-supervised compatibility probe를 실행했다.

```text
TRAIN 2,000 images
44 views -> observed support 4 views + independent/wide hidden views
6 parent hypotheses per image = 12,000 episodes
image-group train 1,500 / heldout 500
predictor target = hidden candidate support (truth label 미사용)
truth parent = 마지막 audit에만 사용
```

## 2026-07-12 - v208 Hidden View Token Prediction

v207의 scalar support 대신 TRAIN 2,000장 × 44 bbox view = 88,000개의
ResNet18 penultimate embedding(512-d)을 생성했다. observed 4-view token set에서
hidden/wide token mean을 self-supervised 방식으로 예측했다.

동일 seed 207 비교:

| metric | v207 scalar | v208 token |
|---|---:|---:|
| observed-only top1 | 92.60% | 92.60% |
| hidden-only top1 | 92.80% | 92.80% |
| predictive compatibility top1 | 93.00% | 93.20% |
| candidate AUC | 0.9815 | 0.9819 |
| observed -> compatibility fixed / broken | 7 / 5 | 10 / 7 |

해석:

```text
token prediction은 scalar 대비 +0.2%p 추가 개선과 net +3을 냈지만,
여전히 hidden support 자체가 점수의 대부분을 설명한다.

현재 predictor는 candidate 개념 embedding을 직접 조건으로 받지 않고, candidate가
선택한 observed view 집합을 통해서만 간접 조건화된다. 또한 token set을 평균내므로
부위별 어긋남이 소실된다. 다음 단계에서 이 방향을 계속한다면 candidate concept
token + cross-attention/set loss가 필요하다.
```

## 2026-07-12 - v209 Candidate-Conditioned Token Set Predictor

candidate concept prototype와 observed 4-view token을 Transformer encoder/decoder에
넣고 hidden 4-view token set을 Chamfer cosine loss로 예측했다. correctness label은
학습에 사용하지 않았다.

동일 seed 207 heldout:

| metric | value |
|---|---:|
| train / heldout images | 1,500 / 500 |
| final train set loss | 0.2867 |
| heldout set error | 0.2973 |
| candidate AUC | 0.9855 |
| hidden-only top1 | 94.40% |
| set compatibility top1 | 94.40% |
| hidden vs compatibility changed choice | 0 / 500 |

해석:

```text
candidate-conditioned set prediction은 학습되고 correct candidate의 set error도 더
낮았지만, 최종 compatibility 순위는 hidden support 순위와 500개 모두 동일했다.

즉 predictor는 후보가 관측을 설명하는 신호를 얻었지만, 현재 점수식에서 hidden
class support가 압도적이라 독립적인 판단축으로 작동하지 못했다. 이 상태에서 모델을
더 키우는 것은 의미가 작다. 다음 분석은 prediction error가 hidden support와 독립인
hard-conflict subset에서만 유용한지, 아니면 concept prototype이 candidate 차이를
충분히 담지 못하는지를 구분해야 한다.
```

## 2026-07-12 - v210 Direct prototype_parent_match Surrogate

초점을 상하위 분류나 관측 예측이 아니라 다음 함수의 직접 근사로 변경했다.

```text
target = 1[candidate_parent == true_parent]       # TRAIN only
runtime input = candidate support + wide support
                + train-only parent prototype와 view token의 관계값
candidate 이름/정답은 runtime feature에서 제외
```

## 2026-07-12 - v210 TEST1000 + v211 Approval Gate

v210 train model을 cls10 TEST1000 44-view token에 label-free 적용했다.

```text
selected parent: 97.00%
v210 candidate:  96.80%
always switch: fixed 20 / broken 22
```

따라서 v210은 방향 후보를 만들지만 단독 최종 판단자는 아니다. TRAIN heldout에서
selected-vs-candidate match 관계를 읽는 v211 approval gate를 학습했다.

| 정책 | threshold | switches | final | fixed / broken |
|---|---:|---:|---:|---:|
| min train precision 95% | 0.0180 | 44 | 96.80% | 20 / 22 |
| min train precision 99.5% | 0.9333 | 11 | 97.70% | 9 / 2 |

해석:

```text
느슨한 Gate는 v210 후보를 사실상 모두 승인해 실패했다. TRAIN precision 제약을
99.5%로 높이자 TEST label을 사용하지 않고도 9개를 복구하고 2개만 깨뜨려
97.0% -> 97.7%로 상승했다.

이는 v210 match surrogate가 전환 방향 신호로 사용 가능함을 보여주지만 legacy
label-derived v153 99.4%에는 아직 큰 차이가 있다. 현재 병목은 candidate 방향보다
hard-protection approval 학습과 match surrogate의 4.2% top1 오차다.
```

동일 seed 207 image-group heldout:

| metric | value |
|---|---:|
| candidate-level AUC | 0.9964 |
| binary match accuracy | 97.37% |
| positive match precision / recall | 89.35% / 95.60% |
| wide-support top1 parent | 93.20% |
| nearest prototype top1 parent | 95.00% |
| learned surrogate top1 parent | 95.80% |
| wide -> surrogate fixed / broken | 16 / 3 |
| nearest prototype -> surrogate fixed / broken | 12 / 8 |

해석:

```text
prototype_parent_match의 역할을 직접 목표로 두자 이전 우회 실험보다 강한 결과가
나왔다. 같은 heldout에서 v207 93.0%, v208 93.2%, v209 94.4%였던 top1이
v210에서 95.8%로 상승했다.

이는 필요한 기능이 미래 관측 재구성보다 candidate-image identity membership
추정에 가깝다는 가설을 지지한다. 다만 train-only parent prototype을 만들 때 parent
supervision을 사용하므로 자동 개념 형성 문제는 아직 해결하지 않았다. 현재 단계는
label-derived runtime feature를 train-supervised/runtime-safe surrogate로 교체한 것.
```

결과:

| metric | value |
|---|---:|
| hidden support prediction MAE | 0.0120 |
| candidate correctness AUC | 0.9815 |
| observed-only top1 parent | 92.60% |
| hidden-only top1 parent | 92.80% |
| predictive compatibility top1 | 93.00% |
| observed -> compatibility fixed / broken | 7 / 5 |

해석:

```text
숨겨진 관측 예측은 가능했고 compatibility 신호도 존재한다.
그러나 top1 개선은 +0.4%p, net +2에 그쳤다.

현재 scalar parent support 예측은 올바른 후보의 support 크기를 다시 추정하는
성격이 강하고, 잘못된 후보가 어떤 시각 증거를 잘못 예측했는지는 충분히 표현하지
못한다. 다음 확장은 scalar probability가 아니라 hidden view embedding/token set을
예측하고, hard competing candidate subset에서 검증해야 한다.
```

공정 비교를 위해 v205와 동일한 image-group split seed 205를 사용했다.

| metric | v205 | v206 |
|---|---:|---:|
| heldout accuracy | 94.60% | 94.90% |
| recover precision | 97.03% | 98.07% |
| recover recall | 96.62% | 96.62% |
| protect recall | 72.92% | 83.33% |
| unresolved recall | 29.63% | 40.74% |
| harmful switch | 12 | 8 |
| switch precision | 97.03% | 98.07% |

해석:

```text
base visual embedding 자체보다 base-fresh 관계가 protection에 도움을 줬다.
동일 split에서 harmful switch가 12 -> 8로 감소했고 protect recall은
72.9% -> 83.3%로 상승했다.

아직 protect 90%에는 못 미친다. 남은 병목은 base embedding 부재가 아니라
protect hard-negative 수량(총 185)과 view별 평균 이전의 관계 손실이다.
```

heldout 결과:

| state | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| stable_keep | 97.38% | 98.67% | 98.02% | 452 |
| recover_switch | 97.03% | 96.62% | 96.82% | 473 |
| protect_keep | 62.50% | 72.92% | 67.31% | 48 |
| unresolved_retry | 53.33% | 29.63% | 38.10% | 27 |
| overall | - | - | 94.60% accuracy | 1,000 |

승인 audit:

```text
switch count: 471
beneficial switch: 457
harmful switch: 12
wrong-to-wrong switch: 2
switch precision: 97.03%
protect keep rate: 75.00%
```

해석:

```text
Gate가 base/fresh EvidenceState를 학습할 수 있다는 첫 정식 증거다.
특히 recover와 stable은 heldout 이미지에서도 96~98% 수준으로 분리됐다.

그러나 protect 사례가 185개뿐이고 heldout support가 48개라 보호율은 75%에
머물렀다. 따라서 전체 accuracy 94.6%만 보고 운영 Gate가 완성됐다고 판단하면
안 된다. 다음 병목은 표현보다 hard-protection episode 수집/oversampling 및
switch threshold calibration이다.

이 평가는 TEST가 아니라 TRAIN 내부 image-group heldout이다. 최종 일반화 성능은
Gate 정책을 고정한 뒤 untouched eval에서 별도로 확인해야 한다.
```

39개 request audit:

| policy | parent accuracy | fixed | broken |
|---|---:|---:|---:|
| selected | 51.28% | - | - |
| 모든 fresh crop 강제 적용 | 51.28% | 15 | 15 |
| plan 적용 (`abstain=keep`) | 61.54% | 15 | 11 |

해석:

```text
새 crop 자체는 실제로 selected/candidate와 다른 증거를 만들었다.
특히 wide_context 34개에서는 selected 47.06% -> fresh 58.82%로 상승했다.

다만 모든 재관측 결과를 강제로 사용하면 abstain 사례에서 손실이 발생한다.
따라서 "재관측"과 "재관측 결과 채택"은 분리해야 한다.

v203은 새 관측을 만드는 단계의 순이득을 처음 확인했지만, 이 TEST1000은 이미
탐색에 사용됐으므로 최종 수치가 아니다. 다음 학습은 train에서 fresh crop 결과가
selected보다 우세한 경우만 승인하는 acceptance gate를 만들어 untouched set에서
검증한다.
```

## v212: Bidirectional Parent/Fine message passing

목적:

```text
Fine evidence가 Parent identity를 보완하고,
Parent identity가 Fine candidate를 보정하도록
두 표현을 2-round residual message passing으로 공동 추론한다.
```

구현:

- `tools/train_eval_bidirectional_parent_fine_v212.py`
- Fine node 10개: label-free v131 texture/object/wave evidence
- Parent node 6개: v210 learned parent-match surrogate evidence
- Fine -> Parent: child evidence 평균 메시지
- Parent -> Fine: candidate가 속한 parent 메시지
- loss: fine CE + parent CE + fine-to-parent consistency CE
- `true_parent`, `prototype_parent_match`, `bundle_correct` 등 정답 파생 컬럼은 입력에서 제외
- 수동 parent map은 사용했으므로 자동 concept discovery 실험은 아님

TEST1000 결과:

| model | fine acc | parent acc | fine fixed/broken | parent fixed/broken |
|---|---:|---:|---:|---:|
| selected/MVP | 94.70% | 97.00% | - | - |
| v212 no interaction (`rounds=0`) | 94.20% | 96.30% | 28 / 33 | 19 / 26 |
| v212 bidirectional (`rounds=2`) | 94.10% | 96.00% | 31 / 37 | 24 / 34 |
| leaked legacy v153 reference | 96.60% | 99.40% | - | - |

해석:

```text
양방향 계산은 정상 작동했지만 일반화 성능은 오르지 않았다.
rounds=0도 이미 baseline보다 낮고, rounds=2의 추가 하락은 작다.
따라서 실패의 주원인은 message passing 자체보다 입력 evidence의 비보완성이다.

현재 v210 parent surrogate와 v131 fine evidence는 서로 독립적인 관점이라기보다
동일한 쉬운 질감/후보 패턴을 상당 부분 공유한다. 이 상태에서 상호작용을 늘리면
새 정보가 생기지 않고 오류 확신도 함께 전달된다.

다음 Parent/Fine 상호작용 실험은 구조를 더 깊게 하기 전에,
각 branch가 실제로 다른 샘플을 고치는지와 error correlation을 먼저 측정해야 한다.
```

## v213: Listwise identity-membership ranking

목적:

```text
v210의 runtime-safe identity membership feature와 train-only parent prototype은 유지한다.
candidate를 독립 binary sample로 학습하지 않고,
한 이미지의 parent 후보 6개를 동시에 비교하여 true parent를 1위로 올린다.
```

구현:

- `tools/train_parent_match_listwise_v213.py`
- `tools/apply_parent_match_listwise_v213.py`
- 입력: v210의 16개 runtime-safe evidence + 샘플 내부 상대 feature 16개
- 학습: 6-way listwise cross entropy
- candidate 이름과 정답 값은 feature에서 제외
- parent truth는 TRAIN loss 및 평가 audit에만 사용
- parent map/prototype은 수동 parent 정의를 사용하므로 아직 완전 자동은 아님

동일 seed 207 TRAIN group-heldout:

| method | parent top-1 |
|---|---:|
| nearest parent prototype | 95.00% |
| v210 binary match surrogate | 95.80% |
| v213 listwise membership | 96.60% |

```text
v213 candidate AUC: 99.71%
v210 대비 top-1: +0.80%p
```

TEST1000:

| method | parent accuracy | fixed | broken | net |
|---|---:|---:|---:|---:|
| selected/MVP | 97.00% | - | - | - |
| v210 always-select | 96.80% | 20 | 22 | -2 |
| v213 always-select | 97.50% | 21 | 16 | +5 |
| v211 conservative gate reference | 97.70% | 9 | 2 | +7 |

해석:

```text
identity membership의 정의가 틀린 것이 아니라 후보 간 순위 목적이 빠져 있었다.
binary AUC가 높아도 샘플 내부 top-1을 직접 최적화하지 않으면 최종 선택이 약해진다.
v213은 별도 전환 Gate 없이도 selected보다 +0.50%p 높아졌다.

다음 병목은 membership 후보 생성이 아니라 승인 문제다.
다만 TRAIN selected parent가 100%여서 실제 recovery Gate 사례가 없으므로,
합성 selected 오답보다 OOF selected prediction을 먼저 만들어야 한다.
```

## v214: Real-validation identity approval gate

목적:

```text
합성 selected 오답을 사용하지 않는다.
stage0가 학습하지 않은 TRAIN val 500장의 실제 selected 오류로
v213 identity candidate의 전환 승인 경계를 고정한다.
```

입력:

- selected parent: stage0 unseen validation prediction
- candidate parent/confidence/margin: v213 listwise membership
- candidate-selected membership delta: v210 evidence
- threshold calibration truth: TRAIN validation에서만 사용
- TEST 런타임 입력에는 정답을 사용하지 않음

TRAIN val 500 calibration:

```text
selected parent: 97.40%
v213 candidate: 97.80%
v214 final: 98.80%
switch 9 / fixed 8 / broken 1
```

TEST1000 개발 평가:

| method | parent accuracy | fixed | broken | net |
|---|---:|---:|---:|---:|
| selected/MVP | 97.00% | - | - | - |
| v213 always-select | 97.50% | 21 | 16 | +5 |
| v211 synthetic/conservative gate | 97.70% | 9 | 2 | +7 |
| v214 real-validation gate | 98.10% | 19 | 8 | +11 |

해석:

```text
v210/v213 identity membership 신호는 최종 선택에 실제 도움이 된다.
병목은 membership 표현이 아니라 실제 recovery/protection 사례가 없는 Gate 학습이었다.
unseen TRAIN validation의 실제 오류를 사용하자 합성 recovery보다 복구-파손 균형이 좋아졌다.

단, TEST1000은 이전 개발 과정에서 반복 분석된 셋이므로 untouched 최종 검증이 아니다.
v214 threshold를 이제 고정하고 새로운 외부/untouched set에서 한 번 검증해야 한다.
```

### v214 fixed-policy external validation

평가셋:

```text
dataset/external_test_sources/awa_cls10_external_hf90_mixed_cap60/test
n = 600
```

정책 고정 조건:

- v210 prototype/model: TRAIN에서 저장된 모델 그대로 사용
- v213 listwise model: seed207 TRAIN 모델 그대로 사용
- v214 threshold: TRAIN val500에서 고정한 세 값 그대로 사용
- 외부 결과를 본 뒤 threshold 재탐색/수정하지 않음
- 외부 정답은 마지막 audit에만 사용

| method | parent accuracy | fixed | broken | net |
|---|---:|---:|---:|---:|
| selected/MVP | 96.17% | - | - | - |
| v210 always-select | 97.17% | 15 | 9 | +6 |
| v213 always-select | 97.50% | 14 | 6 | +8 |
| v214 fixed gate | 97.50% | 11 | 3 | +8 |

```text
v214 switch count: 16
selected 대비: +1.33%p
v213과 최종 정확도는 같지만 broken 6 -> 3
```

해석:

```text
identity membership 중심의 v210 -> v213 -> v214 개선이 외부 분포에서도 유지됐다.
listwise ranking은 parent 후보 품질을 높였고, real-validation gate는 같은 순이득을
더 적은 전환과 절반의 파손으로 달성했다.

이 외부셋은 v214 threshold 선택에는 사용되지 않았지만 과거 프로젝트 분석 이력은
있으므로 완전히 새로운 최종 benchmark라고 표현하지 않는다. 고정 정책의 외부
재검증 결과로 기록한다.
```

## v215: Joint Parent/Fine state contract

목적:

```text
Parent를 Fine의 선행 조건으로 두지 않는다.
Gate 입력 직전의 Parent/Fine 독립 판단을 네 상태로 기록하고,
상태별 승인/복구/재관측 행동 계약을 만든다.
```

구현:

- `tools/build_joint_parent_fine_state_v215.py`
- 입력: no-leak v153의 Gate 이전 `selected_parent`, `fine_label`
- 최종 v153 판단은 feature/상태 판정에 사용하지 않음
- 정답은 audit와 향후 TRAIN state target 생성에만 사용

No-leak TEST1000 상태:

| state | count | rate | target action |
|---|---:|---:|---|
| both_valid | 929 | 92.9% | accept_both |
| parent_only | 41 | 4.1% | keep_parent_reobserve_fine |
| fine_only | 16 | 1.6% | repair_parent_from_fine |
| neither | 14 | 1.4% | reobserve_both |

```text
Parent branch accuracy: 97.0%
Fine branch accuracy: 94.5%
Parent/Fine implied-parent disagreement: 40장
Fine -> Parent repair opportunity: 16장
Parent -> Fine repair opportunity: 41장
```

해석:

```text
Fine이 Parent의 조건부 하위 단계가 아니라 독립적인 복구 증거가 될 수 있다는
상태가 no-leak 결과에서도 직접 확인됐다. 동시에 Parent-only와 neither도 존재하므로
Fine을 항상 신뢰하는 역방향 트리 역시 적합하지 않다.

다음 모델은 단일 switch 확률이 아니라
both / parent-only / fine-only / neither 상태와 행동을 공동 추론해야 한다.
```

TRAIN artifact 제약:

```text
기존 train_transition_rows.csv의 selected_parent는 2000장 모두 'unknown'인 legacy bug가 있다.
selected_parent_base는 복구 가능하지만 TRAIN 직접 예측이라 parent 오류가 사실상 없다.
따라서 기존 TRAIN row만으로는 fine-only/neither 상태를 학습할 수 없다.

다음 필수 작업:
stage0 unseen val500에 no-leak Fine evidence/candidate를 생성하여
실제 네 상태를 가진 Joint Gate 학습/검증 데이터를 만든다.
```

## v216a: Chain-free Joint Gate packet + TRAIN-only first fit

목적:

```text
Parent/Fine를 독립 evidence provider로 유지하고,
하나의 Joint Gate가 branch validity와 네 상태를 한 번에 추론한다.
TEST 정답은 런타임 입력에 넣지 않는다.
```

구현:

- `src/dual_line/decision/joint_gate_v216.py`
- `src/dual_line/decision/joint_gate_packets_v216.py`
- `tools/build_joint_gate_packets_v216.py`
- `tools/select_joint_gate_fine_evidence_v216.py`
- `tools/train_eval_joint_gate_v216.py`
- `tools/build_texture_relation_cache_v130.py --sample_keys_csv`

런타임 packet:

| packet | shape | 주요 내용 |
|---|---:|---|
| Parent | 5 | confidence, margin, membership, wide agreement, stability |
| Fine | 7 | confidence, support, coverage, relation agreement, risk |
| Cross | 6 | agreement, delta, independence, shared overconfidence |
| Observation | 4 | uncovered wide/local, duplicate view, observation gap |

모든 packet은 allow-list로 생성하며 truth/correct/oracle/teacher 필드를 읽지 않는다.
health audit 결과 상수/NaN feature는 없었다. `membership_delta`만 Parent/Fine parent가
같은 대부분의 표본에서 0이므로 sparse warning으로 기록했다.

### 발견 및 수정

1. v113 multiclass 입력 정규화 오류

```text
기존 val500 label-free cache의 selected_class = cat/dog/nan
원인: 10-class y_pred 정수를 과거 {0:cat, 1:dog}로 매핑
수정: y_pred_name 우선, 없으면 prob_* 순서로 class map 추론
```

수정 후 val500 v113:

```text
n=500
base accuracy=94.8%
candidate oracle=99.6%
```

2. v130 부분 실행 지원

```text
--sample_keys_csv 추가
2500장 scan 폴더에서 val500만 선택
최신 roi_align_pipeline v130 처리: 86.4초
```

3. TRAIN-only val500 Fine evidence 생성

```text
v117 -> v119 -> v121 -> v130 -> v130c -> v131 -> v150
Fine accuracy=95.0%
Parent accuracy=97.4%
```

val500 네 상태:

| state | count |
|---|---:|
| both_valid | 473 |
| parent_only | 14 |
| fine_only | 2 |
| neither | 11 |

### v216a 결과

동일한 최신 relation/evidence pipeline으로 정렬한 TEST1000에서 validity 신호는 확인됐다.

| metric | TEST1000 |
|---|---:|
| Parent validity AUC | 0.742 |
| Fine validity AUC | 0.886 |
| Parent balanced accuracy | 0.692 |
| Fine balanced accuracy | 0.826 |
| base Parent accuracy | 97.0% |
| v216a Parent accuracy | 97.0% |
| fixed / broken | 0 / 0 |

해석:

```text
성공:
- chain-free packet과 Joint Gate 학습/평가 경로가 연결됨
- runtime truth leakage 없음
- 동일 evidence schema일 때 branch validity가 일반화 신호를 보임

미달:
- val500에 fine_only가 2장뿐이라 Fine -> Parent repair 상태를 학습하지 못함
- state head는 fine_only를 한 번도 예측하지 않아 보수적으로 keep만 수행
- 따라서 v216a는 정확도 개선 모델이 아니라 validity detector 첫 검증 상태
```

다음 우선순위:

```text
TEST threshold 튜닝 금지.
TRAIN 내부 out-of-fold 또는 hard-view augmentation으로 fine_only/neither를 늘린다.
같은 Fine selector와 같은 relation backend로 packet schema를 고정한다.
그 후 v216b에서 repair/action head를 활성화한다.
```

## v216b-1: TRAIN-only real OOF state collection (2026-07-13)

목적:

```text
단일 2000/500 split에서 거의 없었던 fine_only/neither 상태를
TEST 또는 인공 오답 없이 실제 held-out 예측으로 수집한다.
```

구현:

- `tools/build_joint_gate_oof_states_v216b.py`
- AWA cls10 TRAIN 2500장을 class-stratified 5-fold로 분할
- Parent v0.2 head는 각 fold의 나머지 80%만 학습하고 held-out 20% 예측
- Fine v150 selector도 held-out fold를 제외한 candidate row만 학습
- v130/v131 후보 증거는 최신 `roi_align_pipeline`으로 2000장 부분을 재생성
- 모든 2500장 Parent/Fine 결과는 해당 샘플을 학습하지 않은 OOF 예측
- TEST 입력/threshold/결과 사용 없음

결과:

| metric | result |
|---|---:|
| OOF samples | 2500 |
| Parent accuracy | 97.08% |
| Fine accuracy | 96.60% |
| both_valid | 2366 |
| parent_only | 61 |
| fine_only | 49 |
| neither | 24 |
| Parent/Fine parent disagreement | 79 |

기존 val500 대비 희귀 상태:

```text
val500 fine_only = 2
OOF2500 fine_only = 49

val500 neither = 11
OOF2500 neither = 24
```

무결성 확인:

```text
rows=2500
unique sample_key=2500
duplicate=0
각 fold=500장
모든 fold에 four-state가 존재
```

해석:

```text
v216a의 repair 미작동은 Joint Gate 표현만의 문제가 아니라
단일 split에서 fine_only 상태가 2장뿐이었던 학습자료 부족의 영향이 컸다.

OOF 수집으로 parent_only/fine_only/neither가 모두 실제 예측에서 확보됐다.
따라서 다음 v216b-2는 인공 IF rule보다 이 OOF four-state를 이용해
validity/state/action head를 다시 학습하는 것이 우선이다.
```

핵심 산출물:

- `results/modular_benchmark/v216b_joint_gate_oof_states_cls10_train2500/oof_parent_predictions.csv`
- `results/modular_benchmark/v216b_joint_gate_oof_states_cls10_train2500/oof_fine_predictions.csv`
- `results/modular_benchmark/v216b_joint_gate_oof_states_cls10_train2500/oof_joint_states.csv`
- `results/modular_benchmark/v216b_joint_gate_oof_states_cls10_train2500/summary.json`

## v216b-pre3: OOF calibration before habit learning (2026-07-13)

주의: 이 실험은 이후 합의한 정식 v216b-2가 아니다. habit/normality 표현을
먼저 학습하지 않고 OOF state를 바로 action으로 calibration한 선행 진단이며,
정식 순서에서는 v216b-3의 실패 기준선으로만 사용한다.

입력:

```text
v216b-1 OOF 2500 Parent/Fine predictions
+ runtime-safe v210 membership
+ selected v131 Fine evidence
```

학습 action target:

| state | action |
|---|---|
| both_valid | accept_both |
| parent_only | repair_fine |
| fine_only | repair_parent |
| neither | reobserve_both |

OOF validation에서만 repair threshold를 정했으며 TEST threshold tuning은 하지 않았다.

### v216a 대비 판단 성능

| metric | v216a val500 | v216b OOF2500 |
|---|---:|---:|
| TEST Parent validity AUC | 0.742 | 0.934 |
| TEST Fine validity AUC | 0.886 | 0.898 |
| TEST state macro-F1 | 0.280 | 0.434 |
| TEST state accuracy | 86.1% | 88.7% |
| fine_only prediction count | 0 | 5 |

v216b OOF train audit:

```text
state macro-F1 = 0.700
action accuracy = 94.44%
policy fixed/broken = 45/1
Parent 97.08% -> 98.84%
```

TEST1000 최종 전환:

```text
base Parent = 97.00%
v216b policy = 96.90%
switch = 3
fixed = 1
broken = 2
net = -1
```

전환 3건:

```text
fixed:
chihuahua_10315: cat -> dog

broken:
horse_11349: horse -> dog_like
horse_11605: horse -> deer
```

Fine validity floor를 OOF validation에서 함께 보정하는 2D calibration도 1회 확인했다.
선택된 floor가 0.028로 낮아 TEST 결과는 변하지 않았다.

해석:

```text
성공:
- OOF 희귀 상태 증가로 fine_only를 실제 예측하기 시작함
- Parent validity와 joint-state 인식은 v216a보다 크게 개선됨
- TEST fine_only 2개 중 1개를 검출하고 실제 복구함

미달:
- Parent rejection이 매우 강하면 약한 Fine evidence도 repair_parent로 승인됨
- horse 2건은 Parent invalid + Fine weak 상황을 reobserve_both로 보내지 못함
- 병목은 더 이상 four-state 자료 부족이 아니라 repair와 reobserve의 분리임

다음 변경은 threshold 반복이 아니라 action target을 바꿔야 한다.
candidate Fine validity가 약한 fine_only/neither 유사 상태를
repair_parent가 아닌 reobserve_both hard-negative로 학습해야 한다.
```

산출물:

- `tools/build_joint_gate_oof_packets_v216b.py`
- `tools/train_eval_joint_gate_v216b.py`
- `results/modular_benchmark/v216b_oof_packets_cls10_train2500/joint_gate_oof_packets_v216b.npz`
- `results/modular_benchmark/v216b_joint_gate_oof2500_to_cls10_test1000/summary.json`

## v216b-2: Evidence masking + branch habit normality (2026-07-13)

목적:

```text
전체 TRAIN evidence에서 Parent/Fine branch가 정상적으로 판단할 때의
습관과 상황 구조를 self-supervised 방식으로 학습한다.

correct/wrong/state/fixed/broken target은 사용하지 않는다.
```

구현:

- `src/dual_line/decision/habit_normality_v216b2.py`
- `tools/train_habit_normality_contrastive_v216b2.py`

학습 구성:

```text
Parent input = Parent packet + Cross packet + Observation packet
Fine input   = Fine packet + Cross packet + Observation packet

mild view:
  evidence 10% masking + small Gaussian noise

heavy corruption:
  evidence 45% 교체/삭제
  다른 sample branch 혼합
  cross/observation 교란

loss:
  same-sample two-view InfoNCE
  masked evidence denoising reconstruction
  clean vs heavy branch normality
  aligned Parent/Fine vs mismatched branch joint normality
```

학습 입력에는 truth/state 필드가 없으며 OOF packet의 runtime-safe allow-list만 사용했다.

합성 교란 validation:

| detector | AUC |
|---|---:|
| Parent corruption | 0.965 |
| Fine corruption | 0.994 |
| Parent/Fine mismatch | 1.000 |

이 값은 실제 오답 분류 성능이 아니라 학습에 사용한 합성 교란 검출 성능이다.

### 실제 상태와의 사후 audit

아래 정답/state 비교는 학습 후 평가에만 사용했다.

| feature | OOF Parent invalid AUC | TEST Parent invalid AUC |
|---|---:|---:|
| habit_l2 | 0.975 | 0.965 |
| Parent reconstruction error | 0.976 | 0.959 |
| Fine reconstruction error | 0.956 | 0.942 |
| Fine corruption risk | 0.921 | 0.875 |

Neither 탐지:

| feature | OOF AUC | TEST AUC |
|---|---:|---:|
| habit_l2 | 0.913 | 0.960 |
| Parent reconstruction error | 0.935 | 0.958 |
| Fine reconstruction error | 0.912 | 0.939 |

TEST 상태별 평균 예시:

| state | Parent recon error | Fine recon error | habit L2 |
|---|---:|---:|---:|
| both_valid | 0.059 | 0.072 | 0.448 |
| parent_only | 0.289 | 0.379 | 0.877 |
| fine_only | 0.460 | 0.396 | 1.806 |
| neither | 0.938 | 1.183 | 1.055 |

해석:

```text
정답/오답을 가르치지 않고도 branch evidence의 정상 manifold와
Parent/Fine habit 거리 표현이 형성됐다.

특히 both_valid와 rare/error state의 reconstruction/habit gap이
OOF와 TEST에서 같은 방향으로 분리됐다.

따라서 v216b-3는 기존 packet raw feature를 다시 학습하기보다
이 v216b-2 embedding/normality feature를 고정하고,
OOF 실제 희귀 사례로 action calibration만 수행해야 한다.
```

산출물:

- `results/modular_benchmark/v216b2_habit_normality_train2500_to_test1000/habit_normality_v216b2.pt`
- `results/modular_benchmark/v216b2_habit_normality_train2500_to_test1000/train2500_habit_normality_v216b2.npz`
- `results/modular_benchmark/v216b2_habit_normality_train2500_to_test1000/cls10_test1000_habit_normality_v216b2.npz`
- `results/modular_benchmark/v216b2_habit_normality_train2500_to_test1000/summary.json`

## v216b-3: Frozen habit feature + real OOF action calibration (2026-07-13)

목적:

```text
v216b-2 encoder/normality 표현을 고정한다.
v216b-1의 실제 OOF four-state만 이용해 action을 calibration한다.
TEST feedback으로 threshold나 model을 수정하지 않는다.
```

구현:

- `tools/calibrate_habit_actions_v216b3.py`
- class-balanced multinomial logistic regression
- 입력: v216b-2 normality 7개 + embedding compact relation 5개 + branch disagreement
- 원본 OOF fold를 유지해 5-fold cross-fitted calibration prediction 생성
- repair/reobserve threshold는 cross-fitted OOF prediction에서만 결정

OOF cross-fitted 결과:

| metric | result |
|---|---:|
| state macro-F1 | 0.514 |
| fine_only AUC | 0.998 |
| neither AUC | 0.865 |
| fixed / broken | 46 / 5 |
| Parent accuracy | 97.08% -> 98.72% |

TEST1000 결과:

| metric | result |
|---|---:|
| state macro-F1 | 0.453 |
| both_valid AUC | 0.915 |
| parent_only AUC | 0.586 |
| fine_only AUC | 0.999 |
| neither AUC | 0.921 |
| switch | 6 |
| fixed / broken | 3 / 3 |
| Parent accuracy | 97.00% -> 97.00% |

TEST 전환:

```text
fixed:
chihuahua_10315: cat -> dog
wolf_10047: dog -> dog_like
wolf_10387: cat -> dog_like

broken:
deer_10800: deer -> dog_like
horse_11349: horse -> dog_like
horse_11605: horse -> deer
```

Reobserve audit:

```text
request count = 40
neither captured = 11/28
neither recall = 39.3%
neither precision = 27.5%
```

해석:

```text
v216b-2 normality 표현을 먼저 만든 뒤 calibration하자
premature action calibration의 1 fixed / 2 broken / net -1에서
3 fixed / 3 broken / net 0으로 개선됐고 baseline 97%를 유지했다.

fine_only 두 건을 모두 식별했고 neither 방향도 강한 AUC를 보였다.
그러나 parent_only AUC가 0.586으로 낮아 Parent가 맞는 deer/horse를
fine_only로 오인하는 방향성 문제가 남았다.

따라서 다음 병목은 anomaly 검출이 아니라 branch fault attribution이다.
추가 TEST threshold tuning은 금지한다. 다음 개선은 TRAIN 교란에서
Parent-only corruption과 Fine-only corruption을 명시적으로 분리하거나,
현재 reobserve 요청을 실제 새 관측으로 연결하는 방식이어야 한다.
```

산출물:

- `results/modular_benchmark/v216b3_habit_action_calibration_oof2500_to_test1000/habit_action_calibrator_v216b3.pkl`
- `results/modular_benchmark/v216b3_habit_action_calibration_oof2500_to_test1000/oof_calibration_predictions.csv`
- `results/modular_benchmark/v216b3_habit_action_calibration_oof2500_to_test1000/cls10_test1000/predictions.csv`
- `results/modular_benchmark/v216b3_habit_action_calibration_oof2500_to_test1000/summary.json`

## v216b-4: Synthetic branch fault attribution (2026-07-13)

목적:

```text
v216b-2 habit encoder를 고정하고 TRAIN evidence에서
normal / Parent-only corruption / Fine-only corruption / both corruption을 만든다.
어느 branch가 어긋났는지를 정답 라벨 없이 학습한다.
```

구현:

- `src/dual_line/decision/habit_fault_attribution_v216b4.py`
- `tools/train_branch_fault_attribution_v216b4.py`
- frozen v216b-2 embedding 위에 4-class attribution head만 학습
- 기존 OOF calibrator 조건과 C=0.25를 변경하지 않고 1회 재평가

합성 validation:

| metric | result |
|---|---:|
| four-way fault accuracy | 96.15% |
| normal correct | 480/500 |
| Parent fault correct | 473/500 |
| Fine fault correct | 482/500 |
| Both fault correct | 488/500 |

실제 TEST calibration 비교:

| metric | v216b-3 | v216b-4 |
|---|---:|---:|
| state macro-F1 | 0.453 | 0.489 |
| parent_only AUC | 0.586 | 0.592 |
| fine_only AUC | 0.999 | 0.999 |
| neither AUC | 0.921 | 0.916 |
| switches | 6 | 4 |
| fixed / broken | 3 / 3 | 2 / 2 |
| final Parent accuracy | 97.00% | 97.00% |

v216b-4 전환:

```text
fixed:
chihuahua_10315
wolf_10047

broken:
horse_11349
horse_11605
```

해석:

```text
합성 branch corruption은 잘 분류했지만 실제 parent_only 방향 AUC는 거의 오르지 않았다.
따라서 synthetic corruption direction과 실제 model fault direction은 동일하지 않다.

다만 과전환은 6 -> 4, broken은 3 -> 2로 줄었고 baseline 정확도는 유지했다.
즉 attribution feature는 보호 신호로 일부 작동했지만 최종 fault attribution은 아니다.

현 단계 결론:
- anomaly/state existence detection: 성공
- fine_only/neither ranking: 강함
- Parent-vs-Fine fault direction: 미완성
- automatic correction: baseline 유지, 순이득 0
- reobserve candidate generation: 사용 가능한 상태
```

TEST를 보고 별도 veto threshold를 추가하지 않는다. 다음 실험은 독립 validation에서
calibrator와 fault head가 충돌하면 switch 대신 reobserve하도록 학습하거나,
실제 재관측 결과를 action supervision으로 사용하는 방향이어야 한다.

## v217: Parent explains Fine (2026-07-13)

목적:

```text
parent_only 상태를 단순 anomaly로 찾는 대신,
Parent 증거가 각 Fine 후보와 전체 관측을 얼마나 설명하는지 학습한다.
```

구현:

- `tools/train_eval_parent_explains_fine_v217.py`
- TRAIN 2500의 5-fold OOF 후보 점수와 승인 threshold만 사용
- 후보별 object/wave/texture relation, 관측 다양성, Parent 안정성/소속 증거 사용
- candidate label, 정답, `prototype_parent_match`는 runtime feature에서 제외
- TEST 정답은 마지막 accuracy/fixed/broken audit에만 사용

초기 unrestricted 진단:

```text
OOF candidate AUC: 99.84%
TEST Fine: 94.6% -> 94.1%
fixed / broken: 5 / 10
```

깨진 10개 중 9개는 Persian/Siamese, 1개는 German Shepherd/Chihuahua였다.
Parent 증거는 cat/dog parent를 지지할 수 있지만 같은 parent 내부 sibling을
구분할 정보는 부족했다. 따라서 Parent 증거가 sibling 선택까지 개입하는 것은
그래프 역할을 넘는 것으로 판단했다.

최종 graph safety boundary:

```text
selected Fine parent != Parent prediction
인 cross-parent escape에만 Parent -> Fine 제약을 허용한다.

same-parent sibling reranking은 Fine evidence 전용 판단축에 맡긴다.
```

최종 TEST1000:

```text
base Fine exact:       94.60%
v217 Fine exact:       94.80%
Parent accuracy:       97.00%
base joint:            94.40%
v217 joint:            94.60%
switch:                4
fixed / broken:        2 / 0
parent_only recovered: 2 / 26
fine_only preserved:   2 / 2
```

복구:

```text
deer_10800:  fox -> deer
horse_11349: wolf -> horse
```

나머지 두 전환은 neither 상태의 wrong-to-wrong이었다. 따라서 v217은
cross-parent Fine escape를 안전하게 되돌리는 실행기로는 유효하지만,
same-parent `parent_only` 23개를 해결하지 못한다. 다음 판단축은 Parent를 더
강하게 넣는 것이 아니라 Fine sibling evidence의 반증/설명력을 학습해야 한다.
또한 neither normality risk는 v217 실행을 차단하고 reobserve로 보내야 한다.

산출물:

- `results/modular_benchmark/v217_parent_explains_fine_oof2500_to_test1000/summary.json`
- `results/modular_benchmark/v217_parent_explains_fine_oof2500_to_test1000/eval_predictions.csv`
- `results/modular_benchmark/v217_parent_explains_fine_oof2500_to_test1000/parent_explains_fine_v217.pkl`

## v218: Pairwise Fine sibling explanation (2026-07-13)

목적:

```text
same-parent parent_only와 both_valid를 구분한다.
Parent는 sibling 범위만 정하고 sibling 선택에는 개입하지 않는다.
```

구현:

- `tools/train_eval_sibling_explanation_v218.py`
- 현재 Fine 후보와 대안 sibling 후보의 56개 evidence delta를 비교
- object/wave/texture relation, 관측 다양성, 중복/국소 위험의 상대 차이 사용
- 순서를 뒤집은 pair를 함께 학습해 특정 candidate 이름을 feature로 사용하지 않음
- 5-fold OOF TRAIN에서 모델과 승인 threshold를 결정
- TEST feedback 재학습/threshold 조정 없음

결과:

```text
                         OOF TRAIN    TEST1000
same-parent parent_only AUC   95.45%      70.08%
fixed / broken                 2 / 1       4 / 20
Fine exact                                94.6% -> 93.0%
joint                                     94.4% -> 92.8%
```

복구 4개:

```text
Chihuahua -> German Shepherd: 2
Persian -> Siamese:            2
```

broken 20개의 주요 방향:

```text
Persian -> Siamese:            7
Siamese -> Persian:            4
German Shepherd -> Chihuahua:  4
Fox -> Wolf:                    3
Lion -> Tiger:                  1
Chihuahua -> German Shepherd:  1
```

판단:

```text
기존 전체 parent_only AUC 약 0.59와 비교하면,
same-parent 전용 pairwise evidence는 TEST AUC 0.70으로 감지 신호를 개선했다.

하지만 OOF threshold 0.694가 TEST에서 일반화되지 않았고,
관계 신호의 절대 scale과 방향에 강한 sibling/domain 편향이 남았다.
```

따라서 v218은 실행기로 채택하지 않는다. 현재 운영 기준은 v217의
cross-parent fixed 2 / broken 0을 유지한다. v218의 의미는 same-parent 약점에
사용 가능한 힌트가 실제로 존재함을 확인한 것이며, 다음에는 전환보다
reobserve/ranking audit feature로 먼저 사용해야 한다.

산출물:

- `results/modular_benchmark/v218_sibling_explanation_oof2500_to_test1000/summary.json`
- `results/modular_benchmark/v218_sibling_explanation_oof2500_to_test1000/eval_predictions.csv`
- `results/modular_benchmark/v218_sibling_explanation_oof2500_to_test1000/sibling_explanation_v218.pkl`

## v219: Same-parent Fine risk detector (2026-07-13)

목적:

```text
수정하지 않고 same-parent parent_only만 감지한다.
현재 Fine 판단이 view/관측 변화에도 안정적인지 risk로 출력한다.
```

구현:

- `tools/train_eval_same_parent_detector_v219.py`
- v218 pairwise evidence delta
- 이미지별 44개 bbox view의 sibling probability/margin 안정성
- small/medium/large area, upper/lower/left/right/full-width 독립 관측 일치도
- v216b-2 habit normality feature
- 후보 이름은 feature에서 제외
- 5-fold OOF TRAIN 학습, TEST 수정/전환 없음

결과:

```text
                         OOF TRAIN    TEST1000
ROC-AUC                    95.17%       43.97%
PR-AUC                     31.53%        2.87%
review 5% parent_only recall 66.67%       4.35% (1/23)
```

실패 원인:

```text
후보 이름을 직접 입력하지 않았지만,
prototype 거리와 bbox view probability의 절대 scale이
사실상 class identity shortcut으로 작동했다.
```

TEST 평균 risk 예:

```text
Chihuahua: 0.13
Persian:   0.95
Siamese:   0.93
Lion:      0.92
Tiger:     0.98
```

따라서 모델은 "현재 Fine이 틀릴 습관"보다 "이 클래스는 원래 이런 점수
분포를 가진다"를 학습했다. OOF에서는 같은 분포라 잘 보였지만 TEST에서
클래스별 관측 분포가 변하자 risk ordering이 무너졌다.

판단:

```text
v219는 감지기로 채택하지 않는다.
v218의 상대 evidence AUC 0.70보다도 일반화가 나빠졌다.

다음 감지기는 절대 view/prototype 값을 직접 사용하면 안 된다.
자동 발견 sibling node별 정상 habit 기준으로 정규화한 residual이나,
동일 이미지 안에서의 rank/flip만 사용해야 한다.
```

산출물:

- `results/modular_benchmark/v219_same_parent_detector_oof2500_to_test1000/summary.json`
- `results/modular_benchmark/v219_same_parent_detector_oof2500_to_test1000/eval_risk.csv`
- `results/modular_benchmark/v219_same_parent_detector_oof2500_to_test1000/same_parent_detector_v219.pkl`

## v220: Automatic node-relative Fine risk (2026-07-14)

목적:

```text
v219의 class-scale shortcut을 제거한다.
자동 sibling node별 정상 OOF habit의 median/MAD 좌표계에서만 risk를 학습한다.
```

구현:

- `tools/train_eval_node_relative_detector_v220.py`
- 수동 Parent map 미사용
- 자동 multi-child node:
  - cluster_001 = fox/wolf
  - cluster_002 = lion/tiger
  - cluster_003 = persian/siamese
- 각 OOF fold의 TRAIN 정상 Fine만으로 node median/MAD profile 생성
- detector에는 절대 evidence를 전달하지 않고 signed/absolute residual만 전달
- 예측 수정 없음

결과:

```text
                         OOF TRAIN    TEST1000
전체 ROC-AUC               93.20%       47.26%
전체 PR-AUC                27.09%        2.65%
review 5% recall           64.29%        0.00%
```

TEST node별:

```text
cluster_001 fox/wolf:         AUC 31.38%, internal wrong 4
cluster_002 lion/tiger:       internal wrong 0
cluster_003 persian/siamese:  AUC 73.72%, internal wrong 13
```

해석:

```text
node-relative residual은 Persian/Siamese 내부에서는 실제 감지 신호를 개선했다.
하지만 하나의 global detector probability로 서로 다른 node를 다시 합치자
Fox/Wolf 방향이 역전되고 global review ranking이 무너졌다.
```

따라서 class 절대값 shortcut은 제거했지만 node별 오류 방향/발생률 차이는
남아 있다. 다음 구조는 하나의 global head가 아니라 자동 node별 local head
또는 node-local risk percentile이어야 한다. Lion/Tiger처럼 TRAIN/TEST 오류가
없는 node는 supervised head를 억지로 만들지 않고 one-class normality만 유지해야 한다.

산출물:

- `results/modular_benchmark/v220_auto_node_relative_detector_oof2500_to_test1000/summary.json`
- `results/modular_benchmark/v220_auto_node_relative_detector_oof2500_to_test1000/eval_risk.csv`
- `results/modular_benchmark/v220_auto_node_relative_detector_oof2500_to_test1000/node_relative_detector_v220.pkl`

## v221/v221b: 정답 비사용 masked tile habit 학습

목표는 오답 분류가 아니라, 16개 타일 중 일부를 숨기고 나머지 관측으로 숨긴 타일의 질감/파형 증거를 복원하는 것이었다. TRAIN 정답, 오답 상태, 클래스 target은 학습에 사용하지 않았고 TEST 정답은 학습 종료 후 상태별 AUC 감사에만 사용했다.

```text
TRAIN 2,500 / TEST 1,000
v221: 타일 self token + 관계 평균 요약
v221b: 타일 self evidence를 가리고, 주변 타일의 16x16 방향성 관계 context로 복원

v221b node dim             545
v221b relation context    1090
masked reconstruction loss 0.60684
```

v221b 결과:

| 감사 상태 | 수 | observation risk AUC | fine-fault risk AUC |
|---|---:|---:|---:|
| both valid | 929 | 0.500 | 0.556 |
| Parent only (Parent 정답/Fine 오답) | 41 | 0.476 | 0.484 |
| Fine only (Parent 오답/Fine 정답) | 16 | 0.584 | 0.481 |
| neither | 14 | 0.474 | 0.300 |

추가 단일 지표 감사에서 Parent-only의 최고 비누수 신호는 `fine_restore_flip_rate` AUC 0.574였고, Fine-only는 `reconstruction_mse_max` AUC 0.690, `reconstruction_mse_mean` AUC 0.638이었다.

해석:

```text
타일 관계를 이용한 증거 복원 자체는 학습된다.
하지만 Parent 정답/Fine 오답은 관측 불안정으로 설명되지 않는다.
잘못된 Fine shortcut도 여러 타일에서 일관되게 반복되어 복원 가능할 수 있다.
따라서 reconstruction error 단독으로 Fine-only fault gate를 만들 수 없다.
현재 masked-tile 모듈은 자동 수정기가 아니라 관측 predictability 증거 모듈로만 유지한다.
```

이 실험은 TEST feedback으로 threshold나 모델을 재학습하지 않았다. v221에서 실제 방향성 관계가 빠진 구현 불일치만 v221b에서 바로잡았으며, 결과를 보고 추가 스윕하지 않았다.

## v222: structured CNN/relation reconstruction and 3-state fault count

목표는 정답을 복원하는 것이 아니라 structured intervention 이후에도 CNN, 관계, Parent, Fine evidence가 재현되는지를 학습하는 것이다. 최종 calibration target은 both_correct, one_wrong, both_wrong 세 상태다.

입력과 실행 구조:

- CNN tile embedding 512
- tile texture/class evidence
- 16x16 CNN texture relation
- 16x16 wave relation
- 30 structured probes
- tile/relation/Parent/Fine evidence reconstruction
- branch-symmetric response profile
- 3-state OOF calibration

30개 probe는 단일 타일 16개, quadrant/center/border/half-plane 10개, CNN norm/wave edge/Parent support/Fine support top-4 4개로 구성했다. Probe 중복률은 TEST 평균 1.8%, mask 평균 IoU는 0.072였다.

누수 방지:

- 각 OOF fold reconstructor는 해당 fold 이미지를 학습에 사용하지 않음
- probe row가 아니라 original image 단위 fold
- reconstructor는 correctness/state label을 사용하지 않음
- 3-state calibrator만 OOF state label 사용
- TEST threshold 또는 TEST feedback 사용 안 함

TEST1000 AUC:

| feature group | both correct | one wrong | both wrong |
|---|---:|---:|---:|
| shared reconstruction only | 0.749 | **0.751** | 0.624 |
| branch symmetric only | **0.892** | 0.724 | **0.699** |
| naive all concat | 0.907 | 0.741 | 0.604 |

OOF 기준으로는 branch-symmetric-only가 주 결과다. all 단순 결합은 TEST both-wrong을 0.604로 낮췄으므로 shared-object와 branch-response 축을 무조건 한 선형 head에 합치는 것은 적합하지 않다.

v221b의 고정 수식 기반 3-state TEST AUC는 약 0.579/0.594/0.439였다. v222 structured reconstruction은 특히 one-wrong과 both-wrong 분리를 실제로 높였다. CNN backbone 정보와 관계 복원이 단순 reconstruction loss보다 branch response profile에서 가치가 있음을 확인했다.

해석:

- shared reconstruction은 one-wrong 탐지에 가장 강함
- branch-symmetric response는 both-correct와 both-wrong 분리에 더 강함
- 두 축은 독립 판단축으로 유지하고 단순 feature concat 대신 상태별 결합 구조를 검토

## v223 - v222 reconstruction reobserve planner (2026-07-14)

### Goal

- Use v222 as a reobservation planner rather than a standalone fault classifier.
- Preserve all 30 probe-level reconstruction outputs before sample-level aggregation.
- Rank where to observe again without using TEST truth at runtime.

### Structure

```text
v222 reconstructor
  -> probe-level node / relation reconstruction error
  -> Parent / Fine reconstruction error and impact
  -> TRAIN-only robust reference normalization
  -> top-K reobserve plan
```

Two fixed, train-only policies were evaluated once:

1. Generic top-3: rank a single combined reobserve score.
2. Complementary top-3: select one shared-flow, one Parent-check, and one Fine-check probe.

TEST labels were used only for the final audit. They were not inputs to score generation or action selection.

### TEST1000 result

| Metric | Generic | Complementary |
|---|---:|---:|
| one-wrong detection AUC | 0.803 | 0.798 |
| both-wrong detection AUC | 0.881 | 0.873 |
| any-wrong detection AUC | 0.852 | 0.845 |
| one-wrong top-3 gain over random | -0.022 | +0.038 |
| both-wrong top-3 gain over random | +0.196 | -0.130 |

### Interpretation

- Reconstruction signals are useful for detecting that reobservation is needed.
- They do not yet identify a consistently useful location. A sample-wide anomaly raises many probe scores together, so top probes are often arbitrary.
- Explicitly forcing shared/Parent/Fine diversity did not solve localization. This experiment should not be tuned repeatedly on TEST.
- v216b-3 remains the stronger both-wrong detector. v222/v223 should remain a reobservation module, not replace that detector.

### Required next evidence

The next planner must execute the proposed crop/tile action and learn from an observable runtime reward:

```text
before/after Parent-Fine disagreement
before/after candidate margin
same-object support change
fixed / broken / unchanged
```

Until that action-outcome pair exists, v223 is a request detector and action proposal exporter, not a validated recovery policy.

Artifacts:

- `tools/build_reobserve_plan_v223.py`
- `results/modular_benchmark/v223_reobserve_planner_v222_train2500_to_test1000/summary.json`
- `results/modular_benchmark/v223_reobserve_planner_v222_train2500_to_test1000/reobserve_plan_v223.csv`
- `results/modular_benchmark/v223_reobserve_planner_v222_train2500_to_test1000/complementary_reobserve_plan_v223.csv`
- `results/modular_benchmark/v223_reobserve_planner_v222_train2500_to_test1000/probe_profiles_v223.npz`

## v224 - Counterfactual evidence restoration gate (2026-07-14)

### Goal

Use the v222 reconstruction objective as an evidence restoration layer rather than a
location selector. The habit gate compares raw and restored Parent/Fine states and
approves restoration only when TRAIN evidence predicts a benefit.

```text
raw tile evidence
  -> leave-one-tile-out v222 restoration
  -> raw / restored evidence kept independently
  -> v216 habit features + restoration deltas
  -> state detector and branch-specific approval gates
```

Strictness:

- v222 restoration features for TRAIN were generated with five-fold OOF reconstructors.
- TEST truth was used only for the final audit.
- Approval thresholds were selected on OOF TRAIN with `fixed - 2 * broken`.
- No TEST threshold tuning was performed.

### Restoration evidence

| Fine evidence source | TEST accuracy |
|---|---:|
| Existing v150 Fine | 94.5% |
| Raw tile probability mean | 78.6% |
| Reconstructed tile probability mean | 84.1% |
| Existing Fine OR reconstructed candidate oracle | 96.9% |

The reconstruction itself adds real signal: it raises the weak tile-mean classifier by
5.5 percentage points and recovers 24 of the 55 existing Fine errors. It is not strong
enough to replace the existing Fine selector directly.

### State detection

| State | TEST AUC |
|---|---:|
| both valid | 0.915 |
| Parent only | 0.653 |
| Fine only | 0.987 |
| neither | 0.708 |

The restored comparison remains strong for Fine-only detection, but is weaker than
v216b-3 for neither/both-wrong detection.

### Approval result

OOF TRAIN calibration:

```text
Parent restoration: fixed 12 / broken 3
Fine restoration:   fixed 1 / broken 0
```

Frozen TEST policy:

```text
Parent restoration: approved 1, fixed 0 / broken 1
Fine restoration:   approved 0, fixed 0 / broken 0
Parent accuracy: 97.0% -> 96.9%
Fine accuracy:   94.5% -> 94.5%
```

### Interpretation

- v222 can reconstruct class-related evidence, but the current restoration candidate is
  derived from tile probabilities while the strong Fine answer comes from the v150
  multi-view selector. Their score spaces are not aligned.
- The synthetic degradation trust model was extremely conservative and should not be
  treated as a successful runtime fusion rule.
- The correct next integration is to feed restored tile/relationship evidence back into
  the existing v150 candidate/evidence builder, then let the same selector rescore it.
- The current Parent packet contains only five scalar state features, not a complete
  Parent class-score vector. A true Parent replay requires exporting that vector.
- Do not continue tuning v224 thresholds on TEST. The missing interface is more important
  than another gate sweep.

Artifacts:

- `tools/train_eval_counterfactual_restoration_gate_v224.py`
- `results/modular_benchmark/v224_counterfactual_restoration_gate_oof2500_to_test1000/summary.json`
- `results/modular_benchmark/v224_counterfactual_restoration_gate_oof2500_to_test1000/predictions_v224.csv`
- `results/modular_benchmark/v224_counterfactual_restoration_gate_oof2500_to_test1000/restoration_profiles_v224.npz`
- `results/modular_benchmark/v224_counterfactual_restoration_gate_oof2500_to_test1000/counterfactual_restoration_gate_v224.pkl`

## v225 - Relation-consistent restoration cutoff and DINOv2 backbone pivot (2026-07-14)

### Goal

Test the reconstruction hypothesis once with the missing structural constraints, then
switch to a DINO backbone if the frozen criteria are not met.

```text
Level 0: clean no-op
Level 1: one tile feature attenuation
Level 2: one tile removal
Level 3: adjacent two-tile removal
```

Unlike v222, each corruption also removes or attenuates the corresponding rows and
columns of the explicit 16x16 relation tensor. The model predicts residual evidence and
uncertainty; correctness labels are not used for reconstruction training.

### v225 restoration result

The valid run used 2,500 TRAIN images and one frozen TEST1000 audit.

| Metric | Result |
|---|---:|
| Synthetic corrupted-evidence recovery | 17.2% |
| Clean no-op drift | 0.0191 |
| Raw tile-mean Fine | 78.6% |
| Restored tile-mean Fine | 75.6% |
| Existing Fine OR restored oracle | 96.3% |
| Restored candidate fixes | 18 |
| Breaks if always applied | 207 |

The frozen pass criteria were recovery >=20%, clean drift <=0.03, restored Fine >=86%,
and oracle >=96.9%. Only clean preservation passed. Explicit relation masking removed the
v222 shortcut, but the remaining evidence was not sufficient to reconstruct discriminative
Fine evidence. No TEST threshold tuning was performed, and the restoration direction was
stopped here.

Artifacts:

- `tools/train_eval_relation_consistent_restorer_v225.py`
- `results/modular_benchmark/v225_relation_consistent_restorer_train2500_to_test1000/summary.json`
- `results/modular_benchmark/v225_relation_consistent_restorer_train2500_to_test1000/relation_restorer_v225.pt`

### DINOv2 pivot

The modular backbone contract was extended with the official frozen
`facebookresearch/dinov2` `dinov2_vits14` model. The input stays 224x224 with ImageNet
normalization, and the representation is the 384-dimensional normalized CLS token.

First comparison intentionally used the same simple full-image texture head, without ROI,
wave, relation, candidate, or gate assistance.

| Model | Frozen embedding | TEST1000 Fine accuracy | Macro F1 |
|---|---:|---:|---:|
| ResNet18 full baseline | 512 | 94.1% | 94.1% |
| ResNet50 full baseline | 2048 | 97.4% | 97.4% |
| DINOv2 ViT-S/14 full baseline, matched ResNet50 protocol | 384 | **98.1%** | **98.1%** |
| manual v153 Fine reference | mixed evidence | 96.6-96.8% | - |

Transition audit against ResNet50:

```text
fixed 16
broken 9
net +7
```

Per-class DINOv2 recall:

```text
persian 93, siamese 96, chihuahua 100, german shepherd 98,
wolf 100, fox 96, lion 100, tiger 100, horse 98, deer 100
```

The matched protocol is `10 epochs`, `batch_size=64`, `seed=220`, the same as the
ResNet50 baseline. An earlier exploratory DINO head using 20 epochs, batch 128, and seed
41 scored 97.9%; it is not used as the final backbone comparison.

Interpretation: the DINOv2 representation is already stronger than the ResNet50 baseline
and the old Fine pipeline before adding Dual-Line observation logic. The next justified
experiment is not a larger reconstruction sweep. It is to generate DINO tile/ROI/bbox
evidence through the modular branches and measure whether v153-style candidate/gate logic
adds recovery without erasing the 97.9% base.

### Plain end-to-end DINO transfer baseline

To separate feature extraction from conventional backbone fine-tuning, a second baseline
was trained directly from the raw cls10 image folders. It does not import or use any
Dual-Line cache, ROI, tile, wave, candidate, Parent/Fine, or gate component.

```text
pretrained DINOv2-S/14
  -> full backbone fine-tuning (lr 1e-5)
  -> linear 10-class head (lr 1e-3)
  -> TRAIN 2000 / validation 500
  -> one frozen TEST1000 audit
```

| Plain DINO transfer mode | TEST1000 |
|---|---:|
| Frozen DINO + trained MLP head | 98.1% |
| End-to-end DINO fine-tuning + linear head | 97.0% |

The end-to-end result is lower despite reaching 100% TRAIN-like optimization behavior and
96.8% best validation accuracy. With only 2,000 optimization images, updating the full
backbone appears to reduce some pretrained generality. The stronger plain baseline for this
dataset is therefore frozen DINO plus a learned task head, not full fine-tuning.

Artifact:

- `tools/train_eval_dinov2_transfer_baseline.py`
- `results/baseline_plain_finetuned_dinov2_vits14_awa_cls10_seed220/summary.json`

Artifacts:

- `src/dual_line/backbone_adapters.py`
- `results/baseline_full_dinov2_vits14_awa_cls10_seed220/full_texture_head.pt`
- `results/eval_baseline_full_dinov2_vits14_awa_cls10_test_seed220/metrics.json`
- `results/texture_cache_modular_awa_cls10_train_dinov2_vits14_full/`
- `results/texture_cache_modular_awa_cls10_test_dinov2_vits14_full/`

## 2026-07-15 - v226a DINO Parent/Fine evidence projector

목적:

```text
DINO CLS의 단일 정답만 사용하는 대신,
Parent 공통 정체성 증거와 Fine 구별 증거를 독립 head로 학습하고
각 판단이 4x4 공간 어디에서 형성됐는지 Gate 입력으로 보존한다.
```

구조:

```text
Frozen DINOv2-S/14
  -> CLS [B,384]
  -> observer-aligned tile tokens [B,16,384]
  -> independent Parent/Fine CLS heads
  -> independent Parent/Fine tile MIL heads
  -> masking consistency + evidence concentration control
```

TRAIN 2,000 / validation 500만 모델 선택에 사용했다. TEST1000은 마지막 고정
감사에만 사용했으며 Gate, 전환, TEST threshold sweep은 사용하지 않았다.

| v226a result | TRAIN | validation | TEST1000 |
|---|---:|---:|---:|
| Parent accuracy | 100.0% | 100.0% | 99.6% |
| Fine accuracy | 99.9% | 97.6% | 98.4% |
| Fine tile-only accuracy | 99.5% | 95.8% | 97.4% |

TEST state audit:

```text
both_valid   981
parent_only   15
fine_only      3
neither        1
```

공식 frozen DINO MLP baseline 98.1%와의 sample 비교:

```text
fixed   7
broken  4
net    +3
```

해석:

1. v226a는 최종 Gate가 아니라 Parent/Fine 공간 evidence provider다.
2. 기존에 약했던 `Parent 정답 / Fine 오답` 상태가 실제 TEST에서 15장 형성됐다.
3. Fine tile-only 97.4%는 16개 공간 token 자체에도 강한 구별 정보가 있음을 보인다.
4. 다음 v226b는 TEST 조정 없이 OOF/validation evidence로 4-state detector를 학습한다.

Artifacts:

- `src/dual_line/representation/dino_concept_evidence.py`
- `src/dual_line/representation/dino_observer_relation.py`
- `tools/build_dino_spatial_cache_v226.py`
- `tools/train_eval_dino_concept_evidence_v226.py`
- `results/dino_spatial_cache_v226_cls10_train/dino_spatial_cache_v226.npz`
- `results/dino_spatial_cache_v226_cls10_test/dino_spatial_cache_v226.npz`
- `results/v226a_dino_concept_evidence_cls10_seed226/summary.json`

## 2026-07-15 - v226b OOF DINO four-state detector

목적:

```text
v226a Parent/Fine evidence를 이용해
both_valid / parent_only / fine_only / neither 상태를 감지한다.
아직 예측 전환이나 재관측 실행은 하지 않는다.
```

TRAIN 2,500장은 5-fold OOF projector로 다시 예측했다. 각 held-out sample에 대해
16개 타일을 하나씩 제거한 counterfactual flip/drop 특징을 만들고, 기존 v05 wave와
DINO semantic relation의 agreement/conflict 17개를 결합했다.

OOF 상태 수집:

```text
both_valid   2443
parent_only    46
fine_only       5
neither         6
```

Gate 구조는 Parent/Fine validity를 독립적으로 출력한 뒤 두 결과를 조합한다. TEST는
threshold 또는 model selection에 사용하지 않았다.

| TEST1000 detector metric | result |
|---|---:|
| Parent validity AUC | 0.9997 |
| Fine validity AUC | 0.9665 |
| Any-invalid recall | 78.95% (15/19) |
| Both-valid false review | 4.49% (44/981) |
| Total review rate | 5.90% (59/1000) |
| Exact state accuracy | 94.60% |

상태별 TEST 결과:

```text
parent_only 15:
  exact parent_only 8
  anomaly로는 감지 11

fine_only 3:
  anomaly로는 3개 모두 감지
  exact fine_only 귀속 0 (모두 neither)

neither 1:
  exact neither 1
```

해석:

1. DINO 공간 evidence + 타일 counterfactual + wave agreement는 Fine validity 이상을
   강하게 순위화한다.
2. 기존 병목이던 parent_only는 완전 해결은 아니지만 15개 중 11개를 review 대상으로
   올렸고 8개는 정확히 귀속했다.
3. fine_only는 오류 감지는 되지만 어느 branch가 틀렸는지 귀속하지 못했다.
4. 정상 false review 44개가 있으므로 아직 action/switch Gate로 사용하면 안 된다.
5. 다음 단계는 OOF validation에서만 review threshold를 보정하고, TEST 고정 상태에서
   정상 보호율과 parent_only recall의 tradeoff를 확인하는 것이다.

Artifacts:

- `tools/train_eval_dino_state_gate_v226b.py`
- `results/v226b_dino_oof_state_gate_cls10_seed226/dino_state_gate_v226b.pt`
- `results/v226b_dino_oof_state_gate_cls10_seed226/oof_state_features.npz`
- `results/v226b_dino_oof_state_gate_cls10_seed226/summary.json`

## 2026-07-15 - v226c OOF-only review threshold calibration

목적:

```text
v226b Gate와 projector는 고정한다.
OOF fold 0에서만 Parent/Fine validity threshold를 선택한다.
TEST는 고정 threshold의 최종 감사에만 사용한다.
예측 전환은 하지 않고 review 요청 품질만 측정한다.
```

비교 결과:

| operating point | TEST review | invalid captured | false review | review precision |
|---|---:|---:|---:|---:|
| v226b default 0.5 | 59/1000 (5.90%) | 15/19 (78.95%) | 44/981 (4.49%) | 25.42% |
| v226c safe | 15/1000 (1.50%) | 10/19 (52.63%) | 5/981 (0.51%) | 66.67% |
| v226c balanced | 15/1000 (1.50%) | 10/19 (52.63%) | 5/981 (0.51%) | 66.67% |

상태별 주요 결과:

```text
safe:
  parent_only review 6/15
  fine_only review 3/3
  neither review 1/1

balanced:
  parent_only review 6/15
  fine_only review 3/3
  neither review 1/1
```

해석:

1. OOF-only calibration으로 정상 false review를 44개에서 5개로 크게 줄였다.
2. 그 대가로 any-invalid recall은 15/19에서 10/19로 감소했다.
3. safe와 balanced는 다른 threshold를 선택했지만 TEST review 집합은 동일했다.
4. fine_only 3개는 모두 anomaly로 잡지만 정확한 branch 귀속은 여전히 실패한다.
5. calibration fold의 invalid sample이 13개뿐이므로 threshold 일반화는 아직 불안정하다.
6. 현재 권장 사용은 자동 전환이 아니라 1.5% 규모의 고정밀 reobserve 요청이다.

Artifacts:

- `tools/calibrate_dino_state_gate_v226c.py`
- `results/v226c_dino_state_gate_calibration_cls10_seed226/summary.json`
- `results/v226c_dino_state_gate_calibration_cls10_seed226/test_calibrated_state_scores.npz`

## 2026-07-15 - v227a DINO-guided reobservation audit

목적:

```text
v226b default 0.5가 review한 59장을 실제로 다시 본다.
정상 review 44장도 fragile-correct일 수 있으므로 무조건 오탐으로 버리지 않는다.
정답은 view 생성이나 선택에 사용하지 않고 마지막 감사에서만 사용한다.
```

각 sample의 기존 Parent/Fine attention으로 다음 네 crop을 생성했다.

```text
parent_focus
fine_focus
joint_focus
wide_context
```

네 crop을 DINO와 고정 v226a projector로 다시 읽고 평균 확률 ensemble과 view
consensus를 계산했다. Gate/projector 재학습과 실제 prediction switching은 하지 않았다.

| metric | result |
|---|---:|
| review | 59/1000 (5.9%) |
| Parent consensus mean | 98.31% |
| Fine consensus mean | 96.19% |
| Parent fixed / broken | 2 / 1 |
| Fine fixed / broken | 3 / 0 |
| 정상 review both 보존 | 43/44 |

상태별:

```text
parent_only 11:
  ensemble Fine 복구 3
  candidate oracle Fine 복구 6

fine_only 3:
  ensemble Parent 복구 2
  candidate oracle Parent 복구 2

neither 1:
  네 candidate 모두 복구 실패
```

59장에 ensemble을 적용한다고 가정한 audit-only 전체 수치:

```text
Parent 99.6% -> 99.7%
Fine   98.4% -> 98.7%
```

해석:

1. v226b의 정상 review 44장은 단순 낭비가 아니다. 재관측 후 43장이 양쪽 정답을
   유지했고 모든 sample에서 적어도 하나의 both-correct view가 존재했다.
2. label-free attention crop과 단순 ensemble만으로 Fine 3개를 broken 없이 복구했다.
3. parent_only의 후보 oracle은 6개인데 ensemble 복구는 3개이므로 다음 병목은 view
   생성보다 view 선택/승인이다.
4. wide_context는 네 family 중 가장 약했다. 현재 review 집합에서는 국소 및 joint
   focus가 더 유효하다.
5. 아직 자동 전환 성능으로 확정하지 않는다. 다음 단계는 TRAIN/OOF에서 재관측
   evidence 승격 조건을 학습하고 TEST에 고정 적용하는 것이다.

Artifacts:

- `tools/eval_dino_guided_reobserve_v227.py`
- `results/v227a_dino_guided_reobserve_default05_cls10/summary.json`
- `results/v227a_dino_guided_reobserve_default05_cls10/reobserve_views.npz`
- `results/v227a_dino_guided_reobserve_default05_cls10/reobserve_audit.csv`

## 2026-07-15 - v227b expanded DINO reobservation candidates

목적:

```text
v227a의 후보 4개를 attention 기반 최대 10개로 확장한다.
후보 생성은 공격적으로 하되 실제 전환은 계속 비활성화한다.
동일 bbox는 중복 투표에서 제거한다.
```

추가 관측축:

```text
Parent/Fine tight 및 context
Parent/Fine disagreement
joint tight 및 context
첫 attention 주변을 제거한 secondary joint
wide context
```

실제 unique 후보 수는 sample당 6~10개, 평균 8.19개였다.

| metric | v227a 4-view | v227b expanded |
|---|---:|---:|
| any Parent-correct view | 57/59 | 58/59 |
| any Fine-correct view | 53/59 | 55/59 |
| any both-correct view | 52/59 | 54/59 |
| Parent ensemble fixed/broken | 2/1 | 1/0 |
| Fine ensemble fixed/broken | 3/0 | 2/1 |
| 정상 review both 보존 | 43/44 | 43/44 |

상태별 candidate oracle 변화:

```text
parent_only Fine 복구: 6/11 -> 7/11
fine_only Parent 복구: 2/3 -> 2/3
neither both 복구:     0/1 -> 1/1
```

구체적인 새 증거:

```text
fox_fox_10243:
  base neither
  compact 후보 모두 실패
  expanded secondary_joint에서 dog_like / fox 정답 생성

fox_fox_10441:
  base parent_only
  expanded secondary_joint에서 Fine fox 정답 생성
```

해석:

1. 후보 확장은 oracle을 확실히 올렸고, 특히 기존 neither 샘플도 복구 가능한 관측을
   처음 만들었다.
2. 반면 모든 후보를 동등 평균한 ensemble은 Fine에서 2 fixed / 1 broken으로 v227a의
   3 / 0보다 나빠졌다.
3. 따라서 공격적 후보 생성은 유지하되 단순 confidence/평균 투표는 사용하면 안 된다.
4. secondary evidence가 실제 복구를 만들었으므로 다음 단계는 독립 증거 다양성,
   view family, 기존 branch 보존을 보는 TRAIN/OOF 승격 Gate다.

Artifacts:

- `tools/eval_dino_guided_reobserve_v227.py --view_policy expanded`
- `results/v227b_dino_guided_reobserve_expanded_cls10/summary.json`
- `results/v227b_dino_guided_reobserve_expanded_cls10/reobserve_views.npz`
- `results/v227b_dino_guided_reobserve_expanded_cls10/reobserve_audit.csv`

## 2026-07-15 - v228 OOF reobservation promotion Gate

목적:

```text
expanded 후보는 공격적으로 생성한다.
모든 후보를 평균내지 않고 기존 branch를 실제로 고치는 후보만 승격한다.
승격 모델과 threshold는 TRAIN OOF에서만 학습/선택한다.
TEST는 고정 모델의 최종 감사에만 사용한다.
```

TRAIN 구성:

```text
OOF review samples: 149/2500
expanded candidate rows: 1228
Parent benefit-positive rows: 52
Fine benefit-positive rows: 118

class 이름/identity feature: 사용 안 함
feature:
  기존/후보 confidence, margin, entropy
  확률 분포 거리와 변화량
  Parent/Fine agreement
  기존 validity
  bbox geometry
  view family
```

Promotion model도 기존 projector fold를 따라 cross-validation prediction을 만들었고,
그 OOF score에서 broken penalty 2.0으로 threshold를 선택했다.

| policy | OOF fixed/broken | TEST fixed/broken |
|---|---:|---:|
| Parent promotion | 2/0 | 0/0 |
| Fine promotion | 13/1 | **3/0** |

최종 TEST1000:

| metric | base | v228 |
|---|---:|---:|
| Parent accuracy | 99.6% | 99.6% |
| Fine accuracy | 98.4% | **98.7%** |
| Parent+Fine both accuracy | 98.1% | **98.4%** |
| Parent/Fine graph conflict | 5 | 5 |

승인된 세 Fine 전환:

```text
chihuahua_10045:          german+shepherd -> chihuahua
german+shepherd_10239:    chihuahua -> german+shepherd
persian+cat_10030:        siamese+cat -> persian+cat
```

세 전환은 모두 `parent_disagreement` view에서 나왔고 모두 정답이었다.

해석:

1. v227b의 확장 후보는 oracle을 올렸지만 단순 평균은 2 fixed / 1 broken이었다.
2. v228은 동일 후보에서 TRAIN OOF로 안전한 세 후보만 골라 3 / 0을 만들었다.
3. Parent threshold는 매우 보수적으로 선택되어 TEST 전환을 승인하지 않았다. 이는
   Parent positive 사례가 적은 현재 조건에서 적절한 동작이다.
4. TEST threshold 조정 없이 실제 prediction switching을 켠 첫 안전한 확장 후보 결과다.
5. 아직 단일 TEST1000 검증이므로 다른 외부 split에서 고정 Gate 검증이 필요하다.

Artifacts:

- `tools/train_eval_dino_reobserve_promotion_v228.py`
- `results/v228_dino_reobserve_promotion_gate_cls10_seed228/summary.json`
- `results/v228_dino_reobserve_promotion_gate_cls10_seed228/reobserve_promotion_gate_v228.pkl`
- `results/v228_dino_reobserve_promotion_gate_cls10_seed228/oof_candidate_training_rows.npz`
- `results/v228_dino_reobserve_promotion_gate_cls10_seed228/test_promotion_audit.csv`

## 2026-07-15 - v229 group-aware promotion Gate (negative result)

목적:

```text
v228이 각 candidate를 독립적으로 평가한 한계를 보완한다.
동일 candidate label의 반복 수, view family 수, bbox 다양성,
기존 Parent와 Fine 후보의 정렬을 sample-level 집합 feature로 추가한다.
```

추가 feature 20개:

```text
Parent/Fine label support count 및 ratio
독립 view family count
label별 confidence mean/max/std
bbox area variance 및 center spread
candidate Fine과 base Parent 정렬
candidate Parent와 base Fine-parent 정렬
graph conflict 해소/생성 여부
```

모델과 threshold는 v228과 동일하게 TRAIN OOF에서만 학습/선택했고 TEST로
재조정하지 않았다.

| policy | OOF fixed/broken | TEST fixed/broken |
|---|---:|---:|
| Parent group promotion | 10/3 | 2/2 |
| Fine group promotion | 14/2 | 2/0 |

최종 TEST1000:

```text
Parent 99.6% -> 99.6%
Fine   98.4% -> 98.6%
Both   98.1% -> 98.3%
```

v228 비교:

```text
v228 Fine 3 fixed / 0 broken, final 98.7%
v229 Fine 2 fixed / 0 broken, final 98.6%

v228 Parent 0 / 0
v229 Parent 2 / 2
```

해석:

1. 반복/다양성 feature 자체는 OOF 복구량을 늘렸지만 TEST로 일반화되지 않았다.
2. Parent threshold가 0.9991에서 0.1734로 낮아져 불필요한 전환 4개를 승인했고
   그중 2개를 깨뜨렸다.
3. Fine에서는 v228이 고친 Persian 1개를 놓쳐 총 성능도 낮아졌다.
4. 후보 반복은 정답 증거 반복과 shortcut 반복을 아직 구분하지 못한다.
5. TEST를 보고 threshold 또는 feature rule을 다시 조정하지 않는다.
6. 현재 채택 모델은 v228이며 v229는 부정 실험으로 보존한다.

Artifacts:

- `tools/train_eval_dino_reobserve_group_promotion_v229.py`
- `results/v229_group_aware_reobserve_promotion_cls10_seed229/summary.json`
- `results/v229_group_aware_reobserve_promotion_cls10_seed229/group_promotion_gate_v229.pkl`
- `results/v229_group_aware_reobserve_promotion_cls10_seed229/test_group_promotion_audit.csv`

## 2026-07-15 - v230a learning trajectory evidence nodes

목적:

```text
완료된 TRAIN의 정답/오답 사례만으로 Gate를 학습하지 않는다.
동일 probe를 학습 전부터 반복 관측하여,
모델이 어떤 증거를 일시적으로 외우고 어떤 증거를 안정적으로 유지하는지 추적한다.
```

구조:

```text
고정 probe 300장
- TRAIN seen: 200
- heldout: 100

frozen DINO tile token
-> label-free MiniBatchKMeans evidence node 24개
-> Parent/Fine projector를 14 epoch 학습
-> epoch별 attention, probability, tile sensitivity, weight update 저장
-> 학습 후 node support/persistence/분산을 감사
```

노드 생성에는 라벨을 사용하지 않았다. Parent/Fine projector 학습과 마지막 역할
감사에는 TRAIN 라벨을 사용했다. 역할명은 현재 학습된 분류기가 아니라 다음 조건을
사용한 heuristic audit이다.

| 역할 | 노드 수 | 초기 attention | 최종 attention | 평균 heldout support |
|---|---:|---:|---:|---:|
| stable_shared | 11 | 0.0363 | 0.0829 | 11.55 |
| transient | 2 | 0.0463 | 0.0169 | 1.00 |
| sample_specific | 10 | 0.0481 | 0.0035 | 0.40 |
| watchlist | 1 | 0.0270 | 0.0196 | 2.00 |

핵심 대비:

```text
stable_shared attention: 평균 2.44배 증가
sample_specific attention: 초기의 7.75%만 잔존

evidence_node_006:
  support 85, heldout 30, class 7
  attention 0.0786 -> 0.2265
  대표 타일: 털/몸통 경계와 연속 질감

evidence_node_014:
  support 58, heldout 21, class 8
  attention 0.0180 -> 0.1179
  대표 타일: 눈과 얼굴 주변의 객체 구조

evidence_node_001:
  peak epoch 0, attention 0.0549 -> 0.0228
  대표 타일: 과노출/배경성 패턴
```

probe 성능:

```text
epoch 0  Parent 23.33%, Fine 9.33%
epoch 1  Parent 99.33%, Fine 96.00%
epoch 2  Parent 100.00%, Fine 98.00%
epoch 14 Parent 100.00%, Fine 99.00%

final seen Fine 100.00%
final heldout Fine 97.00%
```

해석:

1. 모델 학습 과정을 관찰하면 여러 sample/class/heldout에 재등장하며 커지는 증거와
   초기에만 강하고 사라지는 촬영 조건성 증거를 분리할 수 있다는 첫 신호다.
2. 이는 오답을 외운 Gate가 아니라, 모델이 어떤 증거에 끌리고 그 습관이 학습 중
   어떻게 굳어지는지를 기록하는 Gate 입력 후보에 가깝다.
3. 아직 최종 오류 수정 Gate도 아니고 node의 semantic 의미를 증명한 것도 아니다.
4. DINO backbone은 frozen이므로 이번 실험은 전체 backbone 학습 궤적이 아니라
   Parent/Fine projector가 DINO 증거를 채택하는 궤적을 측정한다.
5. epoch 1에서 성능이 급상승해 초기 형성 과정이 뭉개졌다. 다음 실험은 epoch 수를
   늘리는 것보다 첫 epoch 내부의 batch 0/1/2/4/8/16 snapshot을 추가하는 편이 낫다.

Artifacts:

- `tools/train_analyze_dino_learning_trajectory_v230.py`
- `results/v230a_learning_trajectory_nodes_cls10_seed230/summary.json`
- `results/v230a_learning_trajectory_nodes_cls10_seed230/learning_trajectory_v230.npz`
- `results/v230a_learning_trajectory_nodes_cls10_seed230/evidence_nodes.csv`
- `results/v230a_learning_trajectory_nodes_cls10_seed230/evidence_nodes.json`
- `results/v230a_learning_trajectory_nodes_cls10_seed230/trajectory_projector_v230.pt`
- `results/v230a_learning_trajectory_nodes_cls10_seed230/representatives/`

## 2026-07-15 - v230b first-epoch intra-batch trajectory

목적:

```text
v230a에서 epoch 0 -> epoch 1 사이에 뭉개졌던 증거 형성 과정을
동일 probe와 동일 자동 evidence node로 batch 단위 관찰한다.
```

추가 snapshot:

```text
epoch 0 initial
epoch 1 batch 1 / 2 / 4 / 8 / 16
epoch 2~14 end
```

학습 초기 probe 변화:

| snapshot | Parent | Fine | seen Fine | heldout Fine |
|---|---:|---:|---:|---:|
| initial | 23.33% | 9.33% | 11.00% | 6.00% |
| batch 1 | 46.67% | 24.67% | 26.00% | 22.00% |
| batch 2 | 64.67% | 46.00% | 48.00% | 42.00% |
| batch 4 | 84.67% | 75.33% | 77.50% | 71.00% |
| batch 8 | 97.33% | 93.00% | 94.50% | 90.00% |
| batch 16 | 99.33% | 96.00% | 97.50% | 93.00% |
| epoch 14 | 100.00% | 99.00% | 100.00% | 97.00% |

역할별 평균 attention trajectory:

| 역할 | initial | batch 4 | batch 8 | batch 16 | final |
|---|---:|---:|---:|---:|---:|
| stable_shared | 0.0363 | 0.0416 | 0.0449 | 0.0547 | 0.0830 |
| transient | 0.0463 | 0.0417 | 0.0394 | 0.0354 | 0.0167 |
| sample_specific | 0.0481 | 0.0425 | 0.0394 | 0.0296 | 0.0034 |
| watchlist | 0.0270 | 0.0345 | 0.0331 | 0.0307 | 0.0195 |

최종 stable node와 나머지 node를 초기 attention delta로 구분한 사후 진단:

```text
batch 1  AUC 0.682
batch 2  AUC 0.720
batch 4  AUC 0.735
batch 8  AUC 0.902
batch 16 AUC 1.000
```

해석:

1. batch 4까지는 객체 공통 증거와 sample-specific shortcut이 아직 섞여 있다.
2. batch 8부터 stable/shared node가 증가하고 임시 node가 감소하는 분리가 선명하다.
3. seen과 heldout가 거의 함께 상승하므로 단순 TRAIN sample 암기만으로 설명하기 어렵다.
4. Gate가 관찰할 핵심은 최종 confidence 하나가 아니라 node별 채택 속도, heldout 전이,
   지속성, 감소/증가 방향이다.
5. AUC는 최종 heuristic role을 사용한 retrospective diagnostic이다. 아직 미래 node 역할이나
   외부 TEST 오류를 예측하는 독립 성능으로 해석하면 안 된다.
6. 다음 Gate 입력 후보는 `early growth`, `late persistence`, `seen-heldout support`,
   `sample concentration`, `Parent/Fine adoption gap`이다.

Implementation:

- 기존 독립 도구에 `--early_snapshot_batches`를 추가했으며 별도 체인을 만들지 않았다.
- snapshot별 `epoch`, `batch_in_epoch`, `global_step`, weight vector, attention과 확률을 저장한다.

Artifacts:

- `tools/train_analyze_dino_learning_trajectory_v230.py`
- `results/v230b_intra_epoch_trajectory_cls10_seed230/summary.json`
- `results/v230b_intra_epoch_trajectory_cls10_seed230/learning_trajectory_v230.npz`
- `results/v230b_intra_epoch_trajectory_cls10_seed230/evidence_nodes.csv`
- `results/v230b_intra_epoch_trajectory_cls10_seed230/representatives/`

## 2026-07-15 - v230c LearningHabitProfile and train-only normality audit

목적:

```text
v230b의 학습 궤적을 최종 분석표로만 두지 않고,
새 이미지에서 Parent/Fine가 어떤 학습 습관 노드에 의존했는지 읽을 수 있는
연속형 runtime profile로 변환한다.
```

노드별 18개 habit feature:

```text
batch 4/8/16 early growth
batch 16 이후 late growth
final/mean attention
positive step ratio와 persistence
sample/heldout/class support
sample concentration
Parent/Fine early growth
Parent/Fine final gap과 adoption time
```

각 새 이미지의 DINO 16개 타일을 v230b centroid 24개에 배치하고, Parent와 Fine의
predicted-class attention으로 node feature를 각각 가중 집계했다. 최종 runtime profile은
149차원이며 정답 라벨을 포함하지 않는다.

| 대상 | n | Parent | Fine | Parent/Fine node cosine |
|---|---:|---:|---:|---:|
| TRAIN | 2,500 | 100.00% | 99.52% | 0.8684 |
| TEST1000 | 1,000 | 99.70% | 98.60% | 0.8721 |

TEST 상태 감사:

```text
both correct        984
Parent only correct  13
Fine only correct     2
neither correct       1
```

프로필이 오류 신호를 실제로 갖는지 확인하기 위해 TRAIN 2,500장 profile만 사용하여
StandardScaler -> PCA(95%, 31차원) -> IsolationForest 정상성 모델을 학습했다.
TRAIN 정답/오답 라벨은 사용하지 않았고 TEST 라벨은 마지막 AUC 감사에만 사용했다.

| 상태 | TEST 수 | 정상성 위험 AUC |
|---|---:|---:|
| 둘 중 하나 이상 오답 | 16 | 0.862 |
| Parent 정답 / Fine 오답 | 13 | 0.829 |
| Parent 오답 / Fine 정답 | 2 | 0.997 |
| 둘 다 오답 | 1 | 0.981 |

해석:

1. 최종 confidence가 아닌 학습 중 증거 채택 속도/지속성/전이성을 새 이미지의
   Parent/Fine 판단 상태에 연결할 수 있게 됐다.
2. 특히 기존 병목이던 Parent-only-correct 상태가 AUC 0.829로 나타난 것은 긍정 신호다.
3. 오답 16개, Fine-only 2개, neither 1개로 표본이 매우 작다. 높은 AUC를 확정 성능이나
   threshold 선택 근거로 사용하면 안 된다.
4. v230c는 감지 profile과 비지도 정상성 감사까지만 수행한다. prediction switching은 하지 않는다.
5. 다음 단계는 다른 split에서 profile을 고정 검증하고, 그 후에만 review/reobserve 입력으로 사용한다.

Artifacts:

- `tools/build_dino_learning_habit_profile_v230.py`
- `tools/eval_dino_learning_habit_profile_v230.py`
- `results/v230c_learning_habit_profile_cls10_train/learning_habit_profile_v230.npz`
- `results/v230c_learning_habit_profile_cls10_test1000/learning_habit_profile_v230.npz`
- `results/v230c_learning_habit_normality_audit_cls10/summary.json`
- `results/v230c_learning_habit_normality_audit_cls10/habit_normality_v230.pkl`

## 2026-07-15 - v231 trajectory-guided multi-expert defer and safe correction

목적:

```text
v230c 학습 궤적 정상성으로 불안정 샘플을 먼저 감지하고,
재관측 후보의 DINO 증거와 TRAIN-only prototype/kNN 지지를 결합하여
Parent와 Fine 판단을 독립적으로 keep/switch/defer한다.
```

구조:

```text
TRAIN 5-fold OOF
  -> v230c trajectory normality review
  -> v228 계열 expanded DINO reobserve views
  -> view consensus / marginal entropy / trajectory node normality
  -> frozen DINO kNN support
  -> Parent/Fine base/candidate correctness OvA heads
  -> candidate utility - base utility
  -> TRAIN OOF threshold

TEST
  -> 고정된 detector, heads, threshold만 적용
  -> TEST label은 마지막 fixed/broken 감사에만 사용
```

TEST 가중치 갱신과 TEST threshold 선택은 수행하지 않았다.

### v231a 초기형

| 지표 | Parent | Fine |
|---|---:|---:|
| fixed | 1 | 2 |
| broken | 0 | 2 |

```text
Parent accuracy 99.7% -> 99.8%
Fine accuracy   98.6% -> 98.6%
Both accuracy   98.4% -> 98.5%
```

초기형은 TRAIN OOF 최적화 과정에서 음수 utility threshold를 허용했다. 따라서 후보가
base보다 나쁘다고 예측된 경우에도 전환될 수 있었고, 실제로 wolf Fine 1개가 이 이유로
깨졌다. 이는 전환 게이트의 의미와 맞지 않는 구조적 문제였다.

### v231b 안전형

변경:

```text
broken penalty = 4
minimum candidate utility delta = 0
기본 행동 = keep
```

TRAIN OOF 정책:

| branch | threshold | switch | fixed | broken | net |
|---|---:|---:|---:|---:|---:|
| Parent | 0.0686 | 5 | 5 | 0 | +5 |
| Fine | 0.3934 | 11 | 9 | 1 | +8 |

TEST 감지:

```text
review 요청             76 / 1000
base invalid            16
invalid captured        12 / 16
정상 review             64
```

TEST 최종:

| 지표 | base | v231b | fixed / broken |
|---|---:|---:|---:|
| Parent | 99.7% | 99.8% | 1 / 0 |
| Fine | 98.6% | 98.8% | 2 / 0 |
| Parent and Fine | 98.4% | 98.7% | 3 / 0 |

실제 승인된 전환:

```text
persian+cat_10093 Parent: 1 -> 0, fixed
chihuahua_10045 Fine:     3 -> 2, fixed
fox_10441 Fine:           4 -> 5, fixed
```

해석:

1. 학습 궤적 기반 정상성은 단순 감지에서 실제 안전 수정까지 연결됐다.
2. TEST를 보고 threshold를 고르지 않고도 3개를 수정하고 broken 0을 유지했다.
3. 현재 병목은 detector가 놓친 invalid 4개와, review에 잡혔지만 승인 가능한 정답 후보를
   만들거나 식별하지 못한 사례다.
4. 다음 개선은 review 범위를 공격적으로 넓히기보다, 포착된 오류에 대한 후보 품질과
   branch별 correction utility를 높이는 쪽이 적합하다.

Artifacts:

- `tools/train_eval_trajectory_guided_defer_v231.py`
- `docs/v231_trajectory_guided_correction_design_ko.md`
- `results/v231a_trajectory_guided_defer_cls10_seed231/summary.json`
- `results/v231b_trajectory_guided_defer_safe_cls10_seed231/summary.json`
- `results/v231b_trajectory_guided_defer_safe_cls10_seed231/test_action_audit.csv`

## 2026-07-15 - v232 ResNet18 trajectory-guided correction control

목적:

```text
v231의 감지와 수정이 DINO 표현력에만 의존하는지 확인한다.
동일 TRAIN/TEST, Parent/Fine projector, trajectory profile, OOF correction을 유지하고
frozen backbone만 DINOv2-S/14에서 ResNet18로 교체한다.
```

ResNet18의 마지막 convolution feature map `[B,512,7,7]`을 spatial token 49개로
해석하고, adaptive pooling으로 observer-aligned 4x4 token `[B,16,512]`을 만들었다.
기존 DINO cache와 동일한 `cls_token`/`tile_tokens` 계약을 사용하므로 downstream
projector, trajectory, normality, correction 모듈은 변경하지 않았다.

### ResNet18 Parent/Fine evidence baseline

| metric | TRAIN | validation | TEST1000 |
|---|---:|---:|---:|
| Parent | 97.6% | 97.2% | 95.6% |
| Fine | 96.1% | 94.2% | 94.0% |
| Fine tile-only | 94.1% | 92.4% | 92.5% |

초기 evidence projector 기준 TEST state:

```text
both valid   917
Parent only   39
Fine only     23
neither       21
```

### ResNet18 learning trajectory

고정 probe 300장에서 동일한 24개 자동 evidence node와 batch 1/2/4/8/16 snapshot을
사용했다.

```text
initial  Parent 16.0%, Fine 10.0%
batch 8  Parent 91.3%, Fine 85.3%
batch 16 Parent 95.7%, Fine 92.7%
final    Parent 99.3%, Fine 98.0%

final seen Fine    100.0%
final heldout Fine  94.0%
```

node roles:

```text
stable_shared     13
sample_specific    7
watchlist          4
transient           0
```

ResNet18에서도 stable/shared와 sample-specific trajectory 분리가 batch 16에서
AUC 0.923, epoch 2 이후 1.0으로 형성됐다.

TRAIN label을 사용하지 않은 normality audit:

| TEST state | count | AUC |
|---|---:|---:|
| any invalid | 78 | 0.859 |
| Parent correct / Fine wrong | 37 | 0.784 |
| Parent wrong / Fine correct | 16 | 0.912 |
| neither correct | 25 | 0.879 |

전체 오류 AUC는 DINO v230c의 0.862와 거의 같았다. 학습 습관을 통한 이상 감지는
DINO 전용 성질이 아니라 ResNet18 projector에서도 재현됐다.

### v232f safe correction

조건:

```text
TRAIN 5-fold OOF only
review quantile 95%
broken penalty 4
minimum utility delta 0
TEST weight update / threshold selection 없음
```

TRAIN OOF:

| branch | switch | fixed | broken | net |
|---|---:|---:|---:|---:|
| Parent | 24 | 21 | 2 | +19 |
| Fine | 24 | 19 | 2 | +17 |

TEST detection:

```text
review                 66 / 1000
base invalid           78
invalid captured       30 / 78
false review           36
```

TEST correction:

| metric | base | v232f | delta |
|---|---:|---:|---:|
| Parent | 95.9% | 97.0% | +1.1%p |
| Fine | 93.8% | 94.6% | +0.8%p |
| Parent and Fine | 92.2% | 93.6% | +1.4%p |

전환 감사:

```text
Parent: 12 fixed / 1 broken / 2 wrong-to-wrong
Fine:    8 fixed / 0 broken / 2 wrong-to-wrong
total:  20 fixed / 1 broken / 4 wrong-to-wrong
```

비교:

| backbone | base both | final both | gain |
|---|---:|---:|---:|
| DINOv2-S/14 v231b | 98.4% | 98.7% | +0.3%p |
| ResNet18 v232f | 92.2% | 93.6% | +1.4%p |

기존 conventional full-image ResNet18 baseline Fine 정확도는 94.1%이고 v232f Fine은
94.6%다. 다만 head와 관측 구조가 다르므로 이 `+0.5%p`는 참고 비교이며, 구조 효과의
주 비교는 동일 v232 base 93.8%에서 94.6%로 오른 `+0.8%p`다.

해석:

1. trajectory normality와 안전 재관측 승인은 ResNet18에서도 작동한다.
2. DINO보다 base 오류가 많아 실제 correction gain은 ResNet18에서 더 크게 나타났다.
3. 그러나 78개 오류 중 top-5% review가 포착한 것은 30개다. 현재 병목은 correction
   action보다 detector operating point와 포착되지 않은 48개 오류다.
4. TEST를 보고 review 범위를 넓히면 안 된다. 다음 공격성 비교는 TRAIN OOF objective로
   5%/10% review budget을 사전 선택한 뒤 별도 TEST에서 검증해야 한다.

Artifacts:

- `src/dual_line/backbone_adapters.py`
- `tools/build_dino_spatial_cache_v226.py`
- `tools/train_eval_trajectory_guided_defer_v231.py`
- `results/resnet18_spatial_cache_v232_cls10_train/backbone_spatial_cache_v232.npz`
- `results/resnet18_spatial_cache_v232_cls10_test/backbone_spatial_cache_v232.npz`
- `results/v232a_resnet18_concept_evidence_cls10_seed226/summary.json`
- `results/v232c_resnet18_intra_epoch_trajectory_cls10_seed230/summary.json`
- `results/v232e_resnet18_learning_habit_normality_audit_cls10/summary.json`
- `results/v232f_resnet18_trajectory_guided_defer_safe_cls10_seed231/summary.json`
- `results/v232f_resnet18_trajectory_guided_defer_safe_cls10_seed231/test_action_audit.csv`

### v232 TEST truth-permutation leakage audit

`정상 review 36장은 모두 keep`, `초기 오답 review 30장 중 20장만 switch`라는 결과가
TEST truth를 암묵적으로 사용한 것인지 확인했다.

검사:

```text
TEST cache label 1,000개 중 886개를 seed 232로 permutation
이미지, embedding, trajectory, TRAIN data/model/threshold는 동일
원본과 permutation 실행의 action row를 truth 컬럼 제외 후 완전 비교
```

결과:

```text
action rows                 132 / 132
decision rows identical     true
different decision rows     0
branch switches             25 / 25
review samples              66 / 66
```

Permutation 후 정답 기반 accuracy/fixed/broken 감사값은 당연히 크게 바뀌었지만,
review 대상, 후보, base/candidate score, utility, threshold, view, switch 결정은 한 건도
바뀌지 않았다. 따라서 TEST truth는 action 결정에 사용되지 않았다.

주의:

1. correction head와 threshold는 TRAIN OOF truth로 supervised 학습된다.
2. 따라서 이 결과는 TEST-label-free inference이지 완전 label-free correction은 아니다.
3. 코드 청결성 측면에서는 향후 inference runner에서 truth 필드를 완전히 제거하고,
   별도 audit 단계에서만 label을 join하는 편이 적합하다.

Artifact:

- `results/v232_truth_permutation_audit/`
## 2026-07-15 - v233a TRAIN-OOF error-risk ranker

목적:

```text
기존 trajectory normality를 단순 임계값으로 사용하는 대신,
TRAIN 5-fold OOF에서 Parent/Fine 오류 위험의 순위를 직접 학습한다.

이번 모듈은 detector만 교체하며 재관측 후보 생성, 전환 승인, 예측 수정은 하지 않는다.
```

구조:

```text
OOF base evidence
  -> any-invalid / Parent-invalid / Fine-invalid 독립 ranker
  -> direct risk와 branch risk 결합
  -> TRAIN OOF Recall@5/10/20% + AURC로 모델 및 fusion 선택
  -> TRAIN OOF CDF와 threshold 고정

TEST
  -> 고정 model/CDF/threshold 적용
  -> TEST label은 마지막 detection audit에만 사용
```

선택 결과:

```text
any-invalid model      HGB depth 2
fusion                 0.75 * direct + 0.25 * max(Parent, Fine)
normality blend        0.0
```

즉 normality feature는 ranker 입력에는 남았지만, 기존 normality 순위를 별도 가산하는 것은
TRAIN OOF에서 선택되지 않았다.

### 동일 TEST rank budget 비교

| backbone | detector | Recall@5% | Recall@10% | Recall@20% | ROC-AUC |
|---|---|---:|---:|---:|---:|
| DINOv2-S/14 | v230 normality | 10/16, 62.5% | 12/16, 75.0% | 13/16, 81.2% | 0.8620 |
| DINOv2-S/14 | v233a | **14/16, 87.5%** | **16/16, 100.0%** | **16/16, 100.0%** | **0.9823** |
| ResNet18 | v230 normality | 24/78, 30.8% | 38/78, 48.7% | 60/78, 76.9% | 0.8593 |
| ResNet18 | v233a | **38/78, 48.7%** | **63/78, 80.8%** | **71/78, 91.0%** | **0.9584** |

### TRAIN threshold를 고정한 단일 이미지형 감사

| backbone | TRAIN budget | TEST review | captured | recall | precision |
|---|---:|---:|---:|---:|---:|
| DINOv2-S/14 | 5% | 58/1000 | 14/16 | 87.5% | 24.1% |
| DINOv2-S/14 | 10% | 107/1000 | 16/16 | 100.0% | 15.0% |
| ResNet18 | 5% | 55/1000 | 43/78 | 55.1% | 78.2% |
| ResNet18 | 10% | 103/1000 | 64/78 | 82.1% | 62.1% |

해석:

1. ResNet18의 핵심 병목이던 상위 review 구간의 압축이 크게 개선됐다.
2. DINO는 10% TRAIN threshold에서 현재 16개 오류를 전부 review 대상으로 올렸다.
3. direct any-invalid와 branch별 위험을 함께 쓰는 것이 direct/branch 단독보다 OOF에서 좋았다.
4. v233a는 supervised TRAIN-OOF failure ranker이다. TEST label/threshold는 사용하지 않았지만
   완전 비지도 detector라고 부르면 안 된다.
5. 아직 최종 정확도 변화는 없다. 다음 v234는 v233a가 추가 포착한 샘플에서 정답 후보를
   만들고 고르는 correction 문제를 별도로 검증해야 한다.

Artifacts:

- `tools/train_eval_error_risk_ranker_v233.py`
- `results/v233a_dino_error_risk_ranker_cls10_seed233/summary.json`
- `results/v233a_dino_error_risk_ranker_cls10_seed233/error_risk_ranker_v233.pkl`
- `results/v233a_resnet18_error_risk_ranker_cls10_seed233/summary.json`
- `results/v233a_resnet18_error_risk_ranker_cls10_seed233/error_risk_ranker_v233.pkl`

## 2026-07-15 - v234 calibrated multi-expert joint correction

목적:

```text
v233이 review한 샘플에서 base와 각 expanded view를 독립 expert로 취급하고,
Parent/Fine/Joint correctness를 TRAIN OOF에서 학습하여 최종 pair를 안전하게 교체한다.
```

구조:

```text
base expert + reobserve view experts
  -> Parent correctness OvA
  -> Fine correctness OvA
  -> Joint pair correctness OvA
  -> OOF score calibration
  -> candidate pair selection
  -> TRAIN OOF utility threshold
```

TEST label은 review, candidate score, switch 결정에 사용하지 않고 마지막 fixed/broken 및
oracle 감사에만 사용했다.

### ResNet18 v234a

```text
v233 review                 103 / 1000
captured base invalid        64 / 78
correct pair candidate       58 / 64
selected full recovery       40 / 58 oracle-hit
fixed / broken               40 / 2
net                          +38
```

| metric | base | v234a | delta |
|---|---:|---:|---:|
| Parent | 95.9% | 98.0% | +2.1%p |
| Fine | 93.8% | 96.0% | +2.2%p |
| Parent and Fine | 92.2% | **96.0%** | **+3.8%p** |

ResNet18에서는 detector 확대와 joint candidate correction이 모두 효과를 냈다. 기존
v232f의 final both 93.6%보다 2.4%p 높다.

### DINO v234a/v234c

```text
v233 review                 107 / 1000
captured base invalid        16 / 16
correct pair candidate       10 / 16
selected full recovery        3 / 10 oracle-hit
fixed / broken                3 / 1
net                           +2
base both 98.4% -> final 98.6%
```

기존 normality 5% OOF 후보 대신 v233 10% OOF 후보를 250개 샘플/2,040개 view로 다시
생성했지만 DINO 결과는 `3 fixed / 1 broken`으로 변하지 않았다.

DINO 후보 전체 순위 감사:

```text
reviewed errors             16
correct pair가 있는 sample  10
correct pair rank 1           3
correct pair rank 5           2
correct pair rank 6           1
correct pair rank 7           3
correct pair rank 10          1
correct pair 없음             6
```

놓친 oracle-hit 7개의 정답 후보는 최고 오답 후보보다 calibrated score가 중앙값 약 0.41
낮았다. 따라서 DINO 병목은 탐지가 아니라 강한 오답 view를 과신하는 candidate ranking이다.
v234 DINO 98.6%는 broken 0인 기존 v231b 98.7%보다 낮으므로 DINO 운영 모델을 v234로
교체하지 않는다.

해석:

1. v234 multi-expert joint correction은 ResNet18에서 강하게 성공했다.
2. DINO는 base가 이미 98.4%라 correction OOF positive가 적고, 정답 view가 있어도
   confidence/evidence rank에서 밀리는 경우가 많다.
3. DINO 다음 실험은 TEST에서 consensus 계수를 조정하면 안 된다. TRAIN OOF에서 view
   하나가 아니라 source별 view set 전체를 입력받는 set-level ranker를 학습하고 새
   heldout에서 검증해야 한다.
4. 현재 최종 선택은 `DINO v231b`, `ResNet18 v234a`가 적합하다.

Artifacts:

- `tools/train_eval_multi_expert_correction_v234.py`
- `tools/build_v233_review_candidates_v234.py`
- `results/v234a_resnet18_multi_expert_correction_cls10_seed234/summary.json`
- `results/v234c_dino_v233_review_oof_candidates_cls10_seed234/summary.json`
- `results/v234c_dino_multi_expert_v233_oof_cls10_seed234/summary.json`
- `results/v234c_dino_multi_expert_v233_oof_cls10_seed234/test_candidate_scores_v234.npz`

## 2026-07-15 - v235 physically label-free inference verification

목적:

```text
TEST 입력 artifact에서 label/truth/target 키를 물리적으로 제거한 뒤에도
v233 review 탐지와 v234 correction 판단이 동일하게 실행되는지 검증한다.
```

입력 캐시 키:

```text
class_names
cls_token
image_path
name
tile_tokens
```

결과:

```text
n                         1000
input label keys          0
review                    103
switch                     50
accuracy computed       false
```

기존 v234 truth-audit 실행과 비교:

```text
review sample             exact match
selected reobserve view   exact match
switch decision           exact match
final Parent prediction   exact match
final Fine prediction     exact match
```

해석:

1. TEST 정답은 v233/v234 런타임 판단에 필요하지 않다.
2. 정확도, fixed, broken은 추론이 끝난 뒤 별도 audit에서만 계산할 수 있다.
3. TRAIN label은 detector/correction 학습 및 train-only support 구축에 사용되며,
   이는 TEST label 누수와 구분된다.
4. 따라서 v234의 TEST 실행 경로는 예측과 평가를 분리할 수 있음이 확인됐다.

Artifacts:

- `tools/infer_label_free_v235.py`
- `results/resnet18_spatial_cache_v232_cls10_test_unlabeled/backbone_spatial_cache_unlabeled.npz`
- `results/v235_resnet18_label_free_inference_cls10/summary.json`
- `results/v235_resnet18_label_free_inference_cls10/predictions.csv`

### v235 raw-image 3-case end-to-end audit

기존 TEST cache를 재사용하지 않고 클래스 폴더 밖으로 복사해 중립적인 파일명만
부여한 이미지 3장으로 시작했다.

```text
input directory
  sample_001.jpg
  sample_002.jpg
  sample_003.jpg

raw image
-> label-free spatial cache
-> v233 review detector
-> expanded reobservation
-> v234 correction
-> final prediction
```

결과:

| sample | review | switch | chosen view | final Parent | final Fine |
|---|---:|---:|---|---|---|
| sample_001 | no | no | keep | dog | chihuahua |
| sample_002 | yes | yes | parent_context | horse | horse |
| sample_003 | yes | no | parent_context | cat | siamese+cat |

```text
n                         3
review                    2
switch                    1
input label keys          0
accuracy computed       false
```

세 샘플의 review/switch/view/final prediction과 risk score는 기존 1,000장 실행에서
선택한 원본 샘플 결과와 모두 동일했다. 이 검증은 라벨 없는 raw image 입력부터
최종 판단까지의 end-to-end 실행이 가능함을 확인한다.

Artifacts:

- `tools/build_unlabeled_spatial_cache_v235.py`
- `results/v235_label_free_three_raw/spatial_cache/backbone_spatial_cache_unlabeled.npz`
- `results/v235_label_free_three_raw/inference/summary.json`
- `results/v235_label_free_three_raw/inference/predictions.csv`
