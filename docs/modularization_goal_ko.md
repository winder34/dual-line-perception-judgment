# Dual-Line 모듈화 목표

작성일: 2026-07-01

## 결론

다음 단계의 목표는 새 정확도 실험이 아니라 **실험 체인을 교체 가능한 모듈 구조로 바꾸는 것**이다.

현재 `manual v153`은 가장 강한 reference implementation으로 보존한다. 다만 v153까지 가는 전체 체인을 그대로 고정하면 백본 교체, 자동화, 재실험 속도에서 병목이 생긴다.

따라서 모듈화의 핵심은 다음 3가지다.

```text
1. backbone 교체를 쉽게 한다.
2. CSV 중심 중간 산출물 병목을 줄인다.
3. v153 / v153auto / 이후 gate를 같은 인터페이스에서 비교 가능하게 한다.
```

## 현재 기준선

```text
reference:
  manual v153 structured transition gate

automation target:
  v153auto concept graph / AND relation gate

not replacement yet:
  v180, v189, v190, v190b
```

manual v153은 현재까지 가장 안정적인 통합 기준선이다. v153auto는 수동 parent/fine 의미 라벨 없이 근접한 결과를 냈지만, noisy external set에서 broken 억제가 아직 약하다.

## 문제 정의

현재 파이프라인은 실험적으로는 강하지만 운영 구조가 길다.

```text
scan
→ tile / roi cache
→ texture cache
→ relation cache
→ v130 / v131 feature CSV
→ v150 candidate
→ v153 gate
→ report
```

문제는 두 가지다.

```text
계산 병목:
  v130 계열 texture/object relation feature 생성

운영 병목:
  CSV를 계속 읽고 쓰면서 schema가 흩어지고,
  어떤 파일이 어떤 feature 계약을 만족하는지 추적하기 어려움
```

## 모듈화 원칙

### 1. CSV는 report/audit 전용으로 낮춘다

CSV를 완전히 없애지는 않는다. 사람이 확인해야 하는 파일은 CSV가 좋다.

하지만 큰 feature matrix, relation tensor, embedding은 CSV보다 binary artifact가 맞다.

```text
CSV:
  predictions
  index
  class accuracy
  fixed/broken/net report
  audit table

NPZ / Parquet:
  embedding
  candidate feature matrix
  relation tensor
  object-flow tensor

manifest.json:
  backbone id
  feature schema id
  class map
  parent map 사용 여부
  생성 명령
  입력 artifact 경로
```

### 2. backbone은 adapter로 감싼다

downstream module이 `resnet18`, `resnet50`, `dinov2`, custom observer backbone을 직접 알면 안 된다.

공통 출력 계약만 맞춘다.

```text
BackboneArtifact:
  sample_key
  embedding
  embedding_dim
  backbone_id
  preprocess_id
  crop_policy_id
  manifest
```

목표 사용 예:

```text
features = build_texture_cache(backbone="resnet18")
features = build_texture_cache(backbone="resnet50")
features = build_texture_cache(backbone="dinov2")
```

이후 candidate/gate는 embedding dimension과 schema만 보고 동작해야 한다.

### 3. gate는 candidate 생성과 분리한다

candidate module은 가능한 후보와 증거를 만든다. gate module은 그 후보를 승인하거나 차단한다.

```text
Candidate module:
  "무엇을 후보로 볼 것인가"

Gate module:
  "기존 판단을 뒤집어도 되는가"
```

v153은 이 분리가 비교적 잘 된 첫 기준선이다.

## 목표 모듈

### A. Artifact / Schema Module

역할:

```text
sample_key 정렬
class_map 관리
parent_map 관리
feature schema 검증
missing/extra column 확인
manifest 생성
```

필수 함수 후보:

```text
load_manifest(path)
validate_sample_order(a, b)
validate_feature_schema(actual, expected)
write_artifact_manifest(...)
```

### B. Backbone / Texture Cache Module

역할:

```text
full image embedding
roi embedding
candidate crop embedding
backbone별 preprocess
batch inference
```

교체 대상:

```text
resnet18
resnet50
efficientnet
dinov2 candidate
custom observer-token backbone
```

### C. Candidate Feature Module

역할:

```text
v130 texture/object relation
v131 texture-object agreement
object relation
wave relation
candidate bundle feature
```

주의:

```text
v130 계열은 계산 병목이므로 shard/restartable 구조 유지
큰 feature는 CSV보다 NPZ/Parquet 우선
```

### D. Concept Graph Module

역할:

```text
auto cluster
sibling seed
risk/context relation
AND relation
partial support node
```

목표:

```text
manual parent/fine 라벨 없이
v153auto가 쓸 수 있는 concept evidence를 만든다.
```

### E. Gate Module

역할:

```text
manual v153 gate
v153auto gate
evidence-quality blocker
future meta-judge gate
```

공통 입력:

```text
selected prediction
candidate prediction
candidate evidence features
concept graph features
object/texture/wave features
```

공통 출력:

```text
final_prediction
switch_approved
gate_score
gate_reason
fixed
broken
net
```

### F. Observer / Tracking Module

역할:

```text
local observer token
global observer relation
object-flow feature
same-object support
```

현재 판단:

```text
final gate에 단순 추가했을 때는 v153을 넘지 못했다.
나중에는 scan/candidate generation 단계에 더 가깝게 넣는 편이 자연스럽다.
```

### G. Evaluation / Report Module

역할:

```text
accuracy
parent accuracy
fine accuracy
fixed
broken
net
class accuracy
dataset별 비교
manual vs auto 비교
```

필수 원칙:

```text
정확도만 보고 판단하지 않는다.
항상 fixed / broken / net을 함께 본다.
```

## 우선 구현 순서

### Phase 1. Schema와 manifest부터 고정

목표:

```text
모든 주요 artifact가 manifest.json을 갖게 한다.
sample order와 feature schema를 자동 검증한다.
```

산출물:

```text
tools/artifact_schema.py
tools/validate_artifact_schema.py
docs/artifact_schema_ko.md
```

### Phase 2. Backbone adapter 분리

목표:

```text
resnet18 / resnet50 교체를 같은 CLI 옵션으로 처리한다.
downstream은 backbone 이름이 아니라 artifact schema를 본다.
```

산출물:

```text
tools/backbone_adapters.py
tools/build_texture_cache_modular.py
```

### Phase 3. v130/v131 feature cache를 binary 우선으로 재정리

목표:

```text
큰 feature와 tensor는 NPZ/Parquet로 저장한다.
CSV는 index와 audit용으로 제한한다.
shard/merge 구조를 표준화한다.
```

산출물:

```text
tools/build_relation_cache_modular.py
tools/merge_relation_cache_shards.py
```

### Phase 4. Gate runner 통합

목표:

```text
manual v153과 v153auto를 같은 runner에서 선택 실행한다.
```

예상 CLI:

```text
python -m tools.run_gate_modular ^
  --gate manual_v153 ^
  --candidate_artifact results/.../candidate_features.npz ^
  --manifest results/.../manifest.json ^
  --out_dir results/...

python -m tools.run_gate_modular ^
  --gate v153auto ^
  --candidate_artifact results/.../candidate_features.npz ^
  --concept_graph results/.../concept_graph.npz ^
  --manifest results/.../manifest.json ^
  --out_dir results/...
```

### Phase 5. Report runner 통합

목표:

```text
어떤 gate를 돌려도 동일한 report가 나온다.
```

산출물:

```text
tools/report_experiment.py
results/.../metrics.json
results/.../predictions.csv
results/.../transition_audit.csv
results/.../class_accuracy.csv
```

## 하지 않을 것

이번 모듈화 단계에서는 다음을 하지 않는다.

```text
1. 새 정확도 실험을 길게 추가하지 않는다.
2. v153 이후 체인을 그대로 더 쌓지 않는다.
3. CSV를 완전히 제거하려고 하지 않는다.
4. 자동 concept node를 완성된 의미 개념으로 과장하지 않는다.
5. v153auto를 아직 manual v153의 완전 대체로 선언하지 않는다.
```

## 성공 기준

1차 성공 기준:

```text
resnet18 / resnet50 feature cache를 같은 downstream runner에 넣을 수 있다.
manual v153과 v153auto를 같은 gate interface에서 실행할 수 있다.
feature schema mismatch를 실행 전에 잡는다.
fixed / broken / net report가 자동 생성된다.
```

2차 성공 기준:

```text
v130/v131 계열 재실행이 shard/restartable artifact 단위로 안정화된다.
CSV read/write 시간이 전체 병목에서 줄어든다.
새로운 observer 또는 dinov2-style backbone을 추가할 때 기존 gate 코드를 거의 건드리지 않는다.
```

## 다음 작업 한 줄 지시문

```text
Implement the modular artifact/schema layer first: manifest-based feature artifacts, sample-order validation, feature-schema validation, and a common report format for manual v153 and v153auto.
```

## 2026-07-03 시작 상태

Phase 1의 첫 뼈대를 추가했다.

추가된 파일:

```text
src/dual_line/runtime_types.py
src/dual_line/artifacts/schema.py
src/dual_line/artifacts/legacy_adapter.py
src/dual_line/evaluation/transition.py
tools/validate_artifact_schema.py
tools/report_modular_run.py
src/dual_line/backbone_adapters.py
src/dual_line/decision/gates.py
tools/run_gate_modular.py
tools/build_texture_cache_modular.py
tools/merge_texture_cache_shards_modular.py
```

현재 가능한 것:

```text
1. ScanBatch / RepresentationBatch / RelationBatch / CandidateBatch / GateOutput 타입 정의
2. legacy CSV/NPZ에서 sample_key와 numeric feature column 추론
3. manifest.json 생성
4. sample order 비교
5. feature schema 비교
6. fixed / broken / net 계산용 공통 metric 함수 제공
7. legacy prediction CSV를 공통 report schema로 요약
8. backbone adapter contract 정의
9. manual v153 / v153auto를 같은 GateRunner interface로 실행
10. BackboneAdapter 기반 texture cache builder 추가
11. modular texture cache shard merge 도구 추가
```

Smoke test:

```text
python -m tools.validate_artifact_schema \
  --artifact results\v153_structured_transition_gate_cls10\all_predictions.csv \
  --write_manifest results\modular_schema_smoke\v153_cls10_all_predictions_manifest.json \
  --out_json results\modular_schema_smoke\v153_cls10_schema_summary.json
```

결과:

```text
ok = true
n_samples = 1000
n_features = 125
```

v153과 v153auto eval sample order 비교:

```text
python -m tools.validate_artifact_schema \
  --artifact results\v153auto_and_logreg_cls10_test\eval_predictions_v153auto.csv \
  --reference results\v153_structured_transition_gate_cls10\all_predictions.csv \
  --out_json results\modular_schema_smoke\v153auto_vs_v153_sample_order.json
```

결과:

```text
ok = true
sample_order_mismatch_count = 0
```

공통 report smoke test:

```text
python -m tools.report_modular_run \
  --pred_csv results\v153_structured_transition_gate_cls10\all_predictions.csv \
  --out_dir results\modular_schema_smoke\report_v153_cls10_fine \
  --run_name v153_cls10_fine \
  --true_col y_true_name \
  --baseline_col mvp_pred_name \
  --final_col final_pred_name_v153

python -m tools.report_modular_run \
  --pred_csv results\v153auto_and_logreg_cls10_test\eval_predictions_v153auto.csv \
  --out_dir results\modular_schema_smoke\report_v153auto_cls10_fine \
  --run_name v153auto_cls10_fine \
  --true_col y_true_name \
  --baseline_col mvp_pred_name \
  --final_col final_pred_name_v153auto
```

결과:

| run | baseline | final | fixed | broken | net |
|---|---:|---:|---:|---:|---:|
| v153 cls10 fine | 94.70% | 96.80% | 33 | 12 | +21 |
| v153auto cls10 fine | 94.70% | 96.60% | 33 | 14 | +19 |

이제 manual v153과 v153auto를 같은 report contract로 비교할 수 있다.

GateRunner smoke test:

```text
python -m tools.run_gate_modular \
  --gate manual_v153 \
  --pred_csv results\v153_structured_transition_gate_cls10\all_predictions.csv \
  --out_dir results\modular_schema_smoke\gate_manual_v153_cls10

python -m tools.run_gate_modular \
  --gate v153auto \
  --pred_csv results\v153auto_and_logreg_cls10_test\eval_predictions_v153auto.csv \
  --out_dir results\modular_schema_smoke\gate_v153auto_cls10
```

결과:

| gate | baseline | final | fixed | broken | net |
|---|---:|---:|---:|---:|---:|
| manual_v153 | 94.70% | 96.80% | 33 | 12 | +21 |
| v153auto | 94.70% | 96.60% | 33 | 14 | +19 |

BackboneAdapter smoke test:

```text
BackboneAdapterConfig(backbone="resnet18", weights="none", device="cpu")
```

결과:

```text
backbone_id = resnet18:none:input224
embedding_dim = 512
artifact_type = backbone_embedding
feature shape = ["N", 512]
```

Backbone 자유도 1차 구현:

```text
tools/build_texture_cache_modular.py
```

목표:

```text
기존 texture_cache.npz 출력 호환 유지
BackboneAdapter로 resnet18 / resnet50 / efficientnet_b0 교체
artifact_manifest.json에 backbone_id / preprocess_id / scan_policy 기록
num_shards / shard_index로 큰 실행을 나눌 수 있게 준비
```

Smoke test:

```text
python -m tools.build_texture_cache_modular \
  --cache_dir results\dual_line_cache_train_1000 \
  --dataset_root dataset \
  --split train \
  --out_dir results\modular_schema_smoke\texture_modular_train2_resnet18_none_roi \
  --backbone resnet18 \
  --weights none \
  --batch_size 2 \
  --device cpu \
  --limit 2
```

결과:

```text
n = 2
embedding_dim = 512
backbone_id = resnet18:none:input224
scan_policy_id = roi_bbox_norm_xyxy
legacy_npz_key = texture_embedding
schema validation ok = true
```

ResNet50 백본 교체 확인 문서:

```text
docs/backbone_resnet50_check_ko.md
```

다음 작은 작업:

```text
1. modular ResNet50 full-image baseline 재현
2. manifest에 gate_version, candidate_version을 더 명시적으로 채우는 helper 추가
3. Gate runner를 legacy CSV가 아닌 CandidateBatch 입력으로 받을 수 있게 확장
```
