"""Write thin, human-readable notebooks. Never read or modify the preserved legacy notebooks."""
from pathlib import Path
import json
import textwrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebooks'


def md(key,text):
    return dict(cell_type='markdown',id=key,metadata={},source=textwrap.dedent(text).strip()+'\n')


def code(key,text):
    return dict(cell_type='code',id=key,metadata={},execution_count=None,outputs=[],
                source=textwrap.dedent(text).strip()+'\n')


def write(name,cells):
    path=OUT/name
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.12'),colab=dict(name=name),recsys_framework='modular_v2'),
        nbformat=4,nbformat_minor=5)
    # A builder must not erase a researcher's saved outputs on a subsequent regeneration.
    if path.exists():
        previous=json.loads(path.read_text())
        if any(c.get('outputs') or c.get('execution_count') is not None for c in previous['cells']):
            raise FileExistsError(f'Refusing to replace executed notebook: {path}')
    path.write_text(json.dumps(notebook,indent=1)+'\n')
    print(name,len(cells),'cells')


def bootstrap(target,modes):
    return [md('environment','''
## 1. Environment and paths

Place this repository's readable `src/recsys/`, `configs/` and `tests/` directories in the Google Drive project directory below. No other notebook needs to have run in this runtime. The existing dataset and split artifacts must already exist for real experiments. The source path and file hashes are checked before imports; restart the runtime if a different or edited package is cached. Dependency requirements and installation instructions are in the project README; no automatic package installation runs here.
'''),code('paths-and-mode',f'''
from pathlib import Path
import sys
import importlib.util
RUN_MODE = "smoke_test"
if RUN_MODE == "smoke_test":
    import os
    for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_thread_variable] = "1"  # bound synthetic CPU native-library concurrency before imports
SUPPORTED_MODES = {tuple(modes)!r}
if RUN_MODE not in SUPPORTED_MODES:
    raise ValueError(f"Unsupported mode {{RUN_MODE!r}}; choose {{SUPPORTED_MODES}}")
PROJECT_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/"
    "personal_projects/movielens_hybrid_recsys"
)
try:
    from google.colab import drive
except ImportError:
    pass
else:
    drive.mount("/content/drive")
SRC_DIR = PROJECT_DIR / "src"
if not (SRC_DIR / "recsys" / "__init__.py").is_file():
    raise FileNotFoundError(f"Readable source missing: {{SRC_DIR / 'recsys'}}. Copy or clone the repository there.")
import hashlib
_source_hash = hashlib.sha256()
for _file in sorted((SRC_DIR / "recsys").rglob("*.py")):
    _source_hash.update(str(_file.relative_to(SRC_DIR)).encode())
    _source_hash.update(_file.read_bytes())
_loaded = sys.modules.get("recsys")
if _loaded is not None:
    if (not Path(_loaded.__file__).resolve().is_relative_to(SRC_DIR.resolve()) or
        getattr(_loaded, "__notebook_source_hash__", None) != _source_hash.hexdigest()):
        raise RuntimeError("A stale/different recsys module is cached. Restart this runtime before importing.")
sys.path = [str(SRC_DIR)] + [p for p in sys.path if p != str(SRC_DIR) and "recsys_source" not in p]
import recsys
recsys.__notebook_source_hash__ = _source_hash.hexdigest()
from recsys.runtime import environment
from recsys.smoke import run_smoke
print("Source:", recsys.__file__)
print(environment("/content" if Path("/content").exists() else "/tmp"))
'''),code('synthetic-smoke',f'''
if RUN_MODE == "smoke_test":
    print(run_smoke(PROJECT_DIR, {target!r}))
''')]

MODELS=[
('03_Popularity_Genre.ipynb','Popularity and Genre CBF',('popularity','genre'),'baselines',
 'Popularity ranks Train-positive counts. Genre CBF uses normalized multi-hot movie genres and normalized Train-positive user profiles. Missing genres contribute no genre signal. Empty profiles receive a declared Popularity fallback.'),
('04_ItemCF.ipynb','Sparse ItemCF',('itemcf',),'itemcf',
 'Binary Train-positive co-occurrence is normalized by item counts to obtain cosine similarity. Bounded source-item blocks retain Top-K neighbors, then sum neighbor weights across the complete positive user history. No unrestricted dense item-item matrix or history truncation is used.'),
('05_Implicit_ALS.ipynb','Weighted implicit ALS',('implicit_als',),'implicit_als',
 'CPU implicit ALS uses p=1,c=1+positive_confidence for observed ratings >=4; p=0,c=1+low_confidence for observed low ratings; missing pairs use p=0,c=1. The library signed-confidence convention distinguishes observed lows from missing feedback. Preference-zero unknowns are an objective proxy, not verified dislikes.'),
('06_LightGCN.ipynb','LightGCN',('lightgcn',),'lightgcn',
 'LightGCN propagates FP32 embeddings through the symmetric normalized positive-edge bipartite graph. It averages layers 0…L and optimizes BPR against sampled unknowns, rejecting every observed rating. The graph polynomial and its exact adjoint run at epoch boundaries; a post-update propagation exports checkpoint/evaluation vectors. Gradients accumulate across loss microbatches, with one large-batch Adam step per epoch. This differs from minibatch SGD and needs Validation tuning. No edge-by-embedding tensor is created.'),
('07_Two_Tower.ipynb','Hybrid Two-Tower',('two_tower',),'two_tower',
 'The user MLP combines a user-ID embedding with a frozen Train-positive genre profile; the item MLP combines a movie-ID embedding with genres. Normalized vectors use a sampled-pair BPR objective against unknown pairs, rejecting observed positive and low ratings. All profiles are fixed at the retriever training cutoff. No external or pretrained embeddings are used.'),
]

for filename,title,names,target,objective in MODELS:
    cells=[md('overview',f'''# MovieLens 32M — {title}

Independent retrieval experiment. {objective}

All primary experiments use the entire external Train partition, Policy B (Train-observed movies), all-Train-rated seen filtering and the same frozen Validation users/labels. The default mode runs synthetic tests only. Saved outputs from the original `03_MovieLens32M_Retrieval_and_Ranking.ipynb` are preserved as a legacy reference and are not loaded by this notebook.
''')]+bootstrap(target,('smoke_test','full_train','load_evaluate','rank_snapshot'))
    cells += [md('configuration','''## 2. Configuration

Edit the readable model JSON files and the explicit switches below. `full_train` uses every Train row. `load_evaluate` requires a checkpoint path and never fits a model. `rank_snapshot` requires an explicit UTC cutoff and produces later internal-Train candidates; it is not enabled by default and is not run as part of framework construction. Snapshot labels never select retriever epochs. Identical boundaries are required across all six snapshot channels before fusion.
'''),code('model-configuration',f'''
from recsys.config import load_model_config
from recsys.workflows import RetrievalExperiment
MODEL_NAMES = {names!r}
CONFIGS = {{name: load_model_config(PROJECT_DIR / "configs", name) for name in MODEL_NAMES}}
SAVED_CHECKPOINTS = {{name: None for name in MODEL_NAMES}}  # set explicit paths for load_evaluate
SNAPSHOT_CUTOFF_UTC = None  # e.g. an approved UTC boundary strictly inside external Train
SNAPSHOT_LABEL_END_UTC = None  # exclusive; None uses the external Validation start
DEVICE = {('cuda' if names[0] in ('lightgcn','two_tower') else 'cpu')!r}
if RUN_MODE == "rank_snapshot":
    for _config in CONFIGS.values():
        _config["snapshot_cutoff_utc"] = SNAPSHOT_CUTOFF_UTC
        _config["snapshot_label_end_utc"] = SNAPSHOT_LABEL_END_UTC
experiments, candidate_outputs, checkpoint_paths = {{}}, {{}}, {{}}
'''),md('verify','''## 3. Load and verify frozen splits

Verify the authoritative manifest, source/artifact checksums, schemas, Train/Validation counts and temporal boundaries. Expected 31,414,033 / 300,000 / 286,171 counts are integrity assertions, not model dimensions. Development does not deserialize Test labels. The cohort table is read only for Validation eligibility/reporting, never for model features.
'''),code('prepare-data','''
if RUN_MODE != "smoke_test":
    _shared_data = None
    for name in MODEL_NAMES:
        exp = RetrievalExperiment(PROJECT_DIR, name, CONFIGS[name], RUN_MODE,
                                  checkpoint=SAVED_CHECKPOINTS[name], device=DEVICE)
        print(exp.prepare(shared_data=_shared_data))
        _shared_data = exp.data  # only the explicitly grouped baselines share RAM within this notebook
        experiments[name] = exp
'''),md('features','''## 4. Prepare model-specific features

The shared context contains canonical IDs, all observed Train histories, sparse positive preferences, Train-only popularity and normalized movie/user genres. Only this notebook's requested model consumes its relevant features. No Validation-derived statistics, Test labels, tags or links enter the context.
'''),code('feature-summary','''
if RUN_MODE != "smoke_test":
    for name, exp in experiments.items():
        print(name, "observed/positive nnz:", exp.data.ctx.observed.nnz, exp.data.ctx.positive.nnz,
              "catalog:", int(exp.data.ctx.eligible.sum()), "cutoff:", exp.data.ctx.cutoff)
'''),md('initialization','''## 5. Initialize model

Inspect the single-model factory configuration, boundary and device. Parameter allocation occurs inside its fit implementation after resource guards; initialization does not launch training.
'''),code('initialize','''
if RUN_MODE != "smoke_test":
    for exp in experiments.values():
        print(exp.initialize())
'''),md('training','''## 6. Train or load exactly the requested model

A matching verified checkpoint is reused. Expensive work starts only in the selected real mode. CUDA is used for neural training; sparse CPU algorithms remain on CPU. Dynamic RAM/disk/VRAM guards stop infeasible stages rather than sample Train. Neural full-Train early stopping uses a deterministic Validation-only panel. Historical snapshots use fixed epochs and training loss only. No unrelated retriever is trained.
'''),code('train-or-load','''
if RUN_MODE != "smoke_test":
    for name, exp in experiments.items():
        print(name, exp.train())
'''),md('checkpoint','''## 7. Save checkpoint

Artifacts use `artifacts/<model>/<run_id>/checkpoint/`, with model files, mappings, configuration, training history, measured resources and a readable manifest. Writes are verified and immutable. Neural epoch state is saved separately; completed models resume automatically, while interrupted neural training restarts from its seed.
'''),code('save-checkpoint','''
if RUN_MODE != "smoke_test":
    for name, exp in experiments.items():
        checkpoint_paths[name] = exp.save_checkpoint()
        print(name, checkpoint_paths[name])
'''),md('retrieval','''## 8. Validation retrieval

Every channel scores the full eligible unseen catalog in bounded chunks and emits at most Top-100 (fewer when filtering leaves fewer eligible movies). Raw scores and deterministic movie-ID tie breaking are preserved. `rank_snapshot` uses the later internal Train window instead of external Validation.
'''),code('retrieve','''
if RUN_MODE != "smoke_test":
    for name, exp in experiments.items():
        candidate_outputs[name] = exp.evaluate()
'''),md('metrics','''## 9. Compute standard metrics

The shared evaluator reports user-macro Recall@20/50/100, HitRate@20/100, NDCG@20, cohort/bootstrap intervals, coverage and popularity concentration. No-hit queries remain in the fixed eligible-positive population. Candidate catalog exclusions and cold/profile fallback counts are explicit. Latency is batch-amortized, not online request latency.
'''),code('show-metrics','''
import json
import pandas as pd
from IPython.display import display
if RUN_MODE != "smoke_test":
    for name, path in candidate_outputs.items():
        summary = json.loads((path / "metrics.json").read_text())
        display(pd.DataFrame(summary["metrics"][name]))
        print(summary["eligibility"])
'''),md('candidate-output','''## 10. Save candidate outputs

Retrieval and metric computation already streamed the standardized file to `results/<validation|rank_train>/<model>/<run_id>/candidates/`. Required columns are `userId,movieId,score,rank,source`; `fallback` adds explicit provenance. The separate manifest records source hashes, split checksum, training cutoff, catalog/mappings, evaluation ID, K, seed, hyperparameters and candidate SHA256. Copy these explicit artifact paths into Notebook 08; no retraining is needed.
'''),code('candidate-paths','''
if RUN_MODE != "smoke_test":
    from recsys.artifacts import read_artifact
    for name, path in candidate_outputs.items():
        manifest = read_artifact(path)
        print(name, "candidates:", path, "rows:", manifest["candidate_rows"], "run:", manifest["run_id"])
'''),md('resources','''## 11. Resource and performance summary

Inspect training duration, sampled process RSS, CUDA allocator peaks, retrieval latency and file size. Checkpoint metadata retains package versions. No large weights or full candidate tables are printed.
'''),code('resource-summary','''
if RUN_MODE != "smoke_test":
    for name, exp in experiments.items():
        print(name, "training:", exp.training_profile,
              "candidate bytes:", (candidate_outputs[name] / "candidates.parquet").stat().st_size)
        history = json.loads((checkpoint_paths[name] / "training_history.json").read_text())
        if history["epochs"]:
            display(pd.DataFrame(history["epochs"]))
'''),md('limitations',f'''## 12. Limitations

{objective}

The benchmark recovers future observed high ratings, not exposure, watching, CTR or causal satisfaction. Missing-not-at-random feedback, submission-time bursts, weak unknown-pair supervision and temporal changes limit interpretation. Policy B excludes unseen-to-Train items and reports their lost future positives. Exact catalog scoring and sparse graph/index construction may still be slow. Source hashes and manifests must match for reuse; changed common protocol code requires a new run. No real MovieLens training or benchmark scores were produced while building this framework.
''')]
    write(filename,cells)

# Fusion workflow: artifacts only, no model training imports.
cells=[md('overview','''# MovieLens 32M — retrieval fusion

Consume six independently saved candidate artifacts. This notebook never fits retrievers. Choose either all full-Train Validation candidates or all historical rank-training candidates with exactly matching boundaries, catalog, mappings, K and evaluation IDs. Mixing these artifact types fails.
''')]+bootstrap('fusion',('smoke_test','load_evaluate'))
cells += [md('configuration','''## 2. Configuration and saved candidate inputs

Paste the exact candidate artifact directories printed by notebooks 03–07. Run this notebook separately for Validation and for historical rank-training candidates. It never selects a latest run implicitly.
'''),code('fusion-config','''
import json
from recsys.config import default_config
from recsys.contracts import CHANNELS
from recsys.fusion.workflow import FusionExperiment
CONFIG = default_config()
CONFIG.update(json.loads((PROJECT_DIR / "configs" / "fusion.json").read_text()))
CANDIDATE_INPUTS = {source: None for source in CHANNELS}
# Example value: PROJECT_DIR / "results/validation/popularity/<run_id>/candidates"
FUSION_OUTPUT = None
'''),md('verify','''## 3. Verify candidate and frozen-data contracts

Input manifests and checksums must match the same frozen dataset, cutoff, catalog/mappings, query-label identifier and Top-K budget. All six sources are required. Full-Train raw scores are preserved but are not assumed calibrated. The context is rebuilt from the corresponding read-only Train boundary; no retriever is loaded or trained.
'''),code('prepare-fusion','''
if RUN_MODE != "smoke_test":
    if any(path is None for path in CANDIDATE_INPUTS.values()):
        raise ValueError("Set all six explicit candidate artifact paths first.")
    fusion = FusionExperiment(PROJECT_DIR, CANDIDATE_INPUTS, CONFIG)
    print(fusion.reference["partition"], fusion.reference["evaluation_id"])
'''),md('union','''## 4. Union, provenance, cutoff features and RRF

Merge sorted Parquet streams one query at a time. Deduplicate `(userId,movieId)`, retaining every per-source raw score/rank/presence/fallback flag. Missing scores stay NaN and missing ranks use K+1. Add only cutoff-frozen history/popularity/genre features. Future feedback is attached as separate label-side fields, never as features. RRF uses `sum(present/(60+rank))`.
'''),code('run-fusion','''
if RUN_MODE != "smoke_test":
    FUSION_OUTPUT = fusion.run()
    print("Saved fusion features:", FUSION_OUTPUT)
'''),md('diagnostics','''## 5. Diagnostics and ablations

Report candidate counts, overlap Jaccard, unique relevant contributions, union recall ceiling, coverage and popularity concentration. RRF and every remove-one-channel ablation use the same fixed query population, including no-hit queries. Pool recall can use up to 600 union candidates and is distinct from Top-100 ranking recall.
'''),code('fusion-metrics','''
import pandas as pd
from IPython.display import display
if FUSION_OUTPUT is not None:
    summary = json.loads((FUSION_OUTPUT / "metrics.json").read_text())
    display(pd.DataFrame([{**reports[0], "model": name} for name, reports in summary["metrics"].items()]))
    display(pd.Series(summary["overlap_jaccard"], name="Jaccard"))
    display(pd.Series(summary["unique_relevant_interactions"], name="unique positive contribution"))
    print("Union recall ceiling:", summary["union_recall_ceiling"])
    print("Candidate count quantiles:", summary["union_candidate_count_quantiles"])
    print("Resources:", summary["profile"])
'''),md('handoff','''## 6. Share artifacts with ranking

Pass the historical `FUSION_OUTPUT` and the full-Train Validation `FUSION_OUTPUT` separately to Notebook 09. Its training example builder samples unknown contrasts from historical candidates; it does not retrain the six channels. Fusion feature tables remain Parquet streams, not one all-user DataFrame.

Ratings are observational submissions. Catalog exclusions, rating bursts, exposure bias and missing-not-at-random feedback limit interpretation. Six channels need not beat a smaller subset; compare the saved ablations. No benchmark results are supplied in this notebook.
''')]
write('08_Retrieval_Fusion.ipynb',cells)

cells=[md('overview','''# MovieLens 32M — independent LightGBM ranking

Consume historical and full-Train Validation fusion artifacts. This notebook contains no retriever algorithms and never trains retriever snapshots. Generate those independently using `rank_snapshot` in notebooks 03–07 and fuse them in Notebook 08.
''')]+bootstrap('ranking',('smoke_test','full_train','load_evaluate'))
cells += [md('configuration','''## 2. Configuration and dependencies

`full_train` here fits LambdaMART from historical candidates, preserving the external partitions. `load_evaluate` loads an existing ranker. Historical features must come from strictly earlier retrievers, and all label timestamps must remain inside external Train. No in-sample full-Train substitute is accepted.
'''),code('ranker-config','''
import json
from recsys.rankers.workflow import RankingExperiment
CONFIG = json.loads((PROJECT_DIR / "configs" / "lightgbm.json").read_text())
CONFIG.update(seed=42, cpu_threads=4, bootstrap=300)
TRAIN_FUSION = None  # historical rank_train fusion directory
VALIDATION_FUSION = None  # full-Train Validation fusion directory
RANKER_CHECKPOINT = None  # required only for load_evaluate
RANKER_OUTPUT = None
'''),md('verify','''## 3. Verify frozen split, time and feature contracts

Check the split checksum, source/catalog/mapping provenance and fixed Validation population. Historical training cutoff must precede every ranking label; the label window must end before external Validation. Feature names are an explicit allow-list excluding labels, feedback kinds and future statistics.
'''),code('prepare-ranker','''
if RUN_MODE != "smoke_test":
    if VALIDATION_FUSION is None:
        raise ValueError("Set the saved full-Train Validation fusion path.")
    ranking = RankingExperiment(PROJECT_DIR, TRAIN_FUSION, VALIDATION_FUSION, CONFIG,
                                RUN_MODE, checkpoint=RANKER_CHECKPOINT)
'''),md('examples','''## 4. Leakage-aware ranking examples and query groups

Keep retrieved observed positives and observed low ratings, then sample a configurable number of unobserved candidates per query. Unknown zero labels are weak supervision, not verified negatives. Count queries with no retrieved positive and exclude them only from fitting; unconditional evaluation retains them. Query-contiguous Parquet batches are written without materializing the full feature table. Record counts, candidate recall ceiling, boundaries, sample-selection limitations and snapshot/full-model feature-distribution differences.
'''),code('prepare-examples','''
examples = None
if RUN_MODE != "smoke_test":
    examples = ranking.prepare_examples()
    if examples is not None:
        print(json.dumps(json.loads((examples / "sampling_report.json").read_text()), indent=2))
'''),md('fit','''## 5. LambdaMART training or checkpoint load

The migrated LambdaMART objective uses LightGBM's native `Dataset`/`Sequence` interface to stream raw Parquet features in bounded row groups. LightGBM still needs RAM for its compressed bins, labels, gradients and histograms, checked before fitting. The sklearn `LGBMRanker` helper remains available for small in-memory experiments. Both optimize `lambdarank`; the streaming API avoids assembling the complete raw table. A fixed Validation panel controls early stopping; Test is unopened.
'''),code('fit-ranker','''
if RUN_MODE != "smoke_test":
    RANKER_CHECKPOINT = ranking.fit_or_load(examples)
    print("Saved ranker:", RANKER_CHECKPOINT)
'''),md('validation','''## 6. Validation: RRF versus LightGBM

Evaluate every fixed eligible-positive user. Report unconditional and pool-hit-conditional Recall/NDCG/HitRate, coverage, popularity concentration and candidate recall ceiling. Inference ranks the entire unsampled union. A ranker cannot recover positives absent from the pool.
'''),code('ranker-metrics','''
import pandas as pd
from IPython.display import display
if RUN_MODE != "smoke_test":
    RANKER_OUTPUT = ranking.evaluate()
    summary = json.loads((RANKER_OUTPUT / "metrics.json").read_text())
    display(pd.DataFrame([{**reports[0], "model": name} for name, reports in summary["metrics"].items()]))
    display(pd.DataFrame(summary["conditional_on_pool_hit"]["lightgbm"]))
    print("Validation result:", RANKER_OUTPUT)
'''),md('importance','''## 7. Feature importance, resources and persistence

Gain/split importance is descriptive, not causal. Checkpoints contain the model, feature contract, source hashes, input-artifact references, sampling report, history, versions and measured resources. Complete matching stages are reused; no previous run is overwritten.
'''),code('feature-importance','''
if RUN_MODE != "smoke_test":
    importance = pd.read_parquet(RANKER_CHECKPOINT / "feature_importance.parquet")
    display(importance.sort_values("gain", ascending=False))
    print(json.loads((RANKER_CHECKPOINT / "training_history.json").read_text())["profile"])
'''),md('limitations','''## 8. Limitations and handoff

Sampling changes the label balance; dropping no-positive training queries creates selection bias. Earlier snapshot embeddings, score scales, catalogs and histories differ from full-Train inference. The saved feature-shift diagnostics also reflect time/population changes and do not establish unbiased transfer. Exposure bias and rating bursts remain observational limitations. No Test tuning or real training was performed during framework construction.

Pass the selected ranker checkpoint, its Validation result and the matching retrieval/fusion paths to Notebook 10 after reviewing Validation results. No real ranking result is prefilled here.
''')]
write('09_LightGBM_Ranking.ipynb',cells)

cells=[md('overview','''# MovieLens 32M — frozen selection and final evaluation

Review saved Validation comparisons, freeze explicit artifact choices, then enable final Test inference only deliberately. This notebook never fits or retunes models. All histories stay frozen at external Train; Validation is not appended.
''')]+bootstrap('final',('smoke_test','load_evaluate'))
cells += [md('configuration','''## 2. Explicit selection inputs and Test switches

Use the exact paths printed by notebooks 03–09. `FREEZE_SELECTION` seals a Validation-only choice. Merely loading this notebook or choosing `load_evaluate` does not evaluate Test. Enabling Test also requires the exact sealed selection ID.
'''),code('selection-config','''
from recsys.contracts import CHANNELS
from recsys.final_evaluation import freeze_selection, load_selection, evaluate_selected
import json
VALIDATION_CANDIDATES = {source: None for source in CHANNELS}
VALIDATION_FUSION = None
RANKER_CHECKPOINT = None
RANKER_VALIDATION = None
FREEZE_SELECTION = False
SELECTION_PATH = None
RUN_FINAL_TEST = False
CONFIRM_SELECTION_ID = ""
FINAL_OUTPUT = None
'''),md('freeze','''## 3. Freeze matching Validation artifacts

All six sources must share the same split, full-Train cutoff, catalog, mappings, users/labels and budget. Fusion must reference those exact candidates, and the selected ranker must have used that fusion for Validation. No latest-run guessing or Test-based model selection occurs.
'''),code('freeze-selection','''
if RUN_MODE != "smoke_test" and FREEZE_SELECTION:
    if (any(p is None for p in VALIDATION_CANDIDATES.values()) or VALIDATION_FUSION is None or
        RANKER_CHECKPOINT is None or RANKER_VALIDATION is None):
        raise ValueError("Set all explicit Validation selection artifact paths.")
    SELECTION_PATH = freeze_selection(PROJECT_DIR, VALIDATION_CANDIDATES, VALIDATION_FUSION,
                                      RANKER_CHECKPOINT, RANKER_VALIDATION)
    print("Frozen selection:", SELECTION_PATH)
'''),md('review','''## 4. Load saved selections and previous Validation results

This step verifies immutable references and displays already computed results without retraining or reading Test labels. Models can be trained in different Colab runtimes. Their source hashes and contracts identify their provenance.
'''),code('review-validation','''
import pandas as pd
from IPython.display import display
if RUN_MODE != "smoke_test":
    if SELECTION_PATH is None:
        raise ValueError("Set an existing selection path, or freeze explicit Validation inputs above.")
    selection = load_selection(SELECTION_PATH)
    print("Selection ID:", selection["selection_id"])
    _fusion_path = Path(selection["validation_fusion"]["path"])
    _validation = json.loads((_fusion_path / "metrics.json").read_text())
    display(pd.DataFrame([{**reports[0], "model": name} for name, reports in _validation["metrics"].items()]))
    _ranker_result = Path(selection["ranker_validation"]["path"])
    display(pd.DataFrame(json.loads((_ranker_result / "metrics.json").read_text())["metrics"]["lightgbm"]))
'''),md('gate','''## 5. Explicitly gated Test inference

Only this step deserializes Test labels. It loads each selected full-Train checkpoint, writes standardized Test candidates, fuses them and scores the selected LambdaMART model. No fit or tuning call occurs. Completed Test results are reused after checksum verification. Source/data incompatibilities fail clearly.
'''),code('test-gate','''
if RUN_MODE != "smoke_test" and RUN_FINAL_TEST:
    FINAL_OUTPUT = evaluate_selected(PROJECT_DIR, SELECTION_PATH, enabled=True,
                                     confirmation=CONFIRM_SELECTION_ID)
    print("Final result:", FINAL_OUTPUT)
elif RUN_MODE != "smoke_test":
    print("Test is disabled; only saved Validation results were reviewed.")
'''),md('comparison','''## 6. Final comparisons, contributions and resources

Individual channels, candidate union ceiling, RRF, LambdaMART and every remove-one-channel ablation use the same frozen Test population. Report Recall@20/50/100, NDCG@20, HitRate, coverage, popularity concentration, user-bootstrap intervals, training time, sampled CPU RSS and GPU allocator peaks. Conditional ranking metrics are separately labeled.
'''),code('final-comparison','''
if FINAL_OUTPUT is not None:
    final = json.loads((FINAL_OUTPUT / "metrics.json").read_text())
    display(pd.DataFrame([{**reports[0], "model": name,
                           "catalog_coverage": final["catalog_coverage"][name],
                           "popular_top_decile_share": final["top_decile_popularity_share"][name]}
                          for name, reports in final["metrics"].items()]))
    display(pd.DataFrame(final["conditional_on_pool_hit"]["lightgbm"]))
    display(pd.Series(final["unique_relevant_interactions"], name="unique positives"))
    _resources = {name: {**info["training_profile"], "checkpoint_bytes": info["checkpoint_bytes"],
                         "retrieval_seconds": info["evaluation"]["retrieval_seconds"]}
                  for name, info in final["profiles"].items()}
    _resources["lightgbm"] = final["ranker_training"]["profile"]
    display(pd.DataFrame(_resources).T)
    print("Pool recall ceiling:", final["union_recall_ceiling"])
    print("Excluded positive labels:", final["eligibility"])
'''),md('limitations','''## 7. Interpretation and experiment record

Test scores are final estimates for the sealed selection, never tuning signals. Rating timestamps are submission times, not consumption or exposure. Missing exposure logs, unknown-pair supervision, MNAR feedback, bursts and temporal shifts prevent online or causal claims. Candidate pool recall limits ranking. Historical/full-Train feature mismatch is documented in the saved ranker sampling report. No real benchmark numbers were generated during this refactor.
''')]
write('10_Final_Evaluation.ipynb',cells)
