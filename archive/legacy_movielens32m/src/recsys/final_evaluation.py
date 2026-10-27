"""Frozen selection and explicitly gated Test inference. No fit functions are called here."""
from pathlib import Path
import json
import numpy as np
from .artifacts import read_artifact,artifact_reference,dump_json,iter_queries,source_hashes
from .contracts import CHANNELS,compatible,mapping_id,catalog_id
from .runtime import Store,fingerprint,digest,profile
from .data import FrozenData,EXPECTED_ROWS
from .evaluation import generate_candidates,MetricAccumulator
from .fusion.workflow import FusionExperiment
from .features import feature_columns


def freeze_selection(project, validation_candidates, validation_fusion, ranker_checkpoint, ranker_validation):
    manifests = {s:read_artifact(p) for s,p in validation_candidates.items()}
    reference = compatible(manifests)
    if reference['partition'] != 'validation' or reference['training_kind'] != 'full_train':
        raise ValueError('Selection requires complete full-Train Validation candidate outputs')
    fusion = read_artifact(validation_fusion)
    ranker = read_artifact(ranker_checkpoint)
    ranker_result = read_artifact(ranker_validation)
    for source,path in validation_candidates.items():
        if fusion['input_candidates'][source] != artifact_reference(path):
            raise ValueError('Fusion does not belong to the selected retrieval outputs')
    if ranker['validation_fusion'] != artifact_reference(validation_fusion):
        raise ValueError('Ranker was selected against different full-Train retrieval features')
    if ranker_result['run_id'] != ranker['run_id']:
        raise ValueError('Ranker Validation results belong to a different model')
    payload = dict(version='selection_v2',selection_basis='Validation only',test_used=False,
        retrieval={s:artifact_reference(p) for s,p in validation_candidates.items()},
        validation_fusion=artifact_reference(validation_fusion),ranker_checkpoint=artifact_reference(ranker_checkpoint),
        ranker_validation=artifact_reference(ranker_validation),contract={k:reference[k] for k in (
            'split_manifest_sha256','mapping_id','catalog_id','training_cutoff','evaluation_id')})
    selection_id = fingerprint(payload)[:20]
    payload['selection_id'] = selection_id
    store = Store(Path(project)/'results'/'comparisons',selection_id)
    def write(root): dump_json(root/'manifest.json',payload)
    return store.write('selection',write,payload)


def load_selection(path):
    selection = read_artifact(path)
    for reference in list(selection['retrieval'].values()) + [selection['validation_fusion'],
                     selection['ranker_checkpoint'],selection['ranker_validation']]:
        if digest(Path(reference['path'])/'manifest.json') != reference['manifest_sha256']:
            raise ValueError('Frozen selection reference checksum mismatch')
        read_artifact(reference['path'])
    return selection


def evaluate_selected(project, selection_path, *, enabled=False, confirmation='', expected=EXPECTED_ROWS):
    selection = load_selection(selection_path)
    if not enabled or confirmation != selection['selection_id']:
        raise PermissionError('Test disabled: explicitly enable and confirm the frozen selection ID')
    results = Store(Path(project)/'results'/'comparisons',selection['selection_id'])
    completed = results.load('final_test')
    if completed: return completed
    data = FrozenData(project,expected).prepare().open_test(explicitly_enabled=True)
    for key,value in dict(split_manifest_sha256=data.split_sha,mapping_id=mapping_id(data.ctx),
                          catalog_id=catalog_id(data.ctx),training_cutoff=data.ctx.cutoff).items():
        if selection['contract'][key] != value:
            raise ValueError(f'Selected inference history incompatible with {key}')
    from .training import load_model
    candidate_paths,profiles = {},{}
    fusion_config = read_artifact(selection['validation_fusion']['path'])['config']
    for source,reference in selection['retrieval'].items():
        candidate_manifest = read_artifact(reference['path'])
        cp = Path(candidate_manifest['checkpoint']['path'])
        checkpoint = read_artifact(cp)
        if checkpoint['source_hashes'] != source_hashes(source):
            raise ValueError('Selected source version differs; restore the original readable source before Test')
        config = checkpoint['config']
        store = Store(Path(project)/'results'/'test'/source,selection['selection_id'])
        def write(root):
            device = 'cpu'
            if source in ('lightgcn','two_tower'):
                try:
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                except ImportError:
                    pass
            model = load_model(data.ctx,source,cp,device)
            model.name = source
            with profile() as timing:
                summary = generate_candidates(model,data,config,root)
            summary['profile'] = timing
            dump_json(root/'metrics.json',summary)
            meta = dict(candidate_manifest,partition='test',evaluation_id=data.population.identifier,
                label_start=data.label_start,label_end=data.label_end,selection_id=selection['selection_id'],
                artifact_checksum=digest(root/'candidates.parquet'))
            dump_json(root/'manifest.json',meta)
        candidate_paths[source] = store.write('candidates',write,checkpoint)
        training = json.loads((cp/'training_history.json').read_text())
        profiles[source] = dict(training_profile=training['profile'],
                               checkpoint_bytes=sum(p.stat().st_size for p in cp.rglob('*') if p.is_file()),
                               evaluation=json.loads((candidate_paths[source]/'metrics.json').read_text()))
    fusion = FusionExperiment(project,candidate_paths,fusion_config,expected,opened_test_data=data).run()
    from lightgbm import Booster
    ranker_path = Path(selection['ranker_checkpoint']['path'])
    ranker_meta = read_artifact(ranker_path)
    if ranker_meta['source_hashes'] != source_hashes('lightgbm'):
        raise ValueError('Selected ranker source version differs')
    ranker = Booster(model_file=str(ranker_path/'model.txt'))
    def write(root):
        metrics = MetricAccumulator(data.ctx,data.population,fusion_config)
        seen = set()
        with profile() as timing:
            for user,frame in iter_queries(fusion/'features.parquet'):
                seen.add(user)
                scores = ranker.predict(frame[feature_columns()])
                if not np.isfinite(scores).all(): raise ValueError('Nonfinite ranker scores')
                ordered = frame.assign(ranking_score=scores).sort_values(['ranking_score','movieId'],ascending=[False,True])
                metrics.add('lightgbm',user,ordered.movieId.tolist())
            for user in set(data.population.truth)-seen: metrics.add('lightgbm',user,[])
            ranker_summary = metrics.finish(root)
        comparison = json.loads((fusion/'metrics.json').read_text())
        for key in ['metrics','conditional_on_pool_hit','catalog_coverage','top_decile_popularity_share']:
            comparison[key].update(ranker_summary[key])
        sampling = json.loads((ranker_path/'sampling_report.json').read_text())
        snapshot_summary = {r['feature']:r['snapshot'] for r in sampling['feature_distribution_mismatch']}
        from .rankers.features import mismatch
        comparison.update(selection_id=selection['selection_id'],profiles=profiles,ranker_inference_profile=timing,
            test_feature_distribution_mismatch=mismatch(snapshot_summary,comparison['feature_summary']),
            ranker_training=json.loads((ranker_path/'training_history.json').read_text()),
            feature_shift_limitation='Historical snapshot features and full-Train inference features may differ; '
                                     'Validation feature-shift diagnostics are preserved in the ranker sampling report.')
        dump_json(root/'metrics.json',comparison)
        dump_json(root/'manifest.json',selection)
    return results.write('final_test',write,selection)
