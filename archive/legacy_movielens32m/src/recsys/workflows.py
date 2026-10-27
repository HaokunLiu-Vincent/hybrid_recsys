"""Independent retrieval notebook workflow. Nothing trains at import or construction."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from .artifacts import source_hashes, dump_json, read_artifact, artifact_reference
from .contracts import FRAMEWORK_VERSION, CHANNELS, mapping_id, catalog_id
from .data import FrozenData, EXPECTED_ROWS
from .evaluation import generate_candidates
from .runtime import Store, digest, fingerprint, profile, environment, require_memory
from .training import fit_retriever, load_model, validation_selector


class RetrievalExperiment:
    MODES = ('full_train','load_evaluate','rank_snapshot')

    def __init__(self, project, model_name, config, mode='full_train', checkpoint=None, device='cpu',
                 expected=EXPECTED_ROWS):
        if mode not in self.MODES or model_name not in CHANNELS:
            raise ValueError('Unsupported model/mode; smoke_test runs synthetic tests without constructing an experiment')
        self.project, self.name = Path(project), model_name
        self.config, self.mode, self.device = deepcopy(config), mode, device
        self.checkpoint_path = Path(checkpoint) if checkpoint else None
        self.expected, self.model = expected, None
        self.training_profile, self.logs = {}, []

    def prepare(self, shared_data=None):
        if self.mode == 'load_evaluate':
            if self.checkpoint_path is None:
                raise ValueError('load_evaluate requires an explicit checkpoint path; latest is never guessed')
            saved = read_artifact(self.checkpoint_path)
            self.kind = saved['training_kind']
            cutoff, end = saved.get('snapshot_cutoff_utc'), saved.get('snapshot_label_end_utc')
        else:
            self.kind = 'historical_snapshot' if self.mode == 'rank_snapshot' else 'full_train'
            cutoff = self.config.get('snapshot_cutoff_utc') if self.kind == 'historical_snapshot' else None
            end = self.config.get('snapshot_label_end_utc') if self.kind == 'historical_snapshot' else None
        if shared_data is None:
            self.data = FrozenData(self.project, self.expected).prepare(self.kind, cutoff, end)
        else:
            self.data = shared_data
            if (self.data.training_kind != self.kind or self.data.partition == 'test' or
                self.data.project.resolve() != self.project.resolve()):
                raise ValueError('Shared preparation has a different history/label boundary')
            if self.kind == 'historical_snapshot':
                import pandas as pd
                requested = int(pd.Timestamp(cutoff).timestamp())+1
                end_timestamp = int(pd.Timestamp(end).timestamp()) if end else self.data.manifest['boundaries_utc_and_unix']['validation_start_timestamp']
                if requested != self.data.label_start or end_timestamp != self.data.label_end:
                    raise ValueError('Shared historical cutoff/label-end mismatch')
        if self.kind == 'full_train' and self.data.ctx.train_rows != self.data.manifest['row_counts']['train']:
            raise ValueError('Full-Train mode may not reduce Train rows')
        if self.expected == EXPECTED_ROWS and self.config['retrieval_k'] != 100:
            raise ValueError('Primary experiment requires Top-100 from every channel')
        key = 'als' if self.name == 'implicit_als' else self.name
        hashes = source_hashes(self.name)
        self.metadata = dict(framework_version=FRAMEWORK_VERSION, model_name=self.name,
            model_version=FRAMEWORK_VERSION, source_hashes=hashes, split_version='v1',
            split_manifest_sha256=self.data.split_sha, training_kind=self.kind,
            training_cutoff=self.data.ctx.cutoff, training_rows=self.data.ctx.train_rows,
            label_start=self.data.label_start, label_end=self.data.label_end,
            catalog_policy='B', catalog_id=catalog_id(self.data.ctx), mapping_id=mapping_id(self.data.ctx),
            evaluation_id=self.data.population.identifier, top_k=self.config['retrieval_k'],positive_threshold=4.,
            random_seed=self.config['seed'], hyperparameters=self.config.get(key,{}), config=self.config,
            partition=self.data.partition, snapshot_cutoff_utc=cutoff, snapshot_label_end_utc=end)
        self.run_id = fingerprint(self.metadata)[:20]
        if self.mode == 'load_evaluate':
            # Inference may change CPU/GPU device, but never data, code, parameters or the query contract.
            for key in ['model_name','source_hashes','split_manifest_sha256','training_kind','training_cutoff',
                        'mapping_id','catalog_id','config','evaluation_id']:
                if saved[key] != self.metadata[key]:
                    raise ValueError(f'Checkpoint incompatible with current {key}; no silent migration')
            self.run_id = saved['run_id']
        self.metadata['run_id'] = self.run_id
        self.store = Store(self.project/'artifacts'/self.name,self.run_id)
        self.results = Store(self.project/'results'/self.data.partition/self.name,self.run_id)
        print(self.name, self.kind, self.run_id, 'training rows:', self.data.ctx.train_rows)
        return self.data.audit

    def initialize(self):
        """No fitting: print the one model factory, training boundary and device."""
        if not hasattr(self,'data'):
            raise RuntimeError('Call prepare() first')
        return dict(model=self.name,device=self.device,training_cutoff=self.data.ctx.cutoff,
                    train_rows=self.data.ctx.train_rows,positive_edges=self.data.ctx.positive.nnz,
                    catalog_items=int(self.data.ctx.eligible.sum()))

    def train(self):
        if self.mode == 'load_evaluate':
            saved = self.checkpoint_path
        else:
            saved = self.store.load('checkpoint')
        if saved is not None:
            read_artifact(saved)
            self.model = load_model(self.data.ctx,self.name,saved,self.device)
            self.model.name = self.name
            info = json.loads((saved/'training_history.json').read_text())
            self.training_profile, self.logs = info['profile'], info['epochs']
            return 'loaded verified checkpoint; no training'
        if self.mode == 'load_evaluate':
            raise FileNotFoundError('Missing checkpoint; load_evaluate never trains')
        selector = None
        panel = []
        if self.kind == 'full_train' and self.name in ('lightgcn','two_tower'):
            selector,panel = validation_selector(self.data.population.truth,self.config)
        def checkpoint(epoch,model,logs,state):
            import torch
            def write(p):
                model.save(p)
                torch.save(state,p/'training_state.pt')
                dump_json(p/'epochs.json',logs)
            self.store.write(f'epochs/{epoch:03d}',write,self.metadata)
        with profile() as timing:
            self.model = fit_retriever(self.data.ctx,self.name,self.config,selector,checkpoint,self.device)
        self.training_profile, self.logs = timing, self.model.logs
        self.panel = list(map(int,panel))
        return 'trained exactly one requested retriever'

    def save_checkpoint(self):
        if self.model is None:
            raise RuntimeError('Train or load the model before saving')
        if self.mode == 'load_evaluate':
            return self.checkpoint_path
        def write(p):
            self.model.save(p)
            self.data.ctx.save(p)
            dump_json(p/'config.json',self.config)
            dump_json(p/'manifest.json',self.metadata)
            dump_json(p/'training_history.json',dict(epochs=self.logs,profile=self.training_profile,
                environment=environment('/tmp'),validation_tuning_panel=getattr(self,'panel',[]),
                selection='Validation Recall@100' if self.kind == 'full_train' else 'fixed epochs; training loss only'))
        self.checkpoint_path = self.store.write('checkpoint',write,self.metadata)
        return self.checkpoint_path

    def evaluate(self):
        if self.model is None or self.checkpoint_path is None:
            raise RuntimeError('Train/load and save a checkpoint first')
        old = self.results.load('candidates')
        if old is not None:
            read_artifact(old)
            return old
        require_memory(disk=len(self.data.population.users)*self.config['retrieval_k']*64,
                       root=self.results.root,stage='candidate Drive persistence reserve')
        def write(p):
            with profile() as timing:
                summary = generate_candidates(self.model,self.data,self.config,p)
            summary.update(profile=timing,training_profile=self.training_profile)
            dump_json(p/'metrics.json',summary)
            dump_json(p/'manifest.json',dict(self.metadata,checkpoint=artifact_reference(self.checkpoint_path),
                artifact_checksum=digest(p/'candidates.parquet'),candidate_rows=summary['candidate_rows']))
            users = self.data.population.users
            import pandas as pd
            pd.DataFrame(dict(userId=users,cohort=[self.data.population.cohorts[int(u)] for u in users])).to_parquet(
                p/'population.parquet',index=False)
        path = self.results.write('candidates',write,self.metadata)
        self.store.json('candidate_reference_'+self.data.partition,dict(
            candidates=artifact_reference(path),metrics=json.loads((path/'metrics.json').read_text())))
        return path
