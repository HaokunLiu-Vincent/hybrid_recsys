"""Invoke targeted, readable synthetic tests from independent notebooks."""
from pathlib import Path
import sys
import unittest

COMMON = ['test_modular.ModularContracts.test_candidate_contract_and_top_k_upper_bound',
          'test_modular.ModularContracts.test_development_does_not_open_test_labels',
          'test_modular.ModularContracts.test_source_hashes_are_model_specific']
TARGETS = {
    'baselines': COMMON+['test_modular.ModularContracts.test_independent_baseline_checkpoint_and_result_compatibility'],
    'itemcf': COMMON+['test_recsys.CoreTests.test_itemcf_exact_cosine_full_history'],
    'implicit_als': COMMON+['test_recsys.CoreTests.test_optimized_als'],
    'lightgcn': COMMON+['test_recsys.CoreTests.test_lightgcn_exact_adjoint_gradient',
                       'test_recsys.CoreTests.test_neural_training_cpu_synthetic'],
    'two_tower': COMMON+['test_recsys.CoreTests.test_neural_training_cpu_synthetic'],
    'fusion': COMMON+['test_recsys.CoreTests.test_fusion_feature_provenance_and_weak_labels'],
    'ranking': COMMON+['test_modular.ModularContracts.test_parquet_sequence_preserves_queries_and_rows',
                      'test_modular.ModularContracts.test_streaming_lightgbm_sequence_fit',
                      'test_recsys.CoreTests.test_lightgbm_fit_predict'],
    'final': COMMON+['test_modular.ModularContracts.test_six_independent_runs_fusion_and_historical_examples'],
}


def run_smoke(project, target):
    # Tiny CPU sparse tests use a single thread; this is not a benchmark setting.
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(1)
    tests = Path(project)/'tests'
    if not (tests/'test_modular.py').is_file():
        raise FileNotFoundError(f'Copy the readable tests/ directory to {tests}')
    sys.path.insert(0,str(tests))
    suite = unittest.defaultTestLoader.loadTestsFromNames(TARGETS[target])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise RuntimeError('Synthetic tests failed; inspect the errors')
    if result.skipped:
        raise RuntimeError(f'Required smoke integrations unavailable: {result.skipped}. '
                           'Install the documented dependencies after approval, then retry.')
    return dict(tests=result.testsRun,failures=0,skipped=0,scope='synthetic only; no MovieLens training')
