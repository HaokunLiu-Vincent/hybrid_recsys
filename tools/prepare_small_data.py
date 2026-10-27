"""Three explicit local DVC stage entry points; no data download or model fitting."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from recsys.small_pipeline import (ROOT, settings, raw_connection, prepare,
                                  validate_splits, write_immutable_json)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['validate-raw','prepare','validate-splits'])
    parser.add_argument('--config',type=Path,default=ROOT/'params.yaml')
    parser.add_argument('--output-root',type=Path,default=ROOT/'data/splits')
    args=parser.parse_args();config=settings(args.config)
    raw=ROOT/config['dataset']['raw_dir']
    if args.stage=='validate-raw':
        con,report=raw_connection(raw,config['dataset']);con.close()
        write_immutable_json(ROOT/'data/validation/small_raw_v1.json',report)
        print('Raw hashes, DuckDB schemas and quality verified:',report['observed'])
    elif args.stage=='prepare':
        results,_=prepare(raw,args.output_root,config,ROOT/'data/diagnostics/small_protocol_v1.json')
        for version,manifest in results.items():
            print(version,[(r['partition'],r['interactions']) for r in manifest['partition_statistics']])
    else:
        reports=[validate_splits(raw,args.output_root/config['protocol'][key],config)
                 for key in ['version','secondary_version']]
        write_immutable_json(ROOT/'data/validation/small_splits_v1.json',reports)
        print('Frozen split files, source coverage, histories, catalog/cohorts and inner windows verified:',
              [(r['version'],r['files_verified']) for r in reports])


if __name__=='__main__':main()
