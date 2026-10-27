"""Synthetic raw/schema/ZIP/provenance checks; no network or model fitting."""
from pathlib import Path
import json
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd
from recsys.small_data import (DATASET_ID, DataValidationError, load_raw, validate_frames,
                               safe_extract, required_root, publish_archive, verify_provenance, download_dataset, sha256)
from recsys.small_eda import analysis_connection,window_diagnostics,split_feasibility


def write_fixture(root):
    root.mkdir(parents=True,exist_ok=True)
    ratings=pd.DataFrame({'userId':[1,1,2,2,3,3], 'movieId':[10,11,10,12,11,12],
                          'rating':[4.,2.,5.,4.,3.,4.5], 'timestamp':[10,20,20,30,40,50]})
    movies=pd.DataFrame({'movieId':[10,11,12,13], 'title':['One','Two','Three','Unrated'],
                         'genres':['Drama','Comedy','Drama|Comedy','(no genres listed)']})
    ratings.to_csv(root/'ratings.csv',index=False);movies.to_csv(root/'movies.csv',index=False)
    (root/'README.md').write_text('6 ratings across 4 movies. These data were created by 3 users.\nLicense: synthetic test fixture only.\n')
    return ratings,movies


def archive_fixture(root,archive):
    with zipfile.ZipFile(archive,'w') as z:
        for p in root.iterdir():z.write(p,'nested/ml-latest-small/'+p.name)


class SmallRawTests(unittest.TestCase):
    def test_valid_schema_and_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_fixture(root)
            ratings,movies,quality=load_raw(root,verify=False)
            self.assertEqual(quality['ratings'],6);self.assertEqual(quality['users'],3)
            self.assertEqual(quality['metadata_movies_without_ratings'],1)
            self.assertEqual(str(ratings.rating.dtype),'float32')
            self.assertEqual(str(ratings.userId.dtype),'int32')

    def test_no_integer_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ratings,movies=write_fixture(root)
            ratings.userId=ratings.userId.astype('int64')+2**32
            ratings.to_csv(root/'ratings.csv',index=False)
            observed,_,_=load_raw(root,verify=False)
            self.assertEqual(int(observed.userId.min()),2**32+1)
            self.assertEqual(str(observed.userId.dtype),'int64')

    def test_missing_and_wrong_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError):load_raw(root,verify=False)
            write_fixture(root)
            (root/'ratings.csv').write_text('userId,movieId,rating\n1,10,4\n')
            with self.assertRaises(DataValidationError):load_raw(root,verify=False)

    def test_invalid_ratings_nulls_duplicates_and_unknown_movies(self):
        with tempfile.TemporaryDirectory() as tmp:
            ratings,movies=write_fixture(Path(tmp))
            bad=ratings.copy();bad.loc[0,'rating']=3.7
            with self.assertRaises(DataValidationError):validate_frames(bad,movies)
            bad=ratings.copy();bad.loc[0,'rating']=np.nan
            with self.assertRaises(DataValidationError):validate_frames(bad,movies)
            with self.assertRaises(DataValidationError):validate_frames(pd.concat([ratings,ratings.iloc[:1]]),movies)
            bad=ratings.copy();bad.loc[0,'movieId']=99
            with self.assertRaises(DataValidationError):validate_frames(bad,movies)
            bad=ratings.copy();bad.loc[0,'timestamp']=-1
            with self.assertRaises(DataValidationError):validate_frames(bad,movies)

    def test_safe_zip_rejects_traversal_symlink_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,member in enumerate(['../escape.csv','/absolute.csv','back\\slash.csv']):
                zpath=root/f'{i}.zip'
                with zipfile.ZipFile(zpath,'w') as z:z.writestr(member,'not data')
                with self.assertRaises(DataValidationError):safe_extract(zpath,root/f'extract{i}')
            zpath=root/'link.zip'
            with zipfile.ZipFile(zpath,'w') as z:
                entry=zipfile.ZipInfo('link');entry.create_system=3;entry.external_attr=(0o120777<<16)
                z.writestr(entry,'outside')
            with self.assertRaises(DataValidationError):safe_extract(zpath,root/'link-out')
            zpath=root/'duplicate.zip'
            with zipfile.ZipFile(zpath,'w') as z:
                z.writestr('ratings.csv','one');z.writestr('ratings.csv','two')
            with self.assertRaises(DataValidationError):safe_extract(zpath,root/'duplicate-out')

    def test_ambiguous_layout_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_fixture(root/'one');write_fixture(root/'two')
            with self.assertRaises(DataValidationError):required_root(root)

    def test_publication_provenance_and_existing_dataset_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_fixture(root/'files');archive_fixture(root/'files',root/'download.zip')
            target=root/'raw'
            manifest=publish_archive(root/'download.zip',target,{'downloaded_at_utc':'2026-01-01T00:00:00+00:00'})
            self.assertEqual(manifest['source']['dataset_id'],DATASET_ID)
            self.assertEqual(manifest['discrepancies'],{})
            self.assertEqual(verify_provenance(target)['quality']['ratings'],6)
            original=sha256(target/'ratings.csv')
            self.assertEqual(download_dataset(target,'no-cli-should-be-called')['quality']['ratings'],6)
            with self.assertRaises(FileExistsError):publish_archive(root/'download.zip',target,{})
            self.assertEqual(sha256(target/'ratings.csv'),original)
            (target/'ratings.csv').write_text('changed')
            with self.assertRaises(DataValidationError):verify_provenance(target)

    def test_partial_dataset_not_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_fixture(root)
            with self.assertRaises(FileNotFoundError):download_dataset(root,'no-cli-should-be-called')

    def test_duckdb_temporal_ties_and_cold_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_fixture(root/'files');archive_fixture(root/'files',root/'download.zip')
            publish_archive(root/'download.zip',root/'raw',{})
            con,_=analysis_connection(root/'raw')
            before={p.name:sha256(p) for p in (root/'raw').glob('*.csv')}
            summary,users=window_diagnostics(con,20,20,40)
            self.assertEqual(summary['history_rows'],1)
            self.assertEqual(summary['future_rows'],3)  # both timestamp=20 events remain in labels
            self.assertEqual(summary['users_without_train_ratings'],1)
            self.assertEqual(summary['eligible_positive'],1)
            options=[dict(alternative='fixture',validation_start=20,test_start=40,end_exclusive=51)]
            table,cohorts,internal=split_feasibility(con,options)
            self.assertEqual(int(table.future_rows.sum()+table.history_rows.iloc[0]),6)
            self.assertFalse((root/'raw'/'train.parquet').exists())
            self.assertEqual(before,{p.name:sha256(p) for p in (root/'raw').glob('*.csv')})
            con.close()


if __name__=='__main__':unittest.main(verbosity=2)
