"""RRF known answers and fixed-population metrics on synthetic fixtures."""
import unittest
import numpy as np
import pandas as pd
from helpers import fixture
from recsys.data import build_context
from recsys.contracts import Population
from recsys.fusion.small_rrf import union_from_frames,validate_pool,ranked,evaluate_rankings,overlap_diagnostics


def stream(source,user,ids,scores=None):
    return pd.DataFrame([dict(userId=user,movieId=i,score=float(scores[j] if scores else 100-j),
                              rank=j+1,source=source,fallback=False) for j,i in enumerate(ids)])


class RRFTests(unittest.TestCase):
    def setUp(self):
        t,m,_=fixture();self.ctx=build_context(t,m)
        self.ids=sorted(set(self.ctx.items[self.ctx.eligible])-set(self.ctx.items[self.ctx.seen(0)]))[:4]

    def test_exact_rrf_membership_dedup_ties_and_no_score_summing(self):
        a,b,c=self.ids[:3]
        frames={'a':stream('a',1,[a,b],[900,1]),'b':stream('b',1,[b,a],[30,-1])}
        pool=union_from_frames(frames,self.ctx,[1])
        self.assertEqual(len(pool),2)
        self.assertEqual(pool.movieId.tolist(),sorted([a,b]))
        np.testing.assert_allclose(pool.rrf_score,1/61+1/62)
        self.assertTrue(pool.source_count.eq(2).all())
        self.assertEqual(pool.index.tolist(),list(range(len(pool))))
        self.assertEqual(pool.set_index('movieId').loc[a,'a_score'],900)

    def test_missing_source_nan_and_rank_sentinel(self):
        a,b,c=self.ids[:3]
        pool=union_from_frames({'a':stream('a',1,[a,b]),'b':stream('b',1,[c])},self.ctx,[1])
        row=pool.set_index('movieId').loc[a]
        self.assertTrue(np.isnan(row.b_score));self.assertEqual(row.b_rank,101)
        self.assertEqual(row.b_present,0);self.assertEqual(row.rrf_score,1/61)

    def test_bad_seen_or_duplicate_candidate_is_rejected(self):
        observed=int(self.ctx.items[self.ctx.seen(0)][0])
        with self.assertRaises(ValueError):union_from_frames({'a':stream('a',1,[observed])},self.ctx,[1])
        with self.assertRaises(ValueError):union_from_frames({'a':stream('a',1,[self.ids[0],self.ids[0]])},self.ctx,[1])

    def test_overlap_jaccard_and_unique_positive_contributions(self):
        frames={'a':stream('a',1,[10,20]),'b':stream('b',1,[20,30]),'c':stream('c',1,[40])}
        d=overlap_diagnostics(frames,{1:{10,20,50}})
        self.assertEqual(d['union_pairs'],4)
        self.assertAlmostEqual(d['pairwise']['a|b']['mean_query_jaccard'],1/3)
        self.assertEqual(d['unique_contribution']['a']['relevant_pairs'],1)
        self.assertEqual(d['retrieved_relevant_pairs'],2)

    def test_pool_ceiling_separate_from_top_k_and_zero_hit_queries_retained(self):
        ids=list(range(1000,1120))
        frame=pd.DataFrame(dict(userId=[1]*120,movieId=ids,rrf_score=np.arange(120,0,-1,dtype=float)))
        pop=Population({1:{1119},2:{9999}},{1:'warm_20_plus',2:'sparse_1_4'},{},'synthetic')
        per,metrics=evaluate_rankings(frame,pop,self.ctx)
        self.assertEqual(metrics[0]['queries'],2)
        self.assertEqual(metrics[0]['recall100'],0)
        self.assertEqual(metrics[0]['pool_recall'],.5)
        self.assertEqual(metrics[0]['empty_candidate_queries'],1)
        self.assertIsNone(next(m for m in metrics if m['cohort']=='cold_0')['ndcg20'])


if __name__=='__main__':unittest.main(verbosity=2)
