"""Data provenance, temporal isolation and evaluation regression tests."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG
from data_loader import get_data_splits, load_dataset_matrix, StandardScaler
from eval import calculate_metrics_raw_scale, compute_seasonal_naive_denominators
from prepare_data import read_tsf

class ProtocolTests(unittest.TestCase):
    def test_provenance_required_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as d, patch('data_loader.DATA_DIR',d):
            with self.assertRaises(FileNotFoundError): load_dataset_matrix(ELECTRICITY_CONFIG)
            p=Path(d)/'Electricity_16ch.npy'; np.save(p,np.zeros((5,16)))
            (Path(d)/'Electricity_manifest.json').write_text(json.dumps({
                'source_kind':'Monash original release','channel_ids':ELECTRICITY_CONFIG.channel_ids,
                'cache_sha256':'deliberately_invalid'}))
            with self.assertRaisesRegex(ValueError,'checksum'): load_dataset_matrix(ELECTRICITY_CONFIG)

    def test_split_windows_and_fraction_scaler(self):
        for cfg in (ELECTRICITY_CONFIG,SOLAR_CONFIG):
            for fraction in (1.,.25):
                ls,s,raw,(start,b1,b2)=get_data_splits(cfg,train_data_fraction=fraction,batch_size=7)
                self.assertEqual(b2,int(.8*len(raw)))
                for split,lo,hi in [('train',start,b1),('val',b1,b2),('test',b2,len(raw))]:
                    t=ls[split+'_starts']; self.assertTrue(np.all(t+cfg.pred_len<=hi))
                    self.assertTrue(np.all(t>=lo))
                self.assertGreaterEqual(ls['train_starts'][0]-cfg.seq_len,start)
                np.testing.assert_allclose(s.mean,raw[start:b1].mean(0,keepdims=True))
                self.assertEqual(next(iter(ls['train']))[0].shape[0],7)
                # Deliberately corrupt unavailable data: fitting must be unchanged.
                bad=raw.copy(); bad[:start]+=1e7; bad[b1:]-=1e7
                fit=StandardScaler().fit(bad[start:b1]); np.testing.assert_array_equal(s.mean,fit.mean)

    def test_common_origins_and_capacity(self):
        for cfg in (ELECTRICITY_CONFIG,SOLAR_CONFIG):
            maximum=7*cfg.seasonal_m; reference=None
            for length in (cfg.seasonal_m,cfg.seq_len,maximum):
                ls,*_=get_data_splits(cfg,custom_seq_len=length,common_start_idx=maximum)
                values={s:ls[s+'_starts'] for s in ('train','val','test')}
                if reference:
                    for s in values: np.testing.assert_array_equal(reference[s],values[s])
                reference=values

    def test_shuffle_restarts_from_same_seed(self):
        a,*_=get_data_splits(ELECTRICITY_CONFIG,seed=43)
        b,*_=get_data_splits(ELECTRICITY_CONFIG,seed=43)
        np.testing.assert_array_equal(next(iter(a['train']))[0],next(iter(b['train']))[0])

    def test_mase_is_mean_of_channel_ratios(self):
        pred=np.array([[[1.,100.]]]); true=np.zeros_like(pred)
        m=calculate_metrics_raw_scale(pred,true,np.array([1.,100.]))
        self.assertAlmostEqual(m['mase'],1.)
        self.assertAlmostEqual(m['rmse'],np.sqrt((1+10000)/2))

    def test_invalid_fractions_and_context_rejected(self):
        for fraction in (0.,-1.,1.1):
            with self.assertRaises(ValueError): get_data_splits(ELECTRICITY_CONFIG,train_data_fraction=fraction)
        with self.assertRaises(ValueError): get_data_splits(ELECTRICITY_CONFIG,custom_seq_len=168,common_start_idx=48)

if __name__=='__main__': unittest.main()
