#!/usr/bin/env python3

import numpy as np
import sys
from JetCalc.ExpCut import cuts as JEC

species = {
                'light':
                        [('pion', 211),
                        ('kaon', 321),
                        ('proton', 2212),
                        ('Lambda', 3122),
                        ('Sigma0', 3212),
                        ('Xi', 3312),
                        ('Omega', 3334)],
                'heavy':
                        [('D+', 411),
                        ('D0', 421),
                        ('D*+', 10411),
                        ('D0*', 10421)],
                        }
# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'
# results "array" (one element)
# to be overwritten for each event
dtype=[
        ########### Initial condition #####################
        ('initial_entropy', float_t),
        ('Npart', float_t),
        ('Ncoll', float_t),
        ########### Soft part #############################
        ('nsamples', int_t),
        ('dNch_deta', float_t),
        ('dET_deta', float_t),
        ('dN_dy', [(s, float_t) for (s, _) in species.get('light')]),
        ('mean_pT', [(s, float_t) for (s, _) in species.get('light')]),
        ('pT_fluct', [('N', int_t), ('sum_pT', float_t), ('sum_pTsq', float_t)]),
        ('Qn_soft', [('M', int_t), ('Qn', complex_t, 8)]),
        ########### Hard part #############################
        ('dX_dpT_dy_pred', [(s, float_t, JEC['pred-pT'].shape[0])
                                                for (s, _) in species.get('heavy')]),
        ('dX_dpT_dy_ALICE', [(s, float_t, JEC['ALICE']['Raa']['pTbins'].shape[0])
                                                for (s, _) in species.get('heavy')]),
        ('dX_dpT_dy_CMS', [(s, float_t, JEC['CMS']['Raa']['pTbins'].shape[0])
                                                for (s, _) in species.get('heavy')]),
        ('Qn_poi_pred', [(s, [('M', int_t, JEC['pred-pT'].shape[0]),
                                                 ('Qn', complex_t, [JEC['pred-pT'].shape[0], 4])] )
                                                for (s, _) in species.get('heavy')]),
        ('Qn_ref_pred', [('M', int_t), ('Qn', complex_t, 3)]),
        ('Qn_poi_ALICE', [(s, [('M', int_t, JEC['ALICE']['vn_HF']['pTbins'].shape[0]),
                        ('Qn', complex_t, [JEC['ALICE']['vn_HF']['pTbins'].shape[0], 4])] )
                                                for (s, _) in species.get('heavy')]),
        ('Qn_ref_ALICE', [('M', int_t), ('Qn', complex_t, 3)]),
        ('Qn_poi_CMS', [(s, [('M', int_t, JEC['CMS']['vn_HF']['pTbins'].shape[0]),
                        ('Qn', complex_t, [JEC['CMS']['vn_HF']['pTbins'].shape[0], 4])] )
                                                for (s, _) in species.get('heavy')]),
        ('Qn_ref_CMS', [('M', int_t), ('Qn', complex_t, 3)]),
]



def main():

    res = np.fromfile(sys.argv[1],dtype=dtype)
    print(res.shape)
    res = np.array(sorted(res, key=lambda x: x['dET_deta'], reverse=True))

    N = res.shape[0]
    print(res['Qn_poi_CMS'])

if __name__ == "__main__":
    main()
