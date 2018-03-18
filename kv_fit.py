# coding: utf-8

"""Entry point for estimation.

"""
import os
import sys
import json
import pickle
import pandas as pd

from kv_estimator import KleinVellaDoubleIndex


def ordered_series(data_dict, index_order):
    s0 = set(data_dict.keys())
    s1 = set(index_order)
    assert s0 == s1, '\n\nStart value keys != index variables:\n{}\n  !=\n{}\n\n'.format(
        s0.difference(s1), s1.difference(s0)
    )
    return pd.Series(data_dict, index=index_order, dtype=float)


if __name__ == '__main__':

    # Check if model exists.
    assert len(sys.argv) > 1, "No model name provided."
    model_name = sys.argv[1]
    assert os.path.isfile('./models/{}.json'.format(model_name)), "Model json does not exist."

    # Load model & data.
    model = json.load(open('./models/{}.json'.format(model_name)), encoding='utf-8')
    data = pd.read_csv('./data/' + model['data'])

    index_colnames = sorted(set(
        [model['y_name']]
        + model['index_colnames']['index_1']
        + model['index_colnames']['index_2']
    ))

    # Create model instance.
    kv_model = KleinVellaDoubleIndex(
        data=data,
        y_name=model['y_name'],
        index_names=[model['labels']['index_1'], model['labels']['index_2']],
        index_colnames=[model['index_colnames']['index_1'], model['index_colnames']['index_2']]
    )

    # Fit pilot.
    if model['pilot']['coeffs_start']:
        coeffs_start = [
            ordered_series(
                model['pilot']['coeffs_start']['index_1'],
                kv_model.index_colnames[0]
            ),
            ordered_series(
                model['pilot']['coeffs_start']['index_2'],
                kv_model.index_colnames[1]
            )
        ]
    else:
        coeffs_start = [None, None]

    kv_model.fit_pilot(
        coeffs_start=coeffs_start,
        trim_lower=model['pilot']['trim']['lower'],
        trim_upper=model['pilot']['trim']['upper'],
        n_smoothing_stages_pilot=model['pilot']['n_smoothing_stages'],
        maxiter=model['pilot']['maxiter'],
    )

    # Fit final.
    kv_model.fit_final(
        trim_lower=model['final']['trim']['lower'],
        trim_upper=model['final']['trim']['upper'],
        maxiter=model['final']['maxiter']
    )

    # Create directory for output.
    if not os.path.exists('./bld/fitted'):
        os.makedirs('./bld/fitted')

    # Save.
    with open('./bld/fitted/kv_{}.pickle'.format(model_name), mode='wb') as f:
        pickle.dump(kv_model, f)
