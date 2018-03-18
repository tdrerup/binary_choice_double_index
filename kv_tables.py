# coding: utf-8

"""Code to tabulate KV results.

"""

import os
import re
import sys
import json
import pickle
import pandas as pd

from reg_to_tex import (
    produce_reg_df, tex_models, probit_average_partial_effect_table
)


def tabulate_ols_and_probit_results(model_json, model):

    all_covars = model.index_colnames_all

    # Load data.
    data = pd.read_csv('./data/' + model_json['data'])

    # Write OLS / Probit results
    probit_model = produce_reg_df(
        model_json['y_name'] + ' ~ ' + ' + '.join(all_covars), 'Probit', data, reg_type='probit'
    )
    ols_model = produce_reg_df(
        model_json['y_name'] + ' ~ ' + ' + '.join(all_covars), 'OLS', data)

    tex_models([probit_model, ols_model], 'bld/tables/probit_ols_{}.tex'.format(model_name))


def tabulate_kv_coefficients(model):

    with open(
        './bld/tables/kv_coefficients_{}.tex'.format(model_name),
            'w') as kv_coefficients:

        # Coefficients from Klein-Vella model
        kv_coefficients.write('\\begin{table}\n')
        kv_coefficients.write('\\begin{center}\n')
        kv_coefficients.write(
         '        \\caption{{Average partial effects, {}, Converged: {}}}\n'.format(
             re.sub('_', '\_', model_name),
             kv_model.results_final_scaled.mle_retvals['converged']
         )
        )

        table_kv_coefficients = kv_model.results_table()
        for i in kv_model.index_colnames_all:
            table_kv_coefficients = re.sub('\s+ext', '\\\\text', table_kv_coefficients)

        kv_coefficients.write(table_kv_coefficients)
        kv_coefficients.write('\\end{center}\n')
        kv_coefficients.write('\\end{table}\n')


def tabulate_kv_apes(model):

    with open('./bld/tables/kv_apes_{}.tex'.format(model_name), 'w') as kv_apes:

        # Average partial effects from Klein-Vella model
        kv_apes.write('\\begin{table}\n')
        kv_apes.write('\\begin{center}\n')
        kv_apes.write(
         '        \\caption{{Average partial effects, {}, Converged: {}}}\n'.format(
             re.sub('_', '\_', model_name),
             kv_model.results_final_scaled.mle_retvals['converged']
         )
        )

        table_kv_apes = kv_model.average_partial_effects_table(indicator_dict={})
        for i in kv_model.index_colnames_all:
            table_kv_apes = re.sub('\s+ext', '\\\\text', table_kv_apes)

        kv_apes.write(table_kv_apes)
        kv_apes.write('\\end{center}\n')
        kv_apes.write('\\end{table}\n')


def tabulate_probit_ape(model_json, model):

    # Load data.
    data = pd.read_csv('./data/' + model_json['data'])

    # Average partial effects for Probit model
    all_covars = kv_model.index_colnames_all
    probit_model = model_json['y_name'] + ' ~ ' + ' + '.join(all_covars)

    table_probit_apes = probit_average_partial_effect_table(probit_model, data, indicator_dict={})

    for i in kv_model.index_colnames_all:
        table_probit_apes = re.sub(r'\s+ext', r'\\text', table_probit_apes)

    with open('./bld/tables/probit_apes_{}.tex'.format(model_name), 'w') as probit_apes:
        probit_apes.write('\\begin{table}\n')
        probit_apes.write('\\begin{center}\n')

        probit_apes.write(
            '        \\caption{{Average partial effects / Probit, {}, Converged: {}}}\n'.format(
                re.sub('_', '\_', model_name),
                kv_model.results_final_scaled.mle_retvals['converged']
            )
        )
        probit_apes.write(table_probit_apes)
        probit_apes.write('\\end{center}\n')
        probit_apes.write('\\end{table}\n')


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Provide model name and table type as arguments."
    model_name = sys.argv[1]
    table_type = sys.argv[2]

    # Create directory for output.
    if not os.path.exists('./bld/tables'):
        os.makedirs('./bld/tables')

    model_json = json.load(open('./models/{}.json'.format(model_name), encoding='utf-8'))

    with open('./bld/fitted/kv_{}.pickle'.format(model_name), mode='rb') as f:
        kv_model = pickle.load(f)

    if table_type in ['probit_apes', 'all']:
        tabulate_probit_ape(model_json=model_json, model=kv_model)
    if table_type in ['kv_apes', 'all']:
        tabulate_kv_apes(model=kv_model)
    if table_type in ['kv_coefficients', 'all']:
        tabulate_kv_coefficients(model=kv_model)
    if table_type in ['probit_ols', 'all']:
        tabulate_ols_and_probit_results(model_json=model_json, model=kv_model)
    if table_type not in [
        'all',
        'probit_apes',
        'kv_apes',
        'kv_coefficients',
        'probit_ols'
    ]:
        raise ValueError(table_type)
