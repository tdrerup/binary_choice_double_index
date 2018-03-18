"""Code to provide tex-tables for models of choice.

``produce_reg_df(model, name, data)`` returns a dataframe of regression results
and takes three arguments:
model -- A string of the regression to be run, e.g.:
'expectation ~ covariate_a + covariate_b'
name -- A string to label the model
data -- The dataframe to work with

``tex_models(model_list, filename)`` writes a texfile for a set of models and
takes two arguments:
model_list -- A list of dataframes as returned by the function
``produce_reg_df()``
filename -- The name of the tex-file to write, e.g., 'test.tex'
"""


import pandas as pd
import statsmodels.api as sm
import numpy as np
import re

from statsmodels.iolib.summary2 import summary_params
from patsy import dmatrices
from scipy import stats


def produce_reg_df(model, model_name, panel, reg_type='ols'):

    y, x = dmatrices(model, panel)

    if reg_type == 'ols':
        results = sm.OLS(y, x).fit()
        estimates = summary_params(results)[['Coef.', 'Std.Err.', 'P>|t|']]

        '''
        White’s (1980) heteroskedasticity robust standard errors. Defined as
        sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1) where e_i = resid[i]
        HC0_se is a property. It is not evaluated until it is called. When it is
        called the RegressionResults instance will then have another attribute
        cov_HC0, which is the full heteroskedasticity consistent covariance matrix
        and also het_scale, which is in this case just resid**2. HCCM matrices are
        only appropriate for OLS.

        Note: Delete the following two lines for 'regular' standard errors.
        '''
        estimates['Std.Err.'] = results.HC0_se
        estimates['P>|t|'] = stats.t.sf(
            np.abs(estimates['Coef.'] / estimates['Std.Err.']), results.nobs - 1) * 2

    elif reg_type == 'probit':
        model = sm.Probit(y, x)
        results = model.fit()
        margeffs = results.get_margeff()
        estimates = pd.DataFrame(
            [margeffs.margeff, margeffs.margeff_se, margeffs.pvalues],
            index=['Coef.', 'Std.Err.', 'P>|t|'],
            columns=model.exog_names[1:]).T

    estimates = estimates.apply(
        lambda x: ['{0:0.3f}'.format(i) for i in x])

    estimates['Std.Err.'] = estimates['Std.Err.'].apply(
        lambda x: '(' + str(x) + ')')

    for i in range(len(estimates)):
        estimates['Coef.'].iloc[i] = str(estimates['Coef.'].iloc[i]) + (
            (float(estimates['P>|t|'].iloc[i]) <= 0.01) * '_3stars' +
            (0.01 < float(estimates['P>|t|'].iloc[i]) <= 0.05) * '_2stars' +
            (0.05 < float(estimates['P>|t|'].iloc[i]) <= 0.10) * '_1star' +
            (0.1 < float(estimates['P>|t|'].iloc[i])) * ''
        )

    estimates['P>|t|'] = estimates['P>|t|'].apply(lambda x: '')

    # Instead of inserting lines, just replace pvalues by linespace.
    estimates = estimates.rename(columns={
        'P>|t|': 'addlinespace'}
    )

    stacked_estimates = pd.DataFrame(
        estimates.stack(), columns=[model_name])

    if reg_type == 'ols':
        stacked_model_stats = pd.DataFrame(
            [results.nobs, results.rsquared_adj],
            index=['Observations', 'R2'],
            columns=[model_name])
    elif reg_type == 'probit':
        stacked_model_stats = pd.DataFrame(
            [results.nobs, results.prsquared],
            index=['Observations', 'R2'],
            columns=[model_name])

    stacked_model = stacked_estimates.append(stacked_model_stats)

    return stacked_model


def tex_models(model_list, filename):
    ''' '''
    if len(model_list) > 1:
        try:
            merged_models = model_list[0].join(
                model_list[1:],
                how='outer'
            )
            index_order = []
            for m in model_list:
                for v in m.index:
                    if v not in index_order and v not in {'Observations', 'R2'}:
                        index_order.append(v)
            index_order.append('Observations')
            index_order.append('R2')
            merged_models = merged_models.reindex(index_order)
        except ValueError as e:
            print('Models need different labels.', e)
            raise
    else:
        merged_models = model_list[0]

    merged_models.loc['R2'] = merged_models.loc['R2'].apply(
        lambda x: str(np.round(100 * x, 1))
    )
    merged_models.loc['Observations'] = merged_models.loc['Observations'].apply(
        lambda x: format(int(x), ',d')
    )
    merged_models = merged_models.fillna('')

    merged_models.index = pd.Series(merged_models.index).apply(lambda x: str(x))

    with pd.option_context("max_colwidth", 1000):
        merged_tex = merged_models.to_latex(header=True, escape=False)

    merged_tex = re.sub('\'', '', merged_tex)
    merged_tex = re.sub('\_3stars', '\sym{***}', merged_tex)
    merged_tex = re.sub('\_2stars', '\sym{**}', merged_tex)
    merged_tex = re.sub('\_1star', '\sym{*}', merged_tex)
    merged_tex = re.sub(r'\\begin{tabular}{.*}', '', merged_tex)
    merged_tex = re.sub(r'\\end{tabular}', '', merged_tex)
    merged_tex = re.sub(r'\\toprule', '', merged_tex)
    merged_tex = re.sub(r'\\bottomrule', '', merged_tex)
    merged_tex = re.sub(r' \\\\', r'  \\tabularnewline', merged_tex)

    merged_tex = re.sub('\(.* Std\.Err\.\)', '', merged_tex)
    merged_tex = re.sub('\(.*addlinespace\)', r'\\addlinespace', merged_tex)
    merged_tex = re.sub('addlinespace.*?tabularnewline', 'addlinespace', merged_tex)

    merged_tex = re.sub('\\\_', ' ', merged_tex)
    merged_tex = re.sub(', Coef.\)', '', merged_tex)
    merged_tex = re.sub('\n\(', '\n', merged_tex)
    merged_tex = re.sub(':leq:', r'$\leq$', merged_tex)
    merged_tex = re.sub(':g:', r'$>$', merged_tex)
    merged_tex = re.sub(':l:', r'$<$', merged_tex)
    merged_tex = re.sub(':in:', r'$\in$', merged_tex)
    merged_tex = re.sub(':infty:', r'$\infty$', merged_tex)
    merged_tex = re.sub(':times:', r'$\\times$', merged_tex)
    merged_tex = re.sub(':euro:', r'\\euro', merged_tex)
    merged_tex = re.sub(':text:', r'\\text', merged_tex)
    merged_tex = re.sub(':dol:', r'$', merged_tex)
    merged_tex = re.sub(':bs:', r'\\', merged_tex)

    merged_tex = re.sub(
        'No.{} of Observations', '\midrule\n' + r'\\addlinespace' + '\nObservations',
        merged_tex
    )
    merged_tex = re.sub('R2', r'Adj. (pseudo) R$^2$ (\%)', merged_tex)

    with open(filename, 'w') as tex_file:
        tex_file.write('\\begin{tabular}{l' + '{}'.format('c' * len(model_list)) + '}\n')
        tex_file.write('\\toprule\n')
        tex_file.write('\\addlinespace\n')
        tex_file.write(merged_tex)
        tex_file.write('\\addlinespace\n')
        tex_file.write('\\bottomrule\n')
        tex_file.write('\\end{tabular}\n')


def probit_average_partial_effect_table(probit_model, panel, indicator_dict={}):
    """Return table of average partial effects for *probit_model* (in patsy form),
    estimated using data in *panel*.

    For each binary variable in model, calculate APE as the difference between average
    predicted probability with variable set to 1 and average predicted probability with
    variable set to 0.

    For each continuous variable in model, calculate APE as difference in predicted
    probabilities if each value of variable is increased by 1 standard deviation.

    If evaluated for binary variable, checks for possible linked indicator variables.
    Calculates APE as difference between index where only *variable* is 1 among linked
    indicators and index where all linked indicators and *variable* are 0. *indicator_dict*
    hands dictionary of linked variables to function.

    """

    y, x = dmatrices(probit_model, panel)
    model = sm.Probit(y, x)
    fitted_model = model.fit()

    table = '\\begin{tabular}{lr}\n \\tabularnewline \\toprule\n'
    table += '& Average Partial Effect \\tabularnewline \n'
    table += '    \\midrule\n'

    for i in fitted_model.model.exog_names[1:]:
        probit_data = pd.DataFrame(
            fitted_model.model.exog, columns=fitted_model.model.exog_names)

        # Check if variable is binary:
        binary = (probit_data[i].apply(
            lambda x: (x in [0, 1, 0., 1.] or pd.isnull(x)) is True
        )).all()

        if not binary:
            probit_data_plus_std = probit_data.copy()
            probit_data_plus_std[i] = probit_data[i] + probit_data[i].std()

            new_prob = fitted_model.predict(probit_data_plus_std).mean()
            old_prob = fitted_model.predict(probit_data).mean()

            ape = new_prob - old_prob

        elif binary:
            probit_data_at_zero = probit_data.copy()
            probit_data_at_zero[i] = 0

            related_indicators = indicator_dict.get(i)
            if related_indicators:
                for j in related_indicators:
                    if j in probit_data.columns:
                        probit_data_at_zero[j] = 0

            probit_data_at_one = probit_data_at_zero.copy()
            probit_data_at_one[i] = 1

            new_prob = fitted_model.predict(probit_data_at_one).mean()
            old_prob = fitted_model.predict(probit_data_at_zero).mean()

            ape = new_prob - old_prob

        cname = re.sub('_', '\_', i)
        table += '      {} & {:1.3f} \\tabularnewline\n'.format(
            cname,
            ape
        )

    table += '\\bottomrule\n\\end{tabular}\n\n'

    return table
