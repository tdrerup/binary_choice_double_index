# coding: utf-8

"""Main estimation code.

"""

import re
import numpy as np
import pandas as pd

from scipy.stats.mstats import gmean
from statsmodels.base.model import GenericLikelihoodModel
from numba import jit


_norm_pdf_C = np.sqrt(2 * np.pi)


@jit(nopython=True)
def _norm_pdf(x):
    return np.exp(-x ** 2 / 2) / _norm_pdf_C


@jit(nopython=True)
def _kde_local(loc, data, bw, lmbda):
    """Return the locally smoothed kernel density estimate at *loc*
    based on *data* with locally smoothed bandwidth *bw x lmbda*,
    where *lmbda* is either a scalar or a vector of the same length
    as *data*.

    """

    l_s_bw = bw * lmbda
    d = (loc - data).T / l_s_bw
    s = (_norm_pdf(d) / l_s_bw).T

    kde = 0.0
    for r in range(s.shape[0]):
        kde += s[r].prod()
    return kde


@jit(nopython=True)
def _kde_local_array_core(index_std, locs_std, leave_one_out_locs, other_locs, nobs, h, lmbda):
    # Loop over leave-one-out variables and others.
    loo_shape = (index_std.shape[0] - 1, index_std.shape[1])
    loo_index = np.empty(loo_shape, dtype=np.double)
    loo_lmbda = np.empty(loo_shape[0], dtype=np.double)
    out = np.empty(len(locs_std), dtype=np.double) * np.nan
    i = 0
    for j in leave_one_out_locs:
        k_loo = 0
        for k in range(index_std.shape[0]):
            if not k == i:
                loo_index[k_loo, 0] = index_std[k, 0]
                loo_index[k_loo, 1] = index_std[k, 1]
                loo_lmbda[k_loo] = lmbda[k]
                k_loo += 1
        out[j] = _kde_local(locs_std[j], loo_index, h, loo_lmbda) / (nobs - 1)
        i += 1
    for j in other_locs:
        out[j] = _kde_local(locs_std[j], index_std, h, lmbda) / nobs

    return out


def _kde_local_array(locs, index, leave_one_out_locs, other_locs, nobs, h, lmbda):
    """Return locally smoothed density of *index* evaluated
    at each element of *locs*.

    Further parameters:

        * *h* - the baseline bandwidth
        * *lmbda* - the local smoothing parameter adjusting the bandwidth

    In KV (2009), this corresponds to the :math:`f^\hat_s, s \in \{0, 1\}`
    in D1 (but for all observations instead of one ω).

    """

    # Standardise data and locs s.t. the product kernel can be used easily.
    Sigma = np.cov(index.T)
    if len(Sigma.shape) == 0:
        Sigma_inv = Sigma ** -1
        sqrt_det = np.sqrt(Sigma_inv)
        chol_Sigma_inv = sqrt_det
    elif len(Sigma.shape) == 2:
        Sigma_inv = np.linalg.inv(Sigma)
        sqrt_det = np.sqrt(np.linalg.det(Sigma_inv))
        chol_Sigma_inv = np.linalg.cholesky(Sigma_inv)
    index_std = index.dot(chol_Sigma_inv)
    locs_std = locs.dot(chol_Sigma_inv)

    return sqrt_det * _kde_local_array_core(
        index_std,
        locs_std,
        leave_one_out_locs,
        other_locs,
        nobs,
        h,
        lmbda
    )


class KleinVellaDoubleIndex(GenericLikelihoodModel):

    def __init__(self, data, y_name, index_names, index_colnames):
        """Set up the data and basic model. Arguments:

            * *data*: A pandas dataframe with all dependent and explanatory
              variables
            * *y_name*: The name of the dependent variable (string)
            * *index_names*: A 2-element list/tuple with the names of the indices.
              E.g.: ['Structural Equation', 'Control Function']
            * *index_colnames*: A 2-element list of iterables with the names of
              the independent variables (strings). E.g.:

                    [
                        ['age', 'female', 'income'],
                        ['wealth', 'female', 'income']
                    ]

              Both should contain a dedicated continuous
              variable as the first element (responsibility of the user).

        *y_name* and the elements of *index[k]_names* must be present in the
        columns of *data*.

        """

        cols = data.columns
        assert y_name in cols
        self.y_name = y_name
        assert len(index_names) == 2
        assert len(index_colnames) == 2
        self.index_names = tuple(index_names)
        self.index_colnames = []
        self.index_colnames_all = []
        self.index_ncoeffs = np.zeros(2, dtype=np.int)
        for i in range(2):
            for i_n in index_colnames[i]:
                assert i_n in cols, "'{}' not in data columns!".format(i_n)
            self.index_colnames.append(tuple(index_colnames[i]))
            self.index_ncoeffs[i] = len(self.index_colnames[i]) - 1
        for v0 in self.index_colnames[0]:
            if v0 not in self.index_colnames[1]:
                self.index_colnames_all.append(v0)
        for v1 in self.index_colnames[1]:
            self.index_colnames_all.append(v1)
        self.coeffs = [None, None]
        # Retain only data without missings in all relevant variables
        self._data = data.dropna(subset=[y_name] + self.index_colnames_all)
        self._nobs = len(self._data)
        self._data = self._data.set_index(np.arange(self._nobs))
        # Trimming is done ex post, so we can set the data here already.
        super(KleinVellaDoubleIndex, self).__init__(
            endog=self._data[self.y_name],
            exog=self._data[self.index_colnames_all]
        )
        self.endog = self._data[self.y_name]
        self.exog = self._data[self.index_colnames_all]
        # Consistency check - binary dependent variable?
        assert set(self._data[self.y_name].unique()) == {0, 1}, (
            "\n\nY is not a binary variable: {}\n\n".format(set(self._data[self.y_name].unique()))
        )

    def coeffs_from_vec(self, coeffs_vec):
        """Set the attribute *coeffs* based on *coeffs_vec*."""
        coeffs = [self.coeffs[0].copy(), self.coeffs[1].copy()]
        coeffs[0].iloc[1:] = coeffs_vec[:self.index_ncoeffs[0]].copy()
        coeffs[1].iloc[1:] = coeffs_vec[self.index_ncoeffs[0]:].copy()
        return coeffs

    def _coeff_series_to_vec(self, coeffs):
        vec = np.zeros(self.index_ncoeffs.sum(), dtype=np.float)
        vec[:self.index_ncoeffs[0]] = coeffs[0].iloc[1:].values.copy()
        vec[self.index_ncoeffs[0]:] = coeffs[1].iloc[1:].values.copy()
        return vec

    def get_index(self, coeffs):
        """Return the based on a 2-element list of *coeffs* and the data in *self.exog*.

        """

        return pd.DataFrame(
            data=[
                self.exog[coeffs[0].index].dot(coeffs[0]),
                self.exog[coeffs[1].index].dot(coeffs[1])
            ],
            index=[0, 1]
        ).T

    def τ(self, z, a):
        """Return smooth trimming weights, formula in D2 of KV (2009)."""
        return 1 / (1 + np.exp(z * self._nobs ** a))

    def _λ(self, f):
        """Return the estimated local smoothing parameter, formula in D3 of KV (2009)."""
        γ = f / gmean(f)
        d = self.τ(z=1 / np.log(self._nobs) - γ, a=0.01)
        return (d * γ + (1 - d) / np.log(self._nobs)) ** (-1 / 2)

    def λ_multi_stage(self, index, n_stages, h1=None, h2=None):
        """Return the vector of estimated local smoothing parameters in D3/D4 of KV (2009)
        for each element of *index*.

        The parameter *n_stages ∊ {1, 2, 3}* controls the number of stages:

            * 1 just returns a vector of ones
            * 2 returns a vector of parameters from a single smoothing step
            * 3 returns a vector of parameters from two smoothing steps
        """

        if len(index.shape) == 1:
            index = index.reshape((len(index), 1))

        n = len(index)
        all_obs = np.arange(n)
        no_obs = np.array([], dtype=np.int64)
        λ1 = np.ones(n, dtype=np.double)
        if n_stages == 1:
            return λ1
        elif n_stages in {2, 3}:
            assert h1 is not None
            λ2 = self._λ(_kde_local_array(index, index, all_obs, no_obs, self._nobs, h1, λ1))
            if n_stages == 2:
                return λ2
            else:
                assert h2 is not None, "3-stage smoothing currently not implemented."
                return self._λ(_kde_local_array(index, index, all_obs, no_obs, self._nobs, h2, λ2))
        else:
            raise ValueError(n_stages)

    def _xtrim(self, lower, upper):
        """Return trimming indicator series, where trimming is based on
        the covariates directly (and the quantiles to be trimmed at, i.e.
        *lower* and *upper*).

        """
        trm = pd.Series(data=True, index=self._data.index)
        for c in self.index_colnames_all:
            l_limit = np.percentile(self._data[c], 100 * lower)
            u_limit = np.percentile(self._data[c], 100 * upper)
            trm &= self._data[c].apply(lambda x: True if l_limit <= x <= u_limit else False)

        return trm

    def f_s_pilot(self, s, index):
        """Return a pilot density estimate (potentially locally smoothed)
        conditional on the outcome of the dependent variable, as defined
        in D1-D4 of KV (2009).

        In theory (see the paper), the local smoothing step is not needed.
        In practice, it is used in the code by the authors.

        """

        assert s in {0, 1}

        index_s = index[self.endog == s].values
        leave_one_out_locs = index[self.endog == s].index.values
        other_locs = index[self.endog == 1 - s].index.values

        λ = self.λ_multi_stage(index_s, n_stages=self._n_smoothing_stages_pilot, h1=self._h_pilot)

        return _kde_local_array(
            index.values,
            index_s,
            leave_one_out_locs,
            other_locs,
            self._nobs,
            self._h_pilot,
            λ
        )

    def semiparametric_probability_function_pilot(self, index):
        f0 = self.f_s_pilot(0, index)
        f1 = self.f_s_pilot(1, index)
        return f1 / (f1 + f0)

    def _bin_loglikeobs(self, P):
        Y = self.endog
        return Y * np.log(P) + (1 - Y) * np.log(1 - P)

    def _loglikeobs_pilot(self, coeffs_vec):
        """Return the pilot estimator of the log likelihood function, i.e. the Q
        in D6 of KV (2009).

        """

        self.coeffs = self.coeffs_from_vec(coeffs_vec)
        index = self.get_index(self.coeffs)
        P = self.semiparametric_probability_function_pilot(index)

        return self._xtrim_series * self._bin_loglikeobs(P)

    def fit_pilot(
        self,
        coeffs_start=[None, None],
        trim_lower=0.01,
        trim_upper=0.99,
        n_smoothing_stages_pilot=1,
        maxiter=500
    ):
        """Fit the initial model, where trimming is based on the covariates
        directly (as opposed to the index).

        Arguments: *coeffs_start* a 2-element list of start values for the
        coefficient vectors of both indices. The order must be the same as
        the order of *self.index_colnames* and the initial element of each start
        vector must be unity. If the start values are set to *None*, a vector
        of ones will be used.

        """

        for i in range(2):
            if coeffs_start[i] is None:
                coeffs_start[i] = pd.Series(data=1.0, index=self.index_colnames[i])
            else:
                assert tuple(coeffs_start[i].index) == self.index_colnames[i]
                assert coeffs_start[i].iloc[0] in [-1.0, 1.0]
            self.coeffs[i] = coeffs_start[i].copy()

        vec_coeffs_start = self._coeff_series_to_vec(coeffs_start)
        self._xtrim_series = self._xtrim(lower=trim_lower, upper=trim_upper)
        self._h_pilot = self._nobs ** - (1 / 11)
        self._n_smoothing_stages_pilot = n_smoothing_stages_pilot
        self.loglikeobs = self._loglikeobs_pilot
        print("Starting pilot fit.")
        self.results_pilot = self.fit(
            start_params=vec_coeffs_start,
            method='bfgs',
            maxiter=maxiter,
            full_output=1,
            disp=1,
            callback=None,
            retall=1,
            tol=0.001
        )
        self.coeffs = self.coeffs_from_vec(self.results_pilot.params)
        self._coeffs_pilot_vec = self.results_pilot.params.copy()
        self.coeffs_pilot = [self.coeffs[0].copy(), self.coeffs[1].copy()]
        self.index_pilot = self.get_index(self.coeffs_pilot)

    def _itrim(self, coeffs, lower, upper):
        """Return trimmming vector based on product of trimming vectors
        for individual indices.

        """

        index = self.get_index(coeffs)

        trm = pd.Series(data=1, index=self._data.index, dtype=np.double)
        for i in 0, 1:
            l_limit = np.percentile(index[i], 100 * lower)
            u_limit = np.percentile(index[i], 100 * upper)

            trm_l = self.τ(z=l_limit - index[i], a=1 / 12)
            trm_u = 1 - self.τ(z=u_limit - index[i], a=1 / 12)

            trm *= trm_l * trm_u

        return trm

    def f_s(self, index, index_s, leave_one_out_locs, other_locs):
        """Return a locally smoothed density estimate conditional on the outcome
        of the dependent variable, as defined in D1-D4 of KV (2009).

        Usually, *index* should be the index regardless of the outcome, *index_s*
        should be the index for those observations with outcome s ∊ {0, 1},
        *leave_one_out_locs* the integer locations of these outcomes, and *other_locs*
        the integer locations of the outcome 1 - s.

        However, this might be different for calculations such as the ASF.

        """

        λ3 = self.λ_multi_stage(index_s, n_stages=3, h1=self._h1, h2=self._h2)

        return _kde_local_array(
            index,
            index_s,
            leave_one_out_locs,
            other_locs,
            self._nobs,
            self._h3,
            λ3
        )

    def f(self, eval_grid, index_data):
        """Syntactic sugar for local density estimation at a grid for marginal
        or joint densities.

        Both *eval_grid* and *index_data* must be NumPy arrays.

        """

        # Make sure we have 2-d arrays throughout.
        if len(eval_grid.shape) == 1:
            eval_grid = np.reshape(eval_grid, (len(eval_grid), 1))
        elif len(eval_grid.shape) > 2:
            raise ValueError(eval_grid.shape)
        if len(index_data.shape) == 1:
            index_data = np.reshape(index_data, (len(index_data), 1))
        elif len(index_data.shape) > 2:
            raise ValueError(index_data.shape)

        return self.f_s(
            index=eval_grid,
            index_s=index_data,
            leave_one_out_locs=np.array([], dtype=np.int64),
            other_locs=np.arange(len(eval_grid))
        )

    def Δ(self, f, s, ε=0.9):
        """Return the adjustment factors for the probability function defined in D5 of KV (2009).

        """

        N = self._nobs
        c = self._f_pilot_perc1[s]
        α1 = ε * self._r3 / 4
        α2 = ε * self._r3 / 5

        return c * self._h3 ** ε / (1 + np.exp(N ** α1 * (f - N ** -α2)))

    def semiparametric_probability_function(self, index, eval_locs=None):
        """Return the semiparametric probability function defined in D5 of KV (2009).

        If *eval_locs* is *None*, go for estimation mode and evaluate the
        function for each data point. Else evaluate it at *eval_locs*.

        """

        index0 = index[self.endog == 0].values
        index1 = index[self.endog == 1].values

        if eval_locs is None:
            eval_locs = index.values
            f0_leave_one_out_locs = index[self.endog == 0].index.values
            f1_leave_one_out_locs = index[self.endog == 1].index.values
            f0_other_locs = f1_leave_one_out_locs
            f1_other_locs = f0_leave_one_out_locs
        else:
            f0_leave_one_out_locs = np.array([], dtype=np.int64)
            f1_leave_one_out_locs = np.array([], dtype=np.int64)
            f0_other_locs = np.arange(len(eval_locs))
            f1_other_locs = np.arange(len(eval_locs))

        # Density estimates conditional on the outcome.
        f0 = self.f_s(
            index=eval_locs,
            index_s=index0,
            leave_one_out_locs=f0_leave_one_out_locs,
            other_locs=f0_other_locs
        )
        f1 = self.f_s(
            index=eval_locs,
            index_s=index1,
            leave_one_out_locs=f1_leave_one_out_locs,
            other_locs=f1_other_locs
        )

        Δ0 = self.Δ(f=f0, s=0)
        Δ1 = self.Δ(f=f1, s=1)
        return (f1 + Δ1) / (f0 + f1 + Δ0 + Δ1)

    def _loglikeobs_final(self, coeffs_vec_scaled):
        coeffs_vec = coeffs_vec_scaled * self._coeffs_pilot_vec
        self.coeffs = self.coeffs_from_vec(coeffs_vec)
        P = self.semiparametric_probability_function(self.get_index(self.coeffs))
        return self._itrim_series * self._bin_loglikeobs(P)

    def _set_constants_itrim(self, r3, δ, trim_lower, trim_upper):
        # Preliminaries: Set various parameters for local smoothing
        r1 = (r3 - δ) / 4
        r2 = (r3 - δ / 2) / 2
        self._h1 = self._nobs ** -r1
        self._h2 = self._nobs ** -r2
        self._h3 = self._nobs ** -r3
        self._r3 = r3
        # Needed for Δ0, Δ1
        self._f_pilot_perc1 = np.zeros(2)
        self._f_pilot_perc1[0] = np.percentile(
            self.f_s_pilot(s=0, index=self.index_pilot) / (1 - self.endog.mean()), 1
        )
        self._f_pilot_perc1[1] = np.percentile(
            self.f_s_pilot(s=1, index=self.index_pilot) / self.endog.mean(), 1
        )
        # Re-use trimming bounds for ASF, so keep here.
        self.trim_lower = trim_lower
        self.trim_upper = trim_upper
        self._itrim_series = self._itrim(
            coeffs=self.coeffs_pilot,
            lower=trim_lower,
            upper=trim_upper
        )

    def fit_final(
        self,
        r3=1 / 11, δ=0.04,
        trim_lower=0.01,
        trim_upper=0.99,
        maxiter=1000
    ):
        """Fit the final model, where trimming is based on the two indices.

        .. note::

            This routine assumes that *fit_pilot* has been run and that the
            resulting first-step coefficients / index values are stored in
            *self.coeffs_pilot* and *self.index_pilot*, respectively.

            In order to improve numerical precision, we scale the coefficient
            vector with the pilot estimates.

        """

        vec_coeffs_start_scaled = np.ones(self.index_ncoeffs.sum())
        self._set_constants_itrim(r3, δ, trim_lower, trim_upper)
        self.loglikeobs = self._loglikeobs_final
        print("Starting final fit.")
        self.results_final_scaled = self.fit(
            start_params=vec_coeffs_start_scaled,
            method='bfgs',
            maxiter=maxiter,
            full_output=1,
            disp=1,
            callback=None,
            retall=1,
            gtol=1e-5
        )
        self.coeffs = self.coeffs_from_vec(
            self.results_final_scaled.params * self._coeffs_pilot_vec
        )
        self.coeffs_final = [self.coeffs[0].copy(), self.coeffs[1].copy()]
        self.index_final = self.get_index(self.coeffs_final)
        self.std_err_final = self.coeffs_from_vec(
            self.results_final_scaled.bse * np.abs(self._coeffs_pilot_vec)
        )

    def average_structural_function(self, asf_index_loc, asf_loc, r=None, ε=1e-3):
        """Return the value of the average structural function and its
        standard error for *asf_index_loc* ∊ {0, 1}, evaluated at the
        point *asf_loc*.

        I.e. if *asf_index_loc=0*, the index=1 is integrated out.

        """

        index0 = self.index_final[0].values
        index1 = self.index_final[1].values
        endog = self.endog.values
        n_grid = 200

        # Set up mesh.
        if asf_index_loc == 0:
            asf_index = index0
            other_index = index1
        elif asf_index_loc == 1:
            asf_index = index1
            other_index = index0
        else:
            raise ValueError('asf_index_loc = {} ∉ {{0, 1}}'.format(asf_index_loc))

        # Calculate the ASF.
        other_grid = np.linspace(other_index.min(), other_index.max(), n_grid)
        eval_grid = pd.DataFrame({asf_index_loc: asf_loc, 1 - asf_index_loc: other_grid}).values
        pred_grid = self.semiparametric_probability_function(
            index=self.index_final,
            eval_locs=eval_grid
        )
        dens_other_est = self.f(eval_grid=other_grid, index_data=other_index)
        # And now the integral (note we're using an equally spaced grid).
        asf = dens_other_est.dot(pred_grid) * (other_grid[1] - other_grid[0])

        # Set the bandwidth (note the bandwidth is always relative to the standardised index).
        if r is None:
            h = self._h3 * asf_index.std()
        else:
            h = self._nobs ** -r * asf_index.std()

        # Variance of the ASF - Start with squared error.
        eval_n = pd.DataFrame({asf_index_loc: asf_loc, 1 - asf_index_loc: other_index}).values
        pred_n = self.semiparametric_probability_function(index=self.index_final, eval_locs=eval_n)
        error2 = (endog - pred_n) ** 2
        # Density ratio: Use the same adjustment factors as before, but only for the extremes.
        dens_other = self.f(eval_grid=other_index, index_data=other_index)
        dens_joint = self.f(eval_grid=eval_n, index_data=self.index_final.values)
        q_other = np.percentile(dens_other, 5)
        Δ_other = q_other / ε * self.Δ(dens_other / dens_other.std(), 1 - asf_index_loc, ε)
        Δ_other *= self.τ(z=(dens_other - q_other) / dens_other.std(), a=0.4)
        q_joint = np.percentile(dens_joint, 5)
        Δ_joint = q_joint / ε
        Δ_joint *= self.τ(z=(dens_joint - q_joint) / dens_joint.std(), a=0.4)
        w2 = ((dens_other + Δ_other) / (dens_joint + Δ_joint)) ** 2
        # Locally smoothed kernel.
        λ = self.λ_multi_stage(index=asf_index, n_stages=3, h1=self._h1, h2=self._h2)
        kernel2 = _norm_pdf((asf_loc - asf_index) / (h * λ)) ** 2
        # Put everything together.
        σ2 = (error2 * kernel2 * w2 / h).mean()
        asf_se = np.sqrt(σ2) * (self._nobs * h) ** -0.5

        return asf, asf_se

    def results_table(self):
        table = '\\begin{tabular}{lcrrcrr}\n    \\toprule\n    & \hspace*{0ex} '
        table += '& \\multicolumn{{2}}{{c}}{{{}}} '.format(self.index_names[0])
        table += '& \hspace*{0ex} '
        table += '& \\multicolumn{{2}}{{c}}{{{}}} '.format(self.index_names[1])
        table += '\\tabularnewline\n    \\cmidrule{3-4}\\cmidrule{6-7}\n'
        table += '    && Estimate & Std. Err. && Estimate & Std. Err. \\tabularnewline\n'
        table += '    \\midrule\n'

        coeffs = self.coeffs_final
        std_errs = self.std_err_final
        used_colnames = set()
        for i, c in enumerate(self.index_colnames[0]):
            cname = re.sub('_', '\_', c)
            if i == 0:
                table += '    {} && {:1.2f} & \(\cdot\;\;\)'.format(cname, coeffs[0][c])
                table += ' && \(\cdot\;\;\) & \(\cdot\;\;\) \\tabularnewline\n'
                used_colnames.add(c)
            elif c not in self.index_colnames[1]:
                table += '    {} && {:1.2f} & {:1.2f}'.format(cname, coeffs[0][c], std_errs[0][c])
                table += ' && \(\cdot\;\;\) & \(\cdot\;\;\) \\tabularnewline\n'
                used_colnames.add(c)
        for i, c in enumerate(self.index_colnames[1]):
            cname = re.sub('_', '\_', c)
            if i == 0:
                table += '    {} && \(\cdot\;\;\) & \(\cdot\;\;\) && '.format(cname)
                table += '{:1.2f} & \(\cdot\;\;\) \\tabularnewline\n'.format(coeffs[1][c])
                used_colnames.add(c)
            elif c not in self.index_colnames[0]:
                table += '    {} && \(\cdot\;\;\) & \(\cdot\;\;\) && '.format(cname)
                table += '{:1.2f} & {:1.2f} \\tabularnewline\n'.format(
                    coeffs[1][c], std_errs[1][c]
                )
                used_colnames.add(c)
        for c in self.index_colnames[0] + self.index_colnames[1]:
            cname = re.sub('_', '\_', c)
            if c in used_colnames:
                continue
            else:
                table += '    {} && {:1.2f} & {:1.2f} && {:1.2f} & {:1.2f}'.format(
                    cname, coeffs[0][c], std_errs[0][c], coeffs[1][c], std_errs[1][c]
                )
                table += ' \\tabularnewline\n'
                used_colnames.add(c)

        table += '    \\bottomrule\n\\end{tabular}\n\n'
        return table

    def average_partial_effect(self, variable, indicators=None, delta='one std', index_loc=0):
        """Return average partial effect for *variable*.

        For binary variable (:= variable in [0, 1, None]), calculate APE as difference between
        average probabilities for hypothetical indices where all values of *variable* are either
        1 or 0, respectively.

        For continuous *variable*, increase by *delta* (default = 1 standard deviation) for
        all observations.

        If evaluated for binary variable, checks for possible linked indicator variables.
        Calculates APE as difference between index where only *variable* is 1 among linked
        indicators and index where all linked indicators and *variable* are 0.

        Related problem for mfx described here:
        http://www.stata-journal.com/sjpdf.html?articlenum=st0086
        (find "set of" in article to get to problem description)

        """

        coeffs = [
            pd.DataFrame(self.coeffs_final[0], columns=['coef.']),
            pd.DataFrame(self.coeffs_final[1], columns=['coef.'])
        ]

        # Check if variable is binary:
        binary = (self._data[variable].apply(
            lambda x: (x in [0, 1, 0., 1.] or pd.isnull(x)) is True
        )).all()

        if delta == 'one std':
            delta = self._data[variable].std()

        if not binary:
            if index_loc in [0, 1] and variable in coeffs[index_loc].index:
                if index_loc == 0:
                    change0 = delta * coeffs[0].loc[variable][0]
                    change1 = 0
                elif index_loc == 1:
                    change0 = 0
                    change1 = delta * coeffs[1].loc[variable][0]
            elif index_loc == 'both':
                if variable in coeffs[0].index:
                    change0 = delta * coeffs[0].loc[variable][0]
                else:
                    change0 = 0

                if variable in coeffs[1].index:
                    change1 = delta * coeffs[1].loc[variable][0]
                else:
                    change1 = 0
            else:
                return '\(\cdot\;\;\)'

            new_index = self.index_final.copy()
            new_index[0] = new_index[0] + change0
            new_index[1] = new_index[1] + change1

            old_prob = self.semiparametric_probability_function(self.index_final).mean()
            new_prob = self.semiparametric_probability_function(
                self.index_final, new_index.values
            ).mean()

            return '{:1.3f}'.format(new_prob - old_prob)

        elif binary:

            related_indicators = indicators.get(variable)
            index_at_zero = self.index_final.copy()

            if index_loc in [0, 1] and variable in coeffs[index_loc].index:
                for c in coeffs[index_loc].index:
                    if c == variable:
                        index_at_zero[index_loc] = (
                            index_at_zero[index_loc] - coeffs[index_loc].loc[c][0] * self._data[c])
                    if related_indicators is not None and c in related_indicators:
                        index_at_zero[index_loc] = (
                            index_at_zero[index_loc] - coeffs[index_loc].loc[c][0] * self._data[c])

                index_at_one = index_at_zero.copy()
                index_at_one[index_loc] = (
                    index_at_one[index_loc] + coeffs[index_loc].loc[variable][0]
                )

            elif index_loc is 'both':
                for i in [0, 1]:
                    for c in coeffs[i].index:
                        if c == variable:
                            index_at_zero[i] = (
                                index_at_zero[i] - coeffs[i].loc[c][0] * self._data[c]
                            )

                        if related_indicators is not None and c in related_indicators:
                            index_at_zero[i] = (
                                index_at_zero[i] - coeffs[i].loc[c][0] * self._data[c]
                            )

                index_at_one = index_at_zero.copy()
                for i in [0, 1]:
                    if variable in coeffs[i].index:
                        index_at_one[i] = index_at_one[i] + coeffs[i].loc[variable][0]

            else:
                return '\(\cdot\;\;\)'

            prob_at_zero = self.semiparametric_probability_function(
                self.index_final, index_at_zero.values
            ).mean()
            prob_at_one = self.semiparametric_probability_function(
                self.index_final, index_at_one.values
            ).mean()

            return '{:1.3f}'.format(prob_at_one - prob_at_zero)

    def average_partial_effects_table(self, indicator_dict={}):
        '''Return tex-table of average partial effects.

        Indicator dict can include dummies that need special care (cf. *average_partial_effect*).

        '''

        table = '\\begin{tabular}{lcrcrcr}\n \\toprule\n & \hspace*{0ex} '
        table += '& {}'.format(self.index_names[0])
        table += '& \hspace*{0ex} '
        table += '& {}'.format(self.index_names[1])
        table += '& \hspace*{0ex} '
        table += '& Combined \\tabularnewline\n'
        table += '    \\midrule\n'

        for c in self.index_colnames[0]:
            # Only print covariates not included in second model.
            if c not in self.index_colnames[1]:
                cname = re.sub('_', '\_', c)
                table += '      {} && {} && {} && {} \\tabularnewline\n'.format(
                    cname,
                    self.average_partial_effect(
                        variable=c, indicators=indicator_dict, index_loc=0),
                    '\(\cdot\;\;\)',
                    self.average_partial_effect(
                        variable=c, indicators=indicator_dict, index_loc='both')
                )
        for c in self.index_colnames[1]:
            cname = re.sub('_', '\_', c)
            table += '      {} && {} && {} && {}  \\tabularnewline\n'.format(
                cname,
                self.average_partial_effect(
                    variable=c, indicators=indicator_dict, index_loc=0),
                self.average_partial_effect(
                    variable=c, indicators=indicator_dict, index_loc=1),
                self.average_partial_effect(
                    variable=c, indicators=indicator_dict, index_loc='both')
            )

        table += '\\bottomrule\n\\end{tabular}\n\n'

        return table


if __name__ == '__main__':
    pass
