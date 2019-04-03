## A Python implementation of Klein and Vella's semiparametric double-index estimator (JoAE, 2009)

This is a Python implementation of "A semiparametric model for binary response and continuous outcomes under index heteroscedasticity" by Klein and Vella ([2009, Journal of Applied Econometrics](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.1064)). The code is a simplified version of the code used in Drerup, Enke, and von Gaudecker ([2017, Journal of Econometrics](https://www.sciencedirect.com/science/article/pii/S0304407617301033)). An older version of the code embedded in von Gaudecker's [Templates for Reproducible Research Projects in Economics](https://github.com/hmgaudecker/econ-project-templates) can be found [here](http://www.wiwi.uni-bonn.de/gaudecker/research.html#household-finance).

### Current features.

* Model estimation
* Tables:
  * coefficients
  * average partial effects of the KV model
  * average partial effects of a similar probit
  * average partial effects of a similar probit alongside OLS results
* Figures:
  * a 3d plot with predicted probabilities
  * a 2d contour plot with joint densities of the indices
  * plots of the average structural functions along the indices
  * plots of the average structural functions alongside a plot of a comparable probit

All of the code below has been tested on a Mac. Let me know if you are interested in a Windows version. The required changes should be minimal.

*Note: The code contains some changes that have not fully been tested yet. Any feedback is welcome: <tdrerup@uni-bonn.de>*

### Getting started.

While the code *may* work with your current Python (≥ 3.4) installation, it is helpful to work in a preset environment. To do so, first install either [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html). Then type
```
  source set-env.sh
```
to install all required packages in the version specified in `environment.yml`. Activate the environment using
```
  source activate binary_choice_double_index
```
Whenever you work with the code, be sure to activate the environment beforehand.

### Usage Example.

We will show the basic usage of the commands using an example dataset (`./data/example.csv`). Run this example to make sure that the code works.

The specification we are going to estimate is saved in `./models/example.json`. The layout of this file is straightforward. We will discuss the individual components below. To fit the model, type
```
  python kv_fit.py example
```
You should see output like the following:
```
  Starting pilot fit.
  Optimization terminated successfully.
           Current function value: 0.511914
           Iterations: 45
           Function evaluations: 62
           Gradient evaluations: 62
  Starting final fit.
  Optimization terminated successfully.
           Current function value: 0.397809
           Iterations: 46
           Function evaluations: 49
           Gradient evaluations: 49
```
Fitting the model takes about 1 minute on a 2015 MBP. More complicated models or models with more data can take substantially longer. Results (coefficients, final indices, etc.) will be packaged in a pickle and saved in `./bld/results/`.

To obtain tables with

* coefficients
* average partial effects of the KV model
* average partial effects of a similar probit
* average partial effects of a similar probit alongside OLS results

run
```
  python kv_tables.py example all
```

Tables will be saved as *.tex* files inin `./bld/tables`. As you can see from the tables, the example model contained one index with 5 variables and one with only 1.

To visualize the results, type
```
  python kv_figures.py example all
```
This will create the following figures:

* a 3d plot with predicted probabilities
* a 2d contour plot with joint densities of the indices
* plots of the average structural functions along the indices
* plots of the average structural functions alongside a plot of a comparable probit

All figures will be saved in `./bld/figures`.

### Details.

For a detailed description of the tables and figures produced by `kv_tables.py` and `kv_figures.py`, check *The precision of subjective data and the explanatory power of economic models* (2017) in the [Journal of Econometrics](https://www.sciencedirect.com/science/article/pii/S0304407617301033)) or the [working paper version](http://www.wiwi.uni-bonn.de/gaudecker/_static/meas_error_subj_beliefs.pdf).

**Data.**

Store the data as a comma-separated file (*.csv*) in `./data/`. The estimator expects the outcome to be binary (0/1, no categoricals). Explanatory variables are expected to be numerical, so dummies for categorical variables need to be created beforehand.

**Model Setup.**

The folder `./models/` contains json files with information about the model(s) to estimate. Each json contains the following entries:

* `data`: The name of the file that contains the data for this model.
* `y_name`: The name of the outcome variable (as stored in `data`).
* `labels`: A dictionary with the names / labels for the outcome variable and indices (mainly used for plotting)
* `index_colnames`: The names of the variables to include in the two indices:
  * `index_1`: A list of the variables for index 1
  * `index_2`: A list of the variables for index 2
* `pilot`: Settings for the KV pilot estimator
  * `coeffs_start`: Starting values for the coefficients for each index
  * `trim`: Percentiles for trimming of the densities
  * `n_smoothing_stages`: Number of smoothing stages (3 is currently not implemented)
  * `maxiter`: Maximum number of iterations
* `final`: Settings for the final KV estimator
  * `trim`: Percentiles for trimming of the densities
  * `maxiter`: Maximum number of iterations

Use the json for the example as a template for your own specification.
