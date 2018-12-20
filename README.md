
# Artificial Intelligence for Trading

## Momentum Trading

## Project: Multi-Factor Model

## Table of Contents

1. [Project Overview](#overview)
2. [Data](#data)
3. [Statistical Risk Model](#stat_risk_model)
4. [Alpha Factors](#alpha_factors)
   1. [Momentum 1 Year Factor](#momentum)
   2. [Mean Reversion 5 Day Sector Neutral Factor](#mean_reversion)
   3. [Mean Reversion 5 Day Sector Neutral Smoothed Factor](#mean_reversion_smoothed)
   4. [Overnight Sentiment Factor](#overnight)
   5. [Overnight Sentiment Factor Smoothed](#overnight_smoothed)
5. [The Combined Alpha Factor](#alpha_combined)
6. [Conclusion](#conclusion)
7. [Files](#files)
8. [Libraries](#lib)

<a id='overview'></a>
****
### Project Overview

In this project, I will build a **statistical risk model using PCA.** I’ll use this model to build a portfolio along with 5 alpha factors. I’ll create these factors, then evaluate them using factor-weighted returns, quantile analysis, sharpe ratio, and turnover analysis. At the end of the project, I’ll optimize the portfolio using the risk model and factors using multiple optimization formulations.

<a id='data'></a>

### Data

For the dataset, we'll be using the end of day from [Quotemedia](https://www.quotemedia.com) and sector data from [Sharadar](http://www.sharadar.com/).

Udacity doesn't have a license to redistribute the data to us. They are working on alternatives to this [problem](https://github.com/udacity/artificial-intelligence-for-trading/#no-data).

<a id='stat_risk_model'></a>

### Statistical Risk Model

I used `fit_pca` function to fit the PCA model to the returns data

<a id='alpha_factors'></a>

### Alpha Factors

<a id='momentum'></a>

- Momentum 1 Year Factor

<a id='mean_reversion'></a>

- Mean Reversion 5 Day Sector Neutral Factor

<a id='mean_reversion_smoothed'></a>

- Mean Reversion 5 Day Sector Neutral Smoothed Factor

<a id='overnight'></a>

- Overnight Sentiment Factor

<a id='overnight_smoothed'></a>

- Overnight Sentiment Smoothed

<a id='alpha_combined'></a>

### Combined Alpha Factor

<a id='conclusion'></a>

### Conclusion

<a id='files'></a>

### Files

<a id='lib'></a>

### Libraries

These necessary libraries are mentioned in `requirements.txt`:

- [alphalens==0.3.2](https://pypi.org/project/alphalens/)

Alphalens is a Python Library for performance analysis of predictive (alpha) stock factors. Alphalens works great with the Zipline open source backtesting library, and Pyfolio which provides performance and risk analysis of financial portfolios.

- [colour==0.1.5](https://github.com/vaab/colour)

Converts and manipulates common color representation (RGB, HSL, web, …).

- [cvxpy==1.0.3](https://github.com/cvxgrp/cvxpy/)

Modeling language for convex optimization problems. It allows you to express your problem in a natural way that follows the math, rather than in the restrictive standard form required by solvers.

- [cycler==1.0.3](https://matplotlib.org/cycler/)

Composable style cycles.

- [numpy==1.13.3](http://www.numpy.org/)

NumPy is the fundamental package for scientific computing with Python.

- [pandas==0.18.1](https://github.com/pandas-dev/pandas)

Flexible and powerful data analysis / manipulation library for Python, providing labeled data structures similar to R `data.frame` objects, statistical functions, and much more.

- [plotly==2.2.3](https://plot.ly/python/)

Python plotting library for collaborative, interactive, publication-quality graphs.

- [pyparsing==2.2.0](https://github.com/pyparsing/pyparsing/)

The pyparsing module is an alternative approach to creating and executing simple grammars, vs. the traditional lex/yacc approach, or the use of regular expressions.

- [python-dateutil==2.6.1](https://dateutil.readthedocs.io/en/stable/)

Extensions to the standard Python datetime module.

- [pytz==2017.3](https://pythonhosted.org/pytz/)

This library allows accurate and cross platform timezone calculations using Python 2.4 or higher.

- [requests==2.18.4](http://docs.python-requests.org/en/master/)

Requests is an elegant and simple HTTP library for Python, built for human beings.

- [scipy==1.0.0](https://www.scipy.org/)

Python-based ecosystem of open-source software for mathematics, science, and engineering.

- [scikit-learn==0.19.1](https://scikit-learn.org/stable/)

A set of python modules for machine learning and data mining.

- [six==1.11.0](https://github.com/benjaminp/six)

Six is a Python 2 and 3 compatibility library. It provides utility functions for smoothing over the differences between the Python versions with the goal of writing Python code that is compatible on both Python versions.

- [tqdm==4.19.5](https://tqdm.github.io/)

A fast, extensible progress bar for Python and CLI

- [zipline==1.2.0](http://www.zipline.io/index.html)

Zipline is a Pythonic algorithmic trading library. It is an event-driven system for backtesting. Zipline is currently used in production as the backtesting and live-trading engine powering [Quantopian](https://www.quantopian.com/) – a free, community-centered, hosted platform for building and executing trading strategies.