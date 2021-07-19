---
title: 'Human-Learn: Human Benchmarks in a Scikit-Learn Compatible API'
tags:
  - python
  - machine learning
  - scikit-learn
authors:
  - name: Vincent D. Warmerdam
    orcid: 0000-0003-0845-4528
    affiliation: 1
affiliations:
   - name: Personal
     index: 1
date: May 3rd 2020
bibliography: paper.bib
---

# Summary

This package contains scikit-learn compatible tools that make it easier to construct and benchmark rule-based systems designed by humans. There are tools to turn python functions into scikit-learn compatible components and interactive jupyter widgets that allow the user to draw models. One can also use it to design rules on top of existing models that, for example, can trigger a classifier fallback when outliers are detected.

# Statement of need

There's been a transition from rule-based systems to ones that use machine learning. Initially, systems converted data to labels by applying rules.

![](docs/examples/rules.png)

Recently, it's become much more fashionable to take data with labels and to use machine-learning algorithms to figure out appropriate rules.

![](docs/examples/ml.png)

We started wondering if we might have lost something in this transition. Machine learning is a general tool, but it is capable of making bad decisions. Decisions that are very hard to debug too.  Tools like SHAP [@NIPS2017_7062] and LIME [@lime] try to explain why algorithms make certain decisions in hindsight, but even with the benefit of hindsight, it's tough to understand what is happening.

At the same time, it's also true that many classification problems can be done by natural intelligence. This package aims to make it easier to turn the act of exploratory data analysis into a well-understood model. These "human" models are very explainable from the start. If nothing else, they can serve as a simple benchmark representing domain knowledge which is a great starting point for any predictive project.

The library features components to easily turn python functions into scikit-learn compatible components [@sklearn_api]. Both the `FunctionClassifier` and the `FunctionRegressor` can turn python functions into grid-searchable components. Human-learn also hosts a suite of interactive widgets, made with Bokeh, that might help construct models from Jupyter as well.

![](docs/screenshot.png)

The above screenshot demonstrates the `InteractiveChart` interface. This interface allows the user to draw machine learning models. These models can be used for classification, outlier detection, labeling tasks, or general data exploration.

# Acknowledgements

This project was developed in my spare time while being employed at Rasa. They have been very supportive of me working on my own projects on the side, and I would like to recognize them for being a great employer.

I also want to acknowledge that I'm building on the shoulders of giants. The popular drawing widget in this library would not have been possible without the wider bokeh [@bokeh], jupyter [@jupyter] and scikit-learn communities.

There have also been small contributions on Github from Joshua Adelman, Kay Hoogland, and Gabriel Luiz Freitas Almeida.

# References
