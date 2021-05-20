---
title: 'Human-Learn: Human Benchmarks in a Scikit-Learn Compatible API'
tags:
  - python
  - machine learning
  - scikit-learn
authors:
  - name: Vincent D. Warmerdam
    orcid: 0000-0003-0845-4528
affiliations:
   - name: Self
     index: 1
date: May 3rd 2020
bibliography: paper.bib
---

# Summary

This package contains scikit-learn compatible tools that make it easier to construct and benchmark rule based systems that are designed by humans. There's tools to turn python functions into scikit-learn compatible components as well as interactive jupyter widgets that allow the user to draw models. It can also be used to design rules on top of existing models that, for example, can trigger a classifier fallback when outliers are detected.

# Statement of need

There's been a transition from rule-based systems to ones that use machine-learning.

![](docs/examples/rules.png)
![](docs/examples/ml.png)

We started wondering if we might have lost something in this transition. Machine learning can be generally applied but it is also capable of making bad decision that are very hard to understand. Even with the benefit of hindsight, it's often very hard to understand how an algorithm makes a decision.

At the same time, it's also true that many classification problems can be handled by natural intelligence too. The goal of this package is to make it easier to turn the act of exploratory data analysis into a well understand model. These models can serve as a simple benchmark that represents domain knowledge which is a great starting point for any predictive project.

The library features components to easily turn python functions into scikit-learn compatible components but we also host a suite of interactive widgets that might help construct models as well.

![](docs/screenshot.png)

The above screenshot demonstrates the `InteractiveChart` interface that can be used to draw machine learning models. Note that the same UI can also be used for outlier detection, labelling tasks or general data exploration as well.

# Acknowledgements

This project was developed in my spare time while being employed at Rasa. They have been very supportive of me working on my own projects on the side and I would like to recognise them for being a great employer.

I also want to acknowledge that I'm building on the shoulders of giants. The popular drawing widget in this library would not have been possible without the wider bokeh, jupyter and ipywidgets communities.

There have been also been small contributions on Github from Joshua Adelman, Kay Hoogland and Gabriel Luiz Freitas Almeida.

# References
