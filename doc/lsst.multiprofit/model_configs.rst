.. _lsst.multiprofit.model-configs:

====================
Model Configurations
====================

MultiProFit's second layer of data structures is a series of :py:class:`lsst.pex.config.Config` classes provided in :py:mod:`lsst.multiprofit.componentconfig`, :py:mod:`lsst.multiprofit.observationconfig`, :py:mod:`lsst.multiprofit.sourceconfig`, and finally :py:mod:`lsst.multiprofit.modelconfig` (which relies on the former three modules).

Each class in these modules is serializable and can be used to construct an equivalent :py:mod:`lsst.gauss2d` or :py:mod:`lsst.gauss2d.fit` object. The classes are designed for model configuration and can be used to provide initial parameter values, transformations, etc. They are not used or updated at runtime.

.. _lsst.multiprofit-parameter-component-configs:

Parameter and Component Configuration
======================================

Gauss2DFit components are configured by :py:class:`lsst.multiprofit.ComponentConfig` subclasses, while individual parameters are configured by
:py:class:`lsst.multiprofit.ParameterConfig` subclasses. These classes can be used to configure components for both PSF and source models.

In most cases, model fitting classes will provide ways to initialize most parameters before fitting, so the initial values of most parameters are not critical; however, they should still be reasonable defaults. For example, variable sizes and fluxes should not be set to zero.

.. _lsst.multiprofit-source-configs:

Source Configuration
====================

:py:class:`lsst.multiprofit.SourceConfig` similarly implements configuration for a single :py:mod:`lsst.gauss2d.fit` source. Source configurations are composed of instances of the intermediate :py:class:`lsst.multiprofit.ComponentGroupConfig` class. This class has no equivalent in Gauss2DFit - it is an additional abstraction to allow for a limited subset of the model constraints allowed by components sharing model parameters. The most important use is allowing components to share the same centroid. Future improvements may allow for size, shape, or flux/color constraints, although the latter may instead be implemented as priors.

.. _lsst.multiprofit-observation-model-configs:

Observation and Model Configuration
===================================

:py:class:`lsst.multiprofit.ObservationConfig` provides configuration options for observations, including the band and pixel size. The class maps directly onto its Gauss2DFit equivalent, as does :py:class:`lsst.multiprofit.ModelConfig`.

.. _lsst.multiprofit-examples-configs:

Examples
========
MultiProFit config classes are used extensively in the provided catalog fitters; these serve as example usage.
