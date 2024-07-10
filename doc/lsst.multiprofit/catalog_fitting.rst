.. _lsst.multiprofit.catalog-fitting:

===============
Catalog Fitting
===============

MultiProFit's third and final layer of data structures is the batch fitting interfaces provided in :py:mod:`lsst.multiprofit.fitting`. These fitters are designed to fit multiple sources in a co-spatial set of observations one-by-one, given a catalog of detections, and output a catalog of best-fit parameters and errors thereof, along with diagnostics.

As with the Model configuration classes, catalog fitters are configured with :py:class:`lsst.pex.config.Config` instances. The base class interfaces are provided in :py:mod:`lsst.multiprofit.fitting` and begin with :py:class:`lsst.multiprofit.CatalogFitterConfig`. Implementations must override subclasses :py:class:`lsst.multiprofit.CatalogExposureABC` to define methods to retrieve the data for a single row of the source catalog.

Unlike model configuration classes, though, these configurations have no corresponding Gauss2DFit data structures and need to store intermediate data. Fitters are therefore implemented to take inputs as :py:class:`pydantic.BaseModel` instances (providing runtime type checking and validation, amongst other benefits), with :py:class:`lsst.pex.config.Config` members for configuration.

.. _lsst.multiprofit-catalog-fitting-psf:

PSF Fitting
===========

The :py:mod:`lsst.multiprofit.fitting.fit_psf` module implements fitting a single Gaussian mixture PSF model at the position of each source in a catalog.

The :py:class:`lsst.multiprofit.CatalogPsfFitter` class' fit function takes two arguments: a :py:class:`lsst.multiprofit.CatalogExposurePsfABC` instance as input data and a :py:class:`lsst.multiprofit.CatalogPsfFitterConfigData`. The former class must have its abstract methods implemented - which typically involves calling code to render an image of the PSF - while the latter may have functions overridden if needed.

.. _lsst.multiprofit-catalog-fitting-source:

Source Fitting
==============

The :py:mod:`lsst.multiprofit.fitting.fit_source` module implements fitting a single PSF-convolved Gaussian mixture model at the position of each source in a catalog.

The :py:class:`lsst.multiprofit.CatalogSourceFitter` class' fit function takes two arguments: a :py:class:`lsst.multiprofit.CatalogExposureSourceABC` instance as input data and a :py:class:`lsst.multiprofit.CatalogSourceFitterConfigData`. Like the PSF fitter, the former class must have its abstract methods implemented and the latter may have functions overridden if needed. The abstract methods must provided an image of the source and also initialize the parameters of the PSF model.

.. _lsst.multiprofit-catalog-fitting-bootstrap:

Bootstrap Fitting
=================

The :py:mod:`lsst.multiprofit.fitting.fit_bootstrap_model` module implements boostrap fitting of multiple random realizations of a single PSF and/or source model. The :py:class:`CatalogExposurePsfBootstrap` and :py:class:`CatalogSourceFitterBootstrap` classes implement PSF and source fitting, respectively. These are designed to test MultiProFit's performance in terms of runtime and accuracy of recovered parameters and errors thereof.

Examples
========
MultiProFit's catalog fitting interfaces are implemented for Rubin-LSST data in `meas_extensions_multiprofit <https://github.com/lsst-dm/meas_extensions_multiprofit/>`_ (to be added as a Science Pipelines package in the near future).
