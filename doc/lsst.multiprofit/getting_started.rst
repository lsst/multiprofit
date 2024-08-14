.. _lsst.multiprofit.getting-started:

===============
Getting Started
===============

At its core, MultiProFit is a convenient and more user-friendly interface to :py:mod:`lsst.gauss2d.fit`.
MultiProFit has three layers of abstraction to simplify access to Gauss2DFit's interfaces.

The first layer is built around the :py:class:`lsst.multiprofit.Modeller` class, which provides access to Python optimizers and is currently the only provided method to fit models (Gauss2DFit does not yet integrate any C++ optimizers but may do so in the future). This class interfaces directly with :py:class:`lsst.gauss2d.fit.ModelD` instances (or ModelF, though double precision is recommended).

The second layer is a series of :py:class:`lsst.pex.config.Config` classes provided in :py:mod:`lsst.multiprofit.componentconfig`, :py:mod:`lsst.multiprofit.observationconfig`, :py:mod:`lsst.multiprofit.sourceconfig`, and finally :py:mod:`lsst.multiprofit.modelconfig` (which relies on the former three modules). Each class in these modules is serializable and can be used to construct an equivalent :py:mod:`lsst.gauss2d` or :py:mod:`lsst.gauss2d.fit` object. Users should attempt to use these classes first; in most cases, the additional flexibility in model specification from the first layer is not needed.

The final layer is the batch fitting interfaces provided in :py:mod:`lsst.multiprofit.fitting`. These classes provide a way to fit many sources in serial, with users providing a catalog of detections and defining functions to read the data for each source.

The remainder of this guide will cover the first layer and basic usage of the second.

.. _lsst.multiprofit-sources-and-components:

Sources and Components
======================

In Gauss2DFit, Components are the fundamental building block, whereas Sources are user-defined collections of one or more components.

.. _lsst.multiprofit-components:

Components
----------

A component is defined by a centroid, total flux, and profile parameters, which at the minimum include ellipsoidal shape parameters and may also include radial profile parameters.
A single Sersic profile is a common choice for a simple galaxy model:

>>> import lsst.gauss2d.fit as g2f
>>>
>>> comp = g2f.SersicMixComponent()

All Gauss2D(Fit) objects can be printed, albeit with minimal formatting:

>>> print(str(comp))
SersicMixComponent(ellipse=SersicParametricEllipse(size_x=ReffXParameterD(value=0.000000, ), size_y=ReffYParameterD(value=0.000000, ), rho=RhoParameterD(value=0.000000, )), centroid=CentroidParameters(x=CentroidXParameterD(value=0.000000, ), y=CentroidYParameterD(value=0.000000, )), integralmodel=LinearIntegralModel(data={Channel(name=None): IntegralParameterD(value=1.000000, ),}), sersicindex=SersicIndexParameterD(value=0.500000, ))

Components have parameters, which can be set individually:

>>> comp.ellipse.size_x = 1.0
>>> comp.ellipse.size_y = 2.0
>>> comp.sersicindex = 1.0

Component parameters can be initialized directly, but this is more verbose than mutating values after the fact.
This is one of the reasons why using the second Config layer can be more convenient.

.. _lsst.multiprofit-sources:

Sources
-------

Sources are constructed from a list of components:

>>> src = g2f.Source([comp])
>>> print(src)
Source(components=[SersicMixComponent(ellipse=SersicParametricEllipse(size_x=ReffXParameterD(value=1.000000, ), size_y=ReffYParameterD(value=2.000000, ), rho=RhoParameterD(value=0.000000, )), centroid=CentroidParameters(x=CentroidXParameterD(value=0.000000, ), y=CentroidYParameterD(value=0.000000, )), integralmodel=LinearIntegralModel(data={Channel(name=None): IntegralParameterD(value=1.000000, ),}), sersicindex=SersicIndexParameterD(value=1.000000, )),])

.. _lsst.multiprofit-models-and-observations:

Models and Observations
=======================

MultiProFit's :py:class:`lsst.multiprofit.Modeller` requires a :py:class:`lsst.gauss2d.fit.ModelD` object, which in turn needs at least one :py:class:`lsst.gauss2d.fit.ObservationD`.

.. _lsst.multiprofit-observations:

Observations
------------

An Observation corresponds to a typical astronomical exposure and consists of an image, its errors (stored as inverse sigma for efficiency), a boolean inverse mask (where values of 1 indicate good pixels), and a channel. In this case, we will use the default channel NONE.

>>> import lsst.gauss2d as g2d
>>> import numpy as np
>>>
>>> shape = np.array((15, 17))
>>> # Center the component
>>> comp.centroid.y, comp.centroid.x = shape/2.
>>> image_data = np.zeros(shape)
>>> sigma_inv_data = np.ones(shape)
>>> mask_inv_data = np.ones(shape, dtype=bool)
>>> channel = g2f.Channel.NONE
>>> # Initializing these in-line with the observation constructor makes a copy
>>> # for some reason, and image.data is a read-only attr so numpy operators
>>> # like += can't be used on it directly
>>> image = g2d.ImageD(image_data)
>>> sigma_inv = g2d.ImageD(sigma_inv_data)
>>> mask_inv = g2d.ImageB(mask_inv_data)
>>> observation = g2f.ObservationD(image=image, sigma_inv=sigma_inv, mask_inv=mask_inv, channel=channel)

.. _lsst.multiprofit-models:

Models
------

Models consist of one or more Sources, one or more Observations, the corresponding PSF model for each observation, and any number of priors. For our first PSF model, we will make a unit Gaussian model only.

>>> psf_gaussian = g2f.GaussianComponent()
>>> psf_gaussian.ellipse.sigma_x = 1
>>> psf_gaussian.ellipse.sigma_y = 1
>>> psf_model = g2f.PsfModel([psf_gaussian])
>>>
>>> model = g2f.ModelD(data=g2f.DataD([observation]), psfmodels=[psf_model], sources=[src])

Once constructed, models must have an evaluator set up. This design makes repeated model evaluations more efficient. The simplest evaluator makes an image of the model.

>>> model.setup_evaluators(g2f.EvaluatorMode.image)
>>> _ = model.evaluate()
>>> output = model.outputs[0]

Now we will use this output image to initialize our observation:

>>> rng = np.random.default_rng(1)
>>> background = 0.01
>>> gain=1e5
>>> counts = rng.poisson((background + output.data)*gain)
>>> image_data += counts/gain - background
>>> sigma_inv_data += np.sqrt(counts)/gain

The log-likelihood (and prior log likelihood, which will be zero with no priors specified) can now be evaluated:

>>> import matplotlib.pyplot as plt
>>>
>>> model.setup_evaluators(g2f.EvaluatorMode.loglike_image)
>>> print([f"{x:.4e}" for x in model.evaluate()])
['-1.4236e-05', '0.0000e+00']

.. >>> from lsst.multiprofit.plotting import plot_model_rgb
.. >>> plot_model_rgb(model, stretch=1e-3)

.. _lsst.multiprofit-modellers:

Modellers
=========

The :py:class:`lsst.multiprofit.Modeller` class has minimal configuration options of its own.
Instead, the fit_model method takes  qa model and (optional) fit configuration:

>>> import lsst.multiprofit as mpf
>>> modeller = mpf.Modeller()

Finally, we are ready to fit:

>>> # The PSF model parameters must be fixed first
>>> # MultiProFit doesn't support fitting PSF model params
>>> for param in psf_model.parameters(): param.fixed = True
>>>
>>> result = modeller.fit_model(model)

The result object contains fit metadata, the result object from the optimizer (scipy in this case), and the best-fit parameters.
We can delete the more verbose metadata and print the remaining values:

>>> result_dict = dict(result)
>>> for key in ("inputs", "result"): del result_dict[key]
>>> print(result_dict.keys())
dict_keys(['config', 'params', 'params_best', 'n_eval_resid', 'n_eval_func', 'n_eval_jac', 'time_eval', 'time_run'])

More complete documentation for the second (Config classes) and third (batch fitting) layers is in progress.
In the meantime, MultiProFit's unit tests and examples can offer some inspiration.
