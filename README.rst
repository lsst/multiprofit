MultiProFit
###########

*multiprofit* is a Python astronomical source modelling code made by and for
`LSST Data Management <https://www.lsst.org/about/dm>`_. MultiProFit means Multiple Profile Fitting.
The multi-aspects can be multi-object, multi-component, multi-band, multi-instrument, and someday multi-epoch.

*multiprofit* can fit any kind of imaging data while modelling sources and convolution kernels as
Gaussian mixtures, including approximations to Sersic profiles. It is fast and flexible and comes with
interfaces for batch fitting of images with corresponding detection catalogs.

*multiprofit* requires Python 3 and uses `gauss2d <https://github.com/lsst/gauss2d/>`_ and
`gauss2d_fit <https://github.com/lsst/gauss2d_fit/>`_ to evaluate models. It can be set up as part of the
`Rubin/LSST Science Pipelines <https://pipelines.lsst.io/>`_ using
`eups <https://github.com/RobertLuptonTheGood/eups>`_, or installed as a standalone package with pip. It is
recommended but not required to use a conda environment with the latter approach.

History
#######

MultiProFit was originally inspired by `ProFit <https://www.github.com/ICRAR/ProFit>`_. It was initially built
to use either `GalSim <https://github.com/GalSim-developers/GalSim/>`_ or
`libprofit <https://github.com/ICRAR/libprofit/>`_ via `pyprofit <https://github.com/ICRAR/pyprofit/>`_ as
rendering backends but now exclusively uses gauss2d.



.. todo *multiprofit* is available in `PyPI <https://pypi.python.org/pypi/multiprofit>`_
   .. and thus can be easily installed via::

.. pip install multiprofit
