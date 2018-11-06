.. q2-extractor documentation master file, created by
   sphinx-quickstart on Mon Nov  5 14:11:15 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to q2_extractor
========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


* :ref:`genindex`
* :ref:`search`

Extractor
=============

Contains q2Extractor, a "plain" extractor class, 
mostly returning Pandas DataFrames from QIIME2 artifacts

.. automodule:: q2_extractor.Extractor
  :members:

MetaHCRExtractor
=================

An extractor class intended to shuttle QIIME2 results into the MetaHCR
project (https://github.com/metahcr/metahcr_v1).

.. automodule:: q2_extractor.MetaHCRExtractor
  :members:
