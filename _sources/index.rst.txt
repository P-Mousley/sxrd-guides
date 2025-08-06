Guide to surface x-ray diffraction analysis
==============================================

Here are some walkthroughs to aid in the analysis of surface x-ray diffraction measurements

.. list-table::
   :widths: 70 70 70
   :header-rows: 0

   * - **Crystal Truncation Rods**
     - **Reflectivity**
     - **In-plane omega scans**
   * - 
       .. image:: images/example_CTR.png
          :width: 220px
          :target: `ctr-section`_
          :alt: Crystal Truncation Rods
     - 
       .. image:: images/example_reflectivity.png
          :width: 220px
          :target: `reflectivity-section`_
          :alt: Reflectivity
     - 
       .. image:: images/example_inplane_omegas.png
          :width: 220px
          :target: `omega-section`_
          :alt: In-plane omega scans

.. list-table::
   :widths: 70 70 70
   :header-rows: 0

   * - **Specular 00L rods**
     - **GIWAXS**
     - **GISAXS**
   * - 
       .. image:: images/example_CTR.png
          :width: 220px
          :target: `ctr-section`_
          :alt: Crystal Truncation Rods
     - 
       .. image:: images/example_reflectivity.png
          :width: 220px
          :target: `reflectivity-section`_
          :alt: Reflectivity
     - 
       .. image:: images/example_inplane_omegas.png
          :width: 220px
          :target: `omega-section`_
          :alt: In-plane omega scans

.. _reflectivity-section:

Reflectivity ( low angle :math:`\theta-2\theta scans`)
--------------------------------------------------------

These are specular measurements taken at low :math:`\theta` values near the critical edge of your sample, and can be analysed with the following steps:
 - Reduce the data to a 1D reflecitivty profile, which can be done using the `islatu package`_
 - use a program such as GenX to model the reflectiity profile 


Specular 00L rod ( :math:`\theta-2\theta scans`)
-------------------------------------------------

These are specular measurements taken at higher :math:`\theta` values, away from the critical edge of your sample. These can be analysed using the following steps:
 - Reduce the data to a 1D profile,  which can be done using the `islatu package`_
 - Peak positions and widths can be used for quantitative analysis of your 00L profile
 - Out-of-plane strain can be calculated by analysing `HKL positions`_
 - The 00L profile can be loaded into fitting programs such as `ROD`_ for full structure fitting. There are also options for `automating ROD`_  
 
 .. note:: 
   
   For a full structural fit, the specular 00L profile is usually combined with off-specular crystal truncation rod measurements. 

.. _ctr-section:

Crystal truncation rod data
----------------------------

These are measurements taken with a fixed incident angel, and along the L direction at a variety of HK positions, which can take both integer and fraction positiosn of H and K. These can be analysed using the following steps:
 - Map the data from raw images into HKL space, which can be done using the `fast_rsm package`_
 - Reduce the 3D HKL volumes data into 1D profiles, which can be done using the binoviewer package (under development)
 - The off-specular CTR profiles can be loaded into fitting programs such as `ROD`_ for full structure fitting. There are also options for `automating ROD`_ 

.. _omega-section:

In-plane omega scans
-----------------------

These are measurements taken at fixed very low L value, and are collected by rocking the sample around it azimuthal axis. These can be analysed using the following steps:
 - Reduce the raw data to individual intensities for each HK position. See extra information on reducing `omega scan data`_
 - The 00L profile can be loaded into fitting programs such as `ROD`_ for full structure fitting - Note: these are usually combined with off-specular crystal truncation rod measurements. There are also options for `automating ROD`_ 

 .. note:: 
   
   For a full structural fit, the in-plane omega scans are usually combined with off-specular crystal truncation rod measurements

Grazing incidence wide angle x-ray scattering
----------------------------------------------

These are measurements taken with a large detector position close to the sample, and usually require some form of azimuthal integration. These can be processed in the following steps:
 - obtain calibration information so you know the detector distance from your sample
 - azimuthally integrated the data or map the data into 2D co-ordinate system (Q_parallel Vs Q_Perpendicular, exit angles)

 both of these steps can be done using the pyFAI python package. For Diamond data sets the `fast_rsm package`_ already makes use of the pyFAI module. 


Links to website resources
---------------------------

ANArod project from ESRF - https://www.esrf.fr/computing/scientific/joint_projects/ANA-ROD/ 




Links to extra pdf resources
------------------------------

 - `ROD manual 2018 <_static/rodmanual_2018.pdf>`_

 - `from beamtime to structure factors- E.Vlieg 2019 <_static/beamtime_to_SF_EVlieg.pdf>`_



.. toctree::
   :hidden:
   :maxdepth: 2

   omegascans
   fractionalhkl
   autofitting_rod
   symmetry_averaging
   reading_files

.. _Guide for analysing omega scan data: ./omegascans.html
.. _Calculating expected HKL positions from a known material: ./fractionalhkl.html
.. _Setting up automated fitting routines for ROD: ./autofitting_rod.html
.. _omega scan data: ./omegascans.html
.. _HKL positions: ./fractionalhkl.html
.. _automating ROD: ./autofitting_rod.html
.. _islatu package: https://islatu.readthedocs.io
.. _fast_rsm package: https://fast-rsm.readthedocs.io
.. _ROD: https://www.esrf.fr/computing/scientific/joint_projects/ANA-ROD/ 
.. _pyFAI: https://pyfai.readthedocs.io/