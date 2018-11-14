Duke coupled heavy-ion collision bulk and heavy-flavor event generator (hic-hvqgen)
===================================================================================

An ebe package for local tests.

Requirements:

  Pythia8230 installed in system

  cmake2.8+
  
  FORTRAN

  c++11, boost, hdf5, gsl

  python3 with h5py package


For the medium evolution, use the Duke hic-eventgen package:
https://github.com/Duke-QCD/hic-eventgen

   TRENTo + freestream + (2+1D)osu-hydro + frzout + UrQMD

The coupling of heavy-flavor:

   TRENTo/Pythia + Duke Lido model + frag & Recomb + UrQMD

To build packages into run/

.. code::

  ./makepkg run

Then go to run/ and run event

.. code::
  
  cd run
  ./job-wrapper inputfile

