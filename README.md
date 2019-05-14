# MoPy
MoPy is a python package for calculating seismic source parameters.
It is still very much a work in progress and not yet intended for non-development use.


## TODO

* Rethink `ChanelInfo` to allow greater flexibility in how windows are selected
    - Maybe this would be best left as a function?
    - Should be easy to use for amplitude estimations
    - Should be usable for duration estimations (eg in Rodriguez-pradilla and Eaton)
    - Could use different channel info for each of these?
    
* Check on how the continuous fft is simulated via dfft, 
    - Edwards et al (2010) indicate simply multiplying by sampling period will do it
      but then indicates noise spectra are normalized by non-zero padded sample number
      ratios (see page 409 first paragraph on left).
      However, this doesn't really make sense since, if we have properly
      simulated the *continuous* fourier transform amplitudes there should be no
      need for vector-length scaling. 
      
