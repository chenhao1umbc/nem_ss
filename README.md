# nem_ss
Nerual EM for source separation
matlab version with real number toy data

Abandoned because matlab deeplearning toolbox does not support autogradient
of det, norm, inv, cell... the basic functions. It is over simple... It can 
handle some deep network, but when it is customized much. It will not work.
Also it does support complex number gradient, though we did not use complex 
number here.