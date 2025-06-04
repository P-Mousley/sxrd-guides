HKL fractional positions
===========================


To calculate the expected HKL positions of a crystalline material:

1) Obtain the lattice parameters of the expected material (a,b,c) and calculate the values for Qx,Qy,Qz using the following equations:


 
    -  :math:`𝑄 = \frac{2\pi}{d}`.  here d is the crystal spacing which can be either a,b or c depending on which axis you are calculating. 
    -  Once you have the Q value for your expected material :math:`Q_{mat}` you can calculate the expected H,K or L position by doing :math:`Q_{pos}= \frac{Q_{mat}}{Q_{sub}}`  where :math:`Q_{sub}` is the corresponding Q value for the substrate material.
    - e.g. if your substrate has a c-lattice parameter of 1.8 Å and your expected material in your film has a c-lattice parameter of 2.0 Å then your expected positions will be as follows:
        - :math:`Q_{sub}=\frac{2\pi}{1.8}=3.49`
        - :math:`Q_{mat}=\frac{2\pi}{2.0}=3.14`
        -  Expected Q position in data :math:`Q_{pos} = \frac{3.14}{3.49} = 0.9` , therefore you would expect to see intensity at positions of multiples of 0.9 e.g. L=0.9, L= 1.8, L=2.7 etc

You can use this technique to estimate the fractional positions of signals in the H,K and L directions. A program like `VESTA`_ can show the powder diffraction plot for a material to let you know the relative strengths of the signals, so you will probably expect to only see some of the strongest signals. Once you have obtained HKL positions for all expected materials, you can compare with your dataset to see where intensity was actually measured. This can be used to assign intensities measures with the expected materials, and you can get a measure of any strain present e.g. if the signal expected at L=0.9 was seen at L=0.92 then you can calculate the difference in lattice parameter using the equations above in reverse

.. _VESTA: https://jp-minerals.org/vesta/