import numpy as np
#use correction factors which were implemented in matlab scananalysis
#area correction described in Vlieg - 1997 doi: 10.1107/S0021889897002537 , \
#assuming uniform illumination of entire sample so not included factor for shape of beam, may need to 
#add this in at some point - need to numerically calculate area
def areacorr(delta,gamma,alpha,specular=0):
    delta=np.radians(delta)
    gamma=np.radians(gamma)
    alpha=np.radians(alpha)
    
    if specular==0:
#         sin_betaout=np.cos(delta)*np.sin(gamma-alpha);
#         cosbetaout=np.arcsin(sinbetaout);
#         cosbetaout=np.cos(betaout);
#         area_cor= 1/(np.sin(delta)*cosbetaout);

#from correct.c code for ANA program -	cos_beta_out = sqrt(1-sqr(cos(DELTA/RAD)*sin((GAMMA-ALPHA)/RAD)))
#area = sin(DELTA/RAD)/cos_beta_out;
        sin_betaout=np.cos(delta)*np.sin(gamma-alpha);
        cos_betaout=np.sqrt(1-np.square(sin_betaout))
        area_cor=np.sin(delta)/cos_betaout
        


    elif specular==1:
        #possibly needs to be altered for 6 circle- see Vlieg - 1997 doi: 10.1107/S0021889897002537
        area_cor = np.sin(alpha)        
    return area_cor
#polarisation and intercept from 2005 - Schleputz work doi:10.1107/S0108767305014790
#and agrees with 1998 paper for 2+3 diffractometer -  DOI: 10.1107/S0021889897009990 
def polcorr(gamma,delta):
    pperp=1
    gamma=np.radians(gamma)
    delta=np.radians(delta)
    #to match correct.c code from ANA -
    # ---->  phor = 1. - sqr(sin(GAMMA/RAD) * cos(DELTA/RAD));
    #  pver = 1. - sqr(sin(DELTA/RAD));
    # polarization = 1. / (HPOLAR*phor+(1-HPOLAR)*pver)<-----;

    cp = 1/(pperp*(1-(np.cos(delta)**2)*np.sin(gamma)**2) + (1-pperp)*(1-np.sin(delta)**2));
    return cp

def intcorr(delta,gamma,alpha):
    delta=np.radians(delta)
    gamma=np.radians(gamma)
    alpha=np.radians(alpha)
#    ci = np.cos(delta)*np.sin((gamma-alpha))
#from correct.c code for ANA program - intercept = 1/sqrt(1-sqr(cos(DELTA/RAD)*sin((GAMMA-ALPHA)/RAD)));
    ci=1/(np.sqrt(1-np.square(np.cos(delta)*np.sin((gamma-alpha)))))

    return ci

#from 1998 paper for 2+3 diffractometer -  DOI: 10.1107/S0021889897009990 
#for 2+3 diffractometer setup (for I07 assuming this is omega+alpha for sample, and gamma,delta,nu for detector)
#with nu being kept fixed

#to match correct.c code for ANA -    if(SCANTYPE == rock)  lorentz = cos(ALPHA/RAD)*sin(DELTA/RAD);
def lorcorr(delta,alpha):
    delta=np.radians(delta)
    alpha=np.radians(alpha)
    cl=(np.sin(delta)*np.cos(alpha))
    return cl