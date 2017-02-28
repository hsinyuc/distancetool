"""
Calculate the horizon and range for a given 
binary. The horizon is defined as the distance at which a face-on and overhead (ideally located) binary is detected 
with SNR=snr_threshold with single IFO. It represents the farthest this binary could be detected above threshold. 

Usage:

find_horizon_range(m1,m2,asdfile,approx=ls.IMRPhenomD)

Input--
m1,m2: binary component masses in solar mass
asdfile: noise curve file [frequency, strain]

Optional inputs--
approx: frequency domain waveform. Default is IMRPhenomD.

Output--
Horizon redshift, 
Horizon luminosity distance (Mpc), 
50% of the detected sources lie within the luminosity distance (Mpc), 
90% of the detected sources lie within the luminosity distance (Mpc), 
detectable comoving volume (Mpc^3)

Author: 
Hsin-Yu Chen (hsin-yu.chen@ligo.org)

"""


from numpy import *
import cosmolopy.distance as cd
import lal
import lalsimulation as ls
from scipy.optimize import leastsq
from scipy.interpolate import interp1d

global snr_th,cosmo
snr_th=8.
cosmo = {'omega_M_0':0.308, 'omega_lambda_0':0.692, 'omega_k_0':0.0, 'h':0.678}

##find redshift for a given luminosity distance
def findzfromDL(z,DL):
    return DL-cd.luminosity_distance(z, **cosmo)
    
    
##estimate the horizon for recursive evaluation in the main code
def horizon_dist_eval(orig_dist,snr,z0):
    guess_dist=orig_dist*snr/snr_th
    guess_redshift,res=leastsq(findzfromDL,z0,args=guess_dist)
    return guess_redshift[0],guess_dist

##generate the waveform    
def get_htildas(m1,m2,dist,
		   phase=0., 
		   df=1e-2,  
		   s1x=0.0, 
		   s1y=0.0, 
		   s1z=0.0, 
		   s2x=0.0, 
		   s2y=0.0, 
		   s2z=0.0, 
		   fmin=10.,
		   fmax=0.,
		   fref=10.,  
		   iota=0., 
		   lambda1=0., 
		   lambda2=0., 
		   waveFlags=None, 
		   nonGRparams=None, 
		   amp_order=0, 
		   phase_order=-1, 
		   approx=ls.IMRPhenomD,
		   ):
	hplus_tilda, hcross_tilda = ls.SimInspiralChooseFDWaveform(phase, df, m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, fmin, fmax, fref, dist*(1E6 * ls.lal.PC_SI), iota, lambda1, lambda2, waveFlags, nonGRparams, amp_order, phase_order, approx)
	freqs=array([hplus_tilda.f0+i*hplus_tilda.deltaF for i in arange(hplus_tilda.data.length)])
	return hplus_tilda.data.data,hcross_tilda.data.data,freqs

#take the h+ component to calculate SNR
def compute_horizonSNR(hplus_tilda,psd_interp,fsel,df):
	return sqrt(4.*df*sum(abs(hplus_tilda[fsel])**2/psd_interp))        

#calculate the horizon distance/redshift
def find_horizon_range(m1,m2,asdfile,approx=ls.IMRPhenomD):

	fmin=10.
	fref=10.
	df=1e-2
	
	input_freq,strain=loadtxt(asdfile,unpack=True)
	minimum_freq=maximum(10.,min(input_freq))
	maximum_freq=minimum(5000.,max(input_freq))
	interpolate_psd = interp1d(input_freq, strain**2)	
		
	#initial guess of horizon redshift and luminosity distance
	z0=0.1
	input_dist=cd.luminosity_distance(z0,**cosmo)	
	hplus_tilda,hcross_tilda,freqs= get_htildas((1.+z0)*m1,(1.+z0)*m2 ,input_dist,iota=0.,fmin=fmin,fref=fref,df=df,approx=approx)
	fsel=logical_and(freqs>minimum_freq,freqs<maximum_freq)
	psd_interp = interpolate_psd(freqs[fsel]) 	
	input_snr=compute_horizonSNR(hplus_tilda,psd_interp,fsel,df)

	input_redshift=z0; guess_snr=0    
	#evaluate the horizon recursively
	while abs(guess_snr-snr_th)>snr_th*0.001: #require the error within 0.1%
		guess_redshift,guess_dist=horizon_dist_eval(input_dist,input_snr,input_redshift) #horizon guess based on the old SNR		
		
		#observer frame or source frame masses
		hplus_tilda,hcross_tilda,freqs= get_htildas((1.+guess_redshift)*m1,(1.+guess_redshift)*m2 ,guess_dist,iota=0.,fmin=fmin,fref=fref,df=df,approx=approx)
		fsel=logical_and(freqs>minimum_freq,freqs<maximum_freq)
		psd_interp = interpolate_psd(freqs[fsel]) 		
		guess_snr=compute_horizonSNR(hplus_tilda,psd_interp,fsel,df) #calculate the new SNR

		input_snr=guess_snr
		input_redshift=guess_redshift
		input_dist=guess_dist
	horizon_redshift=guess_redshift
	
	#sampled universal antenna power pattern for code sped up
	w_sample,P_sample=genfromtxt("../data/Pw_single.dat",unpack=True)
	P=interp1d(w_sample, P_sample,bounds_error=False,fill_value=0.0)
	n_zstep=100

	z,dz=linspace(horizon_redshift,0,n_zstep,endpoint=False,retstep=True)
	dz=abs(dz)
	unit_volume=zeros(size(z))
	for i in range(0,size(z)):	
		hplus_tilda,hcross_tilda,freqs = get_htildas((1.+z[i])*m1,(1.+z[i])*m2 ,cd.luminosity_distance(z[i],**cosmo) ,fmin=fmin,fref=fref,df=df,approx=approx)
		fsel=logical_and(freqs>minimum_freq,freqs<maximum_freq)
		psd_interp = interpolate_psd(freqs[fsel])  
		optsnr_z=compute_horizonSNR(hplus_tilda,psd_interp,fsel,df)
		w=snr_th/optsnr_z
		unit_volume[i]=(cd.comoving_volume(z[i]+dz/2.,**cosmo)-cd.comoving_volume(z[i]-dz/2.,**cosmo))/(1.+z[i])*P(w)
	vol_sum=sum(unit_volume)
	#Find out the redshifts that 50%/90% of the sources lie within
	z50=max(z[where(cumsum(unit_volume)>=0.5*vol_sum)])
	z90=max(z[where(cumsum(unit_volume)>=0.1*vol_sum)])
	
	return horizon_redshift,cd.luminosity_distance(horizon_redshift,**cosmo), cd.luminosity_distance(z50,**cosmo),  cd.luminosity_distance(z90,**cosmo), vol_sum
	


