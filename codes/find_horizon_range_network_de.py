"""
Calculate the horizon, volume and distance measure for a given type of binary. 
The horizon is defined as the furthest distance of an optimally orientated binary a detector network can detect.

Usage:

find_horizon_range(m1,m2,network,asdfile,pwfile,approx=ls.IMRPhenomD)

Input--
m1,m2: binary component masses in solar mass
network: list of detector acronym in the network, e.g. ['H','L','V']. 'H': LIGO-Hanford  'L': LIGO-Livingston 'V': Virgo 'J': KAGRA 'I':LIGO-India
asdfile: list of noise curve file [frequency, strain].
pwfile: antenna power pattern file depending on the network.

Optional inputs--
approx: frequency domain waveform. Default is IMRPhenomD.

Output--
Range (Mpc);
Redshift at which the detector network can detect 50% of the uniformly distributed sources;
Redshift at which the detector network can detect 10% of the uniformly distributed sources;
Redshift of the horizon;
Constant comoving time volume (Gpc^3);
Redshift within which 50% of the detected sources lie;
Redshift within which 90% of the detected sources lie;
Redshift within which 50% of the detected sources lie, the source distribution follows a star formation rate;
Redshift within which 90% of the detected sources lie, the source distribution follows a star formation rate;
Average redshift of the detected sources;
Average redshift of the detected sources, the source distribution follows a star formation rate.
 

Author: 
Hsin-Yu Chen (hsin-yu.chen@ligo.org)

"""


from numpy import *
#import cosmolopy.distance as cd
import dedist as de
import lal
import lalsimulation as ls
from scipy.optimize import leastsq
from scipy.interpolate import interp1d

global snr_th,cosmo
snr_th=12.
#cosmo = {'omega_M_0':0.308, 'omega_lambda_0':0.692, 'omega_k_0':0.0, 'h':0.678}
cosmo = {'omega_M_0':0.3065, 'omega_lambda_0':0.6935, 'omega_k_0':0.0, 'h':0.679}

##find redshift for a given luminosity distance
def findzfromDL(z,DL):
    return DL-de.luminosity_distance_de(z, **cosmo)
    
    
##estimate the horizon for recursive evaluation in the main code
def horizon_dist_eval(orig_dist,snr,z0):
    guess_dist=orig_dist*snr/snr_th
    guess_redshift,res=leastsq(findzfromDL,z0,args=guess_dist)
    #return guess_redshift[0],guess_dist
    return guess_redshift[0],de.luminosity_distance_de(guess_redshift[0], **cosmo)

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
		   fmin=1.,
		   fmax=0.,
		   fref=1.,  
		   iota=0., 
		   longAscNodes=0.,
		   eccentricity=0.,
		   meanPerAno=0.,
		   LALpars=None,
		   approx=ls.IMRPhenomD
		   ):
	#hplus_tilda, hcross_tilda = ls.SimInspiralChooseFDWaveform(phase, df, m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, fmin, fmax, fref, dist*(1E6 * ls.lal.PC_SI), iota, lambda1, lambda2, waveFlags, nonGRparams, amp_order, phase_order, approx)
	hplus_tilda, hcross_tilda = ls.SimInspiralChooseFDWaveform(m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, dist*(1E6 * ls.lal.PC_SI),iota,phase, longAscNodes,eccentricity,meanPerAno,df,fmin, fmax, fref,LALpars,approx)
	freqs=array([hplus_tilda.f0+i*hplus_tilda.deltaF for i in arange(hplus_tilda.data.length)])
	return hplus_tilda.data.data,hcross_tilda.data.data,freqs

det_index={'H':lal.LALDetectorIndexLHODIFF,'L':lal.LALDetectorIndexLLODIFF,'V':lal.LALDetectorIndexVIRGODIFF,'J':lal.LALDetectorIndexKAGRADIFF,'I':lal.LALDetectorIndexLIODIFF }
def getDetResp(detector,ra,dec,time=0.,psi=0.):
    # Set time
    gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(time))    
    #Get Detector Responses
    det_response = lal.CachedDetectors[det_index[detector]].response    
    # Get Fplus, Fcross
    fplus, fcross = lal.ComputeDetAMResponse(det_response, ra, dec, psi, gmst)        
    return fplus,fcross

#Calculate maximum SNR at the coordinate (ra,dec,psi)
def compute_horizonSNR(hplus_tilda,hcross_tilda,network,ra,dec,psi,psd_interp,fsel,df):
    snrsq=0
    for detector in range(0,size(network)):
        fplus,fcross=getDetResp(network[detector],ra,dec,psi=psi)
        h_tilda = fplus*hplus_tilda + fcross*hcross_tilda
        snrsq+= 4.*df*sum(abs(h_tilda[fsel[detector]])**2/psd_interp[detector])     
    return sqrt(snrsq)

def sfr(z):
    return 0.015*(1.+z)**2.7/(1.+((1.+z)/2.9)**5.6) #msun per yr per Mpc^3


#calculate the horizon distance/redshift
def find_horizon_range(m1,m2,network,asdfile,pwfile,approx=ls.IMRPhenomD):

	fmin=10.
	fref=10.
	df=1.0
	
	ra,dec,psi,iota=genfromtxt('horizon_coord_'+pwfile+'.txt',unpack=True)
	
	psdinterp_dict={}
	minimum_freq=zeros(size(network)); maximum_freq=zeros(size(network))
	for detector in range(0,size(network)):	
		input_freq,strain=loadtxt(asdfile[detector],unpack=True,usecols=[0,1])
		minimum_freq[detector]=maximum(min(input_freq),fmin)
		maximum_freq[detector]=minimum(max(input_freq),5000.)
		psdinterp_dict[network[detector]] = interp1d(input_freq, strain**2)	
	#initial guess of horizon redshift and luminosity distance
	z0=1.0
	input_dist=de.luminosity_distance_de(z0,**cosmo)	
	hplus_tilda,hcross_tilda,freqs= get_htildas((1.+z0)*m1,(1.+z0)*m2 ,input_dist,iota=iota,fmin=fmin,fref=fref,df=df,approx=approx)

	fsel=list(); psd_interp=list()
	for detector in range(0,size(network)):	
		fsel.append(logical_and(freqs>minimum_freq[detector],freqs<maximum_freq[detector]))
		psd_interp.append(psdinterp_dict[network[detector]](freqs[fsel[detector]]))
	input_snr=compute_horizonSNR(hplus_tilda,hcross_tilda,network,ra,dec,psi,psd_interp,fsel,df)

	input_redshift=z0; guess_snr=0; njump=0    
	#evaluate the horizon recursively
	while abs(guess_snr-snr_th)>snr_th*0.001 and njump<10: #require the error within 0.1%
		try:
			guess_redshift,guess_dist=horizon_dist_eval(input_dist,input_snr,input_redshift) #horizon guess based on the old SNR				
			hplus_tilda,hcross_tilda,freqs= get_htildas((1.+guess_redshift)*m1,(1.+guess_redshift)*m2 ,guess_dist,iota=iota,fmin=fmin,fref=fref,df=df,approx=approx)
		except:
			njump=10
			print("Will try interpolation.")							
		fsel=list(); psd_interp=list()
		for detector in range(0,size(network)):	
			fsel.append(logical_and(freqs>minimum_freq[detector],freqs<maximum_freq[detector]))
			psd_interp.append(psdinterp_dict[network[detector]](freqs[fsel[detector]]))		
		guess_snr=compute_horizonSNR(hplus_tilda,hcross_tilda,network,ra,dec,psi,psd_interp,fsel,df) #calculate the new SNR

		input_snr=guess_snr
		input_redshift=guess_redshift
		input_dist=guess_dist
		njump+=1
	horizon_redshift=guess_redshift

	#at high redshift the recursive jumps lead to too big a jump for each step, and the recursive loop converge slowly.
	#so I interpolate the z-SNR curve directly.	
	if njump>=10:
		print("Recursive search for the horizon failed. Interpolation instead.")
		try:
			interp_z=linspace(0.001,100,200); interp_snr=zeros(size(interp_z))
			for i in range(0,size(interp_z)): 
				hplus_tilda,hcross_tilda,freqs= get_htildas((1.+interp_z[i])*m1,(1.+interp_z[i])*m2 ,de.luminosity_distance_de(interp_z[i],**cosmo),iota=iota,fmin=fmin,fref=fref,df=df,approx=approx)
				fsel=list(); psd_interp=list()
				for detector in range(0,size(network)):	
					fsel.append(logical_and(freqs>minimum_freq[detector],freqs<maximum_freq[detector]))
					psd_interp.append(psdinterp_dict[network[detector]](freqs[fsel[detector]]))		
				interp_snr[i]=compute_horizonSNR(hplus_tilda,hcross_tilda,network,ra,dec,psi,psd_interp,fsel,df)
			interpolate_snr = interp1d(interp_snr[::-1],interp_z[::-1])
			horizon_redshift= interpolate_snr(snr_th)	
		except RuntimeError: #If the sources lie outside the given interpolating redshift the sources can not be observe, so I cut down the interpolation range.
			print("some of the SNR at the interpolated redshifts cannot be calculated.")
			interpolate_snr = interp1d(interp_snr[::-1],interp_z[::-1])
			horizon_redshift= interpolate_snr(snr_th)	
		except ValueError:	#horizon outside the interpolated redshifts. Can potentially modify the interpolation range, but we basically can not observe the type of source or the source has to be catastrophically close.
			print("Horizon further than z=100 or less than z=0.001. Need to modify the interpolated redshift range.")
			return	
		
	#sampled universal antenna power pattern for code sped up
	w_sample,P_sample=genfromtxt('pw_'+pwfile+'.txt',unpack=True)
	P=interp1d(w_sample, P_sample,bounds_error=False,fill_value=0.0)
	n_zstep=5000
	print("horizon_redshift",horizon_redshift)
	z,dz=linspace(horizon_redshift,0,n_zstep,endpoint=False,retstep=True)
	dz=abs(dz)
	unit_volume=zeros(size(z)); compensate_detect_frac=zeros(size(z))
	for i in range(0,size(z)):	
		hplus_tilda,hcross_tilda,freqs = get_htildas((1.+z[i])*m1,(1.+z[i])*m2 ,de.luminosity_distance_de(z[i],**cosmo),iota=iota,fmin=fmin,fref=fref,df=df,approx=approx)
		fsel=list(); psd_interp=list()
		for detector in range(0,size(network)):	
			fsel.append(logical_and(freqs>minimum_freq[detector],freqs<maximum_freq[detector]))
			psd_interp.append(psdinterp_dict[network[detector]](freqs[fsel[detector]]))					
		optsnr_z=compute_horizonSNR(hplus_tilda,hcross_tilda,network,ra,dec,psi,psd_interp,fsel,df)
		w=snr_th/optsnr_z
		compensate_detect_frac[i]=P(w)		
		unit_volume[i]=(de.comoving_volume_de(z[i]+dz/2.,**cosmo)-de.comoving_volume_de(z[i]-dz/2.,**cosmo))/(1.+z[i])*P(w)
	
	#Find out the redshift at which we detect 50%/90% of the sources at the redshift
	z_reach50=max(z[where(compensate_detect_frac>=0.5)])
	z_reach90=max(z[where(compensate_detect_frac>=0.1)])

	vol_sum=sum(unit_volume)
	#Find out the redshifts that 50%/90% of the sources lie within assuming constant-comoving-rate density
	z50=max(z[where(cumsum(unit_volume)>=0.5*vol_sum)])
	z90=max(z[where(cumsum(unit_volume)>=0.1*vol_sum)])

    
	#Find out the redshifts that 50%/90% of the sources lie within assuming star formation rate
	sfr_vol_sum=sum(unit_volume*sfr(z)/sfr(0))
	sfr_z50=max(z[where(cumsum(unit_volume*sfr(z)/sfr(0))>=0.5*sfr_vol_sum)])
	sfr_z90=max(z[where(cumsum(unit_volume*sfr(z)/sfr(0))>=0.1*sfr_vol_sum)])

    
	#average redshift
	z_mean=sum(unit_volume*z)/vol_sum
	sfr_z_mean=sum(unit_volume*sfr(z)/sfr(0)*z)/sfr_vol_sum
		
	#return (3.*vol_sum/4./pi)**(1./3.),z_reach50,z_reach90,horizon_redshift,vol_sum/1E9,z50,z90,sfr_z50,sfr_z90,z_mean,sfr_z_mean  
	return horizon_redshift,vol_sum/1e9,sfr_vol_sum/1e9,(3.*vol_sum/4./pi)**(1./3.),sfr_z50,z,compensate_detect_frac,unit_volume,unit_volume*sfr(z)/sfr(0)
