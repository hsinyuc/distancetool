import numpy
import scipy.integrate as si 

global c_light_Mpc_s, H100_s
c_light_Mpc_s = 29979245800./3.08568025e24
H100_s = 100. / 3.08568025e19  


def get_omega_k_0(**cosmo):
	if 'omega_k_0' in cosmo: 
		omega_k_0 = cosmo['omega_k_0'] 
	else: 
		omega_k_0 = 1. - cosmo['omega_M_0'] - cosmo['omega_lambda_0'] 
	return omega_k_0 


def e_z_de(z, **cosmo): 
    if 'w' in cosmo: 
        return (cosmo['omega_M_0'] * (1+z)**3. +  cosmo['omega_k_0'] * (1+z)**2. +  cosmo['omega_lambda_0'] * (1+z)**(3.*(1+cosmo['w'])) )**0.5 
    else: 
        return (cosmo['omega_M_0'] * (1+z)**3. +  cosmo['omega_k_0'] * (1+z)**2. +  cosmo['omega_lambda_0'])**0.5 
    
def luminosity_distance_de(z, **cosmo): 
    da = angular_diameter_distance_de(z, **cosmo) 
    return da * (1+z)**2. 

def angular_diameter_distance_de(z, z0 = 0, **cosmo): 
    omega_k = numpy.atleast_1d(get_omega_k_0(**cosmo)) 
    if (numpy.any(omega_k < 0) and not(z0 == 0)): 
        raise ValueError("Not implemented for Omega_k < 0 and z0 > 0") 

    dm2  = comoving_distance_transverse_de(z, **cosmo) 
    if z0 == 0: 
        return dm2 / (1. + z) 
    dm1 = comoving_distance_transverse_de(z0, **cosmo) 

    d_h_0 = hubble_distance_z_de(0.0, **cosmo) 
    term1 = dm1 * numpy.sqrt(1. + omega_k * (dm2/d_h_0)**2.) 
    term2 = dm2 * numpy.sqrt(1. + omega_k * (dm1/d_h_0)**2.) 
    da12 = (term2 - term1)/(1+z) # only for Omega_k > 0 
    return da12 

def comoving_distance_transverse_de(z, **cosmo):
    d_c = comoving_distance_de(z, 0.0, **cosmo) 
    omega_k_0 = get_omega_k_0(**cosmo) 
    if numpy.all(omega_k_0 == 0.0): 
        return d_c 
    d_h_0 = hubble_distance_z_de(0.0, **cosmo) 
    sqrt_ok0 = numpy.sqrt(numpy.abs(omega_k_0)) 
    if not numpy.isscalar(omega_k_0): 
        sqrt_ok0[omega_k_0 == 0.0] = 1.0 
    argument = sqrt_ok0 * d_c / d_h_0 
    factor = d_h_0 * (1./sqrt_ok0) 
    d_m = ((omega_k_0 > 0.0) * (factor * numpy.sinh(argument)) + (omega_k_0 == 0.0) * d_c + (omega_k_0 < 0.0) * (factor * numpy.sin(argument))) 
    return d_m 

def hubble_distance_z_de(z, **cosmo): 
    H_0 = cosmo['h'] * H100_s 
    return c_light_Mpc_s / (H_0 * e_z_de(z, **cosmo))     

def comoving_distance_de(z, z0 = 0, **cosmo): 
    if 'w' in cosmo: 
        w = cosmo['w'] 
    else: 
        w = -1. 
    dc_func =  numpy.vectorize(lambda z, z0, omega_M_0, omega_lambda_0, omega_k_0, h, w:  si.quad(_comoving_integrand_de, z0, z, limit=1000, args=(omega_M_0, omega_lambda_0, omega_k_0, h, w))) 
    d_co, err = dc_func(z, z0,  
    cosmo['omega_M_0'], 
    cosmo['omega_lambda_0'], 
    cosmo['omega_k_0'], 
    cosmo['h'], 
    w 
    ) 
    return d_co 

def _comoving_integrand_de(z, omega_M_0, omega_lambda_0, omega_k_0, h, w=-1.):
    e_z = (omega_M_0 * (1+z)**3. +  omega_k_0 * (1+z)**2. +  omega_lambda_0 * (1+z)**(3.*(1.+w)))**0.5 

    H_0 = h * H100_s 
    H_z =  H_0 * e_z 
    return c_light_Mpc_s / (H_z) 

def comoving_volume_de(z, **cosmo):
    dm = comoving_distance_transverse_de(z, **cosmo) 

    omega_k_0 = get_omega_k_0(**cosmo) 

    flat_volume = 4. * numpy.pi * dm**3. / 3. 
    if numpy.all(omega_k_0 == 0.0): 
        return flat_volume 
    d_h_0 = hubble_distance_z_de(0.0, **cosmo) 
    sqrt_ok0 = numpy.sqrt(numpy.abs(omega_k_0)) 
    dmdh = dm/d_h_0 
    argument = sqrt_ok0 * dmdh 
    f1 = 4. * numpy.pi * d_h_0**3. / (2. * omega_k_0) 
    f2 = dmdh * numpy.sqrt(1. + omega_k_0 * (dmdh)**2.) 
    f3 = 1./sqrt_ok0 
    if numpy.isscalar(omega_k_0): 
        if omega_k_0 > 0.0: 
            return f1 * (f2 - f3 * numpy.arcsinh(argument)) 
        elif omega_k_0 == 0.0: 
            return flat_volume 
        elif omega_k_0 < 0.0: 
            return f1 * (f2 - f3 * numpy.arcsin(argument)) 
    else: 
        b = numpy.broadcast(omega_k_0,z,dm) 
        Vc = numpy.zeros(b.shape) 
        m1 = numpy.ones(b.shape, dtype=bool) * (omega_k_0 > 0.0) 
        Vc[m1] = (f1 * (f2 - f3 * numpy.arcsinh(argument)))[m1] 
        m1 = numpy.ones(b.shape, dtype=bool) * (omega_k_0 == 0.0) 
        Vc[m1] = flat_volume[m1] 
        m1 = numpy.ones(b.shape, dtype=bool) * (omega_k_0 < 0.0) 
        Vc[m1] = (f1 * (f2 - f3 * numpy.arcsin(argument)))[m1] 
        return Vc 
    
    
