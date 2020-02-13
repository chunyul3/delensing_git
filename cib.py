#############################################################
#this coude is about a figure. Which involve DES, WIDE, CIB, SPT, LSSS.
#For DES, WISE the W is Wg. which coresonging to different i.
#For C_kk the W is Wk
##############################################################
import numpy as np
#import pandas as pd
import math
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
import camb
from camb import model, initialpower
from camb import get_matter_power_interpolator
import sys, platform, os
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import leastsq
from scipy import interpolate 

h = 0.675
alpha = 1.27; beta = 1.02;
z0 = 0.5
nz = 100 #number of steps to use for the radial/redshift integration
kmax=10  #kmax to use
omega_m = 0.27
c = 1
beta_cib = 2
bc = 1
sigma_z = 2
zc = 2
T = 34 # k
#mu = 4955
h_p = 6.63 * 10**(-34) #j*s
k_b = 1.38 * 10**(-23) #j/k
mup = 4.955 * 10**12 #s**-1
#mu = 0.545*10**12
#mu = 150e9
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)

#For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.

results= camb.get_results(pars)
#chistar = results.conformal_time(0)- results.tau_maxvis
chistar = results.conformal_time(0)- results.conformal_time(1090)
chis = np.linspace(0,chistar,nz)
zs=results.redshift_at_comoving_radial_distance(chis)
dchis = (chis[2:]-chis[:-2])/2
chis = chis[1:-1]
dz = (zs[2:]-zs[:-2])/2
zs = zs[1:-1]
w = np.ones(chis.shape)

#Here for lensing we want the power spectrum of the Weyl potential.
PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
    hubble_units=False, k_hunit=False, kmax=kmax,
    var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zs[-1])


def Rou(l,suvy):
    rou_sq = C_ki(l,suvy) ** 2 / C_kk(l,suvy) / C_ii(l,suvy)
    rou = rou_sq ** 0.5
    return rou

def C_kk(l,suvy):
    k = (l + 0.5)/chis
    a = H(zs)**1 / (chis)**2 * Wk(zs) * Wk(zs) * PK.P(zs, k,grid = False)* (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
    return sum(a)


def C_ki(l,suvy):
    k = (l + 0.5)/chis
    a = H(zs)**1 / (chis)**2 * Wi(zs, suvy) * Wk(zs) * PK.P(zs, k,grid = False)* (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz / c
    sa = sum(a)
    return sa
        
def C_ii(l,suvy):
    k = (l + 0.5)/chis
    if suvy == 'CIB_pln':
        return 3500 * (l/3000) ** (-1.25)
    elif suvy == 'CIB_2':
        return 3500**1 * (l/3000) ** (-1.25) * 10**-23
    else:
        a = H(zs)**1 / (chis)**2 * Wi(zs,suvy) * Wi(zs,suvy) * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
        return sum(a)

def H(z):
    return results.h_of_z(z)    #Mpc-1

def dn(z, suvy):
    if suvy == 'LSST':        
        dn = b(z) * z**alpha * np.exp(-(z/z0_lsst)**beta)    
    elif suvy == 'WISE':
        dn = 20 * np.exp( - (z - 0.1)**2 )
    elif suvy == "DESI":
        x = np.linspace(0.65, 1.85, 13)
        y1= np.array([309,2269,1923,2094,1441,1353,1337,523,466,329,126,0,0])
        y2 = np.array([832,986,662,272,51,17,0,0,0,0,0,0,0])
        y3 = np.array([47,55,61,67,72,76,80,83,85,87,87,87,86])
        y = y1 + y2 + y3
        print len(x)
        print len(y)
        tck = interpolate.splrep(x, y)
        y_bspline = interpolate.splev(z, tck)
        if z[0] >=.65 or z[0] <=1.85:
            dn = y_bspline[0]
        else:
            dn = 0
    return dn

    
def b(z, suvy):
    if suvy == 'LSST':        
        b = 1.41
    elif suvy == 'WISE':
        b = 1.41
    elif suvy == 'DESI':
        b = 1.41
    else:
        b = 1.41
    return b
    
def didz(z, suvy):
    if suvy == 'CIB':
        j = chis**2 / (1+zs) * np.exp( - (zs - zc)**2 / 2 / sigma_z**2 ) * f_mu(545e9,zs)
        didz = j / H(zs) / (1+zs)
        return didz

    
def Wi(z, suvy):
    if suvy == 'WISE':
        wg = b(z,suvy) * dn(z, suvy)/sum( dn(z, suvy)*dz )
        return wg
    if suvy == 'DESI':
        wg = b(z,suvy) * dn(z, suvy)/sum( dn(z, suvy)*dz )
        return wg
    elif suvy == 'CIB':
        j = chis**2 / (1+zs) * np.exp( - (zs - zc)**2 / 2 / sigma_z**2 ) * f_mu(150e9,zs)
        didz = j / H(zs) / (1+zs)
        #Imu = 0.16e6
        scal = 3720 / max(didz) / 2
        #bc = 3.9*scal
        bc = scal*(0.83 + 0.742*z + 0.318*z*z)
        wcib = bc * chis**2 / H(z) /(1+z)**2 * np.exp( - (z - zc)**2 / 2 / sigma_z**2 ) * f_mu(545e9,z)**1
        return wcib
    elif suvy =='CIB_2':
        #A = 0.289e6; z0 = 1.34; alpha = 1.50; 
        #A = 0.129e6; z0 = 1.25; alpha = 1.24; 
        A = 1.19e6; z0 = 1.08; alpha = 1.52; 
        wcib2 = A * z**alpha * np.exp( - (z/z0)**alpha)
        return wcib2

def Wk(z):
    return 3 * omega_m / 2 / c * H(0)**2 / H(z) *(1 + z)**1 * chis * (chistar - chis) / chistar

def f_mu(mu,z):
    muz = mu * (1+z)
    f = np.zeros(z.shape) 
    for i, zi in enumerate(z):
        if muz[i] < mup:
            f[i] = ( np.exp(h_p * muz[i] / k_b / T) -1 )**(-1) * muz[i]**(beta_cib +3)
        else:
            f[i] = np.exp(h_p * mup / k_b / T -1)**(-1) * mup**(beta_cib +3) * (muz[i]/mup)**(-2)
    return f

def P(zs, k):
    P = PK.P(zs, k ,grid = False)


j = chis**2 / (1+zs) * np.exp( - (zs - zc)**2 / 2 / sigma_z**2 ) * f_mu(150e9,zs)
didz = j / H(zs) / (1+zs)
scal = 3720 / max(didz) / 2
print 'scal', scal


ls = np.arange(2, 1500+1, 10, dtype=np.float64)
rou1 = np.zeros(ls.shape) 
rou2 = np.zeros(ls.shape) 
rou3 = np.zeros(ls.shape) 
ckk = np.zeros(ls.shape) 
cki = np.zeros(ls.shape) 
cii = np.zeros(ls.shape) 
cii2 = np.zeros(ls.shape) 

for i, li in enumerate(ls):
    ckk[i] = C_kk(li,'CIB') 
    cki[i] = C_ki(li,'CIB') 
    cii[i] = C_ii(li,'CIB_pln') 
    cii2[i] = C_ii(li,'CIB') 
llckk = ckk * (ls * (ls+1))**2 /2/np.pi
llcki = cki * (ls * (ls+1))**2 /2/np.pi
llcii = cii * (ls * (ls+1))**2 /2/np.pi


rou2 = cki / (cii2 * ckk)**0.5
rou1 = cki / (cii * ckk)**0.5

llcii = cii * ls * (ls+1) /2/np.pi/1e12
l3cki = cki*ls

wcib = Wi(zs,'CIB')
wcib2 = Wi(zs,'CIB_2')


plt.figure(figsize = (8,5))
plt.loglog(ls, ckk)
plt.loglog(ls, cii)
plt.loglog(ls, cki)
plt.loglog(ls, cii2)
plt.xlim([1,ls[-1]])
plt.xlabel('l')
plt.ylabel('C')
plt.legend(['C_{kk}','C_{ii} pol','C_{ki}','C_{ii} wi'])
plt.savefig('cibpower.png')

plt.figure(figsize = (8,5))
plt.loglog(ls, cii)
plt.loglog(ls, cii2)
plt.xlim([1,ls[-1]])
plt.xlabel('l')
plt.ylabel('C')
plt.legend(['C_{ii} pol','C_{ii} wi'])
plt.savefig('2ciipower.png')

plt.figure(figsize = (8,5))
plt.loglog(ls, cii)
plt.loglog(ls, cii2)
plt.xlabel('l')
plt.ylabel('C')
plt.legend(['C^{ii}_{pln}','C^ii_w'])

plt.figure(figsize = (8,5))
plt.loglog(ls, llcii)
plt.xlim([1,6000])
plt.ylim([5e-5,0.1])
plt.xlabel('l')
plt.ylabel('C')
plt.legend(['l(l+1)C^{ii}_{pln}'])
plt.savefig('pln_cii.png')

plt.figure(figsize = (8,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(zs, wcib)
plt.plot(zs, wcib2)
plt.xlim([0,4.5])
plt.xlabel('z')
plt.ylabel('Wcib')
plt.legend(['wcib = 0912','wcib2 = 1407'])
#plt.savefig('w_cib_didz_2_545.png')
         
plt.figure(figsize = (8,5))        
plt.plot(ls,rou1)
plt.plot(ls,rou2)
plt.xlim([1,ls[-1]])
plt.xlabel('l')
plt.ylabel('rho')
plt.legend(['CIB pln','CIB wi'])
print 'rho max', max(rou1)
print 'rho max l', np.argmax(rou1)
plt.savefig('cib_rho_didz.png')

plt.figure(figsize = (8,5))        
plt.plot(ls,l3cki)
plt.xlim([1,ls[-1]])
plt.xlabel('l')
plt.ylabel('cki')
plt.legend(['l3cki at 545GHz'])
#plt.savefig('l3cki.png')

plt.figure(figsize = (8,5))        
plt.loglog(ls,cii2)
plt.xlim([1e2,1e3])
plt.xlabel('l')
plt.ylabel('cii')
plt.legend(['cii at 545GHz'])
plt.savefig('cii_wi.png')
