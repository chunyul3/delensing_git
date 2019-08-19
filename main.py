#############################################################
#this coude is about a figure. Which involve DES, WIDE, CIB, SPT, LSSS.
#For DES, WISE the W is Wg. which coresonging to different i.
#For C_kk the W is Wk
##############################################################
import numpy as np
import pandas as pd
import math
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy import special
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
from scipy.interpolate import interp1d
from scipy.integrate import dblquad
import datetime
from decimal import Decimal
from itertools import combinations, permutations

h = 0.675
alpha = 1.27; beta = 1.02;
nz = 500 #number of steps to use for the radial/redshift integration
kmax=10  #kmax to use
omega_m = 0.27
c = 1
beta_cib = 2
bc = 1
sigma_z = 2
zc = 2
T = 34 # k
mu = 4955
h_p = 6.63 * 10**(-34) #j*s
k_b = 1.38 * 10**(-23) #j/k
mup = 4.955 * 10**12 #s**-1
mu = 0.545*10**12


wise_scos_z = np.array([0, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15, 0.2, 0.22, 0.25, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.40, 0.45, 0.5])
wise_scos_dn = np.array([0, 0.001, 0.003, 0.008,0.0125, 0.017, 0.022, 0.026, 0.031, 0.034, 0.036, 0.037, 0.040, 0.042, 0.043, 0.043, 0.041, 0.038, 0.032, 0.026, 0.021, 0.017, 0.013, 0.011, 0.009, 0.007, 0.005, 0.004, 0.002, 0.001, 0])
wise_scos = interp1d( wise_scos_z, wise_scos_dn)


data = np.loadtxt('cmb_tot.dat')
ct_l = data[:, 0]
ct_tot = data[:, 1]
ce_tot = data[:, 2]
cb_tot = data[:, 3]
cte_tot = data[:, 4]

data2 = np.loadtxt('cmb.dat')
ct_l = data2[:, 0]
ct = data2[:, 1]
ce = data2[:, 2]
cb = data2[:, 3]
cte = data2[:, 4]

data3 = np.loadtxt('spt_eb.dat')
spteb = data3[:,1]

zeros = np.zeros(10000)
ct = np.append(ct, zeros)
ct_tot = np.append(ct_tot, zeros)
ce = np.append(ce, zeros)
ce_tot = np.append(ce_tot, zeros)
cb = np.append(cb, zeros)
cb_tot = np.append(cb_tot, zeros)
cte = np.append(cte, zeros)
cte_tot = np.append(cte_tot, zeros)


pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_for_lmax(12000, lens_potential_accuracy=0)
results= camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars,  lmax = 6100,CMB_unit='muK')
totCL=powers['total']
unlensedCL=powers['unlensed_scalar']

#chistar = results.conformal_time(0)- results.tau_maxvis
chistar = results.conformal_time(0)- results.conformal_time(1090)
chis = np.linspace(0,chistar,nz)
zs=results.redshift_at_comoving_radial_distance(chis)

dchis = (chis[2:]-chis[:-2])/2
chis = chis[1:-1]
dz = (zs[2:]-zs[:-2])/2
zs = zs[1:-1]
w = np.ones(chis.shape)

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
    a = sum(a)
    return a

def C_ki(l,suvy):
    if suvy == 'SPT':
        sa = C_kk(l, suvy)
    else:
        k = (l + 0.5)/chis
        a = H(zs)**1 / (chis)**2 * Wi(zs, suvy) * Wk(zs) * PK.P(zs, k,grid = False)* (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz / c
        if suvy == 'LSST tomographic' or suvy == 'DES tomographic':
            sa = np.sum(a, axis = 1)
        else: sa = sum(a)
    return sa

def matrix_cii(l, c11, c12, c21, c22):
    # c11 = ckk + ns for spt
    # c12 = cki
    a0 = [[c11,c12],[c12,c22]]
    a = np.matrix(a0)
    return a 

def cross_rho2(l, c11, c12, c21, c22, ck1, ck2):
    a = matrix_cii(l, c11, c12, c21, c22)
#    print a
    b = a.I
    kk = C_kk(l, 'SPT')
    r11 = ck1**2 * b[0,0]
    r12 = ck1 * ck2 * b[0,1]
    r22 = ck2**2 * b[1,1]
    rho = (r11 + r12 * 2 + r22) / kk
    rho = rho**0.5
    return rho

def cross_rho3(l, ckk, c11, c12, c13, c22, c23, c33, ck1, ck2, ck3):
    c21 = c12; c31 = c13; c32 = c23
    a = [[c11,c12,c13],[c21,c22,c23],[c31,c32,c33]]
    a = np.matrix(a)
    b = a.I
    v1 = np.array([[ck1, ck2, ck3]])
    v2 = v1.T
    rho = v1 * b * v2
    if len(rho) == 1:
        rho = float(rho)
    rho = (rho / ckk)**0.5
    return rho

def tomog_rho(l, suvy):
    ckk = C_kk(l, suvy)
    cki = C_ki(l, suvy)
    dim = len(cki)
    cij = np.zeros( (dim, dim) )
    wg = Wi(zs, suvy)
    for i in range(0, dim):        
        for j in range(0, dim):
            cij[i][j] = C_ij_tomog(l, suvy, wg[i, :], wg[j, :])
    cij = np.matrix( cij )
    inver_cij = cij.I
    cki = np.array( [cki] )
    ckj = cki.T
#    print 'cki', cki
 #   print 'ckj', ckj
    rho = cki * inver_cij * ckj
  #  print rho
    rho = ( float(rho) / ckk )**0.5
    if suvy != 'LSST tomographic' and suvy != 'DES tomographic':
        print 'unknow survey: ', suvy                                
    return rho
                    
def cross_cii(l, suvy1, suvy2):
    c11 = 0
    c12 = 0
    c21 = 0
    c22 = 0
    if suvy1 == 'spt':
        ckk = C_kk(l, 'spt')
        al = l * np.cos(theta)
        bl = l * np.sin(theta)        
        ns = noise_ab(al,bl) * l**2 * (l+1)**2
        if l < 6100:
            c11 = ckk + ns
            c12 = C_ki(l, suvy2)
            c21 = c12
            c22 = C_ii(1, suvy2)
    elif suvy2 == 'spt':
        if l < 6100:
            c11 = C_ii(l, suvy1)
            c12 = C_ki(l, suvy1)
            c21 = c12
            c22 = ckk + ns
    else:
        c11 = C_ii(l, suvy1)
        c22 = C_ii(l, suvy2)
        c12 = C_ij(l, suvy1, suvy2)
    a0 = [[c11,c12],[c12,c22]]
#    a = np.matrix('c11,c12;c12,c22')
    a = np.matrix(a0)
    return a 

def cross_rho(l, suvy1, suvy2):
    a = cross_cii(l, suvy1, suvy2)
    b = np.zeros((2,2))
    c1 = 0
    c2 = 0
    if suvy2 == 'spt':
        c1 = C_ki(l, suvy1); c2 = C_kk(l, suvy2)
        if l < 6100:
            b = a.I
            #print b
    elif suvy1 == 'spt':
        cl = C_ki(l, suvy2); c2 = C_kk(l, suvy1)
        if l < 6100:
            b = a.I
#            print a
            
    else:
        cl = C_ki(l, suvy1); c2 = C_ki(l, suvy2)
        b = a.I
    kk = C_kk(l, suvy2)
    r11 = c1**2 * b[0,0]
    r12 = c1 * c2 * b[0,1]
    r22 = c2**2 * b[1,1]
    rho = (r11 + r12 * 2 + r22) / kk
    rho = rho**0.5
    return rho
        
def C_ii(l,suvy):
    k = (l + 0.5)/chis
    if suvy == 'CIB':
        a = 3500 * (l/3000.0) ** (-1.25)
        return a
    else:
        a = H(zs)**1 / (chis)**2 * Wi(zs,suvy) * Wi(zs,suvy) * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
        if suvy == 'LSST tomographic' or suvy == 'DES tomographic':
            sa = np.sum(a, axis = 1)
        else: sa = sum(a)
        return sa

def C_ij(l,suvy1, suvy2):
    if suvy1 == 'SPT':
        a = C_ki(l, suvy2)
    elif suvy2 == 'SPT':
        a = C_ki(l, suvy1)
    else:
        k = (l + 0.5)/chis
        a = H(zs)**1 / (chis)**2 * Wi(zs,suvy1) * Wi(zs,suvy2) * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
        a = sum(a)
    return a

def C_ij_tomog(l, suvy, wni, wnj):
    if suvy == 'LSST tomographic' or suvy == 'DES tomographic':
        k = (l + 0.5)/chis
        a = H(zs)**1 / (chis)**2 * wni * wnj * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
        a = sum(a)        
    else:
        print 'wrong survey'
    return a

def H(z):
    return results.h_of_z(z)    #Mpc-1

def dn_dz(z, suvy):
    if suvy == 'LSST' or suvy == 'LSST tomographic':        
        z0 = 0.5
        dn = z**alpha * np.exp(-(z/z0)**beta)    
    elif suvy == 'WISE':
        dn = 20 * np.exp( - (z - 0.1)**2 )
    elif suvy == 'DES' or suvy == 'DES tomographic':
        dn = 7.7 * np.exp( -(z - 0.657)**2 / 0.043**2 / 2)
    elif suvy == 'WISE x SCOS':
        dn = np.zeros(len(z))
        for i, zi in enumerate(z):            
          if zi > 0 and zi <0.5:
                dn[i] = wise_scos(zi)
    elif suvy == "DESI":
        x = np.linspace(0.65, 1.85, 13)
        y1= np.array([309,2269,1923,2094,1441,1353,1337,523,466,329,126,0,0])
        y2 = np.array([832,986,662,272,51,17,0,0,0,0,0,0,0])
        y3 = np.array([47,55,61,67,72,76,80,83,85,87,87,87,86])
        y = y1 + y2 + y3
        tck = interpolate.splrep(x, y)
        y_bspline = interpolate.splev(z, tck)
        dn = y_bspline
        y00 = y[0]
        for i, zi in enumerate(z):            
            if zi >=.65 and zi <=1.85:
                dn[i] = dn[i]
            elif zi < 0.65:
                dn[i] = y00/0.65*zi*0
            else:
                dn[i] = 0
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

def W_tmg(z, suvy):
    if suvy == 'LSST tomographic':
        wg = np.zeros((10, len(z)))
        zi0 = np.array([0.4, 0.6, 0.9, 1.2, 1.5, 1.9, 2.2, 2.8, 3.4])
        zim = 0.82 * zi0 - 0.127
        zi = np.array([0] + list(zim) + [3.5])
        for i in range(10):            
            dndzi = 0.5 * dn_dz(z, suvy) * ( special.erfc( (zi[i] - z) / np.sqrt(2) / 0.03 / (1 + z) ) - special.erfc( (zi[i + 1] - z) / np.sqrt(2) / 0.03 / (1 + z) ))
            wg[i,:] = dndzi * b(z, suvy)
        wg = wg / sum (dn_dz(z, suvy)*dz)
    elif suvy == 'DES tomographic':
        wg = np.zeros((4, len(z)))
        zi0 = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
        zi = zi0
        for i in range(4):            
            dndzi = 0.5 * dn_dz(z, suvy) * ( special.erfc( (zi[i] - z) / np.sqrt(2) / 0.03 / (1 + z) ) - special.erfc( (zi[i+1] - z) / np.sqrt(2) / 0.03 / (1 + z) ))
            wg[i,:] = dndzi * b(z, suvy)
        wg = wg / sum(dn_dz(z, suvy)*dz)
    return wg

    
def Wi(z, suvy):
    if suvy == 'WISE':
        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
        return wg
    elif suvy == 'LSST':
        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
        return wg
    elif suvy == 'LSST tomographic':
        return W_tmg(z, suvy)
    elif suvy == 'DES':
        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
        return wg
    elif suvy == 'DES tomographic':
        return W_tmg(z, suvy)
    elif suvy == 'DESI':
        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
        return wg
    elif suvy == 'WISE x SCOS':
        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
        return wg    
    elif suvy == 'CIB':
        wcib = bc * chis**2 / H(z) /(1+z)**2 * np.exp( - (z - zc)**2 / 2 / sigma_z**2 ) * f_mu(z)**1
        a = sum(wcib)
        Imu = 1.4/3*10**6
        wcib = wcib/a*Imu
        wcib = wcib * 21.088
        return wcib    
    elif suvy =='CIB_2':
        A = 0.289; z0 = 1.34; alpha = 1.30; 
        wcib2 = A * z**alpha * np.exp( - (z/z0)**alpha)
        a = max(wcib2)
        wcib2 = wcib2/a
        return wcib2
    else:
        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
        return wg


def Wk(z):
    return 3 * omega_m / 2 / c * H(0)**2 / H(z) *(1 + z)**1 * chis * (chistar - chis) / chistar

def f_mu(z):
    muz = mu * (1+z)
    return np.exp(h_p * muz / k_b / T -1)**(-1) * muz**(beta_cib +3)

def spt_ns(lx, ly):
    se = 0; sb = 0; st = 0; ste = 0; stb = 0; seb = 0
    for xi in range(-3000, 3001, 10):
        for yi in range(-3000, 3001, 10):
            d1 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d2 = (xi**2 + yi**2)**0.5
            if d1 >= 2 and d2 >= 2:
                cosphi = (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                cos2phi = 2 * cosphi**2 - 1
                sin2phi = 2 * cosphi * (1 - cosphi**2)**0.5
                d1 = int(d1)
                d2 = int(d2)
                p1 = ( lx - xi ) * lx + ( ly - yi ) * ly
                p2 = lx*xi + yi*ly
                tn_t = ( p1 * ct[d1] + p2 * ct[d2] )**2 / 2 / (ct_tot[d2] + nfwhm[d2] )/ (ct_tot[d1] + nfwhm[d1] ) 
                f1 = p1 * cte[d1] + p2 * cos2phi * cte[d2]
                f2 = p1 * cte[d1] * cos2phi + p2 * cte[d2]
                F = (ce_tot[d2] * ct_tot[d2] * f1 - cte_tot[d2] * cte_tot[d1] * f2) / ( (ct_tot[d2] + nfwhm[d2]) * (ce_tot[d1] + nfwhm[d1]*2) * (ce_tot[d2]+ nfwhm[d2]*2) * (ct_tot[d1] + nfwhm[d1]) - ( (cte_tot[d1] + nfwhm[d1]) * (cte_tot[d2] + nfwhm[d2]) )**2 )
                tn_te = f1 * F 
                tn_tb = ( p2 * cte[d2] * sin2phi )**2 / (cb_tot[d2] + 2 * nfwhm[d2] )/ (ct_tot[d1] + nfwhm[d1] ) 
                tn_e = ( p1 * ce[d1] + p2 * ce[d2] )**2 * cos2phi**2 / 2 / (ce_tot[d2] + 2 * nfwhm[d2] )/ (ce_tot[d1] + 2 * nfwhm[d1] ) 
                tn_b = ( p1 * cb_tot[d1] + p2 * cb_tot[d2] )**2 * cos2phi**2 / 2 / (cb_tot[d2] + 2 * nfwhm[d2] )/ (cb_tot[d1] + 2 * nfwhm[d1] ) 
                tn_eb = ( p1 * cb_tot[d1] + p2 * ce[d2] )**2 * sin2phi**2 / (ce_tot[d2] + 2 * nfwhm[d2] )/ (cb_tot[d1] + 2 * nfwhm[d1] )
                se += tn_e; sb += tn_b; st += tn_t 
                ste += tn_te; stb += tn_tb; seb += tn_eb
                if xi == -2990 and yi == -2990:
                    print 'st', st
    st = st * 100 / (2 * np.pi)**2
    se = se * 100 / (2 * np.pi)**2
    sb = sb * 100 / (2 * np.pi)**2
    ste = ste * 100 / (2 * np.pi)**2
    stb = stb * 100 / (2 * np.pi)**2
    seb = seb * 100 / (2 * np.pi)**2
    return 1/st, 1/se, 1/sb, 1/ste, 1/stb, 1/seb

def n_fwhm(l):
    a = 1/1.2 * 1e-7* np.exp( l**2 * 1**2 * (3e-4)**2/ 2.355**2 )*25
    return a

def noise_t(lx,ly):
    s = 0
    for xi in rlxnge(-3000, 3001, 10):
        for yi in rlxnge(-3000, 3001, 10):
            d1 = int(((xi-lx)**2 + (yi-ly)**2)**0.5)
            d2 = int((xi**2 + yi**2)**0.5)
            if d1 >= 2 and d2 >= 2:
                tn = ( ( (lx-xi)*lx + (ly-yi)*ly ) * ct[d1] + ( lx*xi + yi*ly )* ct[d2] )**2 / 2 / (ct_tot[d2] + nfwhm[d2] )/ (ct_tot[d1] + nfwhm[d1] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1/s/100.0

def noise_te(lx ,ly):
    s = 0
    for xi in range(-3000, 3001, 10):
        for yi in range(-3000, 3001, 10):
            d1 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d2 = (xi**2 + yi**2)**0.5
            if d1 >= 2 and d2 >= 2:
                cosphi = (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                cos2phi = 2 * cosphi**2 - 1
                d1 = int(d1)
                d2 = int(d2)
                f1 = ( (lx - xi)*lx + (ly - yi)*ly ) * cte[d1] + ( lx*xi + yi*ly )* cos2phi * cte[d2]
                f2 = ( (lx - xi)*lx + (ly - yi)*ly ) * cte[d1] * cos2phi + ( lx*xi + yi*ly )* cte[d2]
                F = (ce_tot[d2] * ct_tot[d2] * f1 - cte_tot[d2] * cte_tot[d1] * f2) / ( (ct_tot[d2] + nfwhm[d2]) * (ce_tot[d1] + nfwhm[d1]*2) * (ce_tot[d2]+ nfwhm[d2]*2) * (ct_tot[d1] + nfwhm[d1]) - ( (cte_tot[d1] + nfwhm[d1]) * (cte_tot[d2] + nfwhm[d2]) )**2 )
                tn = f1 * F / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1/s/100.0

def noise_tb(lx ,ly):
    s = 0
    for xi in range(-3000, 3001, 10):
        for yi in range(-3000, 3001, 10):
            d1 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d2 = (xi**2 + yi**2)**0.5
            if d1 >= 2 and d2 >= 2:
                cosphi = (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                sin2phi = 2 * cosphi * (1 - cosphi**2)**0.5
                d1 = int(d1)
                d2 = int(d2)
                tn = ( ( lx*xi + yi*ly )* cte[d2] * sin2phi )**2 / (cb_tot[d2] + 2 * nfwhm[d2] )/ (ct_tot[d1] + nfwhm[d1] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1/s/100.0


def noise_e(lx ,ly):
    s = 0
    for xi in range(-3000, 3001, 10):
        for yi in range(-3000, 3001, 10):
            d1 = int(((xi-lx)**2 + (yi-ly)**2)**0.5)
            d2 = int((xi**2 + yi**2)**0.5)            
            if d1 >= 2 and d2 >= 2:
                cosphi = 1.0 * (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                cos2phi = 2 * cosphi**2 - 1
                tn = ( ( (lx - xi)*lx + (ly - yi)*ly ) * ce[d1] + ( lx*xi + yi*ly )* ce[d2] )**2 * cos2phi**2 / 2 / (ce_tot[d2] + 2 * nfwhm[d2] )/ (ce_tot[d1] + 2 * nfwhm[d1] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1/s/100.0

def noise_b(lx ,ly):
    s = 0
    for xi in range(-3000, 3001, 10):
        for yi in range(-3000, 3001, 10):
            d1 = int(((xi-lx)**2 + (yi-ly)**2)**0.5)
            d2 = int((xi**2 + yi**2)**0.5)            
            if d1 >= 2 and d2 >= 2:
                cosphi = 1.0 * (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                cos2phi = 2 * cosphi**2 - 1
                tn = ( ( (lx - xi)*lx + (ly - yi)*ly ) * cb_tot[d1] + ( lx*xi + yi*ly )* cb_tot[d2] )**2 * cos2phi**2 / 2 / (cb_tot[d2] + 2 * nfwhm[d2] )/ (cb_tot[d1] + 2 * nfwhm[d1] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1/s/100.0


def noise_eb(lx ,ly):
    s = 0
    for xi in range(-3000, 3001, 10):
        for yi in range(-3000, 3001, 10):
            d1 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d2 = (xi**2 + yi**2)**0.5
            if d1 >= 2 and d2 >= 2:
                cosphi = (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                sin2phi = 2 * cosphi * (1 - cosphi**2)**0.5
                d1 = int(d1)
                d2 = int(d2)
                tn = ( ( (lx - xi)*lx + (ly - yi)*ly ) * cb_tot[d1] + ( lx*xi + yi*ly )* ce[d2] )**2 * sin2phi**2 / (ce_tot[d2] + 2 * nfwhm[d2] )/ (cb_tot[d1] + 2 * nfwhm[d1] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1/s/100.0


def Ct(l):
    if l < 6100:
        a = ct_itp(l)
    else:
        a = 0
    return a

def Ct_tot(l):
    if l < 6100:
        a = ct_tot_itp(l)
    else:
        a = 0
    return a

def P(zs, k):
    P = PK.P(zs, k ,grid = False)


survey_list = ['DESI', 'WISE', 'LSST', 'CIB', 'SPT', 'DES', 'WISE x SCOS']
comb2 = list(combinations(['DESI', 'WISE', 'LSST', 'CIB', 'SPT', 'DES', 'WISE x SCOS'], 2))
comb3 = list(combinations(['DESI', 'WISE', 'LSST', 'CIB', 'SPT', 'DES', 'WISE x SCOS'], 3))
survey_tomog_list = ['LSST tomographic', 'DES tomographic']
#legd1 = ['DESI','WISE','LSST','SPT','CIB pln','DES','WISE x SCOS']
legd1 = []
legd2 = []
legd3 = []
legd4 = []

for sy in survey_list:
    legd1.append(sy)

for sy in comb2:
    legd2.append( sy[0] + '+' + sy[1])

for sy in comb3:
    legd3.append( sy[0] + '+' + sy[1] + '+' + sy[2])

for sy in survey_tomog_list:
    legd4.append(sy)

ls = np.arange(2,1500+1,10)
#ls = [2, 22, 62, 102, 182, 282, 382, 582, 882, 1082, 1482]
ls = np.asarray(ls)

rho = np.zeros(( len(ls), len(survey_list) ))
cki = np.zeros(( len(ls), len(survey_list) ))
cii = np.zeros(( len(ls), len(survey_list) ))
cij = np.zeros(( len(ls), len(comb2) ))
cijk = np.zeros(( len(ls), len(comb3) ))
rho_double = np.zeros(( len(ls), len(comb2) ))
rho_triple = np.zeros(( len(ls), len(comb3) ))
rho_tomog = np.zeros(( len(ls), len(survey_tomog_list) ))

spt_nseb = np.zeros(ls.shape)
ckk = np.zeros(ls.shape)

l_long = np.arange(0,15000,1)
nfwhm = n_fwhm(l_long)

theta = np.pi/2


for i, li in enumerate(ls):
    print li
    start = datetime.datetime.now()
    al = li * np.cos(theta)
    bl = li * np.sin(theta)
    spt_nseb[i] = spteb[i] # data already have * li**2 * (li+1)**2/4
    print  'eb', '%.2E' % Decimal(spt_nseb[i])
    ckk[i] = C_kk(li,'CIB') 

    for j, survey in enumerate(survey_list):
        if j == 4:
            cki[i][j] = ckk[i]
            cii[i][j] = ckk[i] + spt_nseb[i]
        else:
            cki[i][j] = C_ki(li, survey)
            cii[i][j] = C_ii(li, survey)

    for ki, two_suvy in enumerate(comb2):
        sy1 = two_suvy[0]
        sy2 = two_suvy[1]
        cij [i][ki] = C_ij(li, sy1, sy2)
        sy1id = survey_list.index( sy1 )
        sy2id = survey_list.index( sy2 )
        rho_double [i][ki] = cross_rho2(li, cii[i][sy1id], cij[i][ki], cij[i][ki], cii[i][sy2id], cki[i][sy1id], cki[i][sy2id] )

    for zi , surveys in enumerate(comb3):
        sy1 = surveys[0]
        sy2 = surveys[1]
        sy3 = surveys[2]
        sy1id = survey_list.index( sy1 )
        sy2id = survey_list.index( sy2 )
        sy3id = survey_list.index( sy3 )
        sy12id = comb2.index((sy1, sy2))
        sy13id = comb2.index((sy1, sy3))
        sy23id = comb2.index((sy2, sy3))
        rho_triple [i][zi] = cross_rho3(li, ckk[i], cii[i][sy1id], cij[i][sy12id], cij[i][sy13id], cii[i][sy2id], cij[i][sy23id], cii[i][sy3id], cki[i][sy1id], cki[i][sy2id], cki[i][sy3id])
        
    for j , surveys in enumerate(survey_tomog_list):
        rho_tomog [i][j] = tomog_rho(li, surveys)
            
    end = datetime.datetime.now()
    print (end-start)

ckk = ckk.reshape((len(ls), 1))
print ckk.shape
rho = cki / (cii * ckk)**0.5
rho0 = rho[:, 0]
rho1 = rho[:, 1]
rho2 = rho[:, 2]
rho3 = rho[:, 3]
rho4 = rho[:, 4]
rho5 = rho[:, 5]
rho6 = rho[:, 6]

data = np.column_stack((ls, rho0, rho1, rho2, rho3, rho4, rho5, rho6))
header = "ls, DESI, WISE, LSST, CIB, rho_spt, DES, WISE x SCOS"
np.savetxt('rho_single_gener.dat', data, header=header)

spteb = np.column_stack((ls, spt_nseb))
header = "ls, spt_nseb"
np.savetxt('spt_eb_gener.dat', spteb, header=header)

wi_z = zs
w0 = Wi(wi_z, 'DESI')
w1 = Wi(wi_z, 'WISE')
w2 = Wi(wi_z, 'LSST')
w2t = Wi(wi_z, 'LSST tomographic')
w3 = Wi(wi_z, 'CIB')
w4 = Wi(wi_z, 'DES')
w4t = Wi(wi_z, 'DES tomographic')
w5 = Wi(wi_z, 'WISE x SCOS')

plt.figure(figsize = (8,5))        
plt.plot(wi_z, w0)
plt.plot(wi_z, w1)
plt.plot(wi_z, w2)
plt.plot(wi_z, w3)
plt.plot(wi_z, w4)
plt.plot(wi_z, w5)
plt.xlim([0,4.5])
plt.legend(['DESI','WISE','LSST','CIB','DES','WISE x SCOS'])
plt.savefig('weight.png')

plt.figure(figsize = (8,5))       
plt.plot(wi_z, w2, linewidth = 2, color = 'r')
#plt.plot(wi_z, w4, linewidth = 2, color = 'b')
for weight in w2t:
    plt.plot(wi_z, weight, color = 'r')
#for weight in w4t:
 #   plt.plot(wi_z, weight, color = 'b')
plt.xlim([0,4.5])
plt.legend(['LSST'])
plt.savefig('weight_lsst.png')

plt.figure(figsize = (8,5))       
#plt.plot(wi_z, w2, linewidth = 2, color = 'r')
plt.plot(wi_z, w4, linewidth = 2, color = 'b')
#for weight in w2t:
 #   plt.plot(wi_z, weight, color = 'r')
for weight in w4t:
   plt.plot(wi_z, weight, color = 'b')
plt.xlim([0,1.5])
plt.legend(['DES'])
plt.savefig('weight_des.png')

plt.figure(figsize = (8,5))
for i, surveys in enumerate(survey_tomog_list):
    plt.plot(ls, rho_tomog[:,i], linestyle = '--')
plt.plot(ls, rho2)
plt.plot(ls, rho5)
#print rho_tomog.T
plt.xlim([1,ls[-1]])
plt.ylim([0,1])
plt.xlabel('l')
plt.ylabel('rho tomographic')
plt.legend(legd4 + ['LSST','DES'])
plt.savefig('rho_tomog.png')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
         
plt.figure(figsize = (8,5))        
for i, survey in enumerate(survey_list):
    plt.plot(ls, rho[:,i], linewidth = 1)
plt.xlim([1,ls[-1]])
plt.ylim([0,1])
plt.xlabel('l')
plt.ylabel('rho')
plt.legend(legd1)
plt.savefig('rho_5muk_beam1_lm3k_eb_single0.png')

plt.figure(figsize = (16,10))        
for i, survey in enumerate(survey_list):
    plt.plot(ls, rho[:,i], linestyle = '-', linewidth = 3)
for i, survey in enumerate(comb2):
    if survey[0] == 'WISE':
        plt.plot(ls, rho_double[:,i], linestyle = '--', linewidth = 1)
    elif survey[0] == 'DESI':
        plt.plot(ls, rho_double[:,i], linestyle = '-', linewidth = 1)
    elif survey[0] == 'LSST' or 'CIB':
        plt.plot(ls, rho_double[:,i], linestyle = '-.', linewidth = 1)
    else:
        plt.plot(ls, rho_double[:,i], linestyle = ':', linewidth = 1)
plt.xlim([1,ls[-1]])
plt.ylim([0,1])
plt.xlabel('l')
plt.ylabel('rho double')
plt.legend(legd1 + legd2)
#plt.legend(['WISE + LSST','WISE + CIB','WISE + SPT', 'WISE + DES', 'WISE+WISExSCOS','SPT + DES','SPT + WISExSCOS','DES+WISExSCOS','LSST + CIB','LSST + SPT','LSST + DES', 'LSST+WISExSCOS','CIB + SPT', 'CIB + DES', 'CIB+WISExSCOS'])
plt.savefig('rho_comb2.png')

plt.figure(figsize = (24,15))        
for i, survey in enumerate(survey_list):
    plt.plot(ls, rho[:,i], linestyle = '-', linewidth = 3)
for i, survey in enumerate(comb3):
    if survey[0] == 'DESI':
        plt.plot(ls, rho_triple[:, i], linestyle = ':')
    elif survey[0] == 'WISE':
        plt.plot(ls, rho_triple[:, i], linestyle = '--')
    elif survey[0] == 'LSST':
        plt.plot(ls, rho_triple[:, i], linestyle = '-.')
    else:
        plt.plot(ls, rho_triple[:, i], linestyle = '-')
plt.xlim([1,ls[-1]])
plt.ylim([0,1])
plt.xlabel('l')
plt.ylabel('rho triple')
plt.title('rho 3 tracers')
plt.legend( legd1 + legd3 ) 
plt.savefig('rho_comb3.png')

