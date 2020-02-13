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
from tempfile import TemporaryFile
outfile = TemporaryFile()

h = 0.675
nz = 100 #number of steps to use for the radial/redshift integration
kmax=10  #kmax to use
omega_m = 0.27
c = 1
beta_cib = 2
bc = 1
sigma_z = 2
zc = 2
alpha_lsst, beta_lsst = 1.27, 1.02
T = 34 # k
mu = 4955
h_p = 6.63 * 10**(-34) #j*s
k_b = 1.38 * 10**(-23) #j/k
mup = 4.955 * 10**12 #s**-1
mu = 0.545*10**12
dl_intg = 100

wise_scos_z = np.array([0, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15, 0.2, 0.22, 0.25, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.40, 0.45, 0.5])
wise_scos_dn = np.array([0, 0.001, 0.003, 0.008,0.0125, 0.017, 0.022, 0.026, 0.031, 0.034, 0.036, 0.037, 0.040, 0.042, 0.043, 0.043, 0.041, 0.038, 0.032, 0.026, 0.021, 0.017, 0.013, 0.011, 0.009, 0.007, 0.005, 0.004, 0.002, 0.001, 0])
wise_scos = interp1d( wise_scos_z, wise_scos_dn)


data = np.loadtxt('data_input/cmb_tot.dat')
ct_l = data[:, 0]
ct_tot = data[:, 1]
ce_tot = data[:, 2]
cb_tot = data[:, 3]
cte_tot = data[:, 4]

data2 = np.loadtxt('data_input/cmb.dat')
ct_l = data2[:, 0]
ct = data2[:, 1]
ce = data2[:, 2]
cb = data2[:, 3]
cte = data2[:, 4]

#data3 = np.loadtxt('data_input/spt_eb_lm3k.dat')
#spteb = data3[:,1]

#data4 = np.loadtxt('data_input/rho_single_lm3k_d10.dat')
#rho_num_s = data4.shape[1]-1

#data5 = np.loadtxt('data_input/rho_double_lm3k_d10.dat')
#rho_ls = data5[:,0]
#rho_num_d = data5.shape[1]-1

#data6 = np.loadtxt('data_input/rho_triple_lm3k_d10.dat')
#rho_num_t = data6.shape[1]-1

#data7 = np.loadtxt('data_input/cmb_survey_noise_sum2.dat')
#data7 = np.loadtxt('data_input/cmb_survey_noise3.dat')
data7 = np.loadtxt('data_input/cmb_survey_noise.dat')
#nfwhm_SPT4g_95 = data7[:,2]
#nfwhm_SPT4g_150 = data7[:,3]
#nfwhm_SO_lat_93 = data7[:,6]
#nfwhm_SO_lat_145 = data7[:,7]
rho_ls = data7[:,0]
spteb = data7[:,1]
nfwhm_CMBs4 = data7[:,-2]
nfwhm_CMBs4_cob = data7[:,-1]
nfwhm_SPT3g = data7[:,2]
nfwhm_SO = data7[:,3]

rho_ls_long = np.arange(2, rho_ls[-1] + 1)
#rho_sig = np.zeros((rho_num_s, len(rho_ls_long)))
#rho_dub = np.zeros((rho_num_d, len(rho_ls_long)))
#rho_tip = np.zeros((rho_num_t, len(rho_ls_long)))

#for i in range(0,rho_num_s):    
 #   rhoi_itp = interp1d(rho_ls, data4[:,i+1])
 #   rho_sig[i, :] = rhoi_itp(rho_ls_long)

#for i in range(0,rho_num_d):    
 #   rhoi_itp = interp1d(rho_ls, data5[:,i+1])
  #  rho_dub[i, :] = rhoi_itp(rho_ls_long)

#for i in range(0,rho_num_t):    
 #   rhoi_itp = interp1d(rho_ls, data6[:,i+1])
  #  rho_tip[i, :] = rhoi_itp(rho_ls_long)

spteb_itp = interp1d(data7[:,0], data7[:,1])
spt3g_itp = interp1d(data7[:,0], data7[:,2])
cmbs4_itp = interp1d(data7[:,0], data7[:,-2])
cmbs4cob_itp = interp1d(data7[:,0], data7[:,-1])
so_itp = interp1d(data7[:,0], data7[:,3])
sptl = np.arange(2,3003)
sptebitp = spteb_itp(sptl)
spt3gitp = spt3g_itp(sptl)
cmbs4itp = cmbs4_itp(sptl)
cmbs4cobitp = cmbs4cob_itp(sptl)
soitp = so_itp(sptl)
sptebitp = np.append([0,0], sptebitp)
spt3gitp = np.append([0,0], spt3gitp)
cmbs4itp = np.append([0,0], cmbs4itp)
cmbs4cobitp = np.append([0,0], cmbs4cobitp)
soitp = np.append([0,0], soitp)
print sptebitp

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
print chis
dz = (zs[2:]-zs[:-2])/2
zs = zs[1:-1]
w = np.ones(chis.shape)

PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
    hubble_units=False, k_hunit=False, kmax=kmax,
    var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zs[-1])
cl_camb = results.get_lens_potential_cls(12000)
cl_camb_b = cl_camb[:, 0]
print 'len cl_camb_b', len(cl_camb_b)
def Rou(l,suvy):
    rou_sq = C_ki(l,suvy) ** 2 / C_kk(l,suvy) / C_ii(l,suvy)
    rou = rou_sq ** 0.5
    return rou

def cki_uncty(l, suvy, paratype, delta_val):
    # This function is written for CIB
    if suvy == 'CIB':
        if paratype == 'zc':
            k = (l + 0.5)/chis
            a = H(zs)**1 / (chis)**2 * dWi(zs, suvy, paratype, delta_val) * Wk(zs) * PK.P(zs, k,grid = False)* (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz / c
            sa = sum(a)
            return sa
#        drho = 

def dWi(z, suvy, paratype, delta_val):
    if suvy == 'CIB':
        dwcibdz1 = bc * chis**2 / H(z) /(1+z)**2 * np.exp( - (z - zc)**2 / 2 / sigma_z**2 ) * f_mu(z)**1 * 2 * (z - zc) / 2 / sigma_z**2
        dwcibdz2 = bc * chis**2 / H(z) /(1+z)**2 * np.exp( - (z - zc)**2 / 2 / sigma_z**2 ) * f_mu(z)**1 * (z - zc)**2 / sigma_z**3
        dwcib1 = dwcibdz1 * delta_val
        dwcib2 = dwcibdz2 * delta_val
        dwcib = (dwcib1**2 + dwcib2**2)**0.5
        a = 3.984850275674492e+71
#        a = sum(dwcib)
        Imu = 1.4/3*10**6
        dwcib = dwcib/a*Imu
        dwcib = dwcib * 21.088
        return dwcib        
    else:
#        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
 #       return wg
        return 0


def C_kk(l,suvy):
    k = (l + 0.5)/chis
    a = H(zs)**1 / (chis)**2 * Wk(zs) * Wk(zs) * PK.P(zs, k,grid = False)* (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
    a = sum(a)
    return a

def C_ki(l,suvy):
    if suvy == 'SPT':
        sa = C_kk(l, suvy)
    elif suvy == 'CMBs4' or suvy == 'CMBs4_cob':
        sa = C_kk(l, suvy)
    elif suvy == 'SPT3g':
        sa = C_kk(l, suvy)
    elif suvy == 'SO':
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

def cross_rho(l, suvy):
    ckk = C_kk(l, suvy[0])
    dim = len(suvy)
    cki = np.zeros(dim)
    for i, sy in enumerate(suvy):        
        cki[i] = C_ki(l, sy)
    cij = np.zeros( (dim, dim) )
    for i in range(0, dim):        
        for j in range(i, dim):                
            if i == j:
                cij[i][j] = C_ii(l, suvy[i])
            else:
                cij[i][j] = C_ij(l, suvy[i], suvy[j])
                cij[j][i] = cij[i][j]
    cij = np.matrix( cij )
    inver_cij = cij.I
    cki = np.array( [cki] )
    ckj = cki.T
    rho = cki * inver_cij * ckj
    rho = ( float(rho) / ckk )**0.5
    return rho

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
    ng = dn_dz(zs, suvy)
    if suvy == 'LSST tomographic':
        for i in range(0, dim):        
            for j in range(0, dim):                
                cij[i][j] = C_ij_tomog(l, suvy, wg[i, :], wg[j, :])
                if i ==j :
                    cij[i][j] += ns_suvy(suvy, ng, ng[i,:], ng[j,:])**1

    elif suvy == 'DES tomographic':
        for i in range(0, dim):        
            for j in range(0, dim):
                cij[i][j] = C_ij_tomog(l, suvy, wg[i, :], wg[j, :])
    cij = np.matrix( cij )
    inver_cij = cij.I
    cki = np.array( [cki] )
    ckj = cki.T
    rho = cki * inver_cij * ckj
    rho = ( float(rho) / ckk )**0.5
    if suvy != 'LSST tomographic' and suvy != 'DES tomographic':
        print 'unknow survey: ', suvy                                
    return rho
                    

def cross_tomog_rho(l, suvy_tomog, suvy2):
    ckk = C_kk(l, suvy_tomog)
    cki1 = C_ki(l, suvy_tomog)
    dim1 = len(cki1)
    dim2 = len(suvy2)
    cki2 = np.zeros(dim2)
    for i, sy in enumerate(suvy2):        
        cki2[i] = C_ki(l, sy)
    dim = dim1 + dim2
    cij = np.zeros( (dim, dim) )
    wg1 = Wi(zs, suvy_tomog)
    wg2 = np.zeros((dim2, len(zs)))
    for i, sy in enumerate(suvy2):
        wg2[i, :] = Wi(zs, sy)
    ng = dn_dz(zs, suvy_tomog)

    for i in range(0, dim1):        
        for j in range(i, dim1):                
            cij[i][j] = C_ij_tomog(l, suvy_tomog, wg1[i, :], wg1[j, :])
            cij[j][i] = cij[i][j]
            if i ==j :
                cij[i][j] += ns_suvy(suvy_tomog, ng, ng[i,:], ng[j,:])
    for i in range(dim1, dim):        
        for j in range(i, dim):                
            if i == j:
                cij[i][j] = C_ii(l, suvy2[i-dim1])
            else:
                cij[i][j] = C_ij(l, suvy2[i-dim1], suvy2[j-dim1])
                cij[j][i] = cij[i][j]

    for i in range(0, dim1):        
        for j in range(dim1, dim):                
            cij[i][j] = C_ij_tomog(l, suvy_tomog, wg1[i, :], wg2[j-dim1, :])
            cij[j][i] = cij[i][j]

    cij = np.matrix( cij )
    inver_cij = cij.I
    cki = np.array( [np.append(cki1,cki2)] )
    ckj = cki.T
    rho = cki * inver_cij * ckj
    rho = ( float(rho) / ckk )**0.5
    if suvy_tomog != 'LSST tomographic' and suvy_tomog != 'DES tomographic':
        print 'unknow survey: ', suvy                                
    return rho
        
def C_ii(l,suvy):
    k = (l + 0.5)/chis
    if suvy == 'CIB':
        a = 3500 * (l/3000.0) ** (-1.25)
#        shot_noise = 1700
#        a += shot_noise
        return a
    elif suvy == 'LSST':
        a = H(zs)**1 / (chis)**2 * Wi(zs,suvy) * Wi(zs,suvy) * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c 
        sa = sum(a) + (np.pi / 60 / 180)**2 / 26
        return sa
    elif suvy == 'SPT':
        a = sptebitp[l] + C_kk(l, 'SPT')
        return a
    elif suvy == 'SPT3g':
        a = spt3gitp[l] + C_kk(l, 'SPT')
        return a
    elif suvy == 'CMBs4':
        a = cmbs4itp[l] + C_kk(l, 'SPT')
        return a
    elif suvy == 'CMBs4_cob':
        a = cmbs4cobitp[l] + C_kk(l, 'SPT')
        return a
    elif suvy == 'SO':
        a = soitp[l] + C_kk(l, 'SPT')
        return a
    else:
        a = H(zs)**1 / (chis)**2 * Wi(zs,suvy) * Wi(zs,suvy) * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
        if suvy == 'LSST tomographic':
            sa = np.sum(a, axis = 1)
            print 'sa', sa
            sa = sa + l**2 / 26 / 120**2 / 180**2
            print 'sa + noise', sa
        elif suvy == 'DES tomographic':
            sa = np.sum(a, axis = 1)
        else: sa = sum(a)
        return sa

def C_ij(l,suvy1, suvy2):
#    print suvy2
    if suvy1 == 'SPT' or suvy1 == 'SPT3g' or suvy1 == 'CMBs4' or suvy1 == 'SO' or suvy1 == 'CMBs4_cob':
        a = C_ki(l, suvy2)
    elif suvy2 == 'SPT' or suvy2 == 'SPT3g' or suvy2 == 'CMBs4' or suvy2 == 'SO' or suvy2 == 'CMBs4_cob':
        a = C_ki(l, suvy1)
    else:
        k = (l + 0.5)/chis
        a = H(zs)**1 / (chis)**2 * Wi(zs,suvy1) * Wi(zs,suvy2) * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
        a = sum(a)
    return a


def C_ij_tomog(l, suvy, wni, wnj):
    if suvy == 'LSST tomographic':
        k = (l + 0.5)/chis
        a = H(zs)**1 / (chis)**2 * wni * wnj * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c 
        a = sum(a)
 #       print 'a', a
    elif suvy == 'DES tomographic':
        k = (l + 0.5)/chis
        a = H(zs)**1 / (chis)**2 * wni * wnj * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
        a = sum(a)        
    else:
        print 'wrong survey'
    return a

def H(z):
    return results.h_of_z(z)    #Mpc-1

def dn_dz(z, suvy):
    if suvy == 'LSST':        
#        alpha = 1.27; beta = 1.02
        z0 = 0.5
        dn = z**alpha_lsst * np.exp(-(z/z0)**beta_lsst)    
    elif suvy == 'LSST tomographic':
        zi0 = np.array([0.4, 0.6, 0.9, 1.2, 1.5, 1.9, 2.2, 2.8, 3.4])
        zim = 0.82 * zi0 - 0.127
        zi = np.array([0] + list(zim) + [3.5])
        dn = np.zeros((10, len(z)))
        for i in range(10):            
            dn[i,:] = 0.5 * dn_dz(z, 'LSST') * ( special.erfc( (zi[i] - z) / np.sqrt(2) / 0.03 / (1 + z) ) - special.erfc( (zi[i + 1] - z) / np.sqrt(2) / 0.03 / (1 + z) ))
    elif suvy == 'WISE':
        dn = 20 * np.exp( - (z - 0.1)**2 )
    elif suvy == 'CIB_2' or suvy == 'CIB':
        A = 0.289; z0 = 1.34; alpha_cib = 1.50
        dn = A * z**alpha_cib * np.exp( - (z/z0)**alpha_cib)
#        dn = A * z**(alpha_cib-1) * alpha_cib * (1 - alpha_cib*(z/z0)**alpha_cib) * np.exp( - (z/z0)**alpha_cib)        
    elif suvy == 'DES':
        dn = 7.7 * np.exp( -(z - 0.657)**2 / 0.043**2 / 2)
    elif suvy == 'DES tomographic':
        dn = np.zeros((4, len(z)))
        zi0 = np.array([0.5, 0.51, 0.55, 0.65, 0.8])
        #zi0 = np.array([0.5, 0.53, 0.6, 0.65, 0.78])
        #zi0 = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
        zi = zi0
        for i in range(4):            
            dn[i,:] = 0.5 * dn_dz(z, 'DES') * ( special.erfc( (zi[i] - z) / np.sqrt(2) / 0.03 / (1 + z) ) - special.erfc( (zi[i+1] - z) / np.sqrt(2) / 0.03 / (1 + z) ))
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
#    print suvy
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
        dndz = dn_dz(z, suvy)
        for i in range(10):            
            wg[i,:] = dndz[i,:] * b(z, suvy) / sum(dndz[i,:] * dz)
    elif suvy == 'DES tomographic':
        wg = np.zeros((4, len(z)))
        dndz = dn_dz(z, suvy)
        for i in range(4):                        
            wg[i,:] = dndz[i, :] * b(z, suvy) / sum(dndz[i,:] * dz)
    return wg

    
def Wi(z, suvy):
    if suvy == 'WISE':
        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
        return wg
    if suvy == 'CONST':
        wg = b(z,suvy) * 1/sum( 1*dz )
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
        bc = 1.4175980058073239e-65 * (0.83 + 0.742*z + 0.318*z*z)
        wcib = bc * chis**2 / H(z) /(1+z)**2 * np.exp( - (z - zc)**2 / 2 / sigma_z**2 ) * f_mu(545e9,z)**1
        return wcib    
    elif suvy =='CIB_2':
        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
        return wg
    else:
        wg = b(z,suvy) * dn_dz(z, suvy)/sum( dn_dz(z, suvy)*dz )
        return wg


def Wk(z):
    return 3 * omega_m / 2 / c * H(0)**2 / H(z) *(1 + z)**1 * chis * (chistar - chis) / chistar

def Cb_lens(l):
    s2 = 0
    lx = l * np.cos(np.pi/4)
    ly = l * np.sin(np.pi/4)
    for xi in range(-3000, 3001, 10):
        for yi in range(-3000, 3001, 10):
            d1 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d2 = (xi**2 + yi**2)**0.5
            d3 = l
            if d1 >= 2 and d2 >= 2: 
                if d1<=3000 and d2 <=3000:
                    cosphi = (xi * lx + yi * ly) / d3 / d2
                    sin2phi = 2 * cosphi * abs((1 - cosphi**2))**0.5
                    wll = 2 * ((lx-xi)*xi + (ly-yi)*yi) / d1**2 * sin2phi
                    d1 = int(d1); d2 = int(d2); d3 = int(d3)
                    tn2 = wll**2 * ce[d2] * cl_camb_b[d1] / (2 * np.pi)**2
                    s2 += tn2
            else:
                tn = 0
    s2 = s2*100
    return s2

def Cb_res(l, rho_list):
    s = np.zeros(rho_list.shape[0])
    s2 = 0
    lx = l * np.cos(np.pi/4)
    ly = l * np.sin(np.pi/4)
    for xi in range(-1500, 1501, 10):
        for yi in range(-1500, 1501, 10):
            d1 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d2 = (xi**2 + yi**2)**0.5
            d3 = l
            if d1 >= 2 and d2 >= 2: 
                if d1<=3000 and d2 <=3000:
                    cosphi = (xi * lx + yi * ly) / d3 / d2
                    sin2phi = 2 * cosphi * abs((1 - cosphi**2))**0.5
                    wll = 2 * ((lx-xi)*xi + (ly-yi)*yi) / d1**2 * sin2phi
                    d1 = int(d1); d2 = int(d2); d3 = int(d3)
                    tn2 = wll**2 * ce[d2] * cl_camb_b[d1] / (2 * np.pi)**2
                    s2 += tn2
  #                  if math.isnan(tn2):
                    for i, rhoi in enumerate(rho_list):
                        tn = wll**2 * ce[d2] * cl_camb_b[d1] * (1 - (ce[d2] / ce[d2]) * (rhoi[d1-2])**2) / (2 * np.pi)**2
                        s[i] += tn
            else:
                tn = 0
    s = s*10**2
    s2 = s2*10**2
    return s, s2

def f_mu(mu,z):
    muz = mu * (1+z)
    f = np.zeros(z.shape) 
    for i, zi in enumerate(z):
        if muz[i] < mup:
            f[i] = ( np.exp(h_p * muz[i] / k_b / T) -1 )**(-1) * muz[i]**(beta_cib +3)
        else:
            f[i] = np.exp(h_p * mup / k_b / T -1)**(-1) * mup**(beta_cib +3) * (muz[i]/mup)**(-2)
    return f

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

def n_fwhm(l, Noise, fwhm):
    a = 1/1.2 * 1e-7* np.exp( l**2 * fwhm**2 * (3e-4)**2/ 2.355**2 ) * Noise **2
    return a


def noise_t(lx,ly,nfwhm):
    s = 0
    for xi in range(-7000, 7001, dl_intg):
        for yi in range(-7000, 7001, dl_intg):
            d2 = int(((xi-lx)**2 + (yi-ly)**2)**0.5)
            d1 = int((xi**2 + yi**2)**0.5)
            if d1 >= lmin and d2 >= lmin:
                tn = ( ( (lx-xi)*lx + (ly-yi)*ly ) * ct[d2] + ( lx*xi + yi*ly )* ct[d1] )**2 / 2 / (ct_tot[d2] + nfwhm[d2] )/ (ct_tot[d1] + nfwhm[d1] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1.0/s/dl_intg**2

def noise_te(lx ,ly, nfwhm):
    s = 0
    for xi in range(-7000, 7001, dl_intg):
        for yi in range(-7000, 7001, dl_intg):
            d2 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d1 = (xi**2 + yi**2)**0.5
            if d1 >= lmin and d2 >= lmin:
                cosphi = (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                cos2phi = 2 * cosphi**2 - 1
                d1 = int(d1)
                d2 = int(d2)
                f1 = ( (lx - xi)*lx + (ly - yi)*ly ) * cte[d2] + ( lx*xi + yi*ly )* cte[d1]*cos2phi
                f2 = ( (lx - xi)*lx + (ly - yi)*ly ) * cte[d2]*cos2phi + ( lx*xi + yi*ly )* cte[d1]
                F = ((ce_tot[d1]+2*nfwhm[d1]) * (ct_tot[d2]+ nfwhm[d2]) * f1 - cte[d2] * cte[d1] * f2) / ( (ct_tot[d2] + nfwhm[d2]) * (ce_tot[d1] + nfwhm[d1]*2) * (ce_tot[d2]+ nfwhm[d2]*2) * (ct_tot[d1] + nfwhm[d1]) - ( (cte_tot[d1]) * (cte_tot[d2]) )**2 )
                tn = f1 * F / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1.0/s/dl_intg**2

def noise_tb(lx ,ly, nfwhm):
    s = 0
    for xi in range(-7000, 7001, dl_intg):
        for yi in range(-7000, 7001, dl_intg):
            d2 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d1 = (xi**2 + yi**2)**0.5
            if d1 >= lmin and d2 >= lmin:
                cosphi = (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                sin2phi = 2 * cosphi * (1 - cosphi**2)**0.5
                d1 = int(d1)
                d2 = int(d2)
                tn = ( ( lx*xi + yi*ly )* cte[d1] * sin2phi )**2 / (cb_tot[d2] + 2 * nfwhm[d2] )/ (ct_tot[d1] + nfwhm[d1] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1.0/s/dl_intg**2


def noise_e(lx ,ly, nfwhm):
    s = 0
    for xi in range(-7000, 7001, dl_intg):
        for yi in range(-7000, 7001, dl_intg):
            d2 = int(((xi-lx)**2 + (yi-ly)**2)**0.5)
            d1 = int((xi**2 + yi**2)**0.5)            
            if d1 >= lmin and d2 >= lmin:
                cosphi = 1.0 * (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                cos2phi = 2 * cosphi**2 - 1
                tn = ( ( (lx - xi)*lx + (ly - yi)*ly ) * ce[d2] + ( lx*xi + yi*ly )* ce[d1] )**2 * cos2phi**2 / 2 / (ce_tot[d2] + 2 * nfwhm[d2] )/ (ce_tot[d1] + 2 * nfwhm[d1] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1.0/s/dl_intg**2

def noise_b(lx ,ly, nfwhm):
    s = 0
    for xi in range(-7000, 7001, dl_intg):
        for yi in range(-7000, 7001, dl_intg):
            d1 = int(((xi-lx)**2 + (yi-ly)**2)**0.5)
            d2 = int((xi**2 + yi**2)**0.5)            
            if d1 >= lmin and d2 >= lmin:
                cosphi = 1.0 * (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                cos2phi = 2 * cosphi**2 - 1
                tn = ( ( (lx - xi)*lx + (ly - yi)*ly ) * cb_tot[d1] + ( lx*xi + yi*ly )* cb_tot[d2] )**2 * cos2phi**2 / 2 / (cb_tot[d2] + 2 * nfwhm[d2] )/ (cb_tot[d1] + 2 * nfwhm[d1] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1.0/s/dl_intg**2


def noise_eb(lx ,ly, nfwhm):
    s = 0
    for xi in range(-7000, 7001, dl_intg):
        for yi in range(-7000, 7001, dl_intg):
            d2 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d1 = (xi**2 + yi**2)**0.5
            if d1 >= lmin and d2 >= lmin:
                cosphi = (xi * (lx - xi) + yi * (ly - yi)) / d1 / d2
                sin2phi = 2 * cosphi * (1 - cosphi**2)**0.5
                d1 = int(d1)
                d2 = int(d2)
                tn = ( ((lx - xi)*lx + (ly - yi)*ly ) * cb_tot[d2] + ( lx*xi + yi*ly )* ce[d1] )**2 * sin2phi**2 / (ce_tot[d1] + 2 * nfwhm[d1] )/ (cb_tot[d2] + 2 * nfwhm[d2] ) / (2 * np.pi)**2
            else: tn = 0
            s = tn + s
    return 1.0/s/dl_intg**2



def ns_suvy(suvy, wi, wni, wnj):
    if suvy == 'LSST tomographic':
        wnij = np.maximum(wni, wnj)
        pct = sum(wnij) / sum(sum(wi))
        ns = (np.pi / 60 / 180)**2 / 26 / pct
    else: ns = 0
    return ns



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
#cmb_survey_list = ['SPT', 'SPT3g', 'SO','CMBs4','CMBs4_cob']
cmb_survey_list = ['SPT', 'SPT3g', 'SO','CMBs4_cob']
cmb_cib_list = []
comb2 = list(combinations(survey_list, 2))

#comb3 = list(combinations(['DESI', 'WISE', 'LSST', 'CIB', 'SPT', 'DES', 'WISE x SCOS'], 3))
comb3 = list(combinations(survey_list, 3))
survey_tomog_list = ['LSST tomographic', 'DES tomographic']
legd1 = []
legd2 = []
legd3 = []
legd4 = []
legd5 = []
for sy in cmb_survey_list:
    legd5.append( 'CIB ' + sy )

for sy in survey_list:
    legd1.append(sy)

for sy in comb2:
    legd2.append( sy[0] + '+' + sy[1])

for sy in comb3:
    legd3.append( sy[0] + '+' + sy[1] + '+' + sy[2])

for sy in survey_tomog_list:
    legd4.append(sy)

ls = np.arange(2,3010+1,10)
#ls = [2, 22, 62, 102, 182, 282, 382, 582, 882, 1082, 1482]
ls = np.asarray(ls)

rho = np.zeros(( len(ls), len(survey_list) ))
cki = np.zeros(( len(ls), len(survey_list) ))
cii = np.zeros(( len(ls), len(survey_list) ))
cij = np.zeros(( len(ls), len(comb2) ))
cijk = np.zeros(( len(ls), len(comb3) ))
rho_double = np.zeros(( len(ls), len(comb2) ))
rho_triple = np.zeros(( len(ls), len(comb3) ))
rho_several4 = np.zeros(len(ls))
rho_several3 = np.zeros(len(ls))
rho_several2 = np.zeros(len(ls))
rho_several1 = np.zeros(len(ls))
rho_cmb_cib = np.zeros(( len(ls), len(cmb_survey_list) ))
rho_tomog = np.zeros(( len(ls), len(survey_tomog_list) ))
rho_tomog_cross = np.zeros(len(ls))
rho_sptpol = np.zeros(len(ls))
rho_spt3g_95 = np.zeros(len(ls))
rho_spt3g_150 = np.zeros(len(ls))
rho_so_lat_93 = np.zeros(len(ls))
rho_so_lat_145 = np.zeros(len(ls))
rho_cmbs4 = np.zeros(len(ls))
rho_spt3g = np.zeros(len(ls))
rho_so = np.zeros(len(ls))


spt_nseb = np.zeros(ls.shape)
sptpol = np.zeros(ls.shape)
spt3g_95 = np.zeros(ls.shape)
spt3g_150 = np.zeros(ls.shape)
spt3g = np.zeros(ls.shape)
cmbs4_cob = np.zeros(ls.shape)
cmbs4 = np.zeros(ls.shape)
so_lat_93 = np.zeros(ls.shape)
so_lat_145 = np.zeros(ls.shape)
so_lat = np.zeros(ls.shape)

ckk = np.zeros(ls.shape)
ckc = np.zeros(ls.shape)

l_long = np.arange(0,15000,1)
nfwhm_sptpol = n_fwhm(l_long, 5, 1)
nfwhm_spt3g_95 = n_fwhm(l_long, 3.0, 1.6)
nfwhm_spt3g_150 = n_fwhm(l_long, 2.2, 1.2)
nfwhm_spt3g_220 = n_fwhm(l_long, 8.8, 1.1)
nfwhm_spt3g = 1 / ( 1/nfwhm_spt3g_95 + 1/nfwhm_spt3g_150) 
#nfwhm_spt3g = 1 / ( 1/nfwhm_spt3g_95**2 + 1/nfwhm_spt3g_150**2)**0.5
nfwhm_cmbs4 = n_fwhm(l_long, 1, 1.4)
nfwhm_cmbs4_95 = n_fwhm(l_long, 0.68, 1.4)
nfwhm_cmbs4_145 = n_fwhm(l_long, 0.96, 1.4)
nfwhm_cmbs4_cob = 1 / ( 1/nfwhm_cmbs4_95 + 1/nfwhm_cmbs4_145)
nfwhm_SO_SAT_93 = n_fwhm(l_long, 2.1, 17)
nfwhm_SO_SAT_145 = n_fwhm(l_long, 4.2, 11)
nfwhm_SO_SAT_225 = n_fwhm(l_long, 1, 1.4)
nfwhm_SO_LAT_93 = n_fwhm(l_long, 5.8, 2.2)
nfwhm_SO_LAT_145 = n_fwhm(l_long, 6.3, 1.4)
nfwhm_SO = 1 / ( 1/nfwhm_SO_LAT_93 + 1/nfwhm_SO_LAT_145)
#nfwhm_SO = 1 / ( 1/nfwhm_SO_LAT_93**2 + 1/nfwhm_SO_LAT_145**2)**0.5

theta = np.pi/2

nrandom = 100
alpha_lsst_est, beta_lsst_est = 1.27, 1.02
mu_rg, sigma_rg = 2, 0.05 # mean and standard deviation
zcf = np.random.normal(mu_rg, sigma_rg, nrandom)
sigma_zf = np.random.normal(mu_rg, sigma_rg, nrandom)
alp_zf = np.random.normal(alpha_lsst_est, 0.1, nrandom)
bet_zf = np.random.normal(beta_lsst_est, 0.1, nrandom)
shot_noise_rd = np.random.normal(1700, 700.0/3, nrandom)

#zcf[-1] = 2
#sigma_zf[-1] = 2
st_cib = np.zeros((nrandom, len(ls)))
st_cib_wi = np.zeros((nrandom, len(zs)))
st_cib_cki = np.zeros((nrandom, len(ls)))
st_cib_drho = np.zeros((nrandom, len(ls)))
st_cib_sys = np.zeros((len(cmb_survey_list), nrandom, len(ls)))

st_lsst = np.zeros((nrandom, len(ls)))
st_lsst_wi = np.zeros((nrandom, len(zs)))

ls_cres = np.arange(2,1500+1,30) ##remember!!!!!!
cblens = np.zeros(len(ls_cres))
llcblens = np.zeros(len(ls_cres))
cbres_cmb = np.zeros(( len(cmb_survey_list), nrandom, len(ls_cres)))
cbres_cmbcib = np.zeros(( len(cmb_survey_list), nrandom, len(ls_cres)))
llcbres_cmb = np.zeros(( len(cmb_survey_list), nrandom, len(ls_cres)))
llcbres_cmbcib = np.zeros(( len(cmb_survey_list), nrandom, len(ls_cres)))

for rg in range(nrandom):
    start = datetime.datetime.now()
    print rg
    zc = zcf[rg]
    sigma_z = sigma_zf[rg]
    st_cib_wi[rg,:] = Wi(zs, 'CIB')
    st_lsst_wi[rg,:] = Wi(zs, 'LSST')
    alpha_lsst = alp_zf[rg]
    beta_lsst = bet_zf[rg]
    shot_noise = shot_noise_rd[rg]
    shot_noise = 0 
    print 'zc',zc
    print 'sigma_z',sigma_z
    for i, li in enumerate(ls):
        al = li * np.cos(theta)
        bl = li * np.sin(theta)
        spt_nseb[i] = spteb[i] # data already have * li**2 * (li+1)**2/4
        ckk[i] = C_kk(li,'CIB')  
        ckc[i] = C_ki(li,'CONST')  
#        spt3g[i] = noise_eb(al, bl, nfwhm_spt3g) *li**2 * (li+1)**2/4
 #       so_lat[i] = noise_eb(al, bl, nfwhm_SO) *li**2 * (li+1)**2/4
  #      cmbs4[i] = noise_eb(al, bl, nfwhm_cmbs4) *li**2 * (li+1)**2/4
#     cmbs4_cob[i] = noise_eb(al, bl, nfwhm_cmbs4_cob) *li**2 * (li+1)**2/4


        for j, survey in enumerate(survey_list):
            if survey == 'SPT':
                cki[i][j] = ckk[i]
                cii[i][j] = ckk[i] + spt_nseb[i]
            else:
                cki[i][j] = C_ki(li, survey)
                cii[i][j] = C_ii(li, survey)

        st_cib[rg,i] = cki[i][3] / (ckk[i] * cii[i][3])**0.5
        
        if 2 == 1:
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
        
            rho_several4[i] = cross_rho(li, ['CIB','SPT','DES','WISE x SCOS'])
            rho_several3[i] = cross_rho(li, ['CIB','SPT','WISE x SCOS'])
            rho_several2[i] = cross_rho(li, ['SPT','CIB'])

            rho_spt3g[i] = (ckk[i] / (ckk[i] + spt3g[i]))**0.5
            rho_so[i] = (ckk[i] / (ckk[i] + so_lat[i]))**0.5
            rho_cmbs4[i] = (ckk[i] / (ckk[i] + cmbs4[i]))**0.5

        for j, surveys in enumerate(cmb_survey_list):
        #    rho_cmb_cib [i][j] = cross_rho(li, ['CIB',surveys])
            st_cib_sys[j][rg][i] = cross_rho(li, ['CIB',surveys])

        st_cib[rg,i] = cki[i][3] / (ckk[i] * cii[i][3])**0.5
        st_cib_cki[rg,i] = cki[i][3]
        st_lsst[rg,i] = cki[i][2] / (ckk[i] * cii[i][2])**0.5

        
    end = datetime.datetime.now()
    print (end-start)


rho_ls_long = np.arange(2, rho_ls[-1] + 1)
rho_cmbcib = np.zeros((nrandom, len(rho_ls_long)))

for j, surveys in enumerate(cmb_survey_list):
    print surveys
    start = datetime.datetime.now()
    for rg in range(nrandom):
        rhoi_itp = interp1d(ls, st_cib_sys[j][rg,:])
        rho_cmbcib[rg, :] = rhoi_itp(rho_ls_long)
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho_cmbcib)
        cbres_cmbcib[j][:,i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)
        

color_list = ['b','r','y','g']
plt.figure(figsize = (16,10))        
for i, surveys in enumerate(cmb_survey_list):
    cmbi = cbres_cmbcib[i][:,:] / cblens
    print cmbi.shape
    cres_std = np.std(cmbi, axis=0)
    cres_mean = np.mean(cmbi, axis=0)
    plt.plot(ls_cres, cres_mean, color = color_list[i])
    plt.fill_between(ls_cres, cres_mean - 3*cres_std, cres_mean + 3*cres_std, color = color_list[i], alpha=0.5)
#    plt.fill_between(ls_cres, cres_mean - 1*cres_std, cres_mean + 1*cres_std, color = color_list[i], alpha=0.5)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('Bres')
plt.legend(cmb_survey_list)
#plt.savefig('cres_shadow_cmbcib_colori_sn_rdm.png')
plt.savefig('cres_shadow_cmbcib_colori_100_nosn.png')

for i, surveys in enumerate(cmb_survey_list):
    cmbi = cbres_cmbcib[i][:,:] / cblens
    print cmbi.shape
    cres_std = np.std(cmbi, axis=0)
    cres_mean = np.mean(cmbi, axis=0)
    plt.plot(ls_cres, cres_mean, color = color_list[i])
    plt.fill_between(ls_cres, cres_mean - 3*cres_std, cres_mean + 3*cres_std, color = color_list[i], alpha=0.5)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('Bres')
plt.legend(cmb_survey_list)
#plt.savefig('cres_shadow_cmbcib_colori_sn_rdm.png')
plt.savefig('cres_shadow_cmbcib_colori_100_nosn_3sigma.png')


ckk = ckk.reshape((len(ls), 1))
rho = cki / (cii * ckk)**0.5
rho0 = rho[:, 0]
rho1 = rho[:, 1]
rho2 = rho[:, 2]
rho3 = rho[:, 3]
rho4 = rho[:, 4]
rho5 = rho[:, 5]
rho6 = rho[:, 6]

sy23id = comb2.index(('LSST', 'CIB'))
sy34id = comb2.index(('CIB', 'SPT'))
rho23 = rho_double[:,sy23id]
rho34 = rho_double[:,sy34id]


colorlist = ['r','b','m','g']
plt.figure(figsize = (8,5))        
for i, surveys in enumerate(cmb_survey_list):
    for j in range(nrandom):        
        plt.plot(ls, st_cib_sys[i][j,:], linestyle = '-', color = colorlist[i])
plt.xlim([1,ls[-1]])
plt.xlabel('l')
#plt.legend(['sptpol','spt3g','so_lat','cmbs4','cib'] + legd5)
plt.savefig('cmb_cib_rand_sn.png')

data = st_cib_sys
np.save(outfile, data)
#np.savetxt('data_gener/cmb_cib_rand.dat', data)

plt.figure(figsize = (8,5))        
for i in range(nrandom):
    plt.plot(zs, st_lsst_wi[i,:]/max(st_lsst_wi[i,:]),color = 'r', linestyle = '-')
plt.xlim([1,5])
plt.xlabel('l')
plt.legend(['lsst wi'])
#plt.savefig('lsst_rand_wi.png')

plt.figure(figsize = (8,5))        
for i, surveys in enumerate(cmb_survey_list):
    cmbi = st_cib_sys[i][:,:]
    print cmbi.shape
    rho_max = np.amax(cmbi, axis=0)
    rho_min = np.amin(cmbi, axis=0)
    rho_mean = np.mean(cmbi, axis=0)
    plt.plot(ls, rho_mean)
    plt.fill_between(ls, rho_min, rho_max, alpha=0.5)
plt.xlim([1,ls[-1]])
plt.xlabel('l')
plt.legend(cmb_survey_list)
plt.savefig('cmb_cib_rand_shadow_sn.png')

rho_max = np.amax(st_lsst, axis=0)
rho_min = np.amin(st_lsst, axis=0)
rho_mean = np.mean(st_lsst, axis=0)
plt.figure(figsize = (8,5))        
plt.plot(ls, rho_mean)
plt.fill_between(ls, rho_min, rho_max,facecolor='yellow', alpha=0.5)
plt.xlim([1,ls[-1]])
plt.xlabel('l')
plt.legend(['LSSTs'])
#plt.savefig('lsst_rand_shadow.png')

cki_max = np.amax(st_cib_cki, axis=0)
cki_min = np.amin(st_cib_cki, axis=0)
cki_mean = np.mean(st_cib_cki, axis=0)
plt.figure(figsize = (8,5))        
plt.loglog(ls, cki_mean)
plt.fill_between(ls, cki_min, cki_max,facecolor='yellow', alpha=0.5)
plt.xlim([1,ls[-1]])
plt.xlabel('l')
plt.legend(['CIBs'])
plt.savefig('cib_rand_shadow_cki_sn.png')

plt.figure(figsize = (8,5))        
for i in range(nrandom):
    plt.plot(zs, st_lsst_wi[i,:]/max(st_lsst_wi[i,:]),color = 'r', linestyle = '-')
plt.xlim([0,7])
plt.xlabel('l')
plt.legend(['lsst wi'])
#plt.savefig('lsst_rand_wi.png')

plt.figure(figsize = (8,5))        
plt.plot(ls, ckc)
plt.xlim([0,ls[-1]])
plt.xlabel('l')
plt.legend(['ckc'])
plt.savefig('ckc.png')

rho_max = np.amax(st_cib, axis=0)
rho_min = np.amin(st_cib, axis=0)
rho_mean = np.mean(st_cib, axis=0)
dwi = st_cib_wi - np.mean(st_cib_wi)
#drho = 2 * st_cib * ckc / st_cib_cki *dwi
#plt.figure(figsize = (8,5))        
#plt.plot(ls, rho_mean)
#plt.fill_between(ls, rho_min, rho_max,facecolor='yellow', alpha=0.5)
#plt.fill_between(ls, rho_mean - drho, rho_mean + drho,facecolor='blue', alpha=0.5)
#plt.xlim([1,ls[-1]])
#plt.xlabel('l')
#plt.legend(['CIBs'])
#plt.savefig('cib_rand_shadow_compare.png')

#plt.figure(figsize = (8,5))        
#plt.plot(ls, drho)
#plt.xlim([0,ls[-1]])
#plt.xlabel('l')
#plt.legend(['drho'])
#plt.savefig('drho.png')
