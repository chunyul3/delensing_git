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
from rho_fun import rho_generate

##### DEFINE PARAMETERS

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


########### INPUT DATA:
##########  cmb surveys
##########  cmb surveys corss rho with cib
##########  interp plank datas and their rhos
##########  original plank data
##########  
##########  
##########  
#data = np.loadtxt('/home/chunyul3/delensing_git/data_gener/cmb_rho.dat')
#data = np.loadtxt('/home/chunyul3/quicklens/noiselsst_full_real_it2.dat')
noise = np.loadtxt('/home/chunyul3/quicklens/noise_3muk_ql.dat')
cmbls = noise[:,0]

cmbnoises = [noise[:,1], noise[:,2],noise[:,3],noise[:,4],noise[:,5],noise[:,6]]

cmb_list = ['EB', 'EE', 'TT','TB','TE','MV']
cmbrhos = [0,0,0,0,0,0]
ckk = np.zeros(len(cmbls))
noiserhos = range(len(cmb_list))
for i in range(len(cmbls)):
    ckk[i] = rho_generate.C_kk(cmbls[i],'CIB') 


for i in range(len(cmbnoises)):
    cmbrhossq = ckk / ( ckk + cmbnoises[i] * 2 * np.pi / 4 )
    noiserhos[i] = cmbrhossq**0.5
        
print noiserhos


#spt3gcibrhos = range(len(cib_rho_list_string))



data = np.loadtxt('/home/chunyul3/delensing_git/data_input/cmb_tot.dat')
ct_l = data[:, 0]
ct_tot = data[:, 1]
ce_tot = data[:, 2]
cb_tot = data[:, 3]
cte_tot = data[:, 4]

data2 = np.loadtxt('/home/chunyul3/delensing_git/data_input/cmb.dat')
ct_l = data2[:, 0]
ct = data2[:, 1]
ce = data2[:, 2]
cb = data2[:, 3]
cte = data2[:, 4]
zeros = np.zeros(10000)
ct = np.append(ct, zeros)
ct_tot = np.append(ct_tot, zeros)
ce = np.append(ce, zeros)
ce_tot = np.append(ce_tot, zeros)
cb = np.append(cb, zeros)
cb_tot = np.append(cb_tot, zeros)
cte = np.append(cte, zeros)
cte_tot = np.append(cte_tot, zeros)

################ input ckk

##################  START THE CAMB CLACULATION

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
#print chis
dz = (zs[2:]-zs[:-2])/2
zs = zs[1:-1]
w = np.ones(chis.shape)


PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
    hubble_units=False, k_hunit=False, kmax=kmax,
    var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zs[-1])
cl_camb = results.get_lens_potential_cls(12000)
cl_camb_b = cl_camb[:, 0]
print 'len cl_camb_b', len(cl_camb_b)


##################### START THE DEFINITION OF THE FUNCTIONS

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

def Cb_res(l, rho, noise1, noise2):
#    s = np.zeros(rho_list.shape[0])
    s2 = 0
    s = 0
    lx = l * np.cos(np.pi/4)
    ly = l * np.sin(np.pi/4)
    for xi in range(-3000, 3001, 10):
        for yi in range(-3000, 3001, 10):
            d1 = ((xi-lx)**2 + (yi-ly)**2)**0.5
            d2 = (xi**2 + yi**2)**0.5
            d3 = l
            if d1 >= 2 and d2 >= 2: 
                if d1<=2640 and d2 <=2640:
#                if d1<=3000 and d2 <=3000:
                    cosphi = (xi * lx + yi * ly) / d3 / d2
                    sin2phi = 2 * cosphi * abs((1 - cosphi**2))**0.5
#                    sin2phi = 2 * cosphi * (1 - cosphi**2)**0.5
                    wll = 2 * ((lx-xi)*xi + (ly-yi)*yi) / d1**2 * sin2phi
                    d1 = int(d1); d2 = int(d2); d3 = int(d3)
                    tn2 = wll**2 * ce[d2] * cl_camb_b[d1] / (2 * np.pi)**2
                    s2 += tn2
                    rhoi = rho
                    #                  if math.isnan(tn2):
#                    tn = wll**2 * ce[d2] * cl_camb_b[d1] * (1 - (ce[d2] / (ce[d2] + noise1[d2])) * ( ckk[d1-2] / ( ckk[d1-2] + noise2[d1-2] ))) / (2 * np.pi)**2
                    tn = tn2 * (1 - ce[d2] / (ce[d2] + noise1[d2]) * rho[d1-2]**2 ) 
                    s += tn
            else:
                tn = 0
    s = s*10**2
    s2 = s2*10**2
    return s, s2



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

'''
def n_fwhm(l, Noise, fwhm):
    a = 1/1.2 * 1e-7* np.exp( l**2 * fwhm**2 * (3e-4)**2/ 2.355**2 ) * Noise **2
    return a
'''


#legd1 = []
#for sy in cmb_survey_list:
#    legd6.append( 'CIB_bc ' + sy )

############# DEFINE INTEPERATION L RANGE

# sptpolcibrhos spt3gcibrhos solatcibrhos cmbs4cibrhos 
noiseitp = range(len(cmb_list))

ls_cres = np.arange(1,3000+1,10) ##remember!!!!!!
cblens = np.zeros(len(ls_cres))
llcblens = np.zeros(len(ls_cres))
cbres_sptpolcib = np.zeros(( len(noiserhos), len(ls_cres)))

rhols = cmbls

rho_ls_long = np.arange(2, rhols[-1] + 1)



#################### CIB

for i, rho in enumerate(noiserhos):
    rhoi_itp = interp1d(cmbls, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    noiseitp[i] = rhoitp




############## CALCULATE THE CMB-CIB CORSS SURVEY BRES
'''
for j, rho in enumerate(sptpolcibrhositp):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_sptpolcib[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)

'''

cblens = np.zeros(len(ls_cres))
cbres_cmb = np.zeros(( len(cmb_list), len(ls_cres)))
nfwhm_3muk = rho_generate.n_fwhm(ct_l, 3, 0)
noise_cmbsurvey = np.column_stack((nfwhm_3muk*2, nfwhm_3muk*2, nfwhm_3muk*2, nfwhm_3muk*2, nfwhm_3muk*2, nfwhm_3muk*2))

for j, rho in enumerate(noiseitp):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmb[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)

############# SAVE DATA
data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmb):    
    data = np.column_stack((data, bres))

header = "ls, cblens, EB, EE TT, TB, TE, MV"
np.savetxt('cbres_3muk.dat', data, header=header)

############# PLOT

'''
plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_sptpolcib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('SPTPOL_CIB Bres')
plt.legend(cibcmb_bres)
plt.savefig('bres_sptpolcib.png')


'''

plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_cmb):
    cibi = bres
    #cibi = bres / cblens
    plt.plot(ls_cres, cibi)
    #plt.plot(ls_cres, cibi)
#plt.xlim([1,ls_cres_short[-1]])
plt.plot(ls_cres, cblens)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('CMBS survey Bres')
plt.legend(cmb_list)
plt.savefig('bres_3muk.png')

################CIB CMB
'''
plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_sptpolcib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi, linestyle = '-', label = 'sptpol '+ cibcmb_bres[i])

for i, bres in enumerate(cbres_spt3gcib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi, linestyle = '--', label = 'spt3g '+ cibcmb_bres[i])

for i, bres in enumerate(cbres_solatcib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi, linestyle = '-.', label = 'solat '+ cibcmb_bres[i])

for i, bres in enumerate(cbres_cmbs4cib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi, linestyle = ':', label = 'cmbs4 '+ cibcmb_bres[i])

for i, bres in enumerate(cbres_cmbs4cobcib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi, linestyle = '-', label = 'cmbs4cob '+ cibcmb_bres[i])

plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('bres')
plt.legend(loc='upper left', ncol=2)
plt.setp( plt.gca().get_legend().get_frame(), visible=False)

plt.ion()

plt.savefig('bres_redshift_it2.png')
'''
