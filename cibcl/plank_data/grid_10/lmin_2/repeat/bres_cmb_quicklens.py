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
data = np.loadtxt('/home/chunyul3/quicklens/noiselssttomo_full_real_it2.dat')
datan1 = np.loadtxt('/home/chunyul3/quicklens/noiseplnbef_full_real_it1.dat')
datan2 = np.loadtxt('/home/chunyul3/quicklens/noiseplnbcs_full_real_it1.dat')
datan3 = np.loadtxt('/home/chunyul3/quicklens/noisebefbef_full_real_it1.dat')
datan4 = np.loadtxt('/home/chunyul3/quicklens/noisebcsbcs_full_real_it1.dat')
cmbls = datan1[:,0]

cmbnoise1 = [datan1[:,1], datan1[:,2],datan1[:,3],datan1[:,4],datan1[:,5]]
cmbnoise2 = [datan2[:,1], datan2[:,2],datan2[:,3],datan2[:,4],datan2[:,5]]
cmbnoise3 = [datan3[:,1], datan3[:,2],datan3[:,3],datan3[:,4],datan3[:,5]]
cmbnoise4 = [datan4[:,1], datan4[:,2],datan4[:,3],datan4[:,4],datan4[:,5]]

#'''
noisesptpol = data[:,1]

noisespt3g = data[:,2]

noisesolat = data[:,3]

noisecmbs4 = data[:,4]

noisecmbs4cob = data[:,5]
#'''
cmb_survey_list = ['SPT', 'SPT3g', 'SO','CMBs4','CMBs4_cob']
cmbnoises = [noisesptpol, noisespt3g, noisesolat, noisecmbs4, noisecmbs4cob]
cmbrhos = [0,0,0,0,0]
cmbcibrhos = [0,0,0,0,0]
cmblsstrhos = [0,0,0,0,0]
cmbcib1 = [0,0,0,0,0]
cmbcib2 = [0,0,0,0,0]
cmbcib3 = [0,0,0,0,0]
cmbcib4 = [0,0,0,0,0]
ckk = np.zeros(len(cmbls))
for i in range(len(cmbls)):
    ckk[i] = rho_generate.C_kk(cmbls[i],'CIB') 

cib_rho_list_string = ['rho_pln_beff', 'rho_pln_bconst', 'rho_beff_beff', 'rho_bconst_bconst']
data2 = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/grid_10/lmin_2/repeat/cib_rho_noise_1.1e3_norm.dat')
rhocib1 = np.zeros(len(cmbls))
rhocib2 = np.zeros(len(cmbls))
rhocib3 = np.zeros(len(cmbls))
rhocib4 = np.zeros(len(cmbls))
rhocib1[2:] = data2[:,1]
rhocib2[2:] = data2[:,2]
rhocib3[2:] = data2[:,3]
rhocib4[2:] = data2[:,4]

data3 = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/grid_10/lmin_2/repeat/cib_lsst.dat')
rholsst = np.zeros(len(cmbls))
rholsst[2:] = data3[:,3]     ############## data3[:,2] is lsst, data3[:,3] is tomographic


for i in range(len(cmbcib1)):
    cmbrhossq1 = ckk / ( ckk + cmbnoise1[i] * 2 * np.pi / 4 )
    cmbrhossq2 = ckk / ( ckk + cmbnoise2[i] * 2 * np.pi / 4 )
    cmbrhossq3 = ckk / ( ckk + cmbnoise3[i] * 2 * np.pi / 4 )
    cmbrhossq4 = ckk / ( ckk + cmbnoise4[i] * 2 * np.pi / 4 )

    cmbcibrhossq = 1 - 1 / (1/(1-rhocib1**2) + cmbrhossq1/(1-cmbrhossq1))
    cmbcib1[i] = cmbcibrhossq**0.5
    cmbcibrhossq = 1 - 1 / (1/(1-rhocib2**2) + cmbrhossq2/(1-cmbrhossq2))
    cmbcib2[i] = cmbcibrhossq**0.5
    cmbcibrhossq = 1 - 1 / (1/(1-rhocib3**2) + cmbrhossq3/(1-cmbrhossq3))
    cmbcib3[i] = cmbcibrhossq**0.5
    cmbcibrhossq = 1 - 1 / (1/(1-rhocib4**2) + cmbrhossq4/(1-cmbrhossq4))
    cmbcib4[i] = cmbcibrhossq**0.5


for i in range(len(cmbnoises)):
    cmbrhossq = ckk / ( ckk + cmbnoises[i] * 2 * np.pi / 4 )
    cmbrhos[i] = cmbrhossq**0.5
    cmbcibrhossq = 1 - 1 / (1/(1-rhocib1**2) + cmbrhossq/(1-cmbrhossq))
    cmbcib1[i] = cmbcibrhossq**0.5
    cmbcibrhossq = 1 - 1 / (1/(1-rhocib2**2) + cmbrhossq/(1-cmbrhossq))
    cmbcib2[i] = cmbcibrhossq**0.5
    cmbcibrhossq = 1 - 1 / (1/(1-rhocib3**2) + cmbrhossq/(1-cmbrhossq))
    cmbcib3[i] = cmbcibrhossq**0.5
    cmbcibrhossq = 1 - 1 / (1/(1-rhocib4**2) + cmbrhossq/(1-cmbrhossq))
    cmbcib4[i] = cmbcibrhossq**0.5
    cmblsstrhossq = 1 - 1 / (1/(1-rholsst**2) + cmbrhossq/(1-cmbrhossq))
    cmblsstrhos[i] = cmblsstrhossq**0.5
        

sptpolcibrhos = range(len(cib_rho_list_string))
spt3gcibrhos = range(len(cib_rho_list_string))
solatcibrhos = range(len(cib_rho_list_string))
cmbs4cibrhos = range(len(cib_rho_list_string))
cmbs4cobcibrhos = range(len(cib_rho_list_string))
#rho_pln_beff, rho_pln_bconst, rho_beff_beff, rho_bconst_bconst


################ l form 2 to 2642
data = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/sptpolcibrhos.dat')
rhols = data[:,0]
for i in range(len(cib_rho_list_string)):
#    rhosq = 1 - 1 / (1/(1-rho**2) + rhosptpol**2/(1-rhosptpol**2))
    sptpolcibrhos[i] = data[:,i+1]
data = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/spt3gcibrhos.dat')
for i in range(len(cib_rho_list_string)):
    spt3gcibrhos[i] = data[:,i+1]
data = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/solatcibrhos.dat')
for i in range(len(cib_rho_list_string)):
    solatcibrhos[i] = data[:,i+1]
data = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/cmbs4cibrhos.dat')
for i in range(len(cib_rho_list_string)):
    cmbs4cibrhos[i] = data[:,i+1]
data = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/cmbs4cobcibrhos.dat')
for i in range(len(cib_rho_list_string)):
    cmbs4cobcibrhos[i] = data[:,i+1]


############## l form 192 to 2642
data = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/cib_rho_palnk.dat')
plankls = data[:,0]
rho_plank_beff = data[:,1]
rho_plank_bconst = data[:,2]


############## only several data points
data = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/cib_rho_palnk_data.dat')
plank_l = data[:,0]
plank_data_rho_beff = data[:,1]
plank_data_rho_bconst = data[:,2]


#############  l form 2 to 2642
data = np.loadtxt('/home/chunyul3/delensing_git/cibcl/plank_data/cib_rho_noise_1.1e3_norm.dat')
rho_ls = data[:,0]
rho_pln_beff = data[:,1]
rho_pln_bconst = data[:,2]
rho_beff_beff = data[:,3]
rho_bconst_bconst = data[:,4]



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

########################### input noise
nfwhm_sptpol = rho_generate.n_fwhm(ct_l, 5, 1)

nfwhm_spt3g_95 = rho_generate.n_fwhm(ct_l, 3.0, 1.6)
nfwhm_spt3g_150 = rho_generate.n_fwhm(ct_l, 2.2, 1.2)
nfwhm_spt3g = 1 / ( 1/nfwhm_spt3g_95 + 1/nfwhm_spt3g_150) 

nfwhm_SO_LAT_93 = rho_generate.n_fwhm(ct_l, 5.8, 2.2)
nfwhm_SO_LAT_145 = rho_generate.n_fwhm(ct_l, 6.3, 1.4)
nfwhm_SO = 1 / ( 1/nfwhm_SO_LAT_93 + 1/nfwhm_SO_LAT_145)

nfwhm_cmbs4 = rho_generate.n_fwhm(ct_l, 1, 1.4)

nfwhm_cmbs4_95 = rho_generate.n_fwhm(ct_l, 0.68, 1.4)
nfwhm_cmbs4_145 = rho_generate.n_fwhm(ct_l, 0.96, 1.4)
nfwhm_cmbs4_cob = 1 / ( 1/nfwhm_cmbs4_95 + 1/nfwhm_cmbs4_145)

noise_cmbsurvey = np.column_stack((nfwhm_sptpol, nfwhm_spt3g, nfwhm_SO, nfwhm_cmbs4, nfwhm_cmbs4_cob))

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

cmb_survey_list = ['SPT', 'SPT3g', 'SO','cmbs4','CMBs4_cob']
cmb_cib_list = []

legd1 = []
legd2 = []
legd3 = []
legd4 = []

legd5 = []
legd6 = []
cibcmb_bres = ['pln_beff', 'pln_bconst', 'beff_beff', 'bconst_bconst']
for sy in cib_rho_list_string:
    legd1.append( 'sptpolcib' + sy)
for sy in cmb_survey_list:
    legd5.append( 'CIB ' + sy )
for sy in cmb_survey_list:
    legd6.append( 'CIB_bc ' + sy )


# sptpolcibrhos spt3gcibrhos solatcibrhos cmbs4cibrhos 
sptpolcibrhositp = range(len(sptpolcibrhos))
spt3gcibrhositp = range(len(spt3gcibrhos))
solatcibrhositp = range(len(solatcibrhos))
cmbs4cibrhositp = range(len(cmbs4cibrhos))
cmbs4cobcibrhositp = range(len(cmbs4cobcibrhos))
cmbrhositp = range(len(cmbrhos))
cmbcibrhositp1 = range(len(cmbrhos))
cmbcibrhositp2 = range(len(cmbrhos))
cmbcibrhositp3 = range(len(cmbrhos))
cmbcibrhositp4 = range(len(cmbrhos))
cmblsstrhositp = range(len(cmbrhos))

ls_cres = np.arange(1,4000+1,10) ##remember!!!!!!
cblens = np.zeros(len(ls_cres))
llcblens = np.zeros(len(ls_cres))
cbres_sptpolcib = np.zeros(( len(sptpolcibrhos), len(ls_cres)))
cbres_spt3gcib = np.zeros(( len(spt3gcibrhos), len(ls_cres)))
cbres_solatcib = np.zeros(( len(solatcibrhos), len(ls_cres)))
cbres_cmbs4cib = np.zeros(( len(cmbs4cibrhos), len(ls_cres)))
cbres_cmbs4cobcib = np.zeros(( len(cmbs4cibrhos), len(ls_cres)))
cbres_cmb = np.zeros(( len(cmbrhos), len(ls_cres)))
cbres_cmbcib1 = np.zeros(( len(cmbrhos), len(ls_cres)))
cbres_cmbcib2 = np.zeros(( len(cmbrhos), len(ls_cres)))
cbres_cmbcib3 = np.zeros(( len(cmbrhos), len(ls_cres)))
cbres_cmbcib4 = np.zeros(( len(cmbrhos), len(ls_cres)))

cbres_plank_beff = np.zeros(len(ls_cres))
cbres_plank_bconst = np.zeros(len(ls_cres))


############# DEFINE INTEPERATION L RANGE

rho_ls_long = np.arange(2, rho_ls[-1] + 1)
plank_ls_long = np.arange(plankls[0], rho_ls[-1] + 1)

rhoi_itp = interp1d(plank_l, plank_data_rho_beff)
rhoplankbeff = rhoi_itp(plank_ls_long)
rhoi_itp = interp1d(plank_l, plank_data_rho_bconst)
rhoplankbconst = rhoi_itp(plank_ls_long)

for i, rho in enumerate(sptpolcibrhos):
    rhoi_itp = interp1d(rhols, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    sptpolcibrhositp[i] = rhoitp

for i, rho in enumerate(spt3gcibrhos):
    rhoi_itp = interp1d(rhols, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    spt3gcibrhositp[i] = rhoitp

for i, rho in enumerate(solatcibrhos):
    rhoi_itp = interp1d(rhols, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    solatcibrhositp[i] = rhoitp

for i, rho in enumerate(cmbs4cibrhos):
    rhoi_itp = interp1d(rhols, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    cmbs4cibrhositp[i] = rhoitp

for i, rho in enumerate(cmbs4cobcibrhos):
    rhoi_itp = interp1d(rhols, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    cmbs4cobcibrhositp[i] = rhoitp
'''
for i, rho in enumerate(cmbrhos):
    rhoi_itp = interp1d(cmbls, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    cmbrhositp[i] = rhoitp
'''
#################### CIB

for i, rho in enumerate(cmbcib1):
    rhoi_itp = interp1d(cmbls, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    cmbcibrhositp1[i] = rhoitp

for i, rho in enumerate(cmbcib2):
    rhoi_itp = interp1d(cmbls, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    cmbcibrhositp2[i] = rhoitp

for i, rho in enumerate(cmbcib3):
    rhoi_itp = interp1d(cmbls, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    cmbcibrhositp3[i] = rhoitp

for i, rho in enumerate(cmbcib4):
    rhoi_itp = interp1d(cmbls, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    cmbcibrhositp4[i] = rhoitp


################# LSST

for i, rho in enumerate(cmblsstrhos):
    rhoi_itp = interp1d(cmbls, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    cmblsstrhositp[i] = rhoitp

cmbcibplnbconst = [sptpolcibrhositp[1], spt3gcibrhositp[1], solatcibrhositp[1], cmbs4cibrhositp[1], cmbs4cobcibrhositp[1]]
ls_cres_short = np.arange(1,500+1,10) ##remember!!!!!!

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

for j, rho in enumerate(spt3gcibrhositp):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_spt3gcib[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)

for j, rho in enumerate(solatcibrhositp):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_solatcib[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)

for j, rho in enumerate(cmbs4cibrhositp):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmbs4cib[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)

for j, rho in enumerate(cmbs4cobcibrhositp):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmbs4cobcib[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)
'''

'''
for j, rho in enumerate(cmbrhositp):
#for j, rho in enumerate(cmbcibplnbconst):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
#    print 'n1', n1
 #   print 'n2', n2
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmb[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)
'''
###################################################### CIBCMB
'''
for j, rho in enumerate(cmbcibrhositp1):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmbcib1[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)

for j, rho in enumerate(cmbcibrhositp1):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmbcib1[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)

for j, rho in enumerate(cmbcibrhositp2):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmbcib2[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)

for j, rho in enumerate(cmbcibrhositp3):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmbcib3[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)

for j, rho in enumerate(cmbcibrhositp4):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmbcib4[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)
'''

################ LSST CMB

for j, rho in enumerate(cmblsstrhositp):
    start = datetime.datetime.now()
    n1 = noise_cmbsurvey[:,j]
    n2 = noise_cmbsurvey[:,j]
    print j
#    for i, li in enumerate(ls_cres_short):
    for i, li in enumerate(ls_cres):
        s, s2 = Cb_res(li, rho, n1, n2)
        cbres_cmb[j][i] = s
        cblens[i] = s2
    end = datetime.datetime.now()
    print (end-start)


############# SAVE DATA
'''
data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_sptpolcib):    
    data = np.column_stack((data, bres))
header = "ls,  rho_pln_beff ,  rho_pln_bconst ,  rho_beff_beff ,  rho_bconst_bconst "
np.savetxt('cbres_sptpolcib.dat', data, header=header)

data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_spt3gcib):    
    data = np.column_stack((data, bres))
header = "ls, rho_pln_beff ,  rho_pln_bconst ,  rho_beff_beff ,  rho_bconst_bconst "
np.savetxt('cbres_spt3gcib.dat', data, header=header)

data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_solatcib):    
    data = np.column_stack((data, bres))
header = "ls, rho_pln_beff ,  rho_pln_bconst ,  rho_beff_beff ,  rho_bconst_bconst "
np.savetxt('cbres_solatcib.dat', data, header=header)

data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmbs4cib):    
    data = np.column_stack((data, bres))
header = "ls, rho_pln_beff ,  rho_pln_bconst ,  rho_beff_beff ,  rho_bconst_bconst "
np.savetxt('cbres_cmbs4cib.dat', data, header=header)

data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmbs4cobcib):
    data = np.column_stack((data, bres))
header = "ls, rho_pln_beff ,  rho_pln_bconst ,  rho_beff_beff ,  rho_bconst_bconst "
np.savetxt('cbres_cmbs4cobcib.dat', data, header=header)
'''
############################# CIB CMB
'''
data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmbcib1):    
    data = np.column_stack((data, bres))
header = "ls, cblens, cressptpol, cresspt3g, cresso_lat, crescmbs4, crescmbcobs4"
np.savetxt('cbres_cmbcib_plnbef_it2.dat', data, header=header)

data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmbcib1):    
    data = np.column_stack((data, bres))
header = "ls, cblens, cressptpol, cresspt3g, cresso_lat, crescmbs4, crescmbcobs4"
np.savetxt('cbres_cmbcib_plnbst_it2.dat', data, header=header)

data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmbcib1):    
    data = np.column_stack((data, bres))
header = "ls, cblens, cressptpol, cresspt3g, cresso_lat, crescmbs4, crescmbcobs4"
np.savetxt('cbres_cmbcib_befbef_it2.dat', data, header=header)

data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmbcib1):    
    data = np.column_stack((data, bres))
header = "ls, cblens, cressptpol, cresspt3g, cresso_lat, crescmbs4, crescmbcobs4"
np.savetxt('cbres_cmbcib_bcsbcs_it2.dat', data, header=header)
'''
'''
data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmb):    
    data = np.column_stack((data, bres))
header = "ls, cblens, cressptpol, cresspt3g, cresso_lat, crescmbs4, crescmbcobs4"
np.savetxt('cbres_cmbsurvey.dat', data, header=header)
'''
'''
data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmb):    
    data = np.column_stack((data, bres))
header = "ls, cblens, crescibsptpol, crescibspt3g, crescibso_lat, crescibcmbs4, crescibcmbcobs4"
np.savetxt('cbres_cmblsst_quicklens_it2.dat', data, header=header)
'''

data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmb):    
    data = np.column_stack((data, bres))
#data = data[:50, :]
header = "ls, cblens, crescibsptpol, crescibspt3g, crescibso_lat, crescibcmbs4, crescibcmbcobs4"
np.savetxt('cbres_cmblsst_tomg_quicklens_it2.dat', data, header=header)

############# PLOT
######## cib_rho_list_string = ['rho_pln_beff', 'rho_pln_bconst', 'rho_beff_beff', 'rho_bconst_bconst']
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

plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_spt3gcib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('SPT3G_CIB Bres')
plt.legend(cibcmb_bres)
plt.savefig('bres_spt3gcib.png')

plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_solatcib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('SOLAT_CIB Bres')
plt.legend(cibcmb_bres)
plt.savefig('bres_solatcib.png')

plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_cmbs4cib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('CMBS4_CIB Bres')
plt.legend(cibcmb_bres)
plt.savefig('bres_cmbs4cib.png')

plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_cmbs4cobcib):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('CMBS4_95_145CIB Bres')
plt.legend(cibcmb_bres)
plt.savefig('bres_cmbs4cobcib.png')
'''
'''
plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_cmb):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('CMBS survey Bres')
plt.legend(cmb_survey_list)
plt.savefig('bres_short_cmblsst_quicklens_it2.png')
'''

plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_cmb):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi)
#plt.xlim([1,ls_cres_short[-1]])
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('CMBS survey Bres')
plt.legend(cmb_survey_list)
plt.savefig('bres_short_cmblssttomog_quicklens_it2.png')

################CIB CMB
'''
cbres_sptpolcib = [cbres_cmbcib1[0],cbres_cmbcib2[0], cbres_cmbcib3[0], cbres_cmbcib4[0]] 
cbres_spt3gcib = [cbres_cmbcib1[1],cbres_cmbcib2[1], cbres_cmbcib3[1], cbres_cmbcib4[1]] 
cbres_solatcib = [cbres_cmbcib1[2],cbres_cmbcib2[2], cbres_cmbcib3[2], cbres_cmbcib4[2]] 
cbres_cmbs4cib = [cbres_cmbcib1[3],cbres_cmbcib2[3], cbres_cmbcib3[3], cbres_cmbcib4[3]] 
cbres_cmbs4cobcib = [cbres_cmbcib1[4],cbres_cmbcib2[4], cbres_cmbcib3[4], cbres_cmbcib4[4]] 

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
'''
plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_cmb):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('CMBS survey Bres')
plt.legend(cmb_survey_list)
plt.savefig('bres_cmbcibplnbconst.png')

'''
