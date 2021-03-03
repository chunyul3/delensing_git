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


#data = np.loadtxt('/home/chunyul3/quicklens/noise_deep_wide_it2.dat')
#data = np.loadtxt('/home/chunyul3/quicklens/noise_deep_wide.dat')
data = np.loadtxt('/home/chunyul3/quicklens/noise_deep_wide_re.dat')
cmbls = data[:,0]

cmbnoise = [data[:,1], data[:,2]]

cmbs4_list = ['deep','wide']

cmbrhos = [0,0]

ckk = np.zeros(len(cmbls))

for i in range(len(cmbls)):
    ckk[i] = rho_generate.C_kk(cmbls[i],'CIB') 


for i in range(len(cmbnoise)):
    cmbrhossq = ckk / ( ckk + cmbnoise[i] * 2 * np.pi / 4 )
    cmbrhos[i] = cmbrhossq**0.5        

rho_ls = cmbls

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

nfwhm_cmbs4_95 = rho_generate.n_fwhm(ct_l, 0.68, 2)
nfwhm_cmbs4_145 = rho_generate.n_fwhm(ct_l, 0.96, 2)
nfwhm_cmbs4_cob = 1 / ( 1/nfwhm_cmbs4_95 + 1/nfwhm_cmbs4_145)

nfwhm_cmbs4_95_wide = rho_generate.n_fwhm(ct_l, 2.0, 2.0)
nfwhm_cmbs4_145_wide = rho_generate.n_fwhm(ct_l, 2.0, 2.0)
nfwhm_cmbs4_cob_wide = 1 / ( 1/nfwhm_cmbs4_95_wide + 1/nfwhm_cmbs4_145_wide)


#noise_cmbsurvey = np.column_stack((nfwhm_sptpol, nfwhm_spt3g, nfwhm_SO, nfwhm_cmbs4, nfwhm_cmbs4_cob))
noise_cmbsurvey = np.column_stack((nfwhm_cmbs4_cob, nfwhm_cmbs4_cob_wide))


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

def parameter_rho(l, para1, para2):
    rho = para1 * ( l / 100 ) ** para2
    return rho

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
                    tn = tn2 * (1 - ce[d2] / (ce[d2] + noise1[d2]) * rho[d1-2]**2 ) 
                    s += tn
            else:
                tn = 0
    s = s*10**2
    s2 = s2*10**2
    return s, s2



def P(zs, k):
    P = PK.P(zs, k ,grid = False)

'''
def n_fwhm(l, Noise, fwhm):
    a = 1/1.2 * 1e-7* np.exp( l**2 * fwhm**2 * (3e-4)**2/ 2.355**2 ) * Noise **2
    return a
'''

cmb_survey_list = ['SPT', 'SPT3g', 'SO','cmbs4','CMBs4_cob']



############# Calculate Input Rho

cmbrhositp = range(len(cmbrhos))
rho_ls_long = np.arange(2, rho_ls[-1] + 1)

A_list = np.arange(0.5, 2, 0.5)                    #para = 0.5 - 2

alpha_list = np.arange(-0.2, 0.2, 0.05)             #para -0.2 0.2

len_A = len(A_list)
len_alpha = len(alpha_list)

ls_rho = rho_ls_long

rho_para = np.zeros((len(A_list), len(alpha_list), len(ls_rho)))

for i, para1 in enumerate(A_list):
    for j, para2 in enumerate(alpha_list):
        rho = para1 * ( ls_rho / 100 ) ** para2
        rho[rho > 1] = 1
        rho_para[i, j] = rho

fig, axs = plt.subplots( len_A, len_alpha, figsize = ( len_alpha*3 ,len_A*3) )
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
count = 0
number = len_A * len_alpha
for i in range(len_A):
    for j in range(len_alpha):
        axs[count].plot(ls_rho, rho_para[i,j])
        alpstr = '%.2f' % alpha_list[j]
        title = 'A ' + str(A_list[i]) + ' alpha ' + str(alpha_list[j])
        axs[count].set_title(title)
        count = count + 1
plt.savefig('rho_range_less_1.png')

'''
for i, rho in enumerate(cmbrhos):
    rhoi_itp = interp1d(cmbls, rho)
    rhoitp = rhoi_itp(rho_ls_long)
    cmbrhositp[i] = rhoitp

'''

############## CALCULATE THE CMB SURVEY BRES

ls_cres = np.arange(1,2000+1,10) ##remember!!!!!!
cblens = np.zeros(len(ls_cres))
llcblens = np.zeros(len(ls_cres))
cbres_cmb = np.zeros(( number, len(ls_cres)))

count = 0
for i in range(len_A):
    for j in range(len_alpha):
        start = datetime.datetime.now()
        n1 = noise_cmbsurvey[:,1]
        n2 = noise_cmbsurvey[:,1]
        n1 = n1 * 2
        n2 = n2 * 2
        print i, ' of ', len_A, ' A '
        print j, ' of ', len_alpha, ' alpha '
        for ind, li in enumerate(ls_cres):
            s, s2 = Cb_res(li, rho_para[i,j], n1, n2)
            cbres_cmb[ count ][ ind ] = s
            cblens[ind] = s2
            end = datetime.datetime.now()
        count = count + 1
        print (end-start)


data = ls_cres
data = np.column_stack((data, cblens))
for i, bres in enumerate(cbres_cmb):    
    data = np.column_stack((data, bres))
header = "ls, cblens, paras. The bres is listed with parameters loop i in len_A, j in len_alpha"
#np.savetxt('cbres_cmbs4_deep_wide_it2.dat', data, header=header)
#np.savetxt('cbres_cmbs4_deep_wide_re.dat', data, header=header)
#np.savetxt('cbres_cmbs4_deep_wide_re_it1.dat', data, header=header)
np.savetxt('cbres_paras_rho.dat', data, header=header)


############# PLOT

leg = []
for i in A_list:
    for j in alpha_list:
        alpstr = '%.2f' % j
        para_ij = 'A:' + str(i) + ' alpha' + alpstr
        leg = leg + [para_ij]
plt.figure(figsize = (16,10))        
for i, bres in enumerate(cbres_cmb):
    cibi = bres / cblens
    plt.plot(ls_cres, cibi)
plt.xlim([1,ls_cres[-1]])
plt.xlabel('l')
plt.ylabel('bres of different rhos')
plt.legend(leg)
#plt.savefig('bres_cmbs4_deep_wide_it2.png')
#plt.savefig('bres_cmbs4_deep_wide_re.png')
#plt.savefig('bres_cmbs4_deep_wide_re_it1.png')
plt.savefig('bres_of_diff_rhos.png')


fig, axs = plt.subplots( len_A, len_alpha, figsize = ( len_alpha*3 ,len_A*3) )
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
count = 0
number = len_A * len_alpha
for i in range(len_A):
    for j in range(len_alpha):
        axs[count].plot(ls_cres, cbres_cmb[count] / cblens)
        alpstr = '%.2f' % alpha_list[j]
        title = 'A ' + str(A_list[i]) + ' alpha ' + str(alpha_list[j])
        axs[count].set_title(title)
        count = count + 1
plt.savefig('bres_range.png')
