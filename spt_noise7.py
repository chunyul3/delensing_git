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

h = 0.675
alpha = 1.27; beta = 1.02;
nz = 100 #number of steps to use for the radial/redshift integration
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



spt_l = np.linspace(100,1000,10)
spt_sg = np.array([1.68, 0.88, 0.56, 0.4, 0.3, 0.24, 0.2, 0.16, 0.12, 0.104]) # *10**7
spt_sg = [i * 10**-7 for i in spt_sg]
spt_ns = np.array([0.4, 0.64, 0.88, 0.96, 1, 1.08, 1.16, 1.2, 1.28, 1.34])
spt_ns = [i * 10**-7 for i in spt_ns]
spt_itp_sg = interp1d(spt_l, spt_sg)
spt_itp_ns = interp1d(spt_l, spt_ns)

#First set up parameters as usual
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_for_lmax(12000, lens_potential_accuracy=0)
results= camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars,  lmax = 6100,CMB_unit='muK')
totCL=powers['total']
unlensedCL=powers['unlensed_scalar']
cee = unlensedCL[:,1]
print cee[0]
ct_tot = totCL[:,0]
ct = unlensedCL[:,0]
ce_tot = totCL[:,1]
ce = unlensedCL[:,1]
cb_tot = totCL[:,2]
cb = unlensedCL[:,2]
cte_tot = totCL[:,3]
cte = unlensedCL[:,3]
print 'cte', cte
ct_l = np.arange(totCL.shape[0])
ct_itp = interp1d(ct_l, ct)
ct = ct * 2 * np.pi / (ct_l + 1)**2
ct_tot = ct_tot * 2 * np.pi / (ct_l + 1)**2
ce = ce * 2 * np.pi / (ct_l + 1)**2
ce_tot = ce_tot * 2 * np.pi / (ct_l + 1)**2
cb = cb * 2 * np.pi / (ct_l + 1)**2
cb_tot = cb_tot * 2 * np.pi / (ct_l + 1)**2
cte = cte * 2 * np.pi / (ct_l + 1)**2
cte_tot = cte_tot * 2 * np.pi / (ct_l + 1)**2
print 'ct_l:', ct_l[0],'to', ct_l[-1]
ct_tot_itp = interp1d(ct_l, ct_tot)

datactt = np.column_stack((ct_l, ct_tot, ce_tot, cb_tot, cte_tot))
headerctt = "ls, tot, etot, btot, ctetot "
np.savetxt('cmb_tot.dat', datactt, header=headerctt)

datactt = np.column_stack((ct_l, ct, ce, cb, cte))
headerctt = "ls, t, e, b ,cte"
np.savetxt('cmb.dat', datactt, header=headerctt)


zeros = np.zeros(10000)
ct = np.append(ct, zeros)
ct_tot = np.append(ct_tot, zeros)
ce = np.append(ce, zeros)
ce_tot = np.append(ce_tot, zeros)
cb = np.append(cb, zeros)
cb_tot = np.append(cb_tot, zeros)
cte = np.append(cte, zeros)
cte_tot = np.append(cte_tot, zeros)


#For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
#so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts

#chistar = results.conformal_time(0)- results.tau_maxvis
chistar = results.conformal_time(0)- results.conformal_time(1090)
chis = np.linspace(0,chistar,nz)
zs=results.redshift_at_comoving_radial_distance(chis)

#Calculate array of delta_chi, and drop first and last points where things go singular
dchis = (chis[2:]-chis[:-2])/2
chis = chis[1:-1]
dz = (zs[2:]-zs[:-2])/2
zs = zs[1:-1]
w = np.ones(chis.shape)

#Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
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
    a = sum(a)
    return a

def C_ki(l,suvy):
    k = (l + 0.5)/chis
    a = H(zs)**1 / (chis)**2 * Wi(zs, suvy) * Wk(zs) * PK.P(zs, k,grid = False)* (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz / c
    sa = sum(a)
    return sa

def matrix_cii(l, c11, c12, c21, c22):
    # c11 = ckk + ns for spt
    # c12 = cki
    a0 = [[c11,c12],[c12,c22]]
    a = np.matrix(a0)
    return a 

def cross_rho3(l, c11, c12, c21, c22, ck1, ck2):
    a = matrix_cii(l, c11, c12, c21, c22)
    b = a.I
    kk = C_kk(l, 'spt')
    c1 = c11
    c2 = c
    r11 = ck1**2 * b[0,0]
    r12 = ck1 * ck2 * b[0,1]
    r22 = ck2**2 * b[1,1]
    rho = (r11 + r12 * 2 + r22) / kk
    rho = rho**0.5
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
    if suvy == 'CIB_pln':
        return 3500**1 * (l/3000) ** (-1.25)
    elif suvy == 'spt':
        if l > 100 and l < 1000:
            a = spt_itp_sg(l) + spt_itp_ns(l)
        else:
            a = 0
        return a
    else:
        a = H(zs)**1 / (chis)**2 * Wi(zs,suvy) * Wi(zs,suvy) * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
        return sum(a)

def C_ij(l,suvy1, suvy2):
    if suvy1 == 'spt':
        a = C_ki(l, suvy2)
    elif suvy2 == 'spt':
        a = C_ki(l, suvy1)
    else:
        k = (l + 0.5)/chis
        a = H(zs)**1 / (chis)**2 * Wi(zs,suvy1) * Wi(zs,suvy2) * PK.P(zs, k,grid = False) * (4 / (1+zs)**2 / 9 / H(0)**4 / omega_m**2)**1 * dz/ c
        a = sum(a)
    return a

def H(z):
    return results.h_of_z(z)    #Mpc-1

def dn(z, suvy):
    if suvy == 'LSST':        
        z0 = 0.5
        dn = z**alpha * np.exp(-(z/z0)**beta)    
    elif suvy == 'WISE':
        dn = 20 * np.exp( - (z - 0.1)**2 )
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

    return b
    
    
def Wi(z, suvy):
    if suvy == 'WISE':
        wg = b(z,suvy) * dn(z, suvy)/sum( dn(z, suvy)*dz )
        return wg
    if suvy == 'LSST':
        wg = b(z,suvy) * dn(z, suvy)/sum( dn(z, suvy)*dz )
        return wg
    if suvy == 'DESI':
        wg = b(z,suvy) * dn(z, suvy)/sum( dn(z, suvy)*dz )
        return wg
    elif suvy == 'CIB':
        wcib = bc * chis**2 / H(z) /(1+z)**2 * np.exp( - (z - zc)**2 / 2 / sigma_z**2 ) * f_mu(z)**1
        a = sum(wcib)
        Imu = 1.4/3*10**6
        wcib = wcib/a*Imu
        return wcib
    elif suvy =='CIB_2':
        A = 0.289; z0 = 1.34; alpha = 1.30; 
        wcib2 = A * z**alpha * np.exp( - (z/z0)**alpha)
        a = max(wcib2)
        wcib2 = wcib2/a
        return wcib2


def Wk(z):
    return 3 * omega_m / 2 / c * H(0)**2 / H(z) *(1 + z)**1 * chis * (chistar - chis) / chistar

def f_mu(z):
    muz = mu * (1+z)
    return np.exp(h_p * muz / k_b / T -1)**(-1) * muz**(beta_cib +3)

def noise(l):
    a = l
    area = dblquad(lambda x, y: ((x-a) * a * Ct( ((x-a)**2 + y**2)**0.5 ) + a*x*Ct( (x**2 + y**2)**0.5 ) )**2 / 2 / ( Ct_tot((x**2 + y**2)**0.5)* Ct_tot(((x-a)**2 + y**2)**0.5) ), 1, 2500, lambda x: 0, lambda x: 2500)


def n_fwhm(l):
    a = 1/1.2 * 1e-7* np.exp( l**2 * 1**2 * (3e-4)**2/ 2.355**2 )*25
    return a

def noise_t(a,b):
    s = 0
    for xi in range(-3000, 3001, 10):
        for yi in range(-3000, 3001, 10):
            d1 = int(((xi-a)**2 + (yi-b)**2)**0.5)
            d2 = int((xi**2 + yi**2)**0.5)
            if d1 >= 2 and d2 >= 2:
                tn = ( ( (a-xi)*a + (b-yi)*b ) * ct[d1] + ( a*xi + yi*b )* ct[d2] )**2 / 2 / (ct_tot[d2] + nfwhm[d2] )/ (ct_tot[d1] + nfwhm[d1] ) / (2 * np.pi)**2
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
#        print 'l', l
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


#ls = np.arange(102,1000-1,10, dtype=np.float64)
ls = np.arange(2,1500+1,10)
#ls = [2, 22, 62, 102, 182, 282, 382, 582, 882, 1082, 1482]
ls = np.asarray(ls)

rho0 = np.zeros(ls.shape) 
rho1 = np.zeros(ls.shape) 
rho3 = np.zeros(ls.shape) 
rho2 = np.zeros(ls.shape) 
ckk = np.zeros(ls.shape) 
ctt = np.zeros(ls.shape) 
cki0 = np.zeros(ls.shape) 
cki1 = np.zeros(ls.shape) 
cki2 = np.zeros(ls.shape) 
cki3 = np.zeros(ls.shape) 
cki4 = np.zeros(ls.shape) 
cii0 = np.zeros(ls.shape) 
cii1 = np.zeros(ls.shape) 
cii2 = np.zeros(ls.shape) 
cii3 = np.zeros(ls.shape) 
cii4 = np.zeros(ls.shape) 
c03 = np.zeros(ls.shape) 
c24 = np.zeros(ls.shape) 
rho_c03 = np.zeros(ls.shape) 
rho_c24 = np.zeros(ls.shape) 
rho_spt = np.zeros(ls.shape)
spt_nst = np.zeros(ls.shape)
spt_nse = np.zeros(ls.shape)
spt_nsb = np.zeros(ls.shape)
spt_nste = np.zeros(ls.shape)
spt_nstb = np.zeros(ls.shape)
spt_nseb = np.zeros(ls.shape)
spt_nsmv = np.zeros(ls.shape)

l_long = np.arange(0,15000,1)
nfwhm = n_fwhm(l_long)
print len(nfwhm)

theta = np.pi/2

for i, li in enumerate(ls):
    print li
    start = datetime.datetime.now()
    ckk[i] = C_kk(li,'CIB')
    al = li * np.cos(theta)
    bl = li * np.sin(theta)
    spt_nst[i] = noise_t(al,bl) * li**2 * (li+1)**2/4
    spt_nse[i] = noise_e(al,bl) * li**2 * (li+1)**2/4
#    spt_nsb[i] = noise_b(al,bl) * li**2 * (li+1)**2/4
    spt_nseb[i] = noise_eb(al,bl) * li**2 * (li+1)**2/4
    spt_nste[i] = noise_te(al,bl) * li**2 * (li+1)**2/4
    spt_nstb[i] = noise_tb(al,bl) * li**2 * (li+1)**2/4
    spt_nsmv[i] = 1/(1/spt_nst[i] + 1/spt_nse[i] + 1/spt_nste[i] + 1/spt_nstb[i] + 1/spt_nseb[i])
    print 't', spt_nst[i]
    print 'e', spt_nse[i]
    print 'b', spt_nsb[i]
    print 'eb', spt_nseb[i]
    print 'te', spt_nste[i]
    print 'tb', spt_nstb[i]
    print   '%.2E' % Decimal(spt_nst[i])
    ckk[i] = C_kk(li,'CIB') 
    ctt[i] = Ct(li)*li*(li+1)/2/np.pi
#Rou is a function which can calculate the rho directly
#But it will compute a same value for multiple times
#So we choose to do the calculation separectly
    cki0[i] = C_ki(li, 'DESI')
    cki1[i] = C_ki(li, 'WISE')
    cki2[i] = C_ki(li,'LSST')
    cki3[i] = C_ki(li,'CIB') 
    cki4[i] = ckk[i]
    cii0[i] = C_ii(li,'DESI')     
    cii1[i] = C_ii(li,'WISE')     
    cii2[i] = C_ii(li,'LSST')     
    cii3[i] = C_ii(li,'CIB')     
    cii4[i] = ckk[i] + spt_nsmv[i]
    c03[i] = C_ij(li, 'DESI', 'CIB')
    c24[i] = cki2[i]
#    c24[i] = C_ij(li, 'LSST', 'spt')
    rho_spt[i] = ckk[i] / cii4[i]
    rho_c03[i] = cross_rho3(li, cii0[i], c03[i], c03[i], cii3[i], cki0[i], cki3[i])    
    rho_c24[i] = cross_rho3(li, cii2[i], c24[i], c24[i], cii4[i], cki2[i], ckk[i])
#    rho_c03[i] = cross_rho(li, 'DESI', 'CIB')
#    rho_c24[i] = cross_rho(li, 'LSST', 'spt')
    end = datetime.datetime.now()
    print (end-start)

rho0 = cki0 / (cii0 * ckk)**0.5    
rho1 = cki1 / (cii1 * ckk)**0.5
rho2 = cki2 / (cii2 * ckk)**0.5
rho3 = cki3 / (cii3 * ckk)**0.5


dn_desi = dn(zs,'DESI')


data = np.column_stack((ls, rho0, rho1, rho2, rho3, rho_spt, rho_c03, rho_c24,spt_nst))
header = "ls, DESI, WISE, LSST, CIB, rho_spt, DESI_CIB, LSST_SPT,spt_nst "
np.savetxt('spt_lmax3k_fwhm7.dat', data, header=header)
#header = "X-Column, Y-Column"
#np.savetxt('AB_data.dat', data, header=header)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
         
plt.figure(figsize = (8,5))        
plt.plot(ls,rho0)
plt.plot(ls,rho1)
plt.plot(ls,rho2)
plt.plot(ls,rho3)
plt.plot(ls,rho_spt)
plt.plot(ls,rho_c03)
plt.plot(ls,rho_c24)
plt.xlim([1,ls[-1]])
#plt.ylim([0,1])
plt.xlabel('l')
plt.ylabel('rho')
plt.legend(['survey = DESI','survey = WISE','survey = LSST','survey = CIB_{wi}','spt','DESI + CIB', 'LSST + spt' ])
plt.savefig('spt_rho_5muk_beam1_lm3k_mv.png')

plt.figure(figsize = (8,5))        
plt.loglog(ls, ckk)
plt.loglog(ls, spt_nst)
plt.loglog(ls, spt_nse)
plt.loglog(ls, spt_nseb)
plt.loglog(ls, spt_nste)
plt.loglog(ls, spt_nstb)
plt.loglog(ls, spt_nsmv)
plt.xlabel('l')
plt.ylabel('ckk')
plt.legend(['ckk','noise t','noise e','noise eb','noise te','noise tb','noise mv'])
plt.savefig('spt_noise_5muk_beam1_lm3k_mv.png')

