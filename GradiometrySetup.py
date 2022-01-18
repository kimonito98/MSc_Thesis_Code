# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:39:11 2021

@author: Michael


"""

from math import acos, atan2, sin, cos, sqrt, pi
import numpy as np


def legendre_polynomial_ext( l_max, colat ):

    lat = pi/2 - colat
    ct = sin(lat)
    st = cos(lat)
    
    p  = np.zeros((l_max+1,1))
    P = np.zeros((l_max+1,l_max+1))
    dP = np.zeros((l_max+1,l_max+1))
    ddP = np.zeros((l_max+1,l_max+1))
    
    r = np.zeros((2*l_max+1,1))
    
    for i in range(1,2*l_max+2):
        r[i-1] = sqrt(i)
       
    p[0] = 1.0
    if l_max > 0:
        p[1] = r[2]*st
    
    for l in range(2,l_max+1):
        l2 = l*2;
        fac = r[l2]/r[l2-1]
        p[l] = p[l-1]*fac*st
    
    for m in range(l_max+1):
        l = m
        P[l,m] = p[m]
        
        if l < l_max:
             l = m+1
             l2 = l*2
             fac = r[l2]
             P[l,m] = P[l-1,m]*ct*fac
             
             for l in range(m+2,l_max+1):
                l2 = l*2
                fac1 = r[l2]/r[l-m-1]/r[l+m-1]
                fac2 = r[l2-2]
                fac3 = r[l-m-2]*r[l+m-2]/r[l2-4]
                P[l,m] = fac1*(P[l-1,m]*fac2*ct-P[l-2,m]*fac3)
                
    dP[0,0] = 0
    if l_max > 0:
        dP[1,1] = sqrt(3)*ct;
    for m in range(2,l_max+1):
        dP[m,m] = sqrt((2*m+1)/(2*m)) * (ct*P[m-1,m-1]+st*dP[m-1,m-1]);
    
    for m in range(0,l_max):
        dP[m+1,m] = sqrt(2*m+3) * (-st*P[m,m]+ct*dP[m,m])
        
    for m in range(0,l_max-1):
        for l in range(m+2,l_max+1):
            dP[l,m] =   sqrt((2*l-1)*(2*l+1)/(l-m)/(l+m)) * (-st*P[l-1,m]+ct*dP[l-1,m]) - sqrt((2*l+1)*(l+m-1)*(l-m-1)/(l-m)/(l+m)/(2*l-3)) * dP[l-2,m]
    
    ddP[0,0] = 0
    if l_max > 0:
        ddP[1,1] = -sqrt(3)*st
        
    for m in range(2,l_max+1):
        ddP[m,m] = sqrt((2*m+1)/(2*m)) * (-st*P[m-1,m-1]+ct*dP[m-1,m-1]+ct*dP[m-1,m-1]+st*ddP[m-1,m-1])
        
    for m in range(0,l_max):
        ddP[m+1,m] = sqrt(2*m+3) * (-ct*P[m,m]-st*dP[m,m]-st*dP[m,m]+ct*ddP[m,m]);
        
    for m in range(0,l_max-1):
        for l in range(m+2,l_max+1):
            ddP[l,m] = sqrt((2*l-1)*(2*l+1)/(l-m)/(l+m)) * (-ct*P[l-1,m]-st*dP[l-1,m]-st*dP[l-1,m]+ct*ddP[l-1,m]) - sqrt((2*l+1)*(l+m-1)*(l-m-1)/(l-m)/(l+m)/(2*l-3)) * ddP[l-2,m]
    return [P,dP,ddP]



def get_cg_idxs(mindegree,maxdegree):
    idxs = dict()
    c = 0
    for m in range(0,maxdegree+1):
        for l in range(mindegree,maxdegree+1):
            if m == 0:
                idxs['C'+str(l)+','+str(m)] = c
                c += 1
            elif 0 < m <= l:
                idxs['C' + str(l) + ',' + str(m)] = c
                c += 1
                idxs['S' + str(l) + ',' + str(m)] = c
                c += 1
    assert (maxdegree+1)**2 - mindegree**2 == len(idxs)
    return idxs

       
def order_gg_err(err,mindegree,maxdegree, perdegree=False):
    Err_dict = dict()
    c = 0
    for m in range(0,maxdegree+1):
        for l in range(mindegree,maxdegree+1):
            if m == 0:
                Err_dict['C'+str(l)+','+str(m)] = err[c]
                c += 1
            elif 0 < m <= l:
                Err_dict['C'+str(l)+','+str(m)] = err[c]
                c += 1
                Err_dict['S'+str(l)+','+str(m)] = err[c]
                c += 1
    assert c == err.shape[0]
    if perdegree:
        Err_dict2 = dict()
        for l in range(mindegree, maxdegree+1):
            for m in range(0, l + 1):
                Err_dict2['C'+str(l)+','+str(m)] = Err_dict['C'+str(l)+','+str(m)]
                if m != 0:
                    Err_dict2['S' + str(l)+',' + str(m)] = Err_dict['S' + str(l)+',' + str(m)]
        Err_dict = Err_dict2
    return Err_dict


def flatten_dict(coeff_dict, mindegree, maxdegree, by_block = False):
    """ Return a flattened coefficients array, to multiply with design
    matrix or coeff. blocks for gradiometry"""
    n_param = (maxdegree + 1) ** 2 - mindegree ** 2
    arr, c = np.empty((n_param)), 0
    if by_block == False:
        for m in range(0, maxdegree + 1):
            for l in range(mindegree, maxdegree + 1):
                if m == 0:
                    arr[c] = coeff_dict['C' + str(l) +',' + str(m)]
                    c += 1
                elif 0 < m <= l:
                    arr[c] = coeff_dict['C' + str(l) +',' + str(m)]
                    c += 1
                    arr[c] = coeff_dict['S' + str(l) +','+ str(m)]
                    c += 1
    else:
        for l in range(mindegree, maxdegree + 1):
            for m in range(0, l + 1):
                arr[c] = coeff_dict['C' + str(l) +','+ str(m)]
                c += 1
                if m != 0:
                    arr[c] = coeff_dict['S' + str(l)+',' + str(m)]
                    c += 1
    assert c == n_param
    return arr

def legendre( l_max, colat ):
    """ Computes associated legendre function, first and second derivatives
    Full (4-pi) normalization 
    VALIDATED"""

    lat = pi/2 - colat
    ct = sin(lat)
    st = cos(lat)
    
    p  = np.zeros((l_max+1,1))
    P = np.zeros((l_max+1,l_max+1))
    dP = np.zeros((l_max+1,l_max+1))
    ddP = np.zeros((l_max+1,l_max+1))
    
    r = np.zeros((2*l_max+1,1))
    
    for i in range(1,2*l_max+2):
        r[i-1] = sqrt(i)
       
    p[0] = 1.0
    if l_max > 0:
        p[1] = r[2]*st
    
    for l in range(2,l_max+1):
        l2 = l*2
        fac = r[l2]/r[l2-1]
        p[l] = p[l-1]*fac*st
    
    for m in range(l_max+1):
        l = m
        P[l,m] = p[m]
        
        if l < l_max:
             l = m+1
             l2 = l*2
             fac = r[l2]
             P[l,m] = P[l-1,m]*ct*fac
             
             for l in range(m+2,l_max+1):
                l2 = l*2
                fac1 = r[l2]/r[l-m-1]/r[l+m-1]
                fac2 = r[l2-2]
                fac3 = r[l-m-2]*r[l+m-2]/r[l2-4]
                P[l,m] = fac1*(P[l-1,m]*fac2*ct-P[l-2,m]*fac3)
                
    dP[0,0] = 0
    if l_max > 0:
        dP[1,1] = sqrt(3)*ct
    for m in range(2,l_max+1):
        dP[m,m] = sqrt((2*m+1)/(2*m)) * (ct*P[m-1,m-1]+st*dP[m-1,m-1])
    
    for m in range(0,l_max):
        dP[m+1,m] = sqrt(2*m+3) * (-st*P[m,m]+ct*dP[m,m])
        
    for m in range(0,l_max-1):
        for l in range(m+2,l_max+1):
            dP[l,m] =   sqrt((2*l-1)*(2*l+1)/(l-m)/(l+m)) * (-st*P[l-1,m]+ct*dP[l-1,m]) - sqrt((2*l+1)*(l+m-1)*(l-m-1)/(l-m)/(l+m)/(2*l-3)) * dP[l-2,m]
    
    ddP[0,0] = 0
    if l_max > 0:
        ddP[1,1] = -sqrt(3)*st
        
    for m in range(2,l_max+1):
        ddP[m,m] = sqrt((2*m+1)/(2*m)) * (-st*P[m-1,m-1]+ct*dP[m-1,m-1]+ct*dP[m-1,m-1]+st*ddP[m-1,m-1])
        
    for m in range(0,l_max):
        ddP[m+1,m] = sqrt(2*m+3) * (-ct*P[m,m]-st*dP[m,m]-st*dP[m,m]+ct*ddP[m,m])
        
    for m in range(0,l_max-1):
        for l in range(m+2,l_max+1):
            ddP[l,m] = sqrt((2*l-1)*(2*l+1)/(l-m)/(l+m)) * (-ct*P[l-1,m]-st*dP[l-1,m]-st*dP[l-1,m]+ct*ddP[l-1,m]) - sqrt((2*l+1)*(l+m-1)*(l-m-1)/(l-m)/(l+m)/(2*l-3)) * ddP[l-2,m]
    return [P,dP,ddP]        

def compute_design(pos,GM,R,minimum_degree,maximum_degree):
    """ Only Axx, Ayy and Azz
        Sorting format example:
        C20, C30, C40, … // m=0
        {C21, S21}, {C31, S31}, {C41, S41}, … // m=1
        {C22, S22}, {C32, S22}, {C42, S42}, …// m=2
        {C33, S33}, {C34, S34}, …// m=3
        {C44, S44} // m=4 """
    
    epochs = pos.shape[0]
    coeffs = (maximum_degree+1)**2 - minimum_degree**2
    aor = np.zeros( (maximum_degree+3, 1) )
    Axx, Ayy, Azz = np.empty((3, epochs, coeffs))

    for k in range(epochs):
        # divide by the reference radius to improve the numerical precision (xa ~= ya ~= za ~= 1.xxx)
        xa = pos[k,0]/R
        ya = pos[k,1]/R
        za = pos[k,2]/R
    		
        p = R * sqrt(xa*xa + ya*ya)
        r = R * sqrt(xa*xa + ya*ya + za*za)
            
        sin_lambda = pos[k,1]/p
        cos_lambda = pos[k,0]/p
        cos_phi = p/r
        sin_phi = pos[k,2]/r
        tan_phi = sin_phi/cos_phi
        GMor3 = GM/r/r/r
        
        aor[0] = 1.
        aor[1] = R/r
        for l in range(2,aor.shape[0]):
            aor[l] = aor[l-1] * aor[1]
         
        [P,dP,ddP] = legendre( maximum_degree, acos(sin_phi) )
        
        counter_xx = 0
        # counter_xy = 0
        # counter_xz = 0
        counter_yy = 0
        # counter_yz = 0
        counter_zz = 0
        
    	# order m = 0   	
        for n in range(minimum_degree,maximum_degree+1):
            aorn_GMor3 = aor[n] * GMor3

            Pnm = P[n,0]  			
            dPnm = dP[n,0]
            ddPnm = ddP[n,0]
            
            Axx[k,counter_xx] = aorn_GMor3 * (ddPnm - (n+1) * Pnm)
            counter_xx+=1
            # Axy[k,counter_xy] = 0.0
            # counter_xy+=1
            # Axz[k,counter_xz] = (n+2) * aorn_GMor3 * dPnm
            # counter_xz+=1
            Ayy[k,counter_yy] = - aorn_GMor3 * ((n+1) * Pnm - tan_phi * dPnm)
            counter_yy+=1
            # Ayz[k,counter_yz] = 0.0
            # counter_yz+=1
            Azz[k,counter_zz] = (n+1)*(n+2) * aorn_GMor3 * Pnm
            counter_zz+=1
        
        cp = cos_lambda
        sp = sin_lambda
        cp2 = 1.0
        sp2 = 0.0
        
        # order greater than zero: m > 0
        for m in range(1, maximum_degree+1): # loop over parameters:
            
            # recursive computation of cos(m*phi) and sin(m*phi)
            cp3 = cp2 * cp - sp2*sp  # cos(m*phi) = cos((m-1)*phi) * cos(phi) - sin((m-1)*phi) * sin(phi)
            sp3 = sp2 * cp + cp2*sp  # sin(m*phi) = sin((m-1)*phi) * cos(phi) + cos((m-1)*phi) * sin(phi)
            cp2 = cp3
            sp2 = sp3
			
            for n in range( max(minimum_degree,m), maximum_degree+1):
                aorn_GMor3 = aor[n] * GMor3 # (a/r)^n * GM/r^3

                aorn_GMor3_cm = aorn_GMor3 * cp2  # (a/r)^n * cos(m*phi)
                aorn_GMor3_sm = aorn_GMor3 * sp2  # (a/r)^n * sin(m*phi)

                Pnm = P[n,m]  			
                dPnm = dP[n,m]
                ddPnm = ddP[n,m]
			    
                helpp = ddPnm - (n+1) * Pnm
                Axx[k,counter_xx] = aorn_GMor3_cm * helpp
                counter_xx+=1
                Axx[k,counter_xx] = aorn_GMor3_sm * helpp
                counter_xx+=1

                # helpp = m * (tan_phi * Pnm - dPnm) / cos_phi
                # Axy[k,counter_xy] = - aorn_GMor3_sm * helpp
                # counter_xy+=1
                # Axy[k,counter_xy] =   aorn_GMor3_cm * helpp
                # counter_xy+=1

                # helpp = (n+2) * dPnm
                # Axz[k,counter_xz] = aorn_GMor3_cm * helpp
                # counter_xz+=1
                # Axz[k,counter_xz] = aorn_GMor3_sm * helpp
                # counter_xz+=1
				
                helpp = - ( ((n+1) + (m*m)/(cos_phi*cos_phi)) * Pnm - tan_phi * dPnm)
                Ayy[k,counter_yy] = aorn_GMor3_cm * helpp
                counter_yy+=1
                Ayy[k,counter_yy] = aorn_GMor3_sm * helpp
                counter_yy+=1

                # helpp = ((n+2)*m) * Pnm / cos_phi
                # Ayz[k,counter_yz] =   aorn_GMor3_sm * helpp
                # counter_yz+=1
                # Ayz[k,counter_yz] = - aorn_GMor3_cm * helpp
                # counter_yz+=1

                helpp = ((n+1)*(n+2)) * Pnm
                Azz[k,counter_zz] = aorn_GMor3_cm * helpp
                counter_zz+=1
                Azz[k,counter_zz] = aorn_GMor3_sm * helpp
                counter_zz+=1
    return Axx, Ayy, Azz


def compute_design_full(pos, GM, R, minimum_degree, maximum_degree):
    """ Sorting format example:
        C20, C30, C40, … // m=0
        {C21, S21}, {C31, S31}, {C41, S41}, … // m=1
        {C22, S22}, {C32, S22}, {C42, S42}, …// m=2
        {C33, S33}, {C34, S34}, …// m=3
        {C44, S44} // m=4 """

    epochs = pos.shape[0]
    coeffs = (maximum_degree + 1) ** 2 - minimum_degree ** 2
    aor = np.zeros((maximum_degree + 3, 1))
    Axx, Axy, Axz, Ayy, Ayz, Azz = np.empty((6, epochs, coeffs))

    for k in range(epochs):
        # divide by the reference radius to improve the numerical precision (xa ~= ya ~= za ~= 1.xxx)
        xa = pos[k, 0] / R
        ya = pos[k, 1] / R
        za = pos[k, 2] / R

        p = R * sqrt(xa * xa + ya * ya)
        r = R * sqrt(xa * xa + ya * ya + za * za)

        sin_lambda = pos[k, 1] / p
        cos_lambda = pos[k, 0] / p
        cos_phi = p / r
        sin_phi = pos[k, 2] / r
        tan_phi = sin_phi / cos_phi
        GMor3 = GM / r / r / r

        aor[0] = 1.
        aor[1] = R / r
        for l in range(2, aor.shape[0]):
            aor[l] = aor[l - 1] * aor[1]

        [P, dP, ddP] = legendre(maximum_degree, acos(sin_phi))

        counter_xx = 0
        counter_xy = 0
        counter_xz = 0
        counter_yy = 0
        counter_yz = 0
        counter_zz = 0

        # order m = 0
        for n in range(minimum_degree, maximum_degree + 1):
            aorn_GMor3 = aor[n] * GMor3

            Pnm = P[n, 0]
            dPnm = dP[n, 0]
            ddPnm = ddP[n, 0]

            Axx[k, counter_xx] = aorn_GMor3 * (ddPnm - (n + 1) * Pnm)
            counter_xx += 1
            Axy[k, counter_xy] = 0.0
            counter_xy += 1
            Axz[k, counter_xz] = (n + 2) * aorn_GMor3 * dPnm
            counter_xz += 1
            Ayy[k, counter_yy] = - aorn_GMor3 * ((n + 1) * Pnm - tan_phi * dPnm)
            counter_yy += 1
            Ayz[k, counter_yz] = 0.0
            counter_yz += 1
            Azz[k, counter_zz] = (n + 1) * (n + 2) * aorn_GMor3 * Pnm
            counter_zz += 1

        cp = cos_lambda
        sp = sin_lambda
        cp2 = 1.0
        sp2 = 0.0

        # order greater than zero: m > 0
        for m in range(1, maximum_degree + 1):  # loop over parameters:

            # recursive computation of cos(m*phi) and sin(m*phi)
            cp3 = cp2 * cp - sp2 * sp  # cos(m*phi) = cos((m-1)*phi) * cos(phi) - sin((m-1)*phi) * sin(phi)
            sp3 = sp2 * cp + cp2 * sp  # sin(m*phi) = sin((m-1)*phi) * cos(phi) + cos((m-1)*phi) * sin(phi)
            cp2 = cp3
            sp2 = sp3

            for n in range(max(minimum_degree, m), maximum_degree + 1):
                aorn_GMor3 = aor[n] * GMor3  # (a/r)^n * GM/r^3

                aorn_GMor3_cm = aorn_GMor3 * cp2  # (a/r)^n * cos(m*phi)
                aorn_GMor3_sm = aorn_GMor3 * sp2  # (a/r)^n * sin(m*phi)

                Pnm = P[n, m]
                dPnm = dP[n, m]
                ddPnm = ddP[n, m]

                helpp = ddPnm - (n + 1) * Pnm
                Axx[k, counter_xx] = aorn_GMor3_cm * helpp
                counter_xx += 1
                Axx[k, counter_xx] = aorn_GMor3_sm * helpp
                counter_xx += 1

                helpp = m * (tan_phi * Pnm - dPnm) / cos_phi
                Axy[k, counter_xy] = - aorn_GMor3_sm * helpp
                counter_xy += 1
                Axy[k, counter_xy] = aorn_GMor3_cm * helpp
                counter_xy += 1

                helpp = (n + 2) * dPnm
                Axz[k, counter_xz] = aorn_GMor3_cm * helpp
                counter_xz += 1
                Axz[k, counter_xz] = aorn_GMor3_sm * helpp
                counter_xz += 1

                helpp = - (((n + 1) + (m * m) / (cos_phi * cos_phi)) * Pnm - tan_phi * dPnm)
                Ayy[k, counter_yy] = aorn_GMor3_cm * helpp
                counter_yy += 1
                Ayy[k, counter_yy] = aorn_GMor3_sm * helpp
                counter_yy += 1

                helpp = ((n + 2) * m) * Pnm / cos_phi
                Ayz[k, counter_yz] = aorn_GMor3_sm * helpp
                counter_yz += 1
                Ayz[k, counter_yz] = - aorn_GMor3_cm * helpp
                counter_yz += 1

                helpp = ((n + 1) * (n + 2)) * Pnm
                Azz[k, counter_zz] = aorn_GMor3_cm * helpp
                counter_zz += 1
                Azz[k, counter_zz] = aorn_GMor3_sm * helpp
                counter_zz += 1
    return Axx, Axy, Axz, Ayy, Ayz, Azz
    # return Axx, Ayy, Azz

def subtract_central_field_full(Axx, Axy, Axz, Ayy, Ayz, Azz, GM, R, pos):
    epochs = pos.shape[0]
    
    for k in range(epochs):
        xa = pos[k,0]/R
        ya = pos[k,1]/R
        za = pos[k,2]/R
        
        r = R * sqrt(xa*xa + ya*ya + za*za)
        GMor3 = GM/r/r/r

        # contribution of central field in LNOF
        Axx[k] += GMor3
        Ayy[k] += GMor3
        Azz[k] -= 2*GMor3
    return Axx, Axy, Axz, Ayy, Ayz, Azz


def subtract_central_field(Axx, Ayy, Azz, GM, R, pos):
    epochs = pos.shape[0]

    for k in range(epochs):
        xa = pos[k, 0] / R
        ya = pos[k, 1] / R
        za = pos[k, 2] / R

        r = R * sqrt(xa * xa + ya * ya + za * za)
        GMor3 = GM / r / r / r

        # contribution of central field in LNOF
        Axx[k] += GMor3
        Ayy[k] += GMor3
        Azz[k] -= 2 * GMor3
    return Axx, Ayy, Azz

def compute_design_nadirOnly(pos, GM, R, minimum_degree, maximum_degree):
    """ Same as above, but only radial direction"""

    epochs = pos.shape[0]
    coeffs = (maximum_degree + 1) ** 2 - minimum_degree ** 2
    aor = np.zeros((maximum_degree + 3, 1))
    # Axx, Axy, Axz, Ayy, Ayz, Azz = np.empty((6, epochs, coeffs))
    Axx, Ayy, Azz = np.empty((3, epochs, coeffs))

    for k in range(epochs):
        # divide by the reference radius to improve the numerical precision (xa ~= ya ~= za ~= 1.xxx)
        xa = pos[k, 0] / R
        ya = pos[k, 1] / R
        za = pos[k, 2] / R

        p = R * sqrt(xa * xa + ya * ya)
        r = R * sqrt(xa * xa + ya * ya + za * za)

        sin_lambda = pos[k, 1] / p
        cos_lambda = pos[k, 0] / p
        cos_phi = p / r
        sin_phi = pos[k, 2] / r
        tan_phi = sin_phi / cos_phi
        GMor3 = GM / r / r / r

        aor[0] = 1.
        aor[1] = R / r
        for l in range(2, aor.shape[0]):
            aor[l] = aor[l - 1] * aor[1]

        [P, dP, ddP] = legendre(maximum_degree, acos(sin_phi))

        counter_xx = 0

        # order m = 0
        for n in range(minimum_degree, maximum_degree + 1):
            aorn_GMor3 = aor[n] * GMor3

            Pnm = P[n, 0]
            dPnm = dP[n, 0]
            ddPnm = ddP[n, 0]

            Axx[k, counter_xx] = aorn_GMor3 * (ddPnm - (n + 1) * Pnm)
            counter_xx += 1

        cp = cos_lambda
        sp = sin_lambda
        cp2 = 1.0
        sp2 = 0.0

        # order greater than zero: m > 0
        for m in range(1, maximum_degree + 1):  # loop over parameters:

            # recursive computation of cos(m*phi) and sin(m*phi)
            cp3 = cp2 * cp - sp2 * sp  # cos(m*phi) = cos((m-1)*phi) * cos(phi) - sin((m-1)*phi) * sin(phi)
            sp3 = sp2 * cp + cp2 * sp  # sin(m*phi) = sin((m-1)*phi) * cos(phi) + cos((m-1)*phi) * sin(phi)
            cp2 = cp3
            sp2 = sp3

            for n in range(max(minimum_degree, m), maximum_degree + 1):
                aorn_GMor3 = aor[n] * GMor3  # (a/r)^n * GM/r^3

                aorn_GMor3_cm = aorn_GMor3 * cp2  # (a/r)^n * cos(m*phi)
                aorn_GMor3_sm = aorn_GMor3 * sp2  # (a/r)^n * sin(m*phi)

                Pnm = P[n, m]
                dPnm = dP[n, m]
                ddPnm = ddP[n, m]

                helpp = ddPnm - (n + 1) * Pnm
                Axx[k, counter_xx] = aorn_GMor3_cm * helpp
                counter_xx += 1
                Axx[k, counter_xx] = aorn_GMor3_sm * helpp
                counter_xx += 1

    return Axx

def rotate_design(Axx, Ayy, Azz, Rot):
    """ rotates the design matrix or observations (from left-handed LNOF to other frame) """ 
    for i in range(Axx.shape[0]):
        for j in range(Axx.shape[1]):
            a = np.array([Axx[i,j],Ayy[i,j],Azz[i,j]])*np.eye(3)
            Axx[i,j],Ayy[i,j],Azz[i,j] = np.diagonal( np.matmul(Rot[i], np.matmul( a, np.transpose(Rot[i])) ))

    return Axx, Ayy, Azz

def rotate_design_full(Axx, Axy, Axz, Ayy, Ayz, Azz, Rot):
    """ rotates the design matrix or observations (from left-handed LNOF to other frame) """
    for i in range(Axx.shape[0]):
        for j in range(Axx.shape[1]):
            a = np.array([ [Axx[i, j], Axy[i,j], Axz[i,j]],
                           [Axy[i, j], Ayy[i,j], Ayz[i,j]],
                           [Axz[i, j], Ayz[i,j], Azz[i,j]] ])

            a = np.matmul(Rot[i], np.matmul( a, np.transpose(Rot[i])) )
            Axx[i, j] = a[0,0]
            Axy[i, j] = a[1,0]
            Axz[i, j] = a[2,0]
            Ayy[i, j] = a[1,1]
            Ayz[i, j] = a[1,2]
            Azz[i,  j] = a[2,2]

    return Axx, Axy, Axz, Ayy, Ayz, Azz


def order_coeffs(cos,sin,n_min,n_max):
    coeff_dict = dict()
    for l in range(n_min,n_max+1):
        for m in range(0,l+1):
            if m != 0: 
                coeff_dict['S'+str(l)+','+str(m)] = sin[l,m]
            coeff_dict['C'+str(l)+','+str(m)] = cos[l,m]
    return coeff_dict


# def err_pyr(err_dict,coeffs,maxdegree):
#     """ Create a Normalised Error Pyramid array for plotting """
#     err_rs_py = np.zeros((maxdegree+1,maxdegree*2+1))
#     mask = np.zeros_like(err_rs_py)
    
#     for l in range(maxdegree+1):
#         lst = []
#         for m in range(l+1):
#             lst.append(abs(err_dict['C'+str(l)+str(m)]/coeffs['C'+str(l)+str(m)]))
#             if m != 0:
#                 lst.insert(0, abs(err_dict['S'+str(l)+str(m)]/coeffs['S'+str(l)+str(m)]))
#         err_rs_py[l,:] = np.array([0]*int((maxdegree*2+1-len(lst))/2) + lst + [0]*int((maxdegree*2+1-len(lst))/2))
#         mask[l,:] = np.array([1]*int((maxdegree*2+1-len(lst))/2) + [0]*len(lst) + [1]*int((maxdegree*2+1-len(lst))/2) )
#     # err_grad_n = np.delete(err_grad_n, 0, axis=0)
#     # mask = np.delete(mask, 0, axis=0)
#     err_rs_py[err_rs_py > 1 ] = 1
#     return err_pyr, mask

def err_n(err_dict,coeff_dict):
    """ Normalise the formal error by the respective coefficient """
    err_n = np.zeros(( len(err_dict) ))  
    ticks = []
    for c, (key,value) in enumerate(err_dict.items()):
        err_n[c] = value/coeff_dict[key]
        ticks.append(key)
    assert c == err_n.shape[0]-1
    return np.abs(err_n) #, ticks
        











