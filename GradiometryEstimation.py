# -*- coding: utf-8 -*-
"""
Created on Sun May  2 09:30:26 2021

@author: Michael

Gradiometry

"""

import os
if os.getcwd() == 'C:\\Users\\Michael':
    os.chdir('PyPhobos')

# my own
import sys
sys.path.append('/home/mplumaris/anaconda3/lib/python3.8/site-packages')
from kernel.interface import spice_interface
from kernel.simulation import propagation_setup
from kernel.simulation import estimation_setup
from kernel.simulation.estimation_setup import observation_setup
from kernel.astro import conversion, frames
from kernel.math import interpolators

import RadioScienceSetup as rss
import SimulationSetup as ss
import GradiometrySetup as gs

import numpy as np
from matplotlib import pyplot as plt, colors

font_size = 15
plt.rcParams.update({'font.size': font_size})

#%%

spice_interface.load_standard_kernels()
start_epoch = 946774800.0
end_epoch = 948146399.9712064

bodies_to_propagate = ["Vehicle"]
central_bodies = ["Phobos"]
r_Ph, mu_Ph, dmax_Ph = 14.0E3, 7.084E5, 10
mu_Ma, r_Ma, dmax_Ma = 0.4282837566395650E+14, 3396e+3, 10
muStd_Ma =  0.2151084000000000E+06

cfs_Ph = ss.read_Phobos_gravity_field( model = "HM")[2]
c_std, s_std, labels = ss.compute_stats_HT_Phobos(max_degree = 3)
# cfs_Ma = ss.read_Mars_gravity_field(std=False)[2]

bodies = ss.get_bodies( r_Ph, mu_Ph, mu_Ma, r_Ma, model = "HM", Monte_Carlo = False  )
acceleration_models = ss.get_acceleration_models( bodies, central_bodies, dmax_Ma, dmax_Ph )
termination_settings = ss.get_termination_settings( start_epoch, max_days = 1 )

x0_inertial = np.load('x0_inertial.npy')
pareto = np.load('pareto.npy')


#%% CAI Values

mindegree, maxdegree = 1, 6
T_k, K_eff, mass = 1e-12, 16102564.1, 1.44e-25
N_atoms, C, T, T_cool, d = 5*10**5, 0.97, 5, 1, 0.5

sigma_v = np.sqrt(T_k*K_eff/mass)
t_m = 2*T+T_cool
sigma_g = 1/(C * np.sqrt(N_atoms) * K_eff * T**2)
sigma_gamma = sigma_g * np.sqrt(2 * t_m)/d

n_rms = maxdegree - mindegree + 1
n_cfs_cg = (maxdegree+1)**2 - mindegree**2
batch = 1000
W = np.eye(batch)/(sigma_gamma**2)
R_RSW2GRF = np.array([[0,  1,  0], [0,  0, -1], [1,  0,  0]])

#%% Gradiometry, Free Rotation (needs my libraries)

# integrator step = measurement rate instead of interpolating
integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, t_m)
termination_settings = ss.get_termination_settings( start_epoch, max_days = 1 )


dependent_variables_to_save = [
            propagation_setup.dependent_variable.rotation_matrix_to_body_fixed_frame("Phobos"),
            propagation_setup.dependent_variable.longitude( "Vehicle", "Phobos" ),
            propagation_setup.dependent_variable.latitude( "Vehicle", "Phobos" ),
            propagation_setup.dependent_variable.relative_position( "Vehicle", "Mars" ),
            propagation_setup.dependent_variable.relative_velocity( "Vehicle", "Mars" )
]


gf_err = np.empty((len(x0_inertial), 3,  n_rms))
# err_grad_std = np.empty((len(x0_inertial), 3,  n_rms))
# cor_grad = np.empty(( len(x0_inertial), 3,  2))

for c, x0 in enumerate(x0_inertial[0:1]):
    print('Solution '+str(c))
    propagator_settings = propagation_setup.propagator.translational( central_bodies, acceleration_models,
        bodies_to_propagate, x0, termination_settings, output_variables = dependent_variables_to_save)
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
    bodies, integrator_settings, propagator_settings)
    dependent_variables = dynamics_simulator.dependent_variable_history
    states = dynamics_simulator.state_history
    r_PhF = np.empty((len(states),3))
    R_LNOF2GRF = np.empty ((len(states),3,3))

    for ep, (s,d) in enumerate(zip(states.values(),dependent_variables.values())):
        # Compute spacecraft position in Phobos-fixed frame
        R_Inrt2PhF = d[0:9].reshape(3,3)
        r_PhF[ep] = np.matmul(R_Inrt2PhF, s[0:3])
        # Rotation from LNOF --> Phobos-Fixed --> Inertial --> GRF (z switches sign!!!)
        r_PhF[ep][2] = r_PhF[ep][2]*(-1)
        R_LNOF2PhF = frames.lv_to_body_fixed_rotation_matrix(d[9],d[10])
        # rotate 90 positive around z, 90  negative around x
        R_Inrt2GRF = np.matmul(R_RSW2GRF, frames.inertial_to_rsw_rotation_matrix(d[11:17]))
        # misal[ep] = np.arccos( np.dot(R_Inrt2GRF, s[0:3]/np.linalg.norm(s[0:3]) ))
        R_LNOF2GRF[ep] = np.matmul( R_Inrt2GRF, np.matmul( np.transpose(R_Inrt2PhF),R_LNOF2PhF))

    [Axx, Ayy, Azz] = gs.compute_design(r_PhF[5:-10], mu_Ph, r_Ph, mindegree, maxdegree)
    [Axx, Ayy, Azz] = gs.subtract_central_field(Axx, Ayy, Azz, mu_Ph, r_Ph, r_PhF[5:-10] )
    [Axx, Ayy, Azz] = gs.rotate_design(Axx, Ayy, Azz, R_LNOF2GRF[5:-10])
    
    n_ops = int( Azz.shape[0]/batch)
    cov_grad = np.zeros((3, n_param,n_param))
    for i in range(n_ops):
        cov_grad[0] = cov_grad[0] + np.matmul(np.matmul(Axx[i*batch:(i+1)*batch,:].T,W),Axx[i*batch:(i+1)*batch,:])
        cov_grad[1] = cov_grad[1] + np.matmul(np.matmul(Ayy[i*batch:(i+1)*batch,:].T,W),Ayy[i*batch:(i+1)*batch,:])
        cov_grad[2] = cov_grad[2] + np.matmul(np.matmul(Azz[i*batch:(i+1)*batch,:].T,W),Azz[i*batch:(i+1)*batch,:])
    for i in range(3):
        cov_grad[i] = np.linalg.inv(cov_grad[i])
        err_grad = gs.order_gg_err(np.sqrt(np.diagonal(np.abs(cov_grad[i]))), mindegree, maxdegree, False)
        err_grad_rms[c, i], err_grad_std[c,i]  = rss.err_stats(err_grad, cfs_Ph, ingore_S2X= True)
        cg = np.divide(cov_grad[i],np.matmul(np.fromiter(err_grad.values(),dtype=float).reshape(1,n_param).T,
                                                    np.fromiter(err_grad.values(),dtype=float).reshape(1,n_param)))
        cn = np.linalg.cond(cg)
        corr_grad[c, i] = np.array([cn, abs(cg[0, 15])]) if cn < 1e+16 else np.array([1e+16, 1])

#%% Save Gradiometry Solutions

# err_grad_rms = err_grad_rms.reshape(err_grad_rms.shape[0], -1)
# np.savetxt('err_grad_rms.txt', err_grad_rms, delimiter = ',')
# err_grad_std = err_grad_std.reshape(err_grad_std.shape[0], -1)
# np.savetxt('err_grad_std.txt', err_grad_std, delimiter = ',')

# corr_grad = corr_grad.reshape(corr_grad.shape[0], -1)
# np.savetxt('corr_grad.txt', corr_grad, delimiter = ',')

#%% Load Gradiometry Solutions
# n_rms = 10 #maxdegree - mindegree + 1
# l = 34 # len(x0_inertial)
#
# err_grad_rms = np.loadtxt('err_grad_rms.txt', delimiter=",")
# err_grad_rms = err_grad_rms.reshape(l, err_grad_rms.shape[1] // n_rms, n_rms)
#
# corr_grad = np.loadtxt('corr_grad.txt', delimiter=",")
# corr_grad = corr_grad.reshape(l, corr_grad.shape[1] // 2, 2)

#%% Save best axis for each solution
priority = 2 # choose axis according to best accuracy for RMS D&O 2
idx0, idx1, idx2 = np.arange(c), np.empty((c), dtype = int), np.ones((c), dtype=int)
for i, (err,corr) in enumerate(zip(err_grad_rms[0:c],corr_grad[0:c])):
    idx1[i] = np.argmin(err[:,priority-1]) # accuracy OR correlation
    # idx1[i] = np.argmin(np.mean(corr, axis=1), axis = 0)


#%% Formal Error Propagation, Mars coefficient uncertainty 


for c, x0 in enumerate(x0_inertial[0:1]):
    print('Solution '+str(c))
    propagator_settings = propagation_setup.propagator.translational( central_bodies, acceleration_models,
        bodies_to_propagate, x0, termination_settings, output_variables = dependent_variables_to_save)
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
    bodies, integrator_settings, propagator_settings)
    dependent_variables = dynamics_simulator.dependent_variable_history
    R_LNOF2GRF = np.empty ( (len(dependent_variables),3,3) )
    r_MaF, r_MaF_prt = np.empty( (2, len(dependent_variables), 3))
    misal = np.empty((len(dependent_variables), 3))
    for ep, d in enumerate(dependent_variables.values()):
        R_Inrt2MaF = d[0:9].reshape(3,3)
        r_MaF[ep] = np.matmul(R_Inrt2MaF, d[9:12])
        r_MaF[ep][2] = r_MaF[ep][2]*(-1)
        r_MaF_prt[ep] = r_MaF[ep] + np.random.normal(0, 100, 3)
        R_LNOF2MaF = frames.lv_to_body_fixed_rotation_matrix(d[15],d[16])
        R_Inrt2GRF = np.matmul(R_RSW2GRF, frames.inertial_to_rsw_rotation_matrix(d[9:15]))
        R_LNOF2GRF[ep] = np.matmul( R_Inrt2GRF, np.matmul( np.transpose(R_Inrt2MaF),R_LNOF2MaF))
    [Axx, Ayy, Azz] = hf.compute_design(r_MaF[5:-10], mu_Ma, r_Ma, mindegree, maxdegree)
    [Axx, Ayy, Azz] = hf.rotate_design(Axx, Ayy, Azz, R_LNOF2GRF[5:-10])
    cov_V = np.zeros((3, len(Axx)))
    for i in range(len(Axx)):
        cov_V[:,i] = np.diagonal(np.matmul( np.array([Axx[i],Ayy[i],Azz[i]]),
                    np.matmul(cov_cfsMa, np.transpose(np.array([Axx[i],Ayy[i],Azz[i]])) ) ) )
    std_V = np.sqrt(cov_V)

#%%
# np.savetxt('std_V.txt', std_V, delimiter = ',')
# np.savetxt('V_noise_pos.txt', V_noise_pos, delimiter=',')
# np.savetxt('V_noise_cfsStd.txt', V_noise_cfsStd, delimiter=',')

std_V = np.loadtxt('std_V.txt', delimiter=",")
V_noise_pos = np.loadtxt('V_noise_pos.txt', delimiter=",")
V_noise_cfsStd = np.loadtxt('V_noise_cfsStd.txt', delimiter=",")

# %%
# d_sat_M = dependent_variable_list

fig, ax1 = plt.subplots(figsize = (10,5))

ax1.plot(std_V[0], linestyle='solid', color = 'blue', lw=2, label ='$V_{xx}$ ')
# ax1.plot(V_noise_pos[0], lw=2, color = 'b',  label ='$V_{xx}$')

ax1.plot(std_V[1], linestyle='dashed', color = 'orange', lw=2, label ='$V_{yy}$ ')
# ax1.plot(V_noise_pos[1], lw=2, color = 'green', label ='$V_{yy}$')

ax1.plot(std_V[2], linestyle='solid', color = 'green', lw=2, label ='$V_{zz}$ ')
# ax1.plot(V_noise_pos[2], lw=2, color='orange', label ='$V_{zz}$')

ax1.hlines(sigma_gamma/sqrt(t_m), xmin=0, xmax=V_noise_cfsStd.shape[1],
    linestyles = 'dashed', lw=2, color = 'black', label = '$\sigma_{\gamma}$')
ax1.set_ylabel('$[s^{-2}]$')
ax1.set_xlabel('Measurement #')
ax1.set_yscale('log')
ax1.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('$|r_{sat,M}| [km]$', color='tab:gray')
ax2.plot(d_sat_M*1e-3, lw=2, color = 'tab:gray')
ax2.tick_params(axis='y', labelcolor='tab:gray')
# ax2.legend()

# fig.tight_layout() 
# fig.suptitle('Phobos Ephemeris induced error')
fig.suptitle('Mars gravity field induced error')


#%% Interpolator help
"""         
    dependent_variables = dynamics_simulator.dependent_variable_history
    benchmark_interpolator = interpolators.create_one_dimensional_interpolator(dependent_variables,
                                                                               interpolators.lagrange_interpolation(8))
    epochs = np.arange(simulation_start_epoch, tf, grad_rate)
    interpol_dict = dict()
    for epoch in epochs:
        interpol_dict[epoch] = benchmark_interpolator.interpolate(epoch)
    interpolated_pos = np.vstack( list( interpol_dict.values( ) ) )

"""

#%% Plot Misalignment

# ang = np.empty( (len(states), 3) )
# for c, (s,d) in enumerate(zip(states.values(),dependent_variables.values())):
#     R_Inrt2GRF = np.matmul(np.array([[ 0,  1,  0], [ 0,  0, -1], [ 1,  0,  0]]),
#                        frames.inertial_to_rsw_rotation_matrix(d[9:15]))
#     ang[c] = np.arccos( np.dot(R_Inrt2GRF, s[0:3]/np.linalg.norm(s[0:3]) ))
# ang = np.rad2deg(ang)

# fig = plt.figure(figsize = (13,10))
# ax = fig.add_subplot(3, 2, 1, projection='3d')
# (xM,yM,zM) = hf.drawSphere(0, 0, 0, 3389)
# ax.plot_wireframe(xM, yM, zM, color="grey")
# ax.plot(1.01*r_SM[:, 0], 1.01*r_SM[:,1], 1.01*r_SM[:,2], lw = 0.5, color = 'blue')
# ax.plot(r_PM[:, 0],r_PM[:,1], r_PM[:,2], lw = 0.5, color = 'red')
# # ax.arrow(x = r_SM[0,0], y = r_SM[0,1], dx = 1, dy=1)
# set_lims(ax, 0,np.max(rM))
# ax.set_title('Mars-centred, inertial')

# ax2 = fig.add_subplot(3, 2, 2, projection='3d')
# (xP,yP,zP) = hf.drawSphere(0, 0, 0, 11)
# ax2.plot_wireframe(xP, yP, zP, color="red")
# ax2.plot(states_list[:, 0]*0.001, states_list[:,1]*0.001, states_list[:,2]*0.001, color = 'blue')
# set_lims(ax2, -np.max(rP),np.max(rP))
# ax2.set_title('Phobos-centred, inertial')

# ax3 = fig.add_subplot(3, 2, 3, projection='3d')
# ax3.plot_wireframe(xM, yM, zM, color="grey")
# ax3.plot(1.01*pos_SM[:, 0], 1.01*pos_SM[:,1], 1.01*pos_SM[:,2], lw = 0.5, color = 'blue')
# ax3.plot(pos_PM[:, 0],pos_PM[:,1], pos_PM[:,2], lw = 0.5, color = 'red')
# set_lims(ax3, 0,np.max(rM))
# ax3.set_title('Mars-centred, fixed')


# ax4 = fig.add_subplot(3, 2, 4, projection='3d')
# ax4.plot_wireframe(xP, yP, zP, color="red")
# ax4.plot(pos_SP[:, 0], pos_SP[:,1], pos_SP[:,2], color = 'blue')
# set_lims(ax4, -np.max(rP),np.max(rP))
# ax4.set_title('Phobos-centred, fixed')

# ax5 = fig.add_subplot(3, 1, 3)
# ax5.plot(ang[:, 0], label = 'Vxx misalignment')
# ax5.plot(ang[:, 1], label = 'Vyy misalignment')
# ax5.plot(ang[:, 2], label = 'Vzz misalignment')
# ax5.legend()