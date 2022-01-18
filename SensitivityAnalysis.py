# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:30:08 2021

@author: Michael


Perform a sensitivity Analaysis on the Phobos orbits, and compute the maintenance budget

"""

import os
if os.getcwd() == 'C:\\Users\\Michael':
    os.chdir('PyPhobos')

from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.astro import conversion, observations, frames
from tudatpy.kernel.simulation.estimation_setup import observation_setup
from tudatpy.kernel.simulation import estimation_setup
from tudatpy.kernel.math import interpolators

# my own
# import sys
# sys.path.append('/home/mplumaris/anaconda3/lib/python3.8/site-packages')
# from kernel.interface import spice_interface
# from kernel.simulation import propagation_setup
# from kernel.simulation import estimation_setup
# from kernel.simulation.estimation_setup import observation_setup
# 
import SimulationSetup as ss
# import GradiometrySetup as gs
import numpy as np
from math import pi, sqrt, ceil
from matplotlib import pyplot as plt, colors

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid


#%% Setup Simulation

font_size = 15
plt.rcParams.update({'font.size': font_size})

# Load spice kernels.
spice_interface.load_standard_kernels([])
spice_interface.load_standard_kernels(['mar097.bsp'])
# spice_interface.load_standard_kernels(['mar097.bsp', 'NOE-4-2020.bsp'])

start_epoch = 946774800.0


central_bodies = ["Phobos"]
bodies_to_propagate = ["Vehicle"]

r_Ph, mu_Ph, dmax_Ph =  14.0E3, 7.11E5, 4 #084E5, 4
mu_Ma, r_Ma, dmax_Ma = 0.4282837566395650E+14, 3396e+3, 10
mu_De = 9.46e+4

frame_origin = "Mars"
frame_orientation = "ECLIPJ2000"

bodies = ss.get_bodies( frame_origin, frame_orientation,
                    r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De, 
                    Phobos_spice = True, model = "HM")


x0_inertial = np.load('x0_inertial.npy')
pareto = np.load('pareto.npy')


acceleration_models = ss.get_vehicle_acceleration_models( bodies, central_bodies, dmax_Ma, dmax_Ph)

#%% Compute STD for sensitivity

# cfs_MaStd = ss.read_Mars_gravity_field( std = True)[2]
# gf_MaStd = np.insert(gs.flatten_dict(cfs_MaStd, 1, 10, by_block = True), 0, 0.2151084E+06)

c_std, s_std, labels = ss.compute_stats_HT_Phobos(max_degree = 10)
cfs_Ph = ss.read_Phobos_gravity_field( model = "HM")[2]

#%% Investigate and Plot STM and Sensitivity influence

delta_p = np.hstack(( np.array([700]), np.max(c_std, axis = 0), np.max(s_std, axis = 0)))
delta_x0 = np.hstack((np.ones(3)*50, np.ones(3)*3e-3))

delta_r = np.zeros((len(x0_inertial), 2))

termination_settings = ss.get_termination_settings( start_epoch, max_days = 1 )
stepsize = 75
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(start_epoch,
                      stepsize, stepsize, stepsize,  1.0, 1.0)
tf = start_epoch + 86400

for c, x0 in enumerate(x0_inertial):
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies, acceleration_models, bodies_to_propagate, x0, termination_settings)
    sensitivity_parameters = ss.create_sensitivity_parameters(propagator_settings, bodies)
    variational_equations_solver = estimation_setup.SingleArcVariationalEquationsSolver(
        bodies, integrator_settings, propagator_settings, sensitivity_parameters,
        integrate_on_creation = 1 )
    phi = variational_equations_solver.state_transition_matrix_history
    sen = variational_equations_solver.sensitivity_matrix_history
    delta_r[c, 0] = np.linalg.norm(np.matmul(phi[tf], delta_x0)[0:3])
    rootsumsquare = 0
    for cc, s in enumerate(sen[tf].T):
        rootsumsquare += np.linalg.norm((np.dot(s,delta_p[cc])**2)[0:3])
    delta_r[c, 1] = sqrt(rootsumsquare)
  
#%% Plot
fig = plt.figure(figsize=(10, 6))
grid = ImageGrid(fig, 111, nrows_ncols=(1,2),
                 axes_pad=0.2, share_all=False,
                 cbar_location="right",
                 cbar_mode="single",
                label_mode = 'all',
                cbar_pad=0.1,
                aspect = False)


titles = ('Injection Errors', 'Environment Model Errors')

for i, ax in enumerate(grid.axes_all):

    im = ax.scatter(x = pareto[:,0], y = pareto[:,1], c = delta_r[:,i]*1e-3, 
                    cmap=plt.cm.viridis, norm=colors.LogNorm())
    ax.set_yticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('$\overline{r}$'+' [km]')
    ax.set_title(titles[i])
    
    if i == 0:
        ax.set_ylabel('$\overline{euc}$'+' [-]') 
    axins = zoomed_inset_axes(ax, 2, loc=8)
    axins.scatter(x = pareto[:,0], y = pareto[:,1], c = delta_r[:,i]*1e-3, 
                    cmap=plt.cm.viridis, norm=colors.LogNorm())
    axins.set_xlim(20.5, 25)
    axins.set_ylim(0.32, 0.337)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
tick_list = [0.5, 1, 2, 4]
cb = fig.colorbar(im, cax=ax.cax, ticks = tick_list)
cb.set_ticklabels(list(map(str, tick_list)))
cb.set_label('$\Delta$'+'r [km] after 1 day')

#%% Optimise interval and manoeuvre_duration

delta_p = np.hstack(( np.array([700]), np.max(c_std, axis = 0), np.max(s_std, axis = 0)))
delta_x0 = np.hstack((np.ones(3)*50, np.ones(3)*3e-3))

arc_length = [12, 24, 48]
manoeuvre_duration = [1, 2, 3]
stepsize = 75

delta_r = np.zeros((len(x0_inertial)))
delta_v = np.zeros((len(x0_inertial), len(arc_length), len(manoeuvre_duration)))

t0 = start_epoch
t2 = t1= t0 + 86400
for c, x0_init in enumerate(x0_inertial):
    x0 = x0_init
    print('orbit ' + str(c) )
    for i in range(len(arc_length)):
        for j in range(len(manoeuvre_duration)):
            t0, t1, t2 = start_epoch, start_epoch, start_epoch
            injection = True
            x0 = x0_init
            while t2 < start_epoch + 48 * 3600:
                t1 += arc_length[i] * 3600
                t2 = t1 + manoeuvre_duration[j] * 3600
                termination_settings = ss.get_termination_settings( t0, end_epoch = t2 )
                integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(t0,
                      stepsize, stepsize, stepsize,  1.0, 1.0)
                propagator_settings = propagation_setup.propagator.translational(
                    central_bodies, acceleration_models, bodies_to_propagate, x0, termination_settings)
                sensitivity_parameters = ss.create_sensitivity_parameters(propagator_settings, bodies)
                variational_equations_solver = estimation_setup.SingleArcVariationalEquationsSolver(
                    bodies, integrator_settings, propagator_settings, sensitivity_parameters,
                    integrate_on_creation = 1 )
                phi = variational_equations_solver.state_transition_matrix_history
                sen = variational_equations_solver.sensitivity_matrix_history
                delta_r1 = np.matmul(phi[t1], delta_x0)[0:3] + np.matmul(sen[t1], delta_p)[0:3]
                delta_r[c] = np.linalg.norm(delta_r1)
                if injection:
                    delta_r2 = np.matmul(phi[t2], delta_x0)[0:3] + np.matmul(sen[t2], delta_p)[0:3]
                    injection = False
                else:
                    delta_r2 = np.matmul(sen[t2], delta_p)[0:3]
                phi_21 = np.matmul(phi[t2], np.linalg.inv(phi[t1]))
                delta_v1 = np.matmul( np.linalg.inv(phi_21[0:3,3:6]), delta_r2 )
                delta_v2 = np.matmul( phi_21[3:6,3:6], delta_v1 )
                delta_v[c,i,j] += np.linalg.norm(delta_v1) + np.linalg.norm(delta_v2)

                t0 = t2
                x0 = variational_equations_solver.state_history[t2]

np.save('delta_v_2d.npy', delta_v)
#%% plot results
delta_v = np.load('delta_v_2d.npy')[1:]
arc_length = [12, 24, 48]
manoeuvre_duration = [1, 2, 3]
fig = plt.figure(figsize=(17,17))
grid = ImageGrid(fig, 111, nrows_ncols=(3,3), axes_pad=0.2, share_all=False,
                 cbar_location="right", cbar_mode="edge", cbar_size="7%",
                 cbar_pad=0.20, aspect = False, )

cutoff = 10
cmap = plt.cm.viridis
for c, a in enumerate(grid.axes_column):
    for r, ax in enumerate(a):
        idxs = np.where(delta_v[:,r,c]<=cutoff)
        maxi = ceil(np.max(delta_v[:,r,:][idxs]))
        mini = np.round(np.min(delta_v[:,r,:][idxs]),2)
        ticks = [mini, 0.5, 1, 2.5, 5, maxi]
        norm = colors.BoundaryNorm(ticks, cmap.N)
        im = ax.scatter(pareto[:,0][idxs], pareto[:,1][idxs], c = delta_v[:,r,c][idxs],
                        cmap=cmap, norm= norm)
        if r == 0:
            # ax.set_xticklabels([])
            ax.set_title(str(manoeuvre_duration[c]) + '-hour duration')
        if r == 1:
            ax.set_xticklabels([])
        if r == 2:
            ax.set_xlabel('$\overline{r}$'+' [km]')
        cb = ax.cax.colorbar(im, ticks=ticks)
        axins = zoomed_inset_axes(ax, 2, loc=8)
        axins.scatter(pareto[:,0][idxs], pareto[:,1][idxs], c = delta_v[:,r,c][idxs],
                        cmap=cmap, norm =  norm)
        axins.set_xlim(20.2, 25)
        axins.set_ylim(0.32, 0.34)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        if c == 0:
            ax.set_ylabel(str(arc_length[r]) + '-hour frequency \n'+'$\overline{euc}$'+' [-]')
        ax.set_yticklabels([])
        cb.set_label('$\Delta V$' + ' cost [m/s]')
        cb.set_ticklabels(list(map(str, ticks)))
        
# fig.suptitle('Two-impulse cost for 2-day maintenance ')


#%% Monte Carlo manoeuvre number and delta_V

models = ('DR','HF','IED','ISC','PC','RP')
draws_per_orbit = 10
interval = 86400
manoeuvre_duration = 2 * 3600
tf = start_epoch + 7 * 86400 + manoeuvre_duration
stepsize = 50
thresholds = [0.1, 0.2, 0.3]

# delta_v = np.zeros((len(pareto), 3, 2)) # delta_v, # of mans, 
# fails = np.zeros((len(pareto), 3, 2)) # crashes and escapes
# crash_model = np.zeros((3, 6))
for c, x0 in enumerate(x0_inertial):
    if c < 57:
        continue
    print('Orbit '+ str(c))
    x0_init = x0
    bodies = ss.get_bodies( frame_origin, frame_orientation,
                        r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De, 
                        Phobos_spice = True, model = "HM")
    acceleration_models = ss.get_vehicle_acceleration_models(bodies, central_bodies, dmax_Ma, dmax_Ph)
    integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, stepsize)
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies, acceleration_models, bodies_to_propagate, x0, tf)
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
        bodies, integrator_settings, propagator_settings)
    states_HM = dynamics_simulator.state_history
    
    for cc, t in enumerate(thresholds):
        threshold = t * (np.linalg.norm(states_HM[start_epoch][0:3]) - r_Ph)
        dv, mans = 0, 0
        for i in np.random.randint(0, len(models), draws_per_orbit):
            bodies = ss.get_bodies( frame_origin, frame_orientation,
                        r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De, 
                        Phobos_spice = True, model = models[i])
            acceleration_models = ss.get_vehicle_acceleration_models(bodies, central_bodies, dmax_Ma, dmax_Ph)
            x0 = x0_init
            dv_temp, mans_temp = 0, 0
            t0 = t1 = t2 = start_epoch
            injection, crash_or_escape = True, False
            while t2 < tf:
                t1 += interval
                t2 = t1 + manoeuvre_duration
                termination_settings = ss.get_termination_settings( t0, end_epoch = t2 )
                integrator_settings = propagation_setup.integrator.runge_kutta_4(t0, stepsize)
                if injection == True: 
                      x0 = x0 + np.concatenate((np.random.normal(0, 50, 3), np.random.normal(0, 0.03, 3)))
                      injection = False
                propagator_settings = propagation_setup.propagator.translational(
                    central_bodies, acceleration_models, bodies_to_propagate, x0, termination_settings)
                parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
                variational_equations_solver = estimation_setup.SingleArcVariationalEquationsSolver(
                    bodies, integrator_settings, propagator_settings,
                    estimation_setup.create_parameters_to_estimate(parameter_settings, bodies),
                    integrate_on_creation = 1)
                states = variational_equations_solver.state_history
                te = list(states.keys())[-1]
    
                if te != t2: # crashed or escaped
                    crash_model[cc, i] += 1
                    crash_or_escape = True
                    if np.linalg.norm(states[te][0:3]) < 15e3:
                        fails[c, cc, 0] += 1
                    else:
                        fails[c, cc, 1] += 1
                    break
                deviation = states_HM[t1][0:3] - states[t1][0:3]
                if np.linalg.norm(deviation) > threshold:
                    mans_temp += 1
                    phi = variational_equations_solver.state_transition_matrix_history
                    phi_21 = np.matmul( phi[t2], np.linalg.inv(phi[t1]))
                    delta_v1 = np.matmul( np.linalg.inv(phi_21[0:3,3:6]), deviation)
                    delta_v2 = np.matmul( phi_21[3:6,3:6], delta_v1)
                    dv_temp += np.linalg.norm(delta_v1) + np.linalg.norm(delta_v2)
                    x0 = states_HM[t2]
                    tm = t1
                else:
                    x0 = states[t2]
                t0 = t2
            
            if not crash_or_escape:
                dv += dv_temp
                mans += mans_temp
                
        if draws_per_orbit - np.sum(fails[c, cc]) != 0:
            delta_v[c, cc, 0] = dv/(draws_per_orbit - np.sum(fails[c, cc]))
            delta_v[c, cc, 1] = ceil(mans/(draws_per_orbit - np.sum(fails[c, cc])))

# array([ 9.5709571 , 30.69306931, 14.52145215, 17.16171617, 18.15181518,
        # 9.9009901 ])
crash_model = np.mean(crash_model, axis=0)/np.sum(np.mean(crash_model, axis=0)) * 100

fails = fails*np.array([1,-1])/draws_per_orbit
for f in fails:
    for ff in f:
        ff[0] = ff[np.argmax(np.abs(ff))]
fails = fails[:,:,0]*100

#%% 
# np.save('delta_v.npy',delta_v)
# np.save('crash_model.npy',crash_model)
# np.save('fails.npy',fails)

delta_v = np.load('delta_v.npy')[1:]
fails = np.load('fails.npy')[1:]
# crash_model = np.load('crash_model.npy')

thresholds = [0.1, 0.2, 0.3]

#%%

fig = plt.figure(figsize=(20,20))
grid = ImageGrid(fig, 111, nrows_ncols=(3,3), axes_pad=0.1, share_all=False,
                 cbar_location="right", cbar_mode="edge", cbar_size="7%",
                 cbar_pad=0.15, aspect = False)
cmap = plt.cm.viridis

for c, a in enumerate(grid.axes_column):
    for r, ax in enumerate(a):
        if r == 0:
            idxs = np.where(delta_v[:,c,0]<=50)
            im = ax.scatter(pareto[:,0][idxs], pareto[:,1][idxs], c = delta_v[:,c,0][idxs],
                            cmap=cmap, norm=colors.LogNorm())
            axins = zoomed_inset_axes(ax, 2, loc=8)
            axins.scatter(pareto[:,0][idxs], pareto[:,1][idxs], c = delta_v[:,c,0][idxs],
                            cmap=cmap, norm=colors.LogNorm())
            tick_list = [5, 10, 20]
            cb = ax.cax.colorbar(im, ticks=tick_list)
            cb.set_yticklabels(list(map(str, tick_list)))
            cb.set_label('$\overline{\Delta V}$' + ' [m/s]')
            ax.set_title(str(thresholds[c]) + '$h_{v,Ph}$'+' threshold')
        if r == 1:
            norm = colors.BoundaryNorm([0,2,4,6,7], cmap.N)
            ticks = np.arange(0,8)
            im = ax.scatter(x = pareto[:,0], y = pareto[:,1],c = delta_v[:,c,1],
                            cmap=cmap, norm = norm )
            axins = zoomed_inset_axes(ax, 2, loc=8)
            axins.scatter(pareto[:,0][idxs], pareto[:,1][idxs], c = delta_v[:,c,1][idxs],
                            cmap=cmap, norm=norm)
            cb = ax.cax.colorbar(im)
            cb.set_label('$\overline{n. mans}$')
        if r == 2:
            tickloc = [-60, -20, -10, 10, 20, np.max(fails)]
            ticklab = [str((abs(a)))+'% cr.' if a>0 else str(int(abs(a)))+'% es.' for a in tickloc]
            norm = colors.BoundaryNorm(tickloc, cmap.N)
            im = ax.scatter(x = pareto[:,0], y = pareto[:,1], c = fails[:,c],
                            cmap=plt.cm.PiYG, norm = norm)
            axins = zoomed_inset_axes(ax, 2, loc=8)
            axins.scatter(x = pareto[:,0], y = pareto[:,1], c = fails[:,c],
                            cmap=plt.cm.PiYG, norm = norm)
            cb = ax.cax.colorbar(im)
            cb.ax.set_yticklabels(ticklab)

            # idxs2 = np.where((pareto[:,0]<22.5)& (pareto[:,0]>20.2)&
            #                  (pareto[:,1]<0.324)& (pareto[:,1]>0.317))
            # ax.text(31.5, 0.34, str(round(np.mean(fails[:,c][idxs2]),2)) + '% cr.')
            
        axins.set_xlim(20.6, 24.8)
        axins.set_ylim(0.32, 0.337)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        ax.set_yticklabels([])
        ax.set_xlabel('$\overline{r}$'+' [km]')
        ax.set_ylabel('$\overline{euc}$'+' [-]')
#
#%% Plot STD of HT gravity

c_stats, s_stats, labels = ss.compute_stats_HT_Phobos(max_degree = 4, both=True)
cfs_Ph = ss.read_Phobos_gravity_field( model = "HM")[2]

#%%
fig, ax = plt.subplots(figsize = (10,6) )
cols = ['m','orange','b','c','r','g']
xval = np.arange(1,5)
xticks = [50,100,150,200]
# for j in range(2):
for i in range(6):
    ax.scatter(x = xval, y = np.abs(c_stats[0,i]-1), 
                # yerr = np.vstack((np.zeros_like(c_stats[1,i]),c_stats[1,i])),
                s = c_stats[1,i]*1e5,
                label = labels[i], c = cols[i]
                #fmt = 'o'
                )    
    # ax.scatter(xval, -s_stats[0,i], c = cols[i])
    
ax.set_yscale('log')
ax.grid(b=True, which='both')
ax.set_xticks(xval)
ax.set_xticklabels(xval)
ax.set_ylabel('$ \mu(C_{l,HT})C_{l,HM}$') 
# ax.set_yticks(xticks)
# ax.set_yticklabels(xticks)
ax.set_xlabel('Harmonic Degree')
ax.legend( ncol = 3, loc = 'lower right')


