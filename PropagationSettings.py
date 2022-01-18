# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:54:56 2021

@author: Michael

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

import SimulationSetup as ss
import numpy as np
from math import pi, sqrt
from matplotlib import pyplot as plt, colors
from scipy.spatial import distance

def get_integrator_settings(integrator_index: int,
                            settings_index: int,
                            simulation_start_epoch: float):
    """
    Retrieves the integrator settings.
    It selects a combination of integrator to be used (first argument) and
    the related setting (tolerance for variable step size integrators
    or step size for fixed step size integrators).""" 
    integrator = propagation_setup.integrator
    
    # Use variable step-size integrator
    if integrator_index < 4:
        multi_stage_integrators = [propagation_setup.integrator.RKCoefficientSets.rkf_45,
                                propagation_setup.integrator.RKCoefficientSets.rkf_56,
                                propagation_setup.integrator.RKCoefficientSets.rkf_78,
                                propagation_setup.integrator.RKCoefficientSets.rkdp_87]
        current_coefficient_set = multi_stage_integrators[integrator_index]
        current_tolerance = 10.0 ** (-16.0 + settings_index)
        integrator_settings = integrator.runge_kutta_variable_step_size(simulation_start_epoch,
                                                                        1.0,
                                                                        current_coefficient_set,
                                                                        1e-2,#np.finfo(float).eps, #
                                                                        np.inf,     
                                                                        current_tolerance,
                                                                        current_tolerance)
    # Use fixed step-size integrator
    fixed_step_sizes = [50,75,100,150,200,300]
    if integrator_index == 4:
        current_fixed_step_size = fixed_step_sizes[settings_index]
        integrator_settings = propagation_setup.integrator.runge_kutta_4(simulation_start_epoch,
                                                        current_fixed_step_size)
    # ABM variable order, variable step size
    if integrator_index == 5:
        current_tolerance = 10.0 ** (- 12 + 2*settings_index)
        integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(simulation_start_epoch,
                                                                                    1.0,
                                                                                    0.01,
                                                                                    np.inf,
                                                                                    current_tolerance,
                                                                                    current_tolerance)
    # ABM fixed order, variable step size
    if integrator_index > 5 and integrator_index < 9: 
        current_order = 6 + 2*(integrator_index-6) # 6, 8, 10
        current_tolerance = 10.0 ** (- 12 + 2*settings_index)
        integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(simulation_start_epoch,
                                                                                    1.0,
                                                                                    0.01,
                                                                                    np.inf,
                                                                                    current_tolerance,
                                                                                    current_tolerance,
                                                                                    current_order,
                                                                                    current_order)
    # ABM variable order, fixed step size
    if integrator_index == 9:
        current_fixed_step_size = fixed_step_sizes[settings_index]
        integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(simulation_start_epoch,
                                                                            current_fixed_step_size,
                                                                            current_fixed_step_size,
                                                                            current_fixed_step_size, 
                                                                            1.0,
                                                                            1.0)
         
    return integrator_settings

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

termination_settings = ss.get_termination_settings( start_epoch, max_days = 7 )

acceleration_models = ss.get_vehicle_acceleration_models( bodies, central_bodies,
                                                         dmax_Ma, dmax_Ph )

x0_inertial = np.load('x0_inertial.npy')
pareto = np.load('pareto.npy')

#%% Integrator Benchmark Solution
""" terminate exactly on final time or find prime factor numbers only! """
step_sizes = (1, 1.5, 2, 2.5, 3, 4, 6, 8, 10, 15, 20, 30, 60, 100, 150, 300, 450, 600, 800, 1080)
rf_err = np.empty( (len(step_sizes)-1, 6) ) # rows = steps, columns = orbit
titles = []

for o, orb in enumerate(np.linspace(0, len(pareto)-1, 6)):
    titles.append('$\overline{r}=$' + str(round(pareto[int(orb),0],1))+
                  ' km $\overline{euc.}=$'+str(round(pareto[int(orb),1],2)) )
    
    for s, step in enumerate( step_sizes ):
        integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, step)
        propagator_settings = propagation_setup.propagator.translational(
                                central_bodies,
                                acceleration_models,
                                bodies_to_propagate,
                                x0_inertial[int(orb)], 
                                termination_settings) #pareto[orb,0]*constants.JULIAN_DAY) 
    
        dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
        bodies, integrator_settings, propagator_settings)
        states = dynamics_simulator.state_history
        states_list = np.vstack( list( states.values( ) ) )
        if s == 0:
            rf = np.linalg.norm(states_list[-1,0:3])
        else:
            rf_err[s-1,o] = abs(rf-np.linalg.norm(states_list[-1,0:3]))

#%% Plot
fig, axs = plt.subplots( 2, 3, figsize = (15, 10) )
c = 0
for i in range(0,2):
    for j in range(0,3):
        axs[i,j].plot(step_sizes[1:], rf_err[:,c], '-*', c = 'r')
        axs[i,j].set_yscale('log')
        axs[i,j].set_xscale('log')
        axs[i,j].grid(True, which = 'both', ls='--')
        axs[i,j].set_title(titles[c])
        if i ==1:
            axs[i,j].set_xlabel('time step [s]')
        c += 1
axs[0,0].set_ylabel('Error in final position [m]')
axs[1,0].set_ylabel('Error in final position [m]')
#%% Validating selection 1

step = 10
r_diff = np.empty( ( int(7*86400/step)+1, 6) )

for o, orb in enumerate(np.linspace(0, len(pareto)-1, 6)):
    # Forward propagation
    integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, step)
    propagator_settings = propagation_setup.propagator.translational(
                            central_bodies,
                            acceleration_models,
                            bodies_to_propagate,
                            x0_inertial[int(orb)], 
                            termination_settings) 
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
    bodies, integrator_settings, propagator_settings)
    states = dynamics_simulator.state_history
    states_forward = np.vstack( list( states.values( ) ) )
    final_state = states_forward[-1]
    tf = list(states.keys())[-1]
    # Backward propagation
    integrator_settings = propagation_setup.integrator.runge_kutta_4(tf,-step)
    propagator_settings = propagation_setup.propagator.translational(
                            central_bodies,
                            acceleration_models,
                            bodies_to_propagate,
                            final_state, 
                            start_epoch)
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
    bodies, integrator_settings, propagator_settings)
    states = dynamics_simulator.state_history
    states_backward = np.vstack( list( states.values( ) ) )
    states_diff = states_forward[:,0:3]-states_backward[:,0:3]
    r_diff[:,o] = np.sqrt(states_diff[:,0]**2 +states_diff[:,1]**2+states_diff[:,2]**2 )
# these are the same for all orbits
time = np.fromiter(states.keys(), dtype=float)
time_hours = [(t-time[0])/86400. for t in time]

#%% Plot 
fig, axs = plt.subplots( 2, 3, figsize = (13,8) )

c = 0
for i in range(0,2):
    for j in range(0,3):
        axs[i,j].plot(time_hours, r_diff[:,c], 'b')
        axs[i,j].set_yscale('log')
        # axs[i,j].set_ylim(1e-10, 1e-3)
        axs[i,j].grid(True, which = 'major', ls='--')
        axs[i,j].set_title(titles[c])
        if i ==1:
            axs[i,j].set_xlabel('time [days]')
        c += 1
axs[0,0].set_ylabel('Position error [m]')
axs[1,0].set_ylabel('Position error [m]')

#%% Validating selection 2
step = 10
p_diff = np.empty( ( int(7*86400/step)+1, 6) )
t_diff = np.empty( ( int(7*86400/step)+1, 6) )

for o, orb in enumerate(np.linspace(0, len(pareto)-1, 6)):
    propagator_settings = propagation_setup.propagator.translational(
                        central_bodies,
                        acceleration_models,
                        bodies_to_propagate,
                        x0_inertial[int(orb)], 
                        termination_settings) 
    # 1 propagate with benchmark step
    integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, step)
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
    bodies, integrator_settings, propagator_settings)
    benchmark_states = dynamics_simulator.state_history
    epochs = np.fromiter(benchmark_states.keys(), dtype=float)
    #2 propagate with slighlty smaller step
    integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, step - 1)
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
    bodies, integrator_settings, propagator_settings)
    smallerstep_states = dynamics_simulator.state_history
    #3 propagate with encke
    propagator_settings = propagation_setup.propagator.translational(
                        central_bodies,
                        acceleration_models,
                        bodies_to_propagate,
                        x0_inertial[int(orb)], 
                        termination_settings,
                        propagator = propagation_setup.propagator.encke) 
    integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, step)
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
    bodies, integrator_settings, propagator_settings)
    encke_states = dynamics_simulator.state_history
    # compare 1&2, but first create interpolator
    interpolator = interpolators.create_one_dimensional_interpolator(smallerstep_states,interpolators.lagrange_interpolation(8))
    states_diff = dict()
    for epoch in epochs:
        states_diff[epoch] = interpolator.interpolate(epoch) - benchmark_states[epoch]
    pos_diff = np.vstack( list( states_diff.values( ) ) )[:,0:3]
    t_diff[:,o] = np.sqrt(pos_diff[:,0]**2 +pos_diff[:,1]**2+pos_diff[:,2]**2 )
     # compare 1&3, but first create interpolator
    interpolator = interpolators.create_one_dimensional_interpolator(encke_states,interpolators.lagrange_interpolation(8))
    states_diff = dict()
    for epoch in epochs:
        states_diff[epoch] = interpolator.interpolate(epoch) - benchmark_states[epoch]
    pos_diff = np.vstack( list( states_diff.values( ) ) )[:,0:3]
    p_diff[:,o] = np.sqrt(pos_diff[:,0]**2 +pos_diff[:,1]**2+pos_diff[:,2]**2 )
# these are the same for all orbits
time = np.fromiter(states_diff.keys(), dtype=float)
time_hours = [(t-time[0])/86400. for t in time]    

#%% Plot 
fig, axs = plt.subplots( 2, 3, figsize = (13,8) )

c = 0
for i in range(0,2):
    for j in range(0,3):
        axs[i,j].plot(time_hours, t_diff[:,c], 'c')
        axs[i,j].plot(time_hours, p_diff[:,c], 'm')
        # axs[i,j].set_ylabel('Position error [m]')
        if i ==1:
            axs[i,j].set_xlabel('time [days]')
        axs[i,j].set_yscale('log')
        # axs[i,j].set_ylim(1e-10, 1e-3)
        axs[i,j].grid(True, which = 'major', ls='--')
        axs[i,j].set_title(titles[c])
        c += 1
axs[0,0].set_ylabel('Position error [m]')
axs[1,0].set_ylabel('Position error [m]')
#%% Integrator Comparison

step, n_int, n_steps = 8, 10, 6
rf_err = np.empty( (6, n_int,n_steps) )
feval = np.empty( (6, n_int,n_steps) )
benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, step)
# drdt = np.empty( (6) )
titles = []
for o, orb in enumerate(np.linspace(0, len(pareto)-1, 6)):
    titles.append('$r_{mean} = $' + str(round(pareto[int(orb),0],1))+' covg. = '+str(round(pareto[int(orb),1],2)) )
    print('orbit '+str(orb))
    propagator_settings = propagation_setup.propagator.translational(
                            central_bodies,
                            acceleration_models,
                            bodies_to_propagate,
                            x0_inertial[int(orb)], 
                            start_epoch + 86400) 
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
    bodies, benchmark_integrator_settings, propagator_settings)
    states = dynamics_simulator.state_history
    time = np.fromiter(states.keys(), dtype=float)
    states_list = np.vstack( list( states.values( ) ) )
    # r = np.sqrt(states_list[:,0]**2 +states_list[:,1]**2+states_list[:,2]**2 )
    # drdt[o] = np.mean(np.abs(np.gradient(r, time)))
    rf = states_list[-1,0:3]
    for i in range(3,4):
        print('integrator '+ str(i))
        for s in range(0,n_steps):
            print('step '+str(s))
            integrator_settings = get_integrator_settings(i , s, start_epoch)
            dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
                bodies, integrator_settings, propagator_settings)
            states = dynamics_simulator.state_history
            states_list = np.vstack( list( states.values( ) ) )
            rf_err[o,i,s] = np.linalg.norm( rf - states_list[-1,0:3] )
            feval[o,i,s] = list(dynamics_simulator.get_cumulative_number_of_function_evaluations().values())[-1]


#%% Plot 

# np.save('feval.npy',feval)
# np.save('rf_err.npy',rf_err)

# rf_err = np.load('rf_err.npy')
# feval = np.load('feval.npy')

fig, axs = plt.subplots( 2, 3, figsize = (25,20) )

integrators = ('RKF45', 'RKF56', 'RKF78', 'RKF87-DP', 'RK4',
               'ABM var. ord./step', 'ABM ord. 6, var. step',
               'ABM ord. 8, var. step', 'ABM ord. 10, var. step',
               'ABM var. ord., fixed step')
o = 0
for i in range(2):
    for j in range(3):
        for k in range(0,10):
            # omit too tight tolerances which cause error
            if k in (0,1,3) :
                idx = [2,3,4,5]
                axs[i,j].plot(feval[o,k,idx], rf_err[o,k,idx],
                              '-o', lw=2, label = integrators[k] )        
            else:
                axs[i,j].plot(feval[o,k], rf_err[o,k], '-o', lw=2, label = integrators[k] )
        axs[i,j].set_xlabel('Function evaluations [-]')
        axs[i,j].set_yscale('log')
        axs[i,j].set_xscale('log')
        axs[i,j].grid(True, which = 'major', ls='--')
        axs[i,j].set_title(titles[o])
        o += 1
axs[0,0].legend() 
axs[0,0].set_ylabel('Position error [m]')
axs[1,0].set_ylabel('Position error [m]')
           
 #%% Propagator Comparison
stepsize, benchmark_stepsize = 75, 8
benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, benchmark_stepsize)
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(start_epoch,
                      stepsize, stepsize, stepsize,  1.0, 1.0)
p = propagation_setup.propagator
propagators = [p.cowell, p.encke]
               # p.gauss_modified_equinoctial
               # p.unified_state_model_quaternions,
               # p.unified_state_model_modified_rodrigues_parameters, p.unified_state_model_exponential_map]
rf_err = np.empty( (6, len(propagators)) )
feval = np.empty( (6, len(propagators)) )

for o, orb in enumerate(np.linspace(0, len(pareto)-1, 6)):
    titles.append('$r_{mean} = $' + str(round(pareto[int(orb),0],1))+' covg. = '+str(round(pareto[int(orb),1],2)) )
    print('orbit '+str(orb))
    for pp, prop in enumerate(propagators):
        print('propagator '+str(pp))
        propagator_settings = propagation_setup.propagator.translational(
                                central_bodies,
                                acceleration_models,
                                bodies_to_propagate,
                                x0_inertial[int(orb)], 
                                termination_settings,
                                prop) 
        if pp == 0:
            dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
                    bodies, benchmark_integrator_settings, propagator_settings)
            states = dynamics_simulator.state_history
            states_list = np.vstack( list( states.values( ) ) )
            rf = states_list[-1,0:3]
        dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
                bodies, integrator_settings, propagator_settings)
        states = dynamics_simulator.state_history
        states_list = np.vstack( list( states.values( ) ) )
        rf_err[o,pp] = np.linalg.norm( rf - states_list[-1,0:3])
        feval[o,pp] = list(dynamics_simulator.get_cumulative_computation_time_history().values())[-1]

                           
#%% Plot 
fig, axs = plt.subplots( 1, figsize = (8,5) )
axs.scatter(feval[:,0], rf_err[:,0], c = 'r', label = 'Cowell')
axs.scatter(feval[:,1], rf_err[:,1], c = 'b', label = 'Encke')
axs.set_ylabel('Position error [m]')
axs.set_xlabel('Computation time [-]')
axs.set_yscale('log')
# axs[j].set_xscale('log')
axs.grid(True, which = 'major', ls='--')
# axs.set_title(propagators[j])
axs.legend()
            
      