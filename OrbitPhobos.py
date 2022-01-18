#%% -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:24:26 2021

@author: Michael

"""

# setup the correct directory containing all files
import os
if os.getcwd() == 'C:\\Users\\Michael':
    os.chdir('PyPhobos')
# if you are using the "generic" version of tudatpy from the website
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.astro import conversion
from tudatpy.kernel.simulation import estimation_setup
# if you are using my own version of tudatpy, which unlocks specific functionalities 
# import sys
# sys.path.append('/home/mplumaris/anaconda3/lib/python3.8/site-packages')
# from kernel.interface import spice_interface
# from kernel.simulation import propagation_setup
# from kernel.astro import conversion
# from kernel.simulation import estimation_setup

import SimulationSetup as ss

import numpy as np
from math import pi, sqrt
from matplotlib import pyplot as plt, colors, rcParams
from scipy.spatial import distance
from scipy.signal import argrelextrema

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid

rcParams.update({'font.size': 14})

#%% Setup Simulation

# Load spice kernels.
spice_interface.load_standard_kernels([])
# spice_interface.load_standard_kernels(['mar097.bsp'])
spice_interface.load_standard_kernels(['mar097.bsp', 'NOE-4-2020.bsp'])

start_epoch = 946774800.0

central_bodies = ["Phobos"]
bodies_to_propagate = ["Vehicle"]

r_Ph, mu_Ph, dmax_Ph =  14.0E3, 7.11E5, 10 #084E5, 4
mu_Ma, r_Ma, dmax_Ma = 0.4282837566395650E+14, 3396e+3, 10
mu_De = 9.46e+4

frame_origin = "Mars"
frame_orientation = "ECLIPJ2000"

bodies = ss.get_bodies( frame_origin, frame_orientation,
                    r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De, 
                    Phobos_spice = True, model = "HM")

acceleration_models = ss.get_vehicle_acceleration_models( bodies, central_bodies,
                                                         dmax_Ma, dmax_Ph )

#%% Grid Search
stepsize = 75
# integrator_settings = propagation_setup.integrator.runge_kutta_4(start_epoch, stepsize)
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(start_epoch,
                      stepsize, stepsize, stepsize,  1.0, 1.0)
termination_settings = ss.get_termination_settings( start_epoch, max_days = 7.1 )
dependent_variables_to_save = [
    propagation_setup.dependent_variable.latitude( "Vehicle", "Phobos" ),
    propagation_setup.dependent_variable.longitude( "Vehicle", "Phobos" ),
    propagation_setup.dependent_variable.heading_angle("Vehicle", "Phobos")

]
moon_rotational_model = bodies.get_body('Phobos').rotation_model

limit = 10*pi/180

pareto = np.empty((0,5), float)
x0_inertial = np.empty((0,6), float)
step = 10
for r0 in np.linspace(20E3, 45E3, step):
    for v in np.linspace(6, 20, step):
        for psi in np.linspace(-pi/6, -pi/2, step):
            initial_cartesian_state_body_fixed = conversion.spherical_to_cartesian(radial_distance = r0,
                                                                            latitude = 0.,
                                                                            longitude = pi,
                                                                            speed = v,
                                                                            flight_path_angle = 0,
                                                                            heading_angle = psi ) #-pi/2 is in z-plane
            initial_state_inertial_coordinates = conversion.transform_to_inertial_orientation(
                initial_cartesian_state_body_fixed,
                start_epoch,
                moon_rotational_model)
 
            propagator_settings = propagation_setup.propagator.translational(
                                                        central_bodies,
                                                        acceleration_models,
                                                        bodies_to_propagate,
                                                        initial_state_inertial_coordinates,
                                                        termination_settings,
                                                        output_variables = dependent_variables_to_save)
        
            dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
            bodies, integrator_settings, propagator_settings)
            
            states = dynamics_simulator.state_history
            time = np.fromiter(states.keys(), dtype=float)
            survival_days = (time[-1] - start_epoch)/86400
            if survival_days > 7:
                dependent_variables = dynamics_simulator.dependent_variable_history
                lat, lon, chi = np.vstack( list( dependent_variables.values( ) ) ).T
                        
                # find orbital period and number of revolutions
                minima = argrelextrema(lon, np.less)
                for revs, m in enumerate(minima[0]):
                    if abs(lat[m]) < limit and abs(chi[m]-chi[0]) < limit:
                        period = time[m]
                        break
                revs += 1
                
                # in case period not found, discard solution
                if revs == len(minima[0]):
                    continue
                
                # compute mean radius
                r_mean = np.mean( np.linalg.norm( np.vstack( list( states.values( ) ) )[0:m], axis = 1))
                
                # compute coverage metric
                lon = (lon[0:m] + pi)/(2*pi)
                lat = (lat[0:m] + pi/2)/pi
                x = np.vstack((lon,lat)).transpose()
                dist = np.triu(distance.cdist(x, x, 'euclidean'))
                coverage = np.mean(dist[dist != 0])
                
                # compute stability index from eigenvalues
                propagator_settings = propagation_setup.propagator.translational(
                                                        central_bodies,
                                                        acceleration_models,
                                                        bodies_to_propagate,
                                                        initial_state_inertial_coordinates,
                                                        period)    
                parameter_settings = estimation_setup.parameter.initial_states( propagator_settings, bodies )
                variational_equations_solver = estimation_setup.SingleArcVariationalEquationsSolver(
                    bodies, integrator_settings, propagator_settings, estimation_setup.create_parameters_to_estimate( parameter_settings, bodies ),
                    integrate_on_creation = 1 )
                state_transition_matrices = variational_equations_solver.state_transition_matrix_history   
                eigenvalues = np.linalg.eig(state_transition_matrices[period])[0]
                length = 0
                for i in range(6):
                    if np.linalg.norm(eigenvalues[i]) > length:
                        length = np.linalg.norm(eigenvalues[i])
                length = length**(1/revs)
                            
                x0_inertial = np.append(x0_inertial, initial_state_inertial_coordinates.reshape((1,6)), axis=0)
                pareto = np.append(pareto, np.array([[r_mean*1e-3, coverage, length, period - start_epoch, revs ]]), axis=0)
                
print(len(pareto))
#%%
# np.save('x0_inertial.npy', x0_inertial)
# np.save('pareto.npy', pareto)
# np.save('pareto_all.npy', pareto)

#%% Open solution which survive >7 days

# x0_inertial_all = np.load('x0_inertial_all.npy')
# pareto_all = np.load('pareto_all.npy')

#%% Select Pareto Front solutions

# idxs = ss.is_pareto_efficient(np.array([pareto[:,0], 1/pareto[:,1], pareto[:,2]]).transpose(), True)
# x0_inertial = x0_inertial[idxs]
# pareto = pareto[idxs]

#%% Save PF solutions

# np.save('x0_inertial.npy', x0_inertial)
# np.save('pareto.npy', pareto)

#%% Open PF solutions

x0_inertial = np.load('x0_inertial.npy')
pareto = np.load('pareto.npy')
pareto_all = np.load('pareto_all.npy')

#%% Plot All and PF solutions
fig = plt.figure(figsize=(10, 6))
grid = ImageGrid(fig, 111, nrows_ncols=(1,2),
                 axes_pad=0.2, share_all=False,
                 cbar_location="right",
                 cbar_mode="single",
                label_mode = 'all',
                cbar_pad=0.1,
                aspect = False)

cmap = plt.cm.inferno

for i, ax in enumerate(grid.axes_all):
    if i == 0:
        im = ax.scatter(x = pareto_all[:,0], y = pareto_all[:,1],
            c = pareto_all[:,2], cmap=cmap, norm = colors.LogNorm())
    if i == 1:  
        im = ax.scatter(x = pareto[:,0], y = pareto[:,1],
                        c = pareto[:,2], cmap=cmap, norm = colors.LogNorm(),
                        vmin = np.min(pareto_all[:,2]), vmax = np.max(pareto_all[:,2]))
        axins = zoomed_inset_axes(ax, 4, loc='lower center')
        axins.scatter(x = pareto[:,0], y = pareto[:,1], c = pareto[:,2],
                      cmap=cmap, norm = colors.LogNorm(),
                      vmin = np.min(pareto_all[:,2]), vmax = np.max(pareto_all[:,2]))
        axins.set_xlim(20.2, 22.5)
        axins.set_ylim(0.317, 0.324)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        cb = fig.colorbar(im, cax=ax.cax)
        cb.set_label('Stability Index [-]')
    if i == 0:
        ax.set_ylabel('$\overline{euc}$'+' [-]')
    # ax.set_yticklabels([])
    ax.set_xlabel('$\overline{r}$'+' [km]')
    ax.set_xticks([20,30,40,50])
    ax.set_xticklabels([str(t) for t in [20,30,40,50]])
                  
#%% Test and Visualise

# which orbit number?
orb = 70


bodies = ss.get_bodies( frame_origin, frame_orientation,
                    r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De,
                    Phobos_spice = True, model = "HM")

central_bodies = ['Phobos']
bodies_to_propagate = ['Vehicle']
stepsize = 75
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(start_epoch,
                      stepsize, stepsize, stepsize,  1.0, 1.0)

termination_settings = ss.get_termination_settings( start_epoch, max_days = 3.5 )
vehicle_acceleration_models = ss.get_vehicle_acceleration_models(bodies,central_bodies ,
                                         dmax_Ma = 10, dmax_Ph = 4)

dependent_variables_to_save = [
            propagation_setup.dependent_variable.central_body_fixed_cartesian_position("Vehicle", "Phobos"),
            # propagation_setup.dependent_variable.latitude("Vehicle", "Phobos"),
            # propagation_setup.dependent_variable.longitude("Vehicle", "Phobos"),

            ]

propagator_settings = propagation_setup.propagator.translational(
                            central_bodies,
                            vehicle_acceleration_models,
                            bodies_to_propagate,
                            x0_inertial[orb],
                            termination_settings,
                            output_variables = dependent_variables_to_save)

dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
bodies, integrator_settings, propagator_settings)

# states = dynamics_simulator.state_history
# states_list = np.vstack( list( states.values( ) ) )
time = list( dynamics_simulator.dependent_variable_history.keys( ) )
print( (time[-1]-start_epoch)/86400 )
dvl = np.vstack( list( dynamics_simulator.dependent_variable_history.values( ) ) )
dvl[:,0:3] = dvl[:,0:3]*1e-3
r = np.linalg.norm(dvl[:,0:3], axis = 1)
maxx = np.max(r)
# 3D
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
(xs, ys, zs) = ss.drawSphere(0, 0, 0, 14)
ax.plot_wireframe(xs, ys, zs, color="grey")
ax.plot(dvl[:, 0], dvl[:, 1], dvl[:, 2], 'magenta')
ax.set_xlabel('x, km')
ax.set_ylabel('y, km')
ax.set_zlabel('z, km')
ss.set_lims(ax, -maxx, maxx, z = True)  
ax.set_title(str(round(np.max(dvl[:, 0]), 0))+'x'+str(round(np.max(dvl[:, 1]), 0))
                                                    +'x'+str(round(np.max(dvl[:, 2]), 0)) + ' (6:7)')
ax1 = fig.add_subplot(1, 2, 2)
ax1.plot(dvl[:,0], dvl[:,1], 'magenta')
ax1.grid(True)
circle1 = plt.Circle((0, 0), 14, color='grey')
ax1.add_artist(circle1)
ss.set_lims(ax1,-maxx,maxx)
ax1.set_xlabel('x, km')
# ax1.set_ylabel('y, km')

# 2D
# fig, ((ax1, ax2)) = plt.subplots( 1, 2, figsize = (13,6) )
# ax1.plot(dvl[:,0], dvl[:,1])
# ax1.grid(True)
# circle1 = plt.Circle((0, 0), 14, color='grey')
# ax1.add_artist(circle1)
# ss.set_lims(ax1,-maxx,maxx)
# ax2.plot(dvl[:,1], dvl[:,2])
# circle2 = plt.Circle((0., 0.), 14, color='grey')
# ax2.add_artist(circle2)
# ax2.grid(True)
# ss.set_lims(ax2,-maxx,maxx)

# np.save('bfc_low_corr_orb.npy', dvl)

# np.savetxt('Geoffrey_orbit.txt', dvl)

#%% Plot PF and 3 samples from across

# choose which orbit numbers
orbs = np.array([3,51,101])


termination_settings = ss.get_termination_settings( start_epoch, max_days = 3.5 )
dependent_variables_to_save = [ 
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("Vehicle", "Phobos")
         ]
stepsize = 75
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(start_epoch,
                      stepsize, stepsize, stepsize,  1.0, 1.0)
fig = plt.figure( figsize = (20, 5) )
cmap = plt.cm.plasma

# for some strange reason this command im.get_facecolors()[orbs][:,0:3]
# cannot be run before plotting, so we have to save them here manually (to reflect PF) obj.
cols = np.array([[0.660374, 0.134144, 0.588971],
       [0.21435 , 0.016973, 0.599239],
       [0.050383, 0.029803, 0.527975]])

count = 0
ticks = [1, 1, 2, 3, 5, 10]
for row in range(2):
    for col in range(2):
        if row == 0 and col == 0:
            ax = fig.add_subplot(1, 4, 1)
            im = ax.scatter(x = pareto[:,0], y = pareto[:,1],
                        c = pareto[:,2], cmap=cmap, norm = colors.LogNorm())
            ax.set_xlabel('$\overline{r} [km] $')
            ax.set_ylabel('$\overline{euc}$')
            cb = fig.colorbar(im)
            # cb.set_label('$\lambda_n$')
            cb.set_ticks(ticks)
            cb.set_ticklabels([str(t) for t in ticks])
            axins = zoomed_inset_axes(ax, 2, loc=8)
            axins.scatter(x = pareto[:,0], y = pareto[:,1], c = pareto[:,2],
                          cmap=cmap, norm = colors.LogNorm())
            axins.set_xlim(20.2, 25)
            axins.set_ylim(0.32, 0.34)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            ax.set_title('Colour = '+'$\lambda_n$')
            # # get this in console! 
            # # im.get_facecolors()[orbs][:,0:3]
            
        else:
            ax = fig.add_subplot(1, 4, (count + 2), projection='3d')
            x0 = x0_inertial[orbs[count]]
            propagator_settings = propagation_setup.propagator.translational(
                                        central_bodies,
                                        acceleration_models,
                                        bodies_to_propagate,
                                        x0,
                                        termination_settings,
                                        output_variables = dependent_variables_to_save)
    
            dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
            bodies, integrator_settings, propagator_settings)
    
            states = dynamics_simulator.state_history
            states_list = np.vstack( list( states.values( ) ) )
            r = np.sqrt(states_list[:,0]**2 +states_list[:,1]**2+states_list[:,2]**2 )
            r_mean = np.mean(r)
            time = np.fromiter(states.keys(), dtype=float)
    
            dependent_variables = dynamics_simulator.dependent_variable_history
            dvl = np.vstack( list( dependent_variables.values( ) ) )*1e-3
            maxx = np.max(r_mean)*1e-3
    
            (xs, ys, zs) = ss.drawSphere(0, 0, 0, 14)
            ax.plot_wireframe(xs, ys, zs, color="grey")
    
            ax.plot(dvl[:, 0], dvl[:, 1], dvl[:, 2], color=cols[count])
            ax.set_xlabel('km')
            ax.set_ylabel('km')
            ax.set_zlabel('km')
            ss.set_lims(ax, -maxx, maxx, z = True)
            ax.set_title('$\overline{r} = $'+str(int(pareto[orbs[count], 0]))+'km, '
                         +'$\overline{euc} = $'+
                         str(round(pareto[orbs[count], 1],2)) +
                         ', '+'$\lambda_n = $' + str(round(pareto[orbs[count], 2],1)))
            count += 1
fig.tight_layout()

#%% Predictor Corrector 

termination_settings = ss.get_termination_settings( start_epoch, max_days = 1 )

integrator_settings = propagation_setup.integrator.runge_kutta_4( start_epoch, 100.)

dependent_variables_to_save = [
            propagation_setup.dependent_variable.central_body_fixed_cartesian_position( "Vehicle", "Phobos"),
            propagation_setup.dependent_variable.body_fixed_groundspeed_velocity( "Vehicle", "Phobos"),
            # propagation_setup.dependent_variable.rotation_matrix_to_body_fixed_frame("Phobos")
            propagation_setup.dependent_variable.longitude("Vehicle", "Phobos")
] 
moon_rotational_model = bodies.get_body('Phobos').rotation_model

for c, x0 in enumerate(x0_inertial[20:21]):
    print(x0)
    for i in range(1):
        propagator_settings = propagation_setup.propagator.translational(
                            central_bodies,
                            acceleration_models,
                            bodies_to_propagate,
                            x0, 
                            termination_settings,
                            output_variables = dependent_variables_to_save) 
        parameter_settings = estimation_setup.parameter.initial_states( propagator_settings, bodies )
        variational_equations_solver = estimation_setup.SingleArcVariationalEquationsSolver(
            bodies, integrator_settings, propagator_settings, estimation_setup.create_parameters_to_estimate( parameter_settings, bodies ),
            integrate_on_creation = 1 )
        dependent_variables = variational_equations_solver.get_dynamics_simulator().dependent_variable_history
        dependent_variable_list = np.vstack( list( dependent_variables.values( ) ) )
        time = list(dependent_variables.keys())

        x0_bf = dependent_variable_list[0,0:6]
        rdiff, best_diff, t = 10e3, np.zeros((6)), time[-1]
        switch = False
        for c, lat in enumerate( dependent_variable_list[:,6] ):
            if lat <= -3.1 and switch == False and c > 1:
                switch = True
            if switch == True:
                diff = dependent_variable_list[c, 0:6] - x0_bf
                if np.linalg.norm(diff[0:3]) < rdiff:
                    rdiff = np.linalg.norm(diff[0:3])
                    best_diff = diff
                    t = time[c]
            if lat >= 0.1 and switch == True:
                switch = False
                phi = (variational_equations_solver.state_transition_matrix_history)[t]
                dx = np.matmul( np.linalg.inv( np.eye(6) - phi ), best_diff )
                x0 = conversion.transform_to_inertial_orientation( 
                    x0_bf + dx, t, moon_rotational_model)
                print(x0)
                break


