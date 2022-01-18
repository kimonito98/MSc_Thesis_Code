# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:27:32 2021

@author: Michael

Investigate the effect of different environemnt models for the Vehicle, and for Phobos

"""

import os
if os.getcwd() == 'C:\\Users\\Michael':
    os.chdir('PyPhobos')
    
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.astro import conversion
from tudatpy.kernel.simulation import estimation_setup
from tudatpy.kernel.simulation import environment_setup


# from kernel.interface import spice_interface
# from kernel.simulation import propagation_setup
# from kernel.astro import conversion
# from kernel.simulation import estimation_setup
# from kernel.simulation import environment_setup

import SimulationSetup as ss
import numpy as np
from math import pi, sqrt
from matplotlib import pyplot as plt, colors

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

termination_settings = ss.get_termination_settings( start_epoch, max_days = 1 )

acceleration_models = ss.get_vehicle_acceleration_models( bodies, central_bodies,
                                                         dmax_Ma, dmax_Ph )

x0_inertial = np.load('x0_inertial.npy')
pareto = np.load('pareto.npy')


#%% Plot all accelerations to discard all but Mars & Phobos gravity 

bodies = ss.get_bodies( frame_origin, frame_orientation,
                    r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De, 
                    Phobos_spice = True, model = "HM")
# function must include all accelerations!
acceleration_models = ss.get_vehicle_acceleration_models( bodies, central_bodies, dmax_Ma=10, dmax_Ph=2 )
multi_arc_stepsize = 75
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
            start_epoch, multi_arc_stepsize, multi_arc_stepsize, multi_arc_stepsize,  1.0, 1.0)
termination_settings = ss.get_termination_settings( start_epoch, max_days = 1 )

orb = 90
dependent_variables_to_save = [
            propagation_setup.dependent_variable.central_body_fixed_cartesian_position( "Vehicle", "Phobos"),
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.spherical_harmonic_gravity_type, "Vehicle", "Mars" ), 
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.spherical_harmonic_gravity_type, "Vehicle", "Phobos" ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.cannonball_radiation_pressure_type, "Vehicle", "Sun"),
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Vehicle", "Sun" ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Vehicle", "Jupiter" ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Vehicle", "Deimos" ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Vehicle", "Earth" ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.empirical_acceleration_type, "Vehicle", "Phobos" ),            
            ] 

propagator_settings = propagation_setup.propagator.translational(
                            central_bodies,
                            acceleration_models,
                            bodies_to_propagate,
                            x0_inertial[orb], 
                            termination_settings,
                            output_variables = dependent_variables_to_save ) 


dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
bodies, integrator_settings, propagator_settings)

dependent_variables = dynamics_simulator.dependent_variable_history
# dependent_variable_list = np.vstack( list( dependent_variables.values( ) ) )
dependent_variable_list_far = np.vstack( list( dependent_variables.values( ) ) )

time = (np.fromiter(dependent_variables.keys(), dtype=float) - start_epoch)/3600
fig, ax = plt.subplots( 1, 1, figsize = (10,6) )
labels = ('Mars SHG', 'Phobos SHG', 'Sun SRP', 'Sun PMG', 'Jupiter PMG' , 'Deimos PMG', 'Earth PMG', 'Empirical')

cols = ['#1f77b4', '#ff7f0e']
for i in range(2):
    ax.plot(time,dependent_variable_list[:,i+3], label = 'Low, '+ labels[i] )
    ax.plot(time,dependent_variable_list_far[:,i+3], ls = 'dashed', c = cols[i], label = 'High, '+ labels[i] )
    
for i in range(2,8):
    ax.plot(time,dependent_variable_list[:,i+3], label =  labels[i] )
    

ax.tick_params(axis='both', which='major',)
ax.set_yscale('log')
ax.grid(b=True, which='both', color='gray', ls = 'dashdot', alpha = 0.5)
ax.set_xlabel('time [h]')
ax.set_ylabel('Vehicle Acceleration$ [m/s^2]$')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(loc='lower right', bbox_to_anchor=(1.5,0))
#%% Shperical Harmonics Settings
import time
termination_settings = ss.get_termination_settings( start_epoch, max_days = 7 )

rf_err = np.empty( (6,5,5) )
feval = np.empty( (6,5,5) )
labels, titles = [], []
for o, orb in enumerate(np.linspace(0, len(pareto)-1, 6)):
    titles.append('$r_{mean} = $' + str(round(pareto[int(orb),0],1))+' covg. = '+str(round(pareto[int(orb),1],2)) )
    acceleration_models = ss.get_vehicle_acceleration_models( bodies, central_bodies, dmax_Ma = 12, dmax_Ph = 10 )

    propagator_settings = propagation_setup.propagator.translational(
                                central_bodies,
                                acceleration_models,
                                bodies_to_propagate,
                                x0_inertial[int(orb)], 
                                termination_settings) 
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
    bodies, integrator_settings, propagator_settings)
    states = dynamics_simulator.state_history
    states_list = np.vstack( list( states.values( ) ) )
    rf = np.linalg.norm(states_list[-1,0:3])
    for sm, dmax_Ma in enumerate(range(10,1,-2)):
        for sp, dmax_Ph in enumerate(range(10,1,-2)):
            acceleration_models = ss.get_vehicle_acceleration_models( bodies, central_bodies, dmax_Ma, dmax_Ph)
            propagator_settings = propagation_setup.propagator.translational(
                                        central_bodies,
                                        acceleration_models,
                                        bodies_to_propagate,
                                        x0_inertial[int(orb)], 
                                        termination_settings) 
            start_time = time.time()
            dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
            bodies, integrator_settings, propagator_settings)
            states = dynamics_simulator.state_history
            states_list = np.vstack( list( states.values( ) ) )
            rf_err[o,sp,sm] = abs(np.linalg.norm(states_list[-1,0:3])-rf)
            feval[o,sp,sm] = (time.time() - start_time) #list(dynamics_simulator.get_cumulative_computation_time_history().values())[-1]
            if o == 0:
                labels.append('M' + str(dmax_Ma) + ', P' + str(dmax_Ph))

#%%
fig, axs = plt.subplots( 2, 3, figsize = (16, 10) )
marks = ('o', 'D', 'v')
o = 0
for i in range(2):
    for j in range(3):
        l = 0
        for sm in range(5):
            for sp in range(5):
                axs[i,j].scatter(feval[o,sm,sp],rf_err[o,sm,sp], label = labels[l], marker = marks[ int(l/10)] )
                l += 1
        if i > 0:
            axs[i,j].set_xlabel('time [s]')
        axs[i,j].set_yscale('log')
        axs[i,j].grid(True, which = 'major', ls='--')
        axs[i,j].set_title(titles[o])
        o += 1
axs[0,0].set_ylabel('Position error \n wrt Benchmark [m]')   
axs[1,0].set_ylabel('Position error \n wrt Benchmark [m]')   
axs[1,1].legend(loc="center", bbox_to_anchor=(-0.1, -0.5), ncol=5)

fig.subplots_adjust(bottom=0.25)


#%% Inspect difference between propagated and tabulated Phobos ephemeris

def get_bodies(frame_origin,
               frame_orientation,
               r_Ph, mu_Ph,
               mu_Ma, r_Ma,
               mu_De,
               Phobos_spice = True,
               model = "HM"):

    bodies_to_create = ["Mars", "Phobos", "Sun", "Earth", "Deimos", "Jupiter"]

    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, frame_origin, frame_orientation)
    
    
    body_settings.get( "Mars" ).rotation_model_settings = environment_setup.rotation_model.spice(
        frame_orientation, "IAU_Mars")
    
    if Phobos_spice:
        body_settings.get( "Phobos" ).ephemeris_settings = environment_setup.ephemeris.direct_spice(
            frame_origin, frame_orientation)        
    else:
        body_settings.get( "Phobos" ).ephemeris_settings = environment_setup.ephemeris.tabulated(
            dict(), frame_origin, frame_orientation)
        
    # x is positive towards Mars, z is in direction of H, y completes triad
    # body_settings.get( "Phobos" ).rotation_model_settings = environment_setup.rotation_model.spice(
    #     frame_orientation, "IAU_Phobos")
    body_settings.get( "Phobos" ).rotation_model_settings = environment_setup.rotation_model.synchronous(
        "Mars", frame_orientation, target_frame = "IAU_Phobos")  
    # Le Maistre 2019
    cos_Ph, sin_Ph = ss.read_Phobos_gravity_field(model)[0:2]
    # Jacobson and Lainey 2014
    cos_Ph = np.array([[ 1, 0, 0], [0, 0, 0], [0.04727047704434555, 0, 0.0229]])
    sin_Ph = np.zeros((3,3))
    body_settings.get( "Phobos" ).gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
        mu_Ph, r_Ph, cos_Ph, sin_Ph, "IAU_Phobos" )
    
    
    body_settings.get( "Deimos" ).ephemeris_settings = environment_setup.ephemeris.direct_spice(
        frame_origin, frame_orientation)
    body_settings.get( "Deimos" ).gravity_field_settings = environment_setup.gravity_field.central(
        mu_De )
    
    body_settings.add_settings("Vehicle")
    body_settings.get("Vehicle").constant_mass = 400
    body_settings.get("Vehicle").ephemeris_settings = environment_setup.ephemeris.tabulated(
        dict(), frame_origin, frame_orientation)
    body_settings.get("Vehicle").ephemeris_settings.reset_make_multi_arc_ephemeris(True)

    bodies = environment_setup.create_system_of_bodies(body_settings)

    reference_area_radiation = 4.0
    radiation_pressure_coefficient = 1.2
    occulting_bodies = [ "Mars" ]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
    )
    environment_setup.add_radiation_pressure_interface(
                bodies, 'Vehicle' , radiation_pressure_settings )

    return bodies


def get_Phobos_acceleration_models(bodies,
                             central_bodies,
                             dmax_Ma, 
                             dmax_Ph):

    bodies_to_propagate = ["Phobos"]

    accelerations_settings_Phobos = dict(
        Mars=[
            propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
                dmax_Ma, dmax_Ma, dmax_Ph, dmax_Ph )
        ],
        # Sun=[
        #     propagation_setup.acceleration.point_mass_gravity()
        # ],
        Deimos=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Jupiter=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Earth=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
    )
    acceleration_settings = {"Phobos": accelerations_settings_Phobos}

    return propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

#%% Tabulated

bodies = get_bodies( frame_origin, frame_orientation,
                    r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De, 
                    Phobos_spice = True, model = "HM")
multi_arc_stepsize = 75
acceleration_models = ss.get_vehicle_acceleration_models( bodies, central_bodies, dmax_Ma=10, dmax_Ph=2 )
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
            start_epoch, multi_arc_stepsize, multi_arc_stepsize, multi_arc_stepsize,  1.0, 1.0)
central_bodies = ["Phobos"]
bodies_to_propagate = ["Vehicle"]

orb = 1
dependent_variables_to_save = [
            propagation_setup.dependent_variable.keplerian_state( "Phobos", "Mars"),
            propagation_setup.dependent_variable.central_body_fixed_spherical_position( "Phobos", "Mars"),
            # propagation_setup.dependent_variable.rotation_matrix_to_body_fixed_frame("Mars"),
            propagation_setup.dependent_variable.central_body_fixed_cartesian_position( "Phobos", "Mars")
            ] 

propagator_settings = propagation_setup.propagator.translational(
                            central_bodies,
                            acceleration_models,
                            bodies_to_propagate,
                            x0_inertial[orb], 
                            start_epoch + 7 * 86400,
                            output_variables = dependent_variables_to_save ) 

dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
bodies, integrator_settings, propagator_settings)

dependent_variables = dynamics_simulator.dependent_variable_history
dvl_t = np.vstack( list( dependent_variables.values( ) ) )
time = (np.fromiter(dependent_variables.keys(), dtype=float) - start_epoch)/86400

#%% Propagated

central_bodies = ["Mars"]
bodies_to_propagate = ["Phobos"]

dependent_variables_to_save = [
    propagation_setup.dependent_variable.keplerian_state( "Phobos", "Mars"),
    propagation_setup.dependent_variable.central_body_fixed_spherical_position( "Phobos", "Mars"),
    # propagation_setup.dependent_variable.rotation_matrix_to_body_fixed_frame("Mars"),
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("Phobos", "Mars")
                              ]

# Phobos_initial_state = propagation_setup.get_initial_state_of_bodies(
#         bodies_to_propagate, central_bodies, bodies, start_epoch)
Phobos_initial_state = np.array([-5.61563403e+06, -6.99812625e+06,  2.54741607e+06,  1.38189778e+03,
        -1.43974592e+03, -7.96603304e+02])
        
bodies = ss.get_bodies( frame_origin, frame_orientation,
                    r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De,
                    Phobos_spice = False, model = "HM")

acceleration_models = ss.get_Phobos_acceleration_models( bodies, central_bodies,
                                                        dmax_Ma = 10, dmax_Ph = 2 )
multi_arc_stepsize = 75
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
            start_epoch, multi_arc_stepsize, multi_arc_stepsize, multi_arc_stepsize,  1.0, 1.0)
propagator_settings = propagation_setup.propagator.translational(
                            central_bodies,
                            acceleration_models,
                            bodies_to_propagate,
                            Phobos_initial_state, 
                            start_epoch + 7 * 86400,
                            output_variables = dependent_variables_to_save ) 

dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
bodies, integrator_settings, propagator_settings)

dependent_variables = dynamics_simulator.dependent_variable_history
dvl_p = np.vstack( list( dependent_variables.values( ) ) )
time_propagated_Phobos = np.fromiter(dependent_variables.keys(), dtype=float) # - start_epoch)/86400

 

#%% SPherical Elements

fig, ax1 = plt.subplots( 1, 1, figsize = (10,8) )

r_diff = dvl_t[:,9:12] - dvl_p[:,9:12]


ax1.plot(time, r_diff[:,0], label = 'x')
ax1.plot(time, r_diff[:,1], label = 'y')
ax1.plot(time, r_diff[:,2], label = 'z')

ax1.grid(b=True, which='both', color='gray', ls = 'dashdot', alpha = 0.5)
ax1.set_ylabel('Difference in BFRF [km]')
ax1.legend()

# ax1.set_ylabel('(Lainey2021) vs. propagated Phobos')

fig.tight_layout()

#%% Body Fixed Elements

fig, ((ax1, ax2)) = plt.subplots( 2, 1, figsize = (10,8) )

r_diff = dvl_t[:,6:9] - dvl_p[:,6:9]


ax1.plot(time, r_diff[:,0]*1e-3, c = 'k')
ax1.grid(b=True, which='both', color='gray', ls = 'dashdot', alpha = 0.5)
ax1.set_ylabel('Radius difference [km]')
# ax1.legend()
ax1.set_xticklabels([])

# color2 = 'darkorange'
# ax2 = ax1.twinx()
ax2.plot( time, np.rad2deg(r_diff[:,1]), label ='Latitude')
ax2.plot( time, np.rad2deg(r_diff[:,2]), label ='Longitude' )
ax2.legend()

ax2.tick_params(axis='y')
ax2.grid(b=True, which='both', ls = 'dashdot', alpha = 0.3)
# ax2.spines['right'].set_color(color2)
ax2.set_ylabel('Angle difference [deg]')
ax2.set_xlabel( 'time [days]' )

# ax1.set_ylabel('(Lainey2021) vs. propagated Phobos')

fig.tight_layout()

#%% Kepler Elements
# %matplotlib inline 
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots( 3, 2, figsize = (10,10) )
time_hours = time
kepler_elements =  dvl_p[:,0:6] - dvl_t[:,0:6] # -

ax1.set_xlabel( 'time [days]' )

semi_major_axis = [ element/1000 for element in kepler_elements[:,0] ]
ax1.plot( time_hours, semi_major_axis )
ax1.set_ylabel( 'Semi-major axis [km]' )

# Eccentricity
eccentricity = kepler_elements[:,1]
ax2.plot( time_hours, eccentricity )
ax2.set_ylabel( 'Eccentricity [-]' )

# Inclination
inclination = [ np.rad2deg( element ) for element in kepler_elements[:,2] ]
ax3.plot( time_hours, inclination )
ax3.set_ylabel( 'Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = [ np.rad2deg( element ) for element in kepler_elements[:,3] ]
ax4.plot( time_hours, argument_of_periapsis )
ax4.set_ylabel( 'Argument of Periapsis [deg]' )

# Right Ascension of the Ascending Node
raan = [ np.rad2deg( element ) for element in kepler_elements[:,4] ]
ax5.plot( time_hours, raan )
ax5.set_ylabel( 'RAAN [deg]' )
ax6.set_ylabel( 'time [days]' )

# True Anomaly
true_anomaly = [ np.rad2deg( element ) for element in kepler_elements[:,5] ]
ax6.scatter( time_hours, true_anomaly, s=1 )
ax6.set_ylabel( 'True Anomaly [deg]' )
ax6.set_ylabel( 'time [days]' )
ax6.set_yticks(np.arange(0, 361, step=60))

fig.suptitle('Difference in Kplerian Elements')


