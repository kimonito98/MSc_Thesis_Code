# -*- coding: utf-8 -*-
"""
Created on Thu Nov 4 11:01:32 2021

@author: Michael

Validate the RS experiment with:

The OSIRIS-REx Radio Science Experiment at Bennu (McMahon, 2018)

"""

import os
# if os.getcwd() == 'C:\\Users\\Michael':
os.chdir('/mnt/c/Users/Michael/PyPhobos/Bennu_Validation')

import sys
sys.path.append('/home/mplumaris/anaconda3/lib/python3.8/site-packages')

from kernel import constants

from kernel.interface import spice_interface
from kernel.simulation import propagation_setup
from kernel.simulation import estimation_setup
from kernel.simulation import environment_setup
from kernel.simulation.estimation_setup import observation_setup
from kernel.astro import conversion, frames

import RadioScienceSetup as rss
import SimulationSetup as ss
import numpy as np

#%% Setup Environment

spice_interface.load_standard_kernels([])
spice_interface.load_standard_kernels(['de424.bsp','bennu_refdrmc_v1.bsp',"bennu_v17.tpc"])

initial_state_epoch = spice_interface.convert_date_string_to_ephemeris_time('6 MAR 2019')

frame_origin = "Earth"
frame_orientation = "ECLIPJ2000"

bodies_to_create = ["Earth", "Sun"]

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, frame_origin, frame_orientation)
bodies = environment_setup.create_system_of_bodies(body_settings)

body_settings.get("Bennu").rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
    frame_orientation, "IAU_Bennu", "IAU_Bennu", initial_state_epoch )
# body_settings.get("Bennu").rotation_model_settings = environment_setup.rotation_model.spice(
#         frame_orientation, "IAU_Bennu")
cos, sin, cfs = ss.read_Mars_gravity_field("Bennu_gravity_field_McMahon2018.txt",
                                           6, std = False, delimiter = None)
body_settings.get( "Bennu" ).gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
        GM_Bennu, R_Bennu, cos, sin, "IAU_Bennu" )

bodies = environment_setup.create_system_of_bodies(body_settings)


bodies.create_empty_body( "Vehicle" )
bodies.get_body( "Vehicle" ).set_constant_mass(100.0)

reference_area_radiation = 1
radiation_pressure_coefficient = 1.28
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, []
)
environment_setup.add_radiation_pressure_interface(
            bodies, "Vehicle", radiation_pressure_settings )
environment_setup.add_empty_tabulate_ephemeris( bodies, "Vehicle" )

n_lnd = 100
rss.add_ground_stations(bodies, n_lnd, "Bennu")

#%% Vehicle Accelerations

central_bodies = ["Bennu"]
bodies_to_propagate = ["Vehicle"]
acceleration_settings = dict(
    Bennu=[
        propagation_setup.acceleration.spherical_harmonic_gravity(6, 6)
    ],
    Sun=[
        propagation_setup.acceleration.cannonball_radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
)
acceleration_settings = {"Vehicle": acceleration_settings}
vehicle_acceleration_models =  propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

#%% Verify Bennu Keplerian Orbit

GM_Sun = constants.SUN_GRAVITATIONAL_PARAMETER

Bennu_keplerian = conversion.cartesian_to_keplerian(
    spice_interface.get_body_cartesian_state_at_epoch('Bennu', 'Sun', frame_orientation,
                                                      'none', initial_state_epoch), GM_Sun
)
# McMahon Values
# Semimajor_axis  = 1.1259*constants.astronomical_unit
# Eccentricity  = 0.20372
# Inclination  = np.deg2rad(6.034)
# RAAN = np.deg2rad(2.017)
# Argument_of_periapse = np.deg2rad(66.304)
# True_anomaly = np.deg2rad(64.541)
# GM_Sun = 132712440040.944600e9

#%% Setup Estimated Parameters
def create_estimated_parameters( propagator_settings,
                                 bodies,
                                 cos_tup, sin_tup,
                                 n_lnd):

    parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameter_settings.append(
        estimation_setup.parameter.radiation_pressure_coefficient("Vehicle"))
    parameter_settings.append(
        estimation_setup.parameter.gravitational_parameter("Bennu"))
    parameter_settings.append(
        estimation_setup.parameter.constant_rotation_rate("Bennu"))
    parameter_settings.append(
        estimation_setup.parameter.rotation_pole_position("Bennu"))
    parameter_settings.append(
        estimation_setup.parameter.spherical_harmonics_c_coefficients("Bennu", cos_tup[0], cos_tup[1], cos_tup[2], cos_tup[3]))
    parameter_settings.append(
        estimation_setup.parameter.spherical_harmonics_s_coefficients("Bennu", sin_tup[0], sin_tup[1], sin_tup[2], sin_tup[3]))
    for i in range(n_lnd):
        parameter_settings.append(
            estimation_setup.parameter.ground_station_position("Bennu", "Landmark"+str(i+1) ) )
    return estimation_setup.create_parameters_to_estimate( parameter_settings, bodies,
                                                           propagator_settings )

#%% Define A-Priori Values
cos_tup, sin_tup = (1,0,5,5), (1,1,5,5)
n_cfs = ss.count_cfs(cos_tup, sin_tup)
# apr_c = cos.flatten()[cos.flatten() != 0][0:n_cfs[0]]
# apr_s = sin.flatten()[sin.flatten() != 0][0:n_cfs[1]]

c6, c7, c8 = 0.084/6**2.08, 0.084/7**2.08, 0.084/8**2.08
rms6, rms7, rms8 = 0.026/6**2.01, 0.026/7**2.01, 0.026/8**2.01
kaula_c = np.concatenate((np.array([c6]), np.ones(6)*rms6,
                        np.array([c7]), np.ones(7)*rms7,
                        np.array([c8]), np.ones(8)*rms7))
kaula_s = np.concatenate((np.ones(6)*rms6,
                        np.ones(7)*rms7,
                        np.ones(8)*rms8))
apr_c = np.concatenate((cos.flatten()[cos.flatten() != 0][0:n_cfs[0]], kaula_c))
apr_s = np.concatenate((sin.flatten()[sin.flatten() != 0][0:n_cfs[1]], kaula_s))
cos_tup, sin_tup = (1,0,8,8), (1,1,8,8)
n_cfs = ss.count_cfs(cos_tup, sin_tup)

#%% Radio-Science Estimation Parameters

buffer = 3600.
n_GS = 3 # to use out of 3 total (DSN)
range_, vlbi, optical = False, False, True
t_doppler, doppler_noise = 60., 3.33e-13
t_range, range_noise, range_bias = 300, 15, 15
t_vlbi, vlbi_noise, vlbi_bias = 300, 1e-9, 1e-9
t_optical, optical_noise, optical_bias = 200*60, 800e-6, 800e-6

total_duration = 9*86400
t_final = initial_state_epoch + total_duration
percentage_arc_tracked = 1
single_arc_stepsize = 30

names = ['r_sc', 'v_sc', 'Cr', 'GM', 'omega', 'pole', 'cfs', 'lnd']
apr_vals = [1e3, 1, 0.1, 1, np.deg2rad(1.4e-4), np.array([np.deg2rad(1),
                        np.deg2rad(1)]), np.concatenate((apr_c, apr_s)), 0.8]
counts = [3, 3, 1, 1, 1, 2, n_cfs[0]+n_cfs[1], 3*n_lnd]

integrator_settings= propagation_setup.integrator.adams_bashforth_moulton(
    initial_state_epoch, single_arc_stepsize, single_arc_stepsize, single_arc_stepsize, 1.0, 1.0)
vehicle_initial_state = conversion.keplerian_to_cartesian(
        gravitational_parameter=GM_Bennu,
        semi_major_axis=1e3,
        eccentricity=0.0,
        inclination=np.deg2rad(90),
        argument_of_periapsis= 0 ,
        longitude_of_ascending_node=np.deg2rad(-90),
        true_anomaly=0 )

propagator_settings = propagation_setup.propagator.translational(
    central_bodies, vehicle_acceleration_models, bodies_to_propagate,
    vehicle_initial_state, t_final,
    output_variables=[propagation_setup.dependent_variable.central_body_fixed_cartesian_position("Vehicle", "Bennu"),
                        # propagation_setup.dependent_variable.keplerian_state("Vehicle", "Bennu")
                      ]
)

link_ends_per_observable = rss.create_link_ends( bodies, n_GS, n_lnd, "Bennu" )
observation_settings_list = rss.create_observation_settings( link_ends_per_observable,
        range_bias, vlbi_bias, optical_bias, target_body = "Bennu")
inverse_apriori_covariance, idxs, n_params = rss.get_estimation_apriori(names, counts, apr_vals, 1, n_lnd)
parameter_set = create_estimated_parameters(propagator_settings, bodies, cos_tup, sin_tup, n_lnd)
parameter_set_values = parameter_set.values

orbit_determination_manager = estimation_setup.OrbitDeterminationManager(
    bodies, parameter_set, observation_settings_list,
    [integrator_settings], propagator_settings)

simulated_observations = rss.simulate_observations(link_ends_per_observable,
    orbit_determination_manager.observation_simulators, bodies, [initial_state_epoch], buffer,
    total_duration, percentage_arc_tracked,
    t_range, t_doppler, t_vlbi, t_optical,
    n_GS, range_noise, doppler_noise, vlbi_noise, optical_noise,
    range_, vlbi, optical, target_body="Bennu")

pod_input = rss.get_estimation_input( simulated_observations, parameter_set.parameter_set_size,
                                     range_noise, doppler_noise, vlbi_noise, optical_noise,
                                     link_ends_per_observable, inverse_apriori_covariance,
                                      target_body = "Bennu")
pod_output = orbit_determination_manager.perform_estimation( pod_input, estimation_setup.EstimationConvergenceChecker( 1 ) )
f = pod_output.formal_errors
gf_err = rss.err_rms(rss.order_rs_err(f[idxs['cfs'][0]:idxs['cfs'][1]], cos_tup, sin_tup))
parameter_set.reset_values(parameter_set_values)

print(gf_err)
np.savetxt('gf_err.txt', gf_err)

print(np.mean(np.rad2deg(f[idxs['pole'][0]:idxs['pole'][1]])))
print(np.rad2deg(f[idxs['omega'][0]]))

#%% Consider Covariance with Kaula uncertainty
# degrees 6__> 8 are consider parameters

P = pod_output.covariance
Hp = pod_output.design_matrix
Hc = np.concatenate((Hp[:,11+20:11+44], Hp[:,11+44+15:11+44+36]), axis = 1)
p1 = np.empty_like(np.transpose(Hp))
W = pod_input.weights_matrix * np.eye(len(pod_input.weights_matrix))
p1 = np.matmul(P, np.matmul(np.transpose(Hp), W))

C = np.eye((45)) * np.concatenate((kaula_c,kaula_s))**2
p2 = np.matmul(Hc, np.matmul(C, np.transpose(Hc)))
PC = P + np.matmul(p1, np.matmul(p2, np.transpose(p1)))
r = np.sqrt(np.diagonal(P))/np.sqrt(np.diagonal(PC))
gf_err_consider = rss.err_rms(rss.order_rs_err(np.sqrt(np.diagonal(PC)
                )[idxs['cfs'][0]:idxs['cfs'][1]], cos_tup, sin_tup))
np.savetxt('gf_err_consider.txt', gf_err_consider)

#%% Verify Spacecraft Trajectory

dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
            bodies, integrator_settings, propagator_settings)
dvs = np.vstack( list(dynamics_simulator.dependent_variable_history.values( ) ) )
np.save('pos_bcbf.npy',dvs)