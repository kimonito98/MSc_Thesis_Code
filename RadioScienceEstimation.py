# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:01:32 2021

@author: Michael

Radio Science on Phobos

Tracking Data Contribution and  Consider Covariance

"""

import os
if os.getcwd() == 'C:\\Users\\Michael':
    os.chdir('PyPhobos')

import sys
sys.path.append('/home/mplumaris/anaconda3/lib/python3.8/site-packages')

from kernel.interface import spice_interface
from kernel.simulation import propagation_setup
from kernel.simulation import estimation_setup

import RadioScienceSetup as rss
import SimulationSetup as ss
import numpy as np
from matplotlib import pyplot as plt

#%% Setup Envitonment and Intergator

spice_interface.load_standard_kernels([])
# spice_interface.load_standard_kernels(['mar097.bsp'])
spice_interface.load_standard_kernels(['mar097.bsp', 'NOE-4-2020.bsp'])

start_epoch = 946774800.0
# if you want to propagate Phobos for 1 year
Phobos_propagation_duration = 7 * 86400
# end_epoch = start_epoch + Phobos_propagation_duration

x0_inertial = np.load('x0_inertial.npy')
pareto = np.load('pareto.npy')

single_arc_bodies_to_propagate = ["Phobos"]
single_arc_central_bodies = ["Mars"]
multi_arc_bodies_to_propagate = ["Vehicle"]
multi_arc_central_bodies = ["Phobos"]

r_Ph, mu_Ph, dmax_Ph =  14.0E3, 7.11E5, 4 #084E5, 4
mu_Ma, r_Ma, dmax_Ma = 0.4282837566395650E+14, 3396e+3, 10
mu_De = 9.46e+4

frame_origin = "Mars"
frame_orientation = "ECLIPJ2000"

cfs_Ph = ss.read_Phobos_gravity_field( model = "HM")[2]

#%% Define A-Priori Values
dmax = 6 # maximum degree of the estimation
cos_tup, sin_tup = (1,0,dmax,dmax), (1,1,dmax,dmax)
n_c, n_s = ss.count_cfs(cos_tup, sin_tup) # number of coefficients
n_deg_rs = max((cos_tup[2],sin_tup[2])) - min((cos_tup[0],sin_tup[0])) + 1
buffer = 3600. # one hour buffer in between arcs
n_GS = 3 # to use out of 3 total (DSN)
n_lnd = 25 # to use out of 25 total
range_, vlbi, optical = False, False, True
t_doppler, doppler_noise = 60., 1e-13 # doppler integration time and noise
t_range, range_noise, range_bias = 300, 2, 2
t_vlbi, vlbi_noise, vlbi_bias = 300, 1e-9, 1e-9
c_std, s_std, labels = ss.compute_stats_HT_Phobos(max_degree = dmax, max_only=True) # compute std for a-priori values
n_cfs = ss.count_cfs(cos_tup, sin_tup) # total number of harmoinc coeffs

# Uncomment for ame optical performance for all orbits
# t_optical, optical_noise, optical_bias = 10*60, np.deg2rad(0.5), np.deg2rad(0.5)

# Uncomment for different optical performance (scaled with mean radius)
r_min, r_max  = np.min(pareto[:,0]), np.max(pareto[:,0])
t_optical_min, t_optical_max = 5*60, 10*60
noise_opt_min, noise_opt_max = np.deg2rad(0.1), np.deg2rad(0.5)

#%% Perform Estimation

arc_overlap = 0
# manoeuvres per orbit are in accordance with maintenance manoeuvres (see SensitivityAnalysis.py)
mans_per_orbit = np.load('delta_v.npy')[:,2,1] #np.ones((len(pareto)))*7
maneuver_rise_time = 200 # for the de-saturation manoeuvres of the reaction wheels
total_maneuver_time = 4 * maneuver_rise_time # empirical indication
total_duration = 7*86400
t_final = start_epoch + total_duration
percentage_arc_tracked = 0.5 # with RadioScience observables
single_arc_stepsize, multi_arc_stepsize = 75, 100


# Investigate Effect of Tracking Data
cases = 1 #out of a total of 8 cases for tracking settings (see below)
case = 0 # only perform case 0

# CREATE RESULTS VECTORS
gf_err_rs = np.zeros((cases, len(pareto), dmax)) # GF estimation (includes GM)
cor_rs = np.zeros((cases, len(pareto), 5))
err_rs = np.zeros((cases, len(pareto), 10 ))
err_lnd = np.zeros((cases, len(pareto), n_lnd ))
# cor_mat = np.zeros((len(orbs), 214,214)) # fixed n_arcs
consider_ratio = np.zeros((2, len(pareto), 3, 3 + dmax)) # 2 consider par, orbs, 3 uncertainties, Phobos state, GM, cfs

failed = [] # just in case some fail (happends very sporadically)

orbs = np.linspace(0,len(pareto)-1, 1, dtype = int) # to sample across the PF
# orbs = np.arange(0, 102, dtype = int) # to sample all the PF


for case in range(cases):
    if case == 1:
        t_doppler, doppler_noise = 1000, 1e-13/5
        range_, vlbi, optical = False, False, True
        percentage_arc_tracked = 1.0
    if case == 1:
        t_doppler, doppler_noise = 1000, 1e-13/5
        range_, vlbi, optical = True, False, False
        percentage_arc_tracked = 1.0
    elif case == 2:
        # range_, vlbi, optical = False, True, False
        range_, vlbi, optical = True, True, True
        percentage_arc_tracked = 1
    elif case == 3:
        range_, vlbi, optical = True, True, True
        percentage_arc_tracked = 0.5
        range_, vlbi, optical = True, False, True
    elif case == 4:
        range_, vlbi, optical = False, True, True
    elif case == 5:
        range_, vlbi, optical = True, True, True
    elif case == 6:
        t_optical, optical_noise, optical_bias = 3 * 60, np.deg2rad(0.5), np.deg2rad(0.5)
    elif case == 7:
        t_optical, optical_noise, optical_bias = 5 * 60, np.deg2rad(0.1), np.deg2rad(0.1)
    print('Case '+str(case))

    for o, orb in enumerate(orbs):
        x0 = x0_inertial[orb]
        r_mean = pareto[orb,0]
        # Different optical performance (scaled with mean radius)
        t_optical = t_optical_min + (r_mean - r_min) / (r_max - r_min) * (t_optical_max - t_optical_min)
        optical_noise = noise_opt_max - (r_mean - r_min) / (r_max - r_min) * (noise_opt_max - noise_opt_min)
        optical_bias = optical_noise
        names = ['r_Ph', 'v_Ph', 'r_sc', 'v_sc', 'Cr', 'GM', 'desat', 'cfs',
                 'emp', 'r_GS', 'range_bias', 'vlbi_bias', 'optical_bias', 'lnd']
        apr_vals = [100, 0.03, 50, 0.03, 0.1, 700, 4e-3, np.concatenate((c_std, s_std)),
                    1e-9, 5e-3, 2, 1e-9, optical_bias, 12]
        print('Orbit ' + str(orb))
        multi_arc_integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
            start_epoch, multi_arc_stepsize, multi_arc_stepsize, multi_arc_stepsize,  1.0, 1.0)
        # propagation_setup.integrator.runge_kutta_4(start_epoch, multi_arc_stepsize)
        single_arc_integrator_settings= propagation_setup.integrator.adams_bashforth_moulton(
            start_epoch, single_arc_stepsize, single_arc_stepsize, single_arc_stepsize, 1.0, 1.0)
        propagation_setup.integrator.runge_kutta_4(start_epoch, single_arc_stepsize)
        bodies = ss.get_bodies(frame_origin, frame_orientation,
                               r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De,
                               Phobos_spice=False, model="HM")
        rss.add_ground_stations(bodies, n_lnd)
        Phobos_acceleration_models = ss.get_Phobos_acceleration_models(bodies, single_arc_central_bodies, dmax_Ma, dmax_Ph)
        # note: these next 2 lines will not work if Phobos is not retrieved from spice!!!
        # So run it once with spice-Phobos and use the result manually 
        # Phobos_initial_state = propagation_setup.get_initial_state_of_bodies(
        #     single_arc_bodies_to_propagate, single_arc_central_bodies, bodies, start_epoch)
        Phobos_initial_state = np.array([-5.61563403e+06, -6.99812625e+06,  2.54741607e+06,  1.38189778e+03,
           -1.43974592e+03, -7.96603304e+02])
        # add an uncertainty in accordance with Lainey 2021 ephemeris formal error
        Phobos_initial_state += np.concatenate((np.random.normal(0, 1e2, 3), np.random.normal(0, 3e-2, 3)))
        single_arc_propagator_settings = propagation_setup.propagator.translational(
            single_arc_central_bodies, Phobos_acceleration_models, single_arc_bodies_to_propagate,
            Phobos_initial_state, start_epoch + Phobos_propagation_duration,
            # output_variables=[propagation_setup.dependent_variable.relative_distance("Phobos", "Mars")]
        )
        link_ends_per_observable = rss.create_link_ends( bodies, n_GS, n_lnd )
        observation_settings_list = rss.create_observation_settings( link_ends_per_observable,
                                                                     range_bias,
                                                                     vlbi_bias,
                                                                     optical_bias)
        arc_initial_times, arc_propagator_settings_list = [], []
        n_arcs = int(mans_per_orbit[orb]) + 1
        arc_duration = (total_duration/n_arcs)
        thrust_mid_times = list(np.arange(start_epoch + arc_duration*0.5, t_final, arc_duration))
        delta_v_values = list(np.zeros((1, 3))) * len(thrust_mid_times)
        vehicle_acceleration_models = ss.get_vehicle_acceleration_models(bodies, multi_arc_central_bodies,
                                         dmax_Ma, dmax_Ph, thrust_mid_times, delta_v_values, maneuver_rise_time, total_maneuver_time)
        counts = [3, 3, 3 * n_arcs, 3 * n_arcs, 1, 1, 3 * n_arcs, n_cfs[0]+n_cfs[1], 3, 3 * n_GS, n_GS, 2*n_GS, 2*n_lnd, 3*n_lnd] #
        inverse_apriori_covariance, idxs, n_params = rss.get_estimation_apriori(names, counts, apr_vals, n_arcs, n_lnd)
        t_current = start_epoch
        while t_current < t_final:
            arc_initial_times.append(t_current)
            arc_propagator_settings_list.append(propagation_setup.propagator.translational(
                    multi_arc_central_bodies, vehicle_acceleration_models, multi_arc_bodies_to_propagate,
                    x0, t_current + arc_duration + arc_overlap,
                # output_variables=[propagation_setup.dependent_variable.relative_distance("Phobos", "Mars"),
                #                   propagation_setup.dependent_variable.central_body_fixed_cartesian_position("Vehicle","Phobos")]
                ))
            t_current += arc_duration
        multi_arc_propagator_settings = propagation_setup.propagator.multi_arc(arc_propagator_settings_list, True)
        hybrid_arc_propagator_settings = propagation_setup.propagator.hybrid_arc( single_arc_propagator_settings,
                                                                                  multi_arc_propagator_settings)
        parameter_set = rss.create_estimated_parameters(
            multi_arc_propagator_settings, arc_initial_times, single_arc_propagator_settings,
            bodies, cos_tup, sin_tup, link_ends_per_observable, n_GS, n_lnd)
        parameter_set_values = parameter_set.values
    
        orbit_determination_manager = estimation_setup.OrbitDeterminationManager(
            bodies, parameter_set, observation_settings_list,
            [single_arc_integrator_settings, multi_arc_integrator_settings],
            hybrid_arc_propagator_settings)
    
        simulated_observations = rss.simulate_observations(link_ends_per_observable,
            orbit_determination_manager.observation_simulators, bodies, arc_initial_times, buffer,
            arc_duration, percentage_arc_tracked,
            t_range, t_doppler, t_vlbi, t_optical,
            n_GS, range_noise, doppler_noise, vlbi_noise, optical_noise,
            range_, vlbi, optical)
    
        pod_input = rss.get_estimation_input( simulated_observations, parameter_set.parameter_set_size,
                                             range_noise, doppler_noise, vlbi_noise, optical_noise,
                                             link_ends_per_observable, inverse_apriori_covariance = inverse_apriori_covariance )
        pod_output = orbit_determination_manager.perform_estimation( pod_input, estimation_setup.EstimationConvergenceChecker( 1 ) )
        try:
            cn = np.linalg.cond(pod_output.correlations)
        except np.linalg.LinAlgError:
            failed.append(orb)
            continue
        f = pod_output.formal_errors
        gf_err_rs[case, orb] = rss.err_stats(rss.order_rs_err(f[idxs['cfs'][0]:idxs['cfs'][1]],
                                        cos_tup, sin_tup), cfs_Ph)
        # Correlations
        cn_gf = np.linalg.cond(pod_output.correlations[idxs['cfs'][0]:idxs['cfs'][1], idxs['cfs'][0]:idxs['cfs'][1]])
        corr_C2 = abs(pod_output.correlations[idxs['cfs'][0]+2,idxs['cfs'][0]+4]) #C20 & C22
        # lst.append(abs(pod_output.correlations[idxs['cfs'][0],idxs['cfs'][0]])) #C20 & C22)
        cn_Ph = np.linalg.cond(pod_output.correlations[idxs['r_Ph'][0]:idxs['v_Ph'][1], idxs['r_Ph'][0]:idxs['v_Ph'][1]])
        cn_lnd = np.linalg.cond(pod_output.correlations[idxs['lnd0'][0]:idxs['lnd'+str(n_lnd-1)][1],
                                                        idxs['lnd0'][0]:idxs['lnd'+str(n_lnd-1)][1]])
        cor_rs[case, orb] = np.array([cn, cn_gf, corr_C2, cn_Ph, cn_lnd])
        err_rs[case, orb] = np.array([np.linalg.norm(f[idxs['r_sc0'][0]:idxs['r_sc0'][1]]),
                          np.linalg.norm(f[idxs['v_sc0'][0]:idxs['v_sc0'][1]]),
                          np.linalg.norm(f[idxs['desat'][0]:idxs['desat'][1]]),
                          np.linalg.norm(f[idxs['emp'][0]:idxs['emp'][1]]),
                          np.linalg.norm(f[idxs['range_bias'][0]:idxs['range_bias'][1]]),
                          np.linalg.norm(f[idxs['vlbi_bias'][0]:idxs['vlbi_bias'][1]]),
                          np.linalg.norm(f[idxs['optical_bias'][0]:idxs['optical_bias'][1]]),
                            f[idxs['GM'][0]] / mu_Ph,
                            np.linalg.norm(f[idxs['r_Ph'][0]:idxs['r_Ph'][1]]),
                            np.linalg.norm(f[idxs['v_Ph'][0]:idxs['v_Ph'][1]])])
        for lnd in range(n_lnd):
            err_lnd[case, orb, lnd] = np.mean(f[idxs['lnd'+str(lnd)][0]:idxs['lnd'+str(lnd)][1]])
    
        # cor_mat[o] = pod_output.correlations
        parameter_set.reset_values(parameter_set_values)


#%%

# to investigate single arc
# np.save('single_gf_err_rs.npy', gf_err_rs)
# np.save('single_err_rs.npy', err_rs)
# np.save('single_cor_rs.npy', cor_rs)

# to investigate tracking
# np.save('tracking_gf_err_rs.npy', gf_err_rs)
# np.save('tracking_err_rs.npy', err_rs)
# np.save('tracking_cor_rs.npy', cor_rs)
# np.save('tracking_err_lnd.npy', err_lnd)
# np.save('cor_mat_across10.npy', cor_mat)

#%% Consider Coveriance Analysis

    GS_position = (1e-3, 5e-3, 1e-2)
    empricial_sensitivity = (1e-10, 0.5e-10, 1e-11)
    batch_size = 1000
    for u, uncertainty in enumerate((GS_position, empricial_sensitivity)):
        P = pod_output.covariance
        Hp = pod_output.design_matrix
        param = 'r_GS' if u == 0 else 'emp'
        Hc = Hp[:,idxs[param][0]:idxs[param][1]]
        for uu, unc in enumerate(uncertainty):
            p1 = np.empty_like(np.transpose(Hp))
            for hp in range(0,len(Hp),batch_size):
                W = pod_input.weights_matrix[hp:hp+batch_size] * \
                    np.eye(len(pod_input.weights_matrix[hp:hp+batch_size]))
                p1[:,hp:hp+batch_size] = np.matmul(np.transpose(Hp[hp:hp+batch_size]), W)
            p1 = np.matmul(P, p1)
            C = np.eye(idxs[param][1]-idxs[param][0]) * unc**2
            dummy = np.zeros_like(P)
            for hp in range(0,len(Hp),batch_size):
                p2 = np.matmul(Hc[hp:hp+batch_size], np.matmul(C, np.transpose(Hc[hp:hp+batch_size])))
                dummy += np.matmul(p1[:,hp:hp+batch_size], np.matmul(p2, np.transpose(p1[:,hp:hp+batch_size])))
            PC = P + dummy
            r = np.sqrt(np.diagonal(P))/np.sqrt(np.diagonal(PC))
            err_gf_rms_cns = rss.err_stats(rss.order_rs_err(np.sqrt(np.diagonal(PC))[idxs['cfs'][0]:
                                idxs['cfs'][1]], cos_tup, sin_tup), cfs_Ph)
            consider_ratio[u, orb, uu] = np.concatenate((np.array([
                               np.mean(r[idxs['r_Ph'][0]:idxs['r_Ph'][1]]),
                               np.mean(r[idxs['v_Ph'][0]:idxs['v_Ph'][1]]),
                               r[idxs['GM'][0]]]),
                               gf_err_rs[case, orb]/err_gf_rms_cns))

np.save('consider_ratio.npy', consider_ratio)

#%% Plot Formal Error on Spacecraft Trajectory around Phobos

stepsize = 75
output_times = np.arange(start_epoch, start_epoch + 3.5*86400 + stepsize, stepsize)
pfe = estimation_setup.propagate_formal_errors(
        pod_output.covariance,
        orbit_determination_manager.state_transition_interface,
        output_times
    )
pfe = np.array(list(pfe.values()))[:,6:9]
ferr = np.linalg.norm(pfe, axis = 1)

np.savetxt('Geoffrey_orbit_posError.txt', ferr)

dvh = dict()
for d in pod_output.dependent_variable_history[0]:
    dvh.update(d)
x,y,z = (np.vstack(list(dvh.values()))[0:2000,1:].T)*1e-3

from mpl_toolkits.mplot3d.art3d import Line3DCollection

r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
maxx = np.max(r)

# generate a list of (x,y,z) points
points = np.array([x,y,z]).transpose().reshape(-1,1,3)
# set up a list of segments
segs = np.concatenate([points[:-1],points[1:]],axis=1)

# make the collection of segments
lc = Line3DCollection(segs, cmap=plt.get_cmap('jet'))
lc.set_array(ferr) # color the segments by our parameter

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(lc)
(xs, ys, zs) = ss.drawSphere(0, 0, 0, 14)
ax.plot_wireframe(xs, ys, zs, color="grey")

im = ax.scatter(x, y, z, s = 0, c=ferr, cmap=plt.cm.jet)
ss.set_lims(ax, -maxx, maxx, z = True)

cb = fig.colorbar(im, ax=ax)
cb.set_label('Spaceraft Position Error [m]')
plt.show()

#%% Compute surface mapping

surface_mapping = np.zeros((len(x0_inertial)))
fov = np.deg2rad(18.9)
single_arc_stepsize = 60 # slice every 5
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
            start_epoch, single_arc_stepsize, single_arc_stepsize, single_arc_stepsize, 1.0, 1.0)
dependent_variables_to_save = [
        propagation_setup.dependent_variable.longitude("Vehicle", "Phobos"),
        propagation_setup.dependent_variable.latitude("Vehicle", "Phobos")
    ]
r_min, r_max  = np.min(pareto[:,0]), np.max(pareto[:,0])
t_optical_min, t_optical_max = 5*60, 10*60
maneuver_rise_time = 200
total_maneuver_time = 4 * maneuver_rise_time
n_arcs = 3
total_duration = 7*86400
t_final = start_epoch+total_duration
arc_duration = (total_duration/n_arcs)
thrust_mid_times = list(np.arange(start_epoch + arc_duration*0.5, t_final, arc_duration))
delta_v_values = list(np.zeros((1, 3))) * len(thrust_mid_times)
bodies = ss.get_bodies(frame_origin, frame_orientation,
                           r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De,
                           Phobos_spice=True, model="HM")
vehicle_acceleration_models = ss.get_vehicle_acceleration_models(bodies, multi_arc_bodies_to_propagate,
     dmax_Ma, dmax_Ph, thrust_mid_times, delta_v_values, maneuver_rise_time, total_maneuver_time)

for orb, x0 in enumerate(x0_inertial[0:3]):
    propagator_settings = propagation_setup.propagator.translational(
        multi_arc_central_bodies, vehicle_acceleration_models, multi_arc_bodies_to_propagate,
        x0, t_final, output_variables=dependent_variables_to_save,
    )
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
        bodies, integrator_settings, propagator_settings)
    r_mean = pareto[orb, 0]
    t_optical = t_optical_min + (r_mean - r_min) / (r_max - r_min) * (t_optical_max - t_optical_min)
    slice = 20 #int(t_optical/60)
    states = dynamics_simulator.state_history
    states_list = np.vstack(list(states.values()))
    r = np.sqrt(states_list[:, 0] ** 2 + states_list[:, 1] ** 2 + states_list[:, 2] ** 2)

    dependent_variables = dynamics_simulator.dependent_variable_history
    dvl = np.vstack(list(dependent_variables.values()))
    surface_mapping[orb] = rss.check_surface_coverage(dvl[0:-1:slice, 1], dvl[0:-1:slice, 0],
                                                      r[0:-1:slice], fov, r_Ph)

np.save('surface_mapping.npy', surface_mapping)