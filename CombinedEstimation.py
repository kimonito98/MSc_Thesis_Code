
import os
if os.getcwd() == 'C:\\Users\\Michael':
    os.chdir('PyPhobos')

import sys
sys.path.append('/home/mplumaris/anaconda3/lib/python3.8/site-packages')

# my own version 
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


#%% Setup Envitonment and Intergator
spice_interface.load_standard_kernels([])
# spice_interface.load_standard_kernels(['mar097.bsp'])
spice_interface.load_standard_kernels(['mar097.bsp', 'NOE-4-2020.bsp'])

start_epoch = 946774800.0
# if you want to propagate Phobos for 1 year
Phobos_propagation_duration = 7 * 86400

x0_inertial = np.load('x0_inertial.npy')
pareto = np.load('pareto.npy')

single_arc_bodies_to_propagate = ["Phobos"]
single_arc_central_bodies = ["Mars"]
multi_arc_bodies_to_propagate = ["Vehicle"]
multi_arc_central_bodies = ["Phobos"]

cfs_Ph = ss.read_Phobos_gravity_field( model = "HM")[2]
r_Ph, mu_Ph, dmax_Ph =  14.0E3, 7.11E5, 10
mu_Ma, r_Ma, dmax_Ma = 0.4282837566395650E+14, 3396e+3, 10
mu_De = 9.46e+4

frame_origin = "Mars"
frame_orientation = "ECLIPJ2000"

#%% CAI Gradiometry Estimation Settings

tm = 11 # measurement time
sigma_gamma = 3.75e-11 # sensitivity
dmin, dmax = 1, 10 # min and max degree of gravity field
batch = 1000 # number of obs. processed simulataneously
Wcg = np.eye(batch)/(sigma_gamma**2) # weight matrix
R_RSW2GRF = np.array([[0,  1,  0], [0,  0, -1], [1,  0,  0]]) # rotation matrix 
cg_idxs = gs.get_cg_idxs(dmin,dmax)
# c_std, s_std, labels = ss.compute_stats_HT_Phobos(max_degree = dmax, max_only=True)
c_std, s_std = np.load('c_std.npy'), np.load('s_std.npy') # do not compute them each time
apr_dict = dict()
cc, cs = 0, 0
for d in range(1, dmax+1):
    for o in range(d+1):
        apr_dict['C'+str(d)+','+str(o)] = c_std[cc]
        if o != 0:
            apr_dict['S'+str(d)+','+str(o)] = s_std[cs]
            cs += 1
        cc += 1

#%% RS Estimation Values Values
dmax_rs = 6 # maximum degree
cos_tup, sin_tup = (1,0,dmax_rs,dmax_rs), (1,1,dmax_rs,dmax_rs)
n_c, n_s = ss.count_cfs(cos_tup, sin_tup)
n_deg_rs = max((cos_tup[2],sin_tup[2])) - min((cos_tup[0],sin_tup[0])) + 1
buffer = 3600. # 1 hour buffer on each arc
n_GS = 3 # Earth Ground Stations to use where max= 3 (DSN)
n_lnd = 25 # Phobos landmarks to use
range_, vlbi, optical = False, False, True
t_doppler, doppler_noise = 60., 1e-13
t_range, range_noise, range_bias = 300, 2, 2
t_vlbi, vlbi_noise, vlbi_bias = 300, 1e-9, 1e-9
# t_optical, optical_noise, optical_bias = 10*60, np.deg2rad(0.5), np.deg2rad(0.5)
# create full a-priori matrix
rs_idxs = list(rss.get_rs_idxs((1,0,dmax,dmax),(1,1,dmax,dmax)))
inv_apr_cov = np.linalg.inv(np.eye(len(cg_idxs))*np.concatenate((c_std, s_std))**2)
inv_apr_cov = rss.reshape_covariance(inv_apr_cov, rs_idxs, cg_idxs)
rs_idxs = list(rss.get_rs_idxs(cos_tup,sin_tup))

#%% Perform Estimation

# Set percentage of tracking data arc (100%, 50%,...)
# pats = [1.0, 0.5] # this is done for RS only experiment
pats = [0.5] # This is done for combined experiment

# create results arrays
gf_err = np.zeros((3, len(x0_inertial), 3,  dmax - dmin + 1)) # gravity field rms formal error, combined experiment
gf_err_rs = np.zeros((3, len(x0_inertial), n_deg_rs)) # same, RS alone
gf_cor = np.zeros((len(pats), len(x0_inertial), 3,  2)) # condition number & corr c20-c22, combined
err_rs = np.zeros((len(pats), len(pareto), 6)) # formal error non-gravity parameters, RS only
cor_rs = np.zeros((len(pats), len(pareto), 5)) # correlations for RS alone

# dependent variables to save (for gradiometry) if Phobos or Mars is dominant attractor
dependent_variables_to_save_Ph = [
    propagation_setup.dependent_variable.rotation_matrix_to_body_fixed_frame("Phobos"),
    propagation_setup.dependent_variable.longitude("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.latitude("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.relative_position("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.relative_velocity("Vehicle", "Phobos")]
dependent_variables_to_save_Ma = [
    propagation_setup.dependent_variable.rotation_matrix_to_body_fixed_frame("Phobos"),
    propagation_setup.dependent_variable.longitude("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.latitude("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.relative_position("Vehicle", "Mars"),
    propagation_setup.dependent_variable.relative_velocity("Vehicle", "Mars"),]

# Adapted optical performance per orbit
r_min, r_max  = np.min(pareto[:,0]), np.max(pareto[:,0])
t_optical_min, t_optical_max = 5*60, 10*60
noise_opt_min, noise_opt_max = np.deg2rad(0.1), np.deg2rad(0.5)

# arc and manoeuvre settings
arc_overlap = 0
mans_per_orbit = np.load('delta_v.npy')[:,2,1] #np.ones((len(pareto)))*7
maneuver_rise_time = 200
total_maneuver_time = 4 * maneuver_rise_time
total_duration = 7 * 86400
t_final = start_epoch + total_duration
single_arc_stepsize, multi_arc_stepsize = 75, 75
measurement_epochs = np.arange(start_epoch, start_epoch + total_duration, tm )
consider_ratio = np.zeros((2, len(pareto), 3, 3 + 6))


failed = [] # save here the solutions which failed
for pat, percentage_arc_tracked in enumerate(pats):
    # orbs = np.linspace(1,len(pareto)-1, 20, dtype = int) # to sample across
    # orbs = np.arange(20, 30, dtype = int) # to sample all
    # cor_mat_rs = [] #np.zeros((2, 229, 229))
    # cor_mat = np.zeros((10, 120, 120))

    # Careful!!! Only for BEST ORBIT!! best orbit is 26 for RS (100% tracking), 0 for CG (50% )
    # pareto = np.load('pareto_reduced.npy')
    # x0_inertial = np.load('x0_inertial.npy')
    orbs = [26]
    for ob, orb in enumerate(orbs):
        print('Orbit '+str(orb))
        x0 = np.array([-1.26957088e+04, -1.58212186e+04,  5.75914538e+03, -3.15673153e+00,
        4.62706278e+00,  5.75238597e+00])#x0_inertial[orb]
        r_mean = pareto[orb, 0]
        t_optical = t_optical_min + (r_mean - r_min) / (r_max - r_min) * (t_optical_max - t_optical_min)
        optical_noise = noise_opt_max - (r_mean - r_min) / (r_max - r_min) * (noise_opt_max - noise_opt_min)
        optical_bias = optical_noise
        names = ['r_Ph', 'v_Ph', 'r_sc', 'v_sc', 'Cr', 'GM', 'desat', 'cfs',
                 'emp', 'r_GS', 'range_bias', 'vlbi_bias', 'optical_bias', 'lnd']  # ,
        apr_vals = [100, 0.03, 50, 0.03, 0.1, 700, 4e-3, np.concatenate((c_std[0:n_c], s_std[0:n_s])),
                    1e-9, 5e-3, 2, 1e-9, optical_bias, 12]
        multi_arc_integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
            start_epoch, multi_arc_stepsize, multi_arc_stepsize, multi_arc_stepsize,  1.0, 1.0)
        single_arc_integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
            start_epoch, single_arc_stepsize, single_arc_stepsize, single_arc_stepsize,  1.0, 1.0)
        bodies = ss.get_bodies(frame_origin, frame_orientation,
                               r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De,
                               Phobos_spice=False, model="HM")
        rss.add_ground_stations(bodies, n_lnd)
        Phobos_acceleration_models = ss.get_Phobos_acceleration_models(bodies, single_arc_central_bodies, dmax_Ma, dmax_Ph)
        Phobos_initial_state = np.array([-5.61563403e+06, -6.99812625e+06,  2.54741607e+06,  1.38189778e+03,
               -1.43974592e+03, -7.96603304e+02])
        # Phobos_initial_state = propagation_setup.get_initial_state_of_bodies(
            # single_arc_bodies_to_propagate, single_arc_central_bodies, bodies, start_epoch)
        single_arc_propagator_settings = propagation_setup.propagator.translational(
            single_arc_central_bodies, Phobos_acceleration_models, single_arc_bodies_to_propagate,
            Phobos_initial_state, start_epoch + Phobos_propagation_duration
        )

        link_ends_per_observable = rss.create_link_ends( bodies, n_GS, n_lnd )
        observation_settings_list = rss.create_observation_settings( link_ends_per_observable,
                                                                     range_bias, vlbi_bias, optical_bias)
        arc_initial_times, arc_propagator_settings_list = [], []
        n_arcs = int(mans_per_orbit[orb]) + 1
        arc_duration = (total_duration/n_arcs)
        thrust_mid_times = list(np.arange(start_epoch + arc_duration*0.5, t_final, arc_duration))
        delta_v_values = list(np.zeros((1, 3))) * len(thrust_mid_times)
        vehicle_acceleration_models = ss.get_vehicle_acceleration_models(bodies, multi_arc_central_bodies,
                                         dmax_Ma, dmax_Ph, thrust_mid_times, delta_v_values, maneuver_rise_time, total_maneuver_time)
        counts = [3, 3, 3 * n_arcs, 3 * n_arcs, 1, 1, 3 * n_arcs, len(rs_idxs), 3, 3 * n_GS, n_GS, 2*n_GS, 2*n_lnd, 3*n_lnd] #n_GS
        inv_apr_cov_rs, idxs, n_params = rss.get_estimation_apriori(names, counts, apr_vals, n_arcs, n_lnd)
        t_current = start_epoch
        while t_current < t_final:
            arc_initial_times.append(t_current)
            arc_propagator_settings_list.append(propagation_setup.propagator.translational(
                    multi_arc_central_bodies, vehicle_acceleration_models, multi_arc_bodies_to_propagate,
                # for setting Mars as central body
                # ['Mars'], vehicle_acceleration_models, multi_arc_bodies_to_propagate,
                # x0+Phobos_initial_state, t_current + arc_duration + arc_overlap,
                x0, t_current + arc_duration + arc_overlap,
                    output_variables = dependent_variables_to_save_Ph if r_mean <= 22
                    else dependent_variables_to_save_Ma
            ))
            t_current += arc_duration
        multi_arc_propagator_settings = propagation_setup.propagator.multi_arc(arc_propagator_settings_list, True)
        hybrid_arc_propagator_settings = propagation_setup.propagator.hybrid_arc( single_arc_propagator_settings,
                                                                                  multi_arc_propagator_settings)
        # RADIO SCIENCE
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
                                             link_ends_per_observable, inv_apr_cov_rs )
        pod_output = orbit_determination_manager.perform_estimation( pod_input, estimation_setup.EstimationConvergenceChecker( 1 ) )
        # cor_mat_rs.append(pod_output.correlations)
        # cor_mat_rs.append(idxs)
        try:
            cn = np.linalg.cond(pod_output.correlations)
        except np.linalg.LinAlgError:
            failed.append(orb)
            continue

        # Design Correlation and Weight Matrices for RS
        f = pod_output.formal_errors
        gf_err_rs[0, orb] = rss.err_stats(rss.order_rs_err(f[idxs['cfs'][0]:idxs['cfs'][1]],
                                        cos_tup, sin_tup), apr_dict, normalise = False, ignore_S2X=True)
        gf_err_rs[1, orb] = rss.err_stats(rss.order_rs_err(f[idxs['cfs'][0]:idxs['cfs'][1]],
                                         cos_tup, sin_tup), apr_dict, normalise=True, ignore_S2X=True)
        gf_err_rs[2, orb] = rss.err_stats(rss.order_rs_err(f[idxs['cfs'][0]:idxs['cfs'][1]],
                                        cos_tup, sin_tup), cfs_Ph, normalise=True, ignore_S2X=True)

        err_rs[pat,orb] = np.array([ f[idxs['GM'][0]] / mu_Ph,
                                  np.linalg.norm(f[idxs['r_sc0'][0]:idxs['r_sc0'][1]]),
                                  np.linalg.norm(f[idxs['v_sc0'][0]:idxs['v_sc0'][1]]),
                                  np.linalg.norm(f[idxs['r_Ph'][0]:idxs['r_Ph'][1]]),
                                  np.linalg.norm(f[idxs['v_Ph'][0]:idxs['v_Ph'][1]]),
                                  np.mean(f[idxs['lnd0'][0]:idxs['lnd' + str(n_lnd - 1)][1]]) ])
        cn_gf = np.linalg.cond(pod_output.correlations[idxs['cfs'][0]:idxs['cfs'][1], idxs['cfs'][0]:idxs['cfs'][1]])
        corr_C2 = abs(pod_output.correlations[idxs['cfs'][0] + 2, idxs['cfs'][0] + 4])  # C20 & C22
        cn_Ph = np.linalg.cond(pod_output.correlations[idxs['r_Ph'][0]:idxs['v_Ph'][1], idxs['r_Ph'][0]:idxs['v_Ph'][1]])
        cn_lnd = np.linalg.cond(pod_output.correlations[idxs['lnd0'][0]:idxs['lnd' + str(n_lnd - 1)][1],
                                idxs['lnd0'][0]:idxs['lnd' + str(n_lnd - 1)][1]])
        cor_rs[pat,orb] = np.array([cn, cn_gf, corr_C2, cn_Ph, cn_lnd])
        parameter_set.reset_values(parameter_set_values)
        # # GRADIOMETRY
        dvh_timestep, dvh_tm, states_timestep, states_tm = dict(), dict(), dict(), dict()
        for d, s in zip(pod_output.dependent_variable_history[0],pod_output.dynamics_history[0]) :
            dvh_timestep.update(d)
            states_timestep.update(s)
        dvh_interpolator = interpolators.create_one_dimensional_interpolator(dvh_timestep,
                    interpolators.lagrange_interpolation(8))
        states_interpolator = interpolators.create_one_dimensional_interpolator(states_timestep,
                    interpolators.lagrange_interpolation(8))
        for epoch in measurement_epochs:
            dvh_tm[epoch] = dvh_interpolator.interpolate(epoch)
            states_tm[epoch] = states_interpolator.interpolate(epoch)
        dvh_tm = np.vstack(list(dvh_tm.values()))
        states_tm = np.vstack(list(states_tm.values()))
        print('Computing Rotations')
        r_PhF, misal = np.empty((2, len(dvh_tm), 3))
        R_LNOF2GRF = np.empty((len(dvh_tm), 3, 3))
        for epoch, (d,s) in enumerate(zip(dvh_tm, states_tm)):
            # Compute spacecraft position in Phobos-fixed frame
            R_Inrt2PhF = d[0:9].reshape(3, 3)
            r_PhF[epoch] = np.matmul(R_Inrt2PhF, s[6:9])
            # Rotation from LNOF --> Phobos-Fixed --> Inertial --> GRF (z switches sign!!!)
            r_PhF[epoch][2] = r_PhF[epoch][2] * (-1)
            R_LNOF2PhF = frames.lv_to_body_fixed_rotation_matrix(d[9], d[10])
            # rotate 90 positive around z, 90  negative around x
            R_Inrt2GRF = np.matmul(R_RSW2GRF, frames.inertial_to_rsw_rotation_matrix(d[11:17]))
            # misal[epoch] = np.arccos( np.dot(R_Inrt2GRF, d[11:14]/np.linalg.norm(d[11:14]) ))
            R_LNOF2GRF[epoch] = np.matmul(R_Inrt2GRF, np.matmul(np.transpose(R_Inrt2PhF), R_LNOF2PhF))
        if np.linalg.norm(r_PhF[-1]) > 70e3:
            failed.append(orb)
            continue
        print('Computing Design Matrices')
        [Axx, Ayy, Azz] = gs.compute_design(r_PhF[5:-10], mu_Ph, r_Ph, dmin, dmax)
        [Axx, Ayy, Azz] = gs.subtract_central_field(Axx, Ayy, Azz, mu_Ph, r_Ph, r_PhF[5:-10])
        [Axx, Ayy, Azz] = gs.rotate_design(Axx, Ayy, Azz, R_LNOF2GRF[5:-10])

        Hrs = pod_output.design_matrix[:,idxs['cfs'][0]:idxs['cfs'][1]]
        Wrs = pod_input.weights_matrix
        inv_cov_rs = np.zeros((len(rs_idxs), len(rs_idxs)))
        for i in range(0, len(Hrs), batch):
            W = Wrs[i:i + 1000] * np.eye(len(Wrs[i:i + batch]))
            inv_cov_rs += np.matmul(np.matmul(Hrs[i:i + batch].T, W), Hrs[i:i + batch])
        cov = rss.reshape_covariance(inv_cov_rs, rs_idxs, cg_idxs)
        cov = np.repeat(cov[np.newaxis], 3, axis=0)
        cgs = np.zeros((3,120,120))
        for i, a in enumerate((Axx, Ayy, Azz)):
            for j in range(0,len(a),batch):
                cov[i] += np.matmul(np.matmul(a[j:j+batch].T, Wcg[:len(a[j:j+batch]),:len(a[j:j+batch])]), a[j:j+batch])
            cov[i] = np.linalg.inv(inv_apr_cov + cov[i])
            err = gs.order_gg_err(np.sqrt(np.diagonal(np.abs(cov[i]))), dmin, dmax, False)
            gf_err[0, orb, i] = rss.err_stats(err, apr_dict, normalise = False, ignore_S2X=True)
            gf_err[1, orb, i] = rss.err_stats(err, apr_dict, normalise = True, ignore_S2X=True)
            gf_err[2, orb, i] = rss.err_stats(err, cfs_Ph, normalise=True, ignore_S2X=True)
            cg = np.divide(cov[i], np.matmul(np.fromiter(err.values(), dtype=float).reshape(1, len(cov[i])).T,
                                                  np.fromiter(err.values(), dtype=float).reshape(1, len(cov[i]))))
            cgs[i] = cg
            cn = np.linalg.cond(cg)
            gf_cor[pat,orb, i] = np.array([cn, abs(cg[cg_idxs['C2,0'], cg_idxs['C2,2']])])
        # save the correlation matrix for the best axis in terms of accuracy
        # cor_mat[ob] = rss.reshape_covariance(cgs[np.argmin(np.mean(gf_err[pat, orb], axis=1))],
        #                                      list(cg_idxs), rss.get_rs_idxs((1, 0, dmax, dmax), (1, 1, dmax, dmax)))


# Uncomment the following lines for a consider covariance analysis 
#         GS_position = (1e-3, 5e-3, 1e-2)
#         empricial_sensitivity = (1e-9, 1e-10, 0.5e-10)
#         batch_size = 1000
#         for u, uncertainty in enumerate((GS_position, empricial_sensitivity)):
#             P = pod_output.covariance
#             Hp = pod_output.design_matrix
#             param = 'r_GS' if u == 0 else 'emp'
#             Hc = Hp[:,idxs[param][0]:idxs[param][1]]
#             for uu, unc in enumerate(uncertainty):
#                 p1 = np.empty_like(np.transpose(Hp))
#                 for hp in range(0,len(Hp),batch_size):
#                     W = pod_input.weights_matrix[hp:hp+batch_size] * \
#                         np.eye(len(pod_input.weights_matrix[hp:hp+batch_size]))
#                     p1[:,hp:hp+batch_size] = np.matmul(np.transpose(Hp[hp:hp+batch_size]), W)
#                 p1 = np.matmul(P, p1)
#                 C = np.eye(idxs[param][1]-idxs[param][0]) * unc**2
#                 dummy = np.zeros_like(P)
#                 for hp in range(0,len(Hp),batch_size):
#                     p2 = np.matmul(Hc[hp:hp+batch_size], np.matmul(C, np.transpose(Hc[hp:hp+batch_size])))
#                     dummy += np.matmul(p1[:,hp:hp+batch_size], np.matmul(p2, np.transpose(p1[:,hp:hp+batch_size])))
#                 PC = P + dummy
#                 r = np.sqrt(np.diagonal(P))/np.sqrt(np.diagonal(PC))
#                 err_gf_rms_cns = rss.err_stats(rss.order_rs_err(np.sqrt(np.diagonal(PC))[idxs['cfs'][0]:
#                                     idxs['cfs'][1]], cos_tup, sin_tup), cfs_Ph, normalise = False)
#                 consider_ratio[u, orb, uu] = np.concatenate((np.array([
#                                    np.mean(r[idxs['r_Ph'][0]:idxs['r_Ph'][1]]),
#                                    np.mean(r[idxs['v_Ph'][0]:idxs['v_Ph'][1]]),
#                                    r[idxs['GM'][0]]]),
#                                    gf_err_rs[0, orb]/err_gf_rms_cns))
#
# np.save('consider_ratio.npy', consider_ratio)

# this removes the solutions that failed (happens very sporadically)
# if failed:
    gf_err_rs = np.delete(gf_err_rs, failed, axis = 1)
    gf_err = np.delete(gf_err, failed, axis=1)
    err_rs = np.delete(err_rs, failed, axis=1)
    gf_cor = np.delete(gf_cor, failed, axis=1)
    cor_rs = np.delete(cor_rs, failed, axis=1)
    pareto_reduced = np.delete(pareto, failed, axis = 0 )
    x0_inertial_reduced = np.delete(x0_inertial, failed, axis= 0 )

# # save results for combined estimation
np.save('gf_err_rs.npy', gf_err_rs)
np.save('gf_err.npy', gf_err)
np.save('err_rs.npy', err_rs)
np.save('gf_cor.npy', gf_cor)
np.save('cor_rs.npy',cor_rs)
np.save('pareto_reduced.npy',pareto_reduced)
np.save('x0_inertial_reduced.npy',pareto_reduced)

# FOR BEST ORBIT ONLY:
# best_rs = ss.fill_best(rss.order_rs_err(f[idxs['cfs'][0]:idxs['cfs'][1]], cos_tup, sin_tup), cfs_Ph, dmax, normalise = False)
# best_rs_n = ss.fill_best(rss.order_rs_err(f[idxs['cfs'][0]:idxs['cfs'][1]], cos_tup, sin_tup), cfs_Ph, dmax, normalise = True)
# np.save('pyramid_'+str(orb)+'_rs.npy', best_rs)
# np.save('pyramid_'+str(orb)+'_rs_n.npy', best_rs_n)
# best = ss.fill_best(err, cfs_Ph, dmax, normalise = False)
# best_n = ss.fill_best(err, cfs_Ph, dmax, normalise = True)
# np.save('pyramid_'+str(orb)+'.npy', best)
# np.save('pyramid_'+str(orb)+'_n.npy', best_n)


# np.save('cor_grad.npy', cor_mat) # average across many
# np.save('cor_mat', cor_mat) # for 2 orbits of interest
# np.save('cor_mat_rs0', cor_mat_rs[0])
# np.save('cor_mat_rs26', cor_mat_rs[2])



#%% Propagate Phobos Epehemeris error, compare best and worst performance
""" Best 5 0.15m at 100% tracking, Worst 93 2.6 m at 50% tracking
Propagate formal error on estimated Phobos
"""
# order: single arc (r_PhobosMars), multi arc (r_Sc_Phobos)
output_times = np.arange(start_epoch, start_epoch + Phobos_propagation_duration, 400)
pfe = estimation_setup.propagate_formal_errors(
        pod_output.covariance,
        orbit_determination_manager.state_transition_interface,
        output_times
    )
pfe = np.array(list(pfe.values()))
# pfe_MaF_best_1year = np.array(list(pfe.values()))[:,0:3]
# pfe_MaF_worst_1year = np.array(list(pfe.values()))[:,0:3]

# np.save('pfe_MaF_best_1year.npy', pfe_MaF_best_1year)
# np.save('pfe_MaF_worst_1year.npy', pfe_MaF_worst_1year)
# np.save('pfe_times.npy', (np.array(output_times)-start_epoch)/86400)




# %% Noise due to Mars coefficient uncertainty and SPacecraft Position Error
""" CAREFUL: for r_mean < 22, Phobos dominates and cfs uncertainty vanishes
BUT perturbed error may be much more severe! 
So evaluate its extent and, if necessary, place a minimum arc length on it!
Proably wont matter since close orbits have very good estimates for r_Ph_sc

I made a massive mistake in subtract central field! Repeat! But probably wont matter
"""
# order: single arc (r_PhobosMars), multi arc (r_Sc_Phobos)
output_times = np.arange(start_epoch, start_epoch + total_duration, tm)
pfe = estimation_setup.propagate_formal_errors(
        pod_output.covariance,
        orbit_determination_manager.state_transition_interface,
        output_times
    )
pfe = np.array(list(pfe.values()))
# np.save('formal_error_propagation_orb93.npy', pfe)

if r_mean <= 22:
    # Only position Noise
    dmin, dmax = 0, 10
    cfs = gs.flatten_dict(ss.read_Phobos_gravity_field('HM')[2], dmin, dmax)
else:
    # Also cfs uncertainty noise
    dmin, dmax = 0, 12
    cfs = gs.flatten_dict(ss.read_Mars_gravity_field('jmro120d.txt',
                        max_degree = dmax,  std = False)[2], dmin, dmax)
    cfs_Ma_std = gs.flatten_dict(ss.read_Mars_gravity_field('jmro120d.txt',
                        max_degree = dmax,  std = True)[2], dmin, dmax)
    # std(C00) = std(GM)/GM
    cfs_Ma_std[0] = cfs_Ma_std[0]/mu_Ma
    cov_cfsMa = np.eye(len(cfs_Ma_std))*(cfs_Ma_std**2)

dependent_variables_to_save_Ph = [
    propagation_setup.dependent_variable.rotation_matrix_to_body_fixed_frame("Phobos"),
    propagation_setup.dependent_variable.longitude("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.latitude("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.relative_position("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.relative_velocity("Vehicle", "Phobos")]
dependent_variables_to_save_Ma = [
    propagation_setup.dependent_variable.rotation_matrix_to_body_fixed_frame("Mars"),
    propagation_setup.dependent_variable.longitude("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.latitude("Vehicle", "Phobos"),
    propagation_setup.dependent_variable.relative_position("Vehicle", "Mars"),
    propagation_setup.dependent_variable.relative_velocity("Vehicle", "Mars")]


bodies = ss.get_bodies(frame_origin, frame_orientation,
                           r_Ph, mu_Ph, mu_Ma, r_Ma, mu_De,
                           Phobos_spice=True, model="HM")
vehicle_acceleration_models = ss.get_vehicle_acceleration_models(bodies, multi_arc_central_bodies,
            dmax_Ma, dmax_Ph, thrust_mid_times, delta_v_values, maneuver_rise_time, total_maneuver_time)
multi_arc_stepsize = 75
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
        start_epoch, multi_arc_stepsize, multi_arc_stepsize, multi_arc_stepsize,  1.0, 1.0)
termination_settings = ss.get_termination_settings(start_epoch, max_days = 7)
propagator_settings = propagation_setup.propagator.translational( multi_arc_central_bodies,
                        vehicle_acceleration_models, multi_arc_bodies_to_propagate, x0_inertial[0],
                        start_epoch + total_duration,
                      output_variables=dependent_variables_to_save_Ph if r_mean <= 22
                      else dependent_variables_to_save_Ma)
dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
bodies, integrator_settings, propagator_settings)
dependent_variables = dynamics_simulator.dependent_variable_history
states = dynamics_simulator.state_history
dvh_tm = dict()
dvh_interpolator = interpolators.create_one_dimensional_interpolator(dependent_variables,
            interpolators.lagrange_interpolation(8))
for epoch in output_times:
    dvh_tm[epoch] = dvh_interpolator.interpolate(epoch)
dvh_tm = np.vstack(list(dvh_tm.values()))
print('Computing Rotations')
r_F, r_F_prt = np.empty((2, len(dvh_tm), 3))
R_LNOF2GRF = np.empty((len(dvh_tm), 3, 3))
for epoch, d in enumerate(dvh_tm):
    # Compute spacecraft position in Attractor-fixed frame
    R_Inrt2F = d[0:9].reshape(3, 3)
    r_F[epoch] = np.matmul(R_Inrt2F, d[11:14])
    r_F[epoch][2] = r_F[epoch][2] * (-1)
    if r_mean <= 22:
        r_F_prt[epoch] = np.matmul(R_Inrt2F, d[11:14] + pfe[epoch,6:9])
    else:
        r_F_prt[epoch] = np.matmul(R_Inrt2F, d[11:14] + pfe[epoch,0:3] + pfe[epoch,6:9])
    r_F_prt[epoch][2] = r_F_prt[epoch][2] * (-1)
    # Rotation from LNOF --> Phobos-Fixed --> Inertial --> GRF (z switches sign!!!)
    R_LNOF2F = frames.lv_to_body_fixed_rotation_matrix(d[9], d[10])
    # rotate 90 positive around z, 90  negative around x
    R_Inrt2GRF = np.matmul(R_RSW2GRF, frames.inertial_to_rsw_rotation_matrix(d[11:17]))
    # misal[epoch] = np.arccos( np.dot(R_Inrt2GRF, d[11:14]/np.linalg.norm(d[11:14]) ))
    R_LNOF2GRF[epoch] = np.matmul(R_Inrt2GRF, np.matmul(np.transpose(R_Inrt2F), R_LNOF2F))

if r_mean <= 22:
    [Axx, Ayy, Azz] = gs.compute_design(r_F[5:-10], mu_Ph, r_Ph, dmin, dmax)
    [Axx, Ayy, Azz] = gs.subtract_central_field(Axx, Ayy, Azz, mu_Ph, r_Ph, r_F[5:-10])
    [Axx, Ayy, Azz] = gs.rotate_design(Axx, Ayy, Azz, R_LNOF2GRF[5:-10])
else:
    [Axx, Ayy, Azz] = gs.compute_design(r_F[5:-10], mu_Ma, r_Ma, dmin, dmax)
    [Axx, Ayy, Azz] = gs.subtract_central_field(Axx, Ayy, Azz, mu_Ma, r_Ma, r_F[5:-10])
    [Axx, Ayy, Azz] = gs.rotate_design(Axx, Ayy, Azz, R_LNOF2GRF[5:-10])

V_noise_cfsMastd = np.zeros((3, len(Axx)))
if r_mean > 22:
    for i in range(len(Axx)):
        A = np.array([Axx[i],Ayy[i],Azz[i]])
        V_noise_cfsMastd[:,i] = np.sqrt(np.diagonal(np.matmul(
            A, np.matmul(cov_cfsMa, A.T ) ) ) )
np.save('V_noise_cfsMastd.npy', V_noise_cfsMastd )

if r_mean <= 22:
    [Axx_prt, Ayy_prt, Azz_prt] = gs.compute_design(r_F_prt[5:-10], mu_Ph, r_Ph, dmin, dmax)
    [Axx_prt, Ayy_prt, Azz_prt] = gs.subtract_central_field(Axx_prt, Ayy_prt, Azz_prt, mu_Ph, r_Ph, r_F_prt[5:-10])
    [Axx_prt, Ayy_prt, Azz_prt] = gs.rotate_design(Axx_prt, Ayy_prt, Azz_prt, R_LNOF2GRF[5:-10])
else:
    [Axx_prt, Ayy_prt, Azz_prt] = gs.compute_design(r_F_prt[5:-10], mu_Ma, r_Ma, dmin, dmax)
    [Axx_prt, Ayy_prt, Azz_prt] = gs.subtract_central_field(Axx_prt, Ayy_prt, Azz_prt, mu_Ma, r_Ma, r_F_prt[5:-10])
    [Axx_prt, Ayy_prt, Azz_prt] = gs.rotate_design(Axx_prt, Ayy_prt, Azz_prt, R_LNOF2GRF[5:-10])

V_noise_pos = np.zeros((3, len(Axx) ))
for i in range(len(Axx)):
    A = np.vstack((Axx[i]-Axx_prt[i],
                   Ayy[i]-Ayy_prt[i],
                   Azz[i]-Azz_prt[i]))
    a = np.zeros((3))
    for cf in range(1, len(cfs)):
        a += np.abs(A[:,cf] * cfs[cf])**2
    V_noise_pos[:,i] = np.sqrt(a)

np.save('V_noise_pos.npy', V_noise_pos)
np.save('r_F.npy', r_F[5:-10])
np.save('r_F_prt.npy', r_F_prt[5:-10])
np.save('output_times.npy', (output_times[5:-10]-start_epoch)/86400)

