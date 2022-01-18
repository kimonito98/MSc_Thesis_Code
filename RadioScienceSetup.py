# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 19:13:51 2021

@author: Michael

Radio Science Functions
"""

import os
if os.getcwd() == 'C:\\Users\\Michael':
    os.chdir('PyPhobos')
import numpy as np
from math import pi, tan

# Important: uncomment these if you are using the GENERIC tudatpy
# from tudatpy.kernel.simulation import estimation_setup
# from tudatpy.kernel.simulation import environment_setup
# from tudatpy.kernel.astro import conversion, observations
# from tudatpy.kernel.simulation.estimation_setup import observation_setup

# Important: uncomment these if you are using the mike_thesis tudatpy with the latest features
from kernel.simulation import environment_setup
from kernel.astro import conversion, observations
from kernel.simulation.estimation_setup import observation_setup
from kernel.simulation import estimation_setup


def add_ground_stations( bodies, n_landmarks, target_body = "Phobos" ):
    # New Norcia
    environment_setup.add_ground_station(
        bodies.get_body('Earth'),'Station1',[0.0,-0.54175,2.0278],conversion.geodetic_position_type)
    # Madrid
    environment_setup.add_ground_station(
        bodies.get_body('Earth'),'Station2',[0.0,0.70562,-0.07416],conversion.geodetic_position_type)
    # Goldstone
    environment_setup.add_ground_station(
        bodies.get_body('Earth'),'Station3',[0.0,0.61831,-2.04011],conversion.geodetic_position_type)
    # Landmarks
    lats = np.random.uniform(-2*pi/6, 2*pi/6, n_landmarks)
    lons = np.linspace(-pi, pi, n_landmarks)
    for i in range(n_landmarks):
        environment_setup.add_ground_station(
            bodies.get_body(target_body),'Landmark'+str(i+1),[0.0,lats[i],lons[i]],conversion.geodetic_position_type)
 

def create_link_ends( bodies, n_GS, n_lnd, target_body = "Phobos" ):

    ground_station_list = environment_setup.get_ground_station_list( bodies.get_body( "Earth" ) )[0:n_GS]
    transmitter_link_ends = observation_setup.one_way_uplink_link_ends( ground_station_list, ("Vehicle", "" ) )

    landmark_list = environment_setup.get_ground_station_list( bodies.get_body( target_body ) )[0:n_lnd]
    optical_link_ends = observation_setup.one_way_uplink_link_ends( landmark_list, ("Vehicle", "" ) )

    # Set (semi-random) combination of link ends for obsevables
    link_ends_per_observable = dict()
    link_ends_per_observable[ observation_setup.one_way_range_type ] = transmitter_link_ends
    link_ends_per_observable[ observation_setup.one_way_doppler_type ] = transmitter_link_ends
    link_ends_per_observable[ observation_setup.angular_position_type ] = transmitter_link_ends + optical_link_ends

    return link_ends_per_observable

def create_observation_settings( link_ends_per_observable,
                                 range_bias,
                                 vlbi_bias,
                                 optical_bias,
                                 target_body = "Phobos"):

    perturbing_bodies = ["Sun"]
    # Create observation settings for each link/observable
    observation_settings_list = list()
    # Create light time correction and bias settings
    light_time_correction_settings = [
        observation_setup.first_order_relativistic_light_time_correction( perturbing_bodies )]
    range_bias_settings = observation_setup.bias(np.array([range_bias]))
    vlbi_bias_settings = observation_setup.bias(np.array([vlbi_bias, vlbi_bias]))
    optical_bias_settings = observation_setup.bias(np.array([optical_bias, optical_bias]))

    # Iterate over observables
    for observable_key, link_ends_list in link_ends_per_observable.items():
        # Iterate over all link ends
        for link_ends in link_ends_list:
            if observable_key == observation_setup.one_way_range_type:
                observation_settings_list.append( observation_setup.one_way_range(
                    link_ends, light_time_correction_settings, range_bias_settings ) )
            if observable_key == observation_setup.one_way_doppler_type:
                observation_settings_list.append( observation_setup.one_way_open_loop_doppler(
                    link_ends, light_time_correction_settings))
                # observation_settings_list.append( observation_setup.one_way_closed_loop_doppler(
                #     link_ends, t_doppler, light_time_correction_settings, bias_settings = None ) )
            if (observable_key == observation_setup.angular_position_type):
                for i in link_ends.values():
                    if 'Earth' in i:
                        observation_settings_list.append( observation_setup.angular_position(
                            link_ends, light_time_correction_settings, vlbi_bias_settings))
                        break
                    elif target_body in i:
                        observation_settings_list.append( observation_setup.angular_position(
                            link_ends, [], optical_bias_settings))
                        break

    return observation_settings_list

def get_estimation_apriori(names, counts, apr_vals, n_arcs, n_lnd):
    """ Retireve the ordered index of the estimated parameters
     and setup the a-priori matrix"""
    idxs = dict()
    apr = np.zeros((sum(counts)))
    idx = 0
    for n, c, a in zip(names, counts, apr_vals):
        if n in ('r_sc','v_sc'):
            idx2 = idx
            for na in range(n_arcs):
                idxs[n + str(na)] = (idx2, idx2 + 3)
                apr[idx2:idx2 + 3] = np.ones((3))*a
                idx2 += 6
            if n == 'r_sc':
                idx = idx + 3
            else:
                idx = idx2 - 3
        elif n == 'cfs':
            idxs[n] = (idx, idx+c)
            apr[idx:idx+c] = a
            idx += c
        elif n == 'lnd':
            apr[idx:idx + c] = np.ones((c)) * a
            for lnd in range(n_lnd):
                idxs[n + str(lnd)] = (idx, idx + 3)
                idx += 3
        else:
            idxs[n] = (idx, idx + c)
            apr[idx:idx+c] = np.ones((c))*a
            idx += c
    return np.linalg.inv(np.eye(sum(counts)) * apr**2), idxs, sum(counts)

def create_estimated_parameters( multi_arc_propagator_settings, arc_initial_times,
                                 single_arc_propagator_settings,
                                 bodies,
                                 cos_tup, sin_tup,
                                 link_ends_per_observable,
                                 n_GS,
                                 n_lnd):
    parameter_settings = estimation_setup.parameter.initial_states(single_arc_propagator_settings, bodies)

    parameter_settings.append( # 6 -->47
        estimation_setup.parameter.multi_arc_initial_states( multi_arc_propagator_settings,
                                                         bodies, arc_initial_times)[0])

    parameter_settings.append( # 48
        estimation_setup.parameter.radiation_pressure_coefficient("Vehicle"))
    parameter_settings.append( # 49
        estimation_setup.parameter.gravitational_parameter("Phobos"))
    parameter_settings.append( #50 --> 70
        estimation_setup.parameter.desaturation_delta_v_values("Vehicle"))

    parameter_settings.append( # 71-->79
        estimation_setup.parameter.spherical_harmonics_c_coefficients("Phobos", cos_tup[0], cos_tup[1], cos_tup[2], cos_tup[3]))
    parameter_settings.append( # 80-->85
        estimation_setup.parameter.spherical_harmonics_s_coefficients("Phobos", sin_tup[0], sin_tup[1], sin_tup[2], sin_tup[3]))
    parameter_settings.append( # 86-->88
        estimation_setup.parameter.constant_empirical_acceleration_terms("Vehicle", "Phobos"))

    for i in range(n_GS):
        parameter_settings.append(
            estimation_setup.parameter.ground_station_position("Earth", "Station"+str(i+1) ) )
    for i in range(n_GS):
        parameter_settings.append(
            estimation_setup.parameter.observation_bias(link_ends_per_observable[observation_setup.one_way_range_type][i],
                                                        observation_setup.one_way_range_type))
    for i in range(n_GS): # 98 --> 100
        parameter_settings.append(
            estimation_setup.parameter.observation_bias(link_ends_per_observable[observation_setup.angular_position_type][i],
                                                        observation_setup.angular_position_type))
    for i in range(n_lnd): # 98 --> 100
        parameter_settings.append(
            estimation_setup.parameter.observation_bias(link_ends_per_observable[observation_setup.angular_position_type][i+n_GS],
                                                        observation_setup.angular_position_type))
    for i in range(n_lnd):
        parameter_settings.append( # 101--> 130
            estimation_setup.parameter.ground_station_position("Phobos", "Landmark"+str(i+1) ) )

    return estimation_setup.create_parameters_to_estimate( parameter_settings, bodies,
                                                           multi_arc_propagator_settings.single_arc_propagator_settings[0] )

def simulate_observations(link_ends_per_observable,
                          observation_simulators,
                          bodies,
                          arc_initial_times, buffer,
                          arc_duration, percentage_arc_tracked,
                          t_range, t_doppler, t_vlbi, t_optical,
                          n_GS, range_noise, doppler_noise, vlbi_noise, optical_noise,
                          range = True, vlbi = True, optical = True, target_body = "Phobos"):

    # visibility conditions
    ground_station_elevation_angle = np.deg2rad(15)
    landmark_elevation_angle = np.deg2rad(15)
    earth_sun_avoidance_angle = np.deg2rad(30)
    occulting_body = "Earth"

    # Simulate observations
    # note that 1h buffer = manoeuvre time (coincidence)
    range_times, doppler_times, vlbi_times, optical_times = np.empty((4,0), dtype = float)
    for t in arc_initial_times:
        range_times = np.append(range_times, np.arange(t+buffer, t+arc_duration*percentage_arc_tracked- buffer, t_range))
        doppler_times = np.append(doppler_times, np.arange(t + buffer, t + arc_duration * percentage_arc_tracked- buffer, t_doppler))
        vlbi_times = np.append(vlbi_times, np.arange(t + buffer, t + arc_duration * percentage_arc_tracked - buffer, t_vlbi))
        optical_times = np.append(optical_times, np.arange(t, t + arc_duration, t_optical))
    # For Bennu, optical data is collected on the 1st and 9th day
    # optical_times = np.concatenate((np.arange(t+buffer,t+buffer+86400, t_optical),np.arange(t+8*86400+buffer,t+buffer+9*86400, t_optical)))
    observation_simulation_settings = list()
    if range == True:
        rangedict = dict()
        rangedict[observation_setup.one_way_range_type] = link_ends_per_observable[observation_setup.one_way_range_type]
        observation_simulation_settings += observation_setup.create_tabulated_simulation_settings(
            rangedict, range_times)
    dopplerdict = dict()
    dopplerdict[observation_setup.one_way_doppler_type] = link_ends_per_observable[observation_setup.one_way_doppler_type]
    observation_simulation_settings += observation_setup.create_tabulated_simulation_settings(
        dopplerdict, doppler_times)
    if vlbi == True:
        vlbidict = dict()
        vlbidict[observation_setup.angular_position_type] = link_ends_per_observable[observation_setup.angular_position_type][0:n_GS]
        observation_simulation_settings += observation_setup.create_tabulated_simulation_settings(
            vlbidict, vlbi_times)
    if optical == True:
        opticaldict = dict()
        opticaldict[observation_setup.angular_position_type] = link_ends_per_observable[observation_setup.angular_position_type][n_GS:]
        observation_simulation_settings += observation_setup.create_tabulated_simulation_settings(
            opticaldict, optical_times)

    ground_station_list = environment_setup.get_ground_station_list(bodies.get_body("Earth"))
    viability_settings_list = observation_setup.elevation_angle_viability(ground_station_list, ground_station_elevation_angle)
    viability_settings_list += observation_setup.body_avoidance_viability(ground_station_list, "Sun", earth_sun_avoidance_angle)
    viability_settings_list += observation_setup.body_occultation_viability(ground_station_list, occulting_body)
    observation_setup.add_viability_check_to_settings(observation_simulation_settings, viability_settings_list)

    landmark_list = environment_setup.get_ground_station_list(bodies.get_body(target_body))
    viability_settings_list = observation_setup.elevation_angle_viability(landmark_list, landmark_elevation_angle)
    observation_setup.add_viability_check_to_settings(observation_simulation_settings, viability_settings_list)

    observation_noise_amplitudes = dict()
    observation_noise_amplitudes[observation_setup.one_way_range_type] = range_noise
    observation_noise_amplitudes[observation_setup.one_way_doppler_type] = doppler_noise
    observation_noise_amplitudes[observation_setup.angular_position_type] = vlbi_noise

    observation_setup.add_gaussian_noise_to_settings(
        observation_simulation_settings,
        observation_noise_amplitudes[observation_setup.one_way_range_type],
        observation_setup.one_way_range_type)
    observation_setup.add_gaussian_noise_to_settings(
        observation_simulation_settings,
        observation_noise_amplitudes[observation_setup.one_way_doppler_type],
        observation_setup.one_way_doppler_type)
    observation_setup.add_gaussian_noise_to_settings(
        observation_simulation_settings,
        observation_noise_amplitudes[observation_setup.angular_position_type],
        observation_setup.angular_position_type)

    # Simulate required observation
    return observation_setup.simulate_observations(
        observation_simulation_settings, observation_simulators, bodies)


def check_surface_coverage(lats, lons, dist, fov, r_Ph):
    """ CHeck the percetage of the moon which can be observed given following camera FoV.
    Note that the surface is assumed flat as viewed by the camera, in order to compute coverage.
    Give all inputs in radians and km"""
    surface = np.zeros((360, 180))
    
    lats = np.round(np.rad2deg(lats + pi/2))
    lons = np.round(np.rad2deg(lons + pi))

    for c, (lat, lon, d) in enumerate(zip(lats, lons, dist)):
        # compute side of rectagle as viewed by the camera
        side_km = tan(fov) * (d - r_Ph)
        side_deg = round(side_km/(2*pi*r_Ph) * 360)
        # mark the area seen by the camera with "1"
        surface[int(lon)-side_deg:int(lon)+side_deg,int(lat)-side_deg:int(lat)+side_deg] = 1
   
    # compute percentages of covered surface
    return np.count_nonzero(surface)/(360*180)



def get_estimation_input( simulated_observations, number_of_parameters, 
                            range_noise, doppler_noise, vlbi_noise, optical_noise,
                            link_ends_per_observable,
                            inverse_apriori_covariance = np.array([]),
                          target_body = "Phobos"):
    # perturb the sc initial state
    initial_parameter_deviation = np.zeros( number_of_parameters )
    initial_parameter_deviation[ 6 ] = 1.0
    initial_parameter_deviation[ 7 ] = 1.0
    initial_parameter_deviation[ 8 ] = 1.0


    pod_input = estimation_setup.PodInput(simulated_observations, number_of_parameters,
                                          inverse_apriori_covariance=inverse_apriori_covariance,
                                          apriori_parameter_correction=initial_parameter_deviation)
    pod_input.define_estimation_settings( reintegrate_variational_equations = False, save_state_history_per_iteration = True )

    for observable_key, link_ends_list in link_ends_per_observable.items():
        if observable_key == observation_setup.one_way_range_type:
            pod_input.set_constant_weight_per_observable_and_link_end(
                observable_key, link_ends_list, 1.0 / (range_noise ** 2))
        if observable_key == observation_setup.one_way_doppler_type:
            pod_input.set_constant_weight_per_observable_and_link_end(
                observable_key, link_ends_list, 1.0 / (doppler_noise ** 2))
        found = False
        if observable_key == observation_setup.angular_position_type:
            for c, link_ends in enumerate(link_ends_list):
                for i in link_ends.values():
                    if target_body in i and found == False:
                        vlbi_links = link_ends_list[0:c]
                        optical_links = link_ends_list[c:]
                        found = True
            pod_input.set_constant_weight_per_observable_and_link_end(
                observable_key, vlbi_links, 1.0 / (vlbi_noise ** 2))
            pod_input.set_constant_weight_per_observable_and_link_end(
                observable_key, optical_links, 1.0 / (optical_noise ** 2))

    return pod_input

def check_earth_sun_avoidance_angle(r_Earth_Sat, r_Earth_Sun):
    angle = np.zeros((len(r_Earth_Sat)))
    for i in range(len(angle)):
        angle[i] = np.arccos( np.dot(r_Earth_Sat[i], r_Earth_Sun[i])/
                             (np.linalg.norm(r_Earth_Sat[i])*np.linalg.norm(r_Earth_Sun)) )
    return angle


def reset_coeffs(c_tup, s_tup, cfs_dict):
    cfs = np.empty( (0), dtype=float)
    for l in range(c_tup[0], c_tup[2] + 1):
        m0 = c_tup[1] if l == c_tup[0] else 0
        for m in range(m0, l + 1):
            cfs = np.append(cfs, cfs_dict['C' + str(l) + ',' + str(m)])
            if (l == c_tup[2]) and (m == c_tup[3]):
                break
    for l in range(s_tup[0], s_tup[2] + 1):
        m0 = s_tup[1] if l == s_tup[0] else 0
        for m in range(s_tup[1], l + 1):
            cfs = np.append(cfs, cfs_dict['S' + str(l) + ',' + str(m)])
            if (l == s_tup[2]) and (m == s_tup[3]):
                break
    return cfs

def get_rs_idxs(c_tup, s_tup):
    idxs = dict()
    c = 0
    for l in range(c_tup[0], c_tup[2] + 1):
        m0 = c_tup[1] if c == 0 else 0
        for m in range(m0, l + 1):
            idxs['C' + str(l) + ',' + str(m)] = c
            c += 1
            if (l == c_tup[2]) and (m == c_tup[3]):
                break
    for l in range(s_tup[0], s_tup[2] + 1):
        for m in range(s_tup[1], l + 1):
            idxs['S' + str(l) + ',' + str(m)] = c
            c += 1
            if (l == s_tup[2]) and (m == s_tup[3]):
                break
    return idxs

def reshape_covariance(cov_rs, rs_idxs, cg_idxs):
    """ Reshape the covariance matrix from rs indexed to cg indexed (or viceversa!!!)"""
    cov = np.zeros((len(cg_idxs), len(cg_idxs)))
    for i in range(len(cov_rs)):
        for j in range(len(cov_rs)):
            ii, jj = cg_idxs[rs_idxs[i]], cg_idxs[rs_idxs[j]]
            cov[ii, jj] = cov_rs[i, j]
    return cov

def reshape_cg_covariance(cov_cg, rs_idxs, cg_idxs):
    cov = np.zeros((len(cg_idxs), len(cg_idxs)))
    for i in range(len(cov_cg)):
        for j in range(len(cov_cg)):
            ii, jj = rs_idxs[cg_idxs[i]], rs_idxs[cg_idxs[j]]
            cov[ii, jj] = cov_cg[i, j]
    return cov

def order_rs_err(err, c_tup, s_tup):
    Err_dict = dict()
    c = 0
    for l in range(c_tup[0], c_tup[2] + 1):
        if c == 0:
            m0 = c_tup[1]
        else:
            m0 = 0
        for m in range( m0, l + 1 ):
            Err_dict['C' + str(l) + ',' + str(m)] = err[c]
            c += 1
            if (l == c_tup[2]) and (m == c_tup[3]):
                break
    for l in range(s_tup[0], s_tup[2] + 1):
        for m in range( s_tup[1], l + 1 ):
            Err_dict['S' + str(l) + ',' + str(m)] = err[c]
            c += 1
            if (l == s_tup[2]) and (m == s_tup[3]):
                break
    assert c == len(err)
    return Err_dict


def err_stats(err_dict, coeff_dict, normalise = True, ignore_S2X=True, rms_only=True):
    """ Normalise the formal error by the respective coefficient,
        but group by degree, calculate RMS & max/min deviation from rms """

    max_degree = max([int(d.split(',')[0][1:]) for d in list(err_dict.keys())])
    err = np.empty((0, 3))

    if normalise:
        for c, l in enumerate(range(max_degree + 1)):
            lst = []
            for m in range(l + 1):
                if 'C' + str(l) + ',' + str(m) in err_dict:
                    if ignore_S2X and not (l == 2 and m == 1):
                        lst.append(err_dict['C' + str(l) + ',' + str(m)] / coeff_dict['C' + str(l) + ',' + str(m)])
                    elif not ignore_S2X:
                        lst.append(err_dict['C' + str(l) + ',' + str(m)] / coeff_dict['C' + str(l) + ',' + str(m)])
                if 'S' + str(l) + ',' + str(m) in err_dict:
                    if ignore_S2X and l != 2:
                        lst.append(err_dict['S' + str(l) + ',' + str(m)] / coeff_dict['S' + str(l) + ',' + str(m)])
                    elif not ignore_S2X:
                        lst.append(err_dict['S' + str(l) + ',' + str(m)] / coeff_dict['S' + str(l) + ',' + str(m)])
            if lst:
                rms = np.sqrt(np.mean(np.asarray(lst) ** 2))
                # err_rms = np.append(err_rms, rms)
                # err_std = np.append(err_std, np.std(np.asarray(lst)))
                err = np.append(err, np.array([ rms, rms - np.min(np.abs(np.asarray(lst))),
                        rms + np.max(np.abs(np.asarray(lst))) ]).reshape(1,3), axis = 0)
    else:
        for c, l in enumerate(range(max_degree + 1)):
            lst = []
            for m in range(l + 1):
                if 'C' + str(l) + ',' + str(m) in err_dict:
                    if ignore_S2X and not (l == 2 and m == 1):
                        lst.append(err_dict['C' + str(l) + ',' + str(m)])
                    elif not ignore_S2X:
                        lst.append(err_dict['C' + str(l) + ',' + str(m)])
                if 'S' + str(l) + ',' + str(m) in err_dict:
                    if ignore_S2X and l != 2:
                        lst.append(err_dict['S' + str(l) + ',' + str(m)])
                    elif not ignore_S2X:
                        lst.append(err_dict['S' + str(l) + ',' + str(m)] )
            if lst:
                rms = np.sqrt(np.mean(np.asarray(lst) ** 2))
                # err_rms = np.append(err_rms, rms)
                # err_std = np.append(err_std, np.std(np.asarray(lst)))
                err = np.append(err, np.array([ rms, rms - np.min(np.abs(np.asarray(lst))),
                        rms + np.max(np.abs(np.asarray(lst))) ]).reshape(1,3), axis = 0)
    if rms_only:
        return err[:,0]
    else:
        return err

def err_rms(err_dict):
    """ Compute RMS per degree """
    max_degree = max([int(d.split(',')[0][1:]) for d in list(err_dict.keys())])
    err_rms = np.empty((0))
    for c, l in enumerate(range(max_degree + 1)):
        lst = []
        for m in range(l + 1):
            if 'C' + str(l) + ',' + str(m) in err_dict:
                lst.append(err_dict['C' + str(l) + ',' + str(m)])
            if 'S' + str(l) + ',' + str(m) in err_dict:
                lst.append(err_dict['S' + str(l) + ',' + str(m)])
        if lst:
            rms = np.sqrt(np.mean(np.asarray(lst) ** 2))
            err_rms = np.append(err_rms, rms)
    return err_rms

