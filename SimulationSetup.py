# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 15:34:09 2021

@author: Michael
"""


import os
if os.getcwd() == 'C:\\Users\\Michael':
    os.chdir('PyPhobos')
import numpy as np

# Important: uncomment these if you are using the GENERIC tudatpy
# from tudatpy.kernel.simulation import environment_setup
# from tudatpy.kernel.simulation import propagation_setup
# from tudatpy.kernel.simulation import estimation_setup
# from tudatpy.kernel.interface import spice_interface

# Important: uncomment these if you are using the mike_thesis tudatpy with the latest features
from kernel.simulation import environment_setup
from kernel.simulation import propagation_setup
from kernel.simulation import estimation_setup
from kernel.interface import spice_interface




from matplotlib import pyplot as plt


def get_bodies(frame_origin,
               frame_orientation,
               r_Ph, mu_Ph,
               mu_Ma, r_Ma,
               mu_De,
               Phobos_spice = True,
               model = "HM"):
    """ Setup system of bodies and properties """

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
    cos_Ph, sin_Ph = read_Phobos_gravity_field(model)[0:2]
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
    occulting_bodies = [ "Phobos" ]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
    )
    environment_setup.add_radiation_pressure_interface(
                bodies, 'Vehicle' , radiation_pressure_settings )

    return bodies

def read_Phobos_gravity_field(model = "HM"):
    """ HM = Homogeneous, others are heterogeneous models, drawn from a distribution defined in:
        Signature of Phobos’ interior structure in its gravity field and libration 
        https://doi.org/10.1016/j.icarus.2018.11.022 """
    max_degree = 10
    rel_path = "PhobosCubes_ForTUDelft/"
    cos, sin = np.zeros( (2, max_degree+1, max_degree+1) )
    cfs = dict()        
    count = 0
    factor = (11.1/14)**2 
    models = ['HM','DR','HF','IED','ISC','PC','RP']
    files = ['PhobosGravity_Cubes500m_deg10_homogen',
             'PhobosGravity_Cubes500m_500models_all_deg10_disrupted',
             'PhobosGravity_Cubes500m_500models_all_deg10_fractured_E1b',
             'PhobosGravity_Cubes500m_500models_all_deg10_icy_shape3',
             'PhobosGravity_Cubes500m_500models_all_deg10_icy_surfshape',
             'PhobosGravity_Cubes500m_500models_all_deg10_porous_E2c',
             'PhobosGravity_Cubes500m_500models_all_deg10_rubble']
    file_cos = np.loadtxt(rel_path + files[models.index(model)] + '.hscos')
    file_sin = np.loadtxt(rel_path + files[models.index(model)] + '.hssin')
    
    if model == 'HM':
        i = 0
    else:
        i = np.random.randint(0, len(file_cos))
        file_cos = file_cos[i]
        file_sin = file_sin[i]

    for d in range(max_degree+1):
        for o in range(d+1):
            cos[d,o] = file_cos[count]*factor
            sin[d,o] = file_sin[count]*factor
            cfs['C' + str(d) + "," + str(o)] = file_cos[count]*factor
            if o != 0:
                cfs['S' + str(d) + "," + str(o)] = file_sin[count]*factor
            count += 1
    
    # the radius conversion factor does not apply here   
    cos[0,0] = 1.
    cfs['C0,0'] = 1.

    return (cos, sin, cfs)
    

def read_Mars_gravity_field(text_file, max_degree,  std = False, delimiter = ',' ):
    """ Read Mars gravity field terms from Genova 2016 (tudat default is good, so no need"""
    deg_idx, ord_idx, cos_idx, sin_idx =  0, 1, 2, 3
    cos, sin = np.zeros( (2, max_degree+1, max_degree+1) )
    cfs = dict()
    for i in np.loadtxt(text_file, delimiter=delimiter):

        if int(i[deg_idx]) > max_degree:
            break
        cos[int(i[deg_idx]),int(i[ord_idx])] = i[cos_idx]
        sin[int(i[deg_idx]),int(i[ord_idx])] = i[sin_idx]

        if std == False:
            cfs['C' + str(int(i[deg_idx])) +','+ str(int(i[ord_idx]))] = i[cos_idx]
            cfs['S' + str(int(i[deg_idx])) +','+ str(int(i[ord_idx]))] = i[sin_idx]
        else:
            cfs['C' + str(int(i[deg_idx])) +','+ str(int(i[ord_idx]))] = i[cos_idx+2]
            cfs['S' + str(int(i[deg_idx])) +','+ str(int(i[ord_idx]))] = i[sin_idx+2]
    return (cos, sin, cfs)

def read_Earth_gravity_field(text_file, skiprows, usecols, max_degree, std = False):
    deg_idx, ord_idx, cos_idx, sin_idx =  0,1,2,3
    cos, sin = np.zeros( (2, max_degree+1, max_degree+1) )
    cfs = dict()
    for i in np.loadtxt(text_file, skiprows = skiprows, usecols = usecols):
        if i[deg_idx] <= max_degree:
            cos[int(i[deg_idx]), int(i[ord_idx])] = i[cos_idx]
            sin[int(i[deg_idx]), int(i[ord_idx])] = i[sin_idx]

        if std == False:
            cfs['C' + str(int(i[deg_idx])) +','+ str(int(i[ord_idx]))] = i[cos_idx]
            cfs['S' + str(int(i[deg_idx])) +','+ str(int(i[ord_idx]))] = i[sin_idx]
        else:
            cfs['C' + str(int(i[deg_idx])) +','+ str(int(i[ord_idx]))] = i[cos_idx+2]
            cfs['S' + str(int(i[deg_idx])) +','+ str(int(i[ord_idx]))] = i[sin_idx+2]
    return (cos, sin, cfs)

def create_sensitivity_parameters( propagator_settings, bodies):
    """ Parameters for sensitivity analysis """
    parameter_settings = estimation_setup.parameter.initial_states( propagator_settings, bodies )
    # parameter_settings.append( estimation_setup.parameter.gravitational_parameter( "Mars" ) )
    # parameter_settings.append( 
    #         estimation_setup.parameter.spherical_harmonics_c_coefficients("Mars", 1, 0, 10, 10))
    # parameter_settings.append( 
    #         estimation_setup.parameter.spherical_harmonics_s_coefficients("Mars", 1, 1, 10, 10))
    parameter_settings.append( estimation_setup.parameter.gravitational_parameter( "Phobos" ) )
    parameter_settings.append( 
            estimation_setup.parameter.spherical_harmonics_c_coefficients("Phobos", 1, 0, 10, 10))
    parameter_settings.append( 
            estimation_setup.parameter.spherical_harmonics_s_coefficients("Phobos", 1, 1, 10, 10))


    return estimation_setup.create_parameters_to_estimate( parameter_settings, bodies )


def compute_stats_HT_Phobos(max_degree = 10, both = False, max_only = False):
    """ Return the STD of the HT Phfill_bestobos models, to be used as a-priori constraints.
    If both == True, also return the MEAN of the HT models, normalised by the HM values, grouped per degree.
    if max_only == True, return the maximum value of the std
    """
    
    labels = ['DR','HF','IED','ISC','PC','RP']
    rel_path = "PhobosCubes_ForTUDelft/"
    files = ['PhobosGravity_Cubes500m_500models_all_deg10_disrupted',
             'PhobosGravity_Cubes500m_500models_all_deg10_fractured_E1b',
             'PhobosGravity_Cubes500m_500models_all_deg10_icy_shape3',
             'PhobosGravity_Cubes500m_500models_all_deg10_icy_surfshape',
             'PhobosGravity_Cubes500m_500models_all_deg10_porous_E2c',
             'PhobosGravity_Cubes500m_500models_all_deg10_rubble']

    if both:
        c_stats, s_stats = np.zeros( (2, 2, 6, max_degree ) )
    
    else:
        n_cfs = (max_degree+1)**2 - 1
        s_stats = np.zeros(( 6, int((n_cfs-max_degree)/2) ))
        c_stats = np.zeros(( 6, n_cfs - int((n_cfs-max_degree)/2) ))
        
    for mod, i in enumerate(labels):
            
        file_cos = np.loadtxt(rel_path + files[labels.index(i)] + '.hscos')
        file_sin = np.loadtxt(rel_path + files[labels.index(i)] + '.hssin')
        
        if both:
            hm_cos = np.loadtxt(rel_path + 'PhobosGravity_Cubes500m_deg10_homogen' + '.hscos')
            hm_sin = np.loadtxt(rel_path + 'PhobosGravity_Cubes500m_deg10_homogen' + '.hssin')
        
            c = 1
            for d in range(1, max_degree+1):
                c_mean, s_mean, c_std, s_std = [], [], [], []
                for o in range(d+1):
                    if not (d==2 and o==1):
                        c_mean.append(np.mean(file_cos[:,c])/hm_cos[c])
                        c_std.append(np.std(file_cos[:,c]))
                    if o != 0:
                        s_mean.append(np.mean(file_sin[:,c])/hm_cos[c])
                        s_std.append(np.std(file_sin[:,c]))
                    c += 1
                c_stats[0,mod,d-1] = np.mean(np.abs(np.asarray(c_mean)))
                c_stats[1,mod,d-1] = np.mean(np.asarray(c_std))
                s_stats[0,mod,d-1] = np.mean(np.abs(np.asarray(s_mean)))
                s_stats[1,mod,d-1] = np.mean(np.asarray(s_std))
        else:
            cc, cs = 1, 1
            for d in range(1, max_degree+1):
                for o in range(d+1):
                    c_stats[mod,cc-1] = np.std(file_cos[:,cc])
                    if o != 0:
                        s_stats[mod,cs-1] = np.std(file_sin[:,cc])
                        cs += 1
                    cc += 1
                
    if max_only == True:
        c_stats = np.max(c_stats, axis=0)
        s_stats = np.max(s_stats, axis=0)

    return (c_stats, s_stats, labels)

def get_vehicle_acceleration_models(bodies,
                                 central_bodies,
                                 dmax_Ma,
                                 dmax_Ph,
                                 thrust_mid_times = None,
                                delta_v_values = None,
                                maneuver_rise_time = None,
                                total_maneuver_time = None ):
    """ Create acceleration models for the spacecraft """
    bodies_to_propagate = ["Vehicle"]
    
    constant_acceleration = np.ones(3)*1e-9
    sine_acceleration = np.ones(3)*1e-9
    cosine_acceleration = np.ones(3)*1e-9

    accelerations_settings_vehicle = dict(
        Mars = [
            propagation_setup.acceleration.spherical_harmonic_gravity(dmax_Ma,dmax_Ma)
            # propagation_setup.acceleration.point_mass_gravity()
        ],
        Phobos = [
            propagation_setup.acceleration.spherical_harmonic_gravity(dmax_Ph,dmax_Ph),
            # propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.empirical(
                constant_acceleration, sine_acceleration, cosine_acceleration )
        ],
        Sun = [
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Jupiter= [
            propagation_setup.acceleration.point_mass_gravity()
        ],
         Earth= [
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Deimos= [
            propagation_setup.acceleration.point_mass_gravity()
        ])
    if thrust_mid_times is not None:
        accelerations_settings_vehicle['Vehicle'] = [
            propagation_setup.acceleration.momentum_wheel_desaturation_acceleration(
                thrust_mid_times, delta_v_values, maneuver_rise_time, total_maneuver_time)
        ]
    acceleration_settings = {"Vehicle" : accelerations_settings_vehicle}
    
    return propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

def get_Phobos_acceleration_models(bodies,
                             central_bodies,
                             dmax_Ma, 
                             dmax_Ph):
    """ Create acceleration models for Phobos (careful: for mutual, homogeneous values from 
    Jacobson and Lainey 2014 are required) """

    bodies_to_propagate = ["Phobos"]

    accelerations_settings_Phobos = dict(
        Mars=[
            propagation_setup.acceleration.spherical_harmonic_gravity(dmax_Ma, dmax_Ma)
            # propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
            #     dmax_Ma, dmax_Ma, dmax_Ph, dmax_Ph  )
            # propagation_setup.acceleration.point_mass_gravity()
        ],
        Sun=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Deimos=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        # Jupiter=[
        #     propagation_setup.acceleration.point_mass_gravity()
        # ],
        # Earth=[
        #     propagation_setup.acceleration.point_mass_gravity()
        # ],
    )
    acceleration_settings = {"Phobos": accelerations_settings_Phobos}

    return propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

def get_termination_settings( start_epoch, max_days = None, end_epoch = None  ):
    """ Define termination conditions for simulation '"""
    
    termination_variable = propagation_setup.dependent_variable.relative_distance( "Vehicle", "Phobos" )
    crash = propagation_setup.propagator.dependent_variable_termination(
            dependent_variable_settings = termination_variable,
            limit_value = 14.0E3,
            use_as_lower_limit = True,
            terminate_exactly_on_final_condition = False)
    
    escape = propagation_setup.propagator.dependent_variable_termination(
            dependent_variable_settings = termination_variable,
            limit_value = 100.0E3,
            use_as_lower_limit = False,
            terminate_exactly_on_final_condition = False)

    if max_days is not None:
        timestop = propagation_setup.propagator.time_termination( start_epoch + max_days*86400,
                                                                 terminate_exactly_on_final_condition=True)
    else:
        timestop = propagation_setup.propagator.time_termination( end_epoch )
    termination_settings_list = [ crash, escape, timestop]
    termination_settings = propagation_setup.propagator.hybrid_termination( termination_settings_list, fulfill_single_condition = True )
    
    return termination_settings


def termination_longitude( lon ):
    """ Define termination conditions using the longitude (one revolution around Phobos) """
    loop = propagation_setup.propagator.dependent_variable_termination(
    dependent_variable_settings = propagation_setup.dependent_variable.longitude( "Vehicle", "Phobos" ),
    limit_value = lon,
    use_as_lower_limit = True,
    terminate_exactly_on_final_condition = False)
    
    escape = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings = propagation_setup.dependent_variable.relative_distance( "Vehicle", "Phobos" ),
        limit_value = 100.0E3,
        use_as_lower_limit = False,
        terminate_exactly_on_final_condition = False)
    
    crash = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings = propagation_setup.dependent_variable.relative_distance( "Vehicle", "Phobos" ),
        limit_value = 15.0E3,
        use_as_lower_limit = True,
        terminate_exactly_on_final_condition = False)
    termination_settings_list = [ crash, escape, loop]
    
    return propagation_setup.propagator.hybrid_termination( termination_settings_list, fulfill_single_condition = True )

def count_cfs(cos_tup, sin_tup ):
    """ Count the number of GF coefficients from the coefficient tuple input """
    n_c, n_s, c = 0, 0, 0
    for l in range(cos_tup[0], cos_tup[2] + 1):
        if c == 0:
            m0 = cos_tup[1]
        else:
            m0 = 0
        for m in range(m0, l + 1):
            n_c += 1
            if (l == cos_tup[2]) and (m == cos_tup[3]):
                break
    for l in range(sin_tup[0], sin_tup[2] + 1):
        for m in range(sin_tup[1], l + 1):
            n_s += 1
            if (l == sin_tup[2]) and (m == sin_tup[3]):
                break
    return n_c, n_s

def fill_best(err_dict, cfs_dict, dmax, normalise = True):
    """ Return a mesh for a contour plot with colorbar. Normalised?"""
    mesh = np.zeros((dmax,2*dmax+1))
    for k, v in err_dict.items():
        lst = k.split(',')
        d = int(lst[0][1:])
        o = int(lst[1])
        if lst[0][0] == 'C':
            if normalise:
                mesh[d - 1, dmax + o] = v / cfs_dict[k]
            else:
                mesh[d - 1, dmax + o] = v
        else:
            if normalise:
                mesh[d - 1, dmax - o] = v / cfs_dict[k]
            else:
                mesh[d - 1, dmax - o] = v
    return mesh


def drawSphere(xCenter, yCenter, zCenter, r):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

def set_lims(ax, minvals, maxvals, z = False):
    ax.set_xlim(minvals, maxvals)
    ax.set_ylim(minvals, maxvals)
    if z:
        ax.set_zlim(minvals, maxvals)

def is_pareto_efficient(costs, return_mask = True):
    """ Algorithm for non-dominated sorting of objective space """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def plot_keplerian_elements(kepler_elements, time):
    time_hours = [t / 86400 for t in time]
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 10))

    # Semi-major Axis
    semi_major_axis = [element / 1000 for element in kepler_elements[:, 0]]
    ax1.plot(time_hours, semi_major_axis)
    ax1.set_ylabel('Semi-major axis [km]')

    # Eccentricity
    eccentricity = kepler_elements[:, 1]
    ax2.plot(time_hours, eccentricity)
    ax2.set_ylabel('Eccentricity [-]')

    # Inclination
    inclination = [np.rad2deg(element) for element in kepler_elements[:, 2]]
    ax3.plot(time_hours, inclination)
    ax3.set_ylabel('Inclination [deg]')

    # Argument of Periapsis
    argument_of_periapsis = [np.rad2deg(element) for element in kepler_elements[:, 3]]
    ax4.plot(time_hours, argument_of_periapsis)
    ax4.set_ylabel('Argument of Periapsis [deg]')

    # Right Ascension of the Ascending Node
    raan = [np.rad2deg(element) for element in kepler_elements[:, 4]]
    ax5.plot(time_hours, raan)
    ax5.set_ylabel('RAAN [deg]')

    # True Anomaly
    true_anomaly = [np.rad2deg(element) for element in kepler_elements[:, 5]]
    ax6.scatter(time_hours, true_anomaly, s=1)
    ax6.set_ylabel('True Anomaly [deg]')
    ax6.set_yticks(np.arange(0, 361, step=60))
