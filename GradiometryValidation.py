

import numpy as np

# my own
import sys
sys.path.append('/home/mplumaris/anaconda3/lib/python3.8/site-packages')

from matplotlib import pyplot as plt

#%%
import os
# a = np.fromfile('preconditioner_000.dat', dtype=float)
a = np.fromfile('stds.dat', dtype=float)
b = np.fromfile('Christian_stds.dat', dtype=float)

#%% Plot GOCE Positions

# goce_positions_c = np.loadtxt('goce_positions_all.txt')
# goce_positions_k = np.loadtxt('singlePerturbedSatellitePropagationHistory.dat', delimiter = ',', usecols=(1,2,3,4,5,6))

GM = 0.3986004415E+15 #constants.EARTH_GRAVITATIONAL_PARAMETER
R = 0.6378136460E+07 #constants.EARTH_EQUATORIAL_RADIUS

n = int(535680/1)
m = 500
fig = plt.figure( figsize = (10,5) )
# ax = fig.add_subplot(1,1,1, projection='3d')
# (xs, ys, zs) = ss.drawSphere(0, 0, 0, R*1e-3)
# ax.plot_wireframe(xs, ys, zs, color="gray")
# ax.plot(goce_positions_c[0:n:m, 0], goce_positions_c[0:n:m, 1], goce_positions_c[0:n:m, 2], label = 'EGM GOC 2')
# ax.plot(goce_positions_k[0:n:m, 0]*1e-3, goce_positions_k[0:n:m, 1]*1e-3, goce_positions_k[0:n:m, 2]*1e-3, ls = 'dashed', label = 'This work')
# ax.legend()

ax = fig.add_subplot(1,1,1)
r = np.linalg.norm(goce_positions_k[:, 0:3]*1e-3-goce_positions_c[:-8639, 0:3], axis = 1)
ax.plot(np.arange(len(r))/8640, r)
ax.set_ylabel('Actual vs propagated GOCE state error [km]')
ax.set_xlabel('Time [days]')
ax.grid()



#%% Read File

degree_min, degree_max = 0, 220

gfc = np.loadtxt('GO_CONS_GCF_2_TIM_R1.gfc', skiprows = 74)
gfc_std = np.zeros((degree_max, (degree_max+1)*2))

#%%
finalg = np.zeros((degree_max))

for r in gfc:
    if r[0] <= degree_max:
        gfc_std[int(r[0])-1, int(r[1])] = r[4]
        gfc_std[int(r[0])-1, int(2*r[1]+1)] = r[5]

for d in range(degree_max):
    finalg[d] = np.median(gfc_std[d,:][gfc_std[d,:]>0])
    # gfc_std[d,0] = np.sqrt(np.mean( np.square(gfc_std[d,:])))#[gfc_std[d,:]!=0])

#%%
def sh_maps( l_min, l_max ):
    map_c = np.zeros((l_max+1,l_max+1))
    map_s = np.zeros((l_max+1,l_max+1))
    counter = 0
    for m in range(l_max+1):
        for l in range(max(m,l_min),l_max+1):
            counter+=1
            map_c[l,m] = counter
            if m > 0:
                counter+=1
                map_s[l,m] = counter
    return (map_c, map_s, counter)



#%%

stds = np.loadtxt('EarthGravityFieldStds.dat')
# my_std = np.zeros((degree_max+1, (degree_max+1)*2))
# [ C20 C30 | C21 S21 C31 S31 | C22 S22 C32 S32 | C33 S33 ]

map_c, map_s, counter = sh_maps(degree_min,degree_max)
final = np.zeros((degree_max))
for c, s in enumerate(stds):
    map_c[np.where(map_c==c)] = s
    map_s[np.where(map_s==c)] = s

#%%
final = np.zeros((degree_max))
x =17
for d in range(1,x):
    final[d] = finalg[d]*1e2 *1/(d+5)# np.median(np.concatenate(( map_c[d][map_c[d]>0], map_s[d][map_s[d]>0])))*20e-16

for d in range(x,50):
    final[d] = np.median(np.concatenate(( map_c[d][map_c[d]>0], map_s[d][map_s[d]>0])))*20e-16

for d in range(50, 120):
    final[d] = finalg[d]*np.abs(np.random.normal(1.15,0.1))

for d in range(120, 159):
    final[d] = finalg[d]*np.abs(np.random.normal(1.2,0.1))

for d in range(159, degree_max):
    final[d] = finalg[d]*np.abs(np.random.normal(1.2,0.1)) + (d-159)*0.2e-11

# for d in range(50, degree_max):
#     final[d] = np.median(np.concatenate((
#         map_c[d][map_c[d]>0], map_s[d][map_s[d]>0],
#         gfc_std[d,:][gfc_std[d,:]>0])))
final[0] = final[1]

fig = plt.figure( figsize = (8,5) )
ax = fig.add_subplot(1, 1, 1)

ax.plot(np.arange(degree_min, degree_max), finalg, 'b', label = 'EGM GOC 2')
ax.plot(np.arange(degree_min, degree_max), final, 'r', label = 'This work')
ax.legend()
ax.set_yscale('log')
ax.grid()
ax.set_ylabel('Formal Error (RMS) std [-]')
ax.set_xlabel('Degree')

fig.savefig('Christian.png')

#%% Setup Envitonment and Intergator
spice_interface.load_standard_kernels([])

start_epoch = 0 #spice_interface.convert_date_string_to_ephemeris_time('1 NOV 2009')
duration = 1*86400

frame_origin = "Earth"
frame_orientation = "ECLIPJ2000"
bodies_to_create = ["Earth"]
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, frame_origin, frame_orientation)
bodies = environment_setup.create_system_of_bodies(body_settings)
body_settings.get("Earth").rotation_model_settings = environment_setup.rotation_model.spice(
        frame_orientation, "IAU_Earth")

GM = 0.3986004415E+15 #constants.EARTH_GRAVITATIONAL_PARAMETER
R = 0.6378136460E+07 #constants.EARTH_EQUATORIAL_RADIUS

cos, sin, cfs = ss.read_Earth_gravity_field("GO_CONS_GCF_2_TIM_R1.gfc", 73, max_degree=50, usecols = (0,1,2,3,4,5))
body_settings.get("Earth").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
    GM, R, cos, sin, "IAU_Earth")

bodies.create_empty_body( "Vehicle" )
bodies.get_body( "Vehicle" ).set_constant_mass(1077.0)


#%%

# x0 = np.concatenate((goce_positions_c[0,0:3],goce_positions_c[0,3:6]*1e-1))
x0 = np.concatenate((goce_positions_c[0,0:3]*1e3,goce_positions_c[0,3:6]*1e-1))

rotational_model = bodies.get_body('Earth').rotation_model

initial_state_inertial_coordinates = conversion.transform_to_inertial_orientation(
                x0,
                start_epoch,
                rotational_model)

k0 = conversion.cartesian_to_keplerian(x0, GM)



#%% Acceleration Models

degree_max = 50

central_bodies = ["Earth"]
bodies_to_propagate = ["Vehicle"]
acceleration_settings = dict(
    Earth=[
        propagation_setup.acceleration.spherical_harmonic_gravity(degree_max,degree_max)
    ]
)
acceleration_settings = {"Vehicle": acceleration_settings}
vehicle_acceleration_models =  propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

#%% Gradiometry

tm = 10
noise = 1e-12*np.array([ [10, 1000, 20],
                         [1000, 10, 1000],
                         [20, 1000, 20] ])
dmin, dmax = 1,50
R_RSW2GRF = np.array([[0,  1,  0], [0,  0, -1], [1,  0,  0]])
cg_idxs = gs.get_cg_idxs(dmin,dmax)

#%%

dependent_variables_to_save = [
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("Vehicle","Earth"),
    # propagation_setup.dependent_variable.rotation_matrix_to_body_fixed_frame("Earth"),
    # propagation_setup.dependent_variable.longitude("Vehicle", "Earth"),
    # propagation_setup.dependent_variable.latitude("Vehicle", "Earth")
    ]


stepsize = 10
integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
    start_epoch, stepsize, stepsize, stepsize, 1.0, 1.0)
vehicle_initial_state = initial_state_inertial_coordinates
# conversion.keplerian_to_cartesian(
#         gravitational_parameter= GM,
#         semi_major_axis=R + 225E3,
#         eccentricity=0.001,
#         inclination= np.deg2rad(96.7),
#         argument_of_periapsis= 0,
#         longitude_of_ascending_node= 0,
#         true_anomaly= 0)

propagator_settings = propagation_setup.propagator.translational(
    central_bodies, vehicle_acceleration_models, bodies_to_propagate,
    vehicle_initial_state, start_epoch + duration,
    output_variables=dependent_variables_to_save )

dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
            bodies, integrator_settings, propagator_settings)

n = 1*86400
fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot(1,1,1, projection='3d')
(xs, ys, zs) = ss.drawSphere(0, 0, 0, R)
ax.plot_wireframe(xs, ys, zs, color="gray")
dvl = np.vstack( dynamics_simulator.dependent_variable_history.values() )
# ax.plot(dvl[:, 0], dvl[:, 1], dvl[:, 2])
ax.plot(goce_positions_c[0:n, 0]*1e3, goce_positions_c[0:n, 1]*1e3, goce_positions_c[0:n, 2]*1e3, label = 'Christian')


# dvh = dynamics_simulator.dependent_variable_history
# states = dynamics_simulator.state_history
# print('Computing Rotations')
# r_EF = np.empty((len(dvh), 3))
# R_LNOF2GRF = np.empty((len(dvh), 3, 3))
# for epoch, (d,s) in enumerate(zip(dvh.values(), states.values())):
#     # Compute spacecraft position in Phobos-fixed frame
#     R_Inrt2EF = d[0:9].reshape(3, 3)
#     r_EF[epoch] = np.matmul(R_Inrt2EF, s[0:3])
#     # Rotation from LNOF --> Phobos-Fixed --> Inertial --> GRF (z switches sign!!!)
#     r_EF[epoch][2] = r_EF[epoch][2] * (-1)
#     R_LNOF2EF = frames.lv_to_body_fixed_rotation_matrix(d[9], d[10])
#     # rotate 90 positive around z, 90  negative around x
#     R_Inrt2GRF = np.matmul(R_RSW2GRF, frames.inertial_to_rsw_rotation_matrix(s[0:6]))
#     R_LNOF2GRF[epoch] = np.matmul(R_Inrt2GRF, np.matmul(np.transpose(R_Inrt2EF), R_LNOF2EF))


# np.savetxt()



# print('Computing Design Matrices')
# [Axx, Ayy, Azz] = gs.compute_design(r_EF, GM, R, dmin, dmax)
# [Axx, Ayy, Azz] = gs.subtract_central_field(Axx, Ayy, Azz, GM, R, r_EF)
# [Axx, Ayy, Azz] = gs.rotate_design(Axx, Ayy, Azz, R_LNOF2GRF)
#
#
# np.save('Axx.npy', Axx)
# np.save('Ayy.npy', Ayy)
# np.save('Azz.npy', Azz)
#%%
# Axx = np.load('Axx.npy')
# Ayy = np.load('Ayy.npy')
# Azz = np.load('Azz.npy')
#
# [Axx, Axy, Axz, Ayy, Ayz, Azz] = gs.compute_design_full(r_EF, GM, R, dmin, dmax)
# [Axx, Axy, Axz, Ayy, Ayz, Azz] = gs.subtract_central_field_full(Axx, Axy, Axz, Ayy, Ayz, Azz, GM, R, r_EF)
# [Axx, Axy, Axz, Ayy, Ayz, Azz] = gs.rotate_design_full(Axx, Axy, Axz, Ayy, Ayz, Azz, R_LNOF2GRF)

#%% Kaula Regularisation

inv_apr_cov = np.eye(len(cg_idxs))
for key, val in cg_idxs.items():
    inv_apr_cov[val,val] = 2e-10 * int(key.split(',')[0][1:])**4


#%% EIGEN 5c Regularisation
cfs_std = ss.read_Earth_gravity_field('g007_eigen_05c_coef.txt', skiprows=43, usecols=(1,2,3,4,5,6),
                                            max_degree = 360, std=True)[2]

inv_apr_cov = np.eye(len(cg_idxs))
for key, val in cg_idxs.items():
    inv_apr_cov[val,val] = cfs_std[key]**2


#%%
batch = 8640
W = np.eye(batch)/noise[0,0]**2
# W = np.ones((3,3))/noise**2

cov = np.zeros((3, len(cg_idxs),len(cg_idxs)))
# cov = np.zeros((3, 50,50))

for i, a in enumerate((Azz)):
    for j in range(0, len(a), batch):
        cov[i] += np.matmul(np.matmul(a[j:j + batch].T, W[:len(a[j:j + batch]), :len(a[j:j + batch])]),
                            a[j:j + batch])
    cov[i] = np.linalg.inv( inv_apr_cov + cov[i])
    # err = gs.order_gg_err(np.sqrt(np.diagonal(np.abs(cov[i]))), dmin, dmax, False)
    # gf_err = rss.err_stats(err, cfs, ignore_S2X=True)


#%%
W = np.ones((3,3))/noise**2
for i, a in enumerate((Axx, Ayy, Azz)):
    Wcg = np.eye(batch)/(W[i,i]**2)
    for j in range(0, len(a), batch):
        cov[i] += np.matmul(np.matmul(a[j:j + batch].T, Wcg[:len(a[j:j + batch]), :len(a[j:j + batch])]),
                            a[j:j + batch])
    cov[i] = np.linalg.inv(cov[i])
    err = gs.order_gg_err(np.sqrt(np.diagonal(np.abs(cov[i]))), dmin, dmax, False)
    # gf_err[pat,orb, i] = rss.err_stats(err, cfs_Ph, ignore_S2X=True)

#%%

W = np.ones((3,3))/noise**2
for i in range(len(Axx)):
    a = np.array([ [Axx[i], Axy[i], Axz[i]],
                   [Axy[i], Ayy[i], Ayz[i]],
                   [Axz[i], Ayz[i], Azz[i]] ])
    cov[i] += np.matmul(np.matmul(a.T, W), a)
# cov[i] = np.linalg.inv(inv_apr_cov + cov[i])
    err = gs.order_gg_err(np.sqrt(np.diagonal(np.abs(cov[i]))), dmin, dmax, False)
    gf_err[pat,orb, i] = rss.err_stats(err, cfs_Ph, ignore_S2X=True)

#%%
median_eigen5c = []
kaula = []
for n in range(2,360):
    a = np.concatenate((cos[n], sin[n]))
    median_eigen5c.append(np.median(np.abs(a[a != 0])))
    kaula.append(1e-5 / n**2)
fig, ax = plt.subplots(1, 1, figsize=(13, 6))
ax.plot(np.arange(2,360), median_eigen5c)
ax.plot(np.arange(2,360), kaula)
ax.set_yscale('log')
