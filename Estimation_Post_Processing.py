# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 19:17:16 2021

@author: Michael
"""

import os
if os.getcwd() == 'C:\\Users\\Michael':
    os.chdir('PyPhobos')
    
import numpy as np
from math import pi, sqrt, ceil
from matplotlib import pyplot as plt, colors, rcParams
from matplotlib.lines import Line2D


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import PercentFormatter, ScalarFormatter

rcParams.update({'font.size': 16})

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#%% Contribution of Tracking Data

n_lnd = 25
gf_err_rs = np.load('tracking_gf_err_rs.npy')
idxs = np.where(gf_err_rs[0,:,0] != 0)[0]
idxs_low = idxs[0:10]
idxs_high = idxs[10:]
cor_rs = np.load('tracking_cor_rs.npy')[:,idxs] # cn, cn_gf, corr_C2, cn_Ph, cn_lnd]
# rsc, vsc, desat,emp,range,vlbi,optical, GM, rPh, vPh
err_rs = np.load('tracking_err_rs.npy')[:,idxs] 
err_rs = np.delete(err_rs, (2,3,4,5,6,7), axis = 2) 
err_lnd_low = np.load('tracking_err_lnd.npy')[:,idxs_low] 
err_lnd_high = np.load('tracking_err_lnd.npy')[:,idxs_high] 

# divide by Doppler-only case
# for p in range(4):
#     gf_err_rs[:,:,p] = gf_err_rs[:,:,p]/np.mean(gf_err_rs[0,:,p])
#     err_rs[:,:,p] = err_rs[:,:,p]/np.mean(err_rs[0,:,p])

# np.save('err_lnd_equalised.npy', err_rs[:,:,4:])
# err_lnd_equalised = np.load('err_lnd_equalised.npy')

#%%
fig = plt.figure(figsize=(13, 13))

cases = 8
cols = ('k','orange','c','r','y','b','k','m','g')

case_labels = ('Doppler only',
               'w/ range',
            'w/ VLBI',
            'w/ range & opt.',
            'w/ VLBI & opt.',
            r"all",
            r"all, frequent opt.",
            'all, low-noise opt.',
            'all, tuned opt.',
          )
xticks = [  
              '$r_{Ph}$','$v_{Ph}$',
            'GM',
            '$C&S_1$','$C_2$','$C&S_3$','$C&S_4$',
            '$C&S_5$','$C&S_6$',
            # '$ \Delta V_{d}$', '$a_{emp}$',
            # r'$\rho_b$', '$\delta_b$',
            # '$opt_{bias}$'
        ]
ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2)
ax3 = plt.subplot(3,1,3)
axes = [ax1, ax2, ax3]

size = 100
for ax in axes:
    if ax == ax1:
        for case in range(1,cases):
            ax.scatter(x = np.arange(4), y = np.mean(err_rs[case,:], axis=0),
                       s = size, # s = np.mean(cor_rs[case,:,3], axis=0)*1e-8*30)  
                       c = cols[case])
            ax.scatter(x = np.arange(4,8), y = np.mean( gf_err_rs[case,:], axis = 0),
                            s = size, # s = np.mean(cor_rs[case,:,1], axis=0), 
                            c = cols[case], label = case_labels[case])
        ax.set_xticks(np.arange(8))        
        ax.set_xticklabels(xticks)
        ax.set_title('Science and Spacecraft Parameters')    
        ax.set_ylabel('Formal Error wrt Doppler-only [-]')   
    elif ax == ax2:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        for case in range(5, cases):
            ax.errorbar(x = np.arange(n_lnd), y = np.mean(err_lnd_low[case,:], axis = 0),
                            yerr = np.std(err_lnd_low[case,:], axis = 0),
                            c = cols[case], capthick = 2, capsize= 5  )
        # ax.errorbar(x = np.arange(n_lnd), y = np.mean(err_lnd_equalised[case,:], axis = 0),
        #                 yerr = np.std(err_lnd[case,:], axis = 0),
        #                 c = 'g', capthick = 2, capsize= 5 ) 
        ax.set_title('Landmark Positions - Low Orbits')
        ax.set_xticks(ticks = np.arange(n_lnd))
        ax.set_xticklabels(np.arange(n_lnd))
        ax.set_ylabel('Formal Error [m]')
    elif ax == ax3:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        for case in range(5, cases):
            ax.errorbar(x = np.arange(n_lnd), y = np.mean(err_lnd_high[case,:], axis = 0),
                            yerr = np.std(err_lnd_high[case,:], axis = 0),
                            c = cols[case], capthick = 2, capsize= 5  )
        # ax.errorbar(x = np.arange(n_lnd), y = np.mean(err_lnd_equalised[case,:], axis = 0),
        #                 yerr = np.std(err_lnd[case,:], axis = 0),
        #                 c = 'g', capthick = 2, capsize= 5 ) 
        ax.set_title('Landmark Positions - High Orbits')
        ax.set_xticks(ticks = np.arange(n_lnd))
        ax.set_xticklabels(np.arange(n_lnd))
        ax.set_ylabel('Formal Error [m]')
  
    ax1.legend(loc='lower right')#, bbox_to_anchor=(1.01,-1.1))
    ax.grid(b=True, which='both')
    ax.grid(b=True, which='both', color='gray', ls = 'dashdot', alpha = 0.5)
# fig.suptitle('Formal Error wrt Doppler only case', )
fig.tight_layout()
#%% PLot Consider Covariance Ratio. Tracking at 50%

ratio = np.load('consider_ratio.npy')
idxs = np.where(ratio[0,:,0,0] != 0)[0]
ratio = ratio[:,idxs[0:10]]
ratio[:,0] = np.mean(ratio[:,:], axis = 1) # mean across orbits

#%%

xticks = ['$r_{Ph}$', '$v_{Ph}$',
          'GM','$C&S_1$','$C_2$','$C&S_3$','$C&S_4$',
            '$C&S_5$','$C&S_6$']

GS_uncertainty = ('1','5','10')
emp_uncertainty = ('$1$','$0.1$','0.05')


fig = plt.figure(figsize = (10,6))
ax = plt.subplot(1,1,1)
mrk = ('o','X','d')
style = ('solid','dashed','dashdot')

size = 100
color1, color2 = 'black','firebrick'
for i in range(3):
    ax.scatter(x = np.arange(9), y = 1/ratio[0,0,i], marker = mrk[i],
                label = GS_uncertainty[i], color = color1, s = size)

    ax.spines['left'].set_color(color1)
    ax.set_ylabel('$\sigma_{P+r_{GS}}/\sigma_P$', color = color1)
    
    ax.tick_params(axis='y', colors=color1)
    
ax2 = ax.twinx()
for i in range(3):
    ax2.scatter(x = np.arange(9), y = 1/ratio[1,0,i], marker = mrk[i],
               label = emp_uncertainty[i], color = color2, s = size)
    ax2.set_ylabel('$\sigma_{P+a_{emp}}/\sigma_P$', color = color2)
    
    ax2.tick_params(axis='y', colors=color2)
    # ax2.set_yscale('log')
    ax.set_xticks(ticks = np.arange(9)) 
    ax.set_xticklabels(xticks)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height* 0.9])
ax.legend(loc='center left', bbox_to_anchor=(0.5, 1.15), title='GS [mm]', ncol = 3)
ax2.legend(loc='center', bbox_to_anchor=(0.2, 1.15),  title= '$a_{emp} [nm/s^2]$', ncol = 3)        
    
ax.grid(b=True, which='both', color=color1, ls = 'dashdot', alpha = 0.5)
ax2.grid(b=True, which='both', color = color2, ls = 'dashdot',alpha = 0.5) 
ax2.spines['left'].set_color(color1)
ax2.spines['right'].set_color(color2)


#%% Gravity Field Results. RS only. 100%

gf_err_rs = np.load('gf_err_rs.npy')
# idxs = np.where(gf_err_rs[0,:,0] != 0)[0]
# gf_err_rs = gf_err_rs[:,idxs]
gf_cor = np.load('gf_cor.npy')#[:,idxs] # cn combined, corr c20-c22
err_rs = np.load('err_rs.npy')#[:,idxs]  # GM, sc, Ph, lnd 
# pareto = np.load('pareto.npy')#[idxs] 
pareto = np.load('pareto_reduced.npy')#[idxs] 
gf_err = np.load('gf_err.npy')#[:,idxs] 
cor_rs = np.load('cor_rs.npy')#[:,idxs]  # cn, cn_gf, corr_C2, cn_Ph, cn_lnd


p = 2 # wrt 50%
# select axis which performs best for degrees higher than 4 
idxs = np.zeros_like((gf_err[p,:,0,4:]), dtype = int)
idx = np.zeros((len(gf_err[p])), dtype = int)
for orb in range(len(gf_err[p])):
    for deg in range(6):
        idxs[orb,deg] = np.argmin(gf_err[p,orb,:,deg+4])
    idx[orb] = np.bincount(idxs[orb]).argmax()
    gf_err[p,orb,0] = gf_err[p,orb,idx[orb]]
    gf_cor[p,orb,0] = gf_cor[p,orb,idx[orb]]
gf_cor = gf_cor[p,:,0,:]
gf_err = gf_err[p,:,0,:]

#%% Gravity Fiels Estimation, RS only, color is accuracy

cmap = plt.cm.plasma
titles = (
          '$RMS( \overline{ \sigma}_{C&S1X})$',
          '$RMS (\overline{ \sigma}_{C2X})$',
          '$RMS( \overline{ \sigma}_{C&S3X})$',
          '$RMS( \overline{ \sigma}_{C&S4X})$',
          )
fig = plt.figure(figsize=(10,10))
grid = ImageGrid(fig, 111, nrows_ncols=(2,2),
                 axes_pad=0.1,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                label_mode = 'L',
                cbar_pad=0.15,
                aspect = False)

# assign marker for improvement
mrk = ('o','p','^','d')
vals = np.array([1, 2, 5, 10])
s = 60

legend_elements = []
for i in range(2):
    legend_elements.append(Line2D([0], [0], marker=mrk[i], color='w',
                        markerfacecolor='k', label=str(vals[i]), markersize=15))
grid.axes_all[0].legend(handles=legend_elements, loc = 'upper left', ncol=2,
                        title = 'n-fold improvement \n wrt RS at 50%')
for t, ax in enumerate(grid.axes_all):
    mrks = []
    for n in range(len(pareto)):
        mr = mrk[ np.where(vals == find_nearest(vals,
                gf_err_rs[p,n,t]/gf_err_rs[0,n,t]))[0][0]]
        mrks.append(mr)
        im = ax.scatter(x = pareto[n,0], y = pareto[n,1], 
                        c = gf_err_rs[0,n,t],
                        edgecolors = 'k',
                        marker = mr,
                        s = s,
                        cmap = cmap,
                        norm = colors.LogNorm(vmin = np.min(gf_err_rs[0,:]),
                                              vmax = np.max(gf_err_rs[0,:])))
    ax.set_title(titles[t],  y=1.0, pad=-18)
    ax.set_yticklabels([])
    axins = zoomed_inset_axes(ax, 2, loc=8)
    for n in range(len(pareto)):
        axins.scatter(x = pareto[n,0], y = pareto[n,1], 
                            c = gf_err_rs[0,n,t],
                            edgecolors = 'k',
                            s = s,
                            marker = mrks[n],
                            cmap = cmap,
                            norm = colors.LogNorm(vmin = np.min(gf_err_rs[0,:]),
                                                  vmax = np.max(gf_err_rs[0,:])))
    axins.set_xlim(20, 24.8)
    axins.set_ylim(0.32, 0.336)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    if t <= 1:
        ax.set_title(titles[t])
    else:
        ax.set_title(titles[t],  y=1.0, pad=-18)
    if t in (0,2):
        ax.set_ylabel('$\overline{euc}$'+' [-]')
    if t > 1:
        ax.set_xlabel('$\overline{r}$'+' [km]')
ax.cax.colorbar(im, format= PercentFormatter(1))

#%% Gravity Fiels Estimation, RS only at 100%

gf_err_rs = np.load('gf_err_rs_f10.npy')
idxs = np.where(gf_err_rs[0,:,0] != 0)[0]
gf_err_rs = gf_err_rs[:,idxs]
# gf_cor = np.load('gf_cor.npy')[:,idxs] # cn combined, corr c20-c22
# err_rs = np.load('err_rs.npy')[:,idxs]  # GM, sc, Ph, lnd 
pareto = np.load('pareto_reduced.npy')[idxs] 
# gf_err = np.load('gf_err.npy')[:,idxs] 
cor_rs = np.load('cor_rs_f10.npy')[0,idxs]  # cn, cn_gf, corr_C2, cn_Ph, cn_lnd

#%%

p = 2 # absolute, wrt apr, wrt vals
cmap = plt.cm.plasma
titles = (
          'Degree-1',
          '',
          'Degree-3',
          'Degree-4',
          'Degree-5',
          'Degree-6'
          )
fig = plt.figure(figsize=(10,10))
grid = ImageGrid(fig, 111, nrows_ncols=(2,3),
                 axes_pad=0.1,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                label_mode = 'L',
                cbar_pad=0.15,
                aspect = False)

s = 60
mrk = ('o','p','^','d')
vals = np.array([0.2, 0.4, 0.6, 0.8])
mrks = []
legend_elements = []
for i in range(4):
    legend_elements.append(Line2D([0], [0], marker=mrk[i], color='w',
                        markerfacecolor='k', label=str(vals[i]), markersize=15))
grid.axes_all[1].legend(handles=legend_elements, loc = 'center', ncol=2,
                        title = '$C_{20},C_{22}$' + r'&' +' corr:', 
                        bbox_to_anchor = (0.4, 1),
                        columnspacing = 0.5
                        )
for t, ax in enumerate(grid.axes_all):  
    axins = zoomed_inset_axes(ax, 2, loc=8)      
    if t == 1:   
        for n in range(len(pareto)):
            mr = mrk[ np.where(vals == find_nearest(vals,cor_rs[n,2]))[0][0]]
            mrks.append(mr)
            im = ax.scatter(x = pareto[n,0], y = pareto[n,1], 
                    c = gf_err_rs[p,n,t],
                    edgecolors = 'k' if gf_err_rs[1,n,t]>=0.1 else None,
                    s = s,
                    marker = mr,
                    cmap = cmap,
                    norm = colors.LogNorm(vmin = np.min(gf_err_rs[p]),
                                          vmax = np.max(gf_err_rs[p]))
                    )
       
            axins.scatter(x = pareto[n,0], y = pareto[n,1], 
                                c = gf_err_rs[p,n,t],
                                s = s,
                                edgecolors = 'k' if gf_err_rs[1,n,t]>=0.1 else None,
                                cmap = cmap,
                                marker = mrks[-1],
                                norm = colors.LogNorm(vmin = np.min(gf_err_rs[p]),
                                          vmax = np.max(gf_err_rs[p]))
                                )
    else:
        for n in range(len(pareto)):
            im = ax.scatter(x = pareto[n,0], y = pareto[n,1], 
                    c = gf_err_rs[p,n,t],
                    edgecolors = 'k' if gf_err_rs[1,n,t]>=0.1 else None,
                    linewidth=2,
                    s = s,
                    cmap = cmap,
                    norm = colors.LogNorm(vmin = np.min(gf_err_rs[p]),
                                          vmax = np.max(gf_err_rs[p]))
                    )
       
            axins.scatter(x = pareto[n,0], y = pareto[n,1], 
                                c = gf_err_rs[p,n,t],
                                s = s,
                                edgecolors = 'k' if gf_err_rs[1,n,t]>=0.1 else None,
                                linewidth=2,
                                cmap = cmap,
                                norm = colors.LogNorm(vmin = np.min(gf_err_rs[p]),
                                                      vmax = np.max(gf_err_rs[p]))
                                )
        
    ax.set_yticklabels([])
    
    axins.set_xlim(20, 24.8)
    axins.set_ylim(0.32, 0.336)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_title(titles[t],  y=1.0, pad=-18)
    if t in (0,3):
        ax.set_ylabel('$\overline{euc}$'+' [-]')
    if t > 1:
        ax.set_xlabel('$\overline{r}$'+' [km]')
# ax.cax.colorbar(im, format= ScalarFormatter(), label = 'Formal Error (rms) [-]')
# ax.cax.colorbar(im, format= PercentFormatter(1)) #ScalarFormatter(), label = 'Formal Error (rms) [-]')

ticks = [1, 0.1, 0.01, 0.001, 0.0001]
cb = ax.cax.colorbar(im, label = 'Formal Error (rms) [-]')
cb.set_ticklabels([str(t*100)+'%' for t in ticks])
cb = ax.cax.colorbar(im, ticks = ticks, label = 'Formal Error (rms) [-]')
cb.set_ticklabels([str(t) for t in ticks])

#%% Gravity Field Estimation, combined. Rs at 50%

gf_err_rs = np.load('gf_err_rs_f5.npy')
idxs = np.where(gf_err_rs[0,:,0] != 0)[0]
gf_err_rs = gf_err_rs[:,idxs]
gf_cor = np.load('gf_cor.npy')[:,idxs] # cn combined, corr c20-c22
err_rs = np.load('err_rs.npy')[:,idxs]  # GM, sc, Ph, lnd 
pareto = np.load('pareto_reduced.npy')[idxs] 
gf_err = np.load('gf_err_f5.npy')[:,idxs] 
cor_rs = np.load('cor_rs.npy')[0,idxs]  # cn, cn_gf, corr_C2, cn_Ph, cn_lnd


p = 2 # absolute, wrt apr, wrt vals
# select axis which performs best for degrees higher than 4 
idxs = np.zeros_like((gf_err[p,:,0,4:]), dtype = int)
idx = np.zeros((len(gf_err[p])), dtype = int)
for orb in range(len(gf_err[p])):
    for deg in range(6):
        idxs[orb,deg] = np.argmin(gf_err[p,orb,:,deg+4])
    idx[orb] = np.bincount(idxs[orb]).argmax()
    gf_err[p,orb,0] = gf_err[p,orb,idx[orb]]
    gf_cor[0,orb,0] = gf_cor[0,orb,idx[orb]]
gf_cor = gf_cor[0,:,0,:]
gf_err = gf_err[:,:,0,:]

#%%
titles = (
          # 'Degree-1',
          '',
          'Degree-3',
          'Degree-4',
          'Degree-5',
          'Degree-6',
          'Degree-7',
          'Degree-8',
          'Degree-9',
          'Degree-10',

          )

cmap = plt.cm.plasma
fig = plt.figure(figsize=(10,10))
grid = ImageGrid(fig, 111, nrows_ncols=(3,3),
                 axes_pad=0.1,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                label_mode = 'L',
                cbar_pad=0.15,
                aspect = False)

s = 60
mrk = ('o','p','^','d')
vals = np.array([0.2, 0.4, 0.6, 0.8])
mrks = []
legend_elements = []
for i in range(4):
    legend_elements.append(Line2D([0], [0], marker=mrk[i], color='w',
                        markerfacecolor='k', label=str(vals[i]), markersize=15))
grid.axes_all[0].legend(handles=legend_elements, loc = 'center', ncol=2,
                        title = '$C_{20},C_{22}$' + r'&' +' corr:', 
                        bbox_to_anchor = (0.3, 1),
                        columnspacing = 0.5
                        )
for t, ax in enumerate(grid.axes_all):  
    axins = zoomed_inset_axes(ax, 2, loc=8)      
    if t == 0:   
        for n in range(len(pareto)):
            mr = mrk[ np.where(vals == find_nearest(vals,gf_cor[n,1]))[0][0]]
            mrks.append(mr)
            im = ax.scatter(x = pareto[n,0], y = pareto[n,1], 
                    c = gf_err[p,n,t+1],
                    edgecolors = 'k' if gf_err[2,n,t+1]>=1 else None,
                    s = s,
                    marker = mr,
                    cmap = cmap,
                    norm = colors.LogNorm(vmin = np.min(gf_err[p]),
                                          vmax = np.max(gf_err[p]))
                    )
       
            axins.scatter(x = pareto[n,0], y = pareto[n,1], 
                                c = gf_err[p,n,t+1],
                                s = s,
                                edgecolors = 'k' if gf_err[2,n,t+1]>=1 else None,
                                cmap = cmap,
                                marker = mrks[-1],
                                norm = colors.LogNorm(vmin = np.min(gf_err[p]),
                                          vmax = np.max(gf_err[p]))
                                )
    else:
        for n in range(len(pareto)):
            im = ax.scatter(x = pareto[n,0], y = pareto[n,1], 
                    c = gf_err[p,n,t+1],
                    edgecolors = 'k' if gf_err[2,n,t+1]>=1 else None,
                    linewidth=2,
                    s = s,
                    cmap = cmap,
                    norm = colors.LogNorm(vmin = np.min(gf_err[p]),
                                          vmax = np.max(gf_err[p]))
                    )
       
            axins.scatter(x = pareto[n,0], y = pareto[n,1], 
                                c = gf_err[p,n,t+1],
                                s = s,
                                edgecolors = 'k' if gf_err[2,n,t+1]>=1 else None,
                                linewidth=2,
                                cmap = cmap,
                                norm = colors.LogNorm(vmin = np.min(gf_err[p]),
                                                      vmax = np.max(gf_err[p]))
                                )
        
    ax.set_yticklabels([])
    
    axins.set_xlim(20, 24.8)
    axins.set_ylim(0.32, 0.336)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_title(titles[t],  y=1.0, pad=-18)
    if t in (0,3,6):
        ax.set_ylabel('$\overline{euc}$'+' [-]')
    if t > 1:
        ax.set_xlabel('$\overline{r}$'+' [km]')
# ax.cax.colorbar(im, format= ScalarFormatter(), label = 'Formal Error (rms) [-]')
# ax.cax.colorbar(im, format= PercentFormatter(1)) #ScalarFormatter(), label = 'Formal Error (rms) [-]')

ticks = [0.01, 0.001, 0.0001, 0.00001]
# cb = ax.cax.colorbar(im, label = 'Formal Error (rms) [-]')
# cb.set_ticklabels([str(t*100)+'%' for t in ticks])
cb = ax.cax.colorbar(im, ticks = ticks, label = 'Formal Error (rms) [-]')
cb.set_ticklabels([str(t*100)+'%' for t in ticks])


#%% Best Orbit Pyramid

# For RS best acc is 26 (123) and 21 (456)
# FOr combined its 0 and 21 & 1
pareto=np.load('pareto_reduced.npy')


best = [26, 27]
for p in range(len(pareto)):
    plt.scatter(pareto[p,0], pareto[p,1], c = 'blue')
for b in best:
    plt.scatter(pareto[b,0], pareto[b,1], c = 'red')

#%%

pareto=np.load('pareto_reduced.npy')

orb = 26

def myformat(label):
    if label <=0.0001:
        return "{:.3%}".format(label)
    if label <=0.001:
        return "{:.2%}".format(label)
    if label <=0.01:
        return "{:.1%}".format(label)
    if label <=0.1:
        return "{:.0%}".format(label)
    
cmap = plt.cm.plasma

best_rs = np.abs(np.load('pyramid_'+str(orb)+'_rs.npy'))
best_rs_n = np.abs(np.load('pyramid_'+str(orb)+'_rs_n.npy'))
# cut these
best_rs = best_rs[0:6]
best_rs = best_rs[0:6]

best = np.abs(np.load('pyramid_'+str(orb)+'.npy'))
best_n = np.abs(np.load('pyramid_'+str(orb)+'_n.npy'))

fig = plt.figure(figsize=(10,10))
grid = ImageGrid(fig, 111, nrows_ncols=(2,1),
                 axes_pad=0.1,
                 cbar_size = '3%',
                 share_all=False,
                 cbar_location="right",
                 cbar_mode="single",
                label_mode = 'L',
                cbar_pad=0.1,
                aspect = False)

minn = np.min(best_n[best_n!=0])
maxx = np.min(best_n[best_n!=0])
if np.min(best_rs[best_rs!=0]) < minn:
    minn = np.min(best_rs[best_rs!=0])
if np.max(best_rs[best_rs!=0]) > maxx:
    maxx = np.max(best_rs[best_rs!=0])
    
for t, ax in enumerate(grid.axes_all):  
    if t == 1:
        im = ax.imshow(best, aspect='auto', interpolation = 'none',
                       cmap = cmap, norm = colors.LogNorm(vmin = minn, vmax = maxx))
        datan = np.ma.masked_where((best_n == 0 ), best_n)
        for (j,i),label in np.ndenumerate(datan):
            if label!=0 and label<=0.5:
                if ((i == 10 and j in (6,8)) or (i == 8 and j == 9) or (i == 12 and j == 7)
                or (i == 8 and j == 9) or (i == 5 and j == 8) or (i == 9 and j == 6)):
                    color = 'black'
                else:
                    color = 'white'
                ax.text(i,j,myformat(label),ha='center',va='center', fontsize = 'xx-small', color = color)
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels([str(i) for i in np.arange(1,11)])
        ax.set_title('RS '+r'&'+' CG', y=0.9, x=0.2, pad = -18)
        # manually fix the white ones
    if t == 0:
        im = ax.imshow(best_rs, aspect='auto', interpolation = 'none',
                       cmap = cmap, norm = colors.LogNorm(vmin = minn, vmax = maxx))
        datarsn = np.ma.masked_where((best_rs_n == 0 ), best_rs_n)
        for (j,i),label in np.ndenumerate(datarsn):
            if label!=0 and label<=0.5:
                if (i == 9 and j == 0):
                    color = 'black'
                else:
                    color = 'white'
                ax.text(i,j,myformat(label),ha='center',va='center', fontsize = 'xx-small', color = color)
        ax.set_title('RS-only', y=0.9, x=0.2, pad = -18)
        ax.set_yticks(np.arange(6))
        ax.set_yticklabels([str(i) for i in np.arange(1,7)])
    
    ax.cax.colorbar(im, label = 'Formal Error (rms) [-]')   
    ax.set_xticks(np.arange(0,21))
    ax.set_xticklabels([str(i) for i in np.arange(-10,11)])
    ax.set_xlabel('Order')
    ax.set_ylabel('Degree')


fig.suptitle('$\overline{r} = $'+str(int(pareto[orb, 0]))+'km, '
                +'$\overline{euc} = $'+
                str(round(pareto[orb, 1],2)) +
                ', '+'$\lambda_n = $' + str(round(pareto[orb, 2],1)),
                y=0.92)    


#%% Propagate Phobos Ephemeris formal Error

subsample = 100
pfe_times = np.load('pfe_times.npy')[0:-1:subsample]
pfe_best = np.load('pfe_MaF_best_1year.npy')[0:-1:subsample]
pfe_worst = np.load('pfe_MaF_worst_1year.npy')[0:-1:subsample]

fig, ax = plt.subplots(2, 1,figsize = (8,6))

labels = ('Radial','Tangential','Normal')
cols = ('b','orange','g')
titles = ('Low 2D Orbit','High 3D Orbit')
mrks = ('o','d','x')

for i in range(3):
    ax[0].plot(pfe_times, pfe_best[:,i], marker = mrks[i],
            label =  labels[i], c = cols[i])
    ax[1].plot(pfe_times, pfe_worst[:,i], marker = mrks[i],
            label = labels[i], c = cols[i])
for i in range(2):
    ax[i].set_yscale('log')
    ax[i].set_xscale('log')
    ax[i].set_xlim([1, 365])
    # ax[i].set_ylim([3e-3, 2])
    # ax[i].set_title(titles[i])
    ax[i].grid(b=True, which='both', color = 'gray', ls = 'dashdot',alpha = 0.5)


ax[0].legend(loc = 'upper left')
# ax.set_xticklabels([])
plt.subplots_adjust(hspace = 0.001)
ax[1].set_xlabel('Time [days]')
fig.tight_layout()
#%% Noise due to Phobos epehemeris and Mars cfs uncertainty


limit = 20000
V_noise_pos = np.load('V_noise_pos.npy')[:,:limit]
V_noise_cfsMastd = np.load('V_noise_cfsMastd.npy')[:,:limit]
r_MaF = np.load('r_MaF.npy')[:limit]
r_MaF_prt = np.load('r_MaF_prt.npy')[:limit]
time = np.load('output_times.npy')[:limit]

pos_err_sc_M = np.linalg.norm(r_MaF-r_MaF_prt, axis = 1)

fig, axs = plt.subplots(2, 1, figsize = (10,10))
labels = ('flight', 'cross-track','nadir')
labels2 = ('Radial','Tangential','Normal')

cols = ('b','orange','g')
# titles = ('Dynamical Model Uncertainty', 'Spacecraft state Uncertainty')
ylabel = 'Gradient Noise '+'$[s^{-2}]$'
ax1 = axs[0].twinx()
ax2 = axs[1].twinx()
for i in range(3):
    axs[0].plot(time, V_noise_cfsMastd[i], linestyle='dashed', color = cols[i], lw=2, label =labels[i])
    axs[1].plot(time, V_noise_pos[i], linestyle='solid', color = cols[i], lw=2, label =labels[i])

r_col = 'm'
ax1.plot(time, np.linalg.norm(r_MaF, axis = 1), color = r_col)
ax1.set_ylim([9.20e6, 9.6e6])

ax2.plot(time, pos_err_sc_M, color = r_col)

for j in range(2):
    axs[j].grid(b=True, which='both', color='gray', ls = 'dashdot', alpha = 0.5)
    axs[j].set_yscale('log')
    axs[j].set_title(titles[j])
    axs[j].set_ylabel(ylabel)
    
axs[0].legend(loc = 'upper left')
axs[0].set_ylim([1e-16,1e-15])

ax1.tick_params(axis='y', colors=r_col)
ax1.grid(b=True, which='both', color = r_col, ls = 'dashdot',alpha = 0.5)
ax1.spines['right'].set_color(r_col)
ax1.set_ylabel('Spacecraft distance to Mars [m]', color = r_col)

axs[1].set_xlabel('time [days]')
# ax2.set_yscale('log')
ax2.set_ylim([3,15])
ax2.tick_params(axis='y', colors=r_col)
ax2.grid(b=True, which='both', color = r_col, ls = 'dashdot',alpha = 0.5)
ax2.spines['right'].set_color(r_col)
ax2.set_ylabel('POD uncertainty wrt Mars [m]', color = r_col)
fig.tight_layout()

#%% RS Correlations averaged across 30 orbits
orb = 2
# cor_mat =  np.load('cor_mat_across10.npy')
# cor_mat =  np.abs(np.load('cor_mat_rs0.npy'))
cor_mat =  np.abs(np.load('cor_mat_rs26.npy'))
narcs = 6

idxs = {'r_Ph': (0, 3), 'v_Ph': (3, 6), 'r_sc0': (6, 9), 'r_sc1': (12, 15), 'r_sc2': (18, 21), 
        'r_sc3': (24, 27), 'r_sc4': (30, 33), 'r_sc5': (36, 39), 'r_sc6': (42, 45), 'v_sc0': (9, 12),
        'v_sc1': (15, 18), 'v_sc2': (21, 24), 'v_sc3': (27, 30), 'v_sc4': (33, 36), 'v_sc5': (39, 42),
        'v_sc6': (45, 48), 'Cr': (48, 49), 'GM': (49, 50), 'desat': (50, 71), 'cfs': (71, 119),
        'emp': (119, 122), 'r_GS': (122, 131), 'range_bias': (131, 134), 'vlbi_bias': (134, 140),
        'optical_bias': (140, 190), 'lnd0': (190, 193), 'lnd1': (193, 196), 'lnd2': (196, 199), 
        'lnd3': (199, 202), 'lnd4': (202, 205), 'lnd5': (205, 208), 'lnd6': (208, 211), 'lnd7':
            (211, 214), 'lnd8': (214, 217), 'lnd9': (217, 220), 'lnd10': (220, 223), 
            'lnd11': (223, 226), 'lnd12': (226, 229), 'lnd13': (229, 232), 'lnd14': (232, 235), 
            'lnd15': (235, 238), 'lnd16': (238, 241), 'lnd17': (241, 244), 'lnd18': (244, 247), 
            'lnd19': (247, 250), 'lnd20': (250, 253), 'lnd21': (253, 256), 'lnd22': (256, 259), 
            'lnd23': (259, 262), 'lnd24': (262, 265)}

idxs = {'r_Ph': (0, 3), 'v_Ph': (3, 6), 'r_sc0': (6, 9), 'r_sc1': (12, 15), 'r_sc2': (18, 21), 
        'r_sc3': (24, 27), 'r_sc4': (30, 33), 'r_sc5': (36, 39), 'v_sc0': (9, 12), 'v_sc1': (15, 18), 
        'v_sc2': (21, 24), 'v_sc3': (27, 30), 'v_sc4': (33, 36), 'v_sc5': (39, 42), 'Cr': (42, 43), 
        'GM': (43, 44), 'desat': (44, 62), 'cfs': (62, 110), 'emp': (110, 113), 'r_GS': (113, 122), 
        'range_bias': (122, 125), 'vlbi_bias': (125, 131), 'optical_bias': (131, 181), 'lnd0': (181, 184),
        'lnd1': (184, 187), 'lnd2': (187, 190), 'lnd3': (190, 193), 'lnd4': (193, 196), 'lnd5': (196, 199),
        'lnd6': (199, 202), 'lnd7': (202, 205), 'lnd8': (205, 208), 'lnd9': (208, 211), 'lnd10': (211, 214),
        'lnd11': (214, 217), 'lnd12': (217, 220), 'lnd13': (220, 223), 'lnd14': (223, 226), 'lnd15': (226, 229),
        'lnd16': (229, 232), 'lnd17': (232, 235), 'lnd18': (235, 238), 'lnd19': (238, 241), 'lnd20': (241, 244),
        'lnd21': (244, 247), 'lnd22': (247, 250), 'lnd23': (250, 253), 'lnd24': (253, 256)}

des = idxs['desat'][0]
# select entries you dont want
delete=['desat','r_GS','optical_bias','range_bias','vlbi_bias']
for i in range(1,narcs):
    delete.append('r_sc'+str(i))
    delete.append('v_sc'+str(i))
for i in range(25):
    delete.append('lnd'+str(i))
# select columns on entries you dont want and delete 
delete_idxs = []
for d in delete:
    delete_idxs += list(np.arange(idxs[d][0],idxs[d][1]))
    del idxs[d]

# insert coefficient names instead of cfs
cfs = {'C1': 0, 'C2': 2, 'C3': 5, 'C4': 9, 'C5': 14, 'C6': 20,
       'S1': 27, 'S2': 28, 'S3': 30, 'S4': 33, 'S5': 37,  'S6': 42}

for c, (k,v) in enumerate(cfs.items()):
    idxs[k] = (idxs['cfs'][0]+v, idxs['cfs'][0]+v+1)
del idxs['cfs']

new_cor_mat = np.zeros((cor_mat.shape[1]-len(delete_idxs),
                        cor_mat.shape[1]-len(delete_idxs)))
for i in range(1):
    dummy = np.delete(cor_mat, delete_idxs, axis = 0)
    new_cor_mat = np.delete(dummy, delete_idxs, axis = 1)

# new dictionary with labels
idxss = dict()
for k,v in idxs.items():
    if v[0] >= 12: #r sc 1
        idxss[k] = v[0]-(narcs-1)*6 # all states I dont want
    else:
        idxss[k] = v[0]

for k,v in idxss.items():       
    if v >= des - (narcs-1)*6: #44-30: # desat
        idxss[k] = v-narcs*3 # len desat


# tri_0 = new_cor_mat
tri_26 = new_cor_mat

#%% Lower triangle: close orbits, upper: far orbits

fig = plt.figure(figsize=(15,15))
grid = ImageGrid(fig, 111, nrows_ncols=(1,1),
                 axes_pad=0.6, share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                label_mode = 'all',
                cbar_pad=0.1,
                aspect = False)
ax = grid.axes_all[0]
# triangle = np.abs(np.triu(np.mean(new_cor_mat[0:int(orbs/2)],axis = 0)) + 
#                 np.tril(np.mean(new_cor_mat[int(orbs/2):],axis = 0)))
triangle = np.abs(np.tril(tri_0) +  np.triu(tri_26))
for i in range(len(triangle)):
    triangle[i,i] -= 1
im = ax.imshow(triangle, aspect='auto', interpolation='none')
cb = fig.colorbar(im, cax=ax.cax)

ax.set_yticks(list(idxss.values()))
ax.set_yticklabels(list(idxss.keys()))
ax.set_xticks(list(idxss.values()))
ax.set_xticklabels(list(idxss.keys()), rotation=90)

ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=90,
         ha="left", va="center",rotation_mode="anchor")

pareto = np.load('pareto_reduced_f5.npy')

orb = 0
ax.set_ylabel('$\overline{r} = $'+str(int(pareto[orb, 0]))+'km, '
                +'$\overline{euc} = $'+
                str(round(pareto[orb, 1],2)) +
                ', '+'$\lambda_n = $' + str(round(pareto[orb, 2],1)))    
orb = 26
ax.set_xlabel('$\overline{r} = $'+str(int(pareto[orb, 0]))+'km, '
                +'$\overline{euc} = $'+
                str(round(pareto[orb, 1],2)) +
                ', '+'$\lambda_n = $' + str(round(pareto[orb, 2],1)))
ax.xaxis.set_label_position('top') 

#%% Correlation Matrix for Combined Solution
orbs = 10
cor_mat = np.abs(np.load('cor_grad.npy'))
# cor_mat = np.abs(np.load('cor_mat.npy'))

idxss = {'C1': 0, 'C2': 2, 'C3': 5,  'C4': 9,  'C5': 14, 'C6': 20,  'C7': 27, 'C8': 35, 'C9': 44, 'C10': 54,
        'S1': 65,   'S3': 68, 'S4': 71,  'S5': 75, 'S6': 80, 'S7': 86,  'S8': 93, 'S9': 101, 'S10': 110}

#%% Lower triangle: close orbits, upper: far orbits

fig = plt.figure(figsize=(15,15))
grid = ImageGrid(fig, 111, nrows_ncols=(1,1),
                 axes_pad=0.6, share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                label_mode = 'all',
                cbar_pad=0.1,
                aspect = False)
ax = grid.axes_all[0]
# triangle = (np.tril(np.mean(cor_mat[0:int(orbs/2)],axis = 0)) + 
#                 np.triu(np.mean(cor_mat[int(orbs/2):],axis = 0)))
triangle = (np.tril(cor_mat[0]) +  np.triu(cor_mat[1]))
for i in range(len(triangle)):
    triangle[i,i] -= 1
im = ax.imshow(triangle, aspect='auto', interpolation='none')
im.set_clim(0, 1)
cb = fig.colorbar(im, cax=ax.cax)

ax.set_yticks(list(idxss.values()))
ax.set_yticklabels(list(idxss.keys()))
ax.set_xticks(list(idxss.values()))
ax.set_xticklabels(list(idxss.keys()), rotation=-60)


ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=60,
          ha="left", va="center",rotation_mode="anchor")

orb = 0
ax.set_ylabel('$\overline{r} = $'+str(int(pareto[orb, 0]))+'km, '
                +'$\overline{euc} = $'+
                str(round(pareto[orb, 1],2)) +
                ', '+'$\lambda_n = $' + str(round(pareto[orb, 2],1)))    
orb = 26
ax.set_xlabel('$\overline{r} = $'+str(int(pareto[orb, 0]))+'km, '
                +'$\overline{euc} = $'+
                str(round(pareto[orb, 1],2)) +
                ', '+'$\lambda_n = $' + str(round(pareto[orb, 2],1)))
ax.xaxis.set_label_position('top') 

