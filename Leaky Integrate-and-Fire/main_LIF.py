"""
copyright (c) 2017 Mahmood Hoseini

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.

"""

import numpy as np, pylab as plt, time
import network_structure, os, ext_input, scipy.io
from numpy import transpose as npt
plt.ion(); start_time = time.time()

path = '/home/mahmoodhoseini/Desktop/LIF-onSphere/'; # Specifiy the path here
os.chdir(path)

################################################### Parameters
tau_Im, tau_Em = 25, 50 # Membrane time constant in ms for [inh, pyr]
C_I, C_E = 0.20, 0.40 # Capacity for [inh, pyr] neurons in nF
N_I, N_E = 200, 800 # Number of [inh, pyr]
p = 0.1 # Connection probability
tau_ref_I, tau_ref_E = 1, 2 #Absolute refractory period for [inh, pyr] neurons in ms
V_leakI = -70+10*np.random.random((N_I, 1));
V_leakE = -70+10*np.random.random((N_E, 1));
V_th = -40 # [Resting(leak), Threshold] membrane potential for neurons in mV
V_reset = -59; V_peak = 0 # Reset membrane and peak spike voltage respectively in mV
V_synI, V_synE = -68, 50 # Synaptic reversal potential for [inh, exc] in mV
V_K = -80 # SRA reversal potential

dt = 0.05 # Time step in ms
t_max = 5500 # Total simulation time in ms
t_stim = 2500 # Stimulus onset time in ms
num_trials = 20 # Number of trials

g_AE = 2.0e-3 # Synaptic conductance for AMPA channels on pyramidals in uS
g_AI = 6.0e-3 # Synaptic conductance for AMPA channels on interneurons in uS
g_GE = 20.0e-3 # Synaptic conductance for GABA channels on pyramidals in uS
g_GI = 10.0e-3 # Synaptic conductance for GABA channels on interneurons in uS
g_extI, g_extE = 4.0e-3, 4.0e-3; # Synaptic conductance for ext input on [inh, pyr] in uS
g_leakI, g_leakE = 5e-3, 10e-3; # Leak conductance for [inh, pyr] neurons in uS

tau_Gl, tau_Gr, tau_Gd = 1.5, 1.5, 6.0 # Inh -> [Inh, Exc] times in ms
tau_Al, tau_Ar, tau_Ad = 1.5, 0.2, 1.0 # Exc -> [Inh, Exc] times in ms
tau_sra = 20 # Spike rate adaptation recovery time constant in ms

tauR, tauD = 30000.0, 400.0; # SD time scales in time steps
rateI, rateE = 175, 175; stim_factor = 5;

################################################### Synaptic weight matrices
WIIo, WEEo, WIEo, WEIo, polar_ang, azim_ang = network_structure.clusters_on_sphere(N_I, N_E, p, 0.5, 0.5) # Neurons are
# randomly distributed on a sphere. For other topologies see network_structure.py module
cells_membership, inter_elec_dist = network_structure.MEA_setup(polar_ang, azim_ang)
r = cells_membership['E'][0][:20]

np.save(path + 'polar_ang.npy', polar_ang)
np.save(path + 'azim_ang.npy', azim_ang)
np.save(path + 'cell_membership.npy', cells_membership)
np.save(path + 'WEE.npy', WEEo)
np.save(path + 'WEI.npy', WEIo)
        
################################################### Equations of "motion"
extSD_flag = True; # Synaptic depression/recovery for thalamo-cortical inputs
intSD_flag = True; # Synaptic depression/recovery for cortico-cortical inputs
LFP_flag = True; # Calculating LFP traces
save_flag = True; # Save files in the specified path above
plot_flag = True; # Visualizing network output
for trial in range(1, num_trials+1) :
    print 'trial num:', str(trial)
    ExtI, ExtE = ext_input.ext_input(N_I, N_E, rateI, rateE, stim_factor, t_max, t_stim, dt)

    WII, WIE, WEI, WEE = WIIo, WIEo, WEIo, WEEo
    tWI = np.ones((N_I, 1)); tWE = np.ones((N_E, 1));
    ntsteps = int(t_max/dt); 
    g_Esra = np.zeros((N_E, ntsteps));
    V_I = np.zeros((N_I , ntsteps)) + V_reset;
    V_E = np.zeros((N_E , ntsteps)) + V_reset;
    X_GI = np.zeros((N_I, 2)); S_GI = np.zeros((N_I, ntsteps))
    X_AI = np.zeros((N_I, 2)); S_AI = np.zeros((N_I, ntsteps))
    X_extI = np.zeros((N_I, 2)); S_extI = np.zeros((N_I, ntsteps))
    X_GE = np.zeros((N_E, 2)); S_GE = np.zeros((N_E, ntsteps))
    X_AE = np.zeros((N_E, 2)); S_AE = np.zeros((N_E, ntsteps))
    X_extE = np.zeros((N_E, 2)); S_extE = np.zeros((N_E, ntsteps))
    STI = np.zeros((N_I, ntsteps)); STE = np.zeros((N_E, ntsteps))
    SCI = np.zeros((N_I, ntsteps)); SCE = np.zeros((N_E, ntsteps))
    mW = np.zeros((6, ntsteps-1));

    for t in range(1, int(t_max/dt)) :

        # Fast interneurons              
        X_AI[:, 1] = X_AI[:, 0]*(1 - dt/tau_Ar) + tau_Im*dt/tau_Ar*np.dot(WIE, STE[:, t-int(tau_Al/dt)])
        S_AI[:, t] = S_AI[:, t-1]*(1 - dt/tau_Ad) + dt/tau_Ad*X_AI[:, 0]
        X_GI[:, 1] = X_GI[:, 0]*(1 - dt/tau_Gr) + tau_Im*dt/tau_Gr*np.dot(WII, STI[:, t-int(tau_Gl/dt)])
        S_GI[:, t] = S_GI[:, t-1]*(1 - dt/tau_Gd) + dt/tau_Gd*X_GI[:, 0]
        X_extI[:, 1] = X_extI[:, 0]*(1 - dt/tau_Ar) + tau_Im*dt/tau_Ar*tWI[:, 0]*ExtI[:, t]
        S_extI[:, t] = S_extI[:, t-1]*(1 - dt/tau_Ad) + dt/tau_Ad*X_extI[:, 0]
    
        index  = np.where(sum(npt(STI[:, t-int(tau_ref_I/dt):t])) == 0)[0]
        SCI[index, t] =  (- g_extI*S_extI[index, t-1]*(V_I[index, t-1] - V_synE)\
                          - g_GI*S_GI[index, t-1]*(V_I[index, t-1] - V_synI)\
                          - g_AI*S_AI[index, t-1]*(V_I[index, t-1] - V_synE))       
        V_I[index, t] = V_I[index, t-1] + dt/C_I*(SCI[index, t] - g_leakI*(V_I[index, t-1] - V_leakI[index, 0]))
        V_I[np.where(V_I[:, t] >= V_th), t] = V_peak
        STI[np.where(V_I[:, t] >= V_th), t] = 1
        
        # Excitatory neurons
        X_AE[:, 1] = X_AE[:, 0]*(1 - dt/tau_Ar) + tau_Em*dt/tau_Ar*np.dot(WEE, STE[:, t-int(tau_Al/dt)])
        S_AE[:, t] = S_AE[:, t-1]*(1 - dt/tau_Ad) + dt/tau_Ad*X_AE[:, 0]
        X_GE[:, 1] = X_GE[:, 0]*(1 - dt/tau_Gr) + tau_Em*dt/tau_Gr*np.dot(WEI, STI[:, t-int(tau_Gl/dt)])
        S_GE[:, t] = S_GE[:, t-1]*(1 - dt/tau_Gd) + dt/tau_Gd*X_GE[:, 0]
        X_extE[:, 1] = X_extE[:, 0]*(1 - dt/tau_Ar) + tau_Em*dt/tau_Ar*tWE[:, 0]*ExtE[:, t]
        S_extE[:, t] = S_extE[:, t-1]*(1 - dt/tau_Ad) + dt/tau_Ad*X_extE[:, 0]
    
        index  = np.where(sum(npt(STE[:, t-int(tau_ref_E/dt):t])) == 0)[0]
        SCE[index, t] = (- g_extE*S_extE[index, t-1]*(V_E[index, t-1] - V_synE)\
                         - g_GE*S_GE[index, t-1]*(V_E[index, t-1] - V_synI)\
                         - g_AE*S_AE[index, t-1]*(V_E[index, t-1] - V_synE)\
                         - g_Esra[index, t-1]*(V_E[index, t-1] - V_K))
        V_E[index, t] = V_E[index, t-1] + dt/C_E*(SCE[index, t] - g_leakE*(V_E[index, t-1] - V_leakE[index, 0]))
       
        V_E[np.where(V_E[:, t] >= V_th), t] = V_peak
        STE[np.where(V_E[:, t] >= V_th), t] = 1
        g_Esra[:, t] = g_Esra[:, t-1]*(1-dt/tau_sra) + 8e-3*STE[:, t]
        
        if extSD_flag == True:
            tWI = tWI + 1/tauR*(np.ones((N_I, 1))-tWI); # thalamocortical synapse recovery
            tWE = tWE + 1/tauR*(np.ones((N_E, 1))-tWE); # thalamocortical synapse recovery

            tWI[:, 0] = tWI[:, 0] - tWI[:, 0]*ExtI[:, t]/tauD; 
            tWE[:, 0] = tWE[:, 0] - tWE[:, 0]*ExtE[:, t]/tauD; 

            mW[0, t-1] = np.mean(tWI);
            mW[1, t-1] = np.mean(tWE);

        if intSD_flag == True :
            WII = WII + 1/tauR*(WIIo - WII); 
            WEI = WEI + 1/tauR*(WEIo - WEI); 
            WIE = WIE + 1/tauR*(WIEo - WIE); 
            WEE = WEE + 1/tauR*(WEEo - WEE); 

            NZ = np.nonzero(STI[:, t])
            WII[:, NZ] = WII[:, NZ] - WII[:, NZ]/tauD; 
            WEI[:, NZ] = WEI[:, NZ] - WEI[:, NZ]/tauD; 

            NZ = np.nonzero(STE[:, t])
            WIE[:, NZ] = WIE[:, NZ] - WIE[:, NZ]/tauD; 
            WEE[:, NZ] = WEE[:, NZ] - WEE[:, NZ]/tauD;

            mW[2, t-1] = np.mean(WII); mW[3, t-1] = np.mean(WEI);
            mW[4, t-1] = np.mean(WIE); mW[5, t-1] = np.mean(WEE);
        
        X_GI[:, 0] = X_GI[:, 1]; X_GI[:, 1] = 0;
        X_AI[:, 0] = X_AI[:, 1]; X_AI[:, 1] = 0;
        X_AE[:, 0] = X_AE[:, 1]; X_AE[:, 1] = 0;
        X_GE[:, 0] = X_GE[:, 1]; X_GE[:, 1] = 0;
        X_extI[:, 0] = X_extI[:, 1]; X_extI[:, 1] = 0;
        X_extE[:, 0] = X_extE[:, 1]; X_extE[:, 1] = 0;
            

    if LFP_flag == True :
        Nelec = len(inter_elec_dist[:, 0])
        LFPs = np.zeros((Nelec, ntsteps))
        for ii in range(Nelec) :
            LFPs[ii, :] = (np.sum(SCE[cells_membership['E'][ii], :], axis=0) +
                           np.sum(SCI[cells_membership['I'][ii], :], axis=0))
            
    ############################################### Saving results
    if save_flag == True :
        np.save(path + 'SAE'+str(trial)+'.npy', S_AE[:20, :])
        np.save(path + 'SGE'+str(trial)+'.npy', S_GE[:20, :])
        np.save(path + 'SextE'+str(trial)+'.npy', S_extE[:20, :])
        np.save(path + 'SAEr'+str(trial)+'.npy', S_AE[r, :])
        np.save(path + 'SGEr'+str(trial)+'.npy', S_GE[r, :])
        np.save(path + 'SextEr'+str(trial)+'.npy', S_extE[r, :])
        scipy.io.savemat(path + 'LFPs'+str(trial)+'.mat', {'lfp': LFPs})
        
################################################### Plotting
if plot_flag == True :
    fig = plt.figure(); ax = plt.gca()
    cnt = 0
    for ii in range(Nelec) :
        for jj in range(len(cells_membership['I'][ii])) :
            cnt += 1;
            for t in range(int(t_max/dt)) :
                if STI[cells_membership['I'][ii][jj], t] == 1 :
                    ax.vlines(t*dt, cnt, cnt+0.95, color='b');
    for ii in range(Nelec) :
        for jj in range(len(cells_membership['E'][ii])) :
            cnt += 1;
            for t in range(int(t_max/dt)) :
                if STE[cells_membership['E'][ii][jj], t] == 1 :
                    ax.vlines(t*dt, cnt, cnt+0.95, color='r');

    fig = plt.figure(); ax = plt.gca()
    ax.plot(g_extI*S_extI[1, :], 'y')
    ax.plot(g_GI*S_GI[1, :], 'b')
    ax.plot(g_AI*S_AI[1, :], 'r')
    plt.title('inputs to inhibitories')
    
    fig = plt.figure(); ax = plt.gca()
    ax.plot(g_extE*S_extE[1, :], 'c')
    ax.plot(g_GE*S_GE[1, :], 'b')
    ax.plot(g_AE*S_AE[1, :], 'r')
    ax.plot(np.mean(g_Esra, 0), 'k')
    plt.title('inputs to pyramidals')
    
    fig = plt.figure(); ax1 = plt.gca(); ax2 = ax1.twinx()
    ax1.plot(mW[0, :], 'k', label='ext->inh')
    ax1.plot(mW[1, :], 'k--', label='ext->exc')
    ax2.plot(mW[2, :], 'b', label='inh->inh')
    ax2.plot(mW[3, :], 'b--', label='inh->exc')
    ax2.plot(mW[4, :], 'r', label='exc->inh')
    ax2.plot(mW[5, :], 'r--', label='exc->exc')
    ax1.set_xlabel('Time step #'); plt.legend()
    ax1.set_ylabel('Ext source', color='k')
    ax2.set_ylabel('Within Net', color='g')

print "It took ", str(round((time.time() - start_time)/60.0/60.0, 2)) + " hrs to run"
