"""
copyright (c) 2017 Mahmood Hoseini

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.

"""
import numpy as np

def auxiliary(N, ntsteps, rate, dt) :
         
    spk_train = np.random.random((N, ntsteps))
    spk_train[spk_train <= rate*dt] = 1;
    spk_train[(spk_train > rate*dt) & (spk_train != 1)] = 0;
    
    return spk_train


def ext_input(N_I, N_E, r_I, r_E, stim_factor, t_max, t_stim, dt) :
    """ Step function external input."""
    ext_input_I = np.concatenate([auxiliary(N_I, int(t_stim/dt), r_I, dt*1e-3), 
                                  auxiliary(N_I, int((t_max-t_stim)/dt), r_I*stim_factor, dt*1e-3)], axis=1)
    ext_input_E = np.concatenate([auxiliary(N_E, int(t_stim/dt), r_E, dt*1e-3), 
                                  auxiliary(N_E, int((t_max-t_stim)/dt), r_E*stim_factor, dt*1e-3)], axis=1)
        
    return ext_input_I, ext_input_E
    
def ext_input2(N_I, N_E, p, r_I, r_E, stim_factor, t_max, t_stim, dt) :
    " Step function external input to a portion of the neurons."
    ext_input_I = np.zeros((N_I, t_max/dt))
    r = np.random.permutation(N_I)[:p*N_I]
    ext_input_I[r, :] = np.concatenate([auxiliary(p*N_I, t_stim/dt, r_I, dt*1e-3), 
                                  auxiliary(p*N_I, (t_max-t_stim)/dt, r_I*stim_factor, dt*1e-3)], axis=1)
    
    ext_input_E = np.zeros((N_E, t_max/dt))
    r = np.random.permutation(N_E)[:p*N_E]
    ext_input_E[r, :] = np.concatenate([auxiliary(p*N_E, t_stim/dt, r_E, dt*1e-3), 
                                  auxiliary(p*N_E, (t_max-t_stim)/dt, r_E*stim_factor, dt*1e-3)], axis=1)
        
    return ext_input_I, ext_input_E
    
def ext_input3(CI, CE, rI, rE, dt) :
    EII = np.random.random(CI)
    EII[EII <= rI*dt*1e-3] = 1;
    EII[(EII > rI*dt*1e-3) & (EII != 1)] = 0;
    
    EIE = np.random.random(CE)
    EIE[EIE <= rE*dt*1e-3] = 1;
    EIE[(EIE > rE*dt*1e-3) & (EIE != 1)] = 0;
    
    return EII, EIE
    
