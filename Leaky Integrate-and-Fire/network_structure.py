
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
from numpy import sin as S
from numpy import cos as C
from numpy import pi
from numpy import matrix as npm
import pylab

################################################
############ Random network ####################
################################################
def MatMaker (n1, n2, prob) :
    A = np.random.rand(n1, n2)
    A[A > prob] = 0;
           
    return A

def Random_network (N_I, N_E, pii, pie, pei, pee) :
    '''This functions recieves 6 inputs and returns synaptic weight matrices.
       Inputs :
               N_I        : Number of interneurons
               N_E        : Number of pyramidals
               pii, ...   : Connection probability for II, EE, EI, and IE connections
       Outputs :
               Four synaptic weight matrices each for one type of connection
    '''
    W_II = MatMaker(N_I, N_I, pii); np.fill_diagonal(W_II, 0)
    W_EE = MatMaker(N_E, N_E, pee); np.fill_diagonal(W_EE, 0)
    W_EI = MatMaker(N_E, N_I, pei)        
    W_IE = MatMaker(N_I, N_E, pie)
    
    return W_II, W_EE, W_IE, W_EI

def Hetro_clusters(sizeI, sizeE, p, coeff) :
    '''This functions recieves two arrays of the same length one for inh and
       the other for pyr. Each list contains the number of neurons in that cluster.
       Returns the synaptic weight matrices with inter cluster connection probability
       p and intra-cluster connection probability p*coeff.
       Inputs :
               sizeI : List of cluster sizes for interneurons
               sizeE : List of cluster sizes for pyramidals
               p     : Intra-cluster connection probability
               coeff : Clustering ratio (=pin/pout)
       Outputs :
               Eight synaptic weight matrices each for one type of connection 
               (4 inter and 4 intra)
    '''
    N_I = sizeI.sum(); N_E = sizeE.sum()

    W_IIin = np.zeros((N_I, N_I));
    W_IIout = MatMaker(N_I, N_I, p*coeff)
    for ii in range(1, len(sizeI)+1) :
        end = sizeI[:ii].sum();
        begin = end - sizeI[ii-1];
        W_IIin[begin:end, begin:end] = MatMaker(sizeI[ii-1], sizeI[ii-1], p);
        W_IIout[begin:end, begin:end] = np.zeros((sizeI[ii-1], sizeI[ii-1])); 
    np.fill_diagonal(W_IIin, 0)

    W_EEin = np.zeros((N_E, N_E));
    W_EEout = MatMaker(N_E, N_E, p*coeff)
    for ii in range(1, len(sizeE)+1) :
        end = sizeE[:ii].sum();
        begin = end - sizeE[ii-1];
        W_EEin[begin:end, begin:end] = MatMaker(sizeE[ii-1], sizeE[ii-1], p);
        W_EEout[begin:end, begin:end] = np.zeros((sizeE[ii-1], sizeE[ii-1])); 
    np.fill_diagonal(W_EEin, 0)
    
    W_IEin = np.zeros((N_I, N_E));
    W_IEout = MatMaker(N_I, N_E, p*coeff)
    for ii in range(1, len(sizeE)+1) :
        end = sizeI[:ii].sum();
        begin = end - sizeI[ii-1];
        endE = sizeE[:ii].sum();
        beginE = endE - sizeE[ii-1];
        W_IEin[begin:end, beginE:endE] = MatMaker(sizeI[ii-1], sizeE[ii-1], p);
        W_IEout[begin:end, beginE:endE] = np.zeros((sizeI[ii-1], sizeE[ii-1]));
    
    W_EIin = np.zeros((N_E, N_I));
    W_EIout = MatMaker(N_E, N_I, p*coeff)
    for ii in range(1, len(sizeE)+1) :
        end = sizeE[:ii].sum();
        begin = end - sizeE[ii-1];
        endI = sizeI[:ii].sum();
        beginI = endI - sizeI[ii-1];
        W_EIin[begin:end, beginI:endI] = MatMaker(sizeE[ii-1], sizeI[ii-1], p);
        W_EIout[begin:end, beginI:endI] = np.zeros((sizeE[ii-1], sizeI[ii-1])); 
    
    return W_IIin, W_IIout, W_EEin, W_EEout, W_IEin, W_IEout, W_EIin, W_EIout
 
##############################################       
########### Small world network ##############
##############################################
# McDonnell, Mark D., et al. "Input-rate modulation of gamma oscillations is sensitive 
# to network topology, delays and short-term plasticity." Brain research 1434 (2012)
def Ring_net(rows, cols, num_conn) :
    ''' This functions makes a ring-like network.'''
    W = np.zeros((rows, cols))
    vec = np.zeros(cols)
    vec[1:num_conn/2 + 1] = 2
    vec[cols-num_conn/2 : cols] = 2
    for ii in range(rows) :
        W[ii , :] = vec*np.random.random(cols)
        vec = np.roll(vec, 1)
    return W

def Rewiring(matrix, p_rw) :
    ''' This function gets a matrix and rewire its connections with p_rw prob.'''
    [n_rows, n_cols] = np.shape(matrix)
    for ii in range(n_rows) :
        for jj in range(n_cols) :
            if (matrix[ii, jj] != 0 and np.random.rand() < p_rw) :
                    matrix[ii, jj] = 0
                    Bool = True
                    while (Bool) :
                        m = np.random.randint(0, n_cols)
                        if matrix[ii, m] == 0 and m != ii :
                            matrix[ii, m] = np.random.random()
                            Bool = False
    return matrix


def Small_world_forward_backward_network(N_I, N_E, k_I, k_E, p_rw) :
    '''This functions recieves number of inh and exc neurons, along with number
       of inh/exc inputs to each individual neuron (k_I/k_E) and also rewiring
       probability. Returns the synaptic weight matrices.
       Inputs :
               N_I : number of interneurons
               N_E : number of pyramidals
               p_rw: rewiring probability
       Outputs :
               Four synaptic weight matrices each for one type of connection 
    '''
    W_II = Ring_net(N_I, N_I, k_I)
    W_II = Rewiring(W_II, p_rw);

    W_EE = Ring_net(N_E, N_E, k_E)
    W_EE = Rewiring(W_EE, p_rw);

    W_IE = np.zeros((N_I, N_E))
    vec = np.zeros(N_I)
    vec[:k_E/2] = 2; vec[-k_E/2:] = 2
    for i in range(N_E) :
        W_IE[:, i] = vec*np.random.random(N_I)
        if (i+1)%4 == 0 :
            vec = np.roll(vec, 1)
    W_IE = np.roll(W_IE, 2, axis=1);
    W_IE = Rewiring(W_IE, p_rw);
    
    W_EI = np.zeros((N_E, N_I))
    vec = np.zeros(N_E)
    vec[:k_I/2+2] = 2; vec[-k_I/2+2:] = 2
    for i in range(N_I) :
        W_EI[:, i] = vec*np.random.random(N_E)
        vec = np.roll(vec, 4)
    W_EI = Rewiring(W_EI, p_rw);
                            
    return W_II, W_EE, W_IE, W_EI
 
######## Clustering coefficient and mean path length functions   
def Clustering_coeff(M) :
    N = npm(M)*npm(M)
    SS = 0 # Summation of matrix N with diagonal elements excluded
    PP = 0 # Like S while the nodes are directly connected
    for ii in range(len(M[:,1])) :
        for jj in range(len(M[:, 1])) :
            SS += N[ii, jj]
            if M[ii, jj] != 0 :
                PP += N[ii, jj]
    return PP/float(SS)
            
def Mean_shortest_path(M) :
    L = []
    num_paths = []
    paths = npm(M)
    binary = np.ones(np.shape(M)) - np.eye(len(M[0, :]))
    for cnt in range(1, len(M[0, :])) :
        NZ = np.array(paths[binary != 0])
        L.append(sum(NZ[0])*cnt)
        num_paths.append(sum(NZ[0]))
        binary = binary - paths
        binary[binary != 1] = 0
        if binary.sum() == 0 :
            break
        paths *= M
        paths[paths != 0] = 1
    return sum(L)/float(sum(num_paths))
   
########### Small world network mean path length and CC vs. rewiring p
def TestSWN(N, k) :
    W, W1, W1, W1 = Small_world_forward_backward_network(N, 4*N, k, 4*k, 0)
    W[W != 0] = 1;
    L0 = Mean_shortest_path(W)
    CC0 = Clustering_coeff(W)
    
    p_rw = np.logspace(-4, 0, 30)
    relL = np.zeros(len(p_rw))
    relCC = np.zeros(len(p_rw))
    for index in range(len(p_rw)) :
        print index
        W, W1, W1, W1 = Small_world_forward_backward_network(N, 4*N, k, 4*k, p_rw[index])
        relL[index] = Mean_shortest_path(W)/L0
        relCC[index] = Clustering_coeff(W)/CC0
        
    ##### Plotting
    pylab.figure(); ax = pylab.gca()
    ax.semilogx(p_rw, relL, 'bs')
    ax.semilogx(p_rw, relCC, 'ro')
    ax.set_xlabel('Rewiring probability')
    ax.legend(('<L>', '<CC>'), prop={'size':14})
    return p_rw, relL, relCC
 
####################################################   
############### Neurons on sphere ##################
####################################################
def spherical_dist(polar1, azim1, polar2, azim2) :
    '''This function recieves polar and azimuthal angles and calculates the
        distance between two points on the sphere.
    '''
    dist = np.arccos(S(polar1)*C(azim1)*S(polar2)*C(azim2) +\
                     S(polar1)*S(azim1)*S(polar2)*S(azim2) + C(polar1)*C(polar2))
    return dist

def dist2elec(polar_ang, azim_ang) :
    elec_polars = [0, pi, pi/3, 2*pi/3, pi/3, 2*pi/3, pi/3, 2*pi/3,   pi/3, 2*pi/3];
    elec_azims =  [0,  0,    0,      0, pi/2,   pi/2,   pi,     pi, 3*pi/2, 3*pi/2];
    N = len(polar_ang)
    D = np.zeros((N, len(elec_polars)));
    for nrn in range(N) :
        for e in range(len(elec_polars)) :
            D[nrn, e] = spherical_dist(polar_ang[nrn], azim_ang[nrn], elec_polars[e], elec_azims[e])
    return D
    
############
def clusters_on_sphere(N_I, N_E, p, sigma, p_rw, IEflag=False) :
    ''' This function puts all neurons randomly on the surface of a unit 
        sphere and uses a two dimensional Gaussian probability distribution
        to connect neurons. It rules out connections that are outside two
        times standard deviation. Finally it rewires connections to add long
        range connections.
        Inputs :
               N_I   : number of interneurons
               N_E   : number of pyramidals
               p     : connection probability
               sigma : std of the gausian function (<1)
               p_rw  : rewiring probability
               IEflag: clustering from Exc to Inh neurons
        Outputs :
               Four synaptic weight matrices each for one type of connection
               and also the location of Exc neurons on the sphere.
    '''
    Ncells = N_E + N_I
    W_II = MatMaker(N_I, N_I, p*2); np.fill_diagonal(W_II, 0)
    W_EI = MatMaker(N_E, N_I, p*2)        
    
    p_aux = p/(1 - 1/(sigma*np.sqrt(8*pi))); # Larger p for thining purposes
    W_EE = MatMaker(N_E, N_E, p_aux); np.fill_diagonal(W_EE, 0)
    polar_ang = np.random.normal(pi/2, 0.75, size=Ncells);
    polar_ang[polar_ang < 0] = 0; polar_ang[polar_ang > pi] = pi;
    azim_ang = 2*pi*np.random.random(Ncells)

    dist_to_elec = dist2elec(polar_ang, azim_ang)
    ## Thining!
    for row in range(N_E) :
        for col in range(row+1, N_E) :
            dist = spherical_dist(polar_ang[row], azim_ang[row], polar_ang[col], azim_ang[col])
            same_cluster = 0; # Assume nrns don't belong to the same elec unless next if
            if (np.argmin(dist_to_elec[row, :]) == np.argmin(dist_to_elec[col, :])) :
                same_cluster = 1;
            
            if (same_cluster==0 or W_EE[row, col]>np.exp(-dist**2/2/sigma**2) or dist>2*sigma) :
                W_EE[row, col] = 0
            else :
                W_EE[row, col] = np.random.gamma(2, 2)/2/2/pi

            if (same_cluster==0 or W_EE[col, row]>np.exp(-dist**2/2/sigma**2) or dist>2*sigma) :
                W_EE[col, row] = 0
            else :
                W_EE[col, row] = np.random.gamma(2, 2)/2/2/pi
    
    W_EE = Rewiring(W_EE, p_rw); 
     
    #######  Exc -> Inh                       
    W_IE = MatMaker(N_I, N_E, p*2)
    if IEflag :
        for row in range(N_I) :
            for col in range(N_E) :
                dist = spherical_dist(polar_ang[row+N_E], azim_ang[row+N_E], polar_ang[col], azim_ang[col])
                same_cluster = 0; # Assume nrns don't belong to the same elec unless next if
                if (np.argmin(dist_to_elec[row, :]) == np.argmin(dist_to_elec[col, :])) :
                    same_cluster = 1;
            
                if (same_cluster==0 or W_IE[row, col]>np.exp(-dist**2/2/sigma**2) or dist>2*sigma) :
                    W_IE[row, col] = 0
                else :
                    W_IE[row, col] = np.random.gamma(2, 2)/2/2/pi
                    
        W_IE = Rewiring(W_IE, p_rw);
                            
    return W_II, W_EE, W_IE, W_EI, polar_ang, azim_ang

    
########### Clustered network mean path length and CC vs. rewiring p 
def TestCN(N, p, sigma) :
    ''' This function calculates the ratio of mean path length and clustering
        coefficient of a rewired clustered network to a non-rewired one.
        Inputs :
            N     : Number of neurons
            p     : connection probability
            sigma : std of gaussian function
        Outputs :
            p_rw  : rewiring probabilities
            relL  : relative mean path length
            relCC : relative clustering coefficient
    '''
    W1, W0, W1, W1, ang, ang = clusters_on_sphere(0, N, p, sigma, 0)
    W0[W0 != 0] = 1;
    L0 = Mean_shortest_path(W0)
    CC0 = Clustering_coeff(W0)
    print L0, CC0
    p_rw = np.logspace(-4, 0, 30)
    relL = np.zeros(len(p_rw))
    relCC = np.zeros(len(p_rw))
    for index in range(len(p_rw)) :
        print index
        #W1, W, W1, W1, ang, ang = clusters_on_sphere(N/4, N, p, sigma, p_rw[index])
        W = Rewiring(W0, p_rw[index])
        relCC[index] = Clustering_coeff(W)/CC0
        relL[index] = Mean_shortest_path(W)/L0
        print relL[index], relCC[index]
        print '###########'
    ##### Plotting
    pylab.figure(); ax = pylab.gca()
    ax.semilogx(p_rw, relL, 'bs')
    ax.semilogx(p_rw, relCC, 'ro')
    ax.set_xlabel('Rewiring probability')
    ax.legend(('<L>', '<CC>'), prop={'size':14})
    return p_rw, relL, relCC, W0, W
    
    
######### Setting up electrode array and getting LFPs
def MEA_setup(polar_ang, azim_ang) :
    ''' This function sets up the MEA of 10 equidistant electrodes. Using location of
        the neurons on sphere, it determine the contribution of each neuron on LFP
        signals (cells withing electrode basin).
        Inputs :
               polar_ang   : polar angle of individual neurons
               azim_ang    : azimuthal angle of individual neurons
        Outputs :
               cells_membership : a dict of cell number arrays that contribute to each electrode
               inter_elec_dist  : Inter-electrode distance matrix
    '''
    elec_polars = [0, pi, pi/3, 2*pi/3, pi/3, 2*pi/3, pi/3, 2*pi/3,   pi/3, 2*pi/3];
    elec_azims =  [0,  0,    0,      0, pi/2,   pi/2,   pi,     pi, 3*pi/2, 3*pi/2];
     
    Nelec = len(elec_polars)
    Ncells = len(polar_ang)                
    inter_elec_dist = np.zeros((Nelec, Nelec))
    cells_membership = {}
    cells_membership['E'] = {}
    cells_membership['I'] = {}
    elec_basins = 0.7
    for ii in range(Nelec) :
        for jj in range(ii+1, Nelec) :
            inter_elec_dist[ii, jj] = spherical_dist(elec_polars[ii], elec_azims[ii], elec_polars[jj], elec_azims[jj])
            inter_elec_dist[jj, ii] = inter_elec_dist[ii, jj]
         
        cells_membership['E'][ii] = [];
        cells_membership['I'][ii] = [];
        for cell in range(Ncells) :
            d = spherical_dist(elec_polars[ii], elec_azims[ii], polar_ang[cell], azim_ang[cell])
            if (d < elec_basins and cell < 4*Ncells/5) :
                cells_membership['E'][ii].append(cell)
            elif (d < elec_basins and cell >= 4*Ncells/5) :
                cells_membership['I'][ii].append(cell-4*Ncells/5)
                
    return cells_membership, inter_elec_dist
    

def plot_weight_matrix () :
    W1, W_EE, W1, W1, pang, aang = clusters_on_sphere(200, 800, 0.2, 0.5, 0.02)
    cm, d = MEA_setup(pang, aang)
    
    nte = 0
    for key in cm['E'].keys() :
        for item in cm['E'][key] : nte+=1

    CW = np.zeros((nte, nte))
    cnt1 = -1
    for key1 in cm['E'].keys() :
        for item1 in cm['E'][key1] : 
            cnt1+=1; cnt2=0
            for key2 in cm['E'].keys() :
                for item2 in cm['E'][key2] :
                    CW[cnt1, cnt2] = W_EE[item1, item2]; 
                    cnt2+=1
    pylab.imshow(CW)