# Overview
This repository contains the MATLAB and python codes used to simulate a network of leaky integrate-and-fire neurons according to the discriptions in Hoseini et al., Dynamics and sources of response variability and its coordination in visual cortex (2019).

Our goal is to provide full transparency and to enable readers to read it and find out the exact details of what was done.

# Description of the modules
1. ext_input.py: This module generates the independent thalamocortical Poissonian inputs. The stimulus was modeled as a sudden increase at stimulus onset time.
2. network_structure.py: This module generates 4 synaptic weight matrices that crosspond to 4 types of connections (inh->inh, inh->exc, exc->exc, exc->inh) for 3 topologies: i) random all-to-all, ii) two-dimensional small-world (see McDonnell, Mark D., et al. "Input-rate modulation of gamma oscillations is sensitive to network topology, delays and short-term plasticity." Brain research 1434 (2012)), and iii) three-dimensional small-world (see Materials and Methods).
3. main_LIF.py: Runs a network specified parameters and saves/plots outputs.

# License
The code in this repository by M. Hoseini is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License

Note that the dataset associated with the paper is licensed under a Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License. This means you are free to use the data to reproduce and validate our results.
