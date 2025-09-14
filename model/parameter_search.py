import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkit as nk
import model.opinion_simulation as o_s
from datetime import datetime
from model.create_data import create_opinions, create_graph

def history_check(history):
    """
    if history is exactly equal for some time span then we need not update opinions anymore
    """
    return (history[-2] == history[-1]).all()

def history_slim(list,length):
    """
    slim the data stored
    """
    return list[:length] + list[-length:]

def one_experiment(pop_size, opinion_dims, num_clusters, graph_type,
                      E_in, E_out, R_in, R_out, T_in, T_out, group_dependent_param, group_method,
                      num_iterations, num_simulations, simulations, existing_simulations):
    """
    run an experiment for some parameter values (includes multiple simulations of same parameter values)
    """
    for idz, sim in enumerate(range(num_simulations)):
        idz += existing_simulations
        sim += existing_simulations
        # create opinions
        x = create_opinions(pop_size,opinion_dims,num_clusters,random_seed=idz)
        # create graph
        G = create_graph(pop_size,graph_type='complete')
        # create simulation for parameters
        a = o_s.platform_simulation(x, G, group_method=group_method,
                            ideological_limit=100000, edge_creation_limit=0,
                            R_in=R_in,R_out=R_out,T_in=T_in,T_out=T_out,E_in=E_in,E_out=E_out,bounded=True)
        
        checkSpan = 0
        # run iterations with influence network
        for i in range(num_iterations):
            
            a.run_iteration(feed_mute = 'influence')
            
            if history_check(a.opinion_history[-2:]):
                checkSpan +=1
                if checkSpan >= 100:
                    print('!Opinions unchanged for 100 iterations! - simulation complete')
                    break
            else:
                checkSpan = 0
        
        DER = a.der()
        STD = a.std_dev()
        HIST = history_slim(a.opinion_history,20)
        GROUP = history_slim(a.group_history,20)
        simulations.append((T_in,T_out,R_in,R_out,E_in,E_out,sim,DER,STD,HIST,GROUP))

def run_experiments(pop_size=100, opinion_dims=1, num_clusters=2, graph_type='complete',
                      R_in=None, R_out=None, T_in=None, T_out=None, E_in=None, E_out=None,
                      group_dependent_param=None, group_method='hdb',
                      param_min=None, param_max=1, param_count=10,
                      num_iterations=20, num_simulations=5,existing_simulations=0):
    """
    simulate over different T_in and T_out values/R_in and R_out values/E_in and E_out values
    depending on which is the group dependent parameter
    """
    simulations = []
    # parameter values for group dependent parameter search
    if param_min == None:
        param_min = param_max/param_count
    param_range = np.linspace(param_min,param_max,param_count)

    for idx, param_in in enumerate(param_range):
        for param_out in param_range:
            # assign T/R as group dependent parameter
            if group_dependent_param == 'T':
                T_in, T_out = param_in, param_out
            elif group_dependent_param == 'R':
                R_in, R_out = param_in, param_out
            elif group_dependent_param == 'E':
                E_in, E_out = param_in, param_out

            # run one experiment
            one_experiment(pop_size, opinion_dims, num_clusters, graph_type,
                           E_in, E_out, R_in, R_out, T_in, T_out, group_dependent_param, group_method,
                           num_iterations, num_simulations,simulations,existing_simulations)

            print('%s\tE_in %.2f, E_out %.2f, R_in %.2f, R_out %.2f, T_in %.2f, T_out %.2f' %(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),E_in, E_out, R_in, R_out, T_in, T_out))
    
    df = pd.DataFrame(simulations, columns=['T_in','T_out','R_in','R_out','E_in','E_out','Simulation','DER','STD',
                                        'x','Group History'])
    
    return df
