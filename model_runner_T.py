from model.parameter_search import run_experiments
from datetime import datetime
import logging
logging.basicConfig(filename='model_runner_T.log', encoding='utf-8', level=logging.DEBUG)
from multiprocessing import Process
import os.path
import pandas as pd

##########
# consistent parameters
pop_size = 100
opinion_dims = 1
num_clusters = 2
graph_type = 'complete'
param_count = 20
num_iterations = 1000
num_simulations = 10
num_processes = 2
add_to_existing = True

##########
# run simulations for T
##########
vals_a = [0.01,0.1 ,0.25,0.01,0.1 ,0.25,0.01,0.1 ,0.25]
vals_b = [0.01,0.01,0.01,0.1 ,0.1 ,0.1 ,0.25,0.25,0.25]
group_methods = ['hdb']*len(vals_a)

logging.info('\n\nT simulations with '+group_methods[0]+' grouping method\n')
logging.info('Pop. Size: %s, # Iters.: %s, # Sims.: %s'%(pop_size,num_iterations,num_simulations))
logging.info('Parameter Grid Granularity: %s, Parallel Processes: %s\n'%(param_count,num_processes))
logging.info('Macro Grid Values: %s,%s\n'%(vals_a,vals_b))


# start time
print('\nStart: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
logging.info(('Start: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

def val_selection(R_vals,E_vals,g_methods):
    for R, E, g_method in zip(R_vals,E_vals,g_methods):

        fname = 'sim_data/'+g_method+'_T_sim_R'+str(R)+'_E'+str(E)+'.pkl'
        if os.path.isfile(fname):
            if add_to_existing:
                df = pd.read_pickle(fname)
                existing_simulations = df['Simulation'].max() + 1
            else:
                print('%s already exists, nothing to be added'%(fname))
                continue
        else:
            existing_simulations = 0
            df = pd.DataFrame(columns=['T_in','T_out','R_in','R_out','E_in','E_out','Simulation','DER','STD',
                                        'x','Group History'])
        
        df_new = run_experiments(pop_size=pop_size, opinion_dims=opinion_dims, num_clusters=num_clusters,
                               graph_type=graph_type, R_in=R, R_out=R, E_in=E, E_out=E, group_dependent_param='T', group_method=g_method,
                               param_min=None, param_max=1, param_count=param_count,
                               num_iterations=num_iterations, num_simulations=num_simulations,existing_simulations=existing_simulations)
        # store
        df = pd.concat([df,df_new],ignore_index=True)
        df.to_pickle(fname)

# run parallel processes
vals_per_process = max(int(len(vals_a)/num_processes),1)
processes = []
for i in range(num_processes):
    if i+1==num_processes:
        range = [i*vals_per_process,None]
    else:
        range = [i*vals_per_process,(i+1)*vals_per_process]
    processes.append(Process(target=val_selection, args=(vals_a[range[0]:range[1]],vals_b[range[0]:range[1]],group_methods[range[0]:range[1]])))
    
for i, p in enumerate(processes):
    p.start()
    print('Process '+str(i)+' : '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

for p in processes:
    p.join()

# end time
print('End  : '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
logging.info(('End  : '+datetime.now().strftime('%Y-%m-%d %H:%M:%S')))