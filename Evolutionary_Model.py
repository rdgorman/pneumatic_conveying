import pickle
import random
import array
import math
import platform
import sys
import glob
import requests
import copy

import System_Pressure_Drop_Functions as syspdf
import Miscellaneous_Functions as mf
import Coefficient_Solver as cf
import numpy as np

from deap import creator, base, tools, algorithms
from itertools import repeat
from pathlib import Path

if platform.system() == 'Windows':
    Delimiter = '\\'
else:
    Delimiter = '/'

class Evolutionary_Model:
    # This class uses an evolutionary model to find the best set of pneumatic conveying methods for the data set.
    class Miscellaneous_Functions:
        # This class contains miscellaneous functions used in both dilute and dense phase.
            
        def Generate_Individual(ind_cls, attrs):
            # This function generates individuals.
            ind = ind_cls(a() for a in attrs)
            return ind
        
        def checkBounds(mins, maxs):
            # This function checks the bounds of the offspring in the evolutionary model.
            def decorator(func):
                def wrapper(*args, **kargs):
                    offspring = func(*args, **kargs)
                    for child in offspring:
                        for i in range(len(child)):
                            if child[i] > maxs[i]:
                                child[i] = int(maxs[i])
                                # child[i] = int(child[i] %maxs[i])
                            elif child[i] < mins[i]:
                                child[i] = int(mins[i])
                                # child[i] = int(maxs[i] - (mins[i] - child[i]))
                    return offspring
                return wrapper
            return decorator
                        
        def Initial_Population(p_cls, ind_cls, Method_Maxes, Population_Size):
            # This function generates an initial population that contains all members of the populations.            
            max_method = max(Method_Maxes)
            population_size_total = max_method * Population_Size
            
            gene_pools = []
            for m in Method_Maxes:
                gene_pool = []
                
                for i in range(int(math.floor(population_size_total/m))):
                    for k in range(m):
                        gene_pool.append(k + 1)
                        
                for k in range(population_size_total - len(gene_pool)):
                    gene_pool.append(random.randint(1, m))
                
                random.shuffle(gene_pool)

                gene_pools.append(gene_pool)
                
            population = []
            for i in range(population_size_total):
                individual = ind_cls()
                for g in gene_pools:
                    individual.append(g[i])
                population.append(individual)
                          
            return p_cls(i for i in population)
    
        def Dump_Population(population, Dump_Location):
            # Dump population for external evaluation.
            for k, ind in zip(range(len(population)), population):
                with open(r'{}{}{}.pkl'.format(Dump_Location, Delimiter, k), 'wb') as filet:
                    pickle.dump(ind, filet)
                    
        def Dump_Individual(individual, Dump_Location):
            # Dump individual for external evaluation.
            p = Path(r'{}{}{}.pkl'.format(Dump_Location, Delimiter, individual))
            if not p.exists():
                with open(r'{}{}{}.pkl'.format(Dump_Location, Delimiter, individual), 'wb') as filet:
                    pickle.dump(list(individual), filet)
            
        def mean_filtered(fitnesses):
            # Mean of filtered data.
            fitnesses_filtered = [x for x in fitnesses if sum(x) < 1.e10]
            try:
                result = np.mean(fitnesses_filtered)
            except:
                result = np.NaN
            return result
            
        def std_filtered(fitnesses):
            # Standard deviation of filtered data.
            fitnesses_filtered = [x for x in fitnesses if sum(x) < 1.e10]
            try:
                result = np.std(fitnesses_filtered)
            except:
                result = np.NaN
            return result
            
        def min_filtered(fitnesses):
            # Minimum of filtered data.
            fitnesses_filtered = [x for x in fitnesses if sum(x) < 1.e10]
            try:
                result = np.min(fitnesses_filtered)
            except:
                result = np.NaN
            return result
            
        def max_filtered(fitnesses):
            # Maximum of filtered data.
            fitnesses_filtered = [x for x in fitnesses if sum(x) < 1.e10]
            try:
                result = np.max(fitnesses_filtered)
            except:
                result = np.NaN
            return result
            
        def count_errors(fitnesses):
            # Count error solutions.
            fitnesses_filtered = [x for x in fitnesses if sum(x) >= 1.e10]
            try:
                result = len(fitnesses_filtered)
            except:
                result = np.NaN
            return result
            
        def count_good(fitnesses):
            # Count good solutions.
            fitnesses_filtered = [x for x in fitnesses if sum(x) < 1.e10]
            try:
                result = len(fitnesses_filtered)
            except:
                result = np.NaN
            return result
            
        def eaSimple_Dump(population, toolbox, cxpb, mutpb, ngen, Dump_Location, Pareto, stats=None, halloffame=None, verbose=__debug__):
            """This function reproduce the simplest evolutionary algorithm as
            presented in chapter 7 of [Back2000]_. It is a modification of eaSimple from deap, and includes a break at generations to dump invalid individuals for inspection. 

            .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
               Basic Algorithms and Operators", 2000.
            """
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if halloffame is not None:
                halloffame.update(population)

            record = stats.compile(population) if stats else {}
            logbook.record(gen=0, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
            
            # Begin the generational process
            for gen in range(1, ngen + 1):
                print('Starting Generation {}'.format(gen))
                # Select the next generation individuals
                offspring = toolbox.select(population, len(population))

                # Vary the pool of individuals
                offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Update the hall of fame with the generated individuals
                if halloffame is not None:
                    halloffame.update(offspring)

                # Replace the current population by the offspring
                population[:] = offspring

                # Append the current generation statistics to the logbook
                record = stats.compile(population) if stats else {}
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                if verbose:
                    print(logbook.stream)
                
                Dumped = False
                count = 0
                for ind in population:
                    if np.isnan(ind.fitness.values[0]):
                        Evolutionary_Model.Miscellaneous_Functions.Dump_Individual(ind, Dump_Location)
                        count += 1
                        Dumped = True
                        
                if Dumped is True:
                    if len(glob.glob(r'{}\In Progress\*'.format(Dump_Location))) == 0:
                        ifttt_url = r'https://maker.ifttt.com/trigger/generation_complete/with/key/frbHb7pc90cg1jaq0IyJ5dJrRM7eV-zo0GZZFA50LpY'
                        if Pareto is True:
                            Pareto_Output = 'Pareto Run'
                        else:
                            Pareto_Output = 'Corrected Run'
                        requests.post(ifttt_url, data={'value1': gen-1, 'value2': count, 'value3': Pareto_Output})
                        print('Items dumped, stopping for evaluation of {} individuals'.format(count))
                        print('')
                        print(halloffame)
                        print('')
                        print(logbook)
                        sys.exit()
                    else:
                        print('Evaluations in Progress ({} individuals remaining)'.format(count))
                        sys.exit()
                    
            ifttt_url = r'https://maker.ifttt.com/trigger/generation_complete/with/key/frbHb7pc90cg1jaq0IyJ5dJrRM7eV-zo0GZZFA50LpY'
            requests.post(ifttt_url, data={'value1': 'All', 'value2': 0})
            return population, logbook
            
    class Dilute_Phase:
        # This class finds the methods for a dilute phase system.
        @staticmethod
        def Evolutionary_Run(Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Method_Maxes, Population_Size, Number_Of_Generations, Individual_Snapshots, Individual_Snapshots_Location, Dump_Location, seed=None, Test=None, Parallel=None):
            
            # Define minimum and maximum values for each method type.
            mins = []
            maxs = []
            
            for m in Method_Maxes:
                mins.append(1)
                maxs.append(m)
                
            # Set optional arguments to function if they are not input by the user.
            if Test is None:
                Test = False
                
            if Parallel is None:
                Parallel = False
            
            if seed is not None:
                random.seed(seed)
        
            # Build evolutionary model using DEAP.
            creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
            creator.create('Individual', list, fitness=creator.FitnessMin)
            
            hof = tools.HallOfFame(10)
            
            toolbox = base.Toolbox()
            
            toolbox.register('select', tools.selTournament, tournsize=3)
            
            # Define statistics for population.
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            
            stats.register('good', Evolutionary_Model.Miscellaneous_Functions.count_good)
            stats.register('errors', Evolutionary_Model.Miscellaneous_Functions.count_errors)
            
            stats.register('min', np.min)
            stats.register('avg', np.mean)
            stats.register('std', np.std)
            stats.register('max', np.max)
            
            stats.register('min_filt', Evolutionary_Model.Miscellaneous_Functions.min_filtered)
            stats.register('avg_filt', Evolutionary_Model.Miscellaneous_Functions.mean_filtered)
            stats.register('std_filt', Evolutionary_Model.Miscellaneous_Functions.std_filtered)
            stats.register('max_filt', Evolutionary_Model.Miscellaneous_Functions.max_filtered)
            
            # Set up chromosomes.
            toolbox.register('attr_0', random.randint, mins[0], maxs[0]) # Horizontal Pipe
            toolbox.register('attr_1', random.randint, mins[1], maxs[1]) # Vertical Upward Pipe
            toolbox.register('attr_2', random.randint, mins[2], maxs[2]) # Vertical Downward Pipe
            toolbox.register('attr_3', random.randint, mins[3], maxs[3]) # Bend H-H
            toolbox.register('attr_4', random.randint, mins[4], maxs[4]) # Bend H-U
            toolbox.register('attr_5', random.randint, mins[5], maxs[5]) # Bend H-D
            toolbox.register('attr_6', random.randint, mins[6], maxs[6]) # Bend U-H
            toolbox.register('attr_7', random.randint, mins[7], maxs[7]) # Bend D-H
            toolbox.register('attr_8', random.randint, mins[8], maxs[8]) # Acceleration Of Solids
            
            toolbox.register('individual', Evolutionary_Model.Miscellaneous_Functions.Generate_Individual, creator.Individual, (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3, toolbox.attr_4, toolbox.attr_5, toolbox.attr_6, toolbox.attr_7, toolbox.attr_8))
            
            toolbox.register('population', tools.initRepeat, list, toolbox.individual)
            toolbox.register('population_initial', Evolutionary_Model.Miscellaneous_Functions.Initial_Population, list, creator.Individual, Method_Maxes, Population_Size)
            
            # population = toolbox.population(n=Population_Size)
            population = toolbox.population_initial()
            
            toolbox.register('mate', tools.cxTwoPoint)
            # toolbox.register('mutate', tools.mutESLogNormal, c=1.0, indpb=0.3)
            toolbox.register('mutate', tools.mutUniformInt, low=min(mins), up=max(maxs), indpb=0.3)
            toolbox.decorate('mate', Evolutionary_Model.Miscellaneous_Functions.checkBounds(mins, maxs))
            toolbox.decorate('mutate', Evolutionary_Model.Miscellaneous_Functions.checkBounds(mins, maxs))
            
            # Define everything for the arguments.
            Dilute_Function_Dict = {
                'f_g_fun': {},
                'v_s_fun': {},
                'v_fao_singh_fun': {},
                'v_fao_rossetti_fun': {},
                'v_fao_de_moraes_fun': {},
                'v_fao_chambers_fun': {},
                'v_fao_pan_fun': {},
                'v_fao_das_fun': {},
                'Iteration_Evaluator_Pressure': {},
                'Iteration_Evaluator_Solids_Flow': {},
                }
            
            # Load pipeline data into a dictionary.
            Pipeline_Data, Coefficient_Dictionary = cf.Coefficient_Solver.Miscellaneous_Functions.Pipeline_Loads(Pipeline_Files, Coefficient_Dictionary_File)
            
            args = [Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Pipeline_Data, Coefficient_Dictionary_File, Coefficient_Dictionary, Pickle_Jar, Coefficient_Dump_Location, Dilute_Function_Dict, Test, Parallel, Individual_Snapshots, Individual_Snapshots_Location, Dump_Location]
            
            toolbox.register('evaluate', Evolutionary_Model.Dilute_Phase.Evaluate_Dump, args=args)
            
            if ' - Pareto' in sys.argv[0]:
                Pareto = True
            else:
                Pareto = False
            
            cxpb =  0.6
            mutpb = 0.3
            population, logbook = Evolutionary_Model.Miscellaneous_Functions.eaSimple_Dump(population, toolbox, cxpb, mutpb, Number_Of_Generations, Dump_Location, Pareto, stats=stats, halloffame=hof, verbose=False)
            
            return (hof, population, logbook)
                
        @staticmethod
        def Evaluate(individual, args):
            # This function is used to evaluate an individual from the population.
            # Pull data out of args.
            print(individual)
            if ' - Pareto' in sys.argv[0]:
                Pareto = True
            else:
                Pareto = False
            
            Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Pipeline_Data, Coefficient_Dictionary_File, Coefficient_Dictionary, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test, Parallel, Individual_Snapshots, Individual_Snapshots_Location = args

            # Check if evaluation for this individual exists. If it does then pull the data and return it.
            if tuple(individual) in Individual_Snapshots:
                return Individual_Snapshots[tuple(individual)]
                
            Methods_Input = {
                'Horizontal Pipe': individual[0],
                'Vertical Upward Pipe': individual[1],
                'Vertical Downward Pipe': individual[2],
                'Bend H-H': individual[3],
                'Bend H-U': individual[4],
                'Bend H-D': individual[5],
                'Bend U-H': individual[6],
                'Bend D-H': individual[7],
                'Acceleration Of Solids': individual[8],
                }
            
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
            
            # Train coefficients or pull existing coefficient values.
            # print('  Solving Coefficients')
            Coefficient_Collection, Cost = cf.Coefficient_Solver.Dilute_Phase.Solver(Coefficient_Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test=Test, Parallel=Parallel)
            # print('  Coefficients Done')
            Coefficients = [
                Coefficient_Collection['Horizontal Pipe'],
                Coefficient_Collection['Vertical Upward Pipe'],
                Coefficient_Collection['Vertical Downward Pipe'],
                Coefficient_Collection['Bend H-H'],
                Coefficient_Collection['Bend H-U'],
                Coefficient_Collection['Bend H-D'],
                Coefficient_Collection['Bend U-H'],
                Coefficient_Collection['Bend D-H'],
                Coefficient_Collection['Acceleration Of Solids']                
                ]
                
            # Evaluate the rest of the data using found coefficients.
            Differences = []
            for Data_Point in Evaluation_Data:
                # Pull test data out and assign it to correct variables.
                g = 9.81 # m/s^2
                    
                Pipeline_Name_A = Data_Point[0]
                m_g_A = Data_Point[1]
                T_g_A = Data_Point[2]
                System_Inlet_Pressure_A = Data_Point[3]
                Outlet_Pressures_A = Data_Point[4]
                
                Pipeline_Name_B = Data_Point[5]
                m_g_B = Data_Point[6]
                T_g_B = Data_Point[7]
                System_Inlet_Pressure_B = Data_Point[8]
                Outlet_Pressures_B = Data_Point[9]
                
                m_s_Input = Data_Point[10]
                
                # print('  Trying: {}'.format(Data_Point))
                
                # Pull pipeline and associated data.
                epsilon_A = Pipeline_Data[Pipeline_Name_A]['epsilon']
                Pipeline_Sections_A = Pipeline_Data[Pipeline_Name_A]['Pipeline_Sections']
                d_A = Pipeline_Data[Pipeline_Name_A]['d']
                d_v50_A = Pipeline_Data[Pipeline_Name_A]['d_v50']
                rho_s_A = Pipeline_Data[Pipeline_Name_A]['rho_s']
                Gas_Type_A = Pipeline_Data[Pipeline_Name_A]['Gas_Type']
                
                epsilon_B = Pipeline_Data[Pipeline_Name_B]['epsilon']
                Pipeline_Sections_B = Pipeline_Data[Pipeline_Name_B]['Pipeline_Sections']
                d_B = Pipeline_Data[Pipeline_Name_B]['d']
                d_v50_B = Pipeline_Data[Pipeline_Name_B]['d_v50']
                rho_s_B = Pipeline_Data[Pipeline_Name_B]['rho_s']
                Gas_Type_B = Pipeline_Data[Pipeline_Name_B]['Gas_Type']
                
                
                try:
                    # Find solids flow in A side of coupled system.
                    [m_s_Final_A, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_A, Pipeline_Sections_A, d_A, d_v50_A, rho_s_A, Gas_Type_A, T_g_A, m_g_A, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_A, Outlet_Pressures_A, Function_Dict)
                    # Cost_1 = Cost
                    # print('    A Done: {:.3f} ({})'.format(m_s_Final_A, Cost[0]))
                    
                    # Find solids flow in B side of coupled system.
                    [m_s_Final_B, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_B, Pipeline_Sections_B, d_B, d_v50_B, rho_s_B, Gas_Type_B, T_g_B, m_g_B, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_B, Outlet_Pressures_B, Function_Dict)
                    # Cost_2 = Cost
                    # print('    B Done: {:.3f} ({})'.format(m_s_Final_B, Cost[0]))
                    
                    # Get total mass flow.
                    m_s_Total = m_s_Final_A + m_s_Final_B             
                    # print('    Total {:.3f}: A {:.3f} ({}) - B {:.3f} ({})'.format(m_s_Total, m_s_Final_A, Cost_1[0], m_s_Final_B, Cost_2[0]))  
                
                    # Find the difference between input mass flow and the calculated flow.
                    if Pareto is False:
                        Differences.append(abs(m_s_Total - m_s_Input))
                    else:
                        Differences.append(m_s_Total - m_s_Input)
                    # Differences.append(m_s_Total - m_s_Input)
                except TypeError:
                    Differences.append(1.e10)
            
            # Save snapshot
            with open(Individual_Snapshots_Location, 'wb') as filet:
                pickle.dump(Individual_Snapshots, filet)
                
            if Pareto is False:
                print('{}: {:,.0f}'.format(individual, sum(Differences)))
                
                # Save solution to snapshot.
                Individual_Snapshots[tuple(individual)] = sum(Differences),
                
                return sum(Differences),
            else:
                Differences_Mean = np.mean(Differences)
                Differences_Std = np.std(Differences)
                Differences_Pareto = abs(Differences_Mean) + Differences_Std
                print('{}: {:,.2f} ({:,.2f} - {:,.2f})'.format(individual, Differences_Pareto, Differences_Mean, Differences_Std))
                
                # Save solution to snapshot.
                Individual_Snapshots[tuple(individual)] = Differences_Pareto,
                
                return Differences_Pareto,
            # return sum([d ** 2. for d in Differences]),
        
        @staticmethod
        def Evaluate_Dump(individual, args):
            # This function is used to evaluate an individual from the population.
            # Pull data out of args.
            Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Pipeline_Data, Coefficient_Dictionary_File, Coefficient_Dictionary, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test, Parallel, Individual_Snapshots, Individual_Snapshots_Location, Dump_Location = args
            
            # Check if evaluation for this individual exists. If it does then pull the data and return it. If not, dump the individual for evaluation.
            try:
                result = Individual_Snapshots[tuple(individual)]
                # print('{}: Found'.format(individual))
            except:
                # Evolutionary_Model.Miscellaneous_Functions.Dump_Individual(individual, Dump_Location)
                result = (np.NaN),
                # print('{}: Not Found'.format(individual))
                
            return result
                
        @staticmethod
        def Evaluate_Full_Differences(individual, args):
            # This function is used to evaluate an individual from the population and return the full set of differences
            # Pull data out of args.
            
            Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Pipeline_Data, Coefficient_Dictionary_File, Coefficient_Dictionary, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test, Parallel, Individual_Snapshots, Individual_Snapshots_Location = args
                
            Methods_Input = {
                'Horizontal Pipe': individual[0],
                'Vertical Upward Pipe': individual[1],
                'Vertical Downward Pipe': individual[2],
                'Bend H-H': individual[3],
                'Bend H-U': individual[4],
                'Bend H-D': individual[5],
                'Bend U-H': individual[6],
                'Bend D-H': individual[7],
                'Acceleration Of Solids': individual[8],
                }
            
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
            
            # Train coefficients or pull existing coefficient values.
            Coefficient_Collection, Cost = cf.Coefficient_Solver.Dilute_Phase.Solver(Coefficient_Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test=Test, Parallel=Parallel)
            
            Coefficients = [
                Coefficient_Collection['Horizontal Pipe'],
                Coefficient_Collection['Vertical Upward Pipe'],
                Coefficient_Collection['Vertical Downward Pipe'],
                Coefficient_Collection['Bend H-H'],
                Coefficient_Collection['Bend H-U'],
                Coefficient_Collection['Bend H-D'],
                Coefficient_Collection['Bend U-H'],
                Coefficient_Collection['Bend D-H'],
                Coefficient_Collection['Acceleration Of Solids']                
                ]
                
            # Evaluate the rest of the data using found coefficients.
            Differences = []
            m_s_s = []
            
            # num_print = 10000
            # Function_Dict_Orig = Function_Dict
            
            # for Data_Point, k in zip(Evaluation_Data, range(len(Evaluation_Data))):
            for Data_Point in Evaluation_Data:
                # Pull test data out and assign it to correct variables.
                g = 9.81 # m/s^2
                    
                Pipeline_Name_A = Data_Point[0]
                m_g_A = Data_Point[1]
                T_g_A = Data_Point[2]
                System_Inlet_Pressure_A = Data_Point[3]
                Outlet_Pressures_A = Data_Point[4]
                
                Pipeline_Name_B = Data_Point[5]
                m_g_B = Data_Point[6]
                T_g_B = Data_Point[7]
                System_Inlet_Pressure_B = Data_Point[8]
                Outlet_Pressures_B = Data_Point[9]
                
                m_s_Input = Data_Point[10]
                
                # Pull pipeline and associated data.
                epsilon_A = Pipeline_Data[Pipeline_Name_A]['epsilon']
                Pipeline_Sections_A = Pipeline_Data[Pipeline_Name_A]['Pipeline_Sections']
                d_A = Pipeline_Data[Pipeline_Name_A]['d']
                d_v50_A = Pipeline_Data[Pipeline_Name_A]['d_v50']
                rho_s_A = Pipeline_Data[Pipeline_Name_A]['rho_s']
                Gas_Type_A = Pipeline_Data[Pipeline_Name_A]['Gas_Type']
                
                epsilon_B = Pipeline_Data[Pipeline_Name_B]['epsilon']
                Pipeline_Sections_B = Pipeline_Data[Pipeline_Name_B]['Pipeline_Sections']
                d_B = Pipeline_Data[Pipeline_Name_B]['d']
                d_v50_B = Pipeline_Data[Pipeline_Name_B]['d_v50']
                rho_s_B = Pipeline_Data[Pipeline_Name_B]['rho_s']
                Gas_Type_B = Pipeline_Data[Pipeline_Name_B]['Gas_Type']
                
                try:
                    # Find solids flow in A side of coupled system.
                    [m_s_Final_A, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_A, Pipeline_Sections_A, d_A, d_v50_A, rho_s_A, Gas_Type_A, T_g_A, m_g_A, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_A, Outlet_Pressures_A, Function_Dict)
                    
                    # Find solids flow in B side of coupled system.
                    [m_s_Final_B, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_B, Pipeline_Sections_B, d_B, d_v50_B, rho_s_B, Gas_Type_B, T_g_B, m_g_B, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_B, Outlet_Pressures_B, Function_Dict)
                    
                    # Get total mass flow.
                    m_s_Total = m_s_Final_A + m_s_Final_B            
                
                    # Find the difference between input mass flow and the calculated flow.
                    Differences.append(m_s_Total - m_s_Input)
                    m_s_s.append(m_s_Total)
                    
                except TypeError:
                    Differences.append(1.e10)
                    m_s_s.append(1.e10)
                    
                # if k % num_print == 0:
                    # print('{} Data Points Evaluated for {}'.format(k, individual))
                    # Function_Dict = Function_Dict_Orig
                    
            return Differences, m_s_s
            
        @staticmethod
        def Evaluate_Full_Differences_Pressure_Drop_Types(individual, args):
            # This function is used to evaluate an individual from the population and return the full set of differences
            # Pull data out of args.
            
            Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Pipeline_Data, Coefficient_Dictionary_File, Coefficient_Dictionary, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test, Parallel, Individual_Snapshots, Individual_Snapshots_Location = args
                
            Methods_Input = {
                'Horizontal Pipe': individual[0],
                'Vertical Upward Pipe': individual[1],
                'Vertical Downward Pipe': individual[2],
                'Bend H-H': individual[3],
                'Bend H-U': individual[4],
                'Bend H-D': individual[5],
                'Bend U-H': individual[6],
                'Bend D-H': individual[7],
                'Acceleration Of Solids': individual[8],
                }
            
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
            
            # Train coefficients or pull existing coefficient values.
            Coefficient_Collection, Cost = cf.Coefficient_Solver.Dilute_Phase.Solver(Coefficient_Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test=Test, Parallel=Parallel)
            
            Coefficients = [
                Coefficient_Collection['Horizontal Pipe'],
                Coefficient_Collection['Vertical Upward Pipe'],
                Coefficient_Collection['Vertical Downward Pipe'],
                Coefficient_Collection['Bend H-H'],
                Coefficient_Collection['Bend H-U'],
                Coefficient_Collection['Bend H-D'],
                Coefficient_Collection['Bend U-H'],
                Coefficient_Collection['Bend D-H'],
                Coefficient_Collection['Acceleration Of Solids']                
                ]
                
            # Set up file.
            with open('Section Pressure Dump.csv', 'w') as filet:
                filet.write('Time,Section Type,Orientation,Pressure Drop\n')
            
            # Evaluate the rest of the data using found coefficients.
            Differences = []
            m_s_s = []
            for Data_Point in Evaluation_Data:
                # Pull test data out and assign it to correct variables.
                g = 9.81 # m/s^2
                    
                Pipeline_Name_A = Data_Point[0]
                m_g_A = Data_Point[1]
                T_g_A = Data_Point[2]
                System_Inlet_Pressure_A = Data_Point[3]
                Outlet_Pressures_A = Data_Point[4]
                
                Pipeline_Name_B = Data_Point[5]
                m_g_B = Data_Point[6]
                T_g_B = Data_Point[7]
                System_Inlet_Pressure_B = Data_Point[8]
                Outlet_Pressures_B = Data_Point[9]
                
                m_s_Input = Data_Point[10]
                
                # Pull pipeline and associated data.
                epsilon_A = Pipeline_Data[Pipeline_Name_A]['epsilon']
                Pipeline_Sections_A = Pipeline_Data[Pipeline_Name_A]['Pipeline_Sections']
                d_A = Pipeline_Data[Pipeline_Name_A]['d']
                d_v50_A = Pipeline_Data[Pipeline_Name_A]['d_v50']
                rho_s_A = Pipeline_Data[Pipeline_Name_A]['rho_s']
                Gas_Type_A = Pipeline_Data[Pipeline_Name_A]['Gas_Type']
                
                epsilon_B = Pipeline_Data[Pipeline_Name_B]['epsilon']
                Pipeline_Sections_B = Pipeline_Data[Pipeline_Name_B]['Pipeline_Sections']
                d_B = Pipeline_Data[Pipeline_Name_B]['d']
                d_v50_B = Pipeline_Data[Pipeline_Name_B]['d_v50']
                rho_s_B = Pipeline_Data[Pipeline_Name_B]['rho_s']
                Gas_Type_B = Pipeline_Data[Pipeline_Name_B]['Gas_Type']
                
                try:
                    # Find solids flow in A side of coupled system.
                    [m_s_Final_A, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver_Dump(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_A, Pipeline_Sections_A, d_A, d_v50_A, rho_s_A, Gas_Type_A, T_g_A, m_g_A, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_A, Outlet_Pressures_A, Function_Dict)
                    
                    # Find solids flow in B side of coupled system.
                    [m_s_Final_B, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver_Dump(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_B, Pipeline_Sections_B, d_B, d_v50_B, rho_s_B, Gas_Type_B, T_g_B, m_g_B, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_B, Outlet_Pressures_B, Function_Dict)
                    
                    # Get total mass flow.
                    m_s_Total = m_s_Final_A + m_s_Final_B            
                
                    # Find the difference between input mass flow and the calculated flow.
                    Differences.append(m_s_Total - m_s_Input)
                    m_s_s.append(m_s_Total)
                except TypeError:
                    Differences.append(1.e10)
                    m_s_s.append(1.e10)
                    
            return Differences, m_s_s
            
        def Evaluate_Full_Split(individual, args):
            # This function is used to evaluate an individual from the population and return the full set of differences
            # Pull data out of args.
            
            Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Pipeline_Data, Coefficient_Dictionary_File, Coefficient_Dictionary, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test, Parallel, Individual_Snapshots, Individual_Snapshots_Location, Split_Method = args
                
            Methods_Input = {
                'Horizontal Pipe': individual[0],
                'Vertical Upward Pipe': individual[1],
                'Vertical Downward Pipe': individual[2],
                'Bend H-H': individual[3],
                'Bend H-U': individual[4],
                'Bend H-D': individual[5],
                'Bend U-H': individual[6],
                'Bend D-H': individual[7],
                'Acceleration Of Solids': individual[8],
                }
            
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
            
            # Train coefficients or pull existing coefficient values.
            Coefficient_Collection, Cost = cf.Coefficient_Solver.Dilute_Phase.Solver(Coefficient_Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test=Test, Parallel=Parallel)
            
            Coefficients = [
                Coefficient_Collection['Horizontal Pipe'],
                Coefficient_Collection['Vertical Upward Pipe'],
                Coefficient_Collection['Vertical Downward Pipe'],
                Coefficient_Collection['Bend H-H'],
                Coefficient_Collection['Bend H-U'],
                Coefficient_Collection['Bend H-D'],
                Coefficient_Collection['Bend U-H'],
                Coefficient_Collection['Bend D-H'],
                Coefficient_Collection['Acceleration Of Solids']                
                ]
                
            Split_Coefficients, Cost = cf.Coefficient_Solver.Dilute_Phase.Split_Solver(Coefficient_Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Split_Method)
            
            
            print('Cost: {}'.format(Cost))
            print('')
            print('Split: {}'.format(Split_Coefficients))
            print('')
            print('Evaluating Data')
                
            # Evaluate the rest of the data using found coefficients.
            Differences = []
            m_s_s = []
            for Data_Point in Evaluation_Data:
                # Pull test data out and assign it to correct variables.
                g = 9.81 # m/s^2
                    
                Pipeline_Name_A = Data_Point[0]
                m_g_A = Data_Point[1]
                T_g_A = Data_Point[2]
                System_Inlet_Pressure_A = Data_Point[3]
                Outlet_Pressures_A = Data_Point[4]
                
                Pipeline_Name_B = Data_Point[5]
                m_g_B = Data_Point[6]
                T_g_B = Data_Point[7]
                System_Inlet_Pressure_B = Data_Point[8]
                Outlet_Pressures_B = Data_Point[9]
                
                m_s_Input = Data_Point[10]
                
                # Pull pipeline and associated data.
                epsilon_A = Pipeline_Data[Pipeline_Name_A]['epsilon']
                Pipeline_Sections_A = Pipeline_Data[Pipeline_Name_A]['Pipeline_Sections']
                d_A = Pipeline_Data[Pipeline_Name_A]['d']
                d_v50_A = Pipeline_Data[Pipeline_Name_A]['d_v50']
                rho_s_A = Pipeline_Data[Pipeline_Name_A]['rho_s']
                Gas_Type_A = Pipeline_Data[Pipeline_Name_A]['Gas_Type']
                
                epsilon_B = Pipeline_Data[Pipeline_Name_B]['epsilon']
                Pipeline_Sections_B = Pipeline_Data[Pipeline_Name_B]['Pipeline_Sections']
                d_B = Pipeline_Data[Pipeline_Name_B]['d']
                d_v50_B = Pipeline_Data[Pipeline_Name_B]['d_v50']
                rho_s_B = Pipeline_Data[Pipeline_Name_B]['rho_s']
                Gas_Type_B = Pipeline_Data[Pipeline_Name_B]['Gas_Type']
                
                try:
                    # Find solids flow in A side of coupled system.
                    [m_s_Final_A, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_A, Pipeline_Sections_A, d_A, d_v50_A, rho_s_A, Gas_Type_A, T_g_A, m_g_A, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_A, Outlet_Pressures_A, Function_Dict, Split_Pressure_Drop=True, Split_Method=Split_Method, Split_Pressure_Drop_Coefficients=Split_Coefficients)
                    
                    # Find solids flow in B side of coupled system.
                    [m_s_Final_B, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_B, Pipeline_Sections_B, d_B, d_v50_B, rho_s_B, Gas_Type_B, T_g_B, m_g_B, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_B, Outlet_Pressures_B, Function_Dict, Split_Pressure_Drop=True, Split_Method=Split_Method, Split_Pressure_Drop_Coefficients=Split_Coefficients)

                    
                    # Get total mass flow.
                    m_s_Total = m_s_Final_A + m_s_Final_B                    
                
                    # Find the difference between input mass flow and the calculated flow.
                    Differences.append(m_s_Total - m_s_Input)
                    m_s_s.append(m_s_Total)
                except TypeError:
                    Differences.append(1.e10)
                    m_s_s.append(1.e10)
                    
            return Differences, m_s_s
        
        @staticmethod
        def Evaluate_Symbolic(individual, args):
            # This function is used to evaluate an individual from the population.
            
            print('{}'.format(individual))
            
            # Pull data out of args.
            Training_Data, Evaluation_Data, Pipeline_Files, Pipeline_Data, Coefficient_Dictionary, Coefficient_Dictionary_File, Test, Parallel, toolbox, Reference_Individual, Section_Type, Pickle_Jar = args
            
            # Turn individual into function.
            func = toolbox.compile(expr=individual)
            
            # Make extra arguments input.
            extra_args = [func]

            # Check if evaluation for this individual exists. If it does then pull the data and return it.
            # if tuple(individual) in Individual_Snapshots:
                # return Individual_Snapshots[tuple(individual)]
            
            Methods_Input_Reference = {
                'Horizontal Pipe': Reference_Individual[0],
                'Vertical Upward Pipe': Reference_Individual[1],
                'Vertical Downward Pipe': Reference_Individual[2],
                'Bend H-H': Reference_Individual[3],
                'Bend H-U': Reference_Individual[4],
                'Bend H-D': Reference_Individual[5],
                'Bend U-H': Reference_Individual[6],
                'Bend D-H': Reference_Individual[7],
                'Acceleration Of Solids': Reference_Individual[8],
                }
            
            Methods_Input_Max = {
                'Horizontal Pipe': 87,
                'Vertical Upward Pipe': 28,
                'Vertical Downward Pipe': 28,
                'Bend H-H': 123,
                'Bend H-U': 123,
                'Bend H-D': 123,
                'Bend U-H': 123,
                'Bend D-H': 123,
                'Acceleration Of Solids': 25,
                }
                
            Methods_Input = copy.deepcopy(Methods_Input_Reference)
            Methods_Input[Section_Type] = Methods_Input_Max[Section_Type]
                
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
            
            # Train coefficients or pull existing coefficient values.
            Coefficient_Dump_Location = None
            
            # Define everything for the arguments.
            Function_Dict = {
                'f_g_fun': {},
                'v_s_fun': {},
                'v_fao_singh_fun': {},
                'v_fao_rossetti_fun': {},
                'v_fao_de_moraes_fun': {},
                'v_fao_chambers_fun': {},
                'v_fao_pan_fun': {},
                'v_fao_das_fun': {},
                'Iteration_Evaluator_Pressure': {},
                'Iteration_Evaluator_Solids_Flow': {},
                }
            
            Coefficient_Collection, Cost = cf.Coefficient_Solver.Dilute_Phase.Solver_Symbolic(Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Methods_Input_Reference, Section_Type, Test=Test, Parallel=Parallel, extra_args=extra_args)
            
            Coefficients = [
                Coefficient_Collection['Horizontal Pipe'],
                Coefficient_Collection['Vertical Upward Pipe'],
                Coefficient_Collection['Vertical Downward Pipe'],
                Coefficient_Collection['Bend H-H'],
                Coefficient_Collection['Bend H-U'],
                Coefficient_Collection['Bend H-D'],
                Coefficient_Collection['Bend U-H'],
                Coefficient_Collection['Bend D-H'],
                Coefficient_Collection['Acceleration Of Solids']                
                ]
                
            # Evaluate the rest of the data using found coefficients.
            Differences = []
            Differences_Non_Abs = []           
            
            for Data_Point in Evaluation_Data:
                # Pull test data out and assign it to correct variables.
                g = 9.81 # m/s^2
                    
                Pipeline_Name_A = Data_Point[0]
                m_g_A = Data_Point[1]
                T_g_A = Data_Point[2]
                System_Inlet_Pressure_A = Data_Point[3]
                Outlet_Pressures_A = Data_Point[4]
                
                Pipeline_Name_B = Data_Point[5]
                m_g_B = Data_Point[6]
                T_g_B = Data_Point[7]
                System_Inlet_Pressure_B = Data_Point[8]
                Outlet_Pressures_B = Data_Point[9]
                
                m_s_Input = Data_Point[10]
                
                # Pull pipeline and associated data.
                epsilon_A = Pipeline_Data[Pipeline_Name_A]['epsilon']
                Pipeline_Sections_A = Pipeline_Data[Pipeline_Name_A]['Pipeline_Sections']
                d_A = Pipeline_Data[Pipeline_Name_A]['d']
                d_v50_A = Pipeline_Data[Pipeline_Name_A]['d_v50']
                rho_s_A = Pipeline_Data[Pipeline_Name_A]['rho_s']
                Gas_Type_A = Pipeline_Data[Pipeline_Name_A]['Gas_Type']
                
                epsilon_B = Pipeline_Data[Pipeline_Name_B]['epsilon']
                Pipeline_Sections_B = Pipeline_Data[Pipeline_Name_B]['Pipeline_Sections']
                d_B = Pipeline_Data[Pipeline_Name_B]['d']
                d_v50_B = Pipeline_Data[Pipeline_Name_B]['d_v50']
                rho_s_B = Pipeline_Data[Pipeline_Name_B]['rho_s']
                Gas_Type_B = Pipeline_Data[Pipeline_Name_B]['Gas_Type']
                
                
                try:
                # if True:
                    # Find solids flow in A side of coupled system.
                    [m_s_Final_A, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_A, Pipeline_Sections_A, d_A, d_v50_A, rho_s_A, Gas_Type_A, T_g_A, m_g_A, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_A, Outlet_Pressures_A, Function_Dict, extra_args=extra_args)
                    
                    # Find solids flow in B side of coupled system.
                    [m_s_Final_B, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_B, Pipeline_Sections_B, d_B, d_v50_B, rho_s_B, Gas_Type_B, T_g_B, m_g_B, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_B, Outlet_Pressures_B, Function_Dict, extra_args=extra_args)
                    
                    # Get total mass flow.
                    m_s_Total = m_s_Final_A + m_s_Final_B                    
                
                    # Find the difference between input mass flow and the calculated flow.
                    Differences.append(abs(m_s_Total - m_s_Input))
                    Differences_Non_Abs.append(m_s_Total - m_s_Input)
                    
                except:
                    Differences.append(1.e10)
                
            # Save solution to snapshot.
            # Individual_Snapshots[tuple(individual)] = sum(Differences),

            print('Total Difference: {:,.0f} - Mean: {:,.1f} - Standard Deviation: {:,.1f}'.format(sum(Differences), np.mean(Differences_Non_Abs), np.std(Differences_Non_Abs)))
            
            # Save snapshot
            # with open(Individual_Snapshots_Location, 'wb') as filet:
                # pickle.dump(Individual_Snapshots, filet)
                
            return sum(Differences),
        
    class Dense_Phase:
        # This class finds the methods for a dense phase system.
        @staticmethod
        def Evolutionary_Run(Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Method_Maxes, Population_Size, Number_Of_Generations, Individual_Snapshots, Individual_Snapshots_Location, seed=None, Test=None, Parallel=None):
            
            # Define minimum and maximum values for each method type.
            mins = []
            maxs = []
            
            for m in Method_Maxes:
                mins.append(1)
                maxs.append(m)
                
            # Set optional arguments to function if they are not input by the user.
            if Test is None:
                Test = False
                
            if Parallel is None:
                Parallel = False
            
            if seed is not None:
                random.seed(seed)
        
            # Build evolutionary model using DEAP.
            creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
            creator.create('Individual', list, fitness=creator.FitnessMin)
            
            hof = tools.HallOfFame(1)
            
            toolbox = base.Toolbox()
            
            toolbox.register('select', tools.selTournament, tournsize=3)
            
            # Define statistics for population.
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register('avg', np.mean)
            stats.register('std', np.std)
            stats.register('min', np.min)
            stats.register('max', np.max)
            
            # Set up chromosomes.
            toolbox.register('attr_0', random.randint, mins[0], maxs[0]) # Horizontal Pipe
            toolbox.register('attr_1', random.randint, mins[1], maxs[1]) # Vertical Upward Pipe
            toolbox.register('attr_2', random.randint, mins[2], maxs[2]) # Vertical Downward Pipe
            toolbox.register('attr_3', random.randint, mins[3], maxs[3]) # Bend H-H
            toolbox.register('attr_4', random.randint, mins[4], maxs[4]) # Bend H-U
            toolbox.register('attr_5', random.randint, mins[5], maxs[5]) # Bend H-D
            toolbox.register('attr_6', random.randint, mins[6], maxs[6]) # Bend U-H
            toolbox.register('attr_7', random.randint, mins[7], maxs[7]) # Bend D-H
            toolbox.register('attr_8', random.randint, mins[8], maxs[8]) # Acceleration Of Solids
            
            toolbox.register('individual', Evolutionary_Model.Miscellaneous_Functions.Generate_Individual, creator.Individual, (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3, toolbox.attr_4, toolbox.attr_5, toolbox.attr_6, toolbox.attr_7, toolbox.attr_8))
            
            toolbox.register('population', tools.initRepeat, list, toolbox.individual)
            
            population = toolbox.population(n=Population_Size)
            
            toolbox.register('mate', tools.cxTwoPoint)
            # toolbox.register('mutate', tools.mutESLogNormal, c=1.0, indpb=0.3)
            toolbox.register('mutate', tools.mutUniformInt, low=min(mins), up=max(maxs), indpb=0.3)
            toolbox.decorate('mate', Evolutionary_Model.Miscellaneous_Functions.checkBounds(mins, maxs))
            toolbox.decorate('mutate', Evolutionary_Model.Miscellaneous_Functions.checkBounds(mins, maxs))
            
            # Define everything for the arguments.
            Dense_Function_Dict = {
                'f_g_fun': {},
                'P_o_fun': {},
                'v_s_fun': {},
                'P_o_watson_fun': {},
                }
            
            # Load pipeline data into a dictionary.
            Pipeline_Data, Coefficient_Dictionary = cf.Coefficient_Solver.Miscellaneous_Functions.Pipeline_Loads(Pipeline_Files, Coefficient_Dictionary_File)
            
            args = [Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Pipeline_Data, Coefficient_Dictionary_File, Coefficient_Dictionary, Pickle_Jar, Coefficient_Dump_Location, Dense_Function_Dict, Test, Parallel, Individual_Snapshots, Individual_Snapshots_Location]
            
            toolbox.register('evaluate', Evolutionary_Model.Dense_Phase.Evaluate, args=args)
            
            
            population, logbook = algorithms.eaSimple(population, toolbox,  cxpb=0.6, mutpb=0.3, ngen=Number_Of_Generations, stats=stats, halloffame=hof, verbose=False)
            
            for ind in population:
                print(ind.fitness)
                
            print('')
            print(hof)
            print('')
            print(logbook)
                
        
        @staticmethod
        def Evaluate(individual, args):
            # This function is used to evaluate an individual from the population.
            # Pull data out of args.
            print(individual)
            
            Coefficient_Training_Data, Evaluation_Data, Pipeline_Files, Pipeline_Data, Coefficient_Dictionary_File, Coefficient_Dictionary, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test, Parallel, Individual_Snapshots, Individual_Snapshots_Location = args

            # Check if evaluaiton for this individual exists. If it does then pull the data and return it.
            if tuple(individual )in Individual_Snapshots:
                return Individual_Snapshots[tuple(individual)]
                
            Methods_Input = {
                'Horizontal Pipe': individual[0],
                'Vertical Upward Pipe': individual[1],
                'Vertical Downward Pipe': individual[2],
                'Bend H-H': individual[3],
                'Bend H-U': individual[4],
                'Bend H-D': individual[5],
                'Bend U-H': individual[6],
                'Bend D-H': individual[7],
                'Acceleration Of Solids': individual[8],
                }
            
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
            
            # Train coefficients or pull existing coefficient values.
            Coefficient_Collection, Cost = cf.Coefficient_Solver.Dense_Phase.Solver(Coefficient_Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test=Test, Parallel=Parallel)
            
            Coefficients = [
                Coefficient_Collection['Horizontal Pipe'],
                Coefficient_Collection['Vertical Upward Pipe'],
                Coefficient_Collection['Vertical Downward Pipe'],
                Coefficient_Collection['Bend H-H'],
                Coefficient_Collection['Bend H-U'],
                Coefficient_Collection['Bend H-D'],
                Coefficient_Collection['Bend U-H'],
                Coefficient_Collection['Bend D-H'],
                Coefficient_Collection['Acceleration Of Solids']                
                ]
                
            # Evaluate the rest of the data using found coefficients.
            Differences = []
            for Data_Point in Evaluation_Data:
                # Pull test data out and assign it to correct variables.
                g = 9.81 # m/s^2
                    
                Pipeline_Name = Data_Point[0]
                m_g = Data_Point[1]
                T_g = Data_Point[2]
                System_Inlet_Pressure = Data_Point[3]
                Outlet_Pressures = Data_Point[4]
                
                m_s_Input = Data_Point[5]
                
                # Pull pipeline and associated data.
                epsilon = Pipeline_Data[Pipeline_Name]['epsilon']
                Pipeline_Sections = Pipeline_Data[Pipeline_Name]['Pipeline_Sections']
                d = Pipeline_Data[Pipeline_Name]['d']
                d_v50 = Pipeline_Data[Pipeline_Name]['d_v50']
                rho_s = Pipeline_Data[Pipeline_Name]['rho_s']
                Gas_Type = Pipeline_Data[Pipeline_Name]['Gas_Type']
                
                try:
                    # Find solids flow in the system
                    [m_s_Final, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dense_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure, Outlet_Pressures, Function_Dict)
                    
                    # Find the difference between input mass flow and the calculated flow.
                    Differences.append(abs(m_s_Final - m_s_Input))
                except TypeError:
                    Differences.append(1.e10)
                
            # Save solution to snapshot.
            Individual_Snapshots[tuple(individual)] = sum(Differences),
            
            # Save snapshot
            # with open(Individual_Snapshots_Location, 'wb') as filet:
                # pickle.dump(Individual_Snapshots, filet)
                
            return sum(Differences),