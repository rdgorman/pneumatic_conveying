# This code contains is designed to solve the coefficients for a given set of training data and save the results.

import numpy as np

import Section_Pressure_Drop_Functions as spdf
import System_Pressure_Drop_Functions as syspdf
import Miscellaneous_Functions as mf

from scipy.optimize import least_squares, minimize
from optimparallel import minimize_parallel
from fluids.atmosphere import ATMOSPHERE_1976

import sys
import warnings
import pickle
import platform
import traceback
warnings.filterwarnings("ignore", category=RuntimeWarning) 

if platform.system() == 'Windows':
    Delimiter = '\\'
else:
    Delimiter = '/'

class Coefficient_Solver:
    # This class uses training data to solve coefficients for a given set of training data and saves the results.
    
    class Miscellaneous_Functions:
    # This class contains various generic functions used for both dilute and dense phase systems.
        @staticmethod
        def Output_Coefficients(Type, Methods_Input, Coefficients_Collection, Pickle_Jar, Coefficient_Dump_Location):
            # This function takes a set of coefficients and outputs the results to a pickle file for the entire set of functions. It also appends them to dictionaries for each pressure drop type to determine starting guesses.

            # Pull methods from methods input.
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
            
            # Make coefficient variable collection.
            Methods_ID = '{} - HP {} VUP {} VDP {} BHH {} BHU {} BHD {} BUH {} BDH {} AS {}'.format(Type, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method, Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method)
            
            # Dump found coefficients for methods into the pickle jar.
            with open(r'{}{}{}.pkl'.format(Pickle_Jar, Delimiter, Methods_ID), 'wb') as filet:
                pickle.dump(Coefficients_Collection, filet)
                
            # If coefficient dictionary pickle does not exist then create it. If it exists, load it and append data to it.
            try:
                with open('{}{}Coefficient Dump.pkl'.format(Pickle_Jar, Delimiter), 'rb') as filet:
                    Coefficient_Output = pickle.load(filet)
                
                
                if Horizontal_Pipe_Method not in Coefficient_Output['Horizontal Pipe']:
                    Coefficient_Output['Horizontal Pipe'][Horizontal_Pipe_Method] = {}
                if Horizontal_Pipe_Method not in Coefficient_Output['Vertical Upward Pipe']:
                    Coefficient_Output['Vertical Upward Pipe'][Horizontal_Pipe_Method] = {}
                if Horizontal_Pipe_Method not in Coefficient_Output['Vertical Downward Pipe']:
                    Coefficient_Output['Vertical Downward Pipe'][Horizontal_Pipe_Method] = {}
                if Horizontal_Pipe_Method not in Coefficient_Output['Bend H-H']:
                    Coefficient_Output['Bend H-H'][Horizontal_Pipe_Method] = {}
                if Horizontal_Pipe_Method not in Coefficient_Output['Bend H-U']:
                    Coefficient_Output['Bend H-U'][Horizontal_Pipe_Method] = {}
                if Horizontal_Pipe_Method not in Coefficient_Output['Bend H-D']:
                    Coefficient_Output['Bend H-D'][Horizontal_Pipe_Method] = {}
                if Horizontal_Pipe_Method not in Coefficient_Output['Bend U-H']:
                    Coefficient_Output['Bend U-H'][Horizontal_Pipe_Method] = {}
                if Horizontal_Pipe_Method not in Coefficient_Output['Bend D-H']:
                    Coefficient_Output['Bend D-H'][Horizontal_Pipe_Method] = {}
                if Horizontal_Pipe_Method not in Coefficient_Output['Acceleration Of Solids']:
                    Coefficient_Output['Acceleration Of Solids'][Horizontal_Pipe_Method] = {}
                    
                Coefficient_Output['Horizontal Pipe'][Horizontal_Pipe_Method]['Latest'] = Coefficients_Collection['Horizontal Pipe']
                Coefficient_Output['Vertical Upward Pipe'][Vertical_Upward_Pipe_Method]['Latest'] = Coefficients_Collection['Vertical Upward Pipe']
                Coefficient_Output['Vertical Downward Pipe'][Vertical_Downward_Pipe_Method]['Latest'] = Coefficients_Collection['Vertical Downward Pipe']
                Coefficient_Output['Bend H-H'][Bend_H_H_Method]['Latest'] = Coefficients_Collection['Bend H-H']
                Coefficient_Output['Bend H-U'][Bend_H_U_Method]['Latest'] = Coefficients_Collection['Bend H-U']
                Coefficient_Output['Bend H-D'][Bend_H_D_Method]['Latest'] = Coefficients_Collection['Bend H-D']
                Coefficient_Output['Bend U-H'][Bend_U_H_Method]['Latest'] = Coefficients_Collection['Bend U-H']
                Coefficient_Output['Bend D-H'][Bend_D_H_Method]['Latest'] = Coefficients_Collection['Bend D-H']
                Coefficient_Output['Acceleration Of Solids'][Acceleration_Of_Solids_Method]['Latest'] = Coefficients_Collection['Acceleration Of Solids']
                
                Coefficient_Output['Horizontal Pipe'][Horizontal_Pipe_Method]['Array'].append(Coefficients_Collection['Horizontal Pipe'])
                Coefficient_Output['Vertical Upward Pipe'][Vertical_Upward_Pipe_Method]['Array'].append(Coefficients_Collection['Vertical Upward Pipe'])
                Coefficient_Output['Vertical Downward Pipe'][Vertical_Downward_Pipe_Method]['Array'].append(Coefficients_Collection['Vertical Downward Pipe'])
                Coefficient_Output['Bend H-H'][Bend_H_H_Method]['Array'].append(Coefficients_Collection['Bend H-H'])
                Coefficient_Output['Bend H-U'][Bend_H_U_Method]['Array'].append(Coefficients_Collection['Bend H-U'])
                Coefficient_Output['Bend H-D'][Bend_H_D_Method]['Array'].append(Coefficients_Collection['Bend H-D'])
                Coefficient_Output['Bend U-H'][Bend_U_H_Method]['Array'].append(Coefficients_Collection['Bend U-H'])
                Coefficient_Output['Bend D-H'][Bend_D_H_Method]['Array'].append(Coefficients_Collection['Bend D-H'])
                Coefficient_Output['Acceleration Of Solids'][Acceleration_Of_Solids_Method]['Array'].append(Coefficients_Collection['Acceleration Of Solids'])
                
            except Exception as e:
                # print(traceback.print_exc())
                # print(Coefficients_Collection)
                Coefficient_Output = {
                    'Horizontal Pipe': { Horizontal_Pipe_Method: {
                        'Array': [Coefficients_Collection['Horizontal Pipe']], 
                        'Latest': Coefficients_Collection['Horizontal Pipe']
                        }}, 
                    'Vertical Upward Pipe': { Vertical_Upward_Pipe_Method: {
                        'Array': [Coefficients_Collection['Vertical Upward Pipe']], 
                        'Latest': Coefficients_Collection['Vertical Upward Pipe']
                        }}, 
                    'Vertical Downward Pipe': { Vertical_Downward_Pipe_Method: {
                        'Array': [Coefficients_Collection['Vertical Downward Pipe']], 
                        'Latest': Coefficients_Collection['Vertical Downward Pipe']
                        }}, 
                    'Bend H-H': { Bend_H_H_Method: {
                        'Array': [Coefficients_Collection['Bend H-H']], 
                        'Latest': Coefficients_Collection['Bend H-H']
                        }}, 
                    'Bend H-U': { Bend_H_U_Method: {
                        'Array': [Coefficients_Collection['Bend H-U']], 
                        'Latest': Coefficients_Collection['Bend H-U']
                        }}, 
                    'Bend H-D': { Bend_H_D_Method: {
                        'Array': [Coefficients_Collection['Bend H-D']], 
                        'Latest': Coefficients_Collection['Bend H-D']
                        }}, 
                    'Bend U-H': { Bend_U_H_Method: {
                        'Array': [Coefficients_Collection['Bend U-H']], 
                        'Latest': Coefficients_Collection['Bend U-H']
                        }}, 
                    'Bend D-H': { Bend_D_H_Method: {
                        'Array': [Coefficients_Collection['Bend D-H']], 
                        'Latest': Coefficients_Collection['Bend D-H']
                        }}, 
                    'Acceleration Of Solids': { Acceleration_Of_Solids_Method: {
                        'Array': [Coefficients_Collection['Acceleration Of Solids']], 
                        'Latest': Coefficients_Collection['Acceleration Of Solids']
                        }}
                    }
                    
            # Dump updated coefficient collection.
            with open('{}{}Coefficient Dump.pkl'.format(Pickle_Jar, Delimiter), 'wb') as filet:
                pickle.dump(Coefficient_Output, filet)
            
        @staticmethod
        def Pipeline_Loads(Pipeline_Files, Coefficient_Dictionary_File):
            # This function loads the data for a set of pipelines.
            Pipeline_Data = {}
            for pn, pf, sf, gf in Pipeline_Files:
                epsilon, Pipeline_Sections = mf.Miscellaneous_Functions.Import_Functions.Pipeline_Load(pf)
                d, d_v50, rho_s, m_s = mf.Miscellaneous_Functions.Import_Functions.Solids_Load(sf)
                Gas_Type, T_g, m_g = mf.Miscellaneous_Functions.Import_Functions.Gas_Load(gf)
                Pipeline_Data[pn] = {'epsilon': epsilon, 'Pipeline_Sections': Pipeline_Sections, 'd': d, 'd_v50': d_v50, 'rho_s': rho_s, 'm_s': m_s, 'Gas_Type': Gas_Type, 'T_g': T_g, 'm_g': m_g}            
            # Load coefficient dictionary.
            Coefficient_Dictionary = mf.Miscellaneous_Functions.Import_Functions.Coefficient_Dictionary_Definer(Coefficient_Dictionary_File)
            
            return Pipeline_Data, Coefficient_Dictionary
         
        @staticmethod
        def Construct_Input_Array(Coefficients):
            # This function constructs an input array from coefficients.
            Input_Array = []
            Coefficient_Lengths = []
            for c in Coefficients:
                for l in c:
                    Input_Array.append(l)
                Coefficient_Lengths.append(len(c))
            
            return Input_Array, Coefficient_Lengths
        
        @staticmethod
        def Deconstruct_Input_Array(Input_Array, Coefficient_Lengths):
            # This function takes an input array and coefficient lengths and builds the coefficient input.
            Coefficients = []
            
            count = 0
            for cl in Coefficient_Lengths:
                Intermediary = []
                
                for k in range(cl):
                    Intermediary.append(Input_Array[count])
                    count += 1
                    
                Coefficients.append(Intermediary)
            # print(Coefficients)
            
            return Coefficients
            
    
    class Dilute_Phase:
        # This class solves coefficients for a dilute phase system.
        
        @staticmethod
        def Solver(Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test=None, Parallel=None, extra_args=None):
        
            # This function solves the coefficients for a given system and set of training data.
            Type = 'Dilute'
        
            # Load pipeline data into a dictionary.
            Pipeline_Data, Coefficient_Dictionary = Coefficient_Solver.Miscellaneous_Functions.Pipeline_Loads(Pipeline_Files, Coefficient_Dictionary_File)
        
            # Pull methods from methods input.
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
                        
            # Load coefficients if already found.
            Methods_ID = '{} - HP {} VUP {} VDP {} BHH {} BHU {} BHD {} BUH {} BDH {} AS {}'.format(Type, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method, Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method)
            
            try:
                with open(r'{}{}{}.pkl'.format(Pickle_Jar, Delimiter, Methods_ID), 'rb') as filet:
                    Final_Coefficients = pickle.load(filet)
                    
                return Final_Coefficients, 0.
            except:
                pass
                        
            # Try to pull coefficient dump if it exists. 
            
            try:
                with open('{}{}{}.pkl'.format(Pickle_Jar, Delimiter, Coefficient_Dump_Location)) as filet:
                    Coefficient_Dump = pickle.load(filet)
            except:
                pass
                
            # For each method see if it exists in the coefficient dump. If it does not exist make a first guess.
            Coefficient_Collection = {}
            try:
                Coefficient_Collection['Horizontal Pipe'] = Coefficient_Dump['Horizontal Pipe'][Horizontal_Pipe_Method]['Latest']
            except:
                Coefficient_Collection['Horizontal Pipe'] = [0.5 for x in range(Coefficient_Dictionary['Dilute Horizontal'][Horizontal_Pipe_Method])]
            try:
                Coefficient_Collection['Vertical Upward Pipe'] = Coefficient_Dump['Vertical Upward Pipe'][Vertical_Upward_Pipe_Method]['Latest']
            except:
                Coefficient_Collection['Vertical Upward Pipe'] = [0.5 for x in range(Coefficient_Dictionary['Dilute Vertical'][Vertical_Upward_Pipe_Method])]
            try:
                Coefficient_Collection['Vertical Downward Pipe'] = Coefficient_Dump['Vertical Downward Pipe'][Vertical_Downward_Pipe_Method]['Latest']
            except:
                Coefficient_Collection['Vertical Downward Pipe'] = [0.5 for x in range(Coefficient_Dictionary['Dilute Vertical'][Vertical_Downward_Pipe_Method])]
            try:
                Coefficient_Collection['Bend H-H'] = Coefficient_Dump['Bend H-H'][Bend_H_H_Method]['Latest']
            except:
                Coefficient_Collection['Bend H-H'] = [0.5 for x in range(Coefficient_Dictionary['Dilute Bends'][Bend_H_H_Method])]
            try:
                Coefficient_Collection['Bend H-U'] = Coefficient_Dump['Bend H-U'][Bend_H_U_Method]['Latest']
            except:
                Coefficient_Collection['Bend H-U'] = [0.5 for x in range(Coefficient_Dictionary['Dilute Bends'][Bend_H_U_Method])]
            try:
                Coefficient_Collection['Bend H-D'] = Coefficient_Dump['Bend H-D'][Bend_H_D_Method]['Latest']
            except:
                Coefficient_Collection['Bend H-D'] = [0.5 for x in range(Coefficient_Dictionary['Dilute Bends'][Bend_H_D_Method])]
            try:
                Coefficient_Collection['Bend U-H'] = Coefficient_Dump['Bend U-H'][Bend_U_H_Method]['Latest']
            except:
                Coefficient_Collection['Bend U-H'] = [0.5 for x in range(Coefficient_Dictionary['Dilute Bends'][Bend_U_H_Method])]
            try:
                Coefficient_Collection['Bend D-H'] = Coefficient_Dump['Bend D-H'][Bend_D_H_Method]['Latest']
            except:
                Coefficient_Collection['Bend D-H'] = [0.5 for x in range(Coefficient_Dictionary['Dilute Bends'][Bend_D_H_Method])]
            try:
                Coefficient_Collection['Acceleration Of Solids'] = Coefficient_Dump['Acceleration Of Solids'][Acceleration_Of_Solids_Method]['Latest']
            except:
                Coefficient_Collection['Acceleration Of Solids'] = [0.5 for x in range(Coefficient_Dictionary['Dilute Acceleration'][Acceleration_Of_Solids_Method])]
            
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
            
            input_array, Coefficient_Lengths = Coefficient_Solver.Miscellaneous_Functions.Construct_Input_Array(Coefficients)
            
            args = (Coefficient_Lengths, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict, extra_args)
            
            if Test is None:
                Test = False
            
            if Parallel is None:
                Parallel = False
            
            if Test is True:
                Coefficient_Solver.Dilute_Phase.Iteration_Evaluator_Coefficients(input_array, Coefficient_Lengths, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict, extra_args)
                
                Final_Coefficients_Array = Coefficient_Solver.Miscellaneous_Functions.Deconstruct_Input_Array(input_array, Coefficient_Lengths)
                Cost = 0
            
            else:
                method = 'COBYLA' # Best regular
                
                if Parallel is False:
                    try:
                        # print('    Minimizing COBYLA')
                        solution_array = minimize(Coefficient_Solver.Dilute_Phase.Iteration_Evaluator_Coefficients, input_array, args=args, method=method)
                    except:
                        solution_array = minimize(Coefficient_Solver.Dilute_Phase.Iteration_Evaluator_Coefficients, input_array, args=args, method='Nelder-Mead')                        
                else:
                    solution_array = minimize_parallel(Coefficient_Solver.Dilute_Phase.Iteration_Evaluator_Coefficients, input_array, args=args, method=method)
                
                Cost = solution_array['fun']
                # print('  {:.3f}'.format(Cost))
                Final_Coefficients_Array = Coefficient_Solver.Miscellaneous_Functions.Deconstruct_Input_Array(solution_array['x'], Coefficient_Lengths)
                
            Final_Coefficients = {}
            Final_Coefficients['Horizontal Pipe'] = Final_Coefficients_Array[0]
            Final_Coefficients['Vertical Upward Pipe'] = Final_Coefficients_Array[1]
            Final_Coefficients['Vertical Downward Pipe'] = Final_Coefficients_Array[2]
            Final_Coefficients['Bend H-H'] = Final_Coefficients_Array[3]
            Final_Coefficients['Bend H-U'] = Final_Coefficients_Array[4]
            Final_Coefficients['Bend H-D'] = Final_Coefficients_Array[5]
            Final_Coefficients['Bend U-H'] = Final_Coefficients_Array[6]
            Final_Coefficients['Bend D-H'] = Final_Coefficients_Array[7]
            Final_Coefficients['Acceleration Of Solids'] = Final_Coefficients_Array[8]
            
            if Pickle_Jar is not None:
                Coefficient_Solver.Miscellaneous_Functions.Output_Coefficients(Type, Methods_Input, Final_Coefficients, Pickle_Jar, Coefficient_Dump_Location)
                
            return Final_Coefficients, Cost
            
        def Solver_Symbolic(Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Methods_Input_Reference, Section_Type, Test=None, Parallel=None, extra_args=None):
        
            # This function solves the coefficients for a given system and set of training data.
            Type = 'Dilute'
        
            # Load pipeline data into a dictionary.
            Pipeline_Data, Coefficient_Dictionary = Coefficient_Solver.Miscellaneous_Functions.Pipeline_Loads(Pipeline_Files, Coefficient_Dictionary_File)
        
            # Pull methods from methods input.
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
            
            # Pull methods from methods input reference.
            Horizontal_Pipe_Method_Reference = Methods_Input_Reference['Horizontal Pipe']
            Vertical_Upward_Pipe_Method_Reference = Methods_Input_Reference['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method_Reference = Methods_Input_Reference['Vertical Downward Pipe']
            Bend_H_H_Method_Reference = Methods_Input_Reference['Bend H-H']
            Bend_H_U_Method_Reference = Methods_Input_Reference['Bend H-U']
            Bend_H_D_Method_Reference = Methods_Input_Reference['Bend H-D']
            Bend_U_H_Method_Reference = Methods_Input_Reference['Bend U-H']
            Bend_D_H_Method_Reference = Methods_Input_Reference['Bend D-H']
            Acceleration_Of_Solids_Method_Reference = Methods_Input_Reference['Acceleration Of Solids']
                        
            # Load reference coefficients.
            Methods_ID = '{} - HP {} VUP {} VDP {} BHH {} BHU {} BHD {} BUH {} BDH {} AS {}'.format(Type, Horizontal_Pipe_Method_Reference, Vertical_Upward_Pipe_Method_Reference, Vertical_Downward_Pipe_Method_Reference, Bend_H_H_Method_Reference, Bend_H_U_Method_Reference, Bend_H_D_Method_Reference, Bend_U_H_Method_Reference, Bend_D_H_Method_Reference, Acceleration_Of_Solids_Method_Reference)
            
            with open(r'{}{}{}.pkl'.format(Pickle_Jar, Delimiter, Methods_ID), 'rb') as filet:
                Reference_Coefficients = pickle.load(filet)
            
            # Create input array
            if Section_Type == 'Horizontal Pipe':
                input_array = [0.5 for x in range(Coefficient_Dictionary['Dilute Horizontal'][Horizontal_Pipe_Method])]
            elif Section_Type == 'Vertical Upward Pipe':
                input_array = [0.5 for x in range(Coefficient_Dictionary['Dilute Vertical'][Vertical_Upward_Pipe_Method])]
            elif Section_Type == 'Vertical Downward Pipe':
                input_array = [0.5 for x in range(Coefficient_Dictionary['Dilute Vertical'][Vertical_Downward_Pipe_Method])]
            elif Section_Type == 'Bend H-H':
                input_array = [0.5 for x in range(Coefficient_Dictionary['Dilute Bend'][Bend_H_H_Method])]
            elif Section_Type == 'Bend H-U':
                input_array = [0.5 for x in range(Coefficient_Dictionary['Dilute Bend'][Bend_H_U_Method])]
            elif Section_Type == 'Bend H-D':
                input_array = [0.5 for x in range(Coefficient_Dictionary['Dilute Bend'][Bend_H_D_Method])]
            elif Section_Type == 'Bend U-H':
                input_array = [0.5 for x in range(Coefficient_Dictionary['Dilute Bend'][Bend_U_H_Method])]
            elif Section_Type == 'Bend D-H':
                input_array = [0.5 for x in range(Coefficient_Dictionary['Dilute Bend'][Bend_D_H_Method])]
            elif Section_Type == 'Acceleration Of Solids':
                input_array = [0.5 for x in range(Coefficient_Dictionary['Dilute Acceleration'][Acceleration_Of_Solids_Method])]
            
            args = (Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict, Reference_Coefficients, Section_Type, extra_args)
            
            if Test is None:
                Test = False
            
            if Parallel is None:
                Parallel = False
            
            if Test is True:
                Coefficient_Solver.Dilute_Phase.Iteration_Evaluator_Coefficients_Symbolic(input_array, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict, Reference_Coefficients, Section_Type, extra_args)
                
                Final_Coefficients_Array = input_array
                Cost = 0
            
            else:
                method = 'COBYLA' # Best regular
                
                if Parallel is False:
                    try:
                        solution_array = minimize(Coefficient_Solver.Dilute_Phase.Iteration_Evaluator_Coefficients_Symbolic, input_array, args=args, method=method, options={'maxiter': 50})
                    except:
                        solution_array = minimize(Coefficient_Solver.Dilute_Phase.Iteration_Evaluator_Coefficients_Symbolic, input_array, args=args, method='Nelder-Mead')                        
                else:
                    solution_array = minimize_parallel(Coefficient_Solver.Dilute_Phase.Iteration_Evaluator_Coefficients_Symbolic, input_array, args=args, method=method)
                
                Cost = solution_array['fun']
                Final_Coefficients_Array = solution_array['x']
                
            Final_Coefficients = Reference_Coefficients
            Final_Coefficients[Section_Type] = solution_array['x']
            
            return Final_Coefficients, Cost
            
        @staticmethod
        def Iteration_Evaluator_Coefficients(input_array, Coefficient_Lengths, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict, extra_args):
            # This function is the iterator to perform the solve.
        
            try:
            # if True:
                # Construct coefficients needed for solution from input array.
                Coefficients = Coefficient_Solver.Miscellaneous_Functions.Deconstruct_Input_Array(input_array, Coefficient_Lengths)
                
                # Evaluate error for each data point in test data.
                Differences = []
                for Data_Point in Training_Data:
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
                    
                    # Find solids flow in A side of coupled system.
                    [m_s_Final_A, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_A, Pipeline_Sections_A, d_A, d_v50_A, rho_s_A, Gas_Type_A, T_g_A, m_g_A, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_A, Outlet_Pressures_A, Function_Dict, extra_args=extra_args)
                    
                    # Find solids flow in B side of coupled system.
                    [m_s_Final_B, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_B, Pipeline_Sections_B, d_B, d_v50_B, rho_s_B, Gas_Type_B, T_g_B, m_g_B, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_B, Outlet_Pressures_B, Function_Dict, extra_args=extra_args)

                    
                    # Get total mass flow.
                    m_s_Total = m_s_Final_A + m_s_Final_B
                    
                    # Find the difference between input mass flow and the calculated flow.
                    Differences.append(abs(m_s_Total - m_s_Input))
                # print('Differences: {:,.0f} (Function Calls: {:,.0f})'.format(sum(Differences), sum([len(list(Function_Dict[k].keys())) for k in list(Function_Dict.keys())])))
            
            except:
                Differences = [1.e10]
                
            # print('      Evaluation: {:.0f}'.format(sum(Differences)))
            return sum(Differences)
        
        @staticmethod
        def Iteration_Evaluator_Coefficients_Symbolic(input_array, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict, Reference_Coefficients, Section_Type, extra_args):
            # This function is the iterator to perform the solve.            
            if ' - Pareto' in sys.argv[0]:
                Pareto = True
            else:
                Pareto = False
                
            try:
            # if True:
                # Construct coefficients needed for solution from input array.
                Coefficients_Input = Reference_Coefficients
                Coefficients_Input[Section_Type] = input_array
                
                Coefficients = [
                    Coefficients_Input['Horizontal Pipe'],
                    Coefficients_Input['Vertical Upward Pipe'],
                    Coefficients_Input['Vertical Downward Pipe'],
                    Coefficients_Input['Bend H-H'],
                    Coefficients_Input['Bend H-U'],
                    Coefficients_Input['Bend H-D'],
                    Coefficients_Input['Bend U-H'],
                    Coefficients_Input['Bend D-H'],
                    Coefficients_Input['Acceleration Of Solids']                
                    ]
                
                # Evaluate error for each data point in test data.
                Differences = []
                for Data_Point in Training_Data:
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
                    
                    # Find solids flow in A side of coupled system.
                    [m_s_Final_A, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_A, Pipeline_Sections_A, d_A, d_v50_A, rho_s_A, Gas_Type_A, T_g_A, m_g_A, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_A, Outlet_Pressures_A, Function_Dict, extra_args=extra_args)
                    
                    # Find solids flow in B side of coupled system.
                    [m_s_Final_B, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_B, Pipeline_Sections_B, d_B, d_v50_B, rho_s_B, Gas_Type_B, T_g_B, m_g_B, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_B, Outlet_Pressures_B, Function_Dict, extra_args=extra_args)

                    
                    # Get total mass flow.
                    m_s_Total = m_s_Final_A + m_s_Final_B
                    
                    # Find the difference between input mass flow and the calculated flow.
                    if Pareto is False:
                        Differences.append(abs(m_s_Total - m_s_Input))
                    else:
                        Differences.append(m_s_Total - m_s_Input)
            
            except:
                Differences = [1.e10]
                
            if Pareto is False:
                return sum(Differences)
            else:
                Differences_Mean = np.mean(Differences)
                Differences_Std = np.std(Differences)
                Differences_Pareto = abs(Differences_Mean) + Differences_Std
                return Differences_Pareto
        
        @staticmethod
        def Split_Solver(Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Split_Method):
            
            Input_Dict = {
                1: 10, # Number of coefficients for method 1
                2: 18, # Number of coefficients for method 2
                3: 18, # Number of coefficients for method 3
                4: 12, # Number of coefficients for method 4
                5: 22, # Number of coefficients for method 5
                6: 22, # Number of coefficients for method 6
                7: 0, # Number of coefficients for method 7
                8: 4, # Number of coefficients for method 8
                9: 3, # Number of coefficients for method 9
                10: 4, # Number of coefficients for method 10
                11: 3, # Number of coefficients for method 11
                12: 1, # Number of coefficients for method 12
                13: 1, # Number of coefficients for method 13
            }
            
            # This function solves the coefficients for a given system and set of training data.
            Type = 'Dilute'
        
            # Load pipeline data into a dictionary.
            Pipeline_Data, Coefficient_Dictionary = Coefficient_Solver.Miscellaneous_Functions.Pipeline_Loads(Pipeline_Files, Coefficient_Dictionary_File)
        
            # Pull methods from methods input.
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
                        
            # Load coefficients.
            Methods_ID = '{} - HP {} VUP {} VDP {} BHH {} BHU {} BHD {} BUH {} BDH {} AS {}'.format(Type, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method, Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method)
            
            with open(r'{}{}{}.pkl'.format(Pickle_Jar, Delimiter, Methods_ID), 'rb') as filet:
                Coefficient_Collection = pickle.load(filet)
                
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
                
            method = 'COBYLA'
            
            input_array = [0.5 for x in range(Input_Dict[Split_Method])]
            
            args = (Coefficients, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict, Split_Method)
            
            solution_array = minimize(Coefficient_Solver.Dilute_Phase.Iteration_Evaluator_Split_Coefficients, input_array, args=args, method=method)
            
            Cost = solution_array['fun']
            Split_Coefficients = solution_array['x']
                    
            return Split_Coefficients, Cost
            
        @staticmethod
        def Iteration_Evaluator_Split_Coefficients(input_array, Coefficients, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict, Split_Method):
            Differences = []
            for Data_Point in Training_Data:
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
                
                # Find solids flow in A side of coupled system.
                [m_s_Final_A, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_A, Pipeline_Sections_A, d_A, d_v50_A, rho_s_A, Gas_Type_A, T_g_A, m_g_A, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_A, Outlet_Pressures_A, Function_Dict, Split_Pressure_Drop=True, Split_Method=Split_Method, Split_Pressure_Drop_Coefficients=input_array)
                
                # Find solids flow in B side of coupled system.
                [m_s_Final_B, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dilute_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon_B, Pipeline_Sections_B, d_B, d_v50_B, rho_s_B, Gas_Type_B, T_g_B, m_g_B, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure_B, Outlet_Pressures_B, Function_Dict, Split_Pressure_Drop=True, Split_Method=Split_Method, Split_Pressure_Drop_Coefficients=input_array)

                
                # Get total mass flow.
                m_s_Total = m_s_Final_A + m_s_Final_B
                
                # Find the difference between input mass flow and the calculated flow.
                Differences.append(abs(m_s_Total - m_s_Input))
                
                return sum(Differences)
            
            
    class Dense_Phase:
        # This class solves coefficients for a dense phase system.
        
        @staticmethod
        def Solver(Training_Data, Pipeline_Files, Methods_Input, Coefficient_Dictionary_File, Pickle_Jar, Coefficient_Dump_Location, Function_Dict, Test=None, Parallel=None):
            # This function solves the coefficients for a given system and set of training data.
            
            Type = 'Dense'
            
            # Load pipeline data into a dictionary.
            Pipeline_Data, Coefficient_Dictionary = Coefficient_Solver.Miscellaneous_Functions.Pipeline_Loads(Pipeline_Files, Coefficient_Dictionary_File)
        
            # Pull methods from methods input.
            Horizontal_Pipe_Method = Methods_Input['Horizontal Pipe']
            Vertical_Upward_Pipe_Method = Methods_Input['Vertical Upward Pipe']
            Vertical_Downward_Pipe_Method = Methods_Input['Vertical Downward Pipe']
            Bend_H_H_Method = Methods_Input['Bend H-H']
            Bend_H_U_Method = Methods_Input['Bend H-U']
            Bend_H_D_Method = Methods_Input['Bend H-D']
            Bend_U_H_Method = Methods_Input['Bend U-H']
            Bend_D_H_Method = Methods_Input['Bend D-H']
            Acceleration_Of_Solids_Method = Methods_Input['Acceleration Of Solids']
                        
            # Load coefficients if already found.
            Methods_ID = '{} - HP {} VUP {} VDP {} BHH {} BHU {} BHD {} BUH {} BDH {} AS {}'.format(Type, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method, Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method)
            
            try:
                with open(r'{}{}{}.pkl'.format(Pickle_Jar, Delimiter, Methods_ID), 'rb') as filet:
                    Final_Coefficients = pickle.load(filet)
                    
                return Final_Coefficients, 0.
            except:
                pass
                        
            # Try to pull coefficient dump if it exists. 
            try:
                with open('{}{}{}.pkl'.format(Pickle_Jar, Delimiter, Coefficient_Dump_Location)) as filet:
                    Coefficient_Dump = pickle.load(filet)
            except:
                pass
            
            # For each method see if it exists in the coefficient dump. If it does not exist make a first guess.
            Coefficient_Collection = {}
            try:
                Coefficient_Collection['Horizontal Pipe'] = Coefficient_Dump['Horizontal Pipe'][Horizontal_Pipe_Method]['Latest']
            except:
                Coefficient_Collection['Horizontal Pipe'] = [0.5 for x in range(Coefficient_Dictionary['Dense Horizontal'][Horizontal_Pipe_Method])]
            try:
                Coefficient_Collection['Vertical Upward Pipe'] = Coefficient_Dump['Vertical Upward Pipe'][Vertical_Upward_Pipe_Method]['Latest']
            except:
                Coefficient_Collection['Vertical Upward Pipe'] = [0.5 for x in range(Coefficient_Dictionary['Dense Vertical'][Vertical_Upward_Pipe_Method])]
            try:
                Coefficient_Collection['Vertical Downward Pipe'] = Coefficient_Dump['Vertical Downward Pipe'][Vertical_Downward_Pipe_Method]['Latest']
            except:
                Coefficient_Collection['Vertical Downward Pipe'] = [0.5 for x in range(Coefficient_Dictionary['Dense Vertical'][Vertical_Downward_Pipe_Method])]
            try:
                Coefficient_Collection['Bend H-H'] = Coefficient_Dump['Bend H-H'][Bend_H_H_Method]['Latest']
            except:
                Coefficient_Collection['Bend H-H'] = [0.5 for x in range(Coefficient_Dictionary['Dense Bends'][Bend_H_H_Method])]
            try:
                Coefficient_Collection['Bend H-U'] = Coefficient_Dump['Bend H-U'][Bend_H_U_Method]['Latest']
            except:
                Coefficient_Collection['Bend H-U'] = [0.5 for x in range(Coefficient_Dictionary['Dense Bends'][Bend_H_U_Method])]
            try:
                Coefficient_Collection['Bend H-D'] = Coefficient_Dump['Bend H-D'][Bend_H_D_Method]['Latest']
            except:
                Coefficient_Collection['Bend H-D'] = [0.5 for x in range(Coefficient_Dictionary['Dense Bends'][Bend_H_D_Method])]
            try:
                Coefficient_Collection['Bend U-H'] = Coefficient_Dump['Bend U-H'][Bend_U_H_Method]['Latest']
            except:
                Coefficient_Collection['Bend U-H'] = [0.5 for x in range(Coefficient_Dictionary['Dense Bends'][Bend_U_H_Method])]
            try:
                Coefficient_Collection['Bend D-H'] = Coefficient_Dump['Bend D-H'][Bend_D_H_Method]['Latest']
            except:
                Coefficient_Collection['Bend D-H'] = [0.5 for x in range(Coefficient_Dictionary['Dense Bends'][Bend_D_H_Method])]
            try:
                Coefficient_Collection['Acceleration Of Solids'] = Coefficient_Dump['Acceleration Of Solids'][Acceleration_Of_Solids_Method]['Latest']
            except:
                Coefficient_Collection['Acceleration Of Solids'] = [0.5 for x in range(Coefficient_Dictionary['Dense Acceleration'][Acceleration_Of_Solids_Method])]
            
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
            
            input_array, Coefficient_Lengths = Coefficient_Solver.Miscellaneous_Functions.Construct_Input_Array(Coefficients)
            
            args = (Coefficient_Lengths, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict)
            
            if Test is None:
                Test = False
                
            if Parallel is None:
                Parallel = False
            
            if Test is True:
                Coefficient_Solver.Dense_Phase.Iteration_Evaluator_Coefficients(input_array, Coefficient_Lengths, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict)
                
                Final_Coefficients_Array = Coefficient_Solver.Miscellaneous_Functions.Deconstruct_Input_Array(input_array, Coefficient_Lengths)
                Cost = 0
            
            else:
                
                # method = 'L-BFGS-B' # Best parallel 
                method = 'COBYLA' # Best regular
                # method = 'SLSQP' # Fastest for profiling
                
                if Parallel is False:
                    solution_array = minimize(Coefficient_Solver.Dense_Phase.Iteration_Evaluator_Coefficients, input_array, args=args, method=method)
                else:
                    solution_array = minimize_parallel(Coefficient_Solver.Dense_Phase.Iteration_Evaluator_Coefficients, input_array, args=args)
                
                Cost = solution_array['fun']
                Final_Coefficients_Array = Coefficient_Solver.Miscellaneous_Functions.Deconstruct_Input_Array(solution_array['x'], Coefficient_Lengths)
                
            
            Final_Coefficients = {}
            Final_Coefficients['Horizontal Pipe'] = Final_Coefficients_Array[0]
            Final_Coefficients['Vertical Upward Pipe'] = Final_Coefficients_Array[1]
            Final_Coefficients['Vertical Downward Pipe'] = Final_Coefficients_Array[2]
            Final_Coefficients['Bend H-H'] = Final_Coefficients_Array[3]
            Final_Coefficients['Bend H-U'] = Final_Coefficients_Array[4]
            Final_Coefficients['Bend H-D'] = Final_Coefficients_Array[5]
            Final_Coefficients['Bend U-H'] = Final_Coefficients_Array[6]
            Final_Coefficients['Bend D-H'] = Final_Coefficients_Array[7]
            Final_Coefficients['Acceleration Of Solids'] = Final_Coefficients_Array[8]
            
            Coefficient_Solver.Miscellaneous_Functions.Output_Coefficients(Type, Methods_Input, Final_Coefficients, Pickle_Jar, Coefficient_Dump_Location)           
            
            return Final_Coefficients, Cost
            
        @staticmethod
        def Iteration_Evaluator_Coefficients(input_array, Coefficient_Lengths, Training_Data, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, Pipeline_Data, Coefficient_Dictionary, Function_Dict):
            # This function is the iterator to perform the solve.
            
            try:
                # Construct coefficients needed for solution from input array.
                Coefficients = Coefficient_Solver.Miscellaneous_Functions.Deconstruct_Input_Array(input_array, Coefficient_Lengths)
                
                # Evaluate error for each data point in test data.
                Differences = []
                for Data_Point in Training_Data:
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
                    
                    # Find solids flow in the system
                    [m_s_Final, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost] = syspdf.System_Pressure_Drop.Dense_Phase.System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure, Outlet_Pressures, Function_Dict)
                    
                    # Find the difference between input mass flow and the calculated flow.
                    Differences.append(abs(m_s_Final - m_s_Input))
                # print('Differences: {:,.0f} (Function Calls: {:,.0f})'.format(sum(Differences), sum([len(list(Function_Dict[k].keys())) for k in list(Function_Dict.keys())])))
            
            except TypeError:
                Differences = [1.e10]
            return sum(Differences)