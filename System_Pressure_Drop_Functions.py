# This code contains functions for determining the pressure drop of dilute and dense phase pneumatic conveying systems.

import numpy as np

import Section_Pressure_Drop_Functions as spdf
# from . import Section_Pressure_Drop_Functions as spdf

from scipy.optimize import least_squares, minimize
from fluids.atmosphere import ATMOSPHERE_1976
from optimparallel import minimize_parallel
from datetime import datetime

import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class System_Pressure_Drop:
    # This class contains the pressure drop calculation methods for both dilute and dense phase systems.
    
    class Dilute_Phase:
        # This class calculates the pressure drop for a dilute phase system.
        
        @staticmethod
        def System_Pressure_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict):
            
            # This code pulls the coefficients out of the coefficient input array.
            Horizontal_Pipe_Coefficients, Vertical_Upward_Pipe_Coefficients, Vertical_Downward_Pipe_Coefficients, Bend_H_H_Coefficients, Bend_H_U_Coefficients, Bend_H_D_Coefficients, Bend_U_H_Coefficients, Bend_D_H_Coefficients, Acceleration_Of_Solids_Coefficients = Coefficients
            
            # This code checks that the coefficient file has the correct number of coefficients for the system.
            if len(Horizontal_Pipe_Coefficients) != Coefficient_Dictionary['Dilute Horizontal'][Horizontal_Pipe_Method]:
                print('Wrong number of coefficients for horizontal pipe sections (Method {}: {} given and {} in dictionary)'.format(Horizontal_Pipe_Method, len(Horizontal_Pipe_Coefficients), Coefficient_Dictionary['Dilute Horizontal'][Horizontal_Pipe_Method]))
                sys.exit()
            if len(Vertical_Upward_Pipe_Coefficients) != Coefficient_Dictionary['Dilute Vertical'][Vertical_Upward_Pipe_Method]:
                print('Wrong number of coefficients for upward vertical pipe sections (Method {}: {} given and {} in dictionary)'.format(Vertical_Upward_Pipe_Method, len(Vertical_Upward_Pipe_Coefficients), Coefficient_Dictionary['Dilute Vertical'][Vertical_Upward_Pipe_Method]))
                sys.exit()
            if len(Vertical_Downward_Pipe_Coefficients) != Coefficient_Dictionary['Dilute Vertical'][Vertical_Downward_Pipe_Method]:
                print('Wrong number of coefficients for downward vertical pipe sections (Method {}: {} given and {} in dictionary)'.format(Vertical_Downward_Pipe_Method, len(Vertical_Downward_Pipe_Coefficients), Coefficient_Dictionary['Dilute Vertical'][Vertical_Downward_Pipe_Method]))
                sys.exit()
            if len(Bend_H_H_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_H_H_Method]:
                print('Wrong number of coefficients for horizontal to horizontal bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_H_H_Method, len(Bend_H_H_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_H_H_Method]))
                sys.exit()
            if len(Bend_H_U_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_H_U_Method]:
                print('Wrong number of coefficients for horizontal to upward vertical bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_H_U_Method, len(Bend_H_U_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_H_U_Method]))
                sys.exit()
            if len(Bend_H_D_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_H_D_Method]:
                print('Wrong number of coefficients for horizontal to downward vertical bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_H_D_Method, len(Bend_H_D_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_H_D_Method]))
                sys.exit()
            if len(Bend_U_H_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_U_H_Method]:
                print('Wrong number of coefficients for upward vertical to horizontal bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_U_H_Method, len(Bend_U_H_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_U_H_Method]))
                sys.exit()
            if len(Bend_D_H_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_D_H_Method]:
                print('Wrong number of coefficients for downward vertical to horizontal bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_D_H_Method, len(Bend_D_H_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_D_H_Method]))
                sys.exit()
            if len(Acceleration_Of_Solids_Coefficients) != Coefficient_Dictionary['Dilute Acceleration'][Acceleration_Of_Solids_Method]:
                print('Wrong number of coefficients for acceleration of solids (Method {}: {} given and {} in dictionary)'.format(Acceleration_Of_Solids_Method, len(Acceleration_Of_Solids_Coefficients), Coefficient_Dictionary['Dilute Acceleration'][Acceleration_Of_Solids_Method]))
                sys.exit()
            
            # Create the initial guesses for system inlet pressure and flow split.
            System_Inlet_Pressure = Outlet_Pressures[0] * 1.01
            input_array = [System_Inlet_Pressure]
            flow_guesses_gas = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            flow_guesses_solids = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            for fg in flow_guesses_gas:
                input_array.append(fg)
            for fg in flow_guesses_solids:
                input_array.append(fg)
            
            # Run the root function to solve for flow split and system inlet pressures.
            args = (Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, System_Inlet_Pressure, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict)
            bounds = [[], []]
            for k in range(len(input_array)):
                if k == 0:
                    bounds[0].append(min(Outlet_Pressures))
                    bounds[1].append(np.inf)
                else:
                    bounds[0].append(0.)
                    bounds[1].append(1.)
            solution_array = least_squares(System_Pressure_Drop.Dilute_Phase.Iteration_Evaluator_Pressure, input_array, args=args, bounds=bounds)
            
            # Pull out final inlet pressure and flow splits from the root function.
            Flow_Splits_Gas_Final = []
            Flow_Splits_Solids_Final = []
            for k in range(len(solution_array['x'])):
                if k == 0:
                    System_Inlet_Pressure_Final = solution_array['x'][k]
                elif k <= len(flow_guesses_gas):
                    Flow_Splits_Gas_Final.append(solution_array['x'][k])
                else:
                    Flow_Splits_Solids_Final.append(solution_array['x'][k])
            
            # Evaluate the solved system for section pressures.
            Section_Pressures, Path_Pressures = System_Pressure_Drop.Dilute_Phase.Pressure_Evaluator(System_Inlet_Pressure_Final, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict)
            
            Cost = solution_array['fun']
            
            return System_Inlet_Pressure_Final, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost
            
        @staticmethod
        def System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure, Outlet_Pressures, Function_Dict, extra_args=None, Split_Pressure_Drop=False, Split_Method=None, Split_Pressure_Drop_Coefficients=None):
        
            # This code pulls the coefficients out of the coefficient input array.
            Horizontal_Pipe_Coefficients, Vertical_Upward_Pipe_Coefficients, Vertical_Downward_Pipe_Coefficients, Bend_H_H_Coefficients, Bend_H_U_Coefficients, Bend_H_D_Coefficients, Bend_U_H_Coefficients, Bend_D_H_Coefficients, Acceleration_Of_Solids_Coefficients = Coefficients
            
            # This code checks that the coefficient file has the correct number of coefficients for the system.
            if len(Horizontal_Pipe_Coefficients) != Coefficient_Dictionary['Dilute Horizontal'][Horizontal_Pipe_Method]:
                print('Wrong number of coefficients for horizontal pipe sections (Method {}: {} given and {} in dictionary)'.format(Horizontal_Pipe_Method, len(Horizontal_Pipe_Coefficients), Coefficient_Dictionary['Dilute Horizontal'][Horizontal_Pipe_Method]))
                sys.exit()
            if len(Vertical_Upward_Pipe_Coefficients) != Coefficient_Dictionary['Dilute Vertical'][Vertical_Upward_Pipe_Method]:
                print('Wrong number of coefficients for upward vertical pipe sections (Method {}: {} given and {} in dictionary)'.format(Vertical_Upward_Pipe_Method, len(Vertical_Upward_Pipe_Coefficients), Coefficient_Dictionary['Dilute Vertical'][Vertical_Upward_Pipe_Method]))
                sys.exit()
            if len(Vertical_Downward_Pipe_Coefficients) != Coefficient_Dictionary['Dilute Vertical'][Vertical_Downward_Pipe_Method]:
                print('Wrong number of coefficients for downward vertical pipe sections (Method {}: {} given and {} in dictionary)'.format(Vertical_Downward_Pipe_Method, len(Vertical_Downward_Pipe_Coefficients), Coefficient_Dictionary['Dilute Vertical'][Vertical_Downward_Pipe_Method]))
                sys.exit()
            if len(Bend_H_H_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_H_H_Method]:
                print('Wrong number of coefficients for horizontal to horizontal bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_H_H_Method, len(Bend_H_H_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_H_H_Method]))
                sys.exit()
            if len(Bend_H_U_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_H_U_Method]:
                print('Wrong number of coefficients for horizontal to upward vertical bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_H_U_Method, len(Bend_H_U_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_H_U_Method]))
                sys.exit()
            if len(Bend_H_D_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_H_D_Method]:
                print('Wrong number of coefficients for horizontal to downward vertical bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_H_D_Method, len(Bend_H_D_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_H_D_Method]))
                sys.exit()
            if len(Bend_U_H_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_U_H_Method]:
                print('Wrong number of coefficients for upward vertical to horizontal bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_U_H_Method, len(Bend_U_H_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_U_H_Method]))
                sys.exit()
            if len(Bend_D_H_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_D_H_Method]:
                print('Wrong number of coefficients for downward vertical to horizontal bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_D_H_Method, len(Bend_D_H_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_D_H_Method]))
                sys.exit()
            if len(Acceleration_Of_Solids_Coefficients) != Coefficient_Dictionary['Dilute Acceleration'][Acceleration_Of_Solids_Method]:
                print('Wrong number of coefficients for acceleration of solids (Method {}: {} given and {} in dictionary)'.format(Acceleration_Of_Solids_Method, len(Acceleration_Of_Solids_Coefficients), Coefficient_Dictionary['Dilute Acceleration'][Acceleration_Of_Solids_Method]))
                sys.exit()
            
            # Create the initial guesses for solids feed rate and flow split.
            m_s = 0.5 # kg/s
            input_array = [m_s]
            flow_guesses_gas = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            flow_guesses_solids = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            for fg in flow_guesses_gas:
                input_array.append(fg)
            for fg in flow_guesses_solids:
                input_array.append(fg)
            
            # Determine extra arguments to pass to section pipe flows.
            if Horizontal_Pipe_Method == 87:
                func = extra_args[0]
                extra_args_system = [func]
            else:
                extra_args_system = None
            
            # Run the root function to solve for flow split and solids feed rate.
            args = (Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, System_Inlet_Pressure, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict, extra_args_system, Split_Pressure_Drop, Split_Method, Split_Pressure_Drop_Coefficients)
            bounds = [[], []]
            for k in range(len(input_array)):
                if k == 0:
                    bounds[0].append(0.)
                    bounds[1].append(np.inf)
                else:
                    bounds[0].append(0.)
                    bounds[1].append(1.)
            solution_array = least_squares(System_Pressure_Drop.Dilute_Phase.Iteration_Evaluator_Solids_Flow, input_array, args=args, bounds=bounds)
            
            # if solution_array['fun'] > 20000. and Vertical_Upward_Pipe_Method != 34:
                # bounds = []
                # for k in range(len(input_array)):
                    # if k == 0:
                        # bounds.append((0., np.inf))
                    # else:
                        # bounds.append((0., 1.))
                # method = 'BFGS'
                # solution_array = minimize(System_Pressure_Drop.Dilute_Phase.Iteration_Evaluator_Solids_Flow, input_array, args=args, bounds=bounds, method=method)
            
            # Pull out final inlet pressure and flow splits from the root function.
            Flow_Splits_Gas_Final = []
            Flow_Splits_Solids_Final = []
            for k in range(len(solution_array['x'])):
                if k == 0:
                    m_s_Final = solution_array['x'][k]
                elif k <= len(flow_guesses_gas):
                    Flow_Splits_Gas_Final.append(solution_array['x'][k])
                else:
                    Flow_Splits_Solids_Final.append(solution_array['x'][k])
                    
            # Evaluate the solved system for section pressures.
            Section_Pressures, Path_Pressures, Penalty = System_Pressure_Drop.Dilute_Phase.Pressure_Evaluator(System_Inlet_Pressure, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s_Final, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict, extra_args=extra_args_system, Split_Pressure_Drop=Split_Pressure_Drop, Split_Method=Split_Method, Split_Pressure_Drop_Coefficients=Split_Pressure_Drop_Coefficients)
            
            Cost = solution_array['fun']
            
            return m_s_Final, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost
            
        @staticmethod
        def Iteration_Evaluator_Pressure(input_array, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, System_Inlet_Pressure, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict):
            # This function is used to minimize the difference between the desired output pressures and those calculated in order to find flow splits and system inlet pressure.
            
            # This section pulls the needed data out of the input array
            Flow_Splits_Gas = []
            Flow_Splits_Solids = []
            for k, i in zip(list(range(len(input_array))), input_array):
                if k == 0:
                    System_Inlet_Pressure = i
                elif k <= (len(input_array) - 1) / 2:
                    Flow_Splits_Gas.append(i)
                else:
                    Flow_Splits_Solids.append(i)
                    
            # This calculates the section pressures for the inputted inlet pressure and flow splits.
            try:
                Section_Pressures, Path_Pressures  = System_Pressure_Drop.Dilute_Phase.Pressure_Evaluator(System_Inlet_Pressure, Flow_Splits_Gas, Flow_Splits_Solids, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict)
            except:
                return 1000000.
            
            # This section of code determines which sections are outlet sections.
            Flow_Paths = []
            [Flow_Paths.append(section[1]) for section in Pipeline_Sections if section[1] not in Flow_Paths]
            
            Max_Sections = {}
            for flow in Flow_Paths:
               Max_Sections[flow] = max([section[0] for section in Pipeline_Sections if section[1] == flow])
            
            for flow in Flow_Paths:
                if Pipeline_Sections[Max_Sections[flow]][2] ==  'Splitter':
                    del Max_Sections[flow]
            
            # This section of code finds the calculated outlet pressures.
            Calculated_Outlet_Pressures = []
            for k in list(Max_Sections.keys()):
                Calculated_Outlet_Pressures.append(Section_Pressures[Max_Sections[k] + 1])
            
            # This compares the calculated and given outlet pressures and returns the sum of the differences.
            Differences = []
            Total_Absolute_Difference = 0.
            for op, cop in zip(Outlet_Pressures, Calculated_Outlet_Pressures):
                if np.isnan(op - cop) or np.isinf(op - cop):
                    Differences.append(10000000.)
                else:
                    Differences.append(abs(op - cop))
                Total_Absolute_Difference += abs(op - cop)
                
            for k in range(len(input_array) - len(Differences)):
                Differences.append(0.)
                
            # Determine if the solution satisfies all function input conditions. If not, applies a penalty to the returned differences.
            try:
                Flow_Splits_Gas_Max = max(Flow_Splits_Gas)
                Flow_Splits_Gas_Min =  min(Flow_Splits_Gas)
                Flow_Splits_Solids_Max = max(Flow_Splits_Solids)
                Flow_Splits_Solids_Min =  min(Flow_Splits_Solids)
            except:
                Flow_Splits_Gas_Max = 0.5
                Flow_Splits_Gas_Min =  0.5
                Flow_Splits_Solids_Max = 0.5
                Flow_Splits_Solids_Min =  0.5
            
            penalty = 10000000.
            if System_Inlet_Pressure < max(Outlet_Pressures) or Flow_Splits_Gas_Max > 1. or Flow_Splits_Gas_Min < 0. or Flow_Splits_Solids_Max > 1. or Flow_Splits_Solids_Min < 0.:
                Differences = [abs(d) + penalty for d in Differences]
            
            if sum(Differences) > 1.E100:
                return 1.E100
                
            return sum(Differences)
            
        @staticmethod
        def Iteration_Evaluator_Solids_Flow(input_array, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, System_Inlet_Pressure, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict, extra_args_system, Split_Pressure_Drop, Split_Method, Split_Pressure_Drop_Coefficients):
            # This function is used to minimize the difference between the desired output pressures and those calculated in order to find flow splits and solids feed rate.
            
            # This section pulls the needed data out of the input array
            Flow_Splits_Gas = []
            Flow_Splits_Solids = []
            for k, i in zip(list(range(len(input_array))), input_array):
                if k == 0:
                    m_s = i
                elif k <= (len(input_array) - 1) / 2:
                    Flow_Splits_Gas.append(i)
                else:
                    Flow_Splits_Solids.append(i)
                    
            # This calculates the section pressures for the inputted inlet pressure and flow splits.
            try:
            # if True:
                Section_Pressures, Path_Pressures, Penalty  = System_Pressure_Drop.Dilute_Phase.Pressure_Evaluator(System_Inlet_Pressure, Flow_Splits_Gas, Flow_Splits_Solids, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict, extra_args=extra_args_system, Split_Pressure_Drop=Split_Pressure_Drop, Split_Method=Split_Method, Split_Pressure_Drop_Coefficients=Split_Pressure_Drop_Coefficients)
            except:
                return 1000000.
            
            # This section of code determines which sections are outlet sections.
            Flow_Paths = []
            [Flow_Paths.append(section[1]) for section in Pipeline_Sections if section[1] not in Flow_Paths]
            
            Max_Sections = {}
            for flow in Flow_Paths:
               Max_Sections[flow] = max([section[0] for section in Pipeline_Sections if section[1] == flow])
            
            for flow in Flow_Paths:
                if Pipeline_Sections[Max_Sections[flow]][2] ==  'Splitter':
                    del Max_Sections[flow]
            
            # This section of code finds the calculated outlet pressures.
            Calculated_Outlet_Pressures = []
            for k in list(Max_Sections.keys()):
                Calculated_Outlet_Pressures.append(Section_Pressures[Max_Sections[k] + 1])
            
            # This compares the calculated and given outlet pressures and returns the sum of the differences.
            Differences = []
            Total_Absolute_Difference = 0.
            for op, cop in zip(Outlet_Pressures, Calculated_Outlet_Pressures):
                if np.isnan(op - cop) or np.isinf(op - cop):
                    Differences.append(10000000.)
                else:
                    Differences.append(abs(op - cop))
                Total_Absolute_Difference += abs(op - cop)
                
            for k in range(len(input_array) - len(Differences)):
                Differences.append(0.)
                
            # Determine if the solution satisfies all function input conditions. If not, applies a penalty to the returned differences.
            try:
                Flow_Splits_Gas_Max = max(Flow_Splits_Gas)
                Flow_Splits_Gas_Min =  min(Flow_Splits_Gas)
                Flow_Splits_Solids_Max = max(Flow_Splits_Solids)
                Flow_Splits_Solids_Min =  min(Flow_Splits_Solids)
            except:
                Flow_Splits_Gas_Max = 0.5
                Flow_Splits_Gas_Min =  0.5
                Flow_Splits_Solids_Max = 0.5
                Flow_Splits_Solids_Min =  0.5
            
            # Determine if the solution satisfies all function input conditions. If not, applies a penalty to the returned differences.
            if len(Flow_Splits_Gas) > 0:
            
                penalty = 10000000.
                if m_s < 0. or Flow_Splits_Gas_Max > 1. or Flow_Splits_Gas_Min < 0. or Flow_Splits_Solids_Max > 1. or Flow_Splits_Solids_Min < 0.:
                    Differences = [d + penalty for d in Differences]
                
            # Determine if value is NaN or Inf and apply penalty if so.
            Differences_Value_Check = []
            for d in Differences:
                if np.isnan(d):
                    Differences_Value_Check.append(penalty)
                elif np.isinf(d):
                    Differences_Value_Check.append(penalty)
                else:
                    Differences_Value_Check.append(d)
            Differences = Differences_Value_Check
            
            return abs(sum(Differences))
        
        @staticmethod
        def Pressure_Evaluator(System_Inlet_Pressure, Flow_Splits_Gas, Flow_Splits_Solids, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict, extra_args=None, Split_Pressure_Drop=False, Split_Method=None, Split_Pressure_Drop_Coefficients=None):
            # This function calculates the pressure drop for a dilute phase system given system inlet pressure and flow splits.
            
            # This code pulls the coefficients out of the coefficient input array.
            Horizontal_Pipe_Coefficients, Vertical_Upward_Pipe_Coefficients, Vertical_Downward_Pipe_Coefficients, Bend_H_H_Coefficients, Bend_H_U_Coefficients, Bend_H_D_Coefficients, Bend_U_H_Coefficients, Bend_D_H_Coefficients, Acceleration_Of_Solids_Coefficients = Coefficients
            
            # Import gas properties.
            if Gas_Type == 'Air':
                R_g = ATMOSPHERE_1976.R
            
            # This defines the dictionary for different flows.
            Flow_Splits_Dictionary = {}
            count = 0
            for s in Pipeline_Sections:
                if s[2] == 'Splitter':
                    Flow_Splits_Dictionary[s[0]] = count
                    count += 1
            
            # This makes a list of the flow paths in the pipeline.
            Flow_Paths = list(range(1 + len([s for s in Pipeline_Sections if s[2] == 'Splitter' ]) * 2))
            Flows = [(m_g, m_s)]
            Flows_Dictionary = {0: 0}
            
            Path_Pressures = [[] for k in Flow_Paths]
                
            # This code finds pressure drop for each pipe section.         
            Section_Pressures = [System_Inlet_Pressure]
            Path_Pressures[0].append(System_Inlet_Pressure)
            
            Penalty = False
            for section in Pipeline_Sections:
                section_number = section[0]
                flow_path = section[1]
                section_type = section[2]
                orientation = section[3]
                L_c = section[4]
                D = section[5]
                
                path_flows = Flows[Flows_Dictionary[flow_path]]
                m_g = path_flows[0]
                m_s = path_flows[1]
                
                P_i = Path_Pressures[flow_path][-1]
                
                
                # Determine extra arguments to pass to section pipe flows.
                if Horizontal_Pipe_Method == 87:
                    func = extra_args[0]
                    if section_number == 0:
                        extra_args_section = [func, orientation, L_c]
                    else:
                        extra_args_section = [func, orientation]
                else:
                    extra_args_section = None

                # This section of code calculates the inlet gas properties for the section.
                if P_i > 0. and not np.iscomplex(P_i):
                    if Gas_Type == 'Air':
                        rho_g = ATMOSPHERE_1976.density(T_g, P_i)
                        mu_g = ATMOSPHERE_1976.viscosity(T_g)
                        
                    # This section of code calculates the pressure drop due to solids acceleration in the first section of the system.
                    if section_number == 0:
                        Pd_additional = spdf.Section_Pressure_Drop.Dilute_Phase.Acceleration_Of_Solids(Acceleration_Of_Solids_Method, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g, m_s, R_g, T_g, mu_g, Acceleration_Of_Solids_Coefficients, Function_Dict, extra_args=extra_args_section)
                    else:
                        Pd_additional = 0.
                        
                    if section_type == 'Horizontal':
                        Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Horizontal_Pipe_Sections(Horizontal_Pipe_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, Horizontal_Pipe_Coefficients, Function_Dict, extra_args=extra_args_section)
                        
                    elif section_type == 'Vertical':
                        if orientation == 'Upward':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Vertical_Pipe_Sections(Vertical_Upward_Pipe_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Vertical_Upward_Pipe_Coefficients, Function_Dict, extra_args=extra_args_section)
                            
                        elif orientation == 'Downward':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Vertical_Pipe_Sections(Vertical_Downward_Pipe_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, -1., Vertical_Downward_Pipe_Coefficients, Function_Dict, extra_args=extra_args_section)
                            
                        else:
                            print('Invalid orientation for section {}'.format(section_number))
                            sys.exit()
                            
                    elif section_type == 'Bend':
                        bend_inlet, bend_outlet = orientation.split('-')
                        if bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet in ['N', 'E', 'S', 'W']:
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_H_H_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Bend_H_H_Coefficients, Function_Dict, extra_args=extra_args_section)
                        elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'U':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_H_U_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Bend_H_U_Coefficients, Function_Dict, extra_args=extra_args_section)
                        elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'D':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_H_D_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, -1., Bend_H_D_Coefficients, Function_Dict, extra_args=extra_args_section)
                        elif bend_inlet == 'U':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_U_H_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Bend_U_H_Coefficients, Function_Dict, extra_args=extra_args_section)
                        elif bend_inlet == 'D':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_D_H_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, -1., Bend_D_H_Coefficients, Function_Dict, extra_args=extra_args_section)
                            
                    elif section_type == 'Splitter':
                        if Split_Pressure_Drop is True:
                            if Split_Method == 7:
                                m_g_1 = m_g * Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]]
                                m_s_1 = m_s * Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]]
                                m_g_2 = m_g * (1. - Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]])
                                m_s_2 = m_s * (1. - Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]])
                                
                                extra_args_split = (Acceleration_Of_Solids_Method, Acceleration_Of_Solids_Coefficients, m_g_1, m_s_1, m_g_2, m_s_2)
                                
                                Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Split(Split_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g, m_s, R_g, T_g, mu_g, Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]], Split_Pressure_Drop_Coefficients, Function_Dict, extra_args=extra_args_split)
                                
                            elif Split_Method in [8, 9, 10, 11, 12, 13]:                            
                                m_g_1 = m_g * Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]]
                                m_s_1 = m_s * Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]]
                                m_g_2 = m_g * (1. - Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]])
                                m_s_2 = m_s * (1. - Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]])
                                
                                extra_args_split = (m_g_1, m_s_1, m_g_2, m_s_2)
                                
                                Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Split(Split_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g, m_s, R_g, T_g, mu_g, Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]], Split_Pressure_Drop_Coefficients, Function_Dict, extra_args=extra_args_split)
                                
                            else:
                                Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Split(Split_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g, m_s, R_g, T_g, mu_g, Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]], Split_Pressure_Drop_Coefficients, Function_Dict, extra_args=extra_args_section)
                                
                        else:
                            Pd = 0.
                            
                        inlet_path = flow_path
                        outlet_path_1 = section[5]
                        outlet_path_2 = section[6]
                        
                        Flows.append((m_g * Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], m_s * Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]]))
                        Flows.append((m_g * (1. - Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]]), m_s * (1. - Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]])))
                        
                        Flows_Dictionary[outlet_path_1] = len(Flows) - 2
                        Flows_Dictionary[outlet_path_2] = len(Flows) - 1
                        
                        if Split_Pressure_Drop is True and Split_Method in [7, 8, 9, 10, 11, 12, 13]:
                            Path_Pressures[outlet_path_1].append(P_i - Pd[0] - Pd_additional)
                            Path_Pressures[outlet_path_2].append(P_i - Pd[1] - Pd_additional)
                            Pd = sum(Pd) / 2.
                        else:
                            Path_Pressures[outlet_path_1].append(P_i - Pd - Pd_additional)
                            Path_Pressures[outlet_path_2].append(P_i - Pd - Pd_additional)
                    
                    Section_Pressures.append(P_i - Pd - Pd_additional)
                    if section_type != 'Splitter':
                        Path_Pressures[flow_path].append(P_i - Pd - Pd_additional)
                    
                    if Pd > 1000000.:
                        Penalty = True
                    elif Pd < 0.:
                        Penalty = True
                    
                    if Pd_additional > 1000000.:
                        Penalty = True
                    elif Pd_additional < 0.:
                        Penalty = True
                    
                else:
                    Section_Pressures.append(Section_Pressures[-1])
                    
                    if section_type == 'Splitter':
                        inlet_path = flow_path
                        outlet_path_1 = section[5]
                        outlet_path_2 = section[6]
                        
                        Flows.append((m_g * Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], m_s * Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]]))
                        Flows.append((m_g * (1. - Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]]), m_s * (1. - Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]])))
                        
                        Flows_Dictionary[outlet_path_1] = len(Flows) - 2
                        Flows_Dictionary[outlet_path_2] = len(Flows) - 1
                        
                        Path_Pressures[outlet_path_1].append(Section_Pressures[-1])
                        Path_Pressures[outlet_path_2].append(Section_Pressures[-1])
                    
                    if section_type != 'Splitter':
                        Path_Pressures[flow_path].append(Section_Pressures[-1])
                        
            return Section_Pressures, Path_Pressures, Penalty
        
        @staticmethod
        def System_Solids_Flow_Solver_Dump(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure, Outlet_Pressures, Function_Dict, extra_args=None, Split_Pressure_Drop=False, Split_Method=None, Split_Pressure_Drop_Coefficients=None):
        
            # This code pulls the coefficients out of the coefficient input array.
            Horizontal_Pipe_Coefficients, Vertical_Upward_Pipe_Coefficients, Vertical_Downward_Pipe_Coefficients, Bend_H_H_Coefficients, Bend_H_U_Coefficients, Bend_H_D_Coefficients, Bend_U_H_Coefficients, Bend_D_H_Coefficients, Acceleration_Of_Solids_Coefficients = Coefficients
            
            # This code checks that the coefficient file has the correct number of coefficients for the system.
            if len(Horizontal_Pipe_Coefficients) != Coefficient_Dictionary['Dilute Horizontal'][Horizontal_Pipe_Method]:
                print('Wrong number of coefficients for horizontal pipe sections (Method {}: {} given and {} in dictionary)'.format(Horizontal_Pipe_Method, len(Horizontal_Pipe_Coefficients), Coefficient_Dictionary['Dilute Horizontal'][Horizontal_Pipe_Method]))
                sys.exit()
            if len(Vertical_Upward_Pipe_Coefficients) != Coefficient_Dictionary['Dilute Vertical'][Vertical_Upward_Pipe_Method]:
                print('Wrong number of coefficients for upward vertical pipe sections (Method {}: {} given and {} in dictionary)'.format(Vertical_Upward_Pipe_Method, len(Vertical_Upward_Pipe_Coefficients), Coefficient_Dictionary['Dilute Vertical'][Vertical_Upward_Pipe_Method]))
                sys.exit()
            if len(Vertical_Downward_Pipe_Coefficients) != Coefficient_Dictionary['Dilute Vertical'][Vertical_Downward_Pipe_Method]:
                print('Wrong number of coefficients for downward vertical pipe sections (Method {}: {} given and {} in dictionary)'.format(Vertical_Downward_Pipe_Method, len(Vertical_Downward_Pipe_Coefficients), Coefficient_Dictionary['Dilute Vertical'][Vertical_Downward_Pipe_Method]))
                sys.exit()
            if len(Bend_H_H_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_H_H_Method]:
                print('Wrong number of coefficients for horizontal to horizontal bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_H_H_Method, len(Bend_H_H_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_H_H_Method]))
                sys.exit()
            if len(Bend_H_U_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_H_U_Method]:
                print('Wrong number of coefficients for horizontal to upward vertical bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_H_U_Method, len(Bend_H_U_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_H_U_Method]))
                sys.exit()
            if len(Bend_H_D_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_H_D_Method]:
                print('Wrong number of coefficients for horizontal to downward vertical bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_H_D_Method, len(Bend_H_D_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_H_D_Method]))
                sys.exit()
            if len(Bend_U_H_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_U_H_Method]:
                print('Wrong number of coefficients for upward vertical to horizontal bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_U_H_Method, len(Bend_U_H_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_U_H_Method]))
                sys.exit()
            if len(Bend_D_H_Coefficients) != Coefficient_Dictionary['Dilute Bends'][Bend_D_H_Method]:
                print('Wrong number of coefficients for downward vertical to horizontal bend sections (Method {}: {} given and {} in dictionary)'.format(Bend_D_H_Method, len(Bend_D_H_Coefficients), Coefficient_Dictionary['Dilute Bends'][Bend_D_H_Method]))
                sys.exit()
            if len(Acceleration_Of_Solids_Coefficients) != Coefficient_Dictionary['Dilute Acceleration'][Acceleration_Of_Solids_Method]:
                print('Wrong number of coefficients for acceleration of solids (Method {}: {} given and {} in dictionary)'.format(Acceleration_Of_Solids_Method, len(Acceleration_Of_Solids_Coefficients), Coefficient_Dictionary['Dilute Acceleration'][Acceleration_Of_Solids_Method]))
                sys.exit()
            
            # Create the initial guesses for solids feed rate and flow split.
            m_s = 0.5 # kg/s
            input_array = [m_s]
            flow_guesses_gas = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            flow_guesses_solids = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            for fg in flow_guesses_gas:
                input_array.append(fg)
            for fg in flow_guesses_solids:
                input_array.append(fg)
            
            # Determine extra arguments to pass to section pipe flows.
            if Horizontal_Pipe_Method == 87:
                func = extra_args[0]
                extra_args_system = [func]
            else:
                extra_args_system = None
            
            # Run the root function to solve for flow split and solids feed rate.
            args = (Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, System_Inlet_Pressure, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict, extra_args_system, Split_Pressure_Drop, Split_Method, Split_Pressure_Drop_Coefficients)
            bounds = [[], []]
            for k in range(len(input_array)):
                if k == 0:
                    bounds[0].append(0.)
                    bounds[1].append(np.inf)
                else:
                    bounds[0].append(0.)
                    bounds[1].append(1.)
            solution_array = least_squares(System_Pressure_Drop.Dilute_Phase.Iteration_Evaluator_Solids_Flow, input_array, args=args, bounds=bounds)
            
            # if solution_array['fun'] > 20000. and Vertical_Upward_Pipe_Method != 34:
                # bounds = []
                # for k in range(len(input_array)):
                    # if k == 0:
                        # bounds.append((0., np.inf))
                    # else:
                        # bounds.append((0., 1.))
                # method = 'BFGS'
                # solution_array = minimize(System_Pressure_Drop.Dilute_Phase.Iteration_Evaluator_Solids_Flow, input_array, args=args, bounds=bounds, method=method)
            
            # Pull out final inlet pressure and flow splits from the root function.
            Flow_Splits_Gas_Final = []
            Flow_Splits_Solids_Final = []
            for k in range(len(solution_array['x'])):
                if k == 0:
                    m_s_Final = solution_array['x'][k]
                elif k <= len(flow_guesses_gas):
                    Flow_Splits_Gas_Final.append(solution_array['x'][k])
                else:
                    Flow_Splits_Solids_Final.append(solution_array['x'][k])
                    
            # Evaluate the solved system for section pressures.
            Section_Pressures, Path_Pressures, Penalty = System_Pressure_Drop.Dilute_Phase.Pressure_Evaluator_Dump(System_Inlet_Pressure, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s_Final, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict, extra_args=extra_args_system, Split_Pressure_Drop=Split_Pressure_Drop, Split_Method=Split_Method, Split_Pressure_Drop_Coefficients=Split_Pressure_Drop_Coefficients)
            
            Cost = solution_array['fun']
            
            return m_s_Final, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost
            
        @staticmethod
        def Pressure_Evaluator_Dump(System_Inlet_Pressure, Flow_Splits_Gas, Flow_Splits_Solids, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict, extra_args=None, Split_Pressure_Drop=False, Split_Method=None, Split_Pressure_Drop_Coefficients=None):
            # This function calculates the pressure drop for a dilute phase system given system inlet pressure and flow splits.
            
            # Identify time for unique identification.
            now = datetime.now()
            
            # This code pulls the coefficients out of the coefficient input array.
            Horizontal_Pipe_Coefficients, Vertical_Upward_Pipe_Coefficients, Vertical_Downward_Pipe_Coefficients, Bend_H_H_Coefficients, Bend_H_U_Coefficients, Bend_H_D_Coefficients, Bend_U_H_Coefficients, Bend_D_H_Coefficients, Acceleration_Of_Solids_Coefficients = Coefficients
            
            # Import gas properties.
            if Gas_Type == 'Air':
                R_g = ATMOSPHERE_1976.R
            
            # This defines the dictionary for different flows.
            Flow_Splits_Dictionary = {}
            count = 0
            for s in Pipeline_Sections:
                if s[2] == 'Splitter':
                    Flow_Splits_Dictionary[s[0]] = count
                    count += 1
            
            # This makes a list of the flow paths in the pipeline.
            Flow_Paths = list(range(1 + len([s for s in Pipeline_Sections if s[2] == 'Splitter' ]) * 2))
            Flows = [(m_g, m_s)]
            Flows_Dictionary = {0: 0}
            
            Path_Pressures = [[] for k in Flow_Paths]
                
            # This code finds pressure drop for each pipe section.         
            Section_Pressures = [System_Inlet_Pressure]
            Path_Pressures[0].append(System_Inlet_Pressure)
            
            Penalty = False
            for section in Pipeline_Sections:
                section_number = section[0]
                flow_path = section[1]
                section_type = section[2]
                orientation = section[3]
                L_c = section[4]
                D = section[5]
                
                path_flows = Flows[Flows_Dictionary[flow_path]]
                m_g = path_flows[0]
                m_s = path_flows[1]
                
                P_i = Path_Pressures[flow_path][-1]
                
                
                # Determine extra arguments to pass to section pipe flows.
                if Horizontal_Pipe_Method == 87:
                    func = extra_args[0]
                    if section_number == 0:
                        extra_args_section = [func, orientation, L_c]
                    else:
                        extra_args_section = [func, orientation]
                else:
                    extra_args_section = None

                # This section of code calculates the inlet gas properties for the section.
                if P_i > 0. and not np.iscomplex(P_i):
                    if Gas_Type == 'Air':
                        rho_g = ATMOSPHERE_1976.density(T_g, P_i)
                        mu_g = ATMOSPHERE_1976.viscosity(T_g)
                        
                    # This section of code calculates the pressure drop due to solids acceleration in the first section of the system.
                    if section_number == 0:
                        Pd_additional = spdf.Section_Pressure_Drop.Dilute_Phase.Acceleration_Of_Solids(Acceleration_Of_Solids_Method, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g, m_s, R_g, T_g, mu_g, Acceleration_Of_Solids_Coefficients, Function_Dict, extra_args=extra_args_section)
                    else:
                        Pd_additional = 0.
                        
                    if section_type == 'Horizontal':
                        Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Horizontal_Pipe_Sections(Horizontal_Pipe_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, Horizontal_Pipe_Coefficients, Function_Dict, extra_args=extra_args_section)
                        
                    elif section_type == 'Vertical':
                        if orientation == 'Upward':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Vertical_Pipe_Sections(Vertical_Upward_Pipe_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Vertical_Upward_Pipe_Coefficients, Function_Dict, extra_args=extra_args_section)
                            
                        elif orientation == 'Downward':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Vertical_Pipe_Sections(Vertical_Downward_Pipe_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, -1., Vertical_Downward_Pipe_Coefficients, Function_Dict, extra_args=extra_args_section)
                            
                        else:
                            print('Invalid orientation for section {}'.format(section_number))
                            sys.exit()
                            
                    elif section_type == 'Bend':
                        bend_inlet, bend_outlet = orientation.split('-')
                        if bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet in ['N', 'E', 'S', 'W']:
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_H_H_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Bend_H_H_Coefficients, Function_Dict, extra_args=extra_args_section)
                        elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'U':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_H_U_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Bend_H_U_Coefficients, Function_Dict, extra_args=extra_args_section)
                        elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'D':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_H_D_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, -1., Bend_H_D_Coefficients, Function_Dict, extra_args=extra_args_section)
                        elif bend_inlet == 'U':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_U_H_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Bend_U_H_Coefficients, Function_Dict, extra_args=extra_args_section)
                        elif bend_inlet == 'D':
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Bends(Bend_D_H_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, -1., Bend_D_H_Coefficients, Function_Dict, extra_args=extra_args_section)
                            
                    elif section_type == 'Splitter':
                        if Split_Pressure_Drop is True:
                            Pd = spdf.Section_Pressure_Drop.Dilute_Phase.Split(Split_Method, L_c, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g, m_s, R_g, T_g, mu_g, Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]], Split_Pressure_Drop_Coefficients, Function_Dict, extra_args=extra_args_section)
                        else:
                            Pd = 0.
                            
                        inlet_path = flow_path
                        outlet_path_1 = section[5]
                        outlet_path_2 = section[6]
                        
                        Flows.append((m_g * Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], m_s * Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]]))
                        Flows.append((m_g * (1. - Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]]), m_s * (1. - Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]])))
                        
                        Flows_Dictionary[outlet_path_1] = len(Flows) - 2
                        Flows_Dictionary[outlet_path_2] = len(Flows) - 1
                        
                        Path_Pressures[outlet_path_1].append(P_i - Pd - Pd_additional)
                        Path_Pressures[outlet_path_2].append(P_i - Pd - Pd_additional)
                    
                    Section_Pressures.append(P_i - Pd - Pd_additional)
                    if section_type != 'Splitter':
                        Path_Pressures[flow_path].append(P_i - Pd - Pd_additional)
                    
                    if Pd > 1000000.:
                        Penalty = True
                    elif Pd < 0.:
                        Penalty = True
                    
                    if Pd_additional > 1000000.:
                        Penalty = True
                    elif Pd_additional < 0.:
                        Penalty = True
                    
                else:
                    Section_Pressures.append(Section_Pressures[-1])
                    
                    if section_type == 'Splitter':
                        inlet_path = flow_path
                        outlet_path_1 = section[5]
                        outlet_path_2 = section[6]
                        
                        Flows.append((m_g * Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], m_s * Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]]))
                        Flows.append((m_g * (1. - Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]]), m_s * (1. - Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]])))
                        
                        Flows_Dictionary[outlet_path_1] = len(Flows) - 2
                        Flows_Dictionary[outlet_path_2] = len(Flows) - 1
                        
                        Path_Pressures[outlet_path_1].append(Section_Pressures[-1])
                        Path_Pressures[outlet_path_2].append(Section_Pressures[-1])
                    
                    if section_type != 'Splitter':
                        Path_Pressures[flow_path].append(Section_Pressures[-1])
                     
                try:
                    bend_inlet, bend_outlet = orientation.split('-')
                    if bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet in ['N', 'E', 'S', 'W']:
                        orientation = 'H-H'
                    elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'U':
                        orientation = 'H-U'
                    elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'D':
                        orientation = 'H-D'
                    elif bend_inlet == 'U':
                        orientation = 'U-H'
                    elif bend_inlet == 'D':
                        orientation = 'D-H'
                except:
                    pass
                    
                # Dump file data.
                with open('Section Pressure Dump.csv', 'a') as filet:
                    filet.write('{},{},{},{}\n'.format(now, section_type, orientation, Pd))
                        
            return Section_Pressures, Path_Pressures, Penalty
        
    class Dense_Phase:
        # This class calculates the pressure drop for a dense phase system.
        
        @staticmethod
        def System_Pressure_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict):
            
            # This code pulls the coefficients out of the coefficient input array.
            Horizontal_Pipe_Coefficients, Vertical_Upward_Pipe_Coefficients, Vertical_Downward_Pipe_Coefficients, Bend_H_H_Coefficients, Bend_H_U_Coefficients, Bend_H_D_Coefficients, Bend_U_H_Coefficients, Bend_D_H_Coefficients, Acceleration_Of_Solids_Coefficients = Coefficients
            
            # This code checks that the coefficient file has the correct number of coefficients for the system.
            if len(Horizontal_Pipe_Coefficients) != Coefficient_Dictionary['Dense Horizontal'][Horizontal_Pipe_Method]:
                print('Wrong number of coefficients for horizontal pipe sections')
                sys.exit()
            if len(Vertical_Upward_Pipe_Coefficients) != Coefficient_Dictionary['Dense Vertical'][Vertical_Upward_Pipe_Method]:
                print('Wrong number of coefficients for upward vertical pipe sections')
                sys.exit()
            if len(Vertical_Downward_Pipe_Coefficients) != Coefficient_Dictionary['Dense Vertical'][Vertical_Downward_Pipe_Method]:
                print('Wrong number of coefficients for downward vertical pipe sections')
                sys.exit()
            if len(Bend_H_H_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_H_H_Method]:
                print('Wrong number of coefficients for horizontal to horizontal bend sections')
                sys.exit()
            if len(Bend_H_U_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_H_U_Method]:
                print('Wrong number of coefficients for horizontal to upward vertical bend sections')
                sys.exit()
            if len(Bend_H_D_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_H_D_Method]:
                print('Wrong number of coefficients for horizontal to downward vertical bend sections')
                sys.exit()
            if len(Bend_U_H_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_U_H_Method]:
                print('Wrong number of coefficients for upward vertical to horizontal bend sections')
                sys.exit()
            if len(Bend_D_H_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_D_H_Method]:
                print('Wrong number of coefficients for downward vertical to horizontal bend sections')
                sys.exit()
            if len(Acceleration_Of_Solids_Coefficients) != Coefficient_Dictionary['Dense Acceleration'][Acceleration_Of_Solids_Method]:
                print('Wrong number of coefficients for acceleration of solids')
                sys.exit()
            
            # Create the initial guesses for system inlet pressure and flow split.
            System_Inlet_Pressure = Outlet_Pressures[0] * 1.01
            input_array = [System_Inlet_Pressure]
            flow_guesses_gas = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            flow_guesses_solids = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            for fg in flow_guesses_gas:
                input_array.append(fg)
            for fg in flow_guesses_solids:
                input_array.append(fg)
                
            # Run the root function to solve for flow split and system inlet pressures.
            args = (Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, System_Inlet_Pressure, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict)
            bounds = [[], []]
            for k in range(len(input_array)):
                if k == 0:
                    bounds[0].append(min(Outlet_Pressures))
                    bounds[1].append(np.inf)
                else:
                    bounds[0].append(0.)
                    bounds[1].append(1.)
            solution_array = least_squares(System_Pressure_Drop.Dense_Phase.Iteration_Evaluator_Pressure, input_array, args=args, bounds=bounds)
            
            # Pull out final inlet pressure and flow splits from the root function.
            Flow_Splits_Gas_Final = []
            Flow_Splits_Solids_Final = []
            for k in range(len(solution_array['x'])):
                if k == 0:
                    System_Inlet_Pressure_Final = solution_array['x'][k]
                elif k <= len(flow_guesses_gas):
                    Flow_Splits_Gas_Final.append(solution_array['x'][k])
                else:
                    Flow_Splits_Solids_Final.append(solution_array['x'][k])
            
            # Evaluate the solved system for section pressures.
            Section_Pressures, Path_Pressures, Penalty = System_Pressure_Drop.Dense_Phase.Pressure_Evaluator(System_Inlet_Pressure_Final, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict)
            
            Cost = solution_array['fun']
            
            return System_Inlet_Pressure_Final, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost
        
        @staticmethod
        def System_Solids_Flow_Solver(Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, System_Inlet_Pressure, Outlet_Pressures, Function_Dict):
            
            # This code pulls the coefficients out of the coefficient input array.
            Horizontal_Pipe_Coefficients, Vertical_Upward_Pipe_Coefficients, Vertical_Downward_Pipe_Coefficients, Bend_H_H_Coefficients, Bend_H_U_Coefficients, Bend_H_D_Coefficients, Bend_U_H_Coefficients, Bend_D_H_Coefficients, Acceleration_Of_Solids_Coefficients = Coefficients
            
            # This code checks that the coefficient file has the correct number of coefficients for the system.
            if len(Horizontal_Pipe_Coefficients) != Coefficient_Dictionary['Dense Horizontal'][Horizontal_Pipe_Method]:
                print('Wrong number of coefficients for horizontal pipe sections')
                sys.exit()
            if len(Vertical_Upward_Pipe_Coefficients) != Coefficient_Dictionary['Dense Vertical'][Vertical_Upward_Pipe_Method]:
                print('Wrong number of coefficients for upward vertical pipe sections')
                sys.exit()
            if len(Vertical_Downward_Pipe_Coefficients) != Coefficient_Dictionary['Dense Vertical'][Vertical_Downward_Pipe_Method]:
                print('Wrong number of coefficients for downward vertical pipe sections')
                sys.exit()
            if len(Bend_H_H_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_H_H_Method]:
                print('Wrong number of coefficients for horizontal to horizontal bend sections')
                sys.exit()
            if len(Bend_H_U_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_H_U_Method]:
                print('Wrong number of coefficients for horizontal to upward vertical bend sections')
                sys.exit()
            if len(Bend_H_D_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_H_D_Method]:
                print('Wrong number of coefficients for horizontal to downward vertical bend sections')
                sys.exit()
            if len(Bend_U_H_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_U_H_Method]:
                print('Wrong number of coefficients for upward vertical to horizontal bend sections')
                sys.exit()
            if len(Bend_D_H_Coefficients) != Coefficient_Dictionary['Dense Bends'][Bend_D_H_Method]:
                print('Wrong number of coefficients for downward vertical to horizontal bend sections')
                sys.exit()
            if len(Acceleration_Of_Solids_Coefficients) != Coefficient_Dictionary['Dense Acceleration'][Acceleration_Of_Solids_Method]:
                print('Wrong number of coefficients for acceleration of solids')
                sys.exit()
            
            # Create the initial guesses for solids flow and flow split.
            m_s = 0.1 # kg/s
            input_array = [m_s]
            flow_guesses_gas = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            flow_guesses_solids = [0.5 for s in Pipeline_Sections if s[2] == 'Splitter' ]
            for fg in flow_guesses_gas:
                input_array.append(fg)
            for fg in flow_guesses_solids:
                input_array.append(fg)
                
            # Run the root function to solve for flow split and system inlet pressures.
            args = (Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, System_Inlet_Pressure, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict)
            bounds = [[], []]
            for k in range(len(input_array)):
                if k == 0:
                    bounds[0].append(0.)
                    bounds[1].append(np.inf)
                else:
                    bounds[0].append(0.)
                    bounds[1].append(1.)
            solution_array = least_squares(System_Pressure_Drop.Dense_Phase.Iteration_Evaluator_Solids_Flow, input_array, args=args, bounds=bounds)
            
            # Pull out final inlet pressure and flow splits from the root function.
            Flow_Splits_Gas_Final = []
            Flow_Splits_Solids_Final = []
            for k in range(len(solution_array['x'])):
                if k == 0:
                    m_s_Final = solution_array['x'][k]
                elif k <= len(flow_guesses_gas):
                    Flow_Splits_Gas_Final.append(solution_array['x'][k])
                else:
                    Flow_Splits_Solids_Final.append(solution_array['x'][k])
            
            # Evaluate the solved system for section pressures.
            Section_Pressures, Path_Pressures, Penalty = System_Pressure_Drop.Dense_Phase.Pressure_Evaluator(System_Inlet_Pressure, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s_Final, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict)
            
            Cost = solution_array['fun']
            
            return m_s_Final, Outlet_Pressures, Flow_Splits_Gas_Final, Flow_Splits_Solids_Final, Cost
        
        @staticmethod
        def Iteration_Evaluator_Pressure(input_array, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, System_Inlet_Pressure, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict):
            # This function is used to minimize the difference between the desired output pressures and those calculated in order to find flow splits and system inlet pressure.
            
            # This section pulls the needed data out of the input array
            Flow_Splits_Gas = []
            Flow_Splits_Solids = []
            for k, i in zip(list(range(len(input_array))), input_array):
                if k == 0:
                    System_Inlet_Pressure = i
                elif k <= (len(input_array) - 1) / 2:
                    Flow_Splits_Gas.append(i)
                else:
                    Flow_Splits_Solids.append(i)
                    
            # This calculates the section pressures for the inputted inlet pressure and flow splits.
            try:
                Section_Pressures, Path_Pressures, Penalty  = System_Pressure_Drop.Dense_Phase.Pressure_Evaluator(System_Inlet_Pressure, Flow_Splits_Gas, Flow_Splits_Solids, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict)
                # if Penalty:
                    # return 1000000.
            except:
                return 1000000.
            
            # This section of code determines which sections are outlet sections.
            Flow_Paths = []
            [Flow_Paths.append(section[1]) for section in Pipeline_Sections if section[1] not in Flow_Paths]
            
            Max_Sections = {}
            for flow in Flow_Paths:
               Max_Sections[flow] = max([section[0] for section in Pipeline_Sections if section[1] == flow])
            
            for flow in Flow_Paths:
                if Pipeline_Sections[Max_Sections[flow]][2] ==  'Splitter':
                    del Max_Sections[flow]
            
            # This section of code finds the calculated outlet pressures.
            Calculated_Outlet_Pressures = []
            for k in list(Max_Sections.keys()):
                Calculated_Outlet_Pressures.append(Section_Pressures[Max_Sections[k] + 1])
            
            # This compares the calculated and given outlet pressures and returns the sum of the differences.
            Differences = []
            Total_Absolute_Difference = 0.
            for op, cop in zip(Outlet_Pressures, Calculated_Outlet_Pressures):
                if np.isnan(op - cop) or np.isinf(op - cop):
                    Differences.append(10000000.)
                else:
                    Differences.append(abs(op - cop))
                Total_Absolute_Difference += abs(op - cop)
            
            for k in range(len(input_array) - len(Differences)):
                Differences.append(0.)
            
            # Determine if the solution satisfies all function input conditions. If not, applies a penalty to the returned differences.
            try:
                Flow_Splits_Gas_Max = max(Flow_Splits_Gas)
                Flow_Splits_Gas_Min =  min(Flow_Splits_Gas)
                Flow_Splits_Solids_Max = max(Flow_Splits_Solids)
                Flow_Splits_Solids_Min =  min(Flow_Splits_Solids)
            except:
                Flow_Splits_Gas_Max = 0.5
                Flow_Splits_Gas_Min =  0.5
                Flow_Splits_Solids_Max = 0.5
                Flow_Splits_Solids_Min =  0.5
            
            penalty = 10000000.
            if System_Inlet_Pressure < max(Outlet_Pressures) or Flow_Splits_Gas_Max > 1. or Flow_Splits_Gas_Min < 0. or Flow_Splits_Solids_Max > 1. or Flow_Splits_Solids_Min < 0.:
                Differences = [abs(d) + penalty for d in Differences]
            
            if sum(Differences) > 1.E100:
                return 1.E100
                
            return sum(Differences)
        
        @staticmethod
        def Iteration_Evaluator_Solids_Flow(input_array, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, System_Inlet_Pressure, T_g, m_g, Coefficients, Coefficient_Dictionary, Outlet_Pressures, Function_Dict):
            # This function is used to minimize the difference between the desired output pressures and those calculated in order to find flow splits and system inlet pressure.
            
            # This section pulls the needed data out of the input array
            Flow_Splits_Gas = []
            Flow_Splits_Solids = []
            for k, i in zip(list(range(len(input_array))), input_array):
                if k == 0:
                    m_s = i
                elif k <= (len(input_array) - 1) / 2:
                    Flow_Splits_Gas.append(i)
                else:
                    Flow_Splits_Solids.append(i)
                
            # This calculates the section pressures for the inputted inlet pressure and flow splits.
            try:
                Section_Pressures, Path_Pressures, Penalty  = System_Pressure_Drop.Dense_Phase.Pressure_Evaluator(System_Inlet_Pressure, Flow_Splits_Gas, Flow_Splits_Solids, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict)
                # if Penalty:
                    # return 1000000.
            except:
                return 1000000.
            
            # This section of code determines which sections are outlet sections.
            Flow_Paths = []
            [Flow_Paths.append(section[1]) for section in Pipeline_Sections if section[1] not in Flow_Paths]
            
            Max_Sections = {}
            for flow in Flow_Paths:
               Max_Sections[flow] = max([section[0] for section in Pipeline_Sections if section[1] == flow])
            
            for flow in Flow_Paths:
                if Pipeline_Sections[Max_Sections[flow]][2] ==  'Splitter':
                    del Max_Sections[flow]
            
            # This section of code finds the calculated outlet pressures.
            Calculated_Outlet_Pressures = []
            for k in list(Max_Sections.keys()):
                Calculated_Outlet_Pressures.append(Section_Pressures[Max_Sections[k] + 1])
            
            # This compares the calculated and given outlet pressures and returns the sum of the differences.
            Differences = []
            Total_Absolute_Difference = 0.
            for op, cop in zip(Outlet_Pressures, Calculated_Outlet_Pressures):
                if np.isnan(op - cop) or np.isinf(op - cop):
                    Differences.append(10000000.)
                else:
                    Differences.append(abs(op - cop))
                Total_Absolute_Difference += abs(op - cop)
            
            for k in range(len(input_array) - len(Differences)):
                Differences.append(0.)
            
            # Determine if the solution satisfies all function input conditions. If not, applies a penalty to the returned differences.
            try:
                Flow_Splits_Gas_Max = max(Flow_Splits_Gas)
                Flow_Splits_Gas_Min =  min(Flow_Splits_Gas)
                Flow_Splits_Solids_Max = max(Flow_Splits_Solids)
                Flow_Splits_Solids_Min =  min(Flow_Splits_Solids)
            except:
                Flow_Splits_Gas_Max = 0.5
                Flow_Splits_Gas_Min =  0.5
                Flow_Splits_Solids_Max = 0.5
                Flow_Splits_Solids_Min =  0.5
            
            penalty = 10000000.
            if m_s < 0. or Flow_Splits_Gas_Max > 1. or Flow_Splits_Gas_Min < 0. or Flow_Splits_Solids_Max > 1. or Flow_Splits_Solids_Min < 0.:
                Differences = [d + penalty for d in Differences]
            
            return Differences
        
        @staticmethod
        def Pressure_Evaluator(System_Inlet_Pressure, Flow_Splits_Gas, Flow_Splits_Solids, Horizontal_Pipe_Method, Vertical_Upward_Pipe_Method, Vertical_Downward_Pipe_Method,  Bend_H_H_Method, Bend_H_U_Method, Bend_H_D_Method, Bend_U_H_Method, Bend_D_H_Method, Acceleration_Of_Solids_Method, g, epsilon, Pipeline_Sections, d, d_v50, rho_s, m_s, Gas_Type, T_g, m_g, Coefficients, Coefficient_Dictionary, Function_Dict):
            # This function calculates the pressure drop for a Dense phase system given system inlet pressure and flow splits.
            
            # This code pulls the coefficients out of the coefficient input array.
            Horizontal_Pipe_Coefficients, Vertical_Upward_Pipe_Coefficients, Vertical_Downward_Pipe_Coefficients, Bend_H_H_Coefficients, Bend_H_U_Coefficients, Bend_H_D_Coefficients, Bend_U_H_Coefficients, Bend_D_H_Coefficients, Acceleration_Of_Solids_Coefficients = Coefficients
                        
            # Import gas properties.
            if Gas_Type == 'Air':
                R_g = ATMOSPHERE_1976.R
            
            # This dictionary for different flows.
            Flow_Splits_Dictionary = {}
            count = 0
            for s in Pipeline_Sections:
                if s[2] == 'Splitter':
                    Flow_Splits_Dictionary[s[0]] = count
                    count += 1
            
            # This makes a list of the flow paths in the pipeline.
            Flow_Paths = list(range(1 + len([s for s in Pipeline_Sections if s[2] == 'Splitter' ]) * 2))
            Flows = [(m_g, m_s)]
            Flows_Dictionary = {0: 0}
            
            Path_Pressures = [[] for k in Flow_Paths]
                
            # This code finds pressure drop for each pipe section.         
            Section_Pressures = [System_Inlet_Pressure]
            Path_Pressures[0].append(System_Inlet_Pressure)
            
            Penalty = False
            for section in Pipeline_Sections:
                section_number = section[0]
                flow_path = section[1]
                section_type = section[2]
                orientation = section[3]
                L_c = section[4]
                D = section[5]
                
                path_flows = Flows[Flows_Dictionary[flow_path]]
                
                m_g = path_flows[0]
                m_s = path_flows[1]
                
                P_i = Path_Pressures[flow_path][-1]
                
                # This section of code calculates the inlet gas properties for the section.
                if P_i > 0.:
                    if Gas_Type == 'Air':
                        rho_g = ATMOSPHERE_1976.density(T_g, P_i)
                        mu_g = ATMOSPHERE_1976.viscosity(T_g)
                        
                    # This section of code calculates the pressure drop due to solids acceleration in the first section of the system.
                    if section_number == 0:
                        Pd_additional = spdf.Section_Pressure_Drop.Dense_Phase().Acceleration_Of_Solids(Acceleration_Of_Solids_Method, L_c, D, d, rho_g, rho_s, g, m_g, m_s, mu_g, Acceleration_Of_Solids_Coefficients, Function_Dict)
                    else:
                        Pd_additional = 0.
                        
                    if section_type == 'Horizontal':
                        Pd = spdf.Section_Pressure_Drop.Dense_Phase().Horizontal_Pipe_Sections(Horizontal_Pipe_Method, L_c, D, d, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, T_g, R_g, Horizontal_Pipe_Coefficients, Function_Dict)
                        
                    elif section_type == 'Vertical':
                        if orientation == 'Upward':
                            Pd = spdf.Section_Pressure_Drop.Dense_Phase().Vertical_Pipe_Sections(Vertical_Upward_Pipe_Method, L_c, D, d, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, T_g, R_g, 1., Vertical_Upward_Pipe_Coefficients, Function_Dict)
                            
                        elif orientation == 'Downward':
                            Pd = spdf.Section_Pressure_Drop.Dense_Phase().Vertical_Pipe_Sections(Vertical_Downward_Pipe_Method, L_c, D, d, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, T_g, R_g, -1., Vertical_Downward_Pipe_Coefficients, Function_Dict)
                            
                        else:
                            print('Invalid orientation for section {}'.format(section_number))
                            sys.exit()
                        
                    elif section_type == 'Bend':
                        bend_inlet, bend_outlet = orientation.split('-')
                        if bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet in ['N', 'E', 'S', 'W']:
                            Pd = spdf.Section_Pressure_Drop.Dense_Phase().Bends(Bend_H_H_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Bend_H_H_Coefficients, Function_Dict)
                        elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'U':
                            Pd = spdf.Section_Pressure_Drop.Dense_Phase().Bends(Bend_H_U_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Bend_H_U_Coefficients, Function_Dict)
                        elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'D':
                            Pd = spdf.Section_Pressure_Drop.Dense_Phase().Bends(Bend_H_D_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, -1., Bend_H_D_Coefficients, Function_Dict)
                        elif bend_inlet == 'U':
                            Pd = spdf.Section_Pressure_Drop.Dense_Phase().Bends(Bend_U_H_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, 1., Bend_U_H_Coefficients, Function_Dict)
                        elif bend_inlet == 'D':
                            Pd = spdf.Section_Pressure_Drop.Dense_Phase().Bends(Bend_D_H_Method, type, L_c, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, -1., Bend_D_H_Coefficients, Function_Dict)
                    elif section_type == 'Splitter':
                        Pd = 0.
                        inlet_path = flow_path
                        outlet_path_1 = section[5]
                        outlet_path_2 = section[6]
                        
                        Flows.append((m_g * Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], m_s * Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]]))
                        Flows.append((m_g * (1. - Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]]), m_s * (1. - Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]])))
                        
                        Flows_Dictionary[outlet_path_1] = len(Flows) - 2
                        Flows_Dictionary[outlet_path_2] = len(Flows) - 1
                        
                        Path_Pressures[outlet_path_1].append(P_i - Pd - Pd_additional)
                        Path_Pressures[outlet_path_2].append(P_i - Pd - Pd_additional)
                    
                    Section_Pressures.append(P_i - Pd - Pd_additional)
                    if section_type != 'Splitter':
                        Path_Pressures[flow_path].append(P_i - Pd - Pd_additional)
                        
                    if Pd > 1000000.:
                        Penalty = True
                        # print 'Error! Pd > 1000000 in Section {} {}: {}'.format(section_number, section_type, orientation)
                    elif Pd < 0.:
                        Penalty = True
                        # print 'Error! Pd < 0 in Section {} {}: {}'.format(section_number, section_type, orientation)
                    
                else:
                    Section_Pressures.append(Section_Pressures[-1])
                    
                    if section_type == 'Splitter':
                        inlet_path = flow_path
                        outlet_path_1 = section[5]
                        outlet_path_2 = section[6]
                        
                        Flows.append((m_g * Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]], m_s * Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]]))
                        Flows.append((m_g * (1. - Flow_Splits_Gas[Flow_Splits_Dictionary[section_number]]), m_s * (1. - Flow_Splits_Solids[Flow_Splits_Dictionary[section_number]])))
                        
                        Flows_Dictionary[outlet_path_1] = len(Flows) - 2
                        Flows_Dictionary[outlet_path_2] = len(Flows) - 1
                        
                        Path_Pressures[outlet_path_1].append(Section_Pressures[-1])
                        Path_Pressures[outlet_path_2].append(Section_Pressures[-1])
                    
                    if section_type != 'Splitter':
                        Path_Pressures[flow_path].append(Section_Pressures[-1])
            
            return Section_Pressures, Path_Pressures, Penalty