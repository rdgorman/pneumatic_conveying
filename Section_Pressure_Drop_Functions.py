# This code contains functions for determining the pressure drop different sections of dilute and dense phase systems.

import numpy as np

from numba import jit
from scipy.interpolate import CubicSpline
from scipy.optimize import root, minimize, brentq
from optimparallel import minimize_parallel

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

rat_x = [1., 1.5, 2., 3., 4., 6., 8., 10., 12., 14., 16., 20.]
kL_y = [20., 14., 12., 12., 14., 17., 24., 30., 34., 38., 42., 50.]
cs = CubicSpline(rat_x, kL_y, extrapolate=True)

class Section_Pressure_Drop:
    # This class contains the pressure drop calculation methods for both dilute and dense phase sections.
    
    global cs
    
    class Dimensionless_Numbers:
        # This class contains the calculation method for dimensionless numbers used in other methods.
        
        @staticmethod
        @jit(nopython=True)
        def Pipe_Reynolds_Number(rho_g, v_g, D, mu_g):
            # This function calculates the pipe Reynolds number.
            return rho_g * v_g * D / mu_g
        
        @staticmethod
        @jit(nopython=True)
        def Solids_Reynolds_Number(rho_g, v_r, d, mu_g):
            # This function calculates the solids Reynolds number.
            return rho_g * v_r * d / mu_g
        
        @staticmethod
        @jit(nopython=True)
        def Terminal_Reynolds_Number(rho_g, v_t, d, mu_g):
            # This function calculates the terminal Reynolds number.
            return rho_g * v_t * d / mu_g
        
        @staticmethod
        @jit(nopython=True)
        def Froude_Number(v_g, g, D):
            # This function calculates the Froude number.
            return v_g ** 2. / (g * D)
        
        @staticmethod
        @jit(nopython=True)
        def Particle_Froude_Number(v_g, v_s, g, d):
            # This function calculates the particle Froude number.
            return (v_g - v_s) ** 2. / (g * d)
        
        @staticmethod
        @jit(nopython=True)
        def Modified_Froude_Number(v_s, g, D):
            # This function calculates the modified Froude number.
            return v_s ** 2. / (g * D)
        
        @staticmethod
        @jit(nopython=True)
        def Froude_Number_Terminal(v_t, g, D):
            # This function calculates the Froude number.
            return v_t ** 2. / (g * D)
            
        @staticmethod
        @jit(nopython=True)
        def Bend_Outlet_Froude_Number(v_fao, g, D):
            # This function calculates the Froude number.
            return v_fao ** 2. / (g * D)
        
        @staticmethod
        @jit(nopython=True)
        def Slug_Froude_Number(v_sl, g, D):
            # This function calculates the Froude number.
            return v_sl ** 2. / (g * D)
        
        @staticmethod
        def Drag_Coefficient(rho_g, v_g, D, mu_g):
            # This function calculates the drag coefficient.
            Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
            return 24./Re_p * (1. + (3./16. * Re_p)/(1. + (19. /240. * Re_p)/(1. + 1. / 122. * Re_p))) + 8. / 1000. * (np.log((1. + Re_p) / 21000.) * (np.log((1. + Re_p) / 254.)) ** 2.) / (1. + Re_p / 7000.)
        
        @staticmethod
        @jit(nopython=True)
        def Solids_Loading_Ratio(m_g, m_s):
            # This function calculates the solids loading ratio.
            return m_s / m_g
            
        @staticmethod
        @jit(nopython=True)
        def Archimedes_Number(rho_g, rho_s, d, mu_g, g):
            #This function calculates the Archimedes number.
            return rho_g * (rho_s - rho_g) * g * d ** 3. / mu_g ** 2.
        
        @staticmethod
        @jit(nopython=True)
        def Length_Ratio(L_c, D):
            #This function calculates the length ratio.
            return L_c / D
            
        @staticmethod
        @jit(nopython=True)
        def Diameter_Ratio(d, D):
            #This function calculates the diameter ratio.
            return d / D
        
        @staticmethod
        @jit(nopython=True)
        def Median_Diameter_Ratio(d_v50, D):
            #This function calculates the median diameter ratio.
            return d_v50 / D
            
        @staticmethod
        @jit(nopython=True)
        def Density_Ratio(rho_g, rho_s):
            #This function calculates the density ratio.
            return rho_g / rho_s
            
        @staticmethod
        @jit(nopython=True)
        def Roughness_Ratio(epsilon, D):
            #This function calculates the roughness ratio.
            return epsilon / D
            
        @staticmethod
        @jit(nopython=True)
        def Pressure_Ratio(v_g, rho_g, R_g, T_g):
            #This function calculates the pressure ratio.
            
            # This section calculates the ideal gas pressure.
            P_I = rho_g * R_g * T_g
            
            # This section calculates the dynamic pressure.
            P_d = 0.5 * rho_g *v_g ** 2.
            
            return P_I / P_d
    
    class Dilute_Phase:
        # This class contains the pressure drop calculation methods for dilute phase systems.
        
        class Iteration_Functions:
            # This class contains functions that need to be solved by iteration.
            @staticmethod
            @jit(nopython=True)
            def f_g_fun(f_g, Re, epsilon, D):
                # Function for calculating friction factor.
                fun = -2. * np.log10(epsilon / (3.7 * D) + 2.51 / (Re * (f_g) ** 0.5)) - 1. / (f_g) ** 0.5
                for f in f_g:
                    if f < 0.:
                        fun = np.array([0.])
                return fun
                
            @staticmethod
            def v_s_fun(v_s, v_g, v_t, D, g, rho_s, m_s, A_c):
                # Function for calculating solids velocity.
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                try:
                    for a in alpha:
                        if a < 0.:
                            fun = np.array([1000.])
                        else:
                            f_p_star = 0.117 * ((1. - a) * v_g / (g * D) ** 0.5 ) ** -1.15 * (1. - a) / a ** 3.
                            
                            interior = f_p_star * v_s ** 2. / (2. * g * D) * a ** 4.7
                            
                            if interior < 0.:
                                fun = np.array([1000.])
                            else:
                                fun = v_g - v_t * (interior) ** 0.5 - v_s
                except:
                    if alpha < 0.:
                        fun = np.array([1000.])
                    else:
                        f_p_star = 0.117 * ((1. - alpha) * v_g / (g * D) ** 0.5 ) ** -1.15 * (1. - alpha) / alpha ** 3.
                        
                        interior = f_p_star * v_s ** 2. / (2. * g * D) * alpha ** 4.7
                        
                        if interior < 0.:
                            fun = np.array([1000.])
                        else:
                            
                            fun = v_g - v_t * (interior) ** 0.5 - v_s
                            
                if v_g < 0.:
                    fun = np.array([1000.])
                return fun
                
            @staticmethod
            @jit(nopython=True)
            def v_fao_singh_fun(v_fao, a_sw, m_s, D, R_B, rho_g, A_c, R_g, T_g, m_g):
                # Function for calculating outlet gas velocity.
                P_B_1 = 0.13 + a_sw  * m_s * v_fao / D ** 2. * (R_B / D) ** -0.18
                P_B_2 = (rho_g - m_g / (v_fao * A_c)) * R_g * T_g
                
                return np.absolute(P_B_1 - P_B_2)
                
            @staticmethod
            def v_fao_rossetti_fun(v_fao, g, D, m_g, R_B, Re_p, rho_g, A_c, R_g, T_g, R_s):
                # Function for calculating outlet gas velocity.
                Fr_o = Section_Pressure_Drop.Dimensionless_Numbers.Bend_Outlet_Froude_Number(v_fao, g, D)
                rho_go = m_g / (v_fao * A_c)
            
                f_B_g = 0.167 * (1. + 17.062 * (2. * R_B / D) ** -1.219) * Re_p ** -0.17 * (2. * R_B / D) ** 0.84
                f_B_s = (5.4 * R_s ** 1.293) / (Fr_o ** 0.84 * (2. * R_B / D) ** 0.39)
            
                P_B_1 = (f_B_g + f_B_s) * rho_go * v_fao ** 2. / 2.
                P_B_2 = (rho_g - m_g / (v_fao * A_c)) * R_g * T_g
                
                return np.absolute(P_B_1 - P_B_2)
                
            @staticmethod
            @jit(nopython=True)
            def v_fao_de_moraes_fun(v_fao, g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g):
                # Function for calculating outlet gas velocity.
                rho_go = m_g / (v_fao * A_c)
                
                f_B_g = 0.167 * (1. + 17.062 * (2. * R_B / D) ** -1.219) * Re_p ** -0.17 * (2. * R_B / D) ** 0.84
                f_B_s = a_dM * np.exp(b_dM * v_fao)
            
                P_B_1 = (f_B_g + f_B_s) * rho_go * v_fao ** 2. / 2.
                P_B_2 = (rho_g - m_g / (v_fao * A_c)) * R_g * T_g
                
                return np.absolute(P_B_1 - P_B_2)
                
            @staticmethod
            def v_fao_chambers_fun(v_fao, m_g, A_c, B_L, R_s, rho_g, R_g, T_g):
                # Function for calculating outlet gas velocity.
                rho_go = m_g / (v_fao * A_c)
                    
                P_B_1 = B_L * (1. + R_s) * (rho_go * v_fao ** 2.) / 2.
                P_B_2 = (rho_g - m_g / (v_fao * A_c)) * R_g * T_g
                
                return P_B_1 - P_B_2
                
            @staticmethod
            def v_fao_pan_fun(v_fao, g, D, m_g, rho_g, A_c, R_s, R_g, T_g):
                # Function for calculating outlet gas velocity.
                Fr_o = Section_Pressure_Drop.Dimensionless_Numbers.Bend_Outlet_Froude_Number(v_fao, g, D)
                rho_go = m_g / (v_fao * A_c)
                
                P_B_1 = 0.005 * R_s ** 1.49 * Fr_o **1.1182 * rho_go * v_fao ** 2. / 2.
                P_B_2 = (rho_g - m_g / (v_fao * A_c)) * R_g * T_g
                
                return P_B_1 - P_B_2
                
            @staticmethod
            @jit(nopython=True)
            def v_fao_das_fun(v_fao, a_D, b_D, R_s, rho_g, m_g, A_c, R_g, T_g):
                # Function for calculating outlet gas velocity.                
                P_B_1 = a_D * R_s * v_fao ** b_D
                P_B_2 = (rho_g - m_g / (v_fao * A_c)) * R_g * T_g
                
                return P_B_1 - P_B_2
                
        @staticmethod
        def Horizontal_Pipe_Sections(method, L, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, coefficients, Function_Dict, extra_args=None):
            # This class contains the pressure drop calculation methods for horizontal pipe sections.
            
            if method == 1:
                # This is the calculation code for Method 1 - Total pressure: Chand (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_st = Section_Pressure_Drop.Dimensionless_Numbers.Particle_Froude_Number(v_g, v_s, g, d)
                Fr_st_st = Section_Pressure_Drop.Dimensionless_Numbers.Modified_Froude_Number(v_s, g, D)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_g * 3. / 8. * C_D / epsilon * rho_g / rho_s * m_s / m_g * Re_p ** 0.25 * Fr ** -0.5 * Fr_st * Fr_st_st ** -0.5
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 2:
                # This is the calculation code for Method 2 - Total pressure: Chand (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_st = Section_Pressure_Drop.Dimensionless_Numbers.Particle_Froude_Number(v_g, v_s, g, d)
                Fr_st_st = Section_Pressure_Drop.Dimensionless_Numbers.Modified_Froude_Number(v_s, g, D)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                P_s = P_g * 3. / 8. * C_D / epsilon * rho_g / rho_s * m_s / m_g * Re_p ** 0.25 * Fr ** -0.5 * Fr_st * Fr_st_st ** -0.5
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 3:
                # This is the calculation code for Method 3 - Total pressure: Chand (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_st = Section_Pressure_Drop.Dimensionless_Numbers.Particle_Froude_Number(v_g, v_s, g, d)
                Fr_st_st = Section_Pressure_Drop.Dimensionless_Numbers.Modified_Froude_Number(v_s, g, D)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                P_s = P_g * 3. / 8. * C_D / epsilon * rho_g / rho_s * m_s / m_g * Re_p ** 0.25 * Fr ** -0.5 * Fr_st * Fr_st_st ** -0.5
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                            
            elif method == 4:
                # This is the calculation code for Method 4 - Total pressure: Chand (Solids velocity: Yang):
                                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                        v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    # v_s = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_st = Section_Pressure_Drop.Dimensionless_Numbers.Particle_Froude_Number(v_g, v_s, g, d)
                Fr_st_st = Section_Pressure_Drop.Dimensionless_Numbers.Modified_Froude_Number(v_s, g, D)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                P_s = P_g * 3. / 8. * C_D / epsilon * rho_g / rho_s * m_s / m_g * Re_p ** 0.25 * Fr ** -0.5 * Fr_st * Fr_st_st ** -0.5
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 5:
                # This is the calculation code for Method 5 - Total pressure: Chand (Solids velocity: Stevanovic et al.):
                                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_st = Section_Pressure_Drop.Dimensionless_Numbers.Particle_Froude_Number(v_g, v_s, g, d)
                Fr_st_st = Section_Pressure_Drop.Dimensionless_Numbers.Modified_Froude_Number(v_s, g, D)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                P_s = P_g * 3. / 8. * C_D / epsilon * rho_g / rho_s * m_s / m_g * Re_p ** 0.25 * Fr ** -0.5 * Fr_st * Fr_st_st ** -0.5
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 6:
                # This is the calculation code for Method 6 - Total pressure: Hitchcock and Jones:
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = 0.012 * R_s ** -0.1 * 1. / Fr ** 0.5 * (d / D) ** -0.9
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 7:
                # This is the calculation code for Method 7 - Total pressure: Barth:
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section of code calculates the terminal velocity of a particle.
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = 0.005 * (1. - Fr ** -1.) / (1. + 0.00125 * Fr_t ** 2.)
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 8:
                # This is the calculation code for Method 8 - Total pressure: Michaelides:
                
                # This section of code extracts the coefficients used in this method.
                K_p = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = K_p * 1. / Fr ** 0.5
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 9:
                # This is the calculation code for Method 9 - Total pressure: Wypych:
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section of code calculates the terminal velocity of a particle.
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56                    
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = (9. * 10. ** 6. * (Fr / rho_g) ** -1.782) / (R_s ** 0.8 * (Fr_t ** 0.5) ** 0.6 * (L / D) ** 0.8 * (D / d_v50) ** 0.85)
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 10:
                # This is the calculation code for Method 10 - Total pressure: Setia et al.:
                
                # This section of code extracts the coefficients used in this method.
                a_s = coefficients[0]
                b_s = coefficients[0]
                c_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = a_s * R_s ** b_s * Fr ** c_s
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 11:
                # This is the calculation code for Method 11 - Total pressure: Tripathi et al.:
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = 20.807 * R_s ** -0.71 * Fr ** -1.05655
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 12:
                # This is the calculation code for Method 12 - Total pressure: Naveh et al.:
                
                # This section of code extracts the coefficients used in this method.
                v_mp = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.                  
                
                if Ar <= 4. * 10. ** 5.:
                    a_n = 0.0216 * Ar ** -0.2467
                else:
                    a_n = 114.9 * Ar ** -0.9126
                
                f_p = 2. * D * rho_g * g * d / mu_g * 1. / v_mp * a_n
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 13:
                # This is the calculation code for Method 13 - Total pressure: Mason et al.:
                
                # This section of code extracts the coefficients used in this method.
                f_pM = coefficients[0]
                a_M = coefficients[1]
                b_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                P_s = f_pM * L * R_s ** a_M * Fr ** b_M
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 14:
                # This is the calculation code for Method 14 - Total pressure: Mehta et al. (Solids velocity: Naveh et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 15:
                # This is the calculation code for Method 15 - Total pressure: Mehta et al. (Solids velocity: Klinzing et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 16:
                # This is the calculation code for Method 16 - Total pressure: Mehta et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                f_me = coefficients[2]
                a_me = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 17:
                # This is the calculation code for Method 17 - Total pressure: Mehta et al. (Solids velocity: Yang):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 18:
                # This is the calculation code for Method 18 - Total pressure: Mehta et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 19:
                # This is the calculation code for Method 19 - Two phase: Ozbelge:
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section of code calculates the terminal velocity of a particle.
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56           
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_s = g * D / (2. * v_s ** 2.) * (alpha ** -4.7 * (v_r / v_t) ** 2. - 1.)
                
                P_s = 2. * f_s * L * v_s * rho_ds * A_c / D
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 20:
                # This is the calculation code for Method 20 - Combined: Hinkle (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 3. * rho_g * C_D * D / (2. * rho_s * d) * (v_r / v_s) ** 2.
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 21:
                # This is the calculation code for Method 21 - Combined: Hinkle (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 3. * rho_g * C_D * D / (2. * rho_s * d) * (v_r / v_s) ** 2.
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 22:
                # This is the calculation code for Method 22 - Combined: Hinkle (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 3. * rho_g * C_D * D / (2. * rho_s * d) * (v_r / v_s) ** 2.
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 23:
                # This is the calculation code for Method 23 - Combined: Hinkle (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 3. * rho_g * C_D * D / (2. * rho_s * d) * (v_r / v_s) ** 2.
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 24:
                # This is the calculation code for Method 24 - Combined: Hinkle (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 3. * rho_g * C_D * D / (2. * rho_s * d) * (v_r / v_s) ** 2.
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 25:
                # This is the calculation code for Method 25 - Combined: Konno and Saito (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)\
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.114 * (g * D) ** 0.5 / v_s
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 26:
                # This is the calculation code for Method 26 - Combined: Konno and Saito (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.114 * (g * D) ** 0.5 / v_s
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 27:
                # This is the calculation code for Method 27 - Combined: Konno and Saito (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.114 * (g * D) ** 0.5 / v_s
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 28:
                # This is the calculation code for Method 28 - Combined: Konno and Saito (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.114 * (g * D) ** 0.5 / v_s
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 29:
                # This is the calculation code for Method 29 - Combined: Konno and Saito (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.114 * (g * D) ** 0.5 / v_s
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 30:
                # This is the calculation code for Method 30 - Combined: Yang (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.117 / (alpha ** 3. * (1. - alpha) ** 0.15) * (v_t * v_g / (v_r * (g * D) ** 0.5)) ** -1.15
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 31:
                # This is the calculation code for Method 31 - Combined: Yang (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                try:
                    f_p = 0.117 / (alpha ** 3. * (1. - alpha) ** 0.15) * (v_t * v_g / (v_r * (g * D) ** 0.5)) ** -1.15
                except:
                    f_p = 0.                    
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 32:
                # This is the calculation code for Method 32 - Combined: Yang (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                try:
                    f_p = 0.117 / (alpha ** 3. * (1. - alpha) ** 0.15) * (v_t * v_g / (v_r * (g * D) ** 0.5)) ** -1.15
                except:
                    f_p = 0.
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 33:
                # This is the calculation code for Method 33 - Combined: Yang (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.117 / (alpha ** 3. * (1. - alpha) ** 0.15) * (v_t * v_g / (v_r * (g * D) ** 0.5)) ** -1.15
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 34:
                # This is the calculation code for Method 34 - Combined: Yang (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = 0.001
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.117 / (alpha ** 3. * (1. - alpha) ** 0.15) * (v_t * v_g / (v_r * (g * D) ** 0.5)) ** -1.15
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 35:
                # This is the calculation code for Method 35 - Combined: Weber (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.82 * R_s ** -0.3 * Fr ** -0.43 * Fr_t ** 0.125 * (D / d) ** 0.1
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 36:
                # This is the calculation code for Method 36 - Combined: Weber (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.82 * R_s ** -0.3 * Fr ** -0.43 * Fr_t ** 0.125 * (D / d) ** 0.1
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 37:
                # This is the calculation code for Method 37 - Combined: Weber (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.82 * R_s ** -0.3 * Fr ** -0.43 * Fr_t ** 0.125 * (D / d) ** 0.1
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 38:
                # This is the calculation code for Method 38 - Combined: Weber (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.82 * R_s ** -0.3 * Fr ** -0.43 * Fr_t ** 0.125 * (D / d) ** 0.1
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 39:
                # This is the calculation code for Method 39 - Combined: Weber (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 0.82 * R_s ** -0.3 * Fr ** -0.43 * Fr_t ** 0.125 * (D / d) ** 0.1
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 40:
                # This is the calculation code for Method 40 - Combined: Wei et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 1.98 / (alpha * (1. - alpha) ** 0.057) * (v_t / v_r) ** -0.902 * (v_g / (g * D) ** 0.5) ** -1.95
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 41:
                # This is the calculation code for Method 41 - Combined: Wei et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                try:
                    f_p = 1.98 / (alpha * (1. - alpha) ** 0.057) * (v_t / v_r) ** -0.902 * (v_g / (g * D) ** 0.5) ** -1.95
                except:
                    f_p = 0.
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 42:
                # This is the calculation code for Method 42 - Combined: Wei et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                try:
                    f_p = 1.98 / (alpha * (1. - alpha) ** 0.057) * (v_t / v_r) ** -0.902 * (v_g / (g * D) ** 0.5) ** -1.95
                except:
                    f_p = 0.
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 43:
                # This is the calculation code for Method 43 - Combined: Wei et al. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 1.98 / (alpha * (1. - alpha) ** 0.057) * (v_t / v_r) ** -0.902 * (v_g / (g * D) ** 0.5) ** -1.95
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 44:
                # This is the calculation code for Method 44 - Combined: Wei et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = 0.001
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = 0.
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = 1.98 / (alpha * (1. - alpha) ** 0.057) * (v_t / v_r) ** -0.902 * (v_g / (g * D) ** 0.5) ** -1.95
                
                P_s_s = 0.
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 45:
                # This is the calculation code for Method 45 - Force balance: Hinkle (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. / 2. * D / d * v_r ** 2. / v_s ** 2. * C_D
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 46:
                # This is the calculation code for Method 46 - Force balance: Hinkle (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. / 2. * D / d * v_r ** 2. / v_s ** 2. * C_D
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 47:
                # This is the calculation code for Method 47 - Force balance: Hinkle (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. / 2. * D / d * v_r ** 2. / v_s ** 2. * C_D
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 48:
                # This is the calculation code for Method 48 - Force balance: Hinkle (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. / 2. * D / d * v_r ** 2. / v_s ** 2. * C_D
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 49:
                # This is the calculation code for Method 49 - Force balance: Hinkle (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. / 2. * D / d * v_r ** 2. / v_s ** 2. * C_D
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 50:
                # This is the calculation code for Method 50 - Force balance: Yang (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 0.117 * ((1. - alpha) * v_g / (g * D) ** 0.5 ) ** -1.15 * (1. - alpha) / alpha ** 3.
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 51:
                # This is the calculation code for Method 51 - Force balance: Yang (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                try:
                    f_p_star = 0.117 * ((1. - alpha) * v_g / (g * D) ** 0.5 ) ** -1.15 * (1. - alpha) / alpha ** 3.
                except:
                    f_p_star = 0.
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 52:
                # This is the calculation code for Method 52 - Force balance: Yang (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                try:
                    f_p_star = 0.117 * ((1. - alpha) * v_g / (g * D) ** 0.5 ) ** -1.15 * (1. - alpha) / alpha ** 3.
                except:
                    f_p_star = 0.
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 53:
                # This is the calculation code for Method 53 - Force balance: Yang (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 0.117 * ((1. - alpha) * v_g / (g * D) ** 0.5 ) ** -1.15 * (1. - alpha) / alpha ** 3.
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 54:
                # This is the calculation code for Method 54 - Force balance: Yang (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 0.117 * ((1. - alpha) * v_g / (g * D) ** 0.5 ) ** -1.15 * (1. - alpha) / alpha ** 3.
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 55:
                # This is the calculation code for Method 55 - Force balance: Klinzing et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. * D / d * rho_g / (rho_s - rho_g) * (v_g - v_s) ** 2. / v_s ** 2. * C_D * alpha ** -4.7
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 56:
                # This is the calculation code for Method 56 - Force balance: Klinzing et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. * D / d * rho_g / (rho_s - rho_g) * (v_g - v_s) ** 2. / v_s ** 2. * C_D * alpha ** -4.7
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 57:
                # This is the calculation code for Method 57 - Force balance: Klinzing et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. * D / d * rho_g / (rho_s - rho_g) * (v_g - v_s) ** 2. / v_s ** 2. * C_D * alpha ** -4.7
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 58:
                # This is the calculation code for Method 58 - Force balance: Klinzing et al. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. * D / d * rho_g / (rho_s - rho_g) * (v_g - v_s) ** 2. / v_s ** 2. * C_D * alpha ** -4.7
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 59:
                # This is the calculation code for Method 59 - Force balance: Klinzing et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                C_D = Section_Pressure_Drop.Dimensionless_Numbers.Drag_Coefficient(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 3. * D / d * rho_g / (rho_s - rho_g) * (v_g - v_s) ** 2. / v_s ** 2. * C_D * alpha ** -4.7
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 60:
                # This is the calculation code for Method 60 - Force balance: Raheman and Jindal (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                Re_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Reynolds_Number(rho_g, v_r, d, mu_g)
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 1.4 * v_t / v_s * Fr_t * -0.95 * (d / D) ** 0.7 + 1.32 * 10. ** -3. * R_s ** 1.5 + 2.12 * Fr ** -0.7 + 1.36 * 10. ** -6. * Re_s ** 0.9 - 0.024
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 61:
                # This is the calculation code for Method 61 - Force balance: Raheman and Jindal (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                Re_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Reynolds_Number(rho_g, v_r, d, mu_g)
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 1.4 * v_t / v_s * Fr_t * -0.95 * (d / D) ** 0.7 + 1.32 * 10. ** -3. * R_s ** 1.5 + 2.12 * Fr ** -0.7 + 1.36 * 10. ** -6. * Re_s ** 0.9 - 0.024
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 62:
                # This is the calculation code for Method 62 - Force balance: Raheman and Jindal (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                Re_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Reynolds_Number(rho_g, v_r, d, mu_g)
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 1.4 * v_t / v_s * Fr_t * -0.95 * (d / D) ** 0.7 + 1.32 * 10. ** -3. * R_s ** 1.5 + 2.12 * Fr ** -0.7 + 1.36 * 10. ** -6. * Re_s ** 0.9 - 0.024
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 63:
                # This is the calculation code for Method 63 - Force balance: Raheman and Jindal (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                Re_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Reynolds_Number(rho_g, v_r, d, mu_g)
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 1.4 * v_t / v_s * Fr_t * -0.95 * (d / D) ** 0.7 + 1.32 * 10. ** -3. * R_s ** 1.5 + 2.12 * Fr ** -0.7 + 1.36 * 10. ** -6. * Re_s ** 0.9 - 0.024
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 64:
                # This is the calculation code for Method 64 - Force balance: Raheman and Jindal (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                Fr_t = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number_Terminal(v_t, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = 0.001
                Re_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Reynolds_Number(rho_g, v_r, d, mu_g)
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 1.4 * v_t / v_s * Fr_t * -0.95 * (d / D) ** 0.7 + 1.32 * 10. ** -3. * R_s ** 1.5 + 2.12 * Fr ** -0.7 + 1.36 * 10. ** -6. * Re_s ** 0.9 - 0.024
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 65:
                # This is the calculation code for Method 65 - Force balance: Wei et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Re_t = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_t, d, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 1.98 * ((1. - alpha) ** -0.057) / alpha ** 3. * (Re_t / Re_p) ** -0.902 * ((Fr) ** 0.5) ** -1.95
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 66:
                # This is the calculation code for Method 66 - Force balance: Wei et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Re_t = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_t, d, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                try:
                    f_p_star = 1.98 * ((1. - alpha) ** -0.057) / alpha ** 3. * (Re_t / Re_p) ** -0.902 * ((Fr) ** 0.5) ** -1.95
                except:
                    f_p_star = 0.
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 67:
                # This is the calculation code for Method 67 - Force balance: Wei et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Re_t = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_t, d, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                try:
                    f_p_star = 1.98 * ((1. - alpha) ** -0.057) / alpha ** 3. * (Re_t / Re_p) ** -0.902 * ((Fr) ** 0.5) ** -1.95
                except:
                    f_p_star = 0.
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 68:
                # This is the calculation code for Method 68 - Force balance: Wei et al. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Re_t = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_t, d, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 1.98 * ((1. - alpha) ** -0.057) / alpha ** 3. * (Re_t / Re_p) ** -0.902 * ((Fr) ** 0.5) ** -1.95
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 69:
                # This is the calculation code for Method 69 - Force balance: Wei et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Re_t = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_t, d, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates surface area.
                S = np.pi * D * L
                
                # This section calculates the impact and friction factor.
                f_p_star = 1.98 * ((1. - alpha) ** -0.057) / alpha ** 3. * (Re_t / Re_p) ** -0.902 * ((Fr) ** 0.5) ** -1.95
                
                # This section calculates the solids pressure drop factor.
                f_p = f_p_star * v_s / v_g
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.                
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = f_p / 4. * ((1. - alpha) * rho_s * A_c) / 2. * (S * v_s) / v_g * v_s ** 2.
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 70:
                # This is the calculation code for Method 70 - Miscellaneous: Pfeffer et. al. 1. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + 4. * Re_p ** -0.32 * R_s)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 71:
                # This is the calculation code for Method 70 - Miscellaneous: Pfeffer et. al. 1. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + 4. * Re_p ** -0.32 * R_s)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 72:
                # This is the calculation code for Method 72 - Miscellaneous: Pfeffer et. al. 1. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + 4. * Re_p ** -0.32 * R_s)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 73:
                # This is the calculation code for Method 73 - Miscellaneous: Pfeffer et. al. 1. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + 4. * Re_p ** -0.32 * R_s)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 74:
                # This is the calculation code for Method 74 - Miscellaneous: Pfeffer et. al. 1. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + 4. * Re_p ** -0.32 * R_s)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 75:
                # This is the calculation code for Method 75 - Miscellaneous: Pfeffer et. al. 2. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** 0.3
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 76:
                # This is the calculation code for Method 76 - Miscellaneous: Pfeffer et al. 2. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** 0.3
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 77:
                # This is the calculation code for Method 77 - Miscellaneous: Pfeffer et al. 2. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** 0.3
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 78:
                # This is the calculation code for Method 78 - Miscellaneous: Pfeffer et al. 2. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** 0.3
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 79:
                # This is the calculation code for Method 79 - Miscellaneous: Pfeffer et al. 2. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** 0.3
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 80:
                # This is the calculation code for Method 80 - Miscellaneous: Pfeffer et al. generalized (Solids velocity: Naveh et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            elif method == 81:
                # This is the calculation code for Method 81 - Miscellaneous: Pfeffer et al. generalized (Solids velocity: Klinzing et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 82:
                # This is the calculation code for Method 82 - Miscellaneous: Pfeffer et al. generalized:
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                n_e = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 83:
                # This is the calculation code for Method 83 - Miscellaneous: Pfeffer et al. generalized (Solids velocity: Yang):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 84:
                # This is the calculation code for Method 84 - Miscellaneous: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            elif method == 85:
                # This is the calculation code for Method 85 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi = coefficients[0]
                B_pi_1 = coefficients[1]
                B_pi_2 = coefficients[2]
                B_pi_3 = coefficients[3]
                B_pi_4 = coefficients[4]
                B_pi_5 = coefficients[5]
                B_pi_6 = coefficients[6]
                B_pi_7 = coefficients[7]
                B_pi_8 = coefficients[8]
                B_pi_9 = coefficients[9]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_H = A_pi * (m_g ** 2.) / (D ** 4. * rho_g) * L_R ** B_pi_1 * D_R ** B_pi_2 * D_R50 ** B_pi_3 * rho_R ** B_pi_4 * epsilon_R ** B_pi_5 * Re_p ** B_pi_6 * Fr ** B_pi_7 * R_s ** B_pi_8 * P_R ** B_pi_9
                
            elif method == 86:
                # This is the calculation code for Method 86 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                B_pi_1 = coefficients[9]
                B_pi_2 = coefficients[10]
                B_pi_3 = coefficients[11]
                B_pi_4 = coefficients[12]
                B_pi_5 = coefficients[13]
                B_pi_6 = coefficients[14]
                B_pi_7 = coefficients[15]
                B_pi_8 = coefficients[16]
                B_pi_9 = coefficients[17]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_H = (m_g ** 2.) / (D ** 4. * rho_g) * (A_pi_1 * L_R ** B_pi_1 + A_pi_2 * D_R ** B_pi_2 + A_pi_3 * D_R50 ** B_pi_3 + A_pi_4 * rho_R ** B_pi_4 + A_pi_5 * epsilon_R ** B_pi_5 + A_pi_6 * Re_p ** B_pi_6 + A_pi_7 * Fr ** B_pi_7 + A_pi_8 * R_s ** B_pi_8 + A_pi_9 * P_R ** B_pi_9)
                
            elif method == 87:
                # This is the calculation code for Method 87 - Buckingham-Pi Theorem:
                                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                B_pi_1 = coefficients[9]
                B_pi_2 = coefficients[10]
                B_pi_3 = coefficients[11]
                B_pi_4 = coefficients[12]
                B_pi_5 = coefficients[13]
                B_pi_6 = coefficients[14]
                B_pi_7 = coefficients[15]
                B_pi_8 = coefficients[16]
                B_pi_9 = coefficients[17]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code uses the coefficients to find the function inputs for this method.
                L_R_C = A_pi_1 * L_R ** B_pi_1
                D_R_C = A_pi_2 * D_R ** B_pi_2
                D_R50_C = A_pi_3 * D_R50 ** B_pi_3
                rho_R_C = A_pi_4 * rho_R ** B_pi_4
                epsilon_R_C = A_pi_5 * epsilon_R ** B_pi_5
                Re_p_C = A_pi_6 * Re_p ** B_pi_6
                Fr_C = A_pi_7 * Fr ** B_pi_7
                R_s_C = A_pi_8 * R_s ** B_pi_8
                P_R_C = A_pi_9 * P_R ** B_pi_9
                
                # Pull function out of extra args
                try:
                    [func, orientation] = extra_args
                except:
                    [func, orientation, L_t] = extra_args
                
                # Set zero functions based on section type.
                Z_HP = 1.
                Z_VUP = 0.
                Z_VDP = 0.
                Z_BHH = 0.
                Z_BHU = 0.
                Z_BHD = 0.
                Z_BUH = 0.
                Z_BDH = 0.
                Z_AS = 0.
                
                # This section of code calculates the dimensionless pressure drop based on the dimensionless numbers and the passed function.
                # P_H_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C, Z_HP, Z_VUP, Z_VDP, Z_BHH, Z_BHU, Z_BHD, Z_BUH, Z_BDH, Z_AS)
                P_H_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C)
                
                # Convert dimensonless pressure drop to pressure drop.
                P_H = (m_g ** 2.) / (D ** 4. * rho_g) * P_H_star
                
            elif method == 88:
                # This is the calculation code for Method 88 - Miscellaneous: Pfeffer et al. generalized with additional nondimensional terms (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                A_pi = coefficients[1]
                B_pi_1 = coefficients[2]
                # B_pi_2 = coefficients[3]
                # B_pi_3 = coefficients[4]
                # B_pi_4 = coefficients[5]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                # Buckingham_Modifier = A_pi * D_R ** B_pi_1 * D_R50 ** B_pi_2 * rho_R ** B_pi_3 * R_s ** B_pi_4
                Buckingham_Modifier = A_pi * R_s ** B_pi_1
                
                f_m_prime = (1. + Buckingham_Modifier) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            elif method == 89:
                # This is the calculation code for Method 84 - Miscellaneous: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 90:
                # This is the calculation code for Method 90 - Miscellaneous: Pfeffer et al. generalized with additional nondimensional terms (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                A_pi = coefficients[1]
                B_pi_1 = coefficients[2]
                B_pi_2 = coefficients[3]
                # B_pi_3 = coefficients[4]
                # B_pi_4 = coefficients[5]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                # Buckingham_Modifier = A_pi * D_R ** B_pi_1 * D_R50 ** B_pi_2 * rho_R ** B_pi_3 * R_s ** B_pi_4
                Buckingham_Modifier = A_pi * R_s ** B_pi_1 * rho_R ** B_pi_2
                
                f_m_prime = f_g * (1. + Buckingham_Modifier) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            elif method == 91:
                # This is the calculation code for Method 90 - Miscellaneous: Pfeffer et al. generalized with additional nondimensional terms (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                A_pi_1 = coefficients[1]
                A_pi_2 = coefficients[2]
                B_pi_1 = coefficients[3]
                B_pi_2 = coefficients[4]
                # B_pi_3 = coefficients[4]
                # B_pi_4 = coefficients[5]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                # Buckingham_Modifier = A_pi * D_R ** B_pi_1 * D_R50 ** B_pi_2 * rho_R ** B_pi_3 * R_s ** B_pi_4
                Buckingham_Modifier = A_pi_1 * R_s **  B_pi_1 + A_pi_2 * rho_R ** B_pi_2
                
                f_m_prime = f_g * (1. + Buckingham_Modifier) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 92:
                # This is the calculation code for Method 90 - Miscellaneous: Pfeffer et al. generalized with additional nondimensional terms (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                A_pi_1 = coefficients[1]
                # A_pi_2 = coefficients[2]
                B_pi_1 = coefficients[2]
                # B_pi_2 = coefficients[4]
                # B_pi_3 = coefficients[4]
                # B_pi_4 = coefficients[5]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                # Buckingham_Modifier = A_pi * D_R ** B_pi_1 * D_R50 ** B_pi_2 * rho_R ** B_pi_3 * R_s ** B_pi_4
                Buckingham_Modifier = A_pi_1 * R_s **  B_pi_1
                
                f_m_prime = f_g * (1. + Buckingham_Modifier) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 93:
                # This is the calculation code for Method 90 - Miscellaneous: Pfeffer et al. generalized with additional nondimensional terms (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                # A_pi_2 = coefficients[2]
                B_pi_1 = coefficients[1]
                # B_pi_2 = coefficients[4]
                # B_pi_3 = coefficients[4]
                # B_pi_4 = coefficients[5]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                # Buckingham_Modifier = A_pi * D_R ** B_pi_1 * D_R50 ** B_pi_2 * rho_R ** B_pi_3 * R_s ** B_pi_4
                Buckingham_Modifier = A_pi_1 * (1 + R_s) **  B_pi_1 
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 94:
                # This is the calculation code for Method 90 - Miscellaneous: Pfeffer et al. generalized with additional nondimensional terms (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                B_pi_1 = coefficients[2]
                B_pi_2 = coefficients[3]
                # B_pi_3 = coefficients[4]
                # B_pi_4 = coefficients[5]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                Buckingham_Modifier = A_pi_1 * (1 + R_s) **  B_pi_1 * A_pi_2 * D_R ** B_pi_2
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            elif method == 95:
                # This is the calculation code for Method 90 - Miscellaneous: Pfeffer et al. generalized with additional nondimensional terms (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                B_pi_1 = coefficients[2]
                B_pi_2 = coefficients[3]
                # B_pi_3 = coefficients[4]
                # B_pi_4 = coefficients[5]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                Buckingham_Modifier = A_pi_1 * (1 + R_s) **  B_pi_1 * A_pi_2 * (1 + D_R) ** B_pi_2
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            elif method == 96:
                # This is the calculation code for Method 96 - Miscellaneous: Pfeffer et al. generalized with additional nondimensional terms (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                B_pi_1 = coefficients[2]
                B_pi_2 = coefficients[3]
                # B_pi_3 = coefficients[4]
                # B_pi_4 = coefficients[5]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                Buckingham_Modifier = A_pi_1 * (1 + R_s) **  B_pi_1 * A_pi_2 * (1 + rho_R) ** B_pi_2
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            return P_H
            
        @staticmethod
        def Vertical_Pipe_Sections(method, L, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, Direction, coefficients, Function_Dict, extra_args=None):
            # This class contains the pressure drop calculation methods for vertical pipe sections.
            
            if method == 1:
                # This is the calculation code for Method 1 - Total pressure: Michaelides:
                
                # This section of code extracts the coefficients used in this method.
                K_p = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = K_p * 1. / Fr ** 0.5
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 2:
                # This is the calculation code for Method 2 - Total pressure: Setia et al.:
                
                # This section of code extracts the coefficients used in this method.
                a_s = coefficients[0]
                b_s = coefficients[0]
                c_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = a_s * R_s ** b_s * Fr ** c_s
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 3:
                # This is the calculation code for Method 3 - Total pressure: Mason et al.:
                
                # This section of code extracts the coefficients used in this method.
                f_pM = coefficients[0]
                a_M = coefficients[1]
                b_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                P_s = f_pM * L * R_s ** a_M * Fr ** b_M
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 4:
                # This is the calculation code for Method 4 - Total pressure: Mehta et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_V = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 5:
                # This is the calculation code for Method 4 - Total pressure: Mehta et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_V = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
            
            elif method == 6:
                # This is the calculation code for Method 6 - Two phase Ozbelge 1. (Solids velocity: Tripathi et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section of code calculates the terminal velocity of a particle.
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56           
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_s = g * D / (2. * v_s ** 2.) * (alpha ** -4.7 * (v_r / v_t) ** 2. - 1.)
                
                P_s = 2. * f_s * L * v_s * rho_ds * A_c / D
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 7:
                # This is the calculation code for Method 6 - Two phase Ozbelge 1. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = 0.001
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56           
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_s = g * D / (2. * v_s ** 2.) * (alpha ** -4.7 * (v_r / v_t) ** 2. - 1.)
                
                P_s = 2. * f_s * L * v_s * rho_ds * A_c / D
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 8:
                # This is the calculation code for Method 8 - Two phase: Ozbelge 2. (Solids velocity: Tripathi et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_s = 0.0054 * (m_s / m_g * rho_g / rho_s) ** -0.115 * (v_g / v_r * d / D) ** 0.339
                
                P_s = 2. * f_s * L * v_s * rho_ds * A_c / D
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 9:
                # This is the calculation code for Method 9 - Two phase: Ozbelge 2. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = 0.001
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_s = 0.0054 * (m_s / m_g * rho_g / rho_s) ** -0.115 * (v_g / v_r * d / D) ** 0.339
                
                P_s = 2. * f_s * L * v_s * rho_ds * A_c / D
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 10:
                # This is the calculation code for Method 10 - Combined: Michaelides (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                K_p = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = K_p * 1. / Fr ** 0.5
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 11:
                # This is the calculation code for Method 11 - Combined: Michaelides (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                K_p = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = K_p * 1. / Fr ** 0.5
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 12:
                # This is the calculation code for Method 12 - Combined: Setia et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_s = coefficients[0]
                b_s = coefficients[0]
                c_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = a_s * R_s ** b_s * Fr ** c_s
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 13:
                # This is the calculation code for Method 13 - Combined: Setia et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_s = coefficients[0]
                b_s = coefficients[0]
                c_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = a_s * R_s ** b_s * Fr ** c_s
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 14:
                # This is the calculation code for Method 14 - Combined: Mason et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_pM = coefficients[0]
                a_M = coefficients[1]
                b_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = f_pM * L * R_s ** a_M * Fr ** b_M           
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 15:
                # This is the calculation code for Method 15 - Combined: Mason et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_pM = coefficients[0]
                a_M = coefficients[1]
                b_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = f_pM * L * R_s ** a_M * Fr ** b_M           
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 16:
                # This is the calculation code for Method 16 - Combined: Mehta et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                P_V = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)  + P_s
            
            elif method == 17:
                # This is the calculation code for Method 17 - Combined: Mehta et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                P_V = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)  + P_s
                        
            elif method == 18:
                # This is the calculation code for Method 18 - Combined: Ozbelge 1. (Solids velocity: Tripathi et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section of code calculates the terminal velocity of a particle.
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56           
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_s = g * D / (2. * v_s ** 2.) * (alpha ** -4.7 * (v_r / v_t) ** 2. - 1.)
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = 2. * f_s * L * v_s * rho_ds * A_c / D        
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 19:
                # This is the calculation code for Method 19 - Combined: Ozbelge 1. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56           
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_s = g * D / (2. * v_s ** 2.) * (alpha ** -4.7 * (v_r / v_t) ** 2. - 1.)
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = 2. * f_s * L * v_s * rho_ds * A_c / D        
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 20:
                # This is the calculation code for Method 20 - Combined: Ozbelge 2. (Solids velocity: Tripathi et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = v_f - v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_s = 0.0054 * (m_s / m_g * rho_g / rho_s) ** -0.115 * (v_g / v_r * d / D) ** 0.339
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = 2. * f_s * L * v_s * rho_ds * A_c / D        
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 21:
                # This is the calculation code for Method 21 - Combined: Ozbelge 2. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the superficial gas velocity.
                v_f = alpha * v_g
                
                # This section calculates slip velocity. 
                v_r = 0.001
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_s = 0.0054 * (m_s / m_g * rho_g / rho_s) ** -0.115 * (v_g / v_r * d / D) ** 0.339
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = 2. * f_s * L * v_s * rho_ds * A_c / D        
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                        
            elif method == 22:
                # This is the calculation code for Method 22 - Miscellaneous: Pfeffer et al. generalized (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            elif method == 23:
                # This is the calculation code for Method 23 - Miscellaneous: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                        
            elif method == 24:
                # This is the calculation code for Method 24 - Combined: Pfeffer et al. generalized (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
            
            elif method == 25:
                # This is the calculation code for Method 25 - Combined: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
            
            elif method == 26:
                # This is the calculation code for Method 26 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi = coefficients[0]
                B_pi_1 = coefficients[1]
                B_pi_2 = coefficients[2]
                B_pi_3 = coefficients[3]
                B_pi_4 = coefficients[4]
                B_pi_5 = coefficients[5]
                B_pi_6 = coefficients[6]
                B_pi_7 = coefficients[7]
                B_pi_8 = coefficients[8]
                B_pi_9 = coefficients[9]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_V = A_pi * (m_g ** 2.) / (D ** 4. * rho_g) * L_R ** B_pi_1 * D_R ** B_pi_2 * D_R50 ** B_pi_3 * rho_R ** B_pi_4 * epsilon_R ** B_pi_5 * Re_p ** B_pi_6 * Fr ** B_pi_7 * R_s ** B_pi_8 * P_R ** B_pi_9
                
            elif method == 27:
                # This is the calculation code for Method 27 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                B_pi_1 = coefficients[9]
                B_pi_2 = coefficients[10]
                B_pi_3 = coefficients[11]
                B_pi_4 = coefficients[12]
                B_pi_5 = coefficients[13]
                B_pi_6 = coefficients[14]
                B_pi_7 = coefficients[15]
                B_pi_8 = coefficients[16]
                B_pi_9 = coefficients[17]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_V = (m_g ** 2.) / (D ** 4. * rho_g) * (A_pi_1 * L_R ** B_pi_1 + A_pi_2 * D_R ** B_pi_2 + A_pi_3 * D_R50 ** B_pi_3 + A_pi_4 * rho_R ** B_pi_4 + A_pi_5 * epsilon_R ** B_pi_5 + A_pi_6 * Re_p ** B_pi_6 + A_pi_7 * Fr ** B_pi_7 + A_pi_8 * R_s ** B_pi_8 + A_pi_9 * P_R ** B_pi_9)
            
            elif method == 28:
                # This is the calculation code for Method 28 - Buckingham-Pi Theorem:
                                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                B_pi_1 = coefficients[9]
                B_pi_2 = coefficients[10]
                B_pi_3 = coefficients[11]
                B_pi_4 = coefficients[12]
                B_pi_5 = coefficients[13]
                B_pi_6 = coefficients[14]
                B_pi_7 = coefficients[15]
                B_pi_8 = coefficients[16]
                B_pi_9 = coefficients[17]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code uses the coefficients to find the function inputs for this method.
                L_R_C = A_pi_1 * L_R ** B_pi_1
                D_R_C = A_pi_2 * D_R ** B_pi_2
                D_R50_C = A_pi_3 * D_R50 ** B_pi_3
                rho_R_C = A_pi_4 * rho_R ** B_pi_4
                epsilon_R_C = A_pi_5 * epsilon_R ** B_pi_5
                Re_p_C = A_pi_6 * Re_p ** B_pi_6
                Fr_C = A_pi_7 * Fr ** B_pi_7
                R_s_C = A_pi_8 * R_s ** B_pi_8
                P_R_C = A_pi_9 * P_R ** B_pi_9
                
                # Pull function out of extra args
                try:
                    [func, orientation] = extra_args
                except:
                    [func, orientation, L_t] = extra_args
                
                
                # Set zero functions based on section type.
                if orientation == 'Upward':
                    Z_HP = 0.
                    Z_VUP = 1.
                    Z_VDP = 0.
                    Z_BHH = 0.
                    Z_BHU = 0.
                    Z_BHD = 0.
                    Z_BUH = 0.
                    Z_BDH = 0.
                    Z_AS = 0.
                elif orientation == 'Downward':
                    Z_HP = 0.
                    Z_VUP = 0.
                    Z_VDP = 1.
                    Z_BHH = 0.
                    Z_BHU = 0.
                    Z_BHD = 0.
                    Z_BUH = 0.
                    Z_BDH = 0.
                    Z_AS = 0.                    
                
                # This section of code calculates the dimensionless pressure drop based on the dimensionless numbers and the passed function.
                # P_V_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C, Z_HP, Z_VUP, Z_VDP, Z_BHH, Z_BHU, Z_BHD, Z_BUH, Z_BDH, Z_AS)
                P_V_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C)
                
                # Convert dimensonless pressure drop to pressure drop.
                P_V = (m_g ** 2.) / (D ** 4. * rho_g) * P_V_star
            
            elif method == 29:
                # This is the calculation code for Method 29 - Combined: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                B_pi_1 = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                Buckingham_Modifier = (1 + R_s) **  B_pi_1
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
                
            elif method == 30:
                # This is the calculation code for Method 29 - Combined: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                B_pi_1 = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                Buckingham_Modifier = A_pi_1 * (1 + R_s) **  B_pi_1
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
                
            elif method == 31:
                # This is the calculation code for Method 29 - Combined: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                B_pi_1 = coefficients[2]
                B_pi_2 = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                Buckingham_Modifier = A_pi_1 * (1 + R_s) **  B_pi_1 * A_pi_2 * (1 + D_R) ** B_pi_2
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
            
            elif method == 32:
                # This is the calculation code for Method 29 - Combined: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                B_pi_1 = coefficients[2]
                B_pi_2 = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                Buckingham_Modifier = A_pi_1 * (1 + R_s) **  B_pi_1 * A_pi_2 * (1 + rho_R) ** B_pi_2
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
                
            elif method == 33:
                # This is the calculation code for Method 29 - Combined: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                B_pi_1 = coefficients[0]
                B_pi_2 = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                Buckingham_Modifier = (1 + R_s) **  B_pi_1 + (1 + rho_R) ** B_pi_2
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
            
            elif method == 34:
                # This is the calculation code for Method 29 - Combined: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                B_pi_1 = coefficients[0]
                B_pi_2 = coefficients[1]
                B_pi_3 = coefficients[2]
                B_pi_4 = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                Buckingham_Modifier = (1 + R_s) **  B_pi_1 + (1 + D_R) ** B_pi_2 + (1 + D_R50) ** B_pi_3 + (1 + rho_R) ** B_pi_4
                
                f_m_prime = f_g * Buckingham_Modifier
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
            
            return P_V
            
        @staticmethod
        def Bends(method, type, R_B, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, Direction, coefficients, Function_Dict, extra_args=None):
            # This class contains the pressure drop calculation methods for bends.
            
            if method == 1:
                # This is the calculation code for Method 1 - Rinoshika (Gas pressure drop: Mason et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_b_p = coefficients[0]
                b_b_p = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                    
                # This section calculates the equivalent length for the bend. 
                f_T = 0.25 / (np.log10((epsilon / D) / 3.7)) ** 2.
                                
                K_L = cs(R_B / D) * f_T
                
                L_eq = K_L * D / f_g
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_g * rho_g / 2. * v_g ** 2. / D * L_eq
                
                # This section calculates the solids friction factor.
                f_b_p = a_b_p * Fr ** b_b_p
                
                # This section calculates the pressure drop due to solids.
                P_B_s = m_s / m_g * R_B * rho_g * v_g ** 2. / (2. * D) * f_b_p
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 2:
                # This is the calculation code for Method 2 - Rinoshika (Gas pressure drop: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_b_p = coefficients[0]
                b_b_p = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                # This section of code determines the bend type coefficient.
                if type == 'Radius':
                    a_T = 1.
                else:
                    a_T = 25.
                
                # This section calculates the pressure drop due to gas.
                P_B_g = a_T * (0.0194 * v_g ** 2. + 0.0374 * v_g ** 1.75) * (R_B / D) ** 0.84
                
                # This section calculates the solids friction factor.
                f_b_p = a_b_p * Fr ** b_b_p
                
                # This section calculates the pressure drop due to solids.
                P_B_s = m_s / m_g * R_B * rho_g * v_g ** 2. / (2. * D) * f_b_p
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 3:
                # This is the calculation code for Method 3 - Mason et al. (Gas pressure drop: Mason et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_B_M = coefficients[0]
                a_B_M = coefficients[1]
                b_B_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                    
                # This section calculates the equivalent length for the bend. 
                f_T = 0.25 / (np.log10((epsilon / D) / 3.7)) ** 2.
                
                K_L = cs(R_B / D) * f_T
                
                L_eq = K_L * D / f_g
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_g * rho_g / 2. * v_g ** 2. / D * L_eq
                
                # This section calculates the pressure drop due to solids.
                P_B_s = f_B_M * R_s ** a_B_M * Fr ** b_B_M
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 4:
                # This is the calculation code for Method 4 - Mason et al. (Gas pressure drop: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_B_M = coefficients[0]
                a_B_M = coefficients[1]
                b_B_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                # This section of code determines the bend type coefficient.
                if type == 'Radius':
                    a_T = 1.
                else:
                    a_T = 25.
                
                # This section calculates the pressure drop due to gas.
                P_B_g = a_T * (0.0194 * v_g ** 2. + 0.0374 * v_g ** 1.75) * (R_B / D) ** 0.84
                
                # This section calculates the pressure drop due to solids.
                P_B_s = f_B_M * R_s ** a_B_M * Fr ** b_B_M
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 5:
                # This is the calculation code for Method 5 - Singh and Wolfe:
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.              
                
                # This section sets the constant determined by Singh and Wolfe.
                a_sw = 0.00334
                
                # Find the outlet pressure by iterating over pressure drop methods.
                if (a_sw, m_s, D, R_B, rho_g, A_c, R_g, T_g, m_g) in Function_Dict['v_fao_singh_fun']:
                    v_fao = Function_Dict['v_fao_singh_fun'][(a_sw, m_s, D, R_B, rho_g, A_c, R_g, T_g, m_g)]
                else:
                    v_fao = minimize(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_singh_fun, v_g, args=(a_sw, m_s, D, R_B, rho_g, A_c, R_g, T_g, m_g), method='COBYLA')['x']
                    Function_Dict['v_fao_singh_fun'][(a_sw, m_s, D, R_B, rho_g, A_c, R_g, T_g, m_g)] = v_fao
                
                # The total pressure drop is found next.
                P_B = 0.13 + a_sw * m_s * v_fao / D ** 2. * (R_B / D) ** -0.18
                
            elif method == 6:
                # This is the calculation code for Method 6 - Rossetti:
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.      
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)

                # This section of code calculates the gas only friction factor for bends.
                f_B_g = 0.167 * (1. + 17.062 * (2. * R_B / D) ** -1.219) * Re_p ** -0.17 * (2. * R_B / D) ** 0.84
                
                # This section of code calculates the velocity, density, and Froude number at the bend outlet.
                if (g, D, m_g, R_B, Re_p, rho_g, A_c, R_g, T_g, R_s) in Function_Dict['v_fao_rossetti_fun']:
                    v_fao = Function_Dict['v_fao_rossetti_fun'][(g, D, m_g, R_B, Re_p, rho_g, A_c, R_g, T_g, R_s)]
                else:
                    # v_fao = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_rossetti_fun, v_g, args=(g, D, m_g, R_B, Re_p, rho_g, A_c, R_g, T_g, R_s), method="lm").x[0]
                    method = 'COBYLA'
                    solution_array = minimize(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_rossetti_fun, v_g, args=(g, D, m_g, R_B, Re_p, rho_g, A_c, R_g, T_g, R_s), method=method)
                    v_fao = solution_array['x']
                    Function_Dict['v_fao_rossetti_fun'][(g, D, m_g, R_B, Re_p, rho_g, A_c, R_g, T_g, R_s)] = v_fao
                rho_go = m_g / (v_fao * A_c)
                Fr_o = Section_Pressure_Drop.Dimensionless_Numbers.Bend_Outlet_Froude_Number(v_fao, g, D)
                
                # This section of code calculates the solids only friction factor for bends.
                f_B_s = (5.4 * R_s ** 1.293) / (Fr_o ** 0.84 * (2. * R_B / D) ** 0.39)
                
                # The total pressure drop is found next.
                P_B = (f_B_g + f_B_s) * rho_go * v_fao ** 2. / 2.
                
            elif method == 7:
                # This is the calculation code for Method 7 - de Moraes et al.:
                
                # This section of code extracts the coefficients used in this method.
                a_dM = coefficients[0]
                b_dM = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.      
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                # This section of code calculates the velocity and density at the bend outlet.
                if (g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g) in Function_Dict['v_fao_de_moraes_fun']:
                    v_fao = Function_Dict['v_fao_de_moraes_fun'][(g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g)]
                else:
                    v_fao = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_de_moraes_fun, v_g, args=(g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g), method='lm').x[0]
                    # method = 'COBYLA'
                    # solution_array = minimize(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_de_moraes_fun, v_g, args=(g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g), method=method)
                    # v_fao_o = solution_array['x']
                    # print '{:.2f} - {:.2f}'.format(v_fao_o, v_fao)
                    Function_Dict['v_fao_de_moraes_fun'][(g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g)] = v_fao
                rho_go = m_g / (v_fao * A_c)

                # This section of code calculates the gas only friction factor for bends.
                f_B_g = 0.167 * (1. + 17.062 * (2. * R_B / D) ** -1.219) * Re_p ** -0.17 * (2. * R_B / D) ** 0.84
                
                # This section of code calculates the solids only friction factor for bends.
                f_B_s = a_dM * np.exp(b_dM * v_fao)
                
                # The total pressure drop is found next.
                P_B = (f_B_g + f_B_s) * rho_go * v_fao ** 2. / 2.
                
            elif method == 8:
                # This is the calculation code for Method 8 - Chambers and Marcus:
                
                # This section of code extracts the coefficients used in this method.
                B_L = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)

                # This section of code calculates the velocity and density at the bend outlet.
                if (m_g, A_c, B_L, R_s, rho_g, R_g, T_g) in Function_Dict['v_fao_chambers_fun']:
                    v_fao = Function_Dict['v_fao_chambers_fun'][(m_g, A_c, B_L, R_s, rho_g, R_g, T_g)]
                else:
                    v_fao = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_chambers_fun, v_g, args=(m_g, A_c, B_L, R_s, rho_g, R_g, T_g), method="lm").x[0]
                    Function_Dict['v_fao_chambers_fun'][(m_g, A_c, B_L, R_s, rho_g, R_g, T_g)] = v_fao
                rho_go = m_g / (v_fao * A_c)
                
                # The total pressure drop is found next.
                P_B = B_L * (1. + R_s) * (rho_go * v_fao ** 2.) / 2.
                
            elif method == 9:
                # This is the calculation code for Method 9 - Mallick:
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)

                # This section of code sets the bend loss coefficient.
                B_L = 0.5
                
                # This section of code calculates the velocity and density at the bend outlet.
                if (m_g, A_c, B_L, R_s, rho_g, R_g, T_g) in Function_Dict['v_fao_chambers_fun']:
                    v_fao = Function_Dict['v_fao_chambers_fun'][(m_g, A_c, B_L, R_s, rho_g, R_g, T_g)]
                else:
                    v_fao = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_chambers_fun, v_g, args=(m_g, A_c, B_L, R_s, rho_g, R_g, T_g), method="lm").x[0]
                    Function_Dict['v_fao_chambers_fun'][(m_g, A_c, B_L, R_s, rho_g, R_g, T_g)] = v_fao
                rho_go = m_g / (v_fao * A_c)
                
                # The total pressure drop is found next.
                P_B = B_L * (1. + R_s) * (rho_go * v_fao ** 2.) / 2.
                                
            elif method == 10:
                # This is the calculation code for Method 10 - Pan:
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)

                # This section of code calculates the velocity, density, and Froude number at the bend outlet.
                if (g, D, m_g, rho_g, A_c, R_s, R_g, T_g) in Function_Dict['v_fao_pan_fun']:
                    v_fao = Function_Dict['v_fao_pan_fun'][(g, D, m_g, rho_g, A_c, R_s, R_g, T_g)]
                else:
                    v_fao = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_pan_fun, v_g, args=(g, D, m_g, rho_g, A_c, R_s, R_g, T_g), method="lm").x[0]
                    Function_Dict['v_fao_pan_fun'][(g, D, m_g, rho_g, A_c, R_s, R_g, T_g)] = v_fao
                rho_go = m_g / (v_fao * A_c)
                Fr_o = Section_Pressure_Drop.Dimensionless_Numbers.Bend_Outlet_Froude_Number(v_fao, g, D)
                
                # The total pressure drop is found next.
                P_B = 0.005 * R_s ** 1.49 * Fr_o **1.1182 * rho_go * v_fao ** 2. / 2.
                
            elif method == 11:
                # This is the calculation code for Method 10 - Das and Meloy:
                
                # This section of code extracts the coefficients used in this method.
                a_D = coefficients[0]
                b_D = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)

                # This section of code calculates the velocity at the bend outlet.
                if (a_D, b_D, R_s, rho_g, m_g, A_c, R_g, T_g) in Function_Dict['v_fao_das_fun']:
                    v_fao = Function_Dict['v_fao_das_fun'][(a_D, b_D, R_s, rho_g, m_g, A_c, R_g, T_g)]
                else:
                    v_fao = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_das_fun, v_g, args=(a_D, b_D, R_s, rho_g, m_g, A_c, R_g, T_g), method="lm").x[0]
                    Function_Dict['v_fao_das_fun'][(a_D, b_D, R_s, rho_g, m_g, A_c, R_g, T_g)] = v_fao
                
                # The total pressure drop is found next.
                P_B = a_D * R_s * v_fao ** b_D
                
            elif 12 <= method <= 95:
                # This is the calculation code for Methods 12-71 - Schuchart (All Bend Orientations):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                    
                # This section calculates the equivalent length for the bend. 
                f_T = 0.25 / (np.log10((epsilon / D) / 3.7)) ** 2.
                
                K_L = cs(R_B / D) * f_T
                
                L_eq = K_L * D / f_g
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_g * rho_g / 2. * v_g ** 2. / D * L_eq
                
                # P_g: This section of code contains pressure drop calculations in a straight pipe of R_B due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * R_B
                
                # This section calculates the pressure drop due to solids in a straight pipe of length R_B.
                method_p = method - 11
                P_s_R_B = Section_Pressure_Drop.Dilute_Phase.Horizontal_Pipe_Sections(method_p, R_B, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, coefficients, Function_Dict) - P_g
                
                # This section calculates the pressure drop due to solids.
                P_B_s = 210. * (2. * R_B / D ) ** -1.15 * P_s_R_B
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif 96 <= method <= 120:
                # This is the calculation code for Methods 72-85 - Schuchart (H-V and V-H Bend Orientations):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                    
                # This section calculates the equivalent length for the bend. 
                f_T = 0.25 / (np.log10((epsilon / D) / 3.7)) ** 2.
                
                K_L = cs(R_B / D) * f_T
                
                L_eq = K_L * D / f_g
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_g * rho_g / 2. * v_g ** 2. / D * L_eq
                
                # P_g: This section of code contains pressure drop calculations in a straight pipe of R_B due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * R_B
                
                # This section calculates the pressure drop due to solids in a straight pipe of length R_B.
                method_p = method - 95
                P_s_R_B = Section_Pressure_Drop.Dilute_Phase.Vertical_Pipe_Sections(method_p, R_B, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, Direction, coefficients, Function_Dict) - P_g
                
                # This section calculates the pressure drop due to solids.
                P_B_s = 210. * (2. * R_B / D ) ** -1.15 * P_s_R_B
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 121:
                # This is the calculation code for Method 121 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi = coefficients[0]
                B_pi_1 = coefficients[1]
                B_pi_2 = coefficients[2]
                B_pi_3 = coefficients[3]
                B_pi_4 = coefficients[4]
                B_pi_5 = coefficients[5]
                B_pi_6 = coefficients[6]
                B_pi_7 = coefficients[7]
                B_pi_8 = coefficients[8]
                B_pi_9 = coefficients[9]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = R_B
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_B = A_pi * (m_g ** 2.) / (D ** 4. * rho_g) * L_R ** B_pi_1 * D_R ** B_pi_2 * D_R50 ** B_pi_3 * rho_R ** B_pi_4 * epsilon_R ** B_pi_5 * Re_p ** B_pi_6 * Fr ** B_pi_7 * R_s ** B_pi_8 * P_R ** B_pi_9
                
            elif method == 122:
                # This is the calculation code for Method 122 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                B_pi_1 = coefficients[9]
                B_pi_2 = coefficients[10]
                B_pi_3 = coefficients[11]
                B_pi_4 = coefficients[12]
                B_pi_5 = coefficients[13]
                B_pi_6 = coefficients[14]
                B_pi_7 = coefficients[15]
                B_pi_8 = coefficients[16]
                B_pi_9 = coefficients[17]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = R_B
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_B = (m_g ** 2.) / (D ** 4. * rho_g) * (A_pi_1 * L_R ** B_pi_1 + A_pi_2 * D_R ** B_pi_2 + A_pi_3 * D_R50 ** B_pi_3 + A_pi_4 * rho_R ** B_pi_4 + A_pi_5 * epsilon_R ** B_pi_5 + A_pi_6 * Re_p ** B_pi_6 + A_pi_7 * Fr ** B_pi_7 + A_pi_8 * R_s ** B_pi_8 + A_pi_9 * P_R ** B_pi_9)
                
            elif method == 123:
                # This is the calculation code for Method 123 - Buckingham-Pi Theorem:
                                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                B_pi_1 = coefficients[9]
                B_pi_2 = coefficients[10]
                B_pi_3 = coefficients[11]
                B_pi_4 = coefficients[12]
                B_pi_5 = coefficients[13]
                B_pi_6 = coefficients[14]
                B_pi_7 = coefficients[15]
                B_pi_8 = coefficients[16]
                B_pi_9 = coefficients[17]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = R_B
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code uses the coefficients to find the function inputs for this method.
                L_R_C = A_pi_1 * L_R ** B_pi_1
                D_R_C = A_pi_2 * D_R ** B_pi_2
                D_R50_C = A_pi_3 * D_R50 ** B_pi_3
                rho_R_C = A_pi_4 * rho_R ** B_pi_4
                epsilon_R_C = A_pi_5 * epsilon_R ** B_pi_5
                Re_p_C = A_pi_6 * Re_p ** B_pi_6
                Fr_C = A_pi_7 * Fr ** B_pi_7
                R_s_C = A_pi_8 * R_s ** B_pi_8
                P_R_C = A_pi_9 * P_R ** B_pi_9
                
                # Pull function out of extra args
                try:
                    [func, orientation] = extra_args
                except:
                    [func, orientation, L_t] = extra_args
                
                # Set zero functions based on section type.
                bend_inlet, bend_outlet = orientation.split('-')
                if bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet in ['N', 'E', 'S', 'W']:
                    Z_HP = 0.
                    Z_VUP = 0.
                    Z_VDP = 0.
                    Z_BHH = 1.
                    Z_BHU = 0.
                    Z_BHD = 0.
                    Z_BUH = 0.
                    Z_BDH = 0.
                    Z_AS = 0.
                elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'U':
                    Z_HP = 0.
                    Z_VUP = 0.
                    Z_VDP = 0.
                    Z_BHH = 0.
                    Z_BHU = 1.
                    Z_BHD = 0.
                    Z_BUH = 0.
                    Z_BDH = 0.
                    Z_AS = 0.
                elif bend_inlet in ['N', 'E', 'S', 'W'] and bend_outlet == 'D':
                    Z_HP = 0.
                    Z_VUP = 0.
                    Z_VDP = 0.
                    Z_BHH = 0.
                    Z_BHU = 0.
                    Z_BHD = 1.
                    Z_BUH = 0.
                    Z_BDH = 0.
                    Z_AS = 0.
                elif bend_inlet == 'U':
                    Z_HP = 0.
                    Z_VUP = 0.
                    Z_VDP = 0.
                    Z_BHH = 0.
                    Z_BHU = 0.
                    Z_BHD = 0.
                    Z_BUH = 1.
                    Z_BDH = 0.
                    Z_AS = 0.    
                elif bend_inlet == 'D':
                    Z_HP = 0.
                    Z_VUP = 0.
                    Z_VDP = 0.
                    Z_BHH = 0.
                    Z_BHU = 0.
                    Z_BHD = 0.
                    Z_BUH = 0.
                    Z_BDH = 1.
                    Z_AS = 0.    
                
                # This section of code calculates the dimensionless pressure drop based on the dimensionless numbers and the passed function.
                # P_B_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C, Z_HP, Z_VUP, Z_VDP, Z_BHH, Z_BHU, Z_BHD, Z_BUH, Z_BDH, Z_AS)
                P_B_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C)
                
                # Convert dimensonless pressure drop to pressure drop.
                P_B = (m_g ** 2.) / (D ** 4. * rho_g) * P_B_star
                
            return P_B
            
        @staticmethod
        def Acceleration_Of_Solids(method, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g, m_s, R_g, T_g, mu_g, coefficients, Function_Dict, extra_args=None):
            # This class contains the pressure drop calculation methods for the acceleration of solids.
            
            if method == 1:
                # This is the calculation code for Method 1 - Tripathi et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
        
            elif method == 2:
                # This is the calculation code for Method 2 - Tripathi et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
                
            elif method == 3:
                # This is the calculation code for Method 3 - Tripathi et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
                
            elif method == 4:
                # This is the calculation code for Method 4 - Tripathi et al. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
                
            elif method == 5:
                # This is the calculation code for Method 4 - Tripathi et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
        
            elif method == 6:
                # This is the calculation code for Method 6 - Mehta et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
        
            elif method == 7:
                # This is the calculation code for Method 7 - Mehta et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
                
            elif method == 8:
                # This is the calculation code for Method 8 - Mehta et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
                
            elif method == 9:
                # This is the calculation code for Method 9 - Mehta et al. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
                
            elif method == 10:
                # This is the calculation code for Method 10 - Mehta et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
                
            elif method == 11:
                # This is the calculation code for Method 11 - Marcus et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
        
            elif method == 12:
                # This is the calculation code for Method 12 - Marcus et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
                
            elif method == 13:
                # This is the calculation code for Method 13 - Marcus et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
                
            elif method == 14:
                # This is the calculation code for Method 14 - Marcus et al. (Solids velocity: Yang):
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
                
            elif method == 15:
                # This is the calculation code for Method 15 - Marcus et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
                
            elif method == 16:
                # This is the calculation code for Method 16 - Tomita and Tasiro:
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = rho_g * v_g ** 2. / 2. * (1. - rho_g / rho_s)
                
            elif method == 17:
                # This is the calculation code for Method 17 - Duckworth:
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = R_s * rho_g * v_g ** 2. / 2. * (v_g ** 2. / (g * d) * (rho_g / rho_s) ** 2.)
                
            elif method == 18:
                # This is the calculation code for Method 18 - Agarwal (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
        
            elif method == 19:
                # This is the calculation code for Method 19 - Agarwal (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
                
            elif method == 20:
                # This is the calculation code for Method 20 - Agarwal (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
                
            elif method == 21:
                # This is the calculation code for Method 21 - Agarwal (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = brentq(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_s_fun, 0., v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c))
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
                
            elif method == 22:
                # This is the calculation code for Method 22 - Agarwal (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
                
            elif method == 23:
                # This is the calculation code for Method 23 - Buckingham-Pi Theorem:
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code extracts the coefficients used in this method.
                A_pi = coefficients[0]
                B_pi_2 = coefficients[1]
                B_pi_3 = coefficients[2]
                B_pi_4 = coefficients[3]
                B_pi_5 = coefficients[4]
                B_pi_6 = coefficients[5]
                B_pi_7 = coefficients[6]
                B_pi_8 = coefficients[7]
                B_pi_9 = coefficients[8]
                                
                # This section of code calculates the non-dimensional numbers used for this method.
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_A = A_pi * (m_g ** 2.) / (D ** 4. * rho_g) * D_R ** B_pi_2 * D_R50 ** B_pi_3 * rho_R ** B_pi_4 * epsilon_R ** B_pi_5 * Re_p ** B_pi_6 * Fr ** B_pi_7 * R_s ** B_pi_8 * P_R ** B_pi_9
                
            elif method == 24:
                # This is the calculation code for Method 24 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi_2 = coefficients[0]
                A_pi_3 = coefficients[1]
                A_pi_4 = coefficients[2]
                A_pi_5 = coefficients[3]
                A_pi_6 = coefficients[4]
                A_pi_7 = coefficients[5]
                A_pi_8 = coefficients[6]
                A_pi_9 = coefficients[7]
                B_pi_2 = coefficients[8]
                B_pi_3 = coefficients[9]
                B_pi_4 = coefficients[10]
                B_pi_5 = coefficients[11]
                B_pi_6 = coefficients[12]
                B_pi_7 = coefficients[13]
                B_pi_8 = coefficients[14]
                B_pi_9 = coefficients[15]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                                
                # This section of code calculates the non-dimensional numbers used for this method.
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_A = (m_g ** 2.) / (D ** 4. * rho_g) * (A_pi_2 * D_R ** B_pi_2 + A_pi_3 * D_R50 ** B_pi_3 + A_pi_4 * rho_R ** B_pi_4 + A_pi_5 * epsilon_R ** B_pi_5 + A_pi_6 * Re_p ** B_pi_6 + A_pi_7 * Fr ** B_pi_7 + A_pi_8 * R_s ** B_pi_8 + A_pi_9 * P_R ** B_pi_9)
                
            elif method == 25:
                # This is the calculation code for Method 25 - Buckingham-Pi Theorem:
                                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                B_pi_1 = coefficients[9]
                B_pi_2 = coefficients[10]
                B_pi_3 = coefficients[11]
                B_pi_4 = coefficients[12]
                B_pi_5 = coefficients[13]
                B_pi_6 = coefficients[14]
                B_pi_7 = coefficients[15]
                B_pi_8 = coefficients[16]
                B_pi_9 = coefficients[17]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                                
                # This section of code calculates the non-dimensional numbers used for this method.
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code uses the coefficients to find the function inputs for this method.
                D_R_C = A_pi_2 * D_R ** B_pi_2
                D_R50_C = A_pi_3 * D_R50 ** B_pi_3
                rho_R_C = A_pi_4 * rho_R ** B_pi_4
                epsilon_R_C = A_pi_5 * epsilon_R ** B_pi_5
                Re_p_C = A_pi_6 * Re_p ** B_pi_6
                Fr_C = A_pi_7 * Fr ** B_pi_7
                R_s_C = A_pi_8 * R_s ** B_pi_8
                P_R_C = A_pi_9 * P_R ** B_pi_9
                
                # Pull function out of extra args
                try:
                    [func, orientation] = extra_args
                except:
                    [func, orientation, L_t] = extra_args
                
                # Set zero functions based on section type.
                Z_HP = 0.
                Z_VUP = 0.
                Z_VDP = 0.
                Z_BHH = 0.
                Z_BHU = 0.
                Z_BHD = 0.
                Z_BUH = 0.
                Z_BDH = 0.
                Z_AS = 1.
                
                # Set L_R for acceleration portion of code.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_t, D)            
                L_R_C = A_pi_1 * L_R ** B_pi_1
                
                # This section of code calculates the dimensionless pressure drop based on the dimensionless numbers and the passed function.
                # P_A_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C, Z_HP, Z_VUP, Z_VDP, Z_BHH, Z_BHU, Z_BHD, Z_BUH, Z_BDH, Z_AS)
                P_A_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C)
                
                # Convert dimensonless pressure drop to pressure drop.
                P_A = (m_g ** 2.) / (D ** 4. * rho_g) * P_A_star
                
            return P_A
        
        @staticmethod
        def Split(method, L, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g, m_s, R_g, T_g, mu_g, Gas_Split, Solids_Split, coefficients, Function_Dict, extra_args=None):
            # This class contains the pressure drop calculation methods for the acceleration of solids.
            
            if method == 1:
                # This is the calculation code for Method 1 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi = coefficients[0]
                B_pi_1 = coefficients[1]
                B_pi_2 = coefficients[2]
                B_pi_3 = coefficients[3]
                B_pi_4 = coefficients[4]
                B_pi_5 = coefficients[5]
                B_pi_6 = coefficients[6]
                B_pi_7 = coefficients[7]
                B_pi_8 = coefficients[8]
                B_pi_9 = coefficients[9]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_S = A_pi * (m_g ** 2.) / (D ** 4. * rho_g) * L_R ** B_pi_1 * D_R ** B_pi_2 * D_R50 ** B_pi_3 * rho_R ** B_pi_4 * epsilon_R ** B_pi_5 * Re_p ** B_pi_6 * Fr ** B_pi_7 * R_s ** B_pi_8 * P_R ** B_pi_9
                
            elif method == 2:
                # This is the calculation code for Method 2 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                B_pi_1 = coefficients[9]
                B_pi_2 = coefficients[10]
                B_pi_3 = coefficients[11]
                B_pi_4 = coefficients[12]
                B_pi_5 = coefficients[13]
                B_pi_6 = coefficients[14]
                B_pi_7 = coefficients[15]
                B_pi_8 = coefficients[16]
                B_pi_9 = coefficients[17]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_S = (m_g ** 2.) / (D ** 4. * rho_g) * (A_pi_1 * L_R ** B_pi_1 + A_pi_2 * D_R ** B_pi_2 + A_pi_3 * D_R50 ** B_pi_3 + A_pi_4 * rho_R ** B_pi_4 + A_pi_5 * epsilon_R ** B_pi_5 + A_pi_6 * Re_p ** B_pi_6 + A_pi_7 * Fr ** B_pi_7 + A_pi_8 * R_s ** B_pi_8 + A_pi_9 * P_R ** B_pi_9)
                
            elif method == 3:
                # This is the calculation code for Method 3 - Buckingham-Pi Theorem:
                                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                B_pi_1 = coefficients[9]
                B_pi_2 = coefficients[10]
                B_pi_3 = coefficients[11]
                B_pi_4 = coefficients[12]
                B_pi_5 = coefficients[13]
                B_pi_6 = coefficients[14]
                B_pi_7 = coefficients[15]
                B_pi_8 = coefficients[16]
                B_pi_9 = coefficients[17]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code uses the coefficients to find the function inputs for this method.
                L_R_C = A_pi_1 * L_R ** B_pi_1
                D_R_C = A_pi_2 * D_R ** B_pi_2
                D_R50_C = A_pi_3 * D_R50 ** B_pi_3
                rho_R_C = A_pi_4 * rho_R ** B_pi_4
                epsilon_R_C = A_pi_5 * epsilon_R ** B_pi_5
                Re_p_C = A_pi_6 * Re_p ** B_pi_6
                Fr_C = A_pi_7 * Fr ** B_pi_7
                R_s_C = A_pi_8 * R_s ** B_pi_8
                P_R_C = A_pi_9 * P_R ** B_pi_9
                
                # Pull function out of extra args
                try:
                    [func, orientation] = extra_args
                except:
                    [func, orientation, L_t] = extra_args
                
                # This section of code calculates the dimensionless pressure drop based on the dimensionless numbers and the passed function.
                # P_S_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C, Z_HP, Z_VUP, Z_VDP, Z_BHH, Z_BHU, Z_BHD, Z_BUH, Z_BDH, Z_AS)
                P_S_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C)
                
                # Convert dimensionless pressure drop to pressure drop.
                P_S = (m_g ** 2.) / (D ** 4. * rho_g) * P_S_star
                
            elif method == 4:
                # This is the calculation code for Method 4 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi = coefficients[0]
                B_pi_1 = coefficients[1]
                B_pi_2 = coefficients[2]
                B_pi_3 = coefficients[3]
                B_pi_4 = coefficients[4]
                B_pi_5 = coefficients[5]
                B_pi_6 = coefficients[6]
                B_pi_7 = coefficients[7]
                B_pi_8 = coefficients[8]
                B_pi_9 = coefficients[9]
                B_pi_10 = coefficients[10]
                B_pi_11 = coefficients[11]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_S = A_pi * (m_g ** 2.) / (D ** 4. * rho_g) * L_R ** B_pi_1 * D_R ** B_pi_2 * D_R50 ** B_pi_3 * rho_R ** B_pi_4 * epsilon_R ** B_pi_5 * Re_p ** B_pi_6 * Fr ** B_pi_7 * R_s ** B_pi_8 * P_R ** B_pi_9 * Gas_Split ** B_pi_10 * Solids_Split ** B_pi_11
                
            elif method == 5:
                # This is the calculation code for Method 5 - Buckingham-Pi Theorem:
                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                A_pi_10 = coefficients[9]
                A_pi_11 = coefficients[10]
                B_pi_1 = coefficients[11]
                B_pi_2 = coefficients[12]
                B_pi_3 = coefficients[13]
                B_pi_4 = coefficients[14]
                B_pi_5 = coefficients[15]
                B_pi_6 = coefficients[16]
                B_pi_7 = coefficients[17]
                B_pi_8 = coefficients[18]
                B_pi_9 = coefficients[19]
                B_pi_10 = coefficients[20]
                B_pi_11 = coefficients[21]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code calculates the pressure drop based on the dimensionless numbers and the coefficients.
                P_S = (m_g ** 2.) / (D ** 4. * rho_g) * (A_pi_1 * L_R ** B_pi_1 + A_pi_2 * D_R ** B_pi_2 + A_pi_3 * D_R50 ** B_pi_3 + A_pi_4 * rho_R ** B_pi_4 + A_pi_5 * epsilon_R ** B_pi_5 + A_pi_6 * Re_p ** B_pi_6 + A_pi_7 * Fr ** B_pi_7 + A_pi_8 * R_s ** B_pi_8 + A_pi_9 * P_R ** B_pi_9 + A_pi_10 * Gas_Split ** B_pi_10 + A_pi_11 * Solids_Split ** B_pi_11)
                
            elif method == 6:
                # This is the calculation code for Method 6 - Buckingham-Pi Theorem:
                                
                # This section of code extracts the coefficients used in this method.
                A_pi_1 = coefficients[0]
                A_pi_2 = coefficients[1]
                A_pi_3 = coefficients[2]
                A_pi_4 = coefficients[3]
                A_pi_5 = coefficients[4]
                A_pi_6 = coefficients[5]
                A_pi_7 = coefficients[6]
                A_pi_8 = coefficients[7]
                A_pi_9 = coefficients[8]
                A_pi_10 = coefficients[9]
                A_pi_11 = coefficients[10]
                B_pi_1 = coefficients[11]
                B_pi_2 = coefficients[12]
                B_pi_3 = coefficients[13]
                B_pi_4 = coefficients[14]
                B_pi_5 = coefficients[15]
                B_pi_6 = coefficients[16]
                B_pi_7 = coefficients[17]
                B_pi_8 = coefficients[18]
                B_pi_9 = coefficients[19]
                B_pi_10 = coefficients[20]
                B_pi_11 = coefficients[21]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # Definition of the critical length for this section type.
                L_c = L
                
                # This section of code calculates the non-dimensional numbers used for this method.
                L_R = Section_Pressure_Drop.Dimensionless_Numbers.Length_Ratio(L_c, D)
                D_R = Section_Pressure_Drop.Dimensionless_Numbers.Diameter_Ratio(d, D)
                D_R50 = Section_Pressure_Drop.Dimensionless_Numbers.Median_Diameter_Ratio(d_v50, D)
                rho_R = Section_Pressure_Drop.Dimensionless_Numbers.Density_Ratio(rho_g, rho_s)
                epsilon_R = Section_Pressure_Drop.Dimensionless_Numbers.Roughness_Ratio(epsilon, D)
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                P_R = Section_Pressure_Drop.Dimensionless_Numbers.Pressure_Ratio(v_g, rho_g, R_g, T_g)
                
                # This section of code uses the coefficients to find the function inputs for this method.
                L_R_C = A_pi_1 * L_R ** B_pi_1
                D_R_C = A_pi_2 * D_R ** B_pi_2
                D_R50_C = A_pi_3 * D_R50 ** B_pi_3
                rho_R_C = A_pi_4 * rho_R ** B_pi_4
                epsilon_R_C = A_pi_5 * epsilon_R ** B_pi_5
                Re_p_C = A_pi_6 * Re_p ** B_pi_6
                Fr_C = A_pi_7 * Fr ** B_pi_7
                R_s_C = A_pi_8 * R_s ** B_pi_8
                P_R_C = A_pi_9 * P_R ** B_pi_9
                Gas_Split_C = A_pi_10 * Gas_Split ** B_pi_10
                Solids_Split_C = A_pi_11 * Solids_Split ** B_pi_11
                
                # Pull function out of extra args
                try:
                    [func, orientation] = extra_args
                except:
                    [func, orientation, L_t] = extra_args
                
                # This section of code calculates the dimensionless pressure drop based on the dimensionless numbers and the passed function.
                # P_S_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C, Z_HP, Z_VUP, Z_VDP, Z_BHH, Z_BHU, Z_BHD, Z_BUH, Z_BDH, Z_AS)
                P_S_star = func(L_R_C, D_R_C, D_R50_C, rho_R_C, epsilon_R_C, Re_p_C, Fr_C, R_s_C, P_R_C, Gas_Split_C, Solids_Split_C)
                
                # Convert dimensionless pressure drop to pressure drop.
                P_S = (m_g ** 2.) / (D ** 4. * rho_g) * P_S_star
                
            elif method == 7:
                # This is the calculation code for Method 7 based on the best method for acceleration of solids for each independent flow rate.
                
                # Pull data from extra arguments.
                Acceleration_Of_Solids_Method = extra_args[0]
                Acceleration_Of_Solids_Coefficients = extra_args[1]
                m_g_1 = extra_args[2]
                m_s_1 = extra_args[3]
                m_g_2 = extra_args[4]
                m_s_2 = extra_args[5]
                
                P_S_1 = Section_Pressure_Drop.Dilute_Phase.Acceleration_Of_Solids(Acceleration_Of_Solids_Method, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g_1, m_s_1, R_g, T_g, mu_g, Acceleration_Of_Solids_Coefficients, Function_Dict, extra_args=None)
                P_S_2 = Section_Pressure_Drop.Dilute_Phase.Acceleration_Of_Solids(Acceleration_Of_Solids_Method, D, d, d_v50, rho_g, rho_s, epsilon, g, m_g_2, m_s_2, R_g, T_g, mu_g, Acceleration_Of_Solids_Coefficients, Function_Dict, extra_args=None)
                
                P_S = (P_S_1, P_S_2)
            
            elif method == 8:
                # Method for Mehta et al. acceleration of solids with new coefficients.
                
                # Set outlet diameter
                # D = 0.09718
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                C_g = coefficients[2]
                C_s = coefficients[3]
                
                # Pull data from extra arguments.
                m_g_1 = extra_args[0]
                m_s_1 = extra_args[1]
                m_g_2 = extra_args[2]
                m_s_2 = extra_args[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g_1 = m_g_1 / (rho_g * A_c)
                v_g_2 = m_g_2 / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s_1 = (v_g_1 - v_t ** a_v_s) * D ** b_v_s
                v_s_2 = (v_g_2 - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the pressure drop due to acceleration
                P_S_1 =  C_g * m_g_1 * v_g_1 / (1055. * A_c) + C_s * m_s_1 * v_s_1 / (527.5 * A_c)
                P_S_2 =  C_g * m_g_2 * v_g_2 / (1055. * A_c) + C_s * m_s_2 * v_s_2 / (527.5 * A_c)
                
                P_S = (P_S_1, P_S_2)
                
            elif method == 9:
                # Method for Tripathi et al. acceleration of solids with new coefficients.
                
                # Set outlet diameter
                # D = 0.09718
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                C_s = coefficients[2]
                
                # Pull data from extra arguments.
                m_g_1 = extra_args[0]
                m_s_1 = extra_args[1]
                m_g_2 = extra_args[2]
                m_s_2 = extra_args[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g_1 = m_g_1 / (rho_g * A_c)
                v_g_2 = m_g_2 / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s_1 = (v_g_1 - v_t ** a_v_s) * D ** b_v_s
                v_s_2 = (v_g_2 - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the pressure drop due to acceleration
                P_S_1 =  C_s * m_s_1 * v_s_1 / A_c
                P_S_2 =  C_s * m_s_2 * v_s_2 / A_c
                
                P_S = (P_S_1, P_S_2)
                
            elif method == 10:
                # Method for Mehta et al. acceleration of solids with new coefficients and reduced diameter.
                
                # Set outlet diameter
                D = 0.09718
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                C_g = coefficients[2]
                C_s = coefficients[3]
                
                # Pull data from extra arguments.
                m_g_1 = extra_args[0]
                m_s_1 = extra_args[1]
                m_g_2 = extra_args[2]
                m_s_2 = extra_args[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g_1 = m_g_1 / (rho_g * A_c)
                v_g_2 = m_g_2 / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s_1 = (v_g_1 - v_t ** a_v_s) * D ** b_v_s
                v_s_2 = (v_g_2 - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the pressure drop due to acceleration
                P_S_1 =  C_g * m_g_1 * v_g_1 / (1055. * A_c) + C_s * m_s_1 * v_s_1 / (527.5 * A_c)
                P_S_2 =  C_g * m_g_2 * v_g_2 / (1055. * A_c) + C_s * m_s_2 * v_s_2 / (527.5 * A_c)
                
                P_S = (P_S_1, P_S_2)
                
            elif method == 11:
                # Method for Tripathi et al. acceleration of solids with new coefficients and reduced diameter.
                
                # Set outlet diameter
                D = 0.09718
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                C_s = coefficients[2]
                
                # Pull data from extra arguments.
                m_g_1 = extra_args[0]
                m_s_1 = extra_args[1]
                m_g_2 = extra_args[2]
                m_s_2 = extra_args[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g_1 = m_g_1 / (rho_g * A_c)
                v_g_2 = m_g_2 / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s_1 = (v_g_1 - v_t ** a_v_s) * D ** b_v_s
                v_s_2 = (v_g_2 - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the pressure drop due to acceleration
                P_S_1 =  C_s * m_s_1 * v_s_1 / A_c
                P_S_2 =  C_s * m_s_2 * v_s_2 / A_c
                
                P_S = (P_S_1, P_S_2)
                
            elif method == 12:
            
                # Method for split flow treated as two separate bends. Based on U-H bend Method 9 - Mallick with a custom found bend loss coefficient.
                
                # Set outlet diameter
                # D = 0.09718
                
                # This section of code extracts the coefficients used in this method.
                B_L = coefficients[0]
                
                # Pull data from extra arguments.
                m_g_1 = extra_args[0]
                m_s_1 = extra_args[1]
                m_g_2 = extra_args[2]
                m_s_2 = extra_args[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g_1 = m_g_1 / (rho_g * A_c)
                v_g_2 = m_g_2 / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s_1 = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g_1, m_s_1)
                R_s_2 = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g_2, m_s_2)
                
                # This section of code calculates the velocity and density at the bend outlet.
                if (m_g_1, A_c, B_L, R_s_1, rho_g, R_g, T_g) in Function_Dict['v_fao_chambers_fun']:
                    v_fao_1 = Function_Dict['v_fao_chambers_fun'][(m_g_1, A_c, B_L, R_s_1, rho_g, R_g, T_g)]
                else:
                    v_fao_1 = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_chambers_fun, v_g_1, args=(m_g_1, A_c, B_L, R_s_1, rho_g, R_g, T_g), method="lm").x[0]
                    Function_Dict['v_fao_chambers_fun'][(m_g_1, A_c, B_L, R_s_1, rho_g, R_g, T_g)] = v_fao_1
                rho_go_1 = m_g_1 / (v_fao_1 * A_c)
                
                if (m_g_2, A_c, B_L, R_s_2, rho_g, R_g, T_g) in Function_Dict['v_fao_chambers_fun']:
                    v_fao_2 = Function_Dict['v_fao_chambers_fun'][(m_g_2, A_c, B_L, R_s_2, rho_g, R_g, T_g)]
                else:
                    v_fao_2 = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_chambers_fun, v_g_2, args=(m_g_2, A_c, B_L, R_s_2, rho_g, R_g, T_g), method="lm").x[0]
                    Function_Dict['v_fao_chambers_fun'][(m_g_2, A_c, B_L, R_s_2, rho_g, R_g, T_g)] = v_fao_2
                rho_go_2 = m_g_2 / (v_fao_2 * A_c)
                
                # The total pressure drop is found next.
                P_S_1 = B_L * (1. + R_s_1) * (rho_go_1 * v_fao_1 ** 2.) / 2.
                P_S_2 = B_L * (1. + R_s_2) * (rho_go_2 * v_fao_2 ** 2.) / 2.
                
                P_S = (P_S_1, P_S_2)
                
            elif method == 13:
            
                # Method for split flow treated as two separate bends. Based on U-H bend Method 9 - Mallick with a custom found bend loss coefficient with reduced outlet diameter.
                
                # Set outlet diameter
                D = 0.09718
                
                # This section of code extracts the coefficients used in this method.
                B_L = coefficients[0]
                
                # Pull data from extra arguments.
                m_g_1 = extra_args[0]
                m_s_1 = extra_args[1]
                m_g_2 = extra_args[2]
                m_s_2 = extra_args[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g_1 = m_g_1 / (rho_g * A_c)
                v_g_2 = m_g_2 / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s_1 = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g_1, m_s_1)
                R_s_2 = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g_2, m_s_2)
                
                # This section of code calculates the velocity and density at the bend outlet.
                if (m_g_1, A_c, B_L, R_s_1, rho_g, R_g, T_g) in Function_Dict['v_fao_chambers_fun']:
                    v_fao_1 = Function_Dict['v_fao_chambers_fun'][(m_g_1, A_c, B_L, R_s_1, rho_g, R_g, T_g)]
                else:
                    v_fao_1 = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_chambers_fun, v_g_1, args=(m_g_1, A_c, B_L, R_s_1, rho_g, R_g, T_g), method="lm").x[0]
                    Function_Dict['v_fao_chambers_fun'][(m_g_1, A_c, B_L, R_s_1, rho_g, R_g, T_g)] = v_fao_1
                rho_go_1 = m_g_1 / (v_fao_1 * A_c)
                
                if (m_g_2, A_c, B_L, R_s_2, rho_g, R_g, T_g) in Function_Dict['v_fao_chambers_fun']:
                    v_fao_2 = Function_Dict['v_fao_chambers_fun'][(m_g_2, A_c, B_L, R_s_2, rho_g, R_g, T_g)]
                else:
                    v_fao_2 = root(Section_Pressure_Drop.Dilute_Phase.Iteration_Functions().v_fao_chambers_fun, v_g_2, args=(m_g_2, A_c, B_L, R_s_2, rho_g, R_g, T_g), method="lm").x[0]
                    Function_Dict['v_fao_chambers_fun'][(m_g_2, A_c, B_L, R_s_2, rho_g, R_g, T_g)] = v_fao_2
                rho_go_2 = m_g_2 / (v_fao_2 * A_c)
                
                # The total pressure drop is found next.
                P_S_1 = B_L * (1. + R_s_1) * (rho_go_1 * v_fao_1 ** 2.) / 2.
                P_S_2 = B_L * (1. + R_s_2) * (rho_go_2 * v_fao_2 ** 2.) / 2.
                
                P_S = (P_S_1, P_S_2)
                
            return P_S
        
    class Dense_Phase:
        # This class contains the pressure drop methods for dense phase systems.
        
        class Iteration_Functions:
            # This class contains functions that need to be solved by iteration.
            @staticmethod
            @jit(nopython=True)
            def f_g_fun(f_g, Re, epsilon, D):
                # Function for calculating friction factor.
                fun = -2. * np.log10(epsilon / (3.7 * D) + 2.51 / (Re * (f_g) ** 0.5)) - 1. / (f_g) ** 0.5
                for f in f_g:
                    if f < 0.:
                        fun = np.array([0.])
                return fun
                
            @staticmethod
            @jit(nopython=True)
            def P_o_fun(P_o, P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s):
                # Function for calculating outlet pressure.
                
                P_H_1 = P_i - P_o
                P_H_2 = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.) + P_g
                
                return P_H_1 - P_H_2
                
            @staticmethod
            def v_s_fun(v_s, v_g, v_t, D, g, rho_s, m_s, A_c):
                # Function for calculating solids velocity.
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                for a in alpha:
                    if a < 0.:
                        fun = np.array([1000.])
                    else:
                        f_p_star = 0.117 * ((1. - a) * v_g / (g * D) ** 0.5 ) ** -1.15 * (1. - a) / a ** 3.
                        
                        interior = f_p_star * v_s ** 2. / (2. * g * D) * a ** 4.7
                        
                        if interior < 0.:
                            fun = np.array([1000.])
                        else:
                            fun = v_g - v_t * () ** 0.5 - v_s
                    
                if v_g < 0.:
                    fun = np.array([1000.])
                
                # print('{} - {} - {} - {}'.format(alpha, fun, len(fun), type(fun)))
                return fun
                                
            @staticmethod
            @jit(nopython=True)
            def P_o_watson_fun(P_o, rho_g, R_g, T_g, g, L, D, m_s, m_g, K_w):
                # Function for calculating outlet gas pressure.
                P_i_1 = rho_g * R_g * T_g
                P_i_2 = P_o * np.exp(g * L * D * np.sqrt(np.pi * m_s) / (2. * K_w * R_g * T_g * m_g))
                
                return P_i_1 - P_i_2
        
        @staticmethod
        def Horizontal_Pipe_Sections(method, L, D, d, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, T_g, R_g, coefficients, Function_Dict):
            # This class contains the pressure drop calculation methods for horizontal pipe sections.
            
            if method == 1:
                # This is the calculation code for Method 1 - Total pressure: Mi:
                
                # This section of code extracts the coefficients used in this method.
                phi = coefficients[0]
                phi_w = coefficients[1]
                phi_s = coefficients[2]
                k_gm = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the slug velocity and slug Froude number.
                v_gm = rho_s * g * np.tan(phi_w) * epsilon ** 3. * d ** 2. / (36. * k_gm * (1. - epsilon) * mu_g)
                v_sl = 105. * epsilon * d / D * (np.tan(phi_w) / np.tan(phi)) ** (1. / 3.) * (v_g - v_gm)
                Fr_sl = Section_Pressure_Drop.Dimensionless_Numbers.Slug_Froude_Number(v_sl, g, D)
                
                # This section calculates the  omega angle.
                omega = np.arcsin(np.sin(phi_w) / np.sin(phi_s))
                
                # This section calculates the coefficient of wall friction.
                mu_w = np.tan(phi_w)
                
                # This section calculates the stress transmission coefficient.
                lambda_st = (1. - np.sin(phi_s) * np.cos(omega - phi_w)) / (1. + np.sin(phi_s) * np.cos(omega - phi_w))
                
                # The total pressure drop is found next.
                P_H = (1. + 1.084 * lambda_st * Fr_sl ** 0.5 + 0.542 * Fr_sl ** -0.5) * 2. * g * mu_w * m_s * L / (A_c * v_sl)
                
            elif method == 2:
                # This is the calculation code for Method 2 - Total pressure: Shaul and Kalman:
                
                # This section of code extracts the coefficients used in this method.
                phi = coefficients[0]
                phi_w = coefficients[1]
                phi_s = coefficients[2]
                k_gm = coefficients[3]
                a_S = coefficients[4]
                b_S = coefficients[5]
                c_S = coefficients[6]
                f_S = coefficients[7]
                D_R = coefficients[8]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the slug velocity and slug Froude number.
                v_gm = rho_s * g * np.tan(phi_w) * epsilon ** 3. * d ** 2. / (36. * k_gm * (1. - epsilon) * mu_g)
                v_sl = 105. * epsilon * d / D * (np.tan(phi_w) / np.tan(phi)) ** (1. / 3.) * (v_g - v_gm)
                Fr_sl = Section_Pressure_Drop.Dimensionless_Numbers.Slug_Froude_Number(v_sl, g, D)
                
                # This section calculates the coefficient of wall friction.
                mu_w = np.tan(phi_w)
                
                # This section calculates the stress transmission coefficient.
                lambda_st = (a_S * phi_s + f_S) / (D / D_R) * (L / D) ** (b_S * (phi_s - phi_w) - c_S)
                
                # The total pressure drop is found next.
                P_H = (1. + 1.084 * lambda_st * Fr_sl ** 0.5 + 0.542 * Fr_sl ** -0.5) * 2. * g * mu_w * m_s * L / (A_c * v_sl)
            
            elif method == 3:
                # This is the calculation code for Method 3 - Total pressure: Muschelknautz and Krambrock (Solids velocity: Naveh et al.):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85):
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
            
            elif method == 4:
                # This is the calculation code for Method 4 - Total pressure: Muschelknautz and Krambrock (Solids velocity: Klinzing et al.):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
            
            elif method == 5:
                # This is the calculation code for Method 5 - Total pressure: Muschelknautz and Krambrock (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                a_v_s = coefficients[1]
                b_v_s = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 6:
                # This is the calculation code for Method 6 - Total pressure: Muschelknautz and Krambrock (Solids velocity: Yang):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 7:
                # This is the calculation code for Method 7 - Total pressure: Muschelknautz and Krambrock (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                R_v = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 8:
                # This is the calculation code for Method 8 - Total pressure: Muschelknautz and Krambrock (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 9:
                # This is the calculation code for Method 9 - Total pressure: Jones and Williams:
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = 83. * R_s ** -0.9 * Fr
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 10:
                # This is the calculation code for Method 10 - Total pressure: Mallick and Wypych (Solids velocity: Naveh et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
            
            elif method == 11:
                # This is the calculation code for Method 11 - Total pressure: Mallick and Wypych (Solids velocity: Klinzing et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
            
            elif method == 12:
                # This is the calculation code for Method 12 - Total pressure: Mallick and Wypych (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                a_v_s = coefficients[2]
                b_v_s = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 13:
                # This is the calculation code for Method 13 - Total pressure: Mallick and Wypych (Solids velocity: Yang):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 14:
                # This is the calculation code for Method 14 - Total pressure: Mallick and Wypych (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                R_v = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 15:
                # This is the calculation code for Method 15 - Total pressure: Mallick and Wypych (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 16:
                # This is the calculation code for Method 16 - Total pressure: Guan Generalized:
                
                # This section of code extracts the coefficients used in this method.
                a_G = coefficients[0]
                b_G = coefficients[1]
                c_G = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = a_G * Fr ** b_G * (d / D) ** c_G
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 17:
                # This is the calculation code for Method 17 - Total pressure: Guan:
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = 2.990 * 10. ** -3. * Fr ** -0.946 * (d / D) ** -0.637
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 18:
                # This is the calculation code for Method 18 - Momentum balance (Solids velocity: Naveh et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the ash volume fraction.
                m_star = ((1. - alpha) * rho_s * v_s) / (alpha * rho_g * v_g)
                
                # This section of code calculates the mixture friction coefficient.
                f_mix = f_g + epsilon * (d / D) ** 0.1 * Re_p ** 0.4 * Fr ** -0.5 * rho_s / rho_g * m_star
                
                # The total pressure drop is found next.
                P_H = 4. * f_mix * rho_g * v_g ** 2. / 2. * L / D

            elif method == 19:
                # This is the calculation code for Method 19 - Momentum balance (Solids velocity: Klinzing et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the ash volume fraction.
                m_star = ((1. - alpha) * rho_s * v_s) / (alpha * rho_g * v_g)
                
                # This section of code calculates the mixture friction coefficient.
                f_mix = f_g + epsilon * (d / D) ** 0.1 * Re_p ** 0.4 * Fr ** -0.5 * rho_s / rho_g * m_star
                
                # The total pressure drop is found next.
                P_H = 4. * f_mix * rho_g * v_g ** 2. / 2. * L / D
                
            elif method == 20:
                # This is the calculation code for Method 20 - Momentum balance (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the ash volume fraction.
                m_star = ((1. - alpha) * rho_s * v_s) / (alpha * rho_g * v_g)
                
                # This section of code calculates the mixture friction coefficient.
                f_mix = f_g + epsilon * (d / D) ** 0.1 * Re_p ** 0.4 * Fr ** -0.5 * rho_s / rho_g * m_star
                
                # The total pressure drop is found next.
                P_H = 4. * f_mix * rho_g * v_g ** 2. / 2. * L / D
                
            elif method == 21:
                # This is the calculation code for Method 21 - Momentum balance (Solids velocity: Yang):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]              
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the ash volume fraction.
                m_star = ((1. - alpha) * rho_s * v_s) / (alpha * rho_g * v_g)
                
                # This section of code calculates the mixture friction coefficient.
                f_mix = f_g + epsilon * (d / D) ** 0.1 * Re_p ** 0.4 * Fr ** -0.5 * rho_s / rho_g * m_star
                
                # The total pressure drop is found next.
                P_H = 4. * f_mix * rho_g * v_g ** 2. / 2. * L / D
                
            elif method == 22:
                # This is the calculation code for Method 22 - Momentum balance (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                R_v = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the ash volume fraction.
                m_star = ((1. - alpha) * rho_s * v_s) / (alpha * rho_g * v_g)
                
                # This section of code calculates the mixture friction coefficient.
                f_mix = f_g + epsilon * (d / D) ** 0.1 * Re_p ** 0.4 * Fr ** -0.5 * rho_s / rho_g * m_star
                
                # The total pressure drop is found next.
                P_H = 4. * f_mix * rho_g * v_g ** 2. / 2. * L / D
                
            elif method == 23:
                # This is the calculation code for Method 23 - Momentum balance (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the ash volume fraction.
                m_star = ((1. - alpha) * rho_s * v_s) / (alpha * rho_g * v_g)
                
                # This section of code calculates the mixture friction coefficient.
                f_mix = f_g + epsilon * (d / D) ** 0.1 * Re_p ** 0.4 * Fr ** -0.5 * rho_s / rho_g * m_star
                
                # The total pressure drop is found next.
                P_H = 4. * f_mix * rho_g * v_g ** 2. / 2. * L / D
                
            elif method == 24:
                # This is the calculation code for Method 24 - Other methods: Laouar and Molodtsof:
                
                # This section of code extracts the coefficients used in this method.
                a_L_1 = coefficients[0]
                a_L_2 = coefficients[1]
                b_L_1 = coefficients[2]
                b_L_2 = coefficients[3]
                c_L_1 = coefficients[4]
                d_L_1 = coefficients[5]
                d_L_2 = coefficients[6]
                d_L_3 = coefficients[7]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the volumetric gas flow rates.
                V_g = m_g / rho_g
                
                # This section of code calculates the coefficients.
                b_L = - b_L_1 * (m_s - b_L_2)
                c_L = m_s - c_L_1
                d_L = d_L_1 * (m_s + d_L_2) * (m_s + d_L_3)
                a_L = a_L_1 * d_L - a_L_2
                
                # The total pressure drop is found next.
                P_H = ((a_L - b_L * V_g) / (c_L * V_g + d_L)) * L
                                
            elif method == 25:
                # This is the calculation code for Method 25 - Dense phase: Michaelides:
                
                # This section of code extracts the coefficients used in this method.
                K_p = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = K_p * 1. / Fr ** 0.5
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 26:
                # This is the calculation code for Method 26 - Dense phase: Setia et al.:
                
                # This section of code extracts the coefficients used in this method.
                a_s = coefficients[0]
                b_s = coefficients[0]
                c_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = a_s * R_s ** b_s * Fr ** c_s
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 27:
                # This is the calculation code for Method 27 - Dense phase: Mason et al.:
                
                # This section of code extracts the coefficients used in this method.
                f_pM = coefficients[0]
                a_M = coefficients[1]
                b_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                P_s = f_pM * L * R_s ** a_M * Fr ** b_M
                
                # The total pressure drop is found next.
                P_H = P_g + P_s
                
            elif method == 28:
                # This is the calculation code for Method 28 - Dense phase: Mehta et al. (Solids velocity: Naveh et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 29:
                # This is the calculation code for Method 29 - Total pressure: Mehta et al. (Solids velocity: Klinzing et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 30:
                # This is the calculation code for Method 30 - Total pressure: Mehta et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                f_me = coefficients[2]
                a_me = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 31:
                # This is the calculation code for Method 31 - Total pressure: Mehta et al. (Solids velocity: Yang):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 32:
                # This is the calculation code for Method 32 - Total pressure: Mehta et al. (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                R_v = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                try:
                    P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                except:
                    P_H = 0.
                
            elif method == 33:
                # This is the calculation code for Method 33 - Total pressure: Mehta et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_H = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                
            elif method == 34:
                # This is the calculation code for Method 34 - Dense phase: Pfeffer et al. generalized (Solids velocity: Naveh et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            elif method == 35:
                # This is the calculation code for Method 35 - Dense phase: Pfeffer et al. generalized (Solids velocity: Klinzing et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 36:
                # This is the calculation code for Method 36 - Dense phase: Pfeffer et al. generalized:
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                n_e = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 37:
                # This is the calculation code for Method 37 - Dense phase: Pfeffer et al. generalized (Solids velocity: Yang):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                
            elif method == 38:
                # This is the calculation code for Method 38 - Dense phase: Pfeffer et al. generalized (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                R_v = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                                
            elif method == 39:
                # This is the calculation code for Method 39 - Dense phase: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_H = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            return P_H
            
        @staticmethod
        def Vertical_Pipe_Sections(method, L, D, d, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, T_g, R_g, Direction, coefficients, Function_Dict):
            # This class contains the pressure drop calculation methods for vertical pipe sections.
            
            if method == 1:
                # This is the calculation code for Method 1 - Total pressure: Rabinovich et al.:
                
                # This section of code extracts the coefficients used in this method.
                phi = coefficients[0]
                phi_w = coefficients[1]
                phi_s = coefficients[2]
                k_gm = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the slug velocity and slug Froude number.
                v_gm = rho_s * g * np.tan(phi_w) * epsilon ** 3. * d ** 2. / (36. * k_gm * (1. - epsilon) * mu_g)
                v_sl = 105. * epsilon * d / D * (np.tan(phi_w) / np.tan(phi)) ** (1. / 3.) * (v_g - v_gm)
                Fr_sl = Section_Pressure_Drop.Dimensionless_Numbers.Slug_Froude_Number(v_sl, g, D)
                
                # This section calculates the coefficient of wall friction.
                mu_w = np.tan(phi_w)
                
                # This section calculates the stress transmission coefficient.
                lambda_st = (-4. * phi_s + 4.32) * (L / D) ** -0.6
                
                # The total pressure drop is found next.
                P_V = (1. + 1.084 * lambda_st * Fr_sl ** 0.5 + 0.542 * Fr_sl ** -0.5) * 2. * g * mu_w * m_s * L / (A_c * v_sl)
                
            elif method == 2:
                # This is the calculation code for Method 2 - Total pressure: Shaul and Kalman:
                
                # This section of code extracts the coefficients used in this method.
                phi = coefficients[0]
                phi_w = coefficients[1]
                phi_s = coefficients[2]
                k_gm = coefficients[3]
                a_S = coefficients[4]
                b_S = coefficients[5]
                c_S = coefficients[6]
                f_S = coefficients[7]
                D_R = coefficients[8]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the slug velocity and slug Froude number.
                v_gm = rho_s * g * np.tan(phi_w) * epsilon ** 3. * d ** 2. / (36. * k_gm * (1. - epsilon) * mu_g)
                v_sl = 105. * epsilon * d / D * (np.tan(phi_w) / np.tan(phi)) ** (1. / 3.) * (v_g - v_gm)
                Fr_sl = Section_Pressure_Drop.Dimensionless_Numbers.Slug_Froude_Number(v_sl, g, D)
                
                # This section calculates the coefficient of wall friction.
                mu_w = np.tan(phi_w)
                
                # This section calculates the stress transmission coefficient.
                lambda_st = (a_S * phi_s + f_S) / (D / D_R) * (L / D) ** (b_S * (phi_s - phi_w) - c_S)
                
                # The total pressure drop is found next.
                P_V = (1. + 1.084 * lambda_st * Fr_sl ** 0.5 + 0.542 * Fr_sl ** -0.5) * 2. * g * mu_w * m_s * L / (A_c * v_sl)
            
            elif method == 3:
                # This is the calculation code for Method 3 - Total pressure: Muschelknautz and Krambrock (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # The total pressure drop is found next.
                
                P_V = P_g + P_s
                
            elif method == 4:
                # This is the calculation code for Method 4 - Total pressure: Muschelknautz and Krambrock (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 5:
                # This is the calculation code for Method 5 - Total pressure: Mallick and Wypych (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 6:
                # This is the calculation code for Method 6 - Total pressure: Mallick and Wypych (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 7:
                # This is the calculation code for Method 7 - Total pressure: Guan Generalized:
                
                # This section of code extracts the coefficients used in this method.
                a_G = coefficients[0]
                b_G = coefficients[1]
                c_G = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = a_G * Fr ** b_G * (d / D) ** c_G
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 8:
                # This is the calculation code for Method 8 - Total pressure: Guan:
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = 2.047 * 10. ** -4. * Fr ** -1.095 * (d / D) ** -1.233
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 9:
                # This is the calculation code for Method 9 - Total pressure: Sharma (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # The total pressure drop is found next.
                P_V = R_s * rho_g * g * L * v_g / v_s
                
            elif method == 10:
                # This is the calculation code for Method 10 - Total pressure: Sharma (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # The total pressure drop is found next.
                P_V = R_s * rho_g * g * L * v_g / v_s
                
            elif method == 11:
                # This is the calculation code for Method 11 - Momentum balance (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the mixture density.
                rho_mix = alpha * rho_g - (1. - alpha) * rho_s
                
                # This section of code calculates the ash volume fraction.
                m_star = ((1. - alpha) * rho_s * v_s) / (alpha * rho_g * v_g)
                
                # This section of code calculates the mixture friction coefficient.
                f_mix = f_g + epsilon * (d / D) ** 0.1 * Re_p ** 0.4 * Fr ** -0.5 * rho_s / rho_g * m_star
                
                # The total pressure drop is found next.
                P_V = 4. * f_mix * rho_g * v_g ** 2. / 2. * L / D + rho_mix * g * L
                
            elif method == 12:
                # This is the calculation code for Method 12 - Momentum balance (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the mixture density.
                rho_mix = alpha * rho_g - (1. - alpha) * rho_s
                
                # This section of code calculates the ash volume fraction.
                m_star = ((1. - alpha) * rho_s * v_s) / (alpha * rho_g * v_g)
                
                # This section of code calculates the mixture friction coefficient.
                f_mix = f_g + epsilon * (d / D) ** 0.1 * Re_p ** 0.4 * Fr ** -0.5 * rho_s / rho_g * m_star
                
                # The total pressure drop is found next.
                P_V = 4. * f_mix * rho_g * v_g ** 2. / 2. * L / D + rho_mix * g * L
                
            elif method == 13:
                # This is the calculation code for Method 13 - Other methods: Watson et al.:
                
                # This section of code extracts the coefficients used in this method.
                K_w = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the outlet pressure.
                if (rho_g, R_g, T_g, g, L, D, m_s, m_g, K_w) in Function_Dict['P_o_watson_fun']:
                    P_o = Function_Dict['P_o_watson_fun'][(rho_g, R_g, T_g, g, L, D, m_s, m_g, K_w)]
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_watson_fun, 10000., args=(rho_g, R_g, T_g, g, L, D, m_s, m_g, K_w), method="lm").x[0]
                    Function_Dict['P_o_watson_fun'][(rho_g, R_g, T_g, g, L, D, m_s, m_g, K_w)] = P_o
                
                # This section of code calculates the inlet pressure.
                P_i = P_o * np.exp(g * L * D * np.sqrt(np.pi * m_s) / (2. * K_w * R_g * T_g * m_g))
                                
                # The total pressure drop is found next.
                P_V = P_i - P_o
                
            elif method == 14:
                # This is the calculation code for Method 14 - Other methods: Laouar and Molodtsof:
                
                # This section of code extracts the coefficients used in this method.
                a_L_1 = coefficients[0]
                a_L_2 = coefficients[1]
                b_L_1 = coefficients[2]
                b_L_2 = coefficients[3]
                c_L_1 = coefficients[4]
                d_L_1 = coefficients[5]
                d_L_2 = coefficients[6]
                d_L_3 = coefficients[7]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the volumetric gas flow rates.
                V_g = m_g / rho_g
                
                # This section of code calculates the coefficients.
                b_L = - b_L_1 * (m_s - b_L_2)
                c_L = m_s - c_L_1
                d_L = d_L_1 * (m_s + d_L_2) * (m_s + d_L_3)
                a_L = a_L_1 * d_L - a_L_2
                
                # The total pressure drop is found next.
                P_V = ((a_L - b_L * V_g) / (c_L * V_g + d_L)) * L
            
            elif method == 15:
                # This is the calculation code for Method 15 - Combined: Rabinovich et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                phi = coefficients[0]
                phi_w = coefficients[1]
                phi_s = coefficients[2]
                k_gm = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates the slug velocity and slug Froude number.
                v_gm = rho_s * g * np.tan(phi_w) * epsilon ** 3. * d ** 2. / (36. * k_gm * (1. - epsilon) * mu_g)
                v_sl = 105. * epsilon * d / D * (np.tan(phi_w) / np.tan(phi)) ** (1. / 3.) * (v_g - v_gm)
                Fr_sl = Section_Pressure_Drop.Dimensionless_Numbers.Slug_Froude_Number(v_sl, g, D)
                
                # This section calculates the coefficient of wall friction.
                mu_w = np.tan(phi_w)
                
                # This section calculates the stress transmission coefficient.
                lambda_st = (-4. * phi_s + 4.32) * (L / D) ** -0.6
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = (1. + 1.084 * lambda_st * Fr_sl ** 0.5 + 0.542 * Fr_sl ** -0.5) * 2. * g * mu_w * m_s * L / (A_c * v_sl)
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 16:
                # This is the calculation code for Method 16 - Combined: Rabinovich et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                phi = coefficients[0]
                phi_w = coefficients[1]
                phi_s = coefficients[2]
                k_gm = coefficients[3]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates the slug velocity and slug Froude number.
                v_gm = rho_s * g * np.tan(phi_w) * epsilon ** 3. * d ** 2. / (36. * k_gm * (1. - epsilon) * mu_g)
                v_sl = 105. * epsilon * d / D * (np.tan(phi_w) / np.tan(phi)) ** (1. / 3.) * (v_g - v_gm)
                Fr_sl = Section_Pressure_Drop.Dimensionless_Numbers.Slug_Froude_Number(v_sl, g, D)
                
                # This section calculates the coefficient of wall friction.
                mu_w = np.tan(phi_w)
                
                # This section calculates the stress transmission coefficient.
                lambda_st = (-4. * phi_s + 4.32) * (L / D) ** -0.6
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = (1. + 1.084 * lambda_st * Fr_sl ** 0.5 + 0.542 * Fr_sl ** -0.5) * 2. * g * mu_w * m_s * L / (A_c * v_sl)
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 17:
                # This is the calculation code for Method 17 - Combined: Shaul and Kalman (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                phi = coefficients[0]
                phi_w = coefficients[1]
                phi_s = coefficients[2]
                k_gm = coefficients[3]
                a_S = coefficients[4]
                b_S = coefficients[5]
                c_S = coefficients[6]
                f_S = coefficients[7]
                D_R = coefficients[8]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates the slug velocity and slug Froude number.
                v_gm = rho_s * g * np.tan(phi_w) * epsilon ** 3. * d ** 2. / (36. * k_gm * (1. - epsilon) * mu_g)
                v_sl = 105. * epsilon * d / D * (np.tan(phi_w) / np.tan(phi)) ** (1. / 3.) * (v_g - v_gm)
                Fr_sl = Section_Pressure_Drop.Dimensionless_Numbers.Slug_Froude_Number(v_sl, g, D)
                
                # This section calculates the coefficient of wall friction.
                mu_w = np.tan(phi_w)
                
                # This section calculates the stress transmission coefficient.
                lambda_st = (a_S * phi_s + f_S) / (D / D_R) * (L / D) ** (b_S * (phi_s - phi_w) - c_S)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = (1. + 1.084 * lambda_st * Fr_sl ** 0.5 + 0.542 * Fr_sl ** -0.5) * 2. * g * mu_w * m_s * L / (A_c * v_sl)
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
            
            elif method == 18:
                # This is the calculation code for Method 18 - Combined: Shaul and Kalman (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                phi = coefficients[0]
                phi_w = coefficients[1]
                phi_s = coefficients[2]
                k_gm = coefficients[3]
                a_S = coefficients[4]
                b_S = coefficients[5]
                c_S = coefficients[6]
                f_S = coefficients[7]
                D_R = coefficients[8]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates the slug velocity and slug Froude number.
                v_gm = rho_s * g * np.tan(phi_w) * epsilon ** 3. * d ** 2. / (36. * k_gm * (1. - epsilon) * mu_g)
                v_sl = 105. * epsilon * d / D * (np.tan(phi_w) / np.tan(phi)) ** (1. / 3.) * (v_g - v_gm)
                Fr_sl = Section_Pressure_Drop.Dimensionless_Numbers.Slug_Froude_Number(v_sl, g, D)
                
                # This section calculates the coefficient of wall friction.
                mu_w = np.tan(phi_w)
                
                # This section calculates the stress transmission coefficient.
                lambda_st = (a_S * phi_s + f_S) / (D / D_R) * (L / D) ** (b_S * (phi_s - phi_w) - c_S)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = (1. + 1.084 * lambda_st * Fr_sl ** 0.5 + 0.542 * Fr_sl ** -0.5) * 2. * g * mu_w * m_s * L / (A_c * v_sl)
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
            
            elif method == 19:
                # This is the calculation code for Method 19 - Combined: Muschelknautz and Krambrock (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = P_g + P_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 20:
                # This is the calculation code for Method 20 - Combined: Muschelknautz and Krambrock (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                beta_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
            
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the inlet pressure.
                P_i = rho_g * R_g * T_g
                
                # This section of code calculates the outlet pressure.
                if (P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s) in Function_Dict['P_o_fun']:
                    P_o = Function_Dict['P_o_fun']
                else:
                    P_o = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().P_o_fun, P_i, args=(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s), method="lm").x[0]
                    Function_Dict['P_o_fun'][(P_g, P_i, beta_s, R_s, g, L, v_g, R_g, T_g, v_s)] = P_o
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = P_o * (np.exp((beta_s * R_s * g * L * v_g) / (R_g * T_g * v_s)) - 1.)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = P_g + P_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 21:
                # This is the calculation code for Method 21 - Combined: Mallick and Wypych (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = P_g + P_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 22:
                # This is the calculation code for Method 22 - Combined: Mallick and Wypych (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_p_star = coefficients[0]
                beta_fs = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = f_p_star * v_s / v_g + 2. * beta_fs / (v_s / v_g * Fr)
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = P_g + P_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 23:
                # This is the calculation code for Method 23 - Combined: Guan Generalized (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_G = coefficients[0]
                b_G = coefficients[1]
                c_G = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = a_G * Fr ** b_G * (d / D) ** c_G
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = P_g + P_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 24:
                # This is the calculation code for Method 24 - Combined: Guan Generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_G = coefficients[0]
                b_G = coefficients[1]
                c_G = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = a_G * Fr ** b_G * (d / D) ** c_G
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = P_g + P_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 25:
                # This is the calculation code for Method 25 - Combined: Guan (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = 2.047 * 10. ** -4. * Fr ** -1.095 * (d / D) ** -1.233
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = P_g + P_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 26:
                # This is the calculation code for Method 26 - Combined: Guan (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # This section of code calculates the solids friction factor.
                f_p = 2.047 * 10. ** -4. * Fr ** -1.095 * (d / D) ** -1.233
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s = R_s * f_p * rho_g * v_g ** 2. * L / (2. * D)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = P_g + P_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 27:
                # This is the calculation code for Method 27 - Combined: Sharma (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = R_s * rho_g * g * L * v_g / v_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 28:
                # This is the calculation code for Method 28 - Combined: Sharma (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = R_s * rho_g * g * L * v_g / v_s
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
                
            elif method == 29:
                # This is the calculation code for Method 29 - Combined: Laouar and Molodtsof (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_L_1 = coefficients[0]
                a_L_2 = coefficients[1]
                b_L_1 = coefficients[2]
                b_L_2 = coefficients[3]
                c_L_1 = coefficients[4]
                d_L_1 = coefficients[5]
                d_L_2 = coefficients[6]
                d_L_3 = coefficients[7]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates the volumetric gas flow rates.
                V_g = m_g / rho_g
                
                # This section of code calculates the coefficients.
                b_L = - b_L_1 * (m_s - b_L_2)
                c_L = m_s - c_L_1
                d_L = d_L_1 * (m_s + d_L_2) * (m_s + d_L_3)
                a_L = a_L_1 * d_L - a_L_2
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = ((a_L - b_L * V_g) / (c_L * V_g + d_L)) * L
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
            
            elif method == 30:
                # This is the calculation code for Method 30 - Combined: Laouar and Molodtsof (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_L_1 = coefficients[0]
                a_L_2 = coefficients[1]
                b_L_1 = coefficients[2]
                b_L_2 = coefficients[3]
                c_L_1 = coefficients[4]
                d_L_1 = coefficients[5]
                d_L_2 = coefficients[6]
                d_L_3 = coefficients[7]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates the volumetric gas flow rates.
                V_g = m_g / rho_g
                
                # This section of code calculates the coefficients.
                b_L = - b_L_1 * (m_s - b_L_2)
                c_L = m_s - c_L_1
                d_L = d_L_1 * (m_s + d_L_2) * (m_s + d_L_3)
                a_L = a_L_1 * d_L - a_L_2
                
                # This section of code calculates the friction pressure drop of the vertical pipe section.
                P_V_f = ((a_L - b_L * V_g) / (c_L * V_g + d_L)) * L
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section of code calculates the static pressure drop of the vertical pipe section.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                P_V_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = P_V_f + P_V_s
            
            elif method == 31:
                # This is the calculation code for Method 31 - Dense phase: Michaelides:
                
                # This section of code extracts the coefficients used in this method.
                K_p = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = K_p * 1. / Fr ** 0.5
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
                
            elif method == 32:
                # This is the calculation code for Method 32 - Dense phase: Setia et al.:
                
                # This section of code extracts the coefficients used in this method.
                a_s = coefficients[0]
                b_s = coefficients[0]
                c_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                f_p = a_s * R_s ** b_s * Fr ** c_s
                
                P_s = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 33:
                # This is the calculation code for Method 33 - Dense phase: Mason et al.:
                
                # This section of code extracts the coefficients used in this method.
                f_pM = coefficients[0]
                a_M = coefficients[1]
                b_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                
                P_s = f_pM * L * R_s ** a_M * Fr ** b_M
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 34:
                # This is the calculation code for Method 34 - Dense phase: Mehta et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                try:
                    P_V = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
                except:
                    P_V = 0.
                
            elif method == 35:
                # This is the calculation code for Method 35 - Dense phase: Mehta et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # The total pressure drop is found next.
                P_V = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)
            
            elif method == 36:
                # This is the calculation code for Method 36 - Combined: Michaelides (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                K_p = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = K_p * 1. / Fr ** 0.5
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 37:
                # This is the calculation code for Method 37 - Combined: Michaelides (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                K_p = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = K_p * 1. / Fr ** 0.5
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 38:
                # This is the calculation code for Method 38 - Combined: Setia et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_s = coefficients[0]
                b_s = coefficients[0]
                c_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = a_s * R_s ** b_s * Fr ** c_s
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 39:
                # This is the calculation code for Method 39 - Combined: Setia et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_s = coefficients[0]
                b_s = coefficients[0]
                c_s = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                f_p = a_s * R_s ** b_s * Fr ** c_s
                
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = m_s / m_g * L * rho_g * v_g ** 2. / (2. * D) * f_p             
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 40:
                # This is the calculation code for Method 40 - Combined: Mason et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_pM = coefficients[0]
                a_M = coefficients[1]
                b_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = f_pM * L * R_s ** a_M * Fr ** b_M           
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 41:
                # This is the calculation code for Method 41 - Combined: Mason et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_pM = coefficients[0]
                a_M = coefficients[1]
                b_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # P_g: This section of code contains pressure drop calculations due to the conveying of gas alone.
                P_g_s = alpha * rho_g * Direction * L * g
                P_g_f = f_g * rho_g / 2. * v_g ** 2. / D * L
                
                P_g = P_g_s + P_g_f
                
                # P_s: This section of code contains excess pressure drop calculations due to the presence of solids.
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                P_s_f = f_pM * L * R_s ** a_M * Fr ** b_M           
                
                P_s = P_s_s + P_s_f
                
                # The total pressure drop is found next.
                P_V = P_g + P_s
            
            elif method == 42:
                # This is the calculation code for Method 42 - Combined: Mehta et al. (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                try:
                    P_V = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)  + P_s
                except:
                    P_V = 0.
            
            elif method == 43:
                # This is the calculation code for Method 43 - Combined: Mehta et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_me = coefficients[0]
                a_me = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                P_V = f_me * v_g ** 2. * rho_g * L * (1. + ((v_s ** 2. * rho_s * (1. - alpha)) / (v_g ** 2. * rho_g)) ** a_me) / (2. * D)  + P_s
                        
            elif method == 44:
                # This is the calculation code for Method 44 - Dense phase: Pfeffer et al. generalized (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
            
            elif method == 45:
                # This is the calculation code for Method 45 - Dense phase: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D)
                        
            elif method == 46:
                # This is the calculation code for Method 46 - Combined: Pfeffer et al. generalized (Solids velocity: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56 
                v_s = v_g - v_t
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
            
            elif method == 47:
                # This is the calculation code for Method 47 - Combined: Pfeffer et al. generalized (Solids velocity: Stevanovic et al.):
                
                # This section of code extracts the coefficients used in this method.
                n_e = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                
                # This section calculates voidage.                    
                rho_ds = m_s / (v_s * A_c)
                alpha = 1. - rho_ds / rho_s
                
                # This section calculates the volumetric flow rates.
                V_g = m_g / rho_g
                V_s = m_s / rho_s
                
                # This section calculates the mixture mean velocity.
                v_m = (V_g + V_s) / A_c
                
                # This section calculates the equivalent friction factor.
                f_m_prime = f_g * (1. + R_s) ** (1. - n_e)
                
                # This section calculates static pressure drop.
                P_g_s = alpha * rho_g * Direction * L * g
                P_s_s = rho_s * (1. - alpha) * Direction * L * g
                
                # The total pressure drop is found next.
                P_s = P_g_s + P_s_s
                
                # The total pressure drop is found next.
                P_V = f_m_prime * L * rho_g * v_m ** 2. / (2. * D) + P_s
            
            return P_V
            
        @staticmethod
        def Bends(method, type, R_B, D, d, d_v50, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, R_g, T_g, Direction, coefficients, Function_Dict):
            # This class contains the pressure drop calculation methods for bends.
            
            if method == 1:
                # This is the calculation code for Method 1 - Cai et al. (Gas pressure drop: Mason et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_c_b = coefficients[0]
                b_c_b = coefficients[1]
                c_c_b = coefficients[2]
                d_c_b = coefficients[3]
                e_c_b = coefficients[4]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates the equivalent length for the bend. 
                f_T = 0.25 / (np.log10((epsilon / D) / 3.7)) ** 2.
                
                K_L = cs(R_B / D) * f_T
                
                L_eq = K_L * D / f_g
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_g * rho_g / 2. * v_g ** 2. / D * L_eq
                
                # This section calculates the solids friction factor.
                f_c_s = a_c_b * R_s ** b_c_b * Fr ** c_c_b * (d / D) ** d_c_b * (R_B / D) ** e_c_b
                
                # This section calculates the pressure drop due to solids.
                P_B_s = f_c_s * R_s * np.pi * R_B / (2. * D) * rho_g * v_g ** 2. / (2.)
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 2:
                # This is the calculation code for Method 2 - Cai et al. (Gas pressure drop: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_c_b = coefficients[0]
                b_c_b = coefficients[1]
                c_c_b = coefficients[2]
                d_c_b = coefficients[3]
                e_c_b = coefficients[4]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code determines the bend type coefficient.
                if type == 'Radius':
                    a_T = 1.
                else:
                    a_T = 25.
                
                # This section calculates the pressure drop due to gas.
                P_B_g = a_T * (0.0194 * v_g ** 2. + 0.0374 * v_g ** 1.75) * (R_B / D) ** 0.84
                
                # This section calculates the solids friction factor.
                f_c_s = a_c_b * R_s ** b_c_b * Fr ** c_c_b * (d / D) ** d_c_b * (R_B / D) ** e_c_b
                
                # This section calculates the pressure drop due to solids.
                P_B_s = f_c_s * R_s * np.pi * R_B / (2. * D) * rho_g * v_g ** 2. / (2.)
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 3:
                # This is the calculation code for Method 3 - Cai et al. (Gas pressure drop: Cai et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_c_b = coefficients[0]
                b_c_b = coefficients[1]
                c_c_b = coefficients[2]
                d_c_b = coefficients[3]
                e_c_b = coefficients[4]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code determines the gas friction coefficient for bends.
                f_c_g = (0.29 + 0.304 * (Re_p * (D / (2. * R_B))) ** -0.25) / np.sqrt(2. * R_B / D)
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_c_g * R_B * rho_g * v_g ** 2. / (2. * D)
                
                # This section calculates the solids friction factor.
                f_c_s = a_c_b * R_s ** b_c_b * Fr ** c_c_b * (d / D) ** d_c_b * (R_B / D) ** e_c_b
                
                # This section calculates the pressure drop due to solids.
                P_B_s = f_c_s * R_s * np.pi * R_B / (2. * D) * rho_g * v_g ** 2. / (2.)
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                               
            elif method == 4:
                # This is the calculation code for Method 4 - Sharma et al.:
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                                
                # The total pressure drop is found next.
                P_B = (1. + R_s) * rho_g * v_g ** 2. / 2.
                
            elif method == 5:
                # This is the calculation code for Method 5 - Rinoshika (Gas pressure drop: Mason et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_b_p = coefficients[0]
                b_b_p = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                    
                # This section calculates the equivalent length for the bend. 
                f_T = 0.25 / (np.log10((epsilon / D) / 3.7)) ** 2.
                
                K_L = cs(R_B / D) * f_T
                
                L_eq = K_L * D / f_g
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_g * rho_g / 2. * v_g ** 2. / D * L_eq
                
                # This section calculates the solids friction factor.
                f_b_p = a_b_p * Fr ** b_b_p
                
                # This section calculates the pressure drop due to solids.
                P_B_s = m_s / m_g * R_B * rho_g * v_g ** 2. / (2. * D) * f_b_p
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 6:
                # This is the calculation code for Method 6 - Rinoshika (Gas pressure drop: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_b_p = coefficients[0]
                b_b_p = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                # This section of code determines the bend type coefficient.
                if type == 'Radius':
                    a_T = 1.
                else:
                    a_T = 25.
                
                # This section calculates the pressure drop due to gas.
                P_B_g = a_T * (0.0194 * v_g ** 2. + 0.0374 * v_g ** 1.75) * (R_B / D) ** 0.84
                
                # This section calculates the solids friction factor.
                f_b_p = a_b_p * Fr ** b_b_p
                
                # This section calculates the pressure drop due to solids.
                P_B_s = m_s / m_g * R_B * rho_g * v_g ** 2. / (2. * D) * f_b_p
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 7:
                # This is the calculation code for Method 7 - Rinoshika (Gas pressure drop: Cai et al.):
                
                # This section of code extracts the coefficients used in this method.
                a_b_p = coefficients[0]
                b_b_p = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                # This section of code determines the gas friction coefficient for bends.
                f_c_g = (0.29 + 0.304 * (Re_p * (D / (2. * R_B))) ** -0.25) / np.sqrt(2. * R_B / D)
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_c_g * R_B * rho_g * v_g ** 2. / (2. * D)
                
                # This section calculates the solids friction factor.
                f_b_p = a_b_p * Fr ** b_b_p
                
                # This section calculates the pressure drop due to solids.
                P_B_s = m_s / m_g * R_B * rho_g * v_g ** 2. / (2. * D) * f_b_p
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                            
            elif method == 8:
                # This is the calculation code for Method 8 - Mason et al. (Gas pressure drop: Mason et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_B_M = coefficients[0]
                a_B_M = coefficients[1]
                b_B_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                    
                # This section calculates the equivalent length for the bend. 
                f_T = 0.25 / (np.log10((epsilon / D) / 3.7)) ** 2.
                
                K_L = cs(R_B / D) * f_T
                
                L_eq = K_L * D / f_g
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_g * rho_g / 2. * v_g ** 2. / D * L_eq
                
                # This section calculates the pressure drop due to solids.
                P_B_s = f_B_M * R_s ** a_B_M * Fr ** b_B_M
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 9:
                # This is the calculation code for Method 9 - Mason et al. (Gas pressure drop: Tripathi et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_B_M = coefficients[0]
                a_B_M = coefficients[1]
                b_B_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                # This section of code determines the bend type coefficient.
                if type == 'Radius':
                    a_T = 1.
                else:
                    a_T = 25.
                
                # This section calculates the pressure drop due to gas.
                P_B_g = a_T * (0.0194 * v_g ** 2. + 0.0374 * v_g ** 1.75) * (R_B / D) ** 0.84
                
                # This section calculates the pressure drop due to solids.
                P_B_s = f_B_M * R_s ** a_B_M * Fr ** b_B_M
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 10:
                # This is the calculation code for Method 10 - Mason et al. (Gas pressure drop: Cai et al.):
                
                # This section of code extracts the coefficients used in this method.
                f_B_M = coefficients[0]
                a_B_M = coefficients[1]
                b_B_M = coefficients[2]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                
                # This section of code determines the gas friction coefficient for bends.
                f_c_g = (0.29 + 0.304 * (Re_p * (D / (2. * R_B))) ** -0.25) / np.sqrt(2. * R_B / D)
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_c_g * R_B * rho_g * v_g ** 2. / (2. * D)
                
                # This section calculates the pressure drop due to solids.
                P_B_s = f_B_M * R_s ** a_B_M * Fr ** b_B_M
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif method == 11:
                # This is the calculation code for Method 11 - de Moraes et al.:
                
                # This section of code extracts the coefficients used in this method.
                a_dM = coefficients[0]
                b_dM = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.      
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                # This section of code calculates the velocity and density at the bend outlet.
                if (g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g) in Function_Dict['v_fao_de_moraes_fun']:
                    v_fao = Function_Dict['v_fao_de_moraes_fun'][(g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g)]
                else:
                    v_fao = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_fao_de_moraes_fun, v_g, args=(g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g), method='lm').x[0]
                    Function_Dict['v_fao_de_moraes_fun'][(g, D, m_g, A_c, R_B, Re_p, a_dM, b_dM, rho_g, R_g, T_g)] = v_fao
                rho_go = m_g / (v_fao * A_c)

                # This section of code calculates the gas only friction factor for bends.
                f_B_g = 0.167 * (1. + 17.062 * (2. * R_B / D) ** -1.219) * Re_p ** -0.17 * (2. * R_B / D) ** 0.84
                
                # This section of code calculates the solids only friction factor for bends.
                f_B_s = a_dM * np.exp(b_dM * v_fao)
                
                # The total pressure drop is found next.
                P_B = (f_B_g + f_B_s) * rho_go * v_fao ** 2. / 2.
                
                return P_B
            
            elif method == 12:
                # This is the calculation code for Method 12 - Chambers and Marcus:
                
                # This section of code extracts the coefficients used in this method.
                B_L = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)

                # This section of code calculates the velocity and density at the bend outlet.
                v_fao = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_fao_chambers_fun, v_g, args=(m_g, A_c, B_L, R_s, rho_g, R_g, T_g), method="lm").x[0]
                rho_go = m_g / (v_fao * A_c)
                
                # The total pressure drop is found next.
                P_B = B_L * (1. + R_s) * (rho_go * v_fao ** 2.) / 2.
                
            elif method == 13:
                # This is the calculation code for Method 13 - Das and Meloy:
                
                # This section of code extracts the coefficients used in this method.
                a_D = coefficients[0]
                b_D = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)

                # This section of code calculates the velocity at the bend outlet.
                v_fao = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_fao_das_fun, v_g, args=(a_D, b_D, R_s, rho_g, m_g, A_c, R_g, T_g), method="lm").x[0]
                
                # The total pressure drop is found next.
                P_B = a_D * R_s * v_fao ** b_D
                
            elif 14 <= method <= 52:
                # This is the calculation code for Methods 14-52 - Schuchart (All Bend Orientations):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates the equivalent length for the bend. 
                f_T = 0.25 / (np.log10((epsilon / D) / 3.7)) ** 2.
                
                K_L = cs(R_B / D) * f_T
                
                L_eq = K_L * D / f_g
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_g * rho_g / 2. * v_g ** 2. / D * L_eq
                
                # P_g: This section of code contains pressure drop calculations in a straight pipe of R_B due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * R_B
                
                # This section calculates the pressure drop due to solids in a straight pipe of length R_B.
                method_p = method - 13
                P_s_R_B = Section_Pressure_Drop.Dense_Phase().Horizontal_Pipe_Sections(method_p, R_B, D, d, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, T_g, R_g, coefficients, Function_Dict) - P_g
                
                # This section calculates the pressure drop due to solids.
                P_B_s = 210. * (2. * R_B / D ) ** -1.15 * P_s_R_B
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            elif 53 <= method <= 99:
                # This is the calculation code for Methods 53-99 - Schuchart (H-V and V-H Bend Orientations):
                
                # This section of code extracts the coefficients used in this method.
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the non-dimensional numbers used for this method.
                Re_p = Section_Pressure_Drop.Dimensionless_Numbers.Pipe_Reynolds_Number(rho_g, v_g, D, mu_g)
                
                if Re_p > 2320. and Re_p <= 80. * D / epsilon:
                    f_g = 0.3164 / (Re_p ** 0.25)
                    
                elif Re_p > 80. * D / epsilon and Re_p <= 4160. * (D / (2. * epsilon) ** 0.85): 
                    if (Re_p, epsilon, D) in Function_Dict['f_g_fun']:
                        f_g = Function_Dict['f_g_fun'][(Re_p, epsilon, D)]
                    else:
                        f_g = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().f_g_fun, 0.3164 / (Re_p ** 0.25), args=(Re_p, epsilon, D), method="lm").x[0]
                        Function_Dict['f_g_fun'][(Re_p, epsilon, D)] = f_g
                    
                else:
                    f_g = 1. / (1.74 + 2. * np.log10(D / (2. * epsilon))) ** 2.
                    
                # This section calculates the equivalent length for the bend. 
                f_T = 0.25 / (np.log10((epsilon / D) / 3.7)) ** 2.
                
                K_L = cs(R_B / D) * f_T
                
                L_eq = K_L * D / f_g
                
                # This section calculates the pressure drop due to gas.
                P_B_g = f_g * rho_g / 2. * v_g ** 2. / D * L_eq
                
                # P_g: This section of code contains pressure drop calculations in a straight pipe of R_B due to the conveying of gas alone.
                P_g = f_g * rho_g / 2. * v_g ** 2. / D * R_B
                
                # This section calculates the pressure drop due to solids in a straight pipe of length R_B.
                method_p = method - 52
                P_s_R_B = Section_Pressure_Drop.Dense_Phase().Vertical_Pipe_Sections(method_p, R_B, D, d, rho_g, rho_s, epsilon, mu_g, g, m_g, m_s, T_g, R_g, Direction, coefficients, Function_Dict) - P_g
                
                # This section calculates the pressure drop due to solids.
                P_B_s = 210. * (2. * R_B / D ) ** -1.15 * P_s_R_B
                
                # The total pressure drop is found next.
                P_B = P_B_g + P_B_s
                
            return P_B
            
        @staticmethod
        def Acceleration_Of_Solids(method, L, D, d, rho_g, rho_s, g, m_g, m_s, mu_g, coefficients, Function_Dict):
            # This class contains the pressure drop calculation methods for the acceleration of solids.
            
            if method == 1:
                # This is the calculation code for Method 1 - Sharma et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
            
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = v_g ** 2. / 2. * rho_g * (1. + 2. * R_s * v_s / v_g)
                
            elif method == 2:
                # This is the calculation code for Method 2 - Sharma et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = v_g ** 2. / 2. * rho_g * (1. + 2. * R_s * v_s / v_g)
                
            elif method == 3:
                # This is the calculation code for Method 3 - Sharma et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                               
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = v_g ** 2. / 2. * rho_g * (1. + 2. * R_s * v_s / v_g)
                
            elif method == 4:
                # This is the calculation code for Method 4 - Sharma et al. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = v_g ** 2. / 2. * rho_g * (1. + 2. * R_s * v_s / v_g)
                
            elif method == 5:
                # This is the calculation code for Method 5 - Sharma et al. (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                R_v = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = v_g ** 2. / 2. * rho_g * (1. + 2. * R_s * v_s / v_g)
                
            elif method == 6:
                # This is the calculation code for Method 6 - Sharma et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = v_g ** 2. / 2. * rho_g * (1. + 2. * R_s * v_s / v_g)
            
            elif method == 7:
                # This is the calculation code for Method 7 - Tripathi et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
        
            elif method == 8:
                # This is the calculation code for Method 8 - Tripathi et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
                
            elif method == 9:
                # This is the calculation code for Method 9 - Tripathi et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
                
            elif method == 10:
                # This is the calculation code for Method 10 - Tripathi et al. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
                
            elif method == 11:
                # This is the calculation code for Method 11 - Tripathi et al. (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                R_v = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
                
            elif method == 12:
                # This is the calculation code for Method 12 - Tripathi et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_s * v_s / A_c
        
            elif method == 13:
                # This is the calculation code for Method 13 - Mehta et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
        
            elif method == 14:
                # This is the calculation code for Method 14 - Mehta et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
                
            elif method == 15:
                # This is the calculation code for Method 15 - Mehta et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
                
            elif method == 16:
                # This is the calculation code for Method 16 - Mehta et al. (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
                
            elif method == 17:
                # This is the calculation code for Method 17 - Mehta et al. (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                R_v = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
                
            elif method == 18:
                # This is the calculation code for Method 18 - Mehta et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  m_g * v_g / (1055. * A_c) + m_s * v_s / (527.5 * A_c)
                
            elif method == 19:
                # This is the calculation code for Method 19 - Marcus et al. (Solids velocity: Naveh et al.):
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
        
            elif method == 20:
                # This is the calculation code for Method 20 - Marcus et al. (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
                
            elif method == 21:
                # This is the calculation code for Method 21 - Marcus et al. (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
                
            elif method == 22:
                # This is the calculation code for Method 22 - Marcus et al. (Solids velocity: Yang):
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
                
            elif method == 23:
                # This is the calculation code for Method 23 - Marcus et al. (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                R_v = coefficients[0]
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
                
            elif method == 24:
                # This is the calculation code for Method 15 - Marcus et al. (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A =  (0.5 + R_s * v_s / v_g) * rho_g * v_g ** 2.
                
            elif method == 25:
                # This is the calculation code for Method 25 - Tomita and Tasiro:
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = rho_g * v_g ** 2. / 2. * (1. - rho_g / rho_s)
                
            elif method == 26:
                # This is the calculation code for Method 26 - Duckworth:
                
                # This section of code calculates the non-dimensional numbers used for this method.
                R_s = Section_Pressure_Drop.Dimensionless_Numbers.Solids_Loading_Ratio(m_g, m_s)
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = R_s * rho_g * v_g ** 2. / 2. * (v_g ** 2. / (g * d) * (rho_g / rho_s) ** 2.)
                
            elif method == 27:
                # This is the calculation code for Method 27 - Agarwal (Solids velocity: Naveh et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_s = v_g * (1. - 0.02 * (Ar * (rho_s - rho_g) / rho_g * (d / D) ** 2. ) ** 0.14)
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
        
            elif method == 28:
                # This is the calculation code for Method 28 - Agarwal (Solids velocity: Klinzing et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** 0.71) * D ** 0.019
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
                
            elif method == 29:
                # This is the calculation code for Method 29 - Agarwal (Solids velocity: Klinzing et al. Generalized):
                
                # This section of code extracts the coefficients used in this method.
                a_v_s = coefficients[0]
                b_v_s = coefficients[1]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                v_s = (v_g - v_t ** a_v_s) * D ** b_v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
                
            elif method == 30:
                # This is the calculation code for Method 30 - Agarwal (Solids velocity: Yang):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section of code calculates the terminal velocity of a particle.
                Ar = Section_Pressure_Drop.Dimensionless_Numbers.Archimedes_Number(rho_g, rho_s, d, mu_g, g)
                v_t = 0.56 * mu_g / (rho_g * d) * Ar ** 0.56
                
                # This section calculates the solids velocity.
                if (v_g, v_t, D, g, rho_s, m_s, A_c) in Function_Dict['v_s_fun']:
                    v_s = Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)]
                else:
                    v_s = root(Section_Pressure_Drop.Dense_Phase.Iteration_Functions().v_s_fun, v_g, args=(v_g, v_t, D, g, rho_s, m_s, A_c), method="lm").x[0]
                    Function_Dict['v_s_fun'][(v_g, v_t, D, g, rho_s, m_s, A_c)] = v_s
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
                
            elif method == 31:
                # This is the calculation code for Method 31 - Agarwal (Solids velocity: Mallick and Wypych):
                
                # This section of code extracts the coefficients used in this method.
                R_v = coefficients[0]
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                Fr = Section_Pressure_Drop.Dimensionless_Numbers.Froude_Number(v_g, g, D)
                v_s = v_g * (0.0177 - 0.0179 * R_v) * Fr + 1.071 * R_v - 0.071
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
                
            elif method == 32:
                # This is the calculation code for Method 32 - Agarwal (Solids velocity: Stevanovic et al.):
                
                # This section of code calculates the gas velocity.
                A_c = D ** 2. / 4. * np.pi
                v_g = m_g / (rho_g * A_c)
                
                # This section calculates the solids velocity.
                v_s = v_g
                
                # This section of code calculates the solids pressure drop due to acceleration.
                P_A = 1.5 * m_s * v_s / A_c
                
            
            return P_A