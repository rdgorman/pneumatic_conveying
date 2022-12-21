class Miscellaneous_Functions:
    # This class contains miscellaneous functions used by other programs.
    
    class Import_Functions:
    # This class loads information needed to calculate the pressure drop.
        @staticmethod
        def Pipeline_Load(Pipeline_File):
            # This function imports pipeline geometry from a .pip file.
            Pipeline_Geometry_Raw_Text = []
            with open(Pipeline_File, 'r') as f:
                for line in f.readlines():
                    Pipeline_Geometry_Raw_Text.append(line)

            D = float(Pipeline_Geometry_Raw_Text[0].split(',')[1])
            epsilon = float(Pipeline_Geometry_Raw_Text[1].split(',')[1])
            Pipeline_Sections = []
            for section in Pipeline_Geometry_Raw_Text[5:]:
                parts = section.replace(' ', '').replace('\n', '').split(',')
                if parts[2] == 'Splitter':
                    Pipeline_Sections.append((int(parts[0]), int(parts[1]), parts[2], parts[3], float(parts[4]), int(parts[5]), int(parts[6]), float(parts[7])))
                else:
                    Pipeline_Sections.append((int(parts[0]), int(parts[1]), parts[2], parts[3], float(parts[4]), float(parts[7])))
            
            return epsilon, Pipeline_Sections
            
        @staticmethod
        def Solids_Load(Solids_File):
            # This function imports solids properties from a .sol file.
            Solids_Raw_Text = []
            with open(Solids_File, 'r') as f:
                for line in f.readlines():
                    Solids_Raw_Text.append(line)
            
            d = float(Solids_Raw_Text[0].split(',')[1])
            dv50 = float(Solids_Raw_Text[1].split(',')[1])
            rho_s = float(Solids_Raw_Text[2].split(',')[1])
            m_s = float(Solids_Raw_Text[3].split(',')[1])
            return d, dv50, rho_s, m_s
            
        @staticmethod
        def Gas_Load(Gas_File):
            # This function imports gas properties from a .gas file.
            Gas_Raw_Text = []
            with open(Gas_File, 'r') as f:
                for line in f.readlines():
                    Gas_Raw_Text.append(line)
                    
            Gas_Type = Gas_Raw_Text[0].split(',')[1].strip()
            T_g = float(Gas_Raw_Text[1].split(',')[1])
            m_g = float(Gas_Raw_Text[2].split(',')[1])
            
            return Gas_Type, T_g, m_g
    
        @staticmethod
        def Coefficient_Dictionary_Definer(Coefficient_Dictionary_File):
            # This creates dictionaries that give the number of coefficients for each calculation method.
            
            # Load coefficient dictionary .csv file.
            lines = []
            with open(Coefficient_Dictionary_File, 'r') as f:
                for line in f.readlines():
                    lines.append(line.replace('\n', '').split(','))
            lines = lines[1:]
            
            # Load .csv file information into dictionaries.
            Coefficient_Dictionary = {}
            for l in lines:
                if l[0] not in list(Coefficient_Dictionary.keys()):
                    Coefficient_Dictionary[l[0]] = {}
                
                try:
                    Coefficient_Dictionary[l[0]][int(l[1])] = int(l[2])
                except:
                    difference = int(l[1].split('-')[0]) - 1
                    difference_type = l[3]
                    for k in range(int(l[1].split('-')[0]), int(l[1].split('-')[1]) + 1):
                        Coefficient_Dictionary[l[0]][k] = Coefficient_Dictionary[l[3]][k - difference]
                        
            return Coefficient_Dictionary