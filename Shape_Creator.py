# This code extracts the shape data from a pipeline and creates an .stl shape file.

import os
import trimesh
import subprocess

import numpy as np

from euclid3 import Point3
from shapely import geometry
from pymesh import stl

# Set program paths.
WorkingDir = os.getcwd()
openscad_path = r'C:\Program Files\OpenSCAD'
meshlab_path = r'C:\Program Files\VCG\MeshLab'

# Set File Name
filename = 'Pipeline'
# filename = 'Test_Rig'

# Other inputs.
points_per_section_input = 1001

# Loads pipeline geometry.
Pipeline_Geometry_Raw_Text = []
with open('{}.pip'.format(filename), 'r') as f:
    for line in f.readlines():
        Pipeline_Geometry_Raw_Text.append(line)

Diameter = float(Pipeline_Geometry_Raw_Text[0].split(',')[1])
Pipeline_Sections = []
for section in Pipeline_Geometry_Raw_Text[5:]:
    parts = section.replace(' ', '').replace('\n', '').split(',')
    if parts[2] == 'Splitter':
        Pipeline_Sections.append((int(parts[0]), int(parts[1]), parts[2], parts[3], float(parts[4]), int(parts[5]), int(parts[6])))
    else:
        Pipeline_Sections.append((int(parts[0]), int(parts[1]), parts[2], parts[3], float(parts[4])))

# This makes a list of the flow paths in the pipeline.
Flow_Paths = list(range(1 + len([s for s in Pipeline_Sections if s[2] == 'Splitter' ]) * 2))

# Plot points that define pipeline.
x_paths = [[] for path in Flow_Paths]
y_paths = [[] for path in Flow_Paths]
z_paths = [[] for path in Flow_Paths]

x_paths[0].append(0.)
y_paths[0].append(0.)
z_paths[0].append(0.)

for section in Pipeline_Sections:    
    section_number = section[0]
    path = section[1]
    section_type = section[2]
    orientation = section[3]
    L_c = section[4]
    
    last_x = x_paths[path][-1]
    last_y = y_paths[path][-1]
    last_z = z_paths[path][-1]
    
    if section_type == 'Horizontal':
        points_per_section = 3
        for p in range(points_per_section):
            if orientation == 'North':
                x_paths[path].append(last_x + L_c / float(points_per_section - 1) * float(p))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z)
            elif orientation == 'East':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y + L_c / float(points_per_section - 1) * float(p))
                z_paths[path].append(last_z)
            elif orientation == 'South':
                x_paths[path].append(last_x - L_c / float(points_per_section - 1) * float(p))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z)
            elif orientation == 'West':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y - L_c / float(points_per_section - 1) * float(p))
                z_paths[path].append(last_z)
            else:
                print('Unrecognized orientation type at Section {}.'.format(section_number))
                quit()
                
    elif section_type == 'Vertical':
        points_per_section = 3
        for p in range(points_per_section):
            if orientation == 'Upward':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y)
                z_paths[path].append(last_z + L_c / float(points_per_section - 1) * float(p))
            elif orientation == 'Downward':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y)
                z_paths[path].append(last_z - L_c / float(points_per_section - 1) * float(p))
            else:
                print('Unrecognized orentation type at Section {}.'.format(section_number))
                quit()
                
    elif section_type == 'Bend':
        points_per_section = points_per_section_input
        for p in range(points_per_section):
            if orientation == 'N-E':
                x_paths[path].append(last_x + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                y_paths[path].append(last_y + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                z_paths[path].append(last_z)
            elif orientation == 'N-W':
                x_paths[path].append(last_x + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                y_paths[path].append(last_y - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                z_paths[path].append(last_z)
            elif orientation == 'N-U':
                x_paths[path].append(last_x + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
            elif orientation == 'N-D':
                x_paths[path].append(last_x + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                
            elif orientation == 'E-N':
                x_paths[path].append(last_x + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                y_paths[path].append(last_y + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                z_paths[path].append(last_z)     
            elif orientation == 'E-S':
                x_paths[path].append(last_x - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                y_paths[path].append(last_y + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                z_paths[path].append(last_z)
            elif orientation == 'E-U':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                z_paths[path].append(last_z + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
            elif orientation == 'E-D':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                z_paths[path].append(last_z - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                
            elif orientation == 'S-E':
                x_paths[path].append(last_x - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                y_paths[path].append(last_y + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                z_paths[path].append(last_z)     
            elif orientation == 'S-W':
                x_paths[path].append(last_x - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                y_paths[path].append(last_y - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                z_paths[path].append(last_z) 
            elif orientation == 'S-U':
                x_paths[path].append(last_x - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
            elif orientation == 'S-D':
                x_paths[path].append(last_x - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                
            elif orientation == 'W-N':
                x_paths[path].append(last_x + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                y_paths[path].append(last_y - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                z_paths[path].append(last_z)     
            elif orientation == 'W-S':
                x_paths[path].append(last_x - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                y_paths[path].append(last_y - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                z_paths[path].append(last_z) 
            elif orientation == 'W-U':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                z_paths[path].append(last_z + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
            elif orientation == 'W-D':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                z_paths[path].append(last_z - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                
            elif orientation == 'U-N':
                x_paths[path].append(last_x + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))     
            elif orientation == 'U-E':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                z_paths[path].append(last_z + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1)))) 
            elif orientation == 'U-S':
                x_paths[path].append(last_x - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
            elif orientation == 'U-W':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                z_paths[path].append(last_z + (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                
            elif orientation == 'D-N':
                x_paths[path].append(last_x + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))     
            elif orientation == 'D-E':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y + (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                z_paths[path].append(last_z - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1)))) 
            elif orientation == 'D-S':
                x_paths[path].append(last_x - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
            elif orientation == 'D-W':
                x_paths[path].append(last_x)
                y_paths[path].append(last_y - (-L_c *np.cos(np.pi / 2. * float(p) / float(points_per_section - 1)) + L_c))
                z_paths[path].append(last_z - (L_c * np.sin(np.pi / 2. * float(p) / float(points_per_section - 1))))
                
            else:
                print('Unrecognized orientation type at Section {}.'.format(section_number))
                quit()
    
    elif section_type == 'Splitter':
        points_per_section = 3
        outlet_path_1 = section[5]
        outlet_path_2 = section[6]
        try:
            inlet, outlet_1, outlet_2 = orientation.split('-')
        except:
            print('Unrecognized orientation type at Section {}.'.format(section_number))
            quit()
            
#         Plot inlet section of feed splitter.
        if inlet == 'N':
            for p in range(points_per_section):
                x_paths[path].append(last_x + L_c / float(points_per_section - 1) * float(p))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z)

            x_paths[outlet_path_1].append(x_paths[path][-1] - Diameter / 2.)
            y_paths[outlet_path_1].append(y_paths[path][-1])
            z_paths[outlet_path_1].append(z_paths[path][-1])

            x_paths[outlet_path_2].append(x_paths[path][-1] - Diameter / 2.)
            y_paths[outlet_path_2].append(y_paths[path][-1])
            z_paths[outlet_path_2].append(z_paths[path][-1])

        elif inlet == 'E':
            for p in range(points_per_section):
                x_paths[path].append(last_x)
                y_paths[path].append(last_y + L_c / float(points_per_section - 1) * float(p))
                z_paths[path].append(last_z)

            x_paths[outlet_path_1].append(x_paths[path][-1])
            y_paths[outlet_path_1].append(y_paths[path][-1] - Diameter / 2.)
            z_paths[outlet_path_1].append(z_paths[path][-1])

            x_paths[outlet_path_2].append(x_paths[path][-1])
            y_paths[outlet_path_2].append(y_paths[path][-1] - Diameter / 2.)
            z_paths[outlet_path_2].append(z_paths[path][-1])

        elif inlet == 'S':
            for p in range(points_per_section):
                x_paths[path].append(laxt_x - L_c / float(points_per_section - 1) * float(p))
                y_paths[path].append(last_y)
                z_paths[path].append(last_z)

            x_paths[outlet_path_1].append(x_paths[path][-1] + Diameter / 2.)
            y_paths[outlet_path_1].append(y_paths[path][-1])
            z_paths[outlet_path_1].append(z_paths[path][-1])

            x_paths[outlet_path_2].append(x_paths[path][-1] + Diameter / 2.)
            y_paths[outlet_path_2].append(y_paths[path][-1])
            z_paths[outlet_path_2].append(z_paths[path][-1])

        elif inlet == 'W':
            for p in range(points_per_section):
                x_paths[path].append(last_x)
                y_paths[path].append(last_y - L_c / float(points_per_section - 1) * float(p))
                z_paths[path].append(last_z)

            x_paths[outlet_path_1].append(x_paths[path][-1])
            y_paths[outlet_path_1].append(y_paths[path][-1] + Diameter / 2.)
            z_paths[outlet_path_1].append(z_paths[path][-1])

            x_paths[outlet_path_2].append(x_paths[path][-1])
            y_paths[outlet_path_2].append(y_paths[path][-1] + Diameter / 2.)
            z_paths[outlet_path_2].append(z_paths[path][-1])

        elif inlet == 'U':
            for p in range(points_per_section):
                x_paths[path].append(last_x)
                y_paths[path].append(last_y)
                z_paths[path].append(last_z + L_c / float(points_per_section - 1) * float(p))

            x_paths[outlet_path_1].append(x_paths[path][-1])
            y_paths[outlet_path_1].append(y_paths[path][-1])
            z_paths[outlet_path_1].append(z_paths[path][-1] - Diameter / 2.)

            x_paths[outlet_path_2].append(x_paths[path][-1])
            y_paths[outlet_path_2].append(y_paths[path][-1])
            z_paths[outlet_path_2].append(z_paths[path][-1] - Diameter / 2.)

        elif inlet == 'D':
            for p in range(points_per_section):
                x_paths[path].append(last_x)
                y_paths[path].append(last_y)
                z_paths[path].append(last_z - L_c / float(points_per_section - 1) * float(p))

            x_paths[outlet_path_1].append(x_paths[path][-1])
            y_paths[outlet_path_1].append(y_paths[path][-1])
            z_paths[outlet_path_1].append(z_paths[path][-1] + Diameter / 2.)

            x_paths[outlet_path_2].append(x_paths[path][-1])
            y_paths[outlet_path_2].append(y_paths[path][-1])
            z_paths[outlet_path_2].append(z_paths[path][-1] + Diameter / 2.)
        
        else:
            print('Unrecognized orientation type at Section {}.'.format(section_number))
            quit()

#         Plot first outlet section of feed splitter.
        last_x = x_paths[outlet_path_1][-1]
        last_y = y_paths[outlet_path_1][-1]
        last_z = z_paths[outlet_path_1][-1]

        for p in range(points_per_section):
            if outlet_1 == 'N':
                x_paths[outlet_path_1].append(last_x + L_c / float(points_per_section - 1) * float(p))
                y_paths[outlet_path_1].append(last_y)
                z_paths[outlet_path_1].append(last_z)
            elif outlet_1 == 'E':
                x_paths[outlet_path_1].append(last_x)
                y_paths[outlet_path_1].append(last_y + L_c / float(points_per_section - 1) * float(p))
                z_paths[outlet_path_1].append(last_z)
            elif outlet_1 == 'S':
                x_paths[outlet_path_1].append(laxt_x - L_c / float(points_per_section - 1) * float(p))
                y_paths[outlet_path_1].append(last_y)
                z_paths[outlet_path_1].append(last_z)
            elif outlet_1 == 'W':
                x_paths[outlet_path_1].append(last_x)
                y_paths[outlet_path_1].append(last_y - L_c / float(points_per_section - 1) * float(p))
                z_paths[outlet_path_1].append(last_z)
            elif outlet_1 == 'U':
                x_paths[outlet_path_1].append(last_x)
                y_paths[outlet_path_1].append(last_y)
                z_paths[outlet_path_1].append(last_z + L_c / float(points_per_section - 1) * float(p))
            elif outlet_1 == 'D':
                x_paths[outlet_path_1].append(last_x)
                y_paths[outlet_path_1].append(last_y)
                z_paths[outlet_path_1].append(last_z - L_c / float(points_per_section - 1) * float(p))
        
            else:
                print('Unrecognized orientation type at Section {}.'.format(section_number))
                quit()

    #         Plot second outlet section of feed splitter.
        last_x = x_paths[outlet_path_2][-1]
        last_y = y_paths[outlet_path_2][-1]
        last_z = z_paths[outlet_path_2][-1]
        
        for p in range(points_per_section):
            if outlet_2 == 'N':
                x_paths[outlet_path_2].append(last_x + L_c / float(points_per_section - 1) * float(p))
                y_paths[outlet_path_2].append(last_y)
                z_paths[outlet_path_2].append(last_z)
            elif outlet_2 == 'E':
                x_paths[outlet_path_2].append(last_x)
                y_paths[outlet_path_2].append(last_y + L_c / float(points_per_section - 1) * float(p))
                z_paths[outlet_path_2].append(last_z)
            elif outlet_2 == 'S':
                x_paths[outlet_path_2].append(last_x - L_c / float(points_per_section - 1) * float(p))
                y_paths[outlet_path_2].append(last_y)
                z_paths[outlet_path_2].append(last_z)
            elif outlet_2 == 'W':
                x_paths[outlet_path_2].append(last_x)
                y_paths[outlet_path_2].append(last_y - L_c / float(points_per_section - 1) * float(p))
                z_paths[outlet_path_2].append(last_z)
            elif outlet_2 == 'U':
                x_paths[outlet_path_2].append(last_x)
                y_paths[outlet_path_2].append(last_y)
                z_paths[outlet_path_2].append(last_z + L_c / float(points_per_section - 1) * float(p))
            elif outlet_2 == 'D':
                x_paths[outlet_path_2].append(last_x)
                y_paths[outlet_path_2].append(last_y)
                z_paths[outlet_path_2].append(last_z - L_c / float(points_per_section - 1) * float(p))
        
            else:
                print('Unrecognized orientation type at Section {}.'.format(section_number))
                quit()
                
    else:
        print('Unrecognized section type at Section {}.'.format(section_number))
        exit()
        
plotting_paths = [[] for path in Flow_Paths]
for path in Flow_Paths:
    for x, y, z in zip(x_paths[path], y_paths[path], z_paths[path]):
        plotting_paths[path].append(Point3(x, y, z))
        
# Define pipe diameter shape for extrusion.
points_per_shape = 101

x_shape_points = []
y_shape_points = []
z_shape_points = []

for p in range(points_per_shape):      
    x_shape_points.append(Diameter / 2. * np.cos(2. * np.pi * float(p) / float(points_per_shape - 1)))
    y_shape_points.append(Diameter / 2. * np.sin(2. * np.pi * float(p) / float(points_per_shape - 1)))
        
shape = []
for x, y in zip(x_shape_points, y_shape_points):
    shape.append(np.array([x, y]))
    
shape_poly = geometry.Polygon(shape)
    
# Extrude pipe along paths.
Meshes = []
pyMeshes = []
for path in Flow_Paths:
    path_uniques = [plotting_paths[path][0]]
    for row in plotting_paths[path]:
        if tuple(row) != tuple(path_uniques[-1]):
            path_uniques.append(row)
    Meshes.append(trimesh.creation.sweep_polygon(shape_poly, path_uniques))
    Meshes[-1].export('Mesh {}.stl'.format(path))
    pyMeshes.append(stl.Stl('Mesh {}.stl'.format(path)))
    os.remove('Mesh {}.stl'.format(path))

# Combine different paths.
combined = pyMeshes[0]
if len(pyMeshes) > 1:
    for k in range(len(pyMeshes) - 1):
        combined = combined.join(pyMeshes[k + 1])
combined.save_stl('{}.stl'.format(filename), update_normals=True)
    
# Run Meshlab to convert repair and simplify the .stl
Result = subprocess.call(r'"{}\meshlabserver" -i "{}\{}.stl" -o "{}\{}.stl" -m vn -s "{}\meshclean.mlx"'.format(meshlab_path, WorkingDir, filename, WorkingDir, filename, WorkingDir), shell=True)

# Run Meshlab to convert .stl to .u3d
Result = subprocess.call('"{}\meshlabserver" -i {}.stl -o {}.ply -m vn'.format(meshlab_path, filename, filename), shell=True)
Result = subprocess.call('"{}\meshlabserver" -i {}.ply -o {}.u3d -m vn'.format(meshlab_path, filename, filename), shell=True)

# Compile 3D PDF.
Result = subprocess.call('pdflatex {}.tex'.format(filename), shell=True)
Result = subprocess.call('pdflatex {}.tex'.format(filename), shell=True)
Result = subprocess.call('pdflatex {}.tex'.format(filename), shell=True)