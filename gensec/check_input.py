import numpy as np
import json

# TODO: Reduce parameters to minimum necessary. If some parameter excludes using another, it could be deleted. For example orientations

def Check_input(parameters):
    '''
    Purposes:  

    1. Check if a parameter exists; otherwise, set it.  
    2. Set parameter.activate to false if not necessary.  
    3. Set to default values if necessary and if default values are available.  
    4. Throw an error if necessary parameters are not set.  
    5. Safely save updated parameters to a new .json file. 
    
     
    '''
    if "protocol" not in parameters:
        raise ValueError("No protocol in input file. Please define if you want to generate and/or search.")
    else:
        if parameters["protocol"]["generate"] is False and parameters["protocol"]["search"] is False:
            raise ValueError("Both generate and search are set to False. Please set one to True.")
    if "check_db" not in parameters["protocol"]:
        parameters["protocol"]["check_db"] = False
    
    if "geometry" not in parameters or "filename" not in parameters["geometry"]:
        raise ValueError("No geometry file given in input. Please add file and format.")
    
    
    if "configuration" not in parameters:
        parameters["configuration"] = {
            "adsorption" : {"activate": False},
            "clashes" : {"intramolecular" : 2.0, "with_fixed_frame" : 1.5},
            "coms" : {"activate": False},
            "orientations" : {"activate" : True},
            "torsions" : {"activate" : False},
        }
        print("No configuration in input file. Clashes set to default values.")
    else:
        if "adsorption" not in parameters["configuration"]:
            parameters["configuration"]["adsorption"] = {"activate": False}
        elif parameters["configuration"]["adsorption"]["activate"] is True:
            if "range" not in parameters["configuration"]["adsorption"]:
                parameters["configuration"]["adsorption"]["range"] = [0.5, 3.0]
                print("Range for adsorption not given. Set to default values [0.5, 3.0].")
            if "point" not in parameters["configuration"]["adsorption"]:
                parameters["configuration"]["adsorption"]["point"] = [0.0, 0.0, 0.0]
                print("Point for adsorption not given. Set to default values [0.0, 0.0, 0.0].")
        
        if "clashes" not in parameters["configuration"]:
            parameters["configuration"]["clashes"] = {"intramolecular" : 2.0, "with_fixed_frame" : 1.5}
            print("No clashes parameters given. Set to default values (intramolecular : 2.0 and fixed frame: 1.5).")
            
        if "coms" not in parameters["configuration"]:
            parameters["configuration"]["coms"] = {"activate": False}
            # TODO: Discuss defaults 
        elif parameters["configuration"]["coms"]["activate"] is True:
            if "z_values" not in parameters["configuration"]["coms"]:
                parameters["configuration"]["coms"]["z_values"] = "identical"
            
        if "orientations" not in parameters["configuration"]:
            parameters["configuration"]["orientations"] = {"activate" : True}
            # TODO: Discuss defaults
            
        if "torsions" not in parameters["configuration"]:
            parameters["configuration"]["torsions"] = {"activate" : False}
            # TODO: Discuss defaults
        
    # TODO: Work on calculator and implement chekck after if search is activated.
    
    if "check_forces" not in parameters:
        parameters["check_forces"] = {"activate" : False}
    elif parameters["check_forces"] is True:
        if "max_force" not in parameters["check_forces"]:
            parameters["check_forces"]["max_force"] = 0.02
            print("No max force given. Set to default value 0.02 eV/A.")
    
    if "fixed_frame" not in parameters:
        parameters["fixed_frame"] = {"activate" : False}
    elif parameters["fixed_frame"] is True:
        if "filename" not in parameters["fixed_frame"]:
            raise ValueError("Fixed frame is activated but no filename is given. Please also remember to add the format.")
        if "is_unit_cell" not in parameters["fixed_frame"]:
            parameters["fixed_frame"]["is_unit_cell"] = False
    
    if "mic" not in parameters:
        parameters["mic"] = {"activate" : False}
    else:
        if parameters["mic"]["activate"] is True:
            if "pbe" not in parameters["mic"]:
                raise ValueError("Mic is activated but no pbe is given. Please add pbe.")   # TODO: check if cell is in fixed frame and set as pbe
    
    if "supercell_finder" not in parameters:
        parameters["supercell_finder"] = {"activate" : False}
    elif parameters["supercell_finder"]["activate"] is True:
        if "unit_cell_method" not in parameters["supercell_finder"]:
            parameters["supercell_finder"]["unit_cell_method"] = "find"
        if parameters["supercell_finder"]["unit_cell_method"] == "find" and "unit_cell_finder" not in parameters:
            parameters["unit_cell_finder"] = {
                "min_angle": 20,
                "max_angle": 90,
                "seperation_factor": 1.0,
                "scan_first": {
                    "activate": False,
                    "first_min_angle": 0,
                    "first_max_angle": 180,
                    "first_n_steps": 10
                },
                "adaptive": {
                    "activate": False,
                    "n_points": 5,
                    "tolerance": 1e-4,
                    "max_iterations": 10
                }
            }
        elif parameters["supercell_finder"]["unit_cell_method"] == "find":
            if "min_angle" not in parameters["unit_cell_finder"]:
                parameters["unit_cell_finder"]["min_angle"] = 20
            if "max_angle" not in parameters["unit_cell_finder"]:
                parameters["unit_cell_finder"]["max_angle"] = 90
            if "seperation_factor" not in parameters["unit_cell_finder"]:
                parameters["unit_cell_finder"]["seperation_factor"] = 1.0
            
            if "scan_first" not in parameters["unit_cell_finder"]:
                parameters["unit_cell_finder"]["scan_first"] = {
                    "activate": False,
                    "first_min_angle": 0,
                    "first_max_angle": 180,
                    "first_n_steps": 10
                }
            elif parameters["unit_cell_finder"]["scan_first"]["activate"] is True:
                if "first_min_angle" not in parameters["unit_cell_finder"]["scan_first"]:
                    parameters["unit_cell_finder"]["scan_first"]["first_min_angle"] = 0
                if "first_max_angle" not in parameters["unit_cell_finder"]["scan_first"]:
                    parameters["unit_cell_finder"]["scan_first"]["first_max_angle"] = 180
                if "first_n_steps" not in parameters["unit_cell_finder"]["scan_first"]:
                    parameters["unit_cell_finder"]["scan_first"]["first_n_steps"] = 10
            
            if "adaptive" not in parameters["unit_cell_finder"]:
                parameters["unit_cell_finder"]["adaptive"] = {
                    "activate": False,
                    "n_points": 5,
                    "tolerance": 1e-4,
                    "max_iterations": 10
                }
            elif parameters["unit_cell_finder"]["adaptive"]["activate"] is True:
                if "n_points" not in parameters["unit_cell_finder"]["adaptive"]:
                    parameters["unit_cell_finder"]["adaptive"]["n_points"] = 5
                if "tolerance" not in parameters["unit_cell_finder"]["adaptive"]:
                    parameters["unit_cell_finder"]["adaptive"]["tolerance"] = 1e-4
                if "max_iterations" not in parameters["unit_cell_finder"]["adaptive"]:
                    parameters["unit_cell_finder"]["adaptive"]["max_iterations"] = 10
        
        if "max_area_diff" not in parameters["supercell_finder"]:
            parameters["supercell_finder"]["max_area_diff"] = 0.1
        if "Z_cell_length" not in parameters["supercell_finder"]:
            parameters["supercell_finder"]["Z_cell_length"] = 100
        if "m_range" not in parameters["supercell_finder"]:
            parameters["supercell_finder"]["m_range"]["type"] = "max"
            parameters["supercell_finder"]["m_range"]["max"] = 15    
        else:
            if parameters["supercell_finder"]["m_range"]["type"] == "max" and "max_s" not in parameters["supercell_finder"]["m_range"]:
                parameters["supercell_finder"]["m_range"]["max_s"] = 15
            if parameters["supercell_finder"]["m_range"]["type"] == "max" and "max_f" not in parameters["supercell_finder"]["m_range"]:
                parameters["supercell_finder"]["m_range"]["max_f"] = parameters["supercell_finder"]["m_range"]["max_s"]        
            if "max_range_f" not in parameters["supercell_finder"]["m_range"] and parameters["supercell_finder"]["m_range"]["type"] == "given_range":
                parameters["supercell_finder"]["max_range_f"] = [3, 3]
            if "max_range_s" not in parameters["supercell_finder"]["m_range"] and parameters["supercell_finder"]["m_range"]["type"] == "given_range":
                parameters["supercell_finder"]["max_range_s"] = [10, 10]
            if parameters["supercell_finder"]["m_range"]["type"] not in ["max", "given_range"]:
                raise implementationError("type for m_range not implemented. Choose between max and given_range.")
        if "max_attempts" not in parameters["supercell_finder"]:
            parameters["supercell_finder"]["max_attempts"] = 1
            
        parameters["configuration"]["coms"]["activate"] = True
        parameters["configuration"]["coms"]["same"] = False
        if "values" not in parameters["configuration"]["coms"]:
            parameters["configuration"]["coms"]["values"] = "given"
    
            
    if "number_of_replicas" not in parameters:
        parameters["number_of_replicas"] = 1
    
    if "trials" not in parameters:
        parameters["trials"] = 1000
        
    if "success" not in parameters:
        parameters["success"] = 1500
        
    if "name" not in parameters:
        parameters["name"] = "Unnamed"
        print("No name given. Set to default value 'Unnamed'.")
        
    safe_parameters = open("checked_parameters.json", "w")
    json.dump(parameters, safe_parameters, indent = 4)
    safe_parameters.close()
    print("Checked parameters and saved to checked_parameters.json.")
    
    return parameters
    