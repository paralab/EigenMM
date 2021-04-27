emm_fmt = """<?xml version="1.0" encoding="utf-8" ?>
<EIGEN_MM>
    <OPTIONS 
		_splitmaxiters="10" 
		_nodesperevaluator="1" 
		_subproblemsperevaluator="1" 
		_totalsubproblems="1" 
		_nevaluators="1" 
		_taskspernode="%d"
		_nevals="-1"
		_nk="10"
		_nb="4"
		_p="0" 
		_nv="10" 
		_raditers="20" 
		_splittol="0.9" 
		_radtol="1e-8" 
		_L="1.1" 
		_R="-1" 
		_terse="0" 
		_details="0" 
		_debug="1" 
		_save_correctness="0"
		_save_operators="0"
		_save_eigenvalues="0"
		_save_eigenbasis="1"
		_correctness_filename="" 
		_operators_filename=""
		_eigenvalues_filename="" 
		_eigenbasis_filename="%s" />
</EIGEN_MM>"""

import sys

if __name__ == "__main__":
    taskspernode = int(sys.argv[1])
    optionsdir = sys.argv[2]
    outputdir = sys.argv[3]
    expname = sys.argv[4]

    emmpath = optionsdir + "/" + expname + "_options.xml"
    
    f = open(emmpath, 'w')
    f_str = emm_fmt % (taskspernode, outputdir + "/" + expname)
    f.write(f_str)
    f.close()
    
