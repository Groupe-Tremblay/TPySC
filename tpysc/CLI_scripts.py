#!/usr/bin/python3

from .TPSC import TPSC
import sys
import json


def print_help():
    print(f"""
TPSC: A Python library that allows the computation of Hubbard model related functions and quantities (such as the self-energy and Green's function) using the Two-Particle-Self-Consistent (TPSC) approach first described in Vick and Tremblay (1997). 

Usage:
  TPSC input_file [output_file]
  
Parse parameters from the input_file in JSON format, perform the computation and print results to screen.
If an output_file path was provided, perform silently and write the results in the output_file in JSON format.
""")
    return

def TPSC_cli():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print_help()
        exit(1)
    
    para_filename = sys.argv[1]
    with open(para_filename, 'r') as infile:
        params = json.load(infile)
        
    tpsc = TPSC(**params)
    tpsc.run()
    
    if len(sys.argv) == 3:
        tpsc.writeResultsJSON(sys.argv[2])
    else:
        tpsc.printResults()
    
    exit(0)
