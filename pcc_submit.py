import argparse

from FECalc.FECalc import FECalc

# Read User Input
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--pcc", help="PCC name")
parser.add_argument("-t", "--target", help="Target molecule")

args = parser.parse_args()

PCC = args.pcc
MOL = args.target

calculator = FECalc(PCC, MOL, f"/project2/andrewferguson/armin/FE_DATA/{PCC}_{MOL}", 
                    "/project/andrewferguson/armin/HTVS_Fentanyl/FECalc/FECalc_settings.JSON")
vals = calculator.run()
