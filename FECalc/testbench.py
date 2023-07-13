from FECalc import FECalc

test = FECalc("GGDEK", "FEN", "/project/andrewferguson/armin/HTVS_Fentanyl/FE_DATA/GGDEK_FEN", "/project/andrewferguson/armin/HTVS_Fentanyl/FECalc/FECalc_settings.JSON")
test._get_atom_ids("/project/andrewferguson/armin/HTVS_Fentanyl/FE_DATA/GGDEK_FEN/complex/em/em.gro")
test.get_FE()