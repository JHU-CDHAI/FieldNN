from ._base import func_convert_Num_factory

# NumField2Settings = {
#     # 'Field': start, end, Min, Max, scale
#     'A1C-V': [3.5, 6.5,  0, 10,   0.1], 
#     'ALT-V': [10,  120,  0, 1000, 10], 
#     'BMI-V': [15,  60,   0, 900,  5],
#     'HDL-V': [15,  100,  0, 150,  5],
#     'LDL-V': [20,  200, -2, 300,  10],
#     'DBP-V': [20,  120,  0, 190,  5],
#     'SBP-V': [50,  200,  0, 300,  5],
    
#     # start, end, Min, Max, scale = 18, 80, 1, 120, 5
#     'static-age': [18, 80, 1, 120, 5],
#     'static-V1stPDa1c': [ 5.6, 6.5, 5, 7, 0.1],
# }

start, end, Min, Max, scale = [3.5, 6.5,  0, 10,   0.1]
func_convert_A1Cwgt = func_convert_Num_factory(start, end, Min, Max, scale)
