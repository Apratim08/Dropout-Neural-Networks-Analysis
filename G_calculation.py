import numpy as np

data = {
    0.1: {"testError": 4.183334112167358, "MEC": 41855.6},
    0.2: {"testError": 1.616668701171875, "MEC": 61721.2},
    0.3: {"testError": 1.7000019550323486, "MEC": 81586.8},
    0.4: {"testError": 1.4999985694885254, "MEC": 96122.0},
    0.5: {"testError": 1.466667652130127, "MEC": 98170.0},
    0.6: {"testError": 1.6833305358886719, "MEC": 100218.0},
    0.7: {"testError": 1.6666650772094727, "MEC": 102266.0},
    0.8: {"testError": 1.8833339214324951, "MEC": 104314.0},
    0.9: {"testError": 1.4833331108093262, "MEC": 106362.0},
    0.99: {"testError": 1.3333320617675781, "MEC": 108205.2}
}

k = 10000 

G_values = {}

for z, values in data.items():
    p = 1 - (values["testError"] / 100)  
    if p > 0:  
        k_p = k * p  
        log2_p = np.log2(p) 
        G = -k_p * log2_p / values["MEC"]  
        G_values[z] = G 
    else:
        G_values[z] = 'Undefined'

print(G_values)
