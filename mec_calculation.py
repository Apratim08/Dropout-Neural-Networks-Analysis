def MECwithP(p):
    input = (28**3) + 28
    bias_per_layer = (2048*p)
    layer1 = min((2048*p)*28 + bias_per_layer, input)
    layer2 = min((2048*p)**2 + bias_per_layer, layer1)
    layer3 = min((2048*p)**2 + bias_per_layer, layer2)
    out = min(2048*p*10 + 10, layer3)
    return input + layer1 + layer2 + layer3 + out


p_val = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
# p_val = [0.1]

mec = []
for i in p_val:
    mec.append(MECwithP(i))

print("MEC for each p:")
for i in range(len(mec)):
    print(p_val[i], ":", mec[i])
