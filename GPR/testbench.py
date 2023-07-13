from GSKGPR import GSKGPR

X = ['FGGFG', 'FGGDG', 'GGDEK', 'FFGKG', 'FFALG', 'GGALL']
Y = [22.5, 26.6, 12.5, 23.4, 25.5, 22.0]
gpr = GSKGPR(X, Y)
model = gpr.fit(0.1, [1, 2, 3, 4, 5], (1e-20, 1e10))
test_mean, test_std = model.predict(['GFGGF', 'KEDGG'], return_std=True)
print(gpr.fit_l)
print(test_mean)
print(test_std)
