import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 257)
C = 512
n = 512*4*4

AE = C*C*32/x
Cir = n*x
print(AE[16])
print(Cir[16])
plt.plot(x, AE, label="Autoencoder")
plt.plot(x, Cir, label="Circular")
plt.show()
