import collections
import matplotlib.pyplot as plt
import numpy as np
AC_data = {
        55:29.44,
        88:27.85,
        89:27.31,
        84:27.62,
        76:27.86,
        71:27.53,
        70:27.05,
        73:27.29,
        64:26.15,
        62:25.82,
        79:27.62,
        98:27.60,
        97:27.92,
        61:24.49,
        67:26.69,
        94:28.09,
        93:28.09,
        92:28.42,
        83:27.30,
        68:27.01,
        78:27.91,
        95:27.38,
        90:28.00,
        82:27.30,
        58:22.73,
        85:27.80,
        91:28.47,
        80:27.43,
        60:23.85,
        66:27.26,
        81:27.63,
        72:26.51,
        75:27.22,
        96:27.55,
        69:26.96,
        74:27.04,
        59:24.67,
        65:26.47,
        63:25.72,
        57:26.36,
        87:27.58,
        99:27.75,
        56:28.11,
        86:28.14,
        77:26.97,
        100:27.41,
        }
od = collections.OrderedDict(sorted(AC_data.items()))
print(od)
x = list(od.keys())
y = list(od.values())
plt.plot(x,y)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.ylabel('BLEU Score in Valid dataset')
plt.xlabel('Epoch')
plt.show()