import numpy as np
from A_fun import A_fun
from F_fun import F_fun
from energyband import energyband
from v_band import v_band

print(f" a = {A_fun(21,1,0.1)}")
print(f" f = {F_fun(21,1,0.1)}")

k = np.array([0.1, 1.3, 2.4])
t = np.linspace(0,20,50)
a = A_fun(t,1,0.1)
#
# print(f" a.shape = {a.shape}")
# print(f" k = {k}")
# print(f"a+k.shape = {(a+k).shape}")
#
# print(f" k =  {k}")
# print(f"a(t) = {a}")
# print(f" k + a(t) = {k + a}")


print(f"v_band(a + k,0,0) = {v_band(a + k,0,0)}")
print("")
print(f"v_band(a+ k,0,1) = {v_band(a + k,0,1)}")
