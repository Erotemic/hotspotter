from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def L2(x):
    return x**2

def L1(x):
    return np.abs(x)

def Geman_McClure(x, a=1.0):
    'a = outlier threshold'
    return (x**2) / (1 + (x**2 / a**2))

def Cauchy(x):
    pass

def Beaton_Tukey(x, a=4.0):
    return (a**2)/6 * (1.0 - (1.0 - (x/a)**2)**3)**(np.array(np.abs(x) <= a, dtype=np.float))
    #np.array[(a**2)/6 * (1 - (1 - (u/a)**2)**3) if np.abs(u) <= a else (a**2)/6 for u in x]

def Beaton_Tukey_weight(x, a=4.0):
    return np.array(np.abs(x) <= a, dtype=np.float) * (1 - (x/a)**2)**2

def visualize_func(func):
    x_radius = 42
    x_data = np.linspace(-x_radius,x_radius, 1000)
    print(func)
    func_name = func.func_name
    func_vars = func.func_code.co_varnames
    print(func_name)
    fig = plt.figure()
    ax  = plt.subplot(111)
    ax.set_title(func_name)
    if len(func_vars) == 1 or True:
        y_data = func(x_data)
        plt.plot(x_data, y_data)
    else:
        pmax = 1
        num  = 10
        for a in np.linspace(-pmax,pmax,num):
            color = plt.get_cmap('jet')((a+pmax)/(pmax*2))
            y_data = func(x_data, a)
            plt.plot(x_data, y_data, color=color, label=('a=%r' % a))

        plt.legend()
    fig.show()

if __name__ == '__main__':
    robust_functions = [L1,L2,Geman_McClure, Beaton_Tukey, Beaton_Tukey_weight]
    for func in iter(robust_functions):
        visualize_func(func)

try:
    __IPYTHON__
except:
    plt.show()
    pass
