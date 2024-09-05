# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:01:48 2024

@author: 86150
"""

### Numpy
a = [-3.14, 0, 3.14]
a

import numpy as np                     # Load the library

a = np.linspace(-np.pi, np.pi, 100)    # Create even grid from -π to π
a

b = np.cos(a)
c = np.sin(a)

b@c


## SciPy
from scipy.stats import norm
from scipy.integrate import quad

ϕ = norm()
value, error = quad(ϕ.pdf, -2, 2)  # Integrate using Gaussian quadrature
value

## NetworkX
import networkx as nx
import matplotlib.pyplot as plt

# genetate a random graph
p = dict((i,(np.random.uniform(0,1),np.random.uniform(0,1))) for i in range(200))
g = nx.random_geometric_graph(200,0.12,pos=p)
pos = nx.get_node_attributes(g,'pos')

# find node nearest the center point (0.5, 0.5)
dists = [(x-0.5)**2+(y-0.5)**2 for x,y in list(pos.values())]
ncenter = np.argmin(dists)

## plot graph, coloring by path length from the center node
p = nx.single_source_shortest_path_length(g,ncenter)
plt.figure()
nx.draw_networkx_edges(g,pos,alpha=0.4)
nx.draw_networkx_nodes(g,
                       pos,
                       nodelist=list(p.keys()),
                       node_size=120,alpha=0.5,
                       node_color=list(p.values()),
                       cmap=plt.cm.jet_r)
plt.show()