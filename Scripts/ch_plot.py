import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

width = [8.5, 6, 4, 2.2]
height = [6.0, 4.5, 3.0, 1.5]
xy = (0,0)
ch = ['recursively enumerable', 'context-sensitive', 'context-free', 'regular']
h_offset = [0.35,0.35,0.35,0.5]
w_offset = [-1.72, -1.4, -1.01, -0.45]
ells = [Ellipse(xy, w, height[i]) for i, w in enumerate(width)]

#a = plt.subplot(111, aspect='equal')
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

for j, e in enumerate(ells):
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.4)
    e.set_facecolor(np.random.rand(3))
    ax.add_artist(e)
    ax.text(w_offset[j], (height[j]/2.17)-h_offset[j], ch[j])

plt.xlim(-5, 5)
plt.ylim(-4, 4)
plt.axis('off')
plt.savefig('../Plots/chomsky_h.eps', format='eps', bbox_inches='tight')
plt.show()