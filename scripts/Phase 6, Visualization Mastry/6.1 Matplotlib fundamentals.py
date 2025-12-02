# Figure - The entire canvas
# Axes - the actual plotting area inside the figure (like a box)
# Axis - x-axis or y-axis

import matplotlib.pyplot as plt

''' # a simple plot with a figure and one Axes
fig, ax = plt.subplots()
    # one plotting area in a figure

# plot data on the axes
ax.plot([1,2,3,4], [10, 20, 25, 30])

# add labels
ax.set_title("Basic Figure/Axes Example")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

plt.show()
'''

'''# Multiple Axes (Subplots)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot([1, 2, 3], [1, 4, 9])
axs[0].set_title("Left plot")

axs[1].plot([1,2,3], [1, 2, 3])
axs[1].set_title("Right Plot")

plt.show()
'''

''' # Subplot layouts in Matplotlib
    # axs[0, 0] -> top left
    # axs[0, 1] -> top right
    # axs[1, 0] -> bottom left
    # axs[1, 1] -> bottom right
'''

'''# sharing Axes
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    # if sharey=False, the scales are different making visual comparision difficult
axs[0].plot([1,2,3], [1, 4,9])
axs[1].plot([1,2,3], [2,3,4])
axs[0].set_title("Left")
axs[1].set_title("Right")
plt.show()
'''

''' # GridSpec for Flexible Layouts
    # GridSpec lets us span rows/columns
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, :]) # top row spans both columns
ax2 = fig.add_subplot(gs[1, 0]) # bottom left
ax3 = fig.add_subplot(gs[1, 1]) # bottom right

ax1.plot([1, 2, 3], [1, 2, 3])
ax2.plot([1, 2, 3], [1, 4, 9])
ax3.plot([1, 2, 3], [3, 2, 1])

plt.tight_layout()
plt.show()
'''

''' # Multiple independent figure with its own axes
fig1, ax1 = plt.subplots()
ax1.plot([1, 2, 3], [4, 5, 6])
ax1.set_title("Figure 1")

fig2, ax2 = plt.subplots()
ax2.plot([1,2,3], [3, 2, 1])
ax2.set_title("Figure 2")

plt.show()
'''

''' # Annotation and styling in Matplotlib
    # 1. Highlighting points with ax.annotate()
    # ax.annotate() -> add text and arrows
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
fig, ax = plt.subplots()
ax.plot(x, y, marker='o')
# annotate the maximum point
ax.annotate('Peak', xy=(4, 30), xytext=(3, 28), arrowprops=dict(facecolor='black', shrink=0.05))
    # 'Peak' is the text, xy is the point arrow is pointing, xytext is the point of text and begining of arrow, shrink -> arrow shrink by 5%
ax.set_title("Annotation Example")
plt.show()
'''

''' # Customizing ticks
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
fig, ax = plt.subplots()
ax.plot(x, y)
# custom ticks
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['One', 'Two', 'Three', 'Four']) 
    # replace numeric ticks with descriptive labels
ax.set_yticks([10, 20, 30])
ax.set_yticklabels(['Low', 'Medium', 'High'])
plt.show()
'''

''' # Legends (explains plots)
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
fig, ax = plt.subplots()
ax.plot(x, y, label='Growth', color='blue')
ax.plot(x, [v*0.8 for v in y], label='Baseline', color='red')
ax.legend(loc='upper left', fontsize=10, frameon=True)
plt.show()
'''

# Colors and Line Styles
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
fig, ax = plt.subplots()
ax.plot(x, y, color='green', linestyle='--', linewidth=2, marker='o', markersize=8)
    # color = 'red' or '#1f77b4'
    # linestyle='--' (dashed), ':' (dotted), '-.' (dash-dot)
ax.set_title("Styling Example")
plt.show()

