import matplotlib.pyplot as plt
plt.plot([0, 1, 2, 3, 4, 5, 6], [0, 23, 33, 37, 41, 46], marker='o')
plt.plot([0, 1, 2, 3, 4, 5], [0, 17, 27, 34, 39, 42], marker='o')
plt.axis([0, 5, 0, 100])
plt.ylabel('Accuracy %')
plt.xlabel('Epoch #')

# legend
plt.legend(('Validation Accuracy', 'Training Accuracy'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=16)

plt.show()