import matplotlib.pyplot as plt
plt.plot([0, 1, 2, 3, 4, 5], [0, 2, 1.8, 1.75, 1.6, 1.5], marker='o')
plt.plot([0, 1, 2, 3, 4, 5], [0, 2.2, 2, 1.8, 1.7, 1.6], marker='o')
plt.axis([0, 5, 0, 100])
plt.ylabel('Loss')
plt.xlabel('Epoch #')

# legend
plt.legend(('Validation Loss', 'Training loss'),
           shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=16)

plt.show()