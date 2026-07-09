import graphviz
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dot = graphviz.Digraph()
dot.node('A', 'Start')
dot.node('B', 'End')
dot.edge('A', 'B')

png_bytes = dot.pipe(format='png')
img = mpimg.imread(io.BytesIO(png_bytes))

fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off')
plt.savefig('test_out.png')
print("Saved to test_out.png")
