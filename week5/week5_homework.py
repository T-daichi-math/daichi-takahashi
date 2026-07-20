import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
x = [0,-1,1]
y = [1,-1,-1]

plt.scatter(x, y, s=250, color="skyblue")
labels = ["A", "B", "C"]

for i in range(3):
    plt.text(x[i], y[i], labels[i],
             fontsize=12,
             ha="center",
             va="center")

# 矢印を追加
plt.annotate(
    "",
    xy=(-1, -1), xytext=(0, 1),
    arrowprops=dict(arrowstyle="->", lw=2)
)

plt.annotate(
    "",
    xy=(1, -1), xytext=(0, 1),
    arrowprops=dict(arrowstyle="->", lw=2)
)

plt.title("Simple Morse Decomposition")
plt.axis("equal")
plt.grid(True)

plt.show()

