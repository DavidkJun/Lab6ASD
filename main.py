import turtle
import random
import math
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt

random.seed(3314)

matrix = [[random.uniform(0, 2) for j in range(11)] for i in range(11)]
k = 1.0 - 1 * 0.005 - 4 * 0.005 - 0.27
multipliedMatrix = np.multiply(matrix, k)
matrix_for_dir = np.floor(multipliedMatrix)
undir_matrix = np.maximum(matrix_for_dir, np.transpose(matrix_for_dir))
print(undir_matrix)

n = 11
B = np.random.uniform(0, 2.0, (n, n))
print("Matrix B:")
print(B)

def ceil100(x):
    return np.ceil(x * 100)

Aundir = np.ones((n, n))
C = np.ceil(B * 100 * Aundir).astype(int)
print("Matrix C:")
print(C)

D = np.where(C > 0, 1, 0)
print("Matrix D:")
print(D)

H = np.where(D == D.T, 1, 0)
print("Matrix H:")
print(H)

Tr = np.triu(np.ones((n, n)), k=1)
print("Matrix Tr:")
print(Tr)

W = (D * H * Tr + C)
W = np.maximum(W, W.T)
print("Matrix W:")
print(W)

def kruskal_mst(weight_matrix):
    G = nx.Graph()
    for i in range(len(weight_matrix)):
        for j in range(i + 1, len(weight_matrix)):
            if weight_matrix[i][j] > 0:
                G.add_edge(i, j, weight=weight_matrix[i][j])
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    return mst


mst_kruskal = kruskal_mst(W)
print("Kruskal's MST:")
print(sorted(mst_kruskal.edges(data=True)))


G = nx.Graph()
for i in range(n):
    for j in range(i + 1, n):
        if W[i][j] > 0:
            G.add_edge(i, j, weight=W[i][j])

positions = []


def drawNumbers():
    global positions
    turtle.speed(0)
    turtle.penup()
    turtle.goto(-300, 300)
    count = 1
    vertical_spacing = -120 * 1.2
    horizontal_spacing = 180 * 1.2
    for i in range(4):
        positions.append(turtle.position())
        turtle.color("white")
        turtle.write(count, align="center")
        turtle.color("black")
        count += 1
        turtle.goto(turtle.xcor(), turtle.ycor() + vertical_spacing)
    for i in range(3):
        positions.append(turtle.position())
        turtle.color("white")
        turtle.write(count, align="center")
        turtle.color("black")
        count += 1
        turtle.goto(turtle.xcor() + horizontal_spacing, turtle.ycor())
    for i in range(4):
        positions.append(turtle.position())
        turtle.color("white")
        turtle.write(count, align="center")
        turtle.color("black")
        count += 1
        turtle.goto(turtle.xcor() - horizontal_spacing / 1.5, turtle.ycor() - vertical_spacing)
    turtle.hideturtle()


rad = 16


def drawCircles():
    turtle.speed(0)

    def draw_circle(x, y):
        turtle.begin_fill()
        turtle.penup()
        turtle.goto(x, y - rad)
        turtle.pendown()
        turtle.circle(rad)
        turtle.end_fill()

    for pos in positions:
        draw_circle(pos[0], pos[1])


def calculateDistance(start, end):
    distance = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    return distance


def getOrto(x, y):
    orthogonal_vector = (-y, x)
    magnitude = math.sqrt(orthogonal_vector[0] ** 2 + orthogonal_vector[1] ** 2)
    unit_vector = (orthogonal_vector[0] / magnitude, orthogonal_vector[1] / magnitude)
    return np.array(unit_vector)


def getStartPosition(pos1, pos_top):
    vec = (pos_top - pos1)
    vec = vec / calculateDistance(pos1, pos_top)
    return pos1 + vec * rad


def drawArrow(pos1, pos2, directed, k, isSearch, weight=None):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    arr_vec = pos2 - pos1
    middle = (pos2 + pos1) / 2
    orto = getOrto(*arr_vec)
    dist_coef = k / calculateDistance(pos1, pos2) * 110
    side = 1 if dist_coef > 40 else -1
    orto = orto * side
    pos_top = middle + orto * dist_coef + orto * 40
    turtle.penup()
    pos_start = getStartPosition(pos1, pos_top)
    turtle.goto(pos_start[0], pos_start[1])
    turtle.pendown()
    turtle.goto(pos_top[0], pos_top[1])
    pos_end = getStartPosition(pos2, pos_top)
    turtle.goto(pos_end[0], pos_end[1])
    turtle.penup()
    if directed:
        drawDirectedArrow(pos_end, pos_top, pos2)
    if isSearch:
        turtle.penup()
        turtle.goto(pos2[0], pos2[1] - rad)
        turtle.pendown()
        turtle.color("green")
        turtle.begin_fill()
        turtle.circle(rad)
        turtle.end_fill()
        turtle.color("black")


    if weight is not None:
        weight_pos = pos_end + (pos_start - pos_end) * 0.2
        turtle.penup()
        turtle.goto(weight_pos[0], weight_pos[1])
        turtle.pendown()
        turtle.write(f'{weight:.1f}', align="center", font=("Arial", 12, "normal"))


def drawDirectedArrow(pos_end, pos_top, pos2):
    width = 14 / 2
    length = 14
    orto2 = getOrto(*(pos_top - pos2))
    vec_back = (pos_top - pos_end) / calculateDistance(pos_top, pos_end)
    a = pos_end + vec_back * length
    turtle.goto(a + orto2 * width)
    turtle.pendown()
    turtle.begin_fill()
    turtle.goto(pos_end)
    turtle.penup()
    turtle.goto(a - orto2 * width)
    turtle.pendown()
    turtle.goto(pos_end)
    turtle.penup()
    turtle.goto(a - orto2 * width)
    turtle.pendown()
    turtle.goto(a + orto2 * width)
    turtle.end_fill()
    turtle.penup()


def drawSelfLoop(position, loop_size, directed):
    turtle.penup()
    turtle.goto(position[0], position[1] + rad)
    turtle.pendown()
    turtle.circle(loop_size)
    if directed:
        arrow_start_angle = 120
        arrow_length = 15
        arrow_width = 10
        arrow_pos_x = position[0] + loop_size * math.cos(math.radians(arrow_start_angle))
        arrow_pos_y = position[1] + 0.25 * rad + loop_size * math.sin(math.radians(arrow_start_angle))
        arrow_dir_x = arrow_length * math.cos(math.radians(arrow_start_angle + 190))
        arrow_dir_y = arrow_length * math.sin(math.radians(arrow_start_angle + 190))
        turtle.penup()
        turtle.goto(arrow_pos_x, arrow_pos_y)
        turtle.pendown()
        turtle.begin_fill()
        turtle.goto(arrow_pos_x + arrow_dir_x, arrow_pos_y + arrow_dir_y)
        turtle.goto(arrow_pos_x - arrow_width, arrow_pos_y - arrow_width)
        turtle.goto(arrow_pos_x, arrow_pos_y)
        turtle.end_fill()


drawNumbers()
drawCircles()


def drawGraph(isDir):
    for i in range(11):
        for j in range(i + 1):
            if undir_matrix[i][j]:
                weight = W[i][j]
                if i == j:
                    drawSelfLoop(positions[i], 30, isDir)
                else:
                    if i == 7:
                        k = 350
                    elif i == 8 and 200 <= calculateDistance(positions[i], positions[j]) <= 280:
                        k = 50
                    elif i > 8:
                        k = 30
                    else:
                        k = 120
                    drawArrow(positions[i], positions[j], directed=isDir, k=k, isSearch=False, weight=weight)

    time.sleep(7)


drawGraph(False)


def drawMST(mst):
    plt.figure(figsize=(8, 6))
    pos = {i: positions[i] for i in range(n)}
    nx.draw(mst, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    labels = nx.get_edge_attributes(mst, 'weight')
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=labels)
    plt.title("Kruskal's MST")
    plt.show()


drawMST(mst_kruskal)
# Обчислення суми ваг мінімального кістяка
def calculate_mst_weight(mst):
    weight_sum = sum(data['weight'] for u, v, data in mst.edges(data=True))
    return weight_sum

mst_weight = calculate_mst_weight(mst_kruskal)
print("Total weight of Kruskal's MST:", mst_weight)
