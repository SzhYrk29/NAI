import random
from PIL import Image
import numpy as np
import math

def search_path_with_astar(start, goal, accessible_fn, h, callback_fn):
    open_set = {tuple(start)}
    closed_set = set()
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): h(start, goal)}

    while open_set:
        callback_fn(closed_set, open_set)

        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

        if current == tuple(goal):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(tuple(start))
            return path[::-1]

        open_set.remove(current)
        closed_set.add(current)

        for neighbor in accessible_fn(current):
            if tuple(neighbor) in closed_set:
                continue

            tentative_g_score = g_score.get(current, float('inf')) + h(neighbor, current)

            if tuple(neighbor) not in open_set:
                open_set.add(tuple(neighbor))
            elif tentative_g_score >= g_score.get(tuple(neighbor), float('inf')):
                continue

            came_from[tuple(neighbor)] = current
            g_score[tuple(neighbor)] = tentative_g_score
            f_score[tuple(neighbor)] = g_score[tuple(neighbor)] + h(neighbor, goal)

    return []

def h_function_euklides(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return math.sqrt(dx ** 2 + dy ** 2)

def h_function_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def h_function_random(a, b):
    distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
    random_value = random.uniform(0, 1)
    return distance + random_value

def getpixel(image, dims, position):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return None
    return image[position[1], position[0]]

def setpixel(image, dims, position, value):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return
    image[position[1], position[0]] = value

def accessible(bitmap, dims, point):
    neighbors = []
    height, width = dims
    for i in range(len(point)):
        for delta in [-1, 1]:
            neighbor = list(point)
            neighbor[i] += delta
            neighbor = tuple(neighbor)
            x, y = neighbor[0], neighbor[1]
            if 0 <= x < width and 0 <= y < height:
                if bitmap[y, x][0] == 0:
                    neighbors.append(neighbor)
    return neighbors

def load_world_map(fname):
    img = Image.open(fname)
    img = img.convert("RGBA")
    pixels = np.array(img)
    dims = pixels.shape[:2]
    return dims, pixels

def save_world_map(fname, image):
    img = Image.fromarray(image)
    img.save(fname)

def find_pixel_position(image, dims, value):
    for y in range(dims[0]):
        for x in range(dims[1]):
            if tuple(image[y, x]) == value:
                return [x, y]
    raise ValueError("Could not find pixel with the given value!")

if __name__ == "__main__":
    metric_choice = input("Choose a metric (euclidean, manhattan, random): ").strip().lower()

    dims, bitmap = load_world_map("img.png")

    start = find_pixel_position(bitmap, dims, (255, 0, 255, 255))  # Cyan pixel
    goal = find_pixel_position(bitmap, dims, (255, 255, 0, 255))    # Yellow pixel

    setpixel(bitmap, dims, start, (0, 0, 0, 255))
    setpixel(bitmap, dims, goal, (0, 0, 0, 255))

    def on_iteration_nothing(closed_set, open_set):
        pass

    if metric_choice == 'euclidean':
        heuristic = h_function_euklides
    elif metric_choice == 'manhattan':
        heuristic = h_function_manhattan
    elif metric_choice == 'random':
        heuristic = h_function_random
    else:
        print("Invalid metric choice. Using Euclidean by default.")
        heuristic = h_function_euklides

    path = search_path_with_astar(start, goal, lambda p: accessible(bitmap, dims, p), heuristic, on_iteration_nothing)

    print(path)

    for p in path:
        setpixel(bitmap, dims, p, (255, 0, 0, 255))  # Mark the path with red

    save_world_map("result.png", bitmap)
