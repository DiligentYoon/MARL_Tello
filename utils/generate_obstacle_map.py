# Writing generate_map.py with all classes in grayscale and mapping
import os
import numpy as np
from PIL import Image, ImageDraw
import argparse

# Class-to-grayscale mapping:
# 0 (free)      → 230 (light gray)
# 1 (obstacle)  → 128 (medium gray)
# 2 (start)     → 200 (lighter gray)
# 3 (goal)      →  80 (darker gray)

GRAYS = {
    0: 230,
    1: 128,
    2: 200,
    3:  80,
}

def generate_map_with_obstacles(width, height, start_size, goal_size,
                                num_rects, rect_min, rect_max,
                                wall_thickness, seed=None):
    if seed is not None:
        np.random.seed(seed)

    grid = np.zeros((height, width), dtype=np.uint8)
    t = wall_thickness

    # (2) start and goal zones
    sx0, sy0 = t, t
    gx1, gy1 = height - t - 1, width - t - 1
    grid[:, sy0:sy0+start_size] = 2
    grid[:, gy1-goal_size+1:gy1+1] = 3

    # (1) boundary walls
    grid[0:t, :] = 1
    grid[-t:, :] = 1
    grid[:, 0:t] = 1
    grid[:, -t:] = 1

    # (3) carve L-shaped guaranteed path
    carve_y = sx0 + start_size - 1
    for x in range(sy0+start_size, gy1-goal_size+1):
        grid[carve_y, x] = 0
    carve_x = gy1 - goal_size + 1
    for y in range(carve_y, gx1-goal_size+1):
        grid[y, carve_x] = 0

    # (4) place random rectangular obstacles
    placed = 0
    attempts = 0
    max_attempts = num_rects * 10
    while placed < num_rects and attempts < max_attempts:
        attempts += 1
        rw = np.random.randint(rect_min, rect_max+1)
        rh = np.random.randint(rect_min, rect_max+1)
        x0 = np.random.randint(t, width - rw - t)
        y0 = np.random.randint(t, height - rh - t)
        region = grid[y0:y0+rh, x0:x0+rw]
        if np.any(region != 0):
            continue
        grid[y0:y0+rh, x0:x0+rw] = 1
        placed += 1

    return grid

def save_grid_as_png(grid, filename, transpose=False):
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    disp = grid.T if transpose else grid
    h, w = disp.shape
    img = Image.new("L", (w, h))  # 'L' mode for grayscale
    for y in range(h):
        for x in range(w):
            v = disp[y, x]
            img.putpixel((x, y), GRAYS.get(v, 0))
    img.save(filename)
    print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate multiple obstacle maps in grayscale")
    parser.add_argument("--width", type=int, default=500, help="map width (px)")
    parser.add_argument("--height", type=int, default=100, help="map height (px)")
    parser.add_argument("--start_size", type=int, default=50, help="start zone size (px)")
    parser.add_argument("--goal_size", type=int, default=50, help="goal zone size (px)")
    parser.add_argument("--num_rects", type=int, default=10, help="number of obstacles")
    parser.add_argument("--rect_min", type=int, default=5, help="min rectangle side (px)")
    parser.add_argument("--rect_max", type=int, default=20, help="max rectangle side (px)")
    parser.add_argument("--wall_thickness", type=int, default=5, help="boundary wall thickness (px)")
    parser.add_argument("--num_maps", type=int, default=100, help="number of maps to generate")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--output_dir", type=str, default="myMap", help="output directory")
    parser.add_argument("--transpose", action="store_true", help="transpose grid for horizontal orientation")
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else np.random.randint(0, 1e6)
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.num_maps):
        seed = base_seed + i
        grid = generate_map_with_obstacles(
            args.width, args.height,
            args.start_size, args.goal_size,
            args.num_rects, args.rect_min, args.rect_max,
            args.wall_thickness, seed
        )
        filename = os.path.join(args.output_dir, f"map_{i:03d}.png")
        save_grid_as_png(grid, filename, transpose=args.transpose)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()

print("generate_map.py updated: all classes use grayscale mapping. Mapping numbers: see GRAYS dict.")

# 0 = free → 230

# 1 = obstacle → 128

# 2 = start zone→ 200

# 3 = goal zone → 80