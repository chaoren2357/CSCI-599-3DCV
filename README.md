# CSCI599
3D Vision Assignment Template for Spring 2024.

The following tutorial will go through you with how to use three.js for your assignment visualization. Please make sure your VScode is installed with "Live Server" plugin.

## How to use
```shell
git clone https://github.com/jingyangcarl/CSCI599.git
cd CSCI599
ls ./ # you should see index.html and README.md showup in the terminal
code ./ # open this folder via vscode locally
# open and right click on the index.html
# select "Open With Live Server" to run the code over localhost.
```

## HW1
You can run by typing `python assignments/assignment1.py`
After `if __name__ == '__main__'`:
- Uncomment `mesh_subdivided = subdivision_loop_halfedge(mesh)` and `mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')` for generate mesh subdivided results.
- Uncomment `mesh_decimated = simplify_quadric_error(mesh, face_count=500)` and `mesh_decimated.export('assets/assignment1/bunny_decimated_500.obj')` for generate mesh decimated results.

Demo model:
- Cube: originally 8 vertices and 6 faces
- Bunny: originally 2503 vertices and 4968 faces
- Face: originally 8807 vertices and 17256 faces

### Loop subdivision
- I use half edge data structure for loop subdivision. My implementation is slow due to the twin setting of half edge.

| Attributes | Trimesh | My implementation |
|-------|-------|-------|
| Cube for 1 iter | ![](/images/cube_subdivided_1_gt.gif) | ![](/images/cube_subdivided_1_he.gif) |
| Time | 0.0067365s | 0.0011031s |
| Cube for 2 iter | ![](/images/cube_subdivided_2_gt.gif) | ![](/images/cube_subdivided_2_he.gif) |
| Time | 0.007020s | 0.011545s |
| Cube for 3 iter | ![](/images/cube_subdivided_3_gt.gif) | ![](/images/cube_subdivided_3_he.gif) |
| Time | 0.008119s | 0.13042s |
| Cube for 4 iter | ![](/images/cube_subdivided_4_gt.gif) | ![](/images/cube_subdivided_4_he.gif) |
| Time | 0.01047s | 2.045044s |


### Quadratic Error based mesh decimation


| Attributes | Trimesh | My implementation |
|-------|-------|-------|
| Bunny to 2000 faces | ![](/images/bunny_decimated_2000_gt.gif) | ![](/images/bunny_decimated_2000_mine.gif) |
| Time | 1.25s | 6.4s |
| Bunny to 500 faces | ![](/images/bunny_decimated_500_gt.gif) | ![](/images/bunny_decimated_500_mine.gif) |
| Time | 1.30s | 6.76s |
| Face to 10000 faces | ![](/images/face_decimated_10000_gt.gif) | ![](/images/face_decimated_10000_mine.gif) |
| Time | 1.09s | 59.90s |
| Face to 2000 faces | ![](/images/face_decimated_2000_gt.gif) | ![](/images/face_decimated_2000_mine.gif) |
| Time | 1.13s | 96.66s |
