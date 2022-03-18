import numpy as np

def remove_specks(color_img):
    window_size = 5
    # remove tiny regions of stray white pixels 
    for i in range(color_img.shape[0] - window_size):
        for j in range(color_img.shape[1] - window_size):
            if np.sum(color_img[i:i+window_size, j:j+window_size, 0] > 0) < 3:
                color_img[i+window_size//2, j+window_size//2, :] = 0

    # zero out all edges of image
    color_img[0:window_size, :, :] = 0
    color_img[-window_size:, :, :] = 0
    color_img[:, 0:window_size, :] = 0
    color_img[:, -window_size:, :] = 0

    return color_img

def closest_nonzero_depth_pixel(pt, depth_img):
    # find the closest nonzero pixel to pt
    nonzero_pixels = np.nonzero(depth_img)
    # print(nonzero_pixels[0].shape)
    pts_combined = np.array([nonzero_pixels[0], nonzero_pixels[1]]).T
    distances = np.sqrt((pts_combined[:, 0] - pt[0]) ** 2 + (pts_combined[:, 1] - pt[1]) ** 2)
    return pts_combined[np.argmin(distances)]

def normalize(vec):
    return vec / np.linalg.norm(vec)

def pixel_to_dist_from_nearest_black_point(image):

    # for each pixel, compute distance to nearest black pixel
    all_black = np.nonzero(image == 0)
    # add all black points to queue
    dq = deque()
    for i in range(len(all_black[0])):
        dq.append(np.array((all_black[0][i], all_black[1][i])))
    
    # initialize distances to infinity
    distances = np.full(image.shape, np.inf)
    distances[all_black] = 0

    # run dijkstra's algorithm
    # while len(q) > 0:
    #     print("Iter")
    #     closest_point, closest_dist = q[0], distances[tuple(q[0])]
    #     for i in range(1, len(q)):
    #         cur_pt_tuple = tuple(q[i])
    #         # print(distances.shape, q[i].shape, distances[tuple(q[i])])
    #         if distances[cur_pt_tuple] < closest_dist:
    #             closest_point, closest_dist = q[i], distances[cur_pt_tuple]
    #     q.remove(closest_point)

    #     # update distances
    #     for i in range(len(q)):
    #         distances[cur_pt_tuple] = min(distances[cur_pt_tuple], closest_dist + np.linalg.norm(q[i] - closest_point))
    
    # run BFS
    iters = 0
    while len(dq) > 0:
        iters += 1
        if iters % 100000 == 0:
            print("Iter", iters)
        next_pt = dq.popleft()
        
        # update distances
        for i in range(-1, 2):
            for j in range(-1, 2):
                cur_pt = next_pt + np.array((i, j))
                if not (cur_pt[0] < 0 or cur_pt[1] < 0 or cur_pt[0] >= image.shape[0]
                        or cur_pt[1] >= image.shape[1]):
                    if (distances[tuple(cur_pt)] == np.inf):
                        distances[tuple(cur_pt)] = distances[tuple(next_pt)] + 1
                        dq.append(cur_pt)
    return distances

def smooth_depth(depth_img):
    depth_cpy = depth_img.copy()
    # smooth the depth image with an average blur of non-zero values in a 3x3 window
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            if depth_img[i, j] == 0:
                continue
            cnt = 0
            depth_img[i, j] = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if (i + di >= 0 and i + di < depth_img.shape[0] and \
                        j + dj >= 0 and j + dj < depth_img.shape[1] and \
                        depth_cpy[i + di, j + dj] > 0):
                        depth_img[i, j] += depth_cpy[i + di, j + dj]
                        cnt += 1
            depth_img[i, j] /= cnt
    return depth_img

def visualize_depth_map_in_3d(depth):
    points = []
    counter = 0
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            depth_val = depth[i, j]
            if depth_val <= 0.1:
                continue
            counter += 1
            if counter % 1 != 0:
                continue
            points.append(np.array([i, j, depth_val]))
    print("showing " + str(len(points)))
    points = np.array(points)

    # # fig = plt.figure()
    # # ax = plt.axes(projection='3d')
    lz = list(zip(*points))
    x = np.array(lz[0]).squeeze()
    y = np.array(lz[1]).squeeze()
    z = np.array(lz[2]).squeeze()

    data = [go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
        )
    )]
    # show the plot
    fig = go.Figure(data=data)
    fig.show()
    # exit()
