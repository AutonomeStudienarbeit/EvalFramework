import numpy as np
import cv2



def calc_radius_triangle_north(p1_roi, p2_roi):
    p1 = np.array((p1_roi[0], p2_roi[1]))
    p2 = np.array(p2_roi)
    p3 = np.array(((p2_roi[0] - p1_roi[0]) // 2 + p1_roi[0], p1_roi[1]))

    # np.linalg.norm returns euclidian distance
    edge_a = np.linalg.norm(p1 - p2)
    h = np.linalg.norm(
        np.array((p3[0], p1[1])) - (p3)
    )
    edge_b = np.linalg.norm(p1 - p3)
    edge_c = np.linalg.norm(p3 - p2)
    # print(f"edge a: {edge_a}, edge b: {edge_b}, edge c: {edge_c}, h: {h}")

    s = (edge_a + edge_b + edge_c) / 2
    # print(f"s: {s}")
    x = s - edge_c
    # print(f"x: {x}")
    r = int(np.sqrt(((s - edge_a) * (s - edge_b) * (s - edge_c)) / s))
    cp = np.array((p1[0] + x, p1[1] - r))
    return r, cp.astype(int)


def get_radius_circle(p1_roi, p2_roi):
    return (p2_roi[0] - p1_roi[0]) // 2, (
        (p2_roi[0] - p1_roi[0]) // 2 + p1_roi[0], (p2_roi[1] - p1_roi[1]) // 2 + p1_roi[1])


def get_radius_rhombus(p1_roi, p2_roi):
    cp = np.array([(p2_roi[0] - p1_roi[0]) // 2 + p1_roi[0], (p2_roi[1] - p1_roi[1]) // 2 + p1_roi[1]])
    tp = np.array(((p2_roi[0] - p1_roi[0]) // 2 + p1_roi[0], p1_roi[1]))
    h_c = np.linalg.norm(cp - tp)
    width = p2_roi[0] - p1_roi[0]
    edge = np.sqrt(np.power(h_c, 2) + np.power(width / 2, 2))
    r = int(np.sqrt(np.power(h_c, 2) - np.power(edge / 2, 2)))
    return r, cp.astype(int)


def calc_radius_triangle_south(p1_roi, p2_roi):
    p1 = np.array(((p2_roi[0] - p1_roi[0]) // 2 + p1_roi[0], p2_roi[1]))
    p2 = np.array((p2_roi[0], p1_roi[1]))
    p3 = np.array(p1_roi)

    # np.linalg.norm returns euclidian distance
    edge_a = np.linalg.norm(p1 - p2)
    h = np.linalg.norm(
        np.array((p1[0], p3[1])) - (p1)
    )
    edge_b = np.linalg.norm(p1 - p3)
    edge_c = np.linalg.norm(p3 - p2)
    # print(f"edge a: {edge_a}, edge b: {edge_b}, edge c: {edge_c}, h: {h}")

    s = (edge_a + edge_b + edge_c) / 2
    # print(f"s: {s}")
    x = s - edge_b
    # print(f"x: {x}")
    r = int(np.sqrt(((s - edge_a) * (s - edge_b) * (s - edge_c)) / s))
    cp = np.array((p3[0] + x, p3[1] + r))
    return r, cp.astype(int)


def get_radius_stop_sign(p1_roi, p2_roi):
    cp = np.array(((p2_roi[0] - p1_roi[0]) // 2 + p1_roi[0], (p2_roi[1] - p1_roi[1]) // 2 + p1_roi[1]))
    r = (p2_roi[0] - p1_roi[0]) // 2
    return r, cp


def add_sticker_in_circle(gt, image, radius, center, sticker):
    height = int(radius / 100 * 50)
    scaling_factor = 100 / sticker.shape[0] * height
    width = int(sticker.shape[1] * scaling_factor / 100)
    scaling_dimensions = (height, width)
    sticker_resized = cv2.resize(sticker, scaling_dimensions)

    # print(f"sticker.shape: {sticker_resized.shape}, radius: {radius}")
    # draw polar coordinates
    # print(f"radius to draw circle on: {radius - sticker_resized.shape[1] + 1}")
    random_radius = np.random.randint(0, radius - sticker_resized.shape[1] + 1)
    random_angle = np.random.randint(0, 360)

    # convert from polar coordinates to cartesian coordinates
    x_offset = int(random_radius * np.cos(random_angle))
    y_offset = int(random_radius * np.sin(random_angle))
    # print(f"x_off: {x_offset}, y_off: {y_offset}")
    # print(f"start: ({x_offset + center[0]}, {y_offset + center[1]})")
    # print(
    #     f"end: ({x_offset + center[0] + sticker_resized.shape[0]}, {y_offset + center[1] + sticker_resized.shape[1]})")

    overlayed_image = image.copy()
    print(overlayed_image.shape)

    print(overlayed_image[
          y_offset + center[1]: y_offset + center[1] + sticker_resized.shape[0],
          x_offset + center[0]: x_offset + center[0] + sticker_resized.shape[1]
          ].shape)

    overlayed_image[
    y_offset + center[1]: y_offset + center[1] + sticker_resized.shape[0],
    x_offset + center[0]: x_offset + center[0] + sticker_resized.shape[1]
    ] = sticker_resized

    return overlayed_image
