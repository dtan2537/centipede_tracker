import numpy as np
import cv2
import math
import os
from scipy.optimize import linear_sum_assignment
from pyswarms.single import GlobalBestPSO
import itertools


calc_dist = lambda x, y: np.linalg.norm(x - y)

def interpolate_points(distance, point1, point2):
    ratio = distance / calc_dist(point1, point2)
    bias_x = ratio * (point2[0] - point1[0])
    bias_y = ratio * (point2[1] - point1[1])
    new_point = [point1[0] + bias_x, point1[1] + bias_y]
    return new_point

def generate_bisecting_legs(A, B, C, line_length):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    BA = A - B
    BC = C - B

    BA_norm = BA / np.linalg.norm(BA)
    BC_norm = BC / np.linalg.norm(BC)
    
    bisector = BA_norm + BC_norm

    norm = np.linalg.norm(bisector)
    if not norm:
        bisector = np.array([BA[1], -BA[0]])
    bisector_norm = bisector / np.linalg.norm(bisector)

    bisector_scaled = (bisector_norm * line_length).astype(int)

    leg_tip1 = B + bisector_scaled
    leg_tip2 = B - bisector_scaled
    
    leg1 = [leg_tip1.tolist(), B.tolist()]
    leg2 = [leg_tip2.tolist(), B.tolist()]
    return leg1, leg2


class centipede:
    ref_norm = [1, 0]
    def __init__(self, segments = 21, ant = 4, frame_height = 0, frame_width = 0):
        #we use 22 segemtns and 40 legs model
        self.tracking_started = False
        self.antenna_body_tolerance = 0.05
        self.legs = (segments - 2) * 2
        self.ant = ant
        self.segments = segments
        self.body_length = None
        self.semgnet_length = None
        self.leg_length = None
        self.segment_points = []
        self.leg_points = []
        self.canvas = np.zeros((frame_height, frame_width))
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.line_thickness = 4
        self.anchor_point = [0, 0]
        self.end_point = [0, 0]
        self.segment_angles = []
        self.segment_weights = []
        self.machine_params = []
        self.upper_bounds = []
        self.lower_bounds = []
        self.is_anchor_switched = False

    def find_closest_point(self, polygon_squeezed, point):
        polygon_squeezed = np.array(polygon_squeezed)
        point = np.array(point)
        distances = np.linalg.norm(polygon_squeezed - point, axis=1)
        min_index = np.argmin(distances)
        return min_index, tuple(polygon_squeezed[min_index])

    def process_midline(self, process_frame):
        contours, _ = cv2.findContours(process_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
        midline = max(contours, key=cv2.contourArea)
        epsilon = 0.003 * cv2.arcLength(midline, True)  # Parameter to adjust the approximation accuracy
        midline_polygon = cv2.approxPolyDP(midline, epsilon, True)
        # cv2.drawContours(frame, [midline_polygon], -1, (0, 255, 0), 2)  # Green color for the lines

        polygon_squeezed = np.squeeze(midline_polygon)
        min1_angle = 360
        min2_angle = 360
        min_vertex = 0
        min_vertex2 = 0
        min_vertex_idx = 0
        for i in range(len(polygon_squeezed)): 
        # Get the current vertex and its two neighboring vertices
            pt1 = polygon_squeezed[i - 1]  # Previous point
            pt2 = polygon_squeezed[i]      # Current point
            pt3 = polygon_squeezed[(i + 1) % len(polygon_squeezed)]  # Next point
            
            # Calculate the angle at the current vertex
            angle = calculate_angle(pt1, pt2, pt3)
            
            # Update minimum angle and associated vertex if a new minimum is found
            if angle < min1_angle:
                min1_angle, min2_angle = angle, min1_angle
                min_vertex2 = min_vertex
                min_vertex2_idx = min_vertex_idx
                min_vertex = pt2
                min_vertex_idx = i
            elif angle < min2_angle:
                min2_angle = angle
                min_vertex2 = pt2
                min_vertex2_idx = i

        perimeter = cv2.arcLength(polygon_squeezed, True)
        return midline_polygon, min_vertex_idx, min_vertex, min_vertex2_idx, min_vertex2, perimeter

    def anchor_correction(self, midline_polygon, min_vertex_idx, min_vertex, min_vertex2_idx, min_vertex2, perimeter):
        tolerance = self.antenna_body_tolerance * self.body_length
        polygon_squeezed = np.squeeze(midline_polygon)
        #vertex_to_anchorpoint/endpoint
        v1_ap_dist = calc_dist(min_vertex, self.anchor_point)
        v1_ep_dist = calc_dist(min_vertex, self.end_point)
        v2_ap_dist = calc_dist(min_vertex2, self.anchor_point)
        v2_ep_dist = calc_dist(min_vertex2,  self.end_point)
        anchor_match_pt, anchor_match_idx, end_match_pt, end_match_idx = (min_vertex, min_vertex_idx, min_vertex2, min_vertex2_idx) if \
            v1_ap_dist < v2_ap_dist else (min_vertex2, min_vertex2_idx, min_vertex, min_vertex_idx)
        is_anchor_off = calc_dist(self.anchor_point, anchor_match_pt) > tolerance
        is_end_off = calc_dist(self.end_point, end_match_pt) > tolerance

        new_anchor_pt = []
        new_end_pt = []
        new_anchor_idx = 0

        old_anchor_new_idx, old_anchor_new_pt = self.find_closest_point(polygon_squeezed, self.anchor_point)
        _, old_end_new_pt = self.find_closest_point(polygon_squeezed, self.end_point)

        if is_anchor_off and is_end_off:
            new_anchor_pt = old_anchor_new_pt
            new_end_pt = old_end_new_pt
            new_anchor_idx = old_anchor_new_idx
        elif is_anchor_off:
            new_anchor_pt = end_match_pt
            new_end_pt = anchor_match_pt
            new_anchor_idx = end_match_idx
            self.is_anchor_switched = True
        elif is_end_off:
            new_anchor_pt = anchor_match_pt
            new_end_pt = new_end_pt
            new_anchor_idx = anchor_match_idx
        else:
            new_anchor_pt = anchor_match_pt
            new_end_pt = end_match_pt
            new_anchor_idx = anchor_match_idx
            self.body_length = perimeter

        self.anchor_point = new_anchor_pt
        self.end_point = new_end_pt
        print(self.anchor_point, self.end_point)

        return new_anchor_idx
        
        #     #update perimeter if nothing is wrong
        # if abs(self.body_length - perimeter) > tolerance:
        #     head_compare_pt = [0, 0]
        #     tail_compare_pt = [0, 0]
        #     if (v1_hp_dist < v2_hp_dist):
        #         head_compare_pt = min_vertex
        #         tail_comapre_pt = min_vertex2
        #     else:
        #         head_comapre_pt = min_vertex2
        #         tail_compare_pt = min_vertex
        #         # min_vertex_idx = min_vertex2_idx
        #         # FINISH LATER PLEASEEEEEE
                
        #     is_head_off = calc_dist(self.head_point, head_comapre_pt) > tolerance
        #     is_tail_off = calc_dist(self.tail_point, tail_compare_pt) > tolerance
        #     if is_head_off and not is_tail_off:
        #         self.tail_point = self.find_closest_point(polygon_squeezed, self.head_point)[1]
        #         self.head_point = tail_comapre_pt
        #         if self.head_point == min_vertex2:
        #             min_vertex_idx = min_vertex2_idx
        #     elif not is_head_off and is_tail_off:
        #         self.tail_point = self.find_closest_point(polygon_squeezed, self.tail_point)[1]
        #         self.head_point = head_comapre_pt
        #         if self.head_point == min_vertex2:
        #             min_vertex_idx = min_vertex2_idx
        #     elif is_head_off and is_tail_off:
        #         min_vertex_idx, self.head_point = self.find_closest_point(polygon_squeezed, self.head_point)[1]
        #         self.tail_point = self.find_closest_point(polygon_squeezed, self.tail_point)[1]
        #     else:
        #         self.head_point = head_compare_pt
        #         self.tail_point = tail_comapre_pt

    def fit_body_curve(self, polygon_squeezed, new_anchor_idx):
        #idea: use a buch of weights to represent segment lengths
        segment_weight_sum = sum(self.segment_weights)
        current_length = 0
        next_point = 0
        curr_idx = new_anchor_idx
        next_idx = curr_idx
        travelled_dist = 0
        for i in range(self.segments):
            curr_point = self.segment_points[-1]
            segment_length = 0
            desired_segment_length = (self.segment_weights[i]/ segment_weight_sum) * self.body_length
            while segment_length < desired_segment_length:
                next_idx = (curr_idx + 1) % len(polygon_squeezed)
                next_point = polygon_squeezed[next_idx]
                travelled_dist = calc_dist(curr_point, next_point)
                if segment_length + travelled_dist > desired_segment_length:
                    break
                segment_length += travelled_dist
                curr_point = next_point
                curr_idx = next_idx
            desired_distance = desired_segment_length - segment_length
            segment_point = interpolate_points(desired_distance, curr_point, next_point)
            self.segment_points.append(segment_point)
        self.segment_points = np.array(self.segment_points, np.int32)

    def find_body(self, process_frame):
        midline_polygon, min_vertex_idx, min_vertex, min_vertex2_idx, min_vertex2, perimeter = self.process_midline(process_frame)
        polygon_squeezed = np.squeeze(midline_polygon)
        if not self.tracking_started:
            self.body_length = perimeter / 2
            self.anchor_point = min_vertex
            self.end_point = min_vertex2
            print(self.end_point)
            self.tracking_started = True
            self.segment_length = self.body_length / (2 * self.segments)
            self.leg_length = 2 * self.segment_length
            self.segment_weights= [self.segment_length] * self.segments
            new_anchor_idx = min_vertex_idx
        else:
            new_anchor_idx = self.anchor_correction(midline_polygon, min_vertex_idx, min_vertex, min_vertex2_idx, min_vertex2, perimeter)
        self.segment_points = [self.anchor_point]
        
        self.fit_body_curve(polygon_squeezed, new_anchor_idx)
        

   
        # self.segment_points.append()
    def draw_body(self, canvas, segment_points, frame=None, draw_compare=0):
        cv2_segment_points = segment_points[:, np.newaxis, :]
        cv2.polylines(canvas, [cv2_segment_points], isClosed=False, color=255, thickness=self.line_thickness)
        if draw_compare:
            cv2.polylines(frame, [cv2_segment_points], isClosed=False, color=(0, 0, 255), thickness=self.line_thickness)


    def generate_legs(self):
        for i in range(1, len(self.segment_points) - 1):
            point1, mid_point, point2 = self.segment_points[i - 1 : i + 2]
            leg1, leg2 = generate_bisecting_legs(point1, mid_point, point2, self.leg_length)
            self.leg_points.append(leg1)
            self.leg_points.append(leg2)

    def draw_legs(self):
        for point1, point2 in self.leg_points:
            cv2.line(self.canvas, point1, point2, color=(0, 0, 255), thickness = self.line_thickness)


    def convert_to_params(self, segment_points):
        segment_lengths = []
        for i in range(len(segment_points) - 1):
            length = calc_dist(segment_points[i], segment_points[i+1])
            segment_lengths.append(length)
        
        segment_angles = []
        for i in range(len(segment_points) - 1):
            curr_point = segment_points[i]
            next_point = segment_points[i + 1]
            ref_point = [curr_point[0] + 10, curr_point[1]]
            angle = calculate_angle(ref_point, curr_point, next_point)
            segment_angles.append(angle)
    
        return segment_lengths, segment_angles


    def calculate_new_point(self, start, length, angle):
        x, y = start
        new_x = x + np.cos(angle) * length
        new_y = y + np.sin(angle) * length
        return [int(new_x), int(new_y)]

    def calculate_new_points(self, anchor_point, segment_lengths, segment_angles):
        head = [anchor_point[0], anchor_point[1]]
        new_segment_points = [head]
        for i in range(len(segment_angles)):
            angle = segment_angles[i]
            length = segment_lengths[i]
            start_point = new_segment_points[-1]
            new_x, new_y = self.params_to_next_point(start_point, angle, length)
            bruh = np.round((new_x, new_y)).astype(int)
            new_segment_points.append(bruh)
        
        segment_lengths, segment_angles = self.convert_to_params(new_segment_points)
        new_segment_points = np.array(new_segment_points)
        return segment_lengths, segment_angles, new_segment_points

    def params_to_next_point(self, point, angle, length):
        angle_radians = math.radians(angle)
        x, y = point
        new_x = x + length * math.cos(angle_radians)
        new_y = y + length * math.sin(angle_radians)
        return (new_x, new_y)

    def update_centipede(self, process_frame, frame = None):
        self.find_body(process_frame)
        self.canvas = np.zeros(self.canvas.shape)
        # print(self.segment_points)
        self.draw_body(self.canvas, self.segment_points, frame=frame, draw_compare=1)

    def noise_test(self):
        self.translate_x = 1
        self.translate_y = 1

    def initialize_centipede(self, frame):
        self.find_body(frame)
        self.generate_bounds()

    def flatten_params(self):
        flattened = list(itertools.chain(self.anchor_point, self.segment_angles, self.segment_lengths))
        return flattened

    def flat_to_params(self, flattened):
        anchor_len = len(self.anchor_point)
        angles_len = len(self.segment_angles)
        length_len = len(self.segment_lengths)
        anchor = flattened[0: anchor_len]
        self.anchor_point = [int(x) for x in anchor]
        self.segment_angles = flattened[anchor_len: anchor_len + angles_len]
        self.segment_length = flattened[anchor_len + angles_len:]

    def generate_bounds(self):
        # anchor, angles, length
        anchor_upper = [self.frame_width, self.frame_height]
        anchor_lower = [0, 0]
        weights_upper = [100] * self.segments
        weights_lower = [0] * self.segments

        upper_bounds = anchor_upper + weights_upper
        lower_bounds = anchor_lower + weights_lower

        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds

class MachineModel:
    def __init__(self):
        pass

    def calculate_iou(self, frame, centi_canvas):
        intersection = np.logical_and(frame, centi_canvas).sum()
        union = np.logical_or(frame, centi_canvas).sum()

        iou = intersection / union if union > 0 else 0.0
        return iou

    def fitness_function(self, params, frame, centi_object, draw_frame):
        centi_object.flat_to_params(params)
        centi_object.update_centipede()
        centi_canvas = centi_object.canvas

        score = 100 - (100 * self.calculate_iou(frame, centi_canvas))

        return score

    def optimize_centipede(self, frame, centi_object, draw_frame = None):
        params_len = len(centi_object.lower_bounds)
        lower_bounds = centi_object.lower_bounds
        upper_bounds = centi_object.upper_bounds
        bounds = (lower_bounds, upper_bounds)

        n_particles = 20

        init_pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_particles, len(lower_bounds)))
        starting_params = centi_object.flatten_params()
        for i in range(init_pos.shape[0]):
            if i % 10:
                # init_pos[i] = starting_params + np.random.uniform(-1, 1, size=len(starting_params))
                init_pos[i] = starting_params
        # print(init_pos[0])
    
        optimizer = GlobalBestPSO(n_particles = n_particles, dimensions=params_len, options={'c1': 1.0, 'c2': 1.0, 'w':0.5}, bounds = bounds, init_pos=init_pos)

        best_params, best_fitness = optimizer.optimize(
            lambda x: [self.fitness_function(p, frame, centi_object, draw_frame) for p in x],
            iters=100
        )

        return best_params, best_fitness


    
                    

def calculate_angle(pt1, pt2, pt3, full_angle = False):
    # Calculate the vectors
    vec1 = np.array(pt1) - np.array(pt2)
    vec2 = np.array(pt3) - np.array(pt2)
    
    # Normalize the vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Calculate the angle between the vectors
    angle = math.acos(np.dot(vec1, vec2))
    angle = np.degrees(angle)

    if full_angle:
        cross_product = np.cross(vec1, vec2)
        if cross_product < 0:
            angle = 360 - angle

    return angle


def process_frame(frame):
    scale_factor = 2
    big_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # adaptive thresholding to use different threshold 
    # values on different regions of the frame.
    blur = ~blur

    ret, bw = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    height, width = bw.shape
    return bw, big_frame, (height, width)



cap = cv2.VideoCapture("polya_t1_d11_skel.avi")


# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter(video_name, fourcc, fps, (int(width), int(height))) 


ret, first_frame = cap.read()
processed_frame, big_frame, (height, width) = process_frame(first_frame)
centi = centipede(frame_height=height, frame_width=width)
centi.initialize_centipede(processed_frame)
machine_model = MachineModel()

#around 44 contours
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    processed_frame, big_frame, (height, width) = process_frame(frame)

    # frame = frame[10:height, 0:width - 20, :]
    # Display the resulting frame

    # conversion of BGR to grayscale is necessary to apply this operation
    # gray = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

    # centi.noise_test()
    # iou = machine_model.calculate_iou(processed_frame, centi.canvas)
    # fitness, params = machine_model.optimize_centipede(processed_frame, centi, big_frame)
    # print(params, fitness)
    # centi.flat_to_params(params)
    centi.update_centipede(processed_frame, frame=big_frame)
    
    # print(fitness)
    # print(iou)

    cv2_segment_points = centi.segment_points[:, np.newaxis, :]
    # print(cv2_segment_points)

    cv2.imshow('Frame', big_frame)
    cv2.imshow('Cartoon', centi.canvas)
    # video.write(frame)
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    cv2.waitKey(0)

# release the video capture object
cap.release()
# video.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()