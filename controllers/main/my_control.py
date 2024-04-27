# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import heapq
from collections import deque

# Global variables
on_ground = True
height_desired = 0.8
timer = None
startpos = None
timer_done = None
var = dict(control_command=[0, 0, 0, 0], state='initial', left=0, right=0, straight=0)
landing_spots = deque()
plt_find = False


# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py lines 296-323. 
# The "item" values that you can later use in the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "range_down": Downward range finder distance (Used instead of Global Z distance)
# "range_front": Front range finder distance
# "range_left": Leftward range finder distance 
# "range_right": Rightward range finder distance
# "range_back": Backward range finder distance
# "yaw": Yaw angle (rad)

# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global on_ground, startpos, var, path, landing_spots, plt_find

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)
    print(var['state'])
    # Take off
    if startpos is None:
        for i in np.arange(3.5, 5, 0.3):
            for j in np.arange(0, 3, 0.3):
                landing_spots.append((i//0.15, j//0.15))
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down']]    
    if on_ground and sensor_data['range_down'] < 0.75:
        var['control_command'] = [0.0, 0.0, height_desired, 0.0]
        var['state'] = 'check_angles'
        return var['control_command']
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----
    map = occupancy_map(sensor_data)
    disc_pos_x = int(sensor_data['x_global'] // res_pos)
    disc_pos_y = int(sensor_data['y_global'] // res_pos)

    if var['state'] == 'check_angles': 
        check_angles(sensor_data, var)

    if var['state'] == 'path_finding':
        start = (disc_pos_x, disc_pos_y)
        goal, grid = goal_definition(map.copy(), landing_spots, var)
        path = a_star(grid, start, goal)
        var['state'] = 'path_following'      
        print(f"path finding : {path}")

    if var['state'] == 'path_following':
        disc_pos_x, disc_pos_y
        path_following(sensor_data, disc_pos_x, disc_pos_y, var, path)
        if disc_pos_x >= 3.5//res_pos and plt_find == False:
            var['state'] = 'plateform_finding'
            plt_find = True
        print(f"path following : {path}")
    
    if var['state'] == 'plateform_finding':
        start = (disc_pos_x, disc_pos_y)
        goal, grid = goal_definition(map.copy(), landing_spots, var)
        path = a_star(grid, start, goal)
        var['state'] = 'path_following'
        var['control_command'] = [0.0, 0.0, height_desired, 0.0]
        print(f"plateform fiding : {path}")

    return var['control_command'] # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]

def goal_definition(grid, landing_spots, var):
    goal = (0, 0)
    for i, row in enumerate(grid):
        for j, element in enumerate(row):
            if element <= 0.2:
                grid[i][j] = 1
            else:
                grid[i][j] = 0
                if i > goal[0] and var['state'] == 'path_finding':
                    goal = (i, j)
    if var['state'] == 'plateform_finding':
        print(f"landing_spots_before_while : {landing_spots}")        
        while goal == (0, 0):
            print(f"landing_spots_in_while : {landing_spots}")  
            landing_spot = landing_spots[0]
            landing_spot_int = (int(landing_spot[0]), int(landing_spot[1]))
            if grid[landing_spot_int[0]][landing_spot_int[1]] == 1:
                landing_spots.popleft()
                goal = landing_spots[0]
            else:
                goal = landing_spots[0]
        print(f"landing_spots_after_while : {landing_spots}")  
        landing_spots.popleft()
    return goal, grid   

def a_star(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    def reconstruct_path(came_from, current):
        path = deque()
        while current in came_from:
            path.appendleft(current)
            current = came_from[current]
        return path
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in [(current[0]+1, current[1]), (current[0]-1, current[1]), (current[0], current[1]+1), (current[0], current[1]-1)]:
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and not grid[neighbor[0]][neighbor[1]]:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def check_angles(sensor_data, var):
    if sensor_data['yaw'] > 1:
        var['left'] = 1
    if sensor_data['yaw'] < -1:
        var['right'] = 1
    if (sensor_data['yaw'] < 0.01 and sensor_data['yaw'] > -0.01 and var['left'] == 1 and var['right'] == 1):
        var['straight'] = 1

    if var['left'] == 0:
        var['control_command'] = [0.0, 0.0, height_desired, 1]
    elif var['right'] == 0:
        var['control_command'] = [0.0, 0.0, height_desired, -1]
    elif var['straight'] == 0:
        var['control_command'] = [0.0, 0.0, height_desired, 1]
    else:
        if plt_find == False:
            var['state'] = 'path_finding'
        else:
            var['state'] = 'plateform_finding'
        var['left'] = 0
        var['right'] = 0
        var['straight'] = 0

def path_following(sensor_data, disc_pos_x, disc_pos_y, var, path):
    if len(path) == 0:
        if plt_find == False:
            var['state'] = 'check_angles'
            return
        else:
            if len(landing_spots) % 7 == 0:
                var['state'] = 'check_angles'
                return
            else:
                var['state'] = 'plateform_finding'
                return
        
    if disc_pos_x < path[0][0] and disc_pos_y == path[0][1]:
        var['control_command'] = [0.25, 0.0, height_desired, 0]
        if sensor_data['range_front'] < res_pos:
            var['control_command'] = [-0.3, 0.0, height_desired, 0]
            var['state'] = 'check_angles'
             
    elif disc_pos_x > path[0][0] and disc_pos_y == path[0][1]:
        var['control_command'] = [-0.25, 0.0, height_desired, 0]
        if sensor_data['range_back'] < res_pos:
             var['control_command'] = [0.3, 0.0, height_desired, 0]
             var['state'] = 'check_angles'
             
    elif disc_pos_x == path[0][0] and disc_pos_y < path[0][1]:
        var['control_command'] = [0.0, 0.25, height_desired, 0]
        if sensor_data['range_left'] < res_pos:
             var['control_command'] = [0.0, -0.3, height_desired, 0]
             var['state'] = 'check_angles'
             
    elif disc_pos_x == path[0][0] and disc_pos_y > path[0][1]:
        var['control_command'] = [0.0, -0.25, height_desired, 0]
        if sensor_data['range_right'] < res_pos:
             var['control_command'] = [0.0, 0.3, height_desired, 0]
             var['state'] = 'check_angles'         
    
    if disc_pos_x == path[0][0] and disc_pos_y == path[0][1]:
        path.popleft()

    '''
    # ---- STATE DETERMINATION ----
    if var['state'] == 'forward':
        forward(disc_pos_x, disc_pos_y, var, map)
    
    if var['state'] == 'check_angles': 
        check_angles(sensor_data, var)

    if var['state'] == 'stop':
        stop(disc_pos_x, disc_pos_y, var, map)

    print(var['state'])

    if var['state'] == 'left':
        left(disc_pos_y, disc_pos_x, var, map)

    if var['state'] == 'right':
        right(disc_pos_y, disc_pos_x, var, map)
            
    # ---- COMMAND DEPENDING ON THE STATE ----
    if var['state'] == 'initial':
        var['control_command'] = [0.0, 0.0, height_desired, 0.0]
    elif var['state'] == 'stop':
        var['control_command'] = [0.0, 0.0, height_desired, 0.0]
    elif var['state'] == 'forward':
        var['control_command'] = [0.3, 0.0, height_desired, 0.0]
    elif var['state'] == 'yaw_left':
        var['control_command'] = [0.0, 0.0, height_desired, 1]
    elif var['state'] == 'yaw_right':
        var['control_command'] = [0.0, 0.0, height_desired, -1]
    elif var['state'] == 'left':
        var['control_command'] = [0.0, 0.3, height_desired, 0.0]
    elif var['state'] == 'right':
        var['control_command'] = [0.0, -0.3, height_desired, 0.0]
    elif var['state'] == 'back':
        var['control_command'] = [0.0, 0.0, height_desired, 0.0]
    elif var['state'] == 'land':
        var['control_command'] = [0.0, 0.0, 0.0, 0.0] 

    return var['control_command'] # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]

def forward(disc_pos_x, disc_pos_y, var, map):
    for i, element in enumerate(reversed(map[disc_pos_x+1:min(disc_pos_x + 5, len(map))])):
        if (element[disc_pos_y] > 0 and element[disc_pos_y + 1] > 0 and element[disc_pos_y - 1] > 0
            and element[disc_pos_y + 2] > 0 and element[disc_pos_y - 2] > 0):
            state = 'forward'
            break
        elif (element[disc_pos_y] > -0.3 and element[disc_pos_y + 1] > -0.3 and element[disc_pos_y - 1] > -0.3
              and element[disc_pos_y + 2] > -0.3 and element[disc_pos_y - 2] > -0.3):
            var['state'] = 'check_angles'
        else:
            var['state'] ='stop'
            break

def check_angles(sensor_data, var):
    if sensor_data['yaw'] > 1:
        var['left'] = 1
    if sensor_data['yaw'] < -1:
        var['right'] = 1
    if (sensor_data['yaw'] < 0.05 and sensor_data['yaw'] > -0.05 and var['left'] == 1 and var['right'] == 1):
        var['straight'] = 1

    if var['left'] == 0:
        var['control_command'] = [0.0, 0.0, height_desired, 1]
    elif var['right'] == 0:
        var['control_command'] = [0.0, 0.0, height_desired, -1]
    elif var['straight'] == 0:
        var['control_command'] = [0.0, 0.0, height_desired, 1]
    else:
        var['state'] = 'forward'
        var['left'] = 0
        var['right'] = 0
        var['straight'] = 0

def stop(disc_pos_x, disc_pos_y, var, map):
    x_column = map[disc_pos_x, :]
    block_1 = 20
    block_2 = 20
    for j, element in enumerate(x_column[disc_pos_y:min(disc_pos_y + 7, len(map))]): 
        if element < -0.5:
            block_1 = j
            break
    for k, element in enumerate(reversed(x_column[max(disc_pos_y - 7, 0):disc_pos_y])):
        if element < -0.5:
            block_2 = k
            break
    if block_1 == block_2:
        if disc_pos_y > max_y/(2*res_pos):
            var['state'] = 'right'
        else:
            var['state'] = 'left'
    elif block_1 < block_2:
        var['state'] = 'right'
    else:
        var['state'] = 'left'
    print(block_1, block_2)

def left(disc_pos_y, disc_pos_x, var, map):
    var['state'] = 'forward'
    if disc_pos_y < 4:
        var['state'] = 'right'
    else:
        for i, element in enumerate(map[disc_pos_x:min(disc_pos_x + 5, len(map))]): 
            if (element[disc_pos_y] < 0 or element[disc_pos_y + 1] < 0 or element[disc_pos_y - 1] < 0):
                var['state'] = 'left'
                break  

def right(disc_pos_y, disc_pos_x, var, map):
    var['state'] = 'forward'
    if disc_pos_y < 4:
        var['state'] = 'left'
    else:
        for i, element in enumerate(map[disc_pos_x:min(disc_pos_x + 5, len(map))]): 
            if (element[disc_pos_y] < 0 or element[disc_pos_y + 1] < 0 or element[disc_pos_y - 1] < 0):
                var['state'] = 'right'
                break '''

    
# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 3.0 # meter
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.15 # meter
conf = 0.2 # certainty given by each measurement
t = 0 # only for plotting

map = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied

def occupancy_map(sensor_data):
    global map, t
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    yaw = sensor_data['yaw']
    
    for j in range(4): # 4 sensors
        yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']
        
        for i in range(int(range_max/res_pos)): # range is 2 meters
            dist = i*res_pos
            idx_x = int(np.round((pos_x - min_x + dist*np.cos(yaw_sensor))/res_pos,0))
            idx_y = int(np.round((pos_y - min_y + dist*np.sin(yaw_sensor))/res_pos,0))

            # make sure the current_setpoint is within the map
            if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > range_max:
                break

            # update the map
            if dist < measurement:
                map[idx_x, idx_y] += conf
            else:
                map[idx_x, idx_y] -= conf
                break
    
    map = np.clip(map, -1, 1) # certainty can never be more than 100%

    # only plot every Nth time step (comment out if not needed)
    if t % 50 == 0:
        plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        plt.savefig("map.png")
        plt.close()
    t +=1

    return map


# Control from the exercises
index_current_setpoint = 0
def path_to_setpoint(path,sensor_data,dt):
    global on_ground, height_desired, index_current_setpoint, timer, timer_done, startpos

    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down']]    
    if on_ground and sensor_data['range_down'] < 0.49:
        current_setpoint = [startpos[0], startpos[1], height_desired, 0.0]
        return current_setpoint
    else:
        on_ground = False

    # Start timer
    if (index_current_setpoint == 1) & (timer is None):
        timer = 0
        print("Time recording started")
    if timer is not None:
        timer += dt
    # Hover at the final setpoint
    if index_current_setpoint == len(path):
        # Uncomment for KF
        control_command = [startpos[0], startpos[1], startpos[2]-0.05, 0.0]

        if timer_done is None:
            timer_done = True
            print("Path planing took " + str(np.round(timer,1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[index_current_setpoint]
    x_drone, y_drone, z_drone, yaw_drone = sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down'], sensor_data['yaw']
    distance_drone_to_goal = np.linalg.norm([current_setpoint[0] - x_drone, current_setpoint[1] - y_drone, current_setpoint[2] - z_drone, clip_angle(current_setpoint[3]) - clip_angle(yaw_drone)])

    # When the drone reaches the goal setpoint, e.g., distance < 0.1m
    if distance_drone_to_goal < 0.1:
        # Select the next setpoint as the goal position
        index_current_setpoint += 1
        # Hover at the final setpoint
        if index_current_setpoint == len(path):
            current_setpoint = [0.0, 0.0, height_desired, 0.0]
            return current_setpoint

    print(current_setpoint)
    return current_setpoint

def clip_angle(angle):
    angle = angle%(2*np.pi)
    if angle > np.pi:
        angle -= 2*np.pi
    if angle < -np.pi:
        angle += 2*np.pi
    return angle