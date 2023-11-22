from controller import Supervisor
import pose as p
import lidarSensor as ls
import math
from enum import Enum
import numpy as np

class State(Enum):
    START = 0
    ALIGN_HEADING = 1
    MOVE_TO_GOAL = 2
    WALL_FOLLOWING = 3
    REVERSE = 4
    FORWARD = 5
    STOP = 6
    COMPLETE = 7

class NavigationSystem:
    def __init__(self, robot,lidar):
        self.robot = robot                        # Supevisor, the robot
        self.robot_node = self.robot.getSelf()    # Handle to supervisor node
        self.robot_pose = p.Pose()
  
        self.leftMotor= self.robot.getDevice('leftMotor')
        self.rightMotor = self.robot.getDevice('rightMotor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.threshold = 0.35 #distance to wall
        self.destination_threshold = 0.25 #close to destination
        
        # Initialise motor velocity
        self.leftMotor.setVelocity(0.0);
        self.rightMotor.setVelocity(0.0);
        
        self.lidar = lidar
        self.lidar_node = self.lidar.get_lidar_node() #supervisor
        self.lidar_pose = self.lidar.get_lidar_pose()
        
        self.global_object_coordinate = []
        self.local_object_coordinate = []
        self.range_image = [0.0] * self.lidar.get_num_point()
        self.min_coordinate = []
        self.min_distance = []
        
        #for wall following
        self.explored_whole_obstacle = False
        self.exploring_obstacle = False
        self.shortest_distance = float('inf')
        self.shortest_coordinate = p.Pose()
        self.start_coordinate = p.Pose() # coordinate in which the robot first met obstacle
        self.time_elapsed = 0 # use for wall following to calculate new distance to destination
        self.min_exploring_time = 3500
        self.exploration_time= 300
        self.arrive = 0 # check if it come back to same place after explored whole obstacle
        
        self.state = State.START
        # to check if robot stuck
        self.time_idle = 0 # use to check if the robot stuck for too long
        self.stuck_threshold = 1400.0 # if time_moving is more than this then robot stuck
        self.reverse_time = 0
        self.reverse_threshold = 300
        self.robot_prev = p.Pose() # to compare if robot move within time_stuck period
        self.prev_state = State.START
        
        self.timestep = robot.getBasicTimeStep()
        
        #occupancy grid coverage
        self.coverage = 0

        
    # ========================================================
    # METHODS FOR EXTERNAL USE
    # ========================================================
    
    def get_global_object_coordinate(self):
        return self.global_object_coordinate   
    
    def get_local_object_coordinate(self):
        return self.local_object_coordinate 
        
    def get_range_image(self):
        return self.range_image
        
    def get_robot_pose(self):
        return self.robot_pose
        
    def get_min_coordinate(self):
        return self.min_coordinate
        
    # ========================================================
    # HELPER METHODS 
    # ========================================================
    
    def angle_difference(self,origin,object,node):
        pose = origin.get_position(node)
        angle = math.atan2(object.y - pose.y, object.x - pose.x) # the angle between two points
        return abs(angle)
    
    def calculate_distance(self, origin,destination,node):
        pose = origin.get_position(node)
        return math.sqrt((destination.x - pose.x)**2+(destination.y - pose.y)**2)   

    # =========================================================
    # METHOD FOR CALCULATING REGION
    # =========================================================
        
    def update_regions(self,occupancy_grid):
        resolution = occupancy_grid.get_resolution() 
        arena_width = occupancy_grid.get_arena_width()
        arena_height = occupancy_grid.get_arena_height()
        grid_dim = occupancy_grid.get_grid_dim()
        
        regions = [] # sub regions: store benefit and middle indices
        
        sub_region = [] # hold smaller region of the grid
        middle_indices = [] #hold the middle coordinate of each sub region
        
        for row in range(0, grid_dim, resolution):
            for col in range(0, grid_dim, resolution):
                # partition grid into smaller region of size resolution
                sub_grid =  occupancy_grid.grid[row: row+resolution, col: col+resolution]
                #get the middle coordinate of the sub region
                middle_row = row + resolution//2
                middle_col = col + resolution//2
                x = middle_col / resolution - arena_width/2
                y = middle_row / resolution - arena_height/2
                middle_indices.append(p.Pose(x,y))
                sub_region.append(sub_grid)
        
        #  get the coverage of each sub region   
        #more coverage = less benefit 
        index = 0  
        for sub in sub_region:
            coverage = 28.0
            for i in range(resolution):
                for j in range(resolution):
                    prob = occupancy_grid.probability(sub[i][j])
                    if(prob < 0.1) or (prob > 0.9):
                        coverage -= 0.1
            temp = [coverage, middle_indices[index]]
            regions.append(temp)
            index += 1
                
        return regions
        
    def utility_planning(self, regions,robot_pose):
        max_utility = float('-inf')
        max_x = 0.0
        max_y = 0.0

        #find destination
        for i in range(len(regions)):
            region = regions[i]
            utility = self.calculate_utility(robot_pose,region[0], region[1])
            if utility > max_utility:
                max_utility = utility
                max_x = region[1].x
                max_y = region[1].y
        destination = p.Pose(max_x,max_y)

        return destination #return the maximum utility coordinate
        
        # highest benefit = cost * 2 + extra
    def calculate_utility(self,robot_pose,benefit,destination):
        return benefit - self.calculate_cost(robot_pose, destination)

    def calculate_cost(self, robot_pose,destination):
        #print('cost: ',math.sqrt((destination.x - robot_pose.x)**2+(destination.y - robot_pose.y)**2))
        return math.sqrt((destination.x - robot_pose.x)**2+(destination.y - robot_pose.y)**2)   

        #coverage of the whole grid
    def calculate_coverage(self,occupancy_grid):
        for i, row_value in enumerate(occupancy_grid.grid):
            for j, column_value in enumerate(row_value):
                p = occupancy_grid.probability(column_value)
                if(p < 0.1) or (p > 0.9):
                    self.coverage += 1.0
        # normalise coverage
        self.coverage = self.coverage / (occupancy_grid.get_grid_dim()**2)
        return self.coverage
 
    # =========================================================
    # METHOD FOR BUG ONE
    # =========================================================

    def check_range_obstacle(self):
        #return a minimum distance of each direction
        self.robot_pose = self.robot_pose.get_position(self.robot_node)
        self.range_image = self.lidar.get_range_image()
        min_distance = self.lidar.partition_range_image(self.range_image)
        return min_distance
        
    def new_point_closer(self,destination,robot_pose):
        current_distance = math.sqrt((destination.x - robot_pose.x)**2+(destination.y - robot_pose.y)**2)  
        return current_distance < self.shortest_distance
        
    def correct_heading(self, destination,robot_pose):
        heading_diff = self.angle_to_goal(destination,robot_pose) #angle between robot and destination
        # return true if robot heading is toward the destination
        heading_diff_threshold = 0.05
        return heading_diff <= heading_diff_threshold and heading_diff >= -heading_diff_threshold
    
    def angle_to_goal(self,destination,robot_pose):
        destination_angle = math.atan2(destination.y - robot_pose.y, destination.x - robot_pose.x)
        #the difference in angle between robot heading and the destination
        heading_diff = destination_angle - robot_pose.theta 
        return heading_diff

    def at_destination(self, destination, robot_pose):
        #threshold considered to be at destination
        return self.calculate_cost(robot_pose, destination) <= self.destination_threshold
        
    def robot_stuck(self, robot_pose):
        #print('distance fron prev point: ',self.calculate_cost(self.robot_prev, robot_pose))
        #print('time idle: ', self.time_idle)
        if self.calculate_cost(self.robot_prev, robot_pose) < 0.0028:
            self.time_idle += self.timestep
        elif self.calculate_cost(self.robot_prev, robot_pose) >= 0.0056:
            self.time_idle = 0
        if self.time_idle > self.stuck_threshold:
            return True
        return False
            
    #for follow obstacle
    def reset(self):
        #print('reset')
        self.explored_whole_obstacle = False
        self.exploring_obstacle = False
        self.shortest_distance = float('inf')
        self.shortest_coordinate = p.Pose()
        self.start_coordinate = p.Pose()
        self.time_elapsed = 0 
        self.time_idle = 0
        self.reverse_time = 0
        self.arrive = 0
        
        
    # =========================================================
    #  BUG ONE
    # =========================================================
        
    # follow right wall
    def wall_following(self,velocity, threshold, obstacle):
        
        av = (velocity*3)
        left_av = av
        right_av = av

        #check front obstacle, turn left
        if (obstacle[0] < threshold):
            left_av = -av
            right_av = av
        # front right detect object, slowly turn left
        elif (obstacle[1] < threshold):
            left_av = av/3
            right_av = av 
        else:
            #checking right wall
            wall_distance = obstacle[2]
                
            if(wall_distance == threshold):
                left_av = av
                right_av = av
                
            #getting close to wall, turn away from wall, turn left
            elif(wall_distance < threshold):
                left_av = av/3
                right_av = av
            #getting away from wall, turn to wall, turn right
            else:
                left_av = av
                right_av = av/6
        
        self.leftMotor.setVelocity(left_av)
        self.rightMotor.setVelocity(right_av)
    
    def bug_one(self, destination, velocity, coverage_threshold ):
        obstacle = self.check_range_obstacle()
        robot_pose = self.robot_pose
        #print('State: ', self.state)
        if(self.coverage > coverage_threshold):
            self.state = State.COMPLETE
        
        if self.state == State.START:
            if self.correct_heading(destination, robot_pose):
                self.state = State.MOVE_TO_GOAL
            else:
                self.state = State.ALIGN_HEADING
                
        elif self.state == State.ALIGN_HEADING:
            if self.correct_heading(destination, robot_pose):
                self.state = State.MOVE_TO_GOAL
            else:
                #rotate robot
                av = (velocity*2)
                left_av = -av
                right_av = av
                self.leftMotor.setVelocity(left_av)
                self.rightMotor.setVelocity(right_av)
                
        elif self.state == State.MOVE_TO_GOAL:
            if self.at_destination(robot_pose,destination):
                self.state = State.STOP
                self.reset()
                #check if robot stuck
            elif self.robot_stuck(robot_pose):
                self.state = State.REVERSE
                self.prev_state = State.MOVE_TO_GOAL
                
            elif (obstacle[0] < self.threshold ): # detect obstacle in front
                self.state = State.WALL_FOLLOWING
                self.time_idle = 0
                self.exploring_obstacle = True
                self.shortest_coordinate = robot_pose
                self.start_coordinate = robot_pose 
                self.shortest_distance = math.sqrt((destination.x - robot_pose.x)**2+(destination.y - robot_pose.y)**2)
                
            else: # move forward
                av = (velocity*3)
                left_av = av
                right_av = av
                self.leftMotor.setVelocity(left_av)
                self.rightMotor.setVelocity(right_av)
            self.robot_prev = robot_pose
            
        elif self.state == State.WALL_FOLLOWING:
            if self.at_destination(destination, robot_pose):
                #print('at dest')
                self.reset()
                self.state = State.STOP
            #back to where robot first detect obstacle
            elif(self.time_elapsed > self.min_exploring_time and self.at_destination(self.start_coordinate,robot_pose)):
                 #print('explored whole obstacle')
                 self.explored_whole_obstacle = True
                 self.exploring_obstacle = False
                 self.time_elapsed = 0
            
            # robot stuck    
            elif (self.robot_stuck(robot_pose)):
                #print('stuck')
                self.state = State.REVERSE
                self.prev_state = State.WALL_FOLLOWING
            elif self.exploring_obstacle:
                #print('following obs')
                self.time_elapsed += self.timestep
                if(self.time_elapsed > self.exploration_time):
                    if self.new_point_closer(destination, robot_pose):
                        self.shortest_distance = math.sqrt((destination.x - robot_pose.x)**2+(destination.y - robot_pose.y)**2)
                        self.shortest_coordinate = robot_pose
                self.wall_following(velocity, self.threshold, obstacle)
            elif self.explored_whole_obstacle:
                #print('explored whole obstacle')
                # robot is back at the closest point again after for second time
                if self.arrive == 1:
                    #print('arrive 1')
                    self.reset()
                    self.state = State.STOP
                #robot at closest point to destination
                elif self.at_destination(self.shortest_coordinate, robot_pose):
                    #print('arrive at shortest distance')
                    self.state = State.ALIGN_HEADING
                    self.arrive += 1
                else:
                    self.wall_following(velocity, self.threshold, obstacle)
            self.robot_prev = robot_pose
                
        elif self.state == State.REVERSE:
            if self.reverse_time > self.reverse_threshold:
                self.state = self.prev_state
                self.time_idle = 0 # reset
                self.reverse_time = 0
                #print('reverse complete')
            else:
            #back up the robot
                av = (velocity*3)
                left_av = -av
                right_av = -av
                self.leftMotor.setVelocity(left_av)
                self.rightMotor.setVelocity(right_av)
                self.reverse_time += self.timestep
                #print('reversing time: ',self.reverse_time)
        
        elif self.state == State.STOP:
            #stop robot movement
            self.leftMotor.setVelocity(0)
            self.rightMotor.setVelocity(0)
            self.state = State.START
            return True
            
        elif self.state == State.COMPLETE:
            self.leftMotor.setVelocity(0)
            self.rightMotor.setVelocity(0)
        
        return False