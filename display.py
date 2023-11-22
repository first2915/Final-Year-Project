import pose as p
import math
import numpy as np
import lidarSensor as ls
import warnings

class display:
    #color for display
    GRAY = 0x999999
    BLACK = 0x000000
    WHITE = 0xFFFFFF
    BLUE = 0x0000FF
    RED = 0xF9584B
    GREEN = 0x00AB66
    
    #log odds values
    log_prev = math.log(0.5/(0.5))
    log_occ = math.log(0.90/(1-0.90))
    log_free = math.log(0.39/(1-0.39))

    alpha = 0.09 # depth of wall
    beta = 0.02 #lidar sensor cone
    
    def __init__(self, robot,display_name, lidar,arena_width = 5, arena_height = 5,resolution = 15):
        if display_name != '':
            self.display = robot.getDevice(display_name)
        else:
            self.display = None
        self.robot = robot
        self.robot_node = self.robot.getSelf() 
        self.robot_pose = p.Pose()
        self.robot_size = 0.25 
        
        self.lidar = lidar
        self.lidar_node = self.lidar.get_lidar_node() #supervisor
        self.lidar_pose = self.lidar.get_lidar_pose()
        self.max_range = self.lidar.get_max_range()
        
        self.prev_pose = p.Pose() # previous robot location in grid
        self.prev_destination = p.Pose()
        
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.resolution = resolution 
        
        if self.display is not None:
                self.display_width = self.display.getWidth()
                self.display_height = self.display.getHeight()
                self.display.setColor(self.GRAY)
                self.display.fillRectangle(0,0,self.display_width, self.display_height)
                self.scale_factor = self.display_width / arena_width 
         
        # number of row and column on each grid
        self.grid_dim = int(self.arena_width * self.resolution) #assuming arena is square
        self.grid = np.full([self.grid_dim, self.grid_dim], self.log_prev)
        self.pixel_size = self.display_width / self.grid_dim #assuming display is square

    # ===============================================================
    # HELPER METHODS END 
    # ================================================================
    
    def scale(self, s):
        return int(s * self.scale_factor)
         
    #return x value of the display
    def mapx(self, x):
        return int(self.display_width / 2.0 + self.scale(x))
        
    #return y value of the display
    def mapy(self, y):
        return int(self.display_height / 2.0 - self.scale(y))
    
    #convert log odds to probability    
    def probability(self,l):
        try:
            p = 1 - (1/(1 + math.exp(l)))
        except OverflowError:
            print('overflow')
            p =  0.92
        return p
        
    #convert distance and angle to coordinate 
    def get_coordinate(self, r, angle): 
        x = r * math.cos(angle) 
        y = r * math.sin(angle)
        return x,y
        
    #from node position to point
    def calculate_distance(self, origin,node,point):
        pose = origin.get_position(node)
        return math.sqrt((point.x - pose.x)**2+(point.y - pose.y)**2)   
    
    #angle between robot heading and the object     
    def angle_difference(self,origin,object,node):
        pose = origin.get_position(node)
        angle = math.atan2(object.y - pose.y, object.x - pose.x) # the angle between two points in radians
        return abs(angle)
        
    def get_resolution(self):
        return self.resolution
        
    def get_arena_width(self):
        return self.arena_width
        
    def get_arena_height(self):
        return self.arena_height
        
    def get_grid_dim(self):
        return self.grid_dim
        
    # ===============================================================
    # HELPER METHODS END 
    # ================================================================
    def set_color(self,i,j):
        p = self.probability(self.grid[i,j])
        if (p < 0.1):
            self.display.setColor(self.WHITE)
        elif (p < 0.4):
            self.display.setColor(0xEEEEEE)

        elif (p > 0.9):
            self.display.setColor(self.BLACK)
        elif (p > 0.6):
            self.display.setColor(0x5B5B5B)
        else:
            self.display.setColor(self.GRAY)   
    
    def inv_sensor_model_range(self,robot_pose, lidarPoint,row,column):
        x = column - robot_pose.x
        y = row - robot_pose.y
        range = math.sqrt(x ** 2 + y ** 2)
        bearing = math.atan2(y,x) - robot_pose.theta
        
        min_angle = float('inf')
        sensor_reading = float('inf')
        laser_angle = self.lidar.get_laser_angle()
        #find the closest point to the cell
        for index, angle in enumerate(laser_angle):
            temp = abs(bearing - angle)
            if temp < min_angle:
                min_angle = temp
                sensor_reading = lidarPoint[index]
        #print('reading ',sensor_reading)

        #region 3 unknown cell
        if(range > min(self.max_range, sensor_reading + self.alpha/2) or (min_angle > self.beta/2)):
            return self.log_prev
        #region 1 cell is probably occupied
        elif ((sensor_reading < self.max_range) and (abs(range - sensor_reading) < self.alpha/2)):
            return self.log_occ
        #region 2 cell is probably empty
        elif range <= sensor_reading:
            return self.log_free
        else:
            return self.log_prev
    
    
    
        # //////////////////////////////////////
        # update color whole grid 
        # /////////////////////////////////////   
        
    def update_occupancy(self, lidarPoint, robot_pose):
        #print(lidarPoint)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
               #convert grid to coordinate on arena
               y = i / self.resolution - self.arena_height/2
               x = j / self.resolution - self.arena_width/2
               self.grid[i,j] = self.grid[i,j] + self.inv_sensor_model_range( robot_pose, lidarPoint,y,x) - self.log_prev
 
    def map(self,robot_pose,destination):
        if self.display is None:
            return
        #draw background
        self.display.setColor(self.GRAY)
        self.display.fillRectangle(0,0,self.display_width, self.display_height)

        for i, row_value in enumerate(self.grid):
            for j, column_value in enumerate(row_value):
                
                # convert to display pixel coordinate 
                pixel_y = (i+1) / self.resolution - self.arena_height/2
                pixel_x = j / self.resolution - self.arena_width/2
                self.set_color(i,j)

                self.display.fillRectangle(self.mapx(pixel_x),self.mapy(pixel_y), int(self.pixel_size), int(self.pixel_size))
     
        #draw robot 
        self.display.setColor(self.RED)
        self.display.fillRectangle(int(self.mapx(robot_pose.x)-self.scale(self.robot_size)/2),
                                        int(self.mapy(robot_pose.y)-self.scale(self.robot_size)/2),
                                        self.scale(self.robot_size),
                                        self.scale(self.robot_size))
        #draw robot heading line
        self.display.setColor(self.BLUE)
        line_x, line_y = self.get_coordinate(self.robot_size/2, robot_pose.theta)
        self.display.drawLine(self.mapx(robot_pose.x), 
                                self.mapy(robot_pose.y),
                                self.mapx(robot_pose.x + line_x), 
                                self.mapy(robot_pose.y + line_y))
        
        #draw destination                        
        self.display.setColor(self.GREEN)
        self.display.fillRectangle(self.mapx(destination.x),self.mapy(destination.y), int(self.pixel_size), int(self.pixel_size))
    
    
    
    # //////////////////////////////////////
    # update color only particular pixel 
    # /////////////////////////////////////
    
    def update_occupancy_two(self, lidarPoint, robot_pose):
        #print(lidarPoint)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
               #convert grid to coordinate on arena
               y = i / self.resolution - self.arena_height/2
               x = j / self.resolution - self.arena_width/2
               temp = self.grid[i,j]
               self.grid[i,j] = self.grid[i,j] + self.inv_sensor_model_range( robot_pose, lidarPoint,y,x) - self.log_prev
               #if log odds change, color it
               if (temp != self.grid[i,j]):
                   self.map_pixel_two(i,j)
                                 
    def map_pixel_two(self,i,j):
        pixel_y = (i+1) / self.resolution - self.arena_height/2
        pixel_x = j / self.resolution - self.arena_width/2
 
        self.set_color(i,j)
        self.display.fillRectangle(self.mapx(pixel_x),self.mapy(pixel_y), int(self.pixel_size), int(self.pixel_size))
 

    def map_two(self,robot_pose,destination):
        #color the robot previous location
        # convert arena coordinate to grid
        i = int((self.prev_pose.y + self.arena_height/2) * self.resolution)
        j = int((self.prev_pose.x + self.arena_width/2) * self.resolution)
        
        self.set_color(i,j)
        # delete robot previous coordinate
        self.display.fillRectangle(int(self.mapx(self.prev_pose.x)-self.scale(self.robot_size)/2),
                                        int(self.mapy(self.prev_pose.y)-self.scale(self.robot_size)/2),
                                        self.scale(self.robot_size),
                                        self.scale(self.robot_size))
        
        #draw robot 
        self.display.setColor(self.RED)
        self.display.fillRectangle(int(self.mapx(robot_pose.x)-self.scale(self.robot_size)/2),
                                        int(self.mapy(robot_pose.y)-self.scale(self.robot_size)/2),
                                        self.scale(self.robot_size),
                                        self.scale(self.robot_size))
        #draw robot heading line
        self.display.setColor(self.BLUE)
        line_x, line_y = self.get_coordinate(self.robot_size/2, robot_pose.theta)
        self.display.drawLine(self.mapx(robot_pose.x), 
                                self.mapy(robot_pose.y),
                                self.mapx(robot_pose.x + line_x), 
                                self.mapy(robot_pose.y + line_y))
        
        # delete last destination point
        i = int((self.prev_destination.y + self.arena_height/2) * self.resolution)
        j = int((self.prev_destination.x + self.arena_width/2) * self.resolution)
 
        self.set_color(i,j)
        self.display.fillRectangle(self.mapx(self.prev_destination.x),self.mapy(self.prev_destination.y), int(self.pixel_size), int(self.pixel_size))
        
        # color the current destination                        
        self.display.setColor(self.GREEN)
        self.display.fillRectangle(self.mapx(destination.x),self.mapy(destination.y), int(self.pixel_size), int(self.pixel_size))
        self.prev_destination = destination
        
        self.prev_pose = robot_pose

    # //////////////////////////////////////
    # /////////////////////////////////////
        
    def sensor_map(self, sensor,robot_pose):
        if self.display is None:
            return
        
        #draw background    
        self.display.setColor(self.GRAY)
        self.display.fillRectangle(0,0,self.display_width, self.display_height)
        
        #draw sensor line
        laser_angle = self.lidar.adjusted_laser_angle(robot_pose)
        self.display.setColor(self.BLUE)
        for i, angle in enumerate(laser_angle):
            if not (math.isinf(sensor[i])):
                x, y = self.get_coordinate(sensor[i], angle)
                self.display.drawLine(int(self.display_width / 2.0), int(self.display_height / 2.0),self.mapx(x), self.mapy(y))      
        
        #draw robot body
        self.robot_pose  = self.robot_pose.get_position(self.robot_node)
        self.display.setColor(self.WHITE)
        self.display.fillRectangle(int((self.display_width / 2.0)-self.scale(self.robot_size)/2),
                                    int((self.display_height / 2.0)-self.scale(self.robot_size)/2 ),
                                    self.scale(self.robot_size),
                                    self.scale(self.robot_size))
        #draw robot heading
        self.display.setColor(self.BLACK)
        line_x, line_y = self.get_coordinate(self.robot_size/2, robot_pose.theta)
        self.display.drawLine(int(self.display_width / 2.0),
                                    int(self.display_height / 2.0),
                                    self.mapx(line_x), 
                                    self.mapy(line_y))
                                    
                                    
                                    