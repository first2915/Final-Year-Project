from controller import Supervisor
import math
import pose as p

#for all method relating to lidar sensor
class LidarSensor:
    def __init__(self, robot):
        self.lidar = robot.getDevice('lidar')
        self.lidar.enable(int(robot.getBasicTimeStep()))
        self.lidar.enablePointCloud() 
        self.lidar_node = robot.getFromDef('LIDAR') #supervisor
        self.lidar_pose = p.Pose()
        self.max_range = self.lidar.getMaxRange()
        self.num_point = self.lidar.getHorizontalResolution()
        
        #define the angle of each laser point in radian
        resolution = 360 / self.num_point
        self.angles = [((360 - i * resolution ) * math.pi / 180 ) for i in range(self.num_point)]
        # subtract from 360 so it starts from 0 degree and for clockwise rotation
        # convert to radian by multiplyig th math.pi/180
        
    def get_lidar_pose(self):
        return self.lidar_pose
        
    def get_max_range(self):
        return self.max_range
    
    def get_point_cloud(self):
        return self.lidar.getPointCloud()
        
    def get_layer_point_cloud(self,layer=0):
        return self.lidar.getLayerPointCloud(layer)

    def get_range_image(self):
        return self.lidar.getRangeImage()
        
    def get_lidar_node(self):
        return self.lidar_node
        
    def get_laser_angle(self):
        return self.angles
    
    def get_num_point(self):
        return self.num_point
        
    #adjust sensor angle relative to robot heading
    def adjusted_laser_angle(self,robot_pose):
        adjusted_angle = [(i + robot_pose.theta + math.pi) % (2*math.pi) - math.pi for i in self.angles]
        #adding robot heading will shift range pi to 3pi
        #subtract by pi to shift range -pi to -pi
        return adjusted_angle

    def convert_to_global(self, pose, object_pose):
        #return the global coordinate of the given object's coordinate
        lidar_matrix = pose.get_pose(self.lidar_node)
        x = lidar_matrix[0] * object_pose.x + lidar_matrix[1] * object_pose.y + lidar_matrix[2] * object_pose.z
        y = lidar_matrix[4] * object_pose.x + lidar_matrix[5] * object_pose.y + lidar_matrix[6] * object_pose.z
        z = lidar_matrix[8] * object_pose.x + lidar_matrix[9] * object_pose.y + lidar_matrix[10] * object_pose.z
        
        x += lidar_matrix[3]
        y += lidar_matrix[7]
        z += lidar_matrix[11]

        return p.Pose(x,y,z)
        
        #laser start from 0 degree(the front) and turn clockwise
    def partition_range_image(self, range_image):
        n = len(range_image)
        # Split the point cloud into 16 equal parts
        k = n//16
        parts = [range_image[i:i+k] for i in range(0, n, k)]
        
        # Group direction together
        direction = []
        direction.append(parts[0]+parts[-1]) # 0 =front
        direction.append(parts[1]+parts[2])  # 1 front right
        direction.append(parts[3]+parts[4])  # 2 right
        direction.append(parts[5]+parts[6])  # 3 back right
        direction.append(parts[7]+parts[8])  # 4 back
        direction.append(parts[9]+parts[10]) # 5 back left
        direction.append(parts[11]+parts[12])# 6 left
        direction.append(parts[13]+parts[14])# 7 front left
        
        min_distance = []
        
        for element in direction:
            min_distance.append(min(element))
            #return minimum distance of all direction 
        return min_distance
    