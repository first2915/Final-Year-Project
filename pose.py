from controller import Supervisor
import math

class Pose():
    def __init__(self,x=0.0,y=0.0,z=0.0, theta = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
    
    def set_position(self,x,y,z, theta):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        
    def get_orientation(self,node):
        orientation = node.getOrientation()
        pitch = -1*math.asin(orientation[6])
        yaw = math.atan2(orientation[3]/math.cos(pitch), orientation[0]/math.cos(pitch))
        return yaw
        
    def get_position(self,node):
        position = node.getPosition()
        rotation = self.get_orientation(node)
        return Pose(position[0], position[1], position[2], rotation)
        
    def get_pose(self,node):
        return node.getPose()

        

  