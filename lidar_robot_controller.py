"""lidar_robot_controller controller."""

from controller import  Supervisor
import navigationSystem as ns
import lidarSensor as ls
import display as d
import math
import pose as p

def run_robot(robot):
    timestep = int(robot.getBasicTimeStep())
    lidar = ls.LidarSensor(robot)
    nav = ns.NavigationSystem(robot,lidar)
    
    #the grid parameters are robot, the display name, lidar sensor, arena width, arena height, and resolution
    #small map
    #occupancyGrid_display = d.display(robot,'occupancyGridDisplay',lidar,5,5,15)
    
    #big map
    occupancyGrid_display = d.display(robot,'occupancyGridDisplay',lidar,10,10,10)
    
    lidar_sensor_display = d.display(robot,'lidarDisplay',lidar)
    regions = []
    reach_region = False
    destination = p.Pose(0.6,0.8) # for starting the map
    coverage_threshold = 0.95
    # test destination unreachable in small map
    #destination = p.Pose(-1.42,-0.864)
   
    
    while robot.step(timestep) != -1:
        #find new region to observe
        if reach_region:
            regions = nav.update_regions(occupancyGrid_display)
            destination = nav.utility_planning(regions, nav.get_robot_pose())
            print('Destination: ', destination.x, ' , ', destination.y) 
        reach_region = nav.bug_one(destination, 2.0, coverage_threshold )
        
        lidar_sensor_display.sensor_map(nav.get_range_image(),nav.get_robot_pose())
        
        # first appraoch: re-generate whole map
        #occupancyGrid_display.update_occupancy(nav.get_range_image(), nav.get_robot_pose())
        #occupancyGrid_display.map(nav.get_robot_pose(),destination)
        
        
        # second appraoch: generate only some cell
        occupancyGrid_display.update_occupancy_two(nav.get_range_image(), nav.get_robot_pose())
        occupancyGrid_display.map_two(nav.get_robot_pose(),destination)
        
        coverage = nav.calculate_coverage(occupancyGrid_display)
        #print('Coverage: ',coverage)
        if( coverage > coverage_threshold):
            print('Coverage: ', coverage)
            print('Simulation ends')
            timestep = -1
        
if __name__ == "__main__":
    # create the Supervised Robot instance.

    robot = Supervisor()
    run_robot(robot)
    
    

