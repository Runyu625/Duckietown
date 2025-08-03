import subprocess
import rospy
import tf
import numpy as np
import sys

from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Header


if __name__=="__main__":

    active_robots = []

    for robot_name in [f"td{number:02d}" for number in range(43)]:        
        try:
            subprocess.check_call(["ping", "-c1",f"{robot_name}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            active_robots.append(robot_name)
        except:
            pass

    print(f"Active robots: {active_robots}")

    if active_robots == []:
        print("No robots detected.")
        sys.exit(0)
        

    #TODO: make script to deactive all the OptiTrack RB, except those of the active robots
    #(not necessary currently, just decluters the network and makes running experiments easier)


    rospy.init_node("robot_pose_receiver")
    tlistener = tf.TransformListener()    
    
    while not rospy.is_shutdown():

        for robot in active_robots:
            try:
                (trans, rot) = tlistener.lookupTransform(robot, "/world", rospy.Time(0))
                rot = tf.transformations.euler_from_quaternion(rot)
                yaw = rot[2] * 180 / np.pi # Getting Yaw (3rd Euler Angle) and converting to Deg (from Rad)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("failed")
                continue


            # TODO: Fix this to be the same as in the duckietown gym
            print(f"trans={trans}")     # OptiTrack -> (-x,+y,-z); ahead is -x, to right is +y, up is -z 
            # [note: in the sim, the coordinates are not in meters but the units are instead the size of the black squares]
            print(f"rot={yaw}")         # OptiTrack -> CW is +ve, bounded in [-180, +180], towards -x is 0
            
            angle = -yaw            
            pos = [-trans[0]/TILE_SIZE, 0.0, trans[1]/TILE_SIZE]

            # load torch policy

            # calculate obersvations using optitrack data (i.e. speed, pos, ori, ....)

            #action = policy(observations)

            # convert action to right and left wheel speed, bounded [-1, 1], if not already done by policy
            # if we need to convert, there is code on ust0 to fetch robot calibration and convert lin + ang vel to wheel speed


            # Not sure how we want to deal with publishing commands, before this was dealt with in a duckiebot class, mb now depending on wheter
            # we are running sim or reality (or if a given duck is meant to be simulated or real, due to the possibility of mixed real and sim)
            # we can instantiate a different duckiebot class? (for each robot there must be a ROS publisher)
            #
            # pub_wheelVel = rospy.Publisher(
            #     f'/{robot_name}/wheels_driver_node/wheels_cmd',
            #     WheelsCmdStamped,
            #     queue_size=1)

            # wheelVel_msg = WheelsCmdStamped()
            # wheelVel_msg.header = Header()
            # wheelVel_msg.header.stamp = rospy.Time.now()
            # wheelVel_msg.vel_left, wheelVel_msg.vel_right = [wheel_vel_left, wheel_vel_right]

            # pub_wheelVel.publish(wheelVel_msg)
