# Imports
import numpy as np

#import rospy

from .natnet import NatNetClient


# cv-tsl-13 OptiTrack Parameters
server_ip     = "192.168.0.13"
multicast_ip  = "239.255.42.99"
command_port  = 1510
data_port     = 1511



class OptiTrack:

    def __init__(self, robot_names): 
        
        self.natnet_client = NatNetClient(
            server              = server_ip,
            multicast           = multicast_ip,
            commandPort         = command_port,
            dataPort            = data_port,
            rigidBodyListener   = self.cbReceivedRBFrame,
            newFrameListener    = None
            #newFrameListener    = self.cbReceivedNewFrame
        )

        self.robots_pos = {name: [] for name in robot_names}
        self.robots_ang = {name: [] for name in robot_names}

        self.natnet_client.run()

    def cbReceivedRBFrame(self, id, position, rotation):
        """receiveRigidBodyFrame is a callback function that gets connected
        to the NatNet client. Called once per rigid body per frame. There are 
        n rigid body frames per camera frame, where n is the number of rigid bodies.

        Args:
            id ([int]): rigid body ID as defined in OptiTrack
            position ([array_like]): (x,y,z) of rigid body, following convention of OptiTrack
            rotation ([array_like]): orientation unit quaternion
        """
        
        # Convert to orientation of duckietown-gym
        self.robots_pos[f"td{id:02d}"] = np.array([position[0], position[2], -position[1]])

        self.robots_ang[f"td{id:02d}"] = RMatrix2Euler(Quat2RMatrix(rotation), "rad")



########################################################################################################################
###############################################  Quaternion Conversion #################################################
########################################################################################################################
"""
References:
[1] - "A Tutorial on Euler Angles and Quaternions" - Moti Ben-Ari
[2] - "Quaternion kinematics for the error-state KF" - Joan Solà 
"""

def Quat2RMatrix(quaternion):
    """Quat2RMatrix gets the rotation matrix equivalent of a quaternion

    Args:
        quaternion ([array_like]): Unit right-handed quaternion in [qx, qy, qz, qw] form

    Returns:
        [np_array]: 3x3 Rotation matrix
    """

    # assert(np.linalg.norm(quaternion) == 1) # ensures unit quaternion
    # TODO: Check that quaternion is right-handed

    qx, qy, qz, qw = quaternion

    RMatrix = np.array(
        [
            [
                (qw * qw) + (qx * qx) - (qy * qy) - (qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                (qw * qw) - (qx * qx) + (qy * qy) - (qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                (qw * qw) - (qx * qx) - (qy * qy) + (qz * qz),
            ],
        ]
    )

    return RMatrix


def RMatrix2Euler(RMatrix, units="rad"):
    """Matrix2Euler converts the Rotation Matrix into Euler angles that follow the roll-pitch-yaw convention.
    Pitch is limited to [-90, 90], due to arcsin use. Other angles span [-180, 180].

    Args:
        matrix ([array_like]): Rotation matrix to be converted
        units (str, optional): Euler angles can either be returned in "deg" or "rad".
        Defaults to "rad".

    Returns:
        [np_array]: Euler Angles in roll-pitch-yaw convention
    """

    EulerAngles = np.array(
        [
            [
                np.arctan2(RMatrix[2, 1], RMatrix[2, 2]),
                np.arcsin(-RMatrix[2, 0]),
                np.arctan2(RMatrix[1, 0], RMatrix[0, 0]),
            ]
        ]
    )

    if units == "deg":
        EulerAngles = np.rad2deg(EulerAngles)

    return EulerAngles

def Quat2Yaw(quaternion, units="rad"):
    """ This simple function combines the two above to return yaw
    """
    RMatrix = Quat2RMatrix(quaternion)
    euler = RMatrix2Euler(RMatrix, units)

    yaw = euler[0][2]  # OptiTrack data is taken to be streamed "Z-up"

    return yaw
