from dis import dis
import itertools
import os
from collections import namedtuple
from ctypes import POINTER
from dataclasses import dataclass
import time
import random

import sys

from cv2 import circle
from numpy import inner
from .optitrack import OptiTrack

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from typing import Any, cast, Dict, List, NewType, Optional, Sequence, Tuple, Union

import subprocess
import sys

try:    import rospy
except: pass

import geometry
import geometry as g
import gym
import math
import numpy as np
import pyglet
import yaml
from geometry import SE2value
from gym import spaces
from gym.utils import seeding
from numpy.random.mtrand import RandomState
from pyglet import gl, image, window
import pathlib

from duckietown_world import (
    get_DB18_nominal,
    get_DB18_uncalibrated,
    get_texture_file,
    MapFormat1,
    MapFormat1Constants,
    MapFormat1Constants as MF1C,
    MapFormat1Object,
    SE2Transform,
)
from duckietown_world.gltf.export import get_duckiebot_color_from_colorname
from duckietown_world.resources import get_resource_path
from duckietown_world.world_duckietown.map_loading import get_transform
from . import logger
from .check_hw import get_graphics_information
from .collision import (
    agent_boundbox,
    generate_norm,
    intersects,
    safety_circle_intersection,
    safety_circle_overlap,
    tile_corners,
    radii_overlap
)
from .distortion import Distortion
from .exceptions import InvalidMapException, NotInLane
from .graphics import (
    bezier_closest,
    bezier_draw,
    bezier_point,
    bezier_tangent,
    create_frame_buffers,
    gen_rot_matrix,
    load_texture,
    Texture,
)
from .objects import CheckerboardObj, DuckiebotObj, DuckieObj, TrafficLightObj, WorldObj
from .objmesh import get_mesh, MatInfo, ObjMesh
from .randomization import Randomizer
from .utils import get_subdir_path
from sklearn.neighbors import NearestNeighbors

DIM = 0.5

TileKind = NewType("TileKind", str)


class TileDict(TypedDict):
    # {"coords": (i, j), "kind": kind, "angle": angle, "drivable": drivable})
    coords: Tuple[int, int]
    kind: TileKind
    angle: int
    drivable: bool
    texture: Texture
    color: np.ndarray
    curves: Any


@dataclass
class DoneRewardInfo:
    done: bool
    done_why: str
    done_code: str
    reward: float


@dataclass
class DynamicsInfo:
    motor_left: float
    motor_right: float


# cv-tsl-13 OptiTrack Parameters
server_ip     = "192.168.0.13"
multicast_ip  = "239.255.42.99"
command_port  = 1510
data_port     = 1511


# Rendering window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Camera image size
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480

# Blue sky horizon color
BLUE_SKY = np.array([0.45, 0.82, 1])

# Color meant to approximate interior walls
WALL_COLOR = np.array([0.64, 0.71, 0.28])

# np.array([0.15, 0.15, 0.15])
GREEN = (0.0, 1.0, 0.0)
# Ground/floor color


# Angle at which the camera is pitched downwards
CAMERA_ANGLE = 19.15 #TODO: check on DB21M

# Camera field of view angle in the Y direction
# Note: robot uses Raspberri Pi camera module V1.3
# https://www.raspberrypi.org/documentation/hardware/camera/README.md
CAMERA_FOV_Y = 75 #TODO: check on DB21M

# Distance from camera to floor (10.8cm)
CAMERA_FLOOR_DIST = 0.108 #TODO: check on DB21M

# Forward distance between the camera (at the front)
# and the center of rotation (6.6cm)
CAMERA_FORWARD_DIST = 0.066 #TODO: check on DB21M

# Distance (diameter) between the center of the robot wheels (10.2cm)
WHEEL_DIST = 0.102 #TODO: check on DB21M

# Total robot width at wheel base, used for collision detection
# Note: the actual robot width is 13cm, but we add a litte bit of buffer
#       to faciliate sim-to-real transfer.
ROBOT_WIDTH = 0.13 + 0.02 #TODO: check on DB21M

# Total robot length
# Note: the center of rotation (between the wheels) is not at the
#       geometric center see CAMERA_FORWARD_DIST
ROBOT_LENGTH = 0.18 #TODO: check on DB21M

# Height of the robot, used for scaling
ROBOT_HEIGHT = 0.12 #TODO: check on DB21M

# Safety radius multiplier
SAFETY_RAD_MULT = 1.8

# Robot safety circle radius
AGENT_SAFETY_RAD = (max(ROBOT_LENGTH, ROBOT_WIDTH) / 2) * SAFETY_RAD_MULT

# Minimum distance spawn position needs to be from all objects
MIN_SPAWN_OBJ_DIST = 0.25

# Road tile dimensions (2ft x 2ft, 61cm wide)
# self.road_tile_size = 0.61

# Maximum forward robot speed in meters/second
DEFAULT_ROBOT_SPEED = 1.20
# approx 2 tiles/second

DEFAULT_FRAMERATE = 5

DEFAULT_MAX_STEPS = 1500

DEFAULT_MAP_NAME = "udem1"

DEFAULT_FRAME_SKIP = 1

DEFAULT_ACCEPT_START_ANGLE_DEG = 60

REWARD_INVALID_POSE = 0

REWARD_COLLISION = -1000

MAX_SPAWN_ATTEMPTS = 5000

LanePosition0 = namedtuple("LanePosition", "dist dot_dir angle_deg angle_rad")


class LanePosition(LanePosition0):
    def as_json_dict(self):
        """Serialization-friendly format."""
        return dict(dist=self.dist, dot_dir=self.dot_dir, angle_deg=self.angle_deg, angle_rad=self.angle_rad)


class Simulator(gym.Env):
    """
    Simple road simulator to test RL training.
    Draws a road with turns using OpenGL, and simulates
    basic differential-drive dynamics.
    """

    metadata = {"render.modes": ["human", "rgb_array", "app"], "video.frames_per_second": 30}

    cur_pos: np.ndarray
    cam_offset: np.ndarray
    road_tile_size: float
    grid_width: int
    grid_height: int
    step_count: int
    timestamp: float
    np_random: RandomState
    grid: List[TileDict]

    def __init__(
        self,
        n_agents: int = 1,
        map_name: str = DEFAULT_MAP_NAME,
        max_steps: int = DEFAULT_MAX_STEPS,
        draw_curve: bool = False,
        draw_bbox: bool = False,
        domain_rand: bool = True,
        frame_rate: float = DEFAULT_FRAMERATE,
        frame_skip: bool = DEFAULT_FRAME_SKIP,
        camera_width: int = DEFAULT_CAMERA_WIDTH,
        camera_height: int = DEFAULT_CAMERA_HEIGHT,
        robot_speed: float = DEFAULT_ROBOT_SPEED,
        accept_start_angle_deg=DEFAULT_ACCEPT_START_ANGLE_DEG,
        full_transparency: bool = False,
        user_tile_start=None,
        seed: int = None,
        distortion: bool = False,
        dynamics_rand: bool = False,
        camera_rand: bool = False,
        randomize_maps_on_reset: bool = False,
        num_tris_distractors: int = 12,
        color_ground: Sequence[float] = (0.15, 0.15, 0.15),
        color_sky: Sequence[float] = BLUE_SKY,
        style: str = "photos",
        enable_leds: bool = False,
        mappo: bool = False,
        simulated: bool = True,
        random_start_tile = True,
        robot_max_tilt_ang: float = 20.0 * np.pi / 180.0
    ):
        """

        :param map_name: Readable name of map
        :param max_steps: Run simulation at most max_steps amount. TODO: run indefinitely if max_steps=0
        :param draw_curve:
        :param draw_bbox:
        :param domain_rand: If true, applies domain randomization
        :param frame_rate: 
        :param frame_skip:
        :param camera_width:
        :param camera_height:
        :param robot_speed:
        :param accept_start_angle_deg:
        :param full_transparency:
        :param user_tile_start: If None, sample randomly. Otherwise (i,j). Overrides map start tile
        :param seed:
        :param distortion: If true, distorts the image with fish-eye approximation
        :param dynamics_rand: If true, perturbs the trim of the Duckiebot
        :param camera_rand: If true randomizes over camera miscalibration
        :param randomize_maps_on_reset: If true, randomizes the map on reset (Slows down training)
        :param style: String that represent which tiles will be loaded. One of ["photos", "synthetic"]
        :param enable_leds: Enables LEDs drawing.
        """

        # Flag to enable/disable domain randomization
        self.domain_rand = domain_rand
        self.randomizer = Randomizer()
        self.random_start_tile = random_start_tile

        # True if running only simulated env, False if controlling real-life robots
        self.simulated = simulated

        print(f"SIMULATION: {self.simulated}")

        robot_desc = {
            "kind": MF1C.KIND_DUCKIEBOT,
            "mesh": get_duckiebot_mesh("red"),
            "pos": [0, 0, 0],     # To be set later in the initialization of the environment
            "angle": 0.0,   # defined for now to initialize robot
            "scale": 1.0,   #TODO: Need to check this. To check for what?
            "optional": None,
            "static": False,
        }
        
        if (self.simulated):
            self.n_agents = n_agents
            self.active_robots = [f"td{i:02d}" for i in range(1, self.n_agents+1)]

        else:
            self.active_robots = []

            # Detecting robots connected to Wi-Fi
            for robot_name in [f"td{number:02d}" for number in range(43)]:        
                try:
                    subprocess.check_call(["ping", "-c1",f"{robot_name}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.active_robots.append(robot_name)
                except:
                    pass
            if (self.active_robots == []):
                print("Could not detect any robot.")
                sys.exit(0)

            self.optitrack = OptiTrack(self.active_robots)

            self.n_agents = len(self.active_robots)

            #TODO: replace ROS with NatNet again (small precision increase)!!!
            rospy.init_node("duckietown_gym")
            # self.tlistener = tf.TransformListener()


        self.agents = [DuckiebotObj(
            robot_desc, self.domain_rand,
            SAFETY_RAD_MULT, WHEEL_DIST, ROBOT_WIDTH, ROBOT_LENGTH,
            simulated=self.simulated, name=robot_name, enable_leds=enable_leds)
                        for robot_name in self.active_robots]

        self.robot_max_tilt_ang = robot_max_tilt_ang

        self.mappo = mappo
        #self.speed = {}
        #self.last_action = {}
        #self.wheelVels = {}
        # self.enable_leds = enable_leds
        information = get_graphics_information()
        logger.info(
            f"Information about the graphics card:",
            pyglet_version=pyglet.version,
            information=information,
            nvidia_around=os.path.exists("/proc/driver/nvidia/version"),
        )

        # first initialize the RNG
        self.seed_value = seed
        self.seed(seed=self.seed_value)
        self.num_tris_distractors = num_tris_distractors
        self.color_ground = color_ground
        self.color_sky = list(color_sky)
        # If true, then we publish all transparency information
        self.full_transparency = full_transparency
        self.dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Map name, set in _load_map()
        self.map_name = None

        # Full map file path, set in _load_map()
        self.map_file_path = None

        # The parsed content of the map_file
        self.map_data = None

        # Maximum number of steps per episode
        self.max_steps = max_steps

        # Flag to draw the road curve
        self.draw_curve = draw_curve

        # Flag to draw bounding boxes
        self.draw_bbox = draw_bbox

        # Frame rate to run at
        self.frame_rate = frame_rate
        self.delta_time = 1.0 / self.frame_rate

        # Number of frames to skip per action
        self.frame_skip = frame_skip

        # Produce graphical output
        self.graphics = True

        # Two-tuple of wheel torques, each in the range [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.camera_width = camera_width
        self.camera_height = camera_height

        self.robot_speed = robot_speed
        # We observe an RGB image with pixels in [0, 255]
        # Note: the pixels are in uint8 format because this is more compact
        # than float32 if sent over the network or stored in a dataset
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8
        )

        self.reward_range = (-1000, 1000)

        # Window for displaying the environment to humans
        self.window = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        # For displaying text
        self.text_label = pyglet.text.Label(font_name="Arial", font_size=14, x=5, y=WINDOW_HEIGHT - 19)

        # Create a frame buffer object for the observation
        self.multi_fbo, self.final_fbo = create_frame_buffers(self.camera_width, self.camera_height, 4)

        # Array to render the image into (for observation rendering)
        self.img_array = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)

        # Create a frame buffer object for human rendering
        self.multi_fbo_human, self.final_fbo_human = create_frame_buffers(WINDOW_WIDTH, WINDOW_HEIGHT, 4)

        # Array to render the image into (for human rendering)
        self.img_array_human = np.zeros(shape=(WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

        # allowed angle in lane for starting position
        self.accept_start_angle_deg = accept_start_angle_deg

        # Load the map
        self._load_map(map_name)

        self._get_waypoints()

        # Distortion params, if so, load the library, only if not bbox mode
        self.distortion = distortion and not draw_bbox
        self.camera_rand = False
        if not draw_bbox and distortion:
            if distortion:
                self.camera_rand = camera_rand

                self.camera_model = Distortion(camera_rand=self.camera_rand)

        # Used by the UndistortWrapper, always initialized to False
        self.undistort = False

        # Dynamics randomization
        self.dynamics_rand = dynamics_rand

        # Start tile
        self.user_tile_start = user_tile_start

        self.style = style

        self.randomize_maps_on_reset = randomize_maps_on_reset

        if self.randomize_maps_on_reset:
            self.map_names = os.listdir(get_subdir_path("maps"))
            self.map_names = [
                _map for _map in self.map_names if not _map.startswith(("calibration", "regress"))
            ]
            self.map_names = [mapfile.replace(".yaml", "") for mapfile in self.map_names]

        # Initialize the state
        self.dot_dir = 0
        self.lane_dist = 0
        self.reset()

        #self.last_action = np.array([0, 0])
        # self.wheelVels = np.array([0, 0])
        self.fully_cooperative = True
        self.degree = 0
        self.mean_pooling = False
        self.comm_radius2 = 0.1


    def find_closest_waypoint(self, agent, lane):
        
        closest_idx = np.argmin(np.linalg.norm(self.discretized_path[lane] - agent.pos[[0, 2]], axis=1))
        

        #print(f">> distance {distance}")
        return closest_idx
            
    
    def get_circleCenter(self, start_point, end_point, radius, turn_direction):

        if (turn_direction == "r"):
            start_p = start_point
            end_p = end_point
        elif (turn_direction == "l"):
            start_p = end_point
            end_p = start_point


        q = np.sqrt((end_p[0]-start_p[0])**2.0 + (end_p[1]-start_p[1])**2.0)
        x_mean = (start_p[0] + end_p[0]) / 2.0
        y_mean = (start_p[1] + end_p[1]) / 2.0
        
        xc = x_mean + np.sqrt(radius**2.0-(q/2.0)**2.0)*(start_p[1]-end_p[1])/q
        yc = y_mean + np.sqrt(radius**2.0-(q/2.0)**2.0)*(end_p[0]-start_p[0])/q  
        


        return [xc, yc]

    def parametric_circle(self, t, circle_center, R):
        print(f"t = {t}")
        x = circle_center[0] + R*np.cos(t)
        y = circle_center[1] + R*np.sin(t)
        return np.asarray([x,y]).T

    def inv_parametric_circle(self, x, xc, R, y_sign_correction: bool):
        
        inside_term = (x-xc)/R
        #print(f"(x-xc)/R={inside_term}")

        t = np.arccos(inside_term)
        
        if y_sign_correction:
            return -t
        
        return t


    def _get_waypoints(self):

        tile_size = self.map_data["tile_size"]          # To save typing
       
        # Waypoints used to generated discretized path for a given map
        keyWaypoints = []

        inner_curve_radius = 0.25
        outer_curve_radius = 0.75

        # Key-Waypoint: [pos_x, pos_y, direction, circle_radius]
        # Positions are given in tile units
        # Direction is either: "l" - left (CCW); "r" - right (CW); "s" - straight

        # Outer loops
        keyWaypoints.append([0.25, 1.0, "r", outer_curve_radius])
        keyWaypoints.append([1.0, 0.25, "s", outer_curve_radius])
        keyWaypoints.append([4.0, 0.25, "r", outer_curve_radius])
        keyWaypoints.append([4.75, 1.0, "s", outer_curve_radius])
        keyWaypoints.append([4.75, 5.0, "r", outer_curve_radius])
        keyWaypoints.append([4.0, 5.75, "s", outer_curve_radius])
        keyWaypoints.append([2.0, 5.75, "r", outer_curve_radius])
        keyWaypoints.append([1.25, 5.0, "l", inner_curve_radius])
        keyWaypoints.append([1.0, 4.75, "r", outer_curve_radius])
        keyWaypoints.append([0.25, 4.0, "r", outer_curve_radius])

        keyWaypoints.append([0.25, 1.0, "s", inner_curve_radius])   # Duplicating 1st point (due to for loop below)
        
        # self.keyWaypoints = np.array(keyWaypoints)    # List to np.array
        
        points_per_tile = 5                            # Constant number of points per tile


        #discretized_path = []

        inner_path = []
        outer_path = []
        for i in range(len(keyWaypoints)-1):

            # For readability
            curr_keyWaypoint = keyWaypoints[i]
            next_keyWaypoint = keyWaypoints[i+1]
            curr_radius = keyWaypoints[i][3]
            curr_turn_dir = keyWaypoints[i][2]

            # Straight vertical line (x=const)
            if (curr_keyWaypoint[0] == next_keyWaypoint[0]):

                distance = int(np.sqrt((next_keyWaypoint[1] - curr_keyWaypoint[1])**2 + (next_keyWaypoint[0] - curr_keyWaypoint[0])**2))  # In tile units
                nr_of_points = distance*points_per_tile
                y_var = np.linspace(curr_keyWaypoint[1],           # Start point
                                    next_keyWaypoint[1],           # End point
                                    nr_of_points + 1
                                    )[0:-1]                         # Makes it so that next_keyWaypoint is not included
                x_var = np.ones(nr_of_points) * curr_keyWaypoint[0]

                for j in range(nr_of_points):
                    outer_path.append([x_var[j], y_var[j]])


            # Straight horizontal line (y=const)
            elif (curr_keyWaypoint[1] == next_keyWaypoint[1]):
                
                distance = int(np.sqrt((next_keyWaypoint[1] - curr_keyWaypoint[1])**2 + (next_keyWaypoint[0] - curr_keyWaypoint[0])**2))  # In tile units
                nr_of_points = distance*points_per_tile
                x_var = np.linspace(curr_keyWaypoint[0],           # Start point
                                    next_keyWaypoint[0],           # End point
                                    nr_of_points + 1
                                    )[0:-1]                         # Makes it so that next_keyWaypoint is not included
                y_var = np.ones(nr_of_points) * curr_keyWaypoint[1]
        
                for j in range(nr_of_points):
                    outer_path.append([x_var[j], y_var[j]])

            

            # Left or Right Turn
            else:

                circle_center = self.get_circleCenter(curr_keyWaypoint[0:2], next_keyWaypoint[0:2], curr_radius, curr_turn_dir)

                print(circle_center)

                if circle_center[1] < ((curr_keyWaypoint[1] + next_keyWaypoint[1])/2.0):
                    y_correction = False
                else:
                    y_correction = True

                start_theta = self.inv_parametric_circle(curr_keyWaypoint[0], circle_center[0], curr_radius, y_correction)
                end_theta   = self.inv_parametric_circle(next_keyWaypoint[0], circle_center[0], curr_radius, y_correction)

                # print(start_theta)
                # print(end_theta)

                arc_length = np.abs( (end_theta - start_theta) *  curr_radius )
                # print(arc_length)
                #print(arc_length, self.points_per_m)
                segment_points = points_per_tile #int(arc_length * points_per_tile)

                print(f"segment = {segment_points}")

                #print(f"Arc length = {arc_length}, nr_segment_points={segment_points}")

                arc_Theta = np.linspace(start_theta, end_theta, segment_points)
                print(f"arc_theta = {arc_Theta.shape}")


                segment = self.parametric_circle(arc_Theta, circle_center, curr_radius)

                for j in range(len(segment)):
                    outer_path.append([segment[j][0], segment[j][1]])

        keyWaypoints = []

        # Inner loop
        keyWaypoints.append([0.75, 1.0, "r", inner_curve_radius])
        keyWaypoints.append([1.0, 0.75, "s", inner_curve_radius])
        keyWaypoints.append([4.0, 0.75, "r", inner_curve_radius])
        keyWaypoints.append([4.25, 1.0, "s", inner_curve_radius])
        keyWaypoints.append([4.25, 5.0, "r", inner_curve_radius])
        keyWaypoints.append([4.0, 5.25, "s", inner_curve_radius])
        keyWaypoints.append([2.0, 5.25, "r", inner_curve_radius])
        keyWaypoints.append([1.75, 5.0, "l", outer_curve_radius])
        keyWaypoints.append([1.0, 4.25, "r", inner_curve_radius])
        keyWaypoints.append([0.75, 4.0, "r", inner_curve_radius])

        keyWaypoints.append([0.75, 1.0, "s", inner_curve_radius])   # Duplicating 1st point (due to for loop below)

        for i in range(len(keyWaypoints)-1):

            # For readability
            curr_keyWaypoint = keyWaypoints[i]
            next_keyWaypoint = keyWaypoints[i+1]
            curr_radius = keyWaypoints[i][3]
            curr_turn_dir = keyWaypoints[i][2]

            # Straight vertical line (x=const)
            if (curr_keyWaypoint[0] == next_keyWaypoint[0]):

                distance = int(np.sqrt((next_keyWaypoint[1] - curr_keyWaypoint[1])**2 + (next_keyWaypoint[0] - curr_keyWaypoint[0])**2))  # In tile units
                nr_of_points = distance*points_per_tile
                y_var = np.linspace(curr_keyWaypoint[1],           # Start point
                                    next_keyWaypoint[1],           # End point
                                    nr_of_points + 1
                                    )[0:-1]                         # Makes it so that next_keyWaypoint is not included
                x_var = np.ones(nr_of_points) * curr_keyWaypoint[0]

                for j in range(nr_of_points):
                    inner_path.append([x_var[j], y_var[j]])


            # Straight horizontal line (y=const)
            elif (curr_keyWaypoint[1] == next_keyWaypoint[1]):
                
                distance = int(np.sqrt((next_keyWaypoint[1] - curr_keyWaypoint[1])**2 + (next_keyWaypoint[0] - curr_keyWaypoint[0])**2))  # In tile units
                nr_of_points = distance*points_per_tile
                x_var = np.linspace(curr_keyWaypoint[0],           # Start point
                                    next_keyWaypoint[0],           # End point
                                    nr_of_points + 1
                                    )[0:-1]                         # Makes it so that next_keyWaypoint is not included
                y_var = np.ones(nr_of_points) * curr_keyWaypoint[1]
        
                for j in range(nr_of_points):
                    inner_path.append([x_var[j], y_var[j]])

            

            # Left or Right Turn
            else:

                circle_center = self.get_circleCenter(curr_keyWaypoint[0:2], next_keyWaypoint[0:2], curr_radius, curr_turn_dir)

                print(circle_center)

                if circle_center[1] < ((curr_keyWaypoint[1] + next_keyWaypoint[1])/2.0):
                    y_correction = False
                else:
                    y_correction = True

                start_theta = self.inv_parametric_circle(curr_keyWaypoint[0], circle_center[0], curr_radius, y_correction)
                end_theta   = self.inv_parametric_circle(next_keyWaypoint[0], circle_center[0], curr_radius, y_correction)

                # print(start_theta)
                # print(end_theta)

                arc_length = np.abs( (end_theta - start_theta) *  curr_radius )
                # print(arc_length)
                #print(arc_length, self.points_per_m)
                segment_points = points_per_tile#int(arc_length * points_per_tile)

                #print(f"Arc length = {arc_length}, nr_segment_points={segment_points}")

                arc_Theta = np.linspace(start_theta, end_theta, segment_points)

                segment = self.parametric_circle(arc_Theta, circle_center, curr_radius)

                for j in range(len(segment)):
                    inner_path.append([segment[j][0], segment[j][1]])


        #self.discretized_path = np.array(discretized_path) * tile_size
        inner_path = np.array(inner_path) * tile_size
        outer_path = np.array(outer_path) * tile_size

        print(inner_path.shape)
        print(outer_path.shape)

        # Inner = 0 ; Outer  = 1;
        self.discretized_path = [inner_path, outer_path]

        #print(self.discretized_path.shape)

        # Top straight

        #print(self.keyWaypoints)



        #for tile in self.map_data['tiles']:
            

    def _init_vlists(self):

        ns = 8
        assert ns >= 2

        # half_size = self.road_tile_size / 2
        TS = self.road_tile_size

        def get_point(u, v):
            pu = u / (ns - 1)
            pv = v / (ns - 1)
            x = -TS / 2 + pu * TS
            z = -TS / 2 + pv * TS
            tu = pu
            tv = 1 - pv
            return (x, 0.0, z), (tu, tv)

        vertices = []
        textures = []
        normals = []
        colors = []
        for i, j in itertools.product(range(ns - 1), range(ns - 1)):
            tl_p, tl_t = get_point(i, j)
            tr_p, tr_t = get_point(i + 1, j)
            br_p, br_t = get_point(i, j + 1)
            bl_p, bl_t = get_point(i + 1, j + 1)
            normal = [0.0, 1.0, 0.0]

            color = (255, 255, 255, 255)
            vertices.extend(tl_p)
            textures.extend(tl_t)
            normals.extend(normal)
            colors.extend(color)

            vertices.extend(tr_p)
            textures.extend(tr_t)
            normals.extend(normal)
            colors.extend(color)

            vertices.extend(bl_p)
            textures.extend(bl_t)
            normals.extend(normal)
            colors.extend(color)

            vertices.extend(br_p)
            textures.extend(br_t)
            normals.extend(normal)
            colors.extend(color)

            #
            # normals.extend([0.0, 1.0, 0.0] * 4)

        # def get_quad_vertices(cx, cz, hs) -> Tuple[List[float], List[float], List[float]]:
        #     v = [
        #         -hs + cx,
        #         0.0,
        #         -hs + cz,
        #         #
        #         hs + cx,
        #         0.0,
        #         -hs + cz,
        #         #
        #         hs + cx,
        #         0.0,
        #         hs + cz,
        #         #
        #         -hs + cx,
        #         0.0,
        #         hs + cz,
        #     ]
        #     n = [0.0, 1.0, 0.0] * 4
        #     t = [0.0, 1.0,
        #          #        #          1.0, 1.0,
        #          #
        #          1.0, 0.0,
        #          #
        #          0.0, 0.0]
        #     return v, n, t

        # Create the vertex list for our road quad
        # Note: the vertices are centered around the origin so we can easily
        # rotate the tiles about their center

        # verts = []
        # texCoords = []
        # normals = []
        #
        # v, n, t = get_quad_vertices(cx=0, cz=0, hs=half_size)
        # verts.extend(v)
        # normals.extend(n)
        # texCoords.extend(t)

        # verts = [
        #     -half_size,
        #     0.0,
        #     -half_size,
        #     #
        #     half_size,
        #     0.0,
        #     -half_size,
        #     #
        #     half_size,
        #     0.0,
        #     half_size,
        #     #
        #     -half_size,
        #     0.0,
        #     half_size,
        # ]
        # texCoords = [1.0, 0.0,
        #              0.0, 0.0,
        #              0.0, 1.0,
        #              1.0, 1.0]
        # Previous choice would reflect the texture
        # logger.info(nv=len(vertices), nt=len(textures), nn=len(normals), vertices=vertices,
        # textures=textures,
        #             normals=normals)
        total = len(vertices) // 3
        self.road_vlist = pyglet.graphics.vertex_list(
            total, ("v3f", vertices), ("t2f", textures), ("n3f", normals), ("c4B", colors)
        )
        logger.info("done")
        # Create the vertex list for the ground quad
        verts = [
            -1,
            -0.8,
            1,
            #
            -1,
            -0.8,
            -1,
            #
            1,
            -0.8,
            -1,  #
            1,
            -0.8,
            1,
        ]
        self.ground_vlist = pyglet.graphics.vertex_list(4, ("v3f", verts))

    def reset(self, segment: bool = False):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        self.step_count = 0
        self.timestamp = 0.0

        # Robot's current speed
        #self.speed = {}
        #self.wheelVels = {}
        #self.last_action = {}
        #self.speed = {i:0 for i in range(self.n_agents)}
        #self.wheelVels = {i: [0, 0] for i in range(self.n_agents)}
        #self.last_action = {i: [0, 0] for i in range(self.n_agents)}
        for agent in self.agents:
            agent.speed = 0.0
            agent.last_action = [0.0, 0.0]
            self.wheelVels = [0.0, 0.0]

        if self.randomize_maps_on_reset:
            map_name = self.np_random.choice(self.map_names)
            logger.info(f"Random map chosen: {map_name}")
            self._load_map(map_name)

        self.randomization_settings = self.randomizer.randomize(rng=self.np_random)

        # Horizon color
        # Note: we explicitly sample white and grey/black because
        # these colors are easily confused for road and lane markings
        if self.domain_rand:
            horz_mode = self.randomization_settings["horz_mode"]
            if horz_mode == 0:
                self.horizon_color = self._perturb(self.color_sky)
            elif horz_mode == 1:
                self.horizon_color = self._perturb(WALL_COLOR)
            elif horz_mode == 2:
                self.horizon_color = self._perturb([0.15, 0.15, 0.15], 0.4)
            elif horz_mode == 3:
                self.horizon_color = self._perturb([0.9, 0.9, 0.9], 0.4)
        else:
            self.horizon_color = self.color_sky

        # Setup some basic lighting with a far away sun
        if self.domain_rand:
            light_pos = self.randomization_settings["light_pos"]
        else:
            # light_pos = [-40, 200, 100, 0.0]

            light_pos = [0.0, 3.0, 0.0, 1.0]

        # DIM = 0.0
        ambient = np.array([0.50 * DIM, 0.50 * DIM, 0.50 * DIM, 1])
        ambient = self._perturb(ambient, 0.3)
        diffuse = np.array([0.70 * DIM, 0.70 * DIM, 0.70 * DIM, 1])
        diffuse = self._perturb(diffuse, 0.99)
        # specular = np.array([0.3, 0.3, 0.3, 1])
        specular = np.array([0.0, 0.0, 0.0, 1])

        # logger.info(light_pos=light_pos, ambient=ambient, diffuse=diffuse, specular=specular)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*light_pos))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat * 4)(*diffuse))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, (gl.GLfloat * 4)(*specular))

        # gl.glLightfv(gl.GL_LIGHT0, gl.GL_CONSTANT_ATTENUATION, (gl.GLfloat * 1)(0.4))
        # gl.glLightfv(gl.GL_LIGHT0, gl.GL_LINEAR_ATTENUATION, (gl.GLfloat * 1)(0.3))
        # gl.glLightfv(gl.GL_LIGHT0, gl.GL_QUADRATIC_ATTENUATION, (gl.GLfloat * 1)(0.1))

        #gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_COLOR_MATERIAL)

        #TODO: should we only randomize in sim environment?
        # Ground color
        self.ground_color = self._perturb(np.array(self.color_ground), 0.3)

        # Distance between the robot's wheels
        self.wheel_dist = self._perturb(WHEEL_DIST)

        # Set default values

        # Distance bewteen camera and ground
        self.cam_height = CAMERA_FLOOR_DIST

        # Angle at which the camera is rotated
        self.cam_angle = [CAMERA_ANGLE, 0, 0]

        # Field of view angle of the camera
        self.cam_fov_y = CAMERA_FOV_Y

        # Perturb using randomization API (either if domain rand or only camera rand
        if self.domain_rand or self.camera_rand:
            self.cam_height *= self.randomization_settings["camera_height"]
            self.cam_angle = [CAMERA_ANGLE * self.randomization_settings["camera_angle"], 0, 0]
            self.cam_fov_y *= self.randomization_settings["camera_fov_y"]

        # Camera offset for use in free camera mode
        self.cam_offset = np.array([0, 0, 0])

        # Create the vertex list for the ground/noise triangles
        # These are distractors, junk on the floor
        numTris = self.num_tris_distractors
        verts = []
        colors = []
        for _ in range(0, 3 * numTris):
            p = self.np_random.uniform(low=[-20, -0.6, -20], high=[20, -0.3, 20], size=(3,))
            c = self.np_random.uniform(low=0, high=0.9)
            c = self._perturb([c, c, c], 0.1)
            verts += [p[0], p[1], p[2]]
            colors += [c[0], c[1], c[2]]

        self.tri_vlist = pyglet.graphics.vertex_list(3 * numTris, ("v3f", verts), ("c3f", colors))

        # Randomize tile parameters
        for tile in self.grid:
            rng = self.np_random if self.domain_rand else None

            kind = tile["kind"]
            fn = get_texture_file(os.path.join("tiles-processed", f"{self.style}", f"{kind}", "texture"))[0]
            # ft = get_fancy_textures(self.style, texture_name)
            t = load_texture(fn, segment=False, segment_into_color=False)
            tt = Texture(t, tex_name=kind, rng=rng)
            tile["texture"] = tt

            # Random tile color multiplier
            tile["color"] = self._perturb([1, 1, 1, 1], 0.2)

        # Randomize object parameters
        for obj in self.objects:
            # Randomize the object color
            obj.color = self._perturb([1, 1, 1, 1], 0.3)

            # Randomize whether the object is visible or not
            if obj.optional and self.domain_rand:
                obj.visible = self.np_random.randint(0, 2) == 0
            else:
                obj.visible = True

        # If the map specifies a starting tile
        if self.user_tile_start:
            logger.info(f"using user tile start: {self.user_tile_start}")
            i, j = self.user_tile_start
            tile = self._get_tile(i, j)
            if tile is None:
                msg = "The tile specified does not exist."
                raise Exception(msg)
            logger.debug(f"tile: {tile}")
        else:
            if self.start_tile is not None:
                tile = self.start_tile
            else:
                # Select a random drivable tile to start on
                start_tiles = {}
                for agent in self.agents:
                    if not self.drivable_tiles:
                        msg = "There are no drivable tiles. Use start_tile or self.user_tile_start"
                        raise Exception(msg)
                    tile_idx = self.np_random.randint(0, len(self.drivable_tiles))
                    tile = self.drivable_tiles[tile_idx]
                    start_tiles[agent] = self.drivable_tiles[tile_idx]

        # If the map specifies a starting pose
        # self.cur_pos = {}
        # self.cur_angle = {}

        starting_positions = []
        starting_angles = []
        for agent in self.agents:
            if self.start_pose is not None:
                # logger.info(f"using map pose start: {self.start_pose}")

                i, j = start_tiles[agent]["coords"]
                x = i * self.road_tile_size + self.start_pose[0]
                z = j * self.road_tile_size + self.start_pose[1]

                # propose_pos = np.array([x, 0, z])
                # propose_angle = self.start_pose[1]

                starting_positions.append(np.array([x, 0, z]))
                starting_angles.append(self.start_pose[1])
                #self.cur_pos[v] = propose_pos
                #self.cur_angle[v] = propose_angle
            else:
                # Keep trying to find a valid spawn position on this tile
                for _ in range(MAX_SPAWN_ATTEMPTS):
                    if self.random_start_tile:
                        i, j = start_tiles[agent]["coords"]
                    else:
                        i, j = [0, 2]
                    # Choose a random position on this tile
                    x = self.np_random.uniform(i, i + 1) * self.road_tile_size
                    # always start in same tile and correct direction and lane
                    # x = self.np_random.uniform(i + 0.2, i + 0.45) * self.road_tile_size
                    # z = self.np_random.uniform(j, j + 1) * self.road_tile_size
                    z = self.np_random.uniform(j - 1, j + 2) * self.road_tile_size

                    propose_pos = np.array([x, 0, z])

                    # Choose a random direction
                    if self.random_start_tile:
                        propose_angle = self.np_random.uniform(0, 2 * math.pi)
                        # propose_angle = self.np_random.uniform(math.pi + 0.6,  2 * math.pi - 0.6)
                    else:
                        # propose_angle = 3*math.pi/2
                        propose_angle = self.np_random.uniform(0.90*3*math.pi/2, 1.1*3*math.pi/2)
                        # Rotate randomly
                        if random.random() <= 0.5:
                            propose_angle += math.pi

                    # logger.debug('Sampled %s %s angle %s' % (propose_pos[0],
                    #                                          propose_pos[1],
                    #                                          np.rad2deg(propose_angle)))

                    # If this is too close to an object or not a valid pose, retry
                    inconvenient = self._inconvenient_spawn(propose_pos)

                    if inconvenient:
                        # msg = 'The spawn was inconvenient.'
                        # logger.warning(msg)
                        continue

                    invalid = not self._valid_pose(propose_pos, propose_angle, safety_factor=1.3)
                    if invalid:
                        # msg = 'The spawn was invalid.'
                        # logger.warning(msg)
                        continue

                    # If the angle is too far away from the driving direction, retry
                    try:
                        lp = self.get_lane_pos2(propose_pos, propose_angle)
                    except NotInLane:
                        continue
                    M = self.accept_start_angle_deg
                    ok = -M < lp.angle_deg < +M
                    if not ok:
                        continue
                    starting_positions.append(propose_pos)
                    starting_angles.append(propose_angle)
                    # Found a valid initial pose
                    #self.cur_pos[v] = propose_pos
                    #self.cur_angle[v] = propose_angle
                    break
                else:
                    msg = f"Could not find a valid starting pose after {MAX_SPAWN_ATTEMPTS} attempts"
                    # logger.warn(msg)
                    # propose_pos = np.array([1, 0, 1])
                    # propose_angle = 1
                    starting_positions.append(np.array([1, 0, 1]))
                    starting_angles.append(1)


                    #self.cur_pos[v] = propose_pos
                    #self.cur_angle[v] = propose_angle

        reset_physical = np.any([agent.invalid_real_pose for agent in self.agents])

        # Simulated or Initial Reset of Real Env
        if not reset_physical:
            for v, agent in enumerate(self.agents):
                agent.pos = starting_positions[v]
                agent.angle = starting_angles[v]
        
        else:
            # Stop all agents
            for agent in self.agents:
                agent.send_wheelVel([0.0, 0.0])


        
        for agent in self.agents:
            closest_lane0 = self.discretized_path[0][self.find_closest_waypoint(agent, 0)]
            closest_lane1 = self.discretized_path[1][self.find_closest_waypoint(agent, 1)]

            if (closest_lane0 < closest_lane1):
                agent.lane = 0
            else:
                agent.lane = 1

            # Currently stopping code
            # should do control theory to ge to start_pos and start_angle from current pos
        #self.real_move2pose(starting_positions, starting_angles)
            


        # self.cur_pos = propose_pos
        # self.cur_angle = propose_angle

        init_vel = np.array([0, 0])

        # Initialize Dynamics model
        for agent in self.agents:
            if self.dynamics_rand:
                trim = 0 + self.randomization_settings["trim"][0]
                p = get_DB18_uncalibrated(delay=0.25, trim=trim)
            else:
                p = get_DB18_nominal(delay=0.25)

            q = self.cartesian_from_weird(agent.pos, agent.angle)
            v0 = geometry.se2_from_linear_angular(init_vel, 0)
            c0 = q, v0
            agent.state = p.initialize(c0=c0, t0=0)

        #Goal points
        # self.agent_xg = []
        # self.agent_zg = []
        # self.placer_x = 0
        # pos = []
        #for i in range(self.n_agents):
        for agent in self.agents:
            #self.agent_xg.append(self.placer_x)
            #self.agent_zg.append(1.5 * self.road_tile_size)
            agent.goal_2D = np.array([0, 1.5 * self.road_tile_size])
            #self.agent_yg.append(y1[i]+70)
            # self.placer_x += 0.5 * self.road_tile_size
            #self.placer_x = 0.5 * self.road_tile_size
            # pos.append(self.cur_pos[v][[0, 2]])
            # agent.pos_2d = [agent.pos[0], agent.pos[2]]
            # print(f"2D: {agent.pos_2d}")          

        # self.goal_xpoints = np.array(self.agent_xg)
        # self.goal_zpoints = np.array(self.agent_zg)

        # agent.pos_2D = np.array(pos)

        # Return first observation
        return self.get_obs()

    # def real_move2pose(self, goal_positions, goal_angles):

    #     for v, agent in enumerate(self.agents):

    #         desired_pos = goal_positions[v]
    #         desired_angle = goal_angles[v]

    #         while True:

    #             dX = desired_pos[0]-agent.pos[0]
    #             dY = desired_pos[1]-agent.pos[1]
    #             rho = np.sqrt(dX**2 + dY**2)

    #             print(f"des_pos={desired_pos}, curr_pos={agent.pos} -> dist={rho}")
                
    #             alpha = np.arctan2(dY, dX) - agent.angle
    #             beta = - agent.angle - alpha

    #             print(f"alpha={alpha}, beta={beta}: robot_angle={agent.angle}")


    def get_obs(self):
        """State: [angle_wrt_lane, lane_distance, not_in_lane(bool), speed]"""
        #todo perception function
        _obs = []

        for agent in self.agents:
            flat_list = []
            pos = agent.pos.tolist()
            angle = agent.angle
            #flat_list.append(angle)
            #flat_list.extend(pos)
            #tile = self._get_tile(pos[0] / self.road_tile_size, pos[2] / self.road_tile_size)
            #flat_list.append(tile['tile_n'] if tile else -1)
            try:
                lp = self.get_lane_pos2(pos, angle)
                self.dot_dir = lp.dot_dir
                self.lane_dist = lp.dist
                # if self.lane_dist < 0:
                #     self.dot_dir *= -1
                flat_list.append(self.dot_dir)
                flat_list.append(self.lane_dist)
                flat_list.append(0)
            except NotInLane:
                flat_list.append(self.dot_dir)
                flat_list.append(self.lane_dist)
                flat_list.append(1)
            # flat_list.extend(agent.pos.tolist())
            # flat_list.append(agent.angle)
            flat_list.append(agent.speed)
            _obs.append(flat_list)
            # print(f"Dot dir: {self.dot_dir}")
            # print(f"Lane dist: {self.lane_dist}")
            # print(f"Dot dir: {self.dot_dir}")
            # print(f"Lane dist: {self.lane_dist}")

        if self.mappo:
            _obs = np.array(_obs).reshape(1, self.n_agents, -1)
        return _obs

    def _load_map(self, map_name: str):
        """
        Load the map layout from a YAML file
        """

        # Store the map name
        if os.path.exists(map_name) and os.path.isfile(map_name):
            # if env is loaded using gym's register function, we need to extract the map name from the complete url
            map_name = os.path.basename(map_name)
            assert map_name.endswith(".yaml")
            map_name = ".".join(map_name.split(".")[:-1])

        self.map_name = map_name
        self.map_file_path = self.dir_name + '/maps/' + map_name + '.yaml'

        # Get the full map file path
        # self.map_file_path = get_resource_path(f"{map_name}.yaml")

        logger.debug(f'loading map file "{self.map_file_path}"')

        with open(self.map_file_path, "r") as f:
            self.map_data = yaml.load(f, Loader=yaml.Loader)

        self._interpret_map(self.map_data)

    def _interpret_map(self, map_data: MapFormat1):
        try:
            if not "tile_size" in map_data:
                msg = "Must now include explicit tile_size in the map data."
                raise InvalidMapException(msg)
            self.road_tile_size = map_data["tile_size"]
            self._init_vlists()

            tiles = map_data["tiles"]
            assert len(tiles) > 0
            assert len(tiles[0]) > 0

            # Create the grid
            self.grid_height = len(tiles)
            self.grid_width = len(tiles[0])
            # noinspection PyTypeChecker
            self.grid = [None] * self.grid_width * self.grid_height

            # We keep a separate list of drivable tiles
            self.drivable_tiles = []

            # For each row in the grid
            for j, row in enumerate(tiles):

                if len(row) != self.grid_width:
                    msg = "each row of tiles must have the same length"
                    raise InvalidMapException(msg, row=row)

                # For each tile in this row
                for i, tile in enumerate(row):
                    tile = tile.strip()

                    if tile == "empty":
                        continue

                    directions = ["S", "E", "N", "W"]
                    default_orient = "E"

                    if "/" in tile:
                        kind, orient = tile.split("/")
                        kind = kind.strip(" ")
                        orient = orient.strip(" ")
                        angle = directions.index(orient)

                    elif "4" in tile:
                        kind = "4way"
                        angle = directions.index(default_orient)

                    else:
                        kind = tile
                        angle = directions.index(default_orient)

                    DRIVABLE_TILES = [
                        "straight",
                        "curve_left",
                        "curve_right",
                        "3way_left",
                        "3way_right",
                        "4way",
                    ]
                    drivable = kind in DRIVABLE_TILES

                    # logger.info(f'kind {kind} drivable {drivable} row = {row}')

                    tile = cast(
                        TileDict, {"coords": (i, j), "kind": kind, "angle": angle, "drivable": drivable}
                    )

                    self._set_tile(i, j, tile)

                    if drivable:
                        tile["curves"] = self._get_curve(i, j)
                        self.drivable_tiles.append(tile)

            default_color = "blue"

            self.mesh = get_duckiebot_mesh(default_color)
            self._load_objects(map_data)

            # Get the starting tile from the map, if specified
            self.start_tile = None
            if "start_tile" in map_data:
                coords = map_data["start_tile"]
                self.start_tile = self._get_tile(*coords)

            # Get the starting pose from the map, if specified
            self.start_pose = None
            if "start_pose" in map_data:
                self.start_pose = map_data["start_pose"]
        except Exception as e:
            msg = "Cannot load map data"
            raise InvalidMapException(msg, map_data=map_data)

    def _load_objects(self, map_data: MapFormat1):
        # Create the objects array
        self.objects = []

        # The corners for every object, regardless if collidable or not
        self.object_corners = []

        # Arrays for checking collisions with N static objects
        # (Dynamic objects done separately)
        # (N x 2): Object position used in calculating reward
        self.collidable_centers = []

        # (N x 2 x 4): 4 corners - (x, z) - for object's boundbox
        self.collidable_corners = []

        # (N x 2 x 2): two 2D norms for each object (1 per axis of boundbox)
        self.collidable_norms = []

        # (N): Safety radius for object used in calculating reward
        self.collidable_safety_radii = []

        # For each object
        try:
            objects = map_data["objects"]
        except KeyError:
            pass
        else:
            if isinstance(objects, list):
                for obj_idx, desc in enumerate(objects):
                    print(f"KIND = {desc['kind']}")
                    kind = desc["kind"]
                    obj_name = f"ob{obj_idx:02d}-{kind}"
                    self.interpret_object(obj_name, desc)
            elif isinstance(objects, dict):
                for obj_name, desc in objects.items():
                    self.interpret_object(obj_name, desc)
            else:
                raise ValueError(objects)

        # If there are collidable objects
        if len(self.collidable_corners) > 0:
            self.collidable_corners = np.stack(self.collidable_corners, axis=0)
            self.collidable_norms = np.stack(self.collidable_norms, axis=0)

            # Stack doesn't do anything if there's only one object,
            # So we add an extra dimension to avoid shape errors later
            if len(self.collidable_corners.shape) == 2:
                self.collidable_corners = self.collidable_corners[np.newaxis]
                self.collidable_norms = self.collidable_norms[np.newaxis]

        self.collidable_centers = np.array(self.collidable_centers)
        self.collidable_safety_radii = np.array(self.collidable_safety_radii)

    def interpret_object(self, objname: str, desc: MapFormat1Object):
        kind = desc["kind"]

        W = self.grid_width
        tile_size = self.road_tile_size
        transform: SE2Transform = get_transform(desc, W, tile_size)
        # logger.info(desc=desc, transform=transform)

        pose = transform.as_SE2()

        pos, angle_rad = self.weird_from_cartesian(pose)

        # c = self.cartesian_from_weird(pos, angle_rad)
        # logger.debug(desc=desc, pose=geometry.SE2.friendly(pose), weird=(pos, angle_rad),
        # c=geometry.SE2.friendly(c))

        # pos = desc["pos"]
        # x, z = pos[0:2]
        # y = pos[2] if len(pos) == 3 else 0.0

        # rotate = desc.get("rotate", 0.0)
        optional = desc.get("optional", False)

        # pos = self.road_tile_size * np.array((x, y, z))

        # Load the mesh

        if kind == MapFormat1Constants.KIND_DUCKIEBOT:
            use_color = desc.get("color", "red")

            mesh = get_duckiebot_mesh(use_color)

        elif kind.startswith("sign"):
            change_materials: Dict[str, MatInfo]
            # logger.info(kind=kind, desc=desc)
            minfo = cast(MatInfo, {"map_Kd": f"{kind}.png"})
            change_materials = {"April_Tag": minfo}
            mesh = get_mesh("sign_generic", change_materials=change_materials)
        elif kind == "floor_tag":
            return
        else:
            mesh = get_mesh(kind)

        if "height" in desc:
            scale = desc["height"] / mesh.max_coords[1]
        else:
            if "scale" in desc:
                scale = desc["scale"]
            else:
                scale = 1.0
        assert not ("height" in desc and "scale" in desc), "cannot specify both height and scale"

        static = desc.get("static", True)
        # static = desc.get('static', False)
        # print('static is now', static)

        obj_desc = {
            "kind": kind,
            "mesh": mesh,
            "pos": pos,
            "angle": angle_rad,
            "scale": scale,
            "optional": optional,
            "static": static,
        }

        if static:
            if kind == MF1C.KIND_TRAFFICLIGHT:
                obj = TrafficLightObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
            else:
                obj = WorldObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
        else:
            if kind == MF1C.KIND_DUCKIEBOT:
                obj = DuckiebotObj(
                    obj_desc, self.domain_rand, SAFETY_RAD_MULT, WHEEL_DIST, ROBOT_WIDTH, ROBOT_LENGTH
                )
            elif kind == MF1C.KIND_DUCKIE:
                obj = DuckieObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT, self.road_tile_size)
            elif kind == MF1C.KIND_CHECKERBOARD:
                obj = CheckerboardObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT, self.road_tile_size)
            else:
                msg = "Object kind unknown."
                raise InvalidMapException(msg, kind=kind)

        self.objects.append(obj)

        # Compute collision detection information

        # angle = rotate * (math.pi / 180)

        # # Find drivable tiles object could intersect with
        # # possible_tiles = find_candidate_tiles(obj.obj_corners, self.road_tile_size)

        # If the object intersects with a drivable tile
        if (
            static
            and kind != MF1C.KIND_TRAFFICLIGHT
            # We want collision checking also for things outside the lanes
            # # and self._collidable_object(obj.obj_corners, obj.obj_norm, possible_tiles)
        ):
            # noinspection PyUnresolvedReferences
            self.collidable_centers.append(pos)  # XXX: changes types during initialization
            self.collidable_corners.append(obj.obj_corners.T)
            self.collidable_norms.append(obj.obj_norm)
            # noinspection PyUnresolvedReferences
            self.collidable_safety_radii.append(obj.safety_radius)  # XXX: changes types during initialization

    def close(self):

        if not self.simulated:
            for agent in self.agents:
                agent.send_wheelVel([0.0, 0.0])

            self.optitrack.natnet_client.stop()
            rospy.signal_shutdown("Closing simulator.")

        sys.exit(0)

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def _set_tile(self, i: int, j: int, tile: TileDict) -> None:
        assert 0 <= i < self.grid_width
        assert 0 <= j < self.grid_height
        index: int = j * self.grid_width + i
        self.grid[index] = tile

    def _get_tile(self, i: int, j: int) -> Optional[TileDict]:
        """
        Returns None if the duckiebot is not in a tile.
        """
        i = int(i)
        j = int(j)
        if i < 0 or i >= self.grid_width:
            return None
        if j < 0 or j >= self.grid_height:
            return None
        self.grid[j * self.grid_width + i]['tile_n'] = j * self.grid_width + i
        return self.grid[j * self.grid_width + i]

    def _perturb(self, val: Union[float, np.ndarray, List[float]], scale: float = 0.1) -> np.ndarray:
        """
        Add noise to a value. This is used for domain randomization.
        """
        assert 0 <= scale < 1

        val = np.array(val)

        if not self.domain_rand:
            return val

        if isinstance(val, np.ndarray):
            noise = self.np_random.uniform(low=1 - scale, high=1 + scale, size=val.shape)
            if val.size == 4:
                noise[3] = 1
        else:
            noise = self.np_random.uniform(low=1 - scale, high=1 + scale)

        res = val * noise

        return res

    def _collidable_object(self, obj_corners, obj_norm, possible_tiles):
        """
        A function to check if an object intersects with any
        drivable tiles, which would mean our agent could run into them.
        Helps optimize collision checking with agent during runtime
        """

        if possible_tiles.shape == 0:
            return False

        drivable_tiles = []
        for c in possible_tiles:
            tile = self._get_tile(c[0], c[1])
            if tile and tile["drivable"]:
                drivable_tiles.append((c[0], c[1]))

        if not drivable_tiles:
            return False

        drivable_tiles = np.array(drivable_tiles)

        # Tiles are axis aligned, so add normal vectors in bulk
        tile_norms = np.array([[1, 0], [0, 1]] * len(drivable_tiles))

        # None of the candidate tiles are drivable, don't add object
        if len(drivable_tiles) == 0:
            return False

        # Find the corners for each candidate tile
        drivable_tiles = np.array(
            [
                tile_corners(self._get_tile(pt[0], pt[1])["coords"], self.road_tile_size).T
                for pt in drivable_tiles
            ]
        )

        # Stack doesn't do anything if there's only one object,
        # So we add an extra dimension to avoid shape errors later
        if len(tile_norms.shape) == 2:
            tile_norms = tile_norms[np.newaxis]
        else:  # Stack works as expected
            drivable_tiles = np.stack(drivable_tiles, axis=0)
            tile_norms = np.stack(tile_norms, axis=0)

        # Only add it if one of the vertices is on a drivable tile
        return intersects(obj_corners, drivable_tiles, obj_norm, tile_norms)

    def get_grid_coords(self, abs_pos: np.array) -> Tuple[int, int]:
        """
        Compute the tile indices (i,j) for a given (x,_,z) world position

        x-axis maps to increasing i indices
        z-axis maps to increasing j indices

        Note: may return coordinates outside of the grid if the
        position entered is outside of the grid.
        """

        x, _, z = abs_pos
        i = math.floor(x / self.road_tile_size)
        j = math.floor(z / self.road_tile_size)

        return int(i), int(j)

    def _get_curve(self, i, j):
        """
        Get the Bezier curve control points for a given tile
        """
        tile = self._get_tile(i, j)
        assert tile is not None

        kind = tile["kind"]
        angle = tile["angle"]

        # Each tile will have a unique set of control points,
        # Corresponding to each of its possible turns

        if kind.startswith("straight"):
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.25],
                            [-0.20, 0, 0.25],
                            [-0.20, 0, 0.50],
                        ],
                        [
                            [0.20, 0, 0.50],
                            [0.20, 0, 0.25],
                            [0.20, 0, -0.25],
                            [0.20, 0, -0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        elif kind == "curve_left":
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, 0.00],
                            [0.00, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [0.50, 0, -0.20],
                            [0.30, 0, -0.20],
                            [0.20, 0, -0.30],
                            [0.20, 0, -0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        elif kind == "curve_right":
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.20],
                            [-0.30, 0, -0.20],
                            [-0.50, 0, -0.20],
                        ],
                        [
                            [-0.50, 0, 0.20],
                            [-0.30, 0, 0.20],
                            [0.30, 0, 0.00],
                            [0.20, 0, -0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        # Hardcoded all curves for 3way intersection
        elif kind.startswith("3way"):
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.25],
                            [-0.20, 0, 0.25],
                            [-0.20, 0, 0.50],
                        ],
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, 0.00],
                            [0.00, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [0.20, 0, 0.50],
                            [0.20, 0, 0.25],
                            [0.20, 0, -0.25],
                            [0.20, 0, -0.50],
                        ],
                        [
                            [0.50, 0, -0.20],
                            [0.30, 0, -0.20],
                            [0.20, 0, -0.20],
                            [0.20, 0, -0.50],
                        ],
                        [
                            [0.20, 0, 0.50],
                            [0.20, 0, 0.20],
                            [0.30, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [0.50, 0, -0.20],
                            [0.30, 0, -0.20],
                            [-0.20, 0, 0.00],
                            [-0.20, 0, 0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        # Template for each side of 4way intersection
        elif kind.startswith("4way"):
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, 0.00],
                            [0.00, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.25],
                            [-0.20, 0, 0.25],
                            [-0.20, 0, 0.50],
                        ],
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.20],
                            [-0.30, 0, -0.20],
                            [-0.50, 0, -0.20],
                        ],
                    ]
                )
                * self.road_tile_size
            )
        else:
            msg = "Cannot get bezier for kind"
            raise InvalidMapException(msg, kind=kind)

        # Rotate and align each curve with its place in global frame
        if kind.startswith("4way"):
            fourway_pts = []
            # Generate all four sides' curves,
            # with 3-points template above
            for rot in np.arange(0, 4):
                mat = gen_rot_matrix(np.array([0, 1, 0]), rot * math.pi / 2)
                pts_new = np.matmul(pts, mat)
                pts_new += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])
                fourway_pts.append(pts_new)

            fourway_pts = np.reshape(np.array(fourway_pts), (12, 4, 3))
            return fourway_pts

        # Hardcoded each curve; just rotate and shift
        elif kind.startswith("3way"):
            threeway_pts = []
            mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
            pts_new = np.matmul(pts, mat)
            pts_new += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])
            threeway_pts.append(pts_new)

            threeway_pts = np.array(threeway_pts)
            threeway_pts = np.reshape(threeway_pts, (6, 4, 3))
            return threeway_pts

        else:
            mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
            pts = np.matmul(pts, mat)
            pts += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])

        return pts

    def closest_curve_point(
        self, pos: np.array, angle: float
    ) -> Tuple[Optional[np.array], Optional[np.array]]:
        """
        Get the closest point on the curve to a given point
        Also returns the tangent at that point.

        Returns None, None if not in a lane.
        """

        i, j = self.get_grid_coords(pos)
        tile = self._get_tile(i, j)

        if tile is None or not tile["drivable"]:
            return None, None

        # Find curve with largest dotproduct with heading
        curves = self._get_tile(i, j)["curves"]
        curve_headings = curves[:, -1, :] - curves[:, 0, :]
        curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
       # print(curve_headings)
        dir_vec = get_dir_vec(angle)

        dot_prods = np.dot(curve_headings, dir_vec)

        # Closest curve = one with largest dotprod
        cps = curves[np.argmax(dot_prods)]

        # Find closest point and tangent to this curve
        t = bezier_closest(cps, pos)
        point = bezier_point(cps, t)
        tangent = bezier_tangent(cps, t)

        return point, tangent

    def get_lane_pos2(self, pos, angle):
        """
        Get the position of the agent relative to the center of the right lane

        Raises NotInLane if the Duckiebot is not in a lane.
        """

        # Get the closest point along the right lane's Bezier curve,
        # and the tangent at that point
        point, tangent = self.closest_curve_point(pos, angle)
        if point is None or tangent is None:
            msg = f"Point not in lane: {pos}"
            raise NotInLane(msg)

        assert point is not None and tangent is not None

        # Compute the alignment of the agent direction with the curve tangent
        dirVec = get_dir_vec(angle)
        dotDir = np.dot(dirVec, tangent)
        dotDir = np.clip(dotDir, -1.0, +1.0)

        # Compute the signed distance to the curve
        # Right of the curve is negative, left is positive
        posVec = pos - point
        upVec = np.array([0, 1, 0])
        rightVec = np.cross(tangent, upVec)
        signedDist = np.dot(posVec, rightVec)

        # Compute the signed angle between the direction and curve tangent
        # Right of the tangent is negative, left is positive
        angle_rad = math.acos(dotDir)

        if np.dot(dirVec, rightVec) < 0:
            angle_rad *= -1

        angle_deg = np.rad2deg(angle_rad)
        # return signedDist, dotDir, angle_deg

        # Make dotDir negative if in wrong lane
        # if signedDist < -0.15:
        #     dotDir = -dotDir

        return LanePosition(dist=signedDist, dot_dir=dotDir, angle_deg=angle_deg, angle_rad=angle_rad)

    def _drivable_pos(self, pos) -> bool:
        """
        Check that the given (x,y,z) position is on a drivable tile
        """

        coords = self.get_grid_coords(pos)
        tile = self._get_tile(*coords)
        if tile is None:
            msg = f"No tile found at {pos} {coords}"
            logger.debug(msg)
            return False

        if not tile["drivable"]:
            msg = f"{pos} corresponds to tile at {coords} which is not drivable: {tile}"
            logger.debug(msg)
            return False

        return True

    def proximity_penalty2(self, pos: g.T3value, angle: float) -> float:
        """
        Calculates a 'safe driving penalty' (used as negative rew.)
        as described in Issue #24

        Describes the amount of overlap between the "safety circles" (circles
        that extend further out than BBoxes, giving an earlier collision 'signal'
        The number is max(0, prox.penalty), where a lower (more negative) penalty
        means that more of the circles are overlapping
        """

        pos = _actual_center(pos, angle)
        if len(self.collidable_centers) == 0:
            static_dist = 0

        # Find safety penalty w.r.t static obstacles
        else:
            d = np.linalg.norm(self.collidable_centers - pos, axis=1)

            if not safety_circle_intersection(d, AGENT_SAFETY_RAD, self.collidable_safety_radii):
                static_dist = 0.0
            else:
                static_dist = safety_circle_overlap(d, AGENT_SAFETY_RAD, self.collidable_safety_radii)

        total_safety_pen = static_dist
        for obj in self.objects:
            # Find safety penalty w.r.t dynamic obstacles
            total_safety_pen += obj.proximity(pos, AGENT_SAFETY_RAD)

        return total_safety_pen

    def _inconvenient_spawn(self, pos):
        """
        Check that agent spawn is not too close to any object
        """

        results = [
            np.linalg.norm(x.pos - pos) < max(x.max_coords) * 0.5 * x.scale + MIN_SPAWN_OBJ_DIST
            for x in self.objects
            if x.visible
        ]
        return np.any(results)

    def _collision(self, agent_corners):
        """
        Tensor-based OBB Collision detection
        """
        # Generate the norms corresponding to each face of BB
        agent_norm = generate_norm(agent_corners)

        # Check collisions with Static Objects
        if len(self.collidable_corners) > 0:
            collision = intersects(agent_corners, self.collidable_corners, agent_norm, self.collidable_norms)
            if collision:
                return True

        # Check collisions with Dynamic Objects
        for obj in self.objects:
            if obj.check_collision(agent_corners, agent_norm):
                return True

        # No collision with any object
        return False

    def _valid_pose(self, pos: g.T3value, angle: float, safety_factor: float = 1.0) -> bool:
        """
        Check that the agent is in a valid pose

        safety_factor = minimum distance
        """

        # Compute the coordinates of the base of both wheels
        pos = _actual_center(pos, angle)
        f_vec = get_dir_vec(angle)
        r_vec = get_right_vec(angle)

        l_pos = pos - (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
        r_pos = pos + (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
        f_pos = pos + (safety_factor * 0.5 * ROBOT_LENGTH) * f_vec

        # Check that the center position and
        # both wheels are on drivable tiles and no collisions

        all_drivable = (
            self._drivable_pos(pos)
            and self._drivable_pos(l_pos)
            and self._drivable_pos(r_pos)
            and self._drivable_pos(f_pos)
        )

        # Recompute the bounding boxes (BB) for the agent
        agent_corners = get_agent_corners(pos, angle)
        no_collision = not self._collision(agent_corners)

        res = no_collision and all_drivable

        # if not res:
        #     logger.debug(f"Invalid pose. Collision free: {no_collision} On drivable area: {all_drivable}")
        #     logger.debug(f"safety_factor: {safety_factor}")
        #     logger.debug(f"pos: {pos}")
        #     logger.debug(f"l_pos: {l_pos}")
        #     logger.debug(f"r_pos: {r_pos}")
        #     logger.debug(f"f_pos: {f_pos}")

        return res

    def _check_intersection_static_obstacles(self, pos: g.T3value, angle: float) -> bool:
        agent_corners = get_agent_corners(pos, angle)
        agent_norm = generate_norm(agent_corners)
        # logger.debug(agent_corners=agent_corners, agent_norm=agent_norm)
        # Check collisions with Static Objects
        if len(self.collidable_corners) > 0:
            collision = intersects(agent_corners, self.collidable_corners, agent_norm, self.collidable_norms)
            if collision:
                return True
        return False

    cur_pose: np.ndarray
    cur_angle: float
    speed: float

    def update_physics(self, agent, delta_time: float = None):
        # print("updating physics")
        if delta_time is None:
            delta_time = self.delta_time
        #self.wheelVels[v] = action * self.robot_speed * 1
        #prev_pos = self.cur_pos[v]
        agent.wheelsVels = agent.action * self.robot_speed * 1
        prev_pos = agent.pos

        # Update agent's position
        if agent.simulated:
            agent.pos, agent.angle = _update_pos(self, agent)
            
        else:
            agent.pos = self.optitrack.robots_pos[agent.name]

            # Taking Yaw Euler Angle
            agent.angle = self.optitrack.robots_ang[agent.name][0][2]

            # Roll and Pitch
            tilt_angles = np.abs(self.optitrack.robots_ang[agent.name][0][0:2])

            # Check if Roll and Pitch are within acceptable values
            # Ensures that: max_tilt < |angle| < 180deg - max_tilt -> stopping of physical robots
            if np.any(tilt_angles > self.robot_max_tilt_ang) or np.any(tilt_angles > np.pi - self.robot_max_tilt_ang):
                print(f"Stopping vehicles because of roll/pitch of {agent.name}")
                print(f"Max robot angle: {self.robot_max_tilt_ang}, tilt angles: {self.optitrack.robots_ang[agent.name][0][0:2]}")
                agent.invalid_real_pose = True
                self.reset()

            # print(f"agent pos = {agent.pos}, agent ang {agent.angle}")
            # try:
            #     (trans, rot) = self.tlistener.lookupTransform(agent.name, "/world", rospy.Time())
            #     print(f"OPTI POS: {trans}")
            #     rot = tf.transformations.euler_from_quaternion(rot)
            #     yaw = rot[2] # Getting Yaw (3rd Euler Angle), in RAD
            #     # TODO: after moving back to Natnet must care that the signs will change (trans and rot)
            #     agent.pos = np.array([-trans[0], 0, trans[1]])
            #     agent.angle = - yaw # In RAD
            #     print(f"AGENT POS: {agent.pos}")
            #     print(f"AGENT ANGLE: {agent.angle}, YAW: {yaw}")
            # except:
            #     print(f"Couldn't update pose of {agent.name}")

            # TODO: This is wrong, action should be given in
            # linear velocity and steering angle and we then convert to linear velocity of the wheels
            agent.send_wheelVel(agent.action)

        # self.step_count += 1
        # self.timestamp += delta_time

        agent.last_action = agent.action
        # Compute the robot's speed
        delta_pos = agent.pos - prev_pos
        angle_vector = np.array([np.cos(agent.angle), 0, -np.sin(agent.angle)])
        sign = np.sign(np.dot(angle_vector, delta_pos))
        if sign == 0:
            sign = 1
        agent.speed = sign * np.linalg.norm(delta_pos) / delta_time

        # Update world objects
        for obj in self.objects:
            if obj.kind == MapFormat1Constants.KIND_DUCKIEBOT:
                if not obj.static:
                    obj_i, obj_j = self.get_grid_coords(obj.pos)
                    same_tile_obj = [
                        o
                        for o in self.objects
                        if tuple(self.get_grid_coords(o.pos)) == (obj_i, obj_j) and o != obj
                    ]

                    obj.step_duckiebot(delta_time, self.closest_curve_point, same_tile_obj)
            else:
                # print("stepping all objects")
                obj.step(delta_time)

    #TODO: obsolete, need to fix
    def get_agent_info(self, v) -> dict:
        info = {}
        pos = self.agents[v].pos
        angle = self.agents[v].angle
        # Get the position relative to the right lane tangent

        info["action"] = list(self.last_action[v])
        if self.full_transparency:
            #             info['desc'] = """
            #
            # cur_pos, cur_angle ::  simulator frame (non cartesian)
            #
            # egovehicle_pose_cartesian :: cartesian frame
            #
            #     the map goes from (0,0) to (grid_height, grid_width)*self.road_tile_size
            #
            # """
            try:
                lp = self.get_lane_pos2(pos, angle)
                info["lane_position"] = lp.as_json_dict()
            except NotInLane:
                pass

            info["robot_speed"] = self.agents[v].speed
            info["proximity_penalty"] = self.proximity_penalty2(pos, angle)
            info["cur_pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]

            info["cur_angle"] = float(angle)
            info["wheel_velocities"] = [self.agents[v].wheelVels[0], self.agents[v].wheelVels[1]]

            # put in cartesian coordinates
            # (0,0 is bottom left)
            # q = self.cartesian_from_weird(self.cur_pos, self.)
            # info['cur_pos_cartesian'] = [float(p[0]), float(p[1])]
            # info['egovehicle_pose_cartesian'] = {'~SE2Transform': {'p': [float(p[0]), float(p[1])],
            #                                                        'theta': angle}}

            info["timestamp"] = self.timestamp
            info["tile_coords"] = list(self.get_grid_coords(pos))
            # info['map_data'] = self.map_data
        misc = {}
        misc["Simulator"] = info
        return misc

    def cartesian_from_weird(self, pos, angle) -> np.ndarray:
        gx, gy, gz = pos
        grid_height = self.grid_height
        tile_size = self.road_tile_size

        # this was before but obviously doesn't work for grid_height = 1
        # cp = [gx, (grid_height - 1) * tile_size - gz]
        cp = [gx, grid_height * tile_size - gz]

        return geometry.SE2_from_translation_angle(np.array(cp), angle)

    def weird_from_cartesian(self, q: SE2value) -> Tuple[list, float]:

        cp, angle = geometry.translation_angle_from_SE2(q)

        gx = cp[0]
        gy = 0
        # cp[1] = (grid_height - 1) * tile_size - gz
        GH = self.grid_height
        tile_size = self.road_tile_size
        # this was before but obviously doesn't work for grid_height = 1
        # gz = (grid_height - 1) * tile_size - cp[1]
        gz = GH * tile_size - cp[1]
        return [gx, gy, gz], angle

    def compute_reward(self, pos, angle, speed, tiles_visited):
        # Compute the collision avoidance penalty
        col_penalty = self.proximity_penalty2(pos, angle)

        reward = 0

        # #reward for every new drivable tile visited
        # tile = self._get_tile(pos[0] / self.road_tile_size, pos[2] / self.road_tile_size)
        # if tile:
        #     if tile['drivable'] and tile['tile_n'] not in tiles_visited:
        #         tiles_visited.append(tile['tile_n'])
        #         reward += 10

        # Get the position relative to the right lane tangent
        try:
            lp = self.get_lane_pos2(pos, angle)
            rew_dist = 0.3 - np.abs(lp.dist)
            rew_dir = speed * (lp.dot_dir ** 4)
            rew_col = col_penalty
            angle_deg = lp.angle_deg
            # reward = 10 * rew_dist + 20 * rew_dir + 40 * rew_col
            reward += 10 * rew_dir
            # print(f'lp.dist: {lp.dist}, lp.dir: {lp.dot_dir}, angle_deg: {angle_deg}' )
            # reward = lp.dot_dir

            # reward = + 5 * (1 - np.abs(lp.dist))
        except NotInLane:
            #reward = 40 * col_penalty
            reward += REWARD_INVALID_POSE
        # else:

            # Compute the reward

        return reward

    def step(self, action: np.ndarray):
        misc = {}
        #pos = []
        not_valid = []
        for v, agent in enumerate(self.agents):
        # for v in range(self.n_agents):
            # [speed, steering]
            curr_action = np.clip(action[v], -1, 1)

            # Actions could be a Python list
            agent.action = np.array(curr_action)

            for _ in range(self.frame_skip):
                self.update_physics(agent)

            #misc[v] = self.get_agent_info(v)

            if not self._valid_pose(agent.pos, agent.angle):
                # print(f"{agent.name} has invalid pose")
                if not self.simulated:
                    agent.invalid_real_pose = True
                not_valid.append(True)
            else:
                not_valid.append(False)

            # d = self._compute_done_reward(v)
            # misc[v]["Simulator"]["msg"] = d.done_why

        #Formation goals
        #agent.pos_2D = np.array(pos)
        # diffs_x = np.abs(agent.pos_2D[:, 0] - self.goal_xpoints)
        # diffs_z = np.abs(agent.pos_2D[:, 1] - self.goal_zpoints)

        # self.get_adjacency_matrix(agent.pos_2D)
        collisions = {}
        collision = False
        if False:
            for agent1 in self.agents:
                collisions[agent1] = {}
                for agent2 in self.agents:
                    if agent1 != agent2:
                        collisions[agent1][agent2] = radii_overlap(agent1.pos, agent2.pos, 0.125, 0.125, scale=self.road_tile_size)
                        if collisions[agent1][agent2]:
                            collision = True
        
        #TODO: I'm sure there's a cleaner way of setting these if-statements
        if any(not_valid):
            dones = self.n_agents * [True]
            reward = self.n_agents * [REWARD_INVALID_POSE]
        elif collision:
            print('!!!!!!!!!!!!!!!!!! COLLISION')
            dones = self.n_agents * [True]
            reward = self.n_agents * [REWARD_COLLISION]
        elif self.step_count > self.max_steps:
            dones = self.n_agents * [True]
            reward = self.n_agents * [0]
        else:
            reward = self._lane_following_reward()
            dones = self.n_agents * [False]

        # elif np.all(diffs_x < 0.2) and np.all(diffs_z < 0.2):
        #     dones = self.n_agents * [True]
        #     reward = self.calc_reward()
        # else:
        #     dones = self.n_agents * [False]
        #     reward = self.calc_reward()

        info = {}
        self.step_count += 1
        # self.timestamp += delta_time

        if self.mappo:
            reward = np.array(reward).reshape(1, self.n_agents, -1)
            dones = np.array(dones).reshape(1, -1)

        return self.get_obs(), reward, dones, info

    def dist2_mat(self, x):
        x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net

    def get_full_obs(self):
        _obs = []
        #for v in range(self.n_agents):
        for agent in self.agents:
            flat_list = []
            flat_list.extend(agent.pos.tolist())
            flat_list.append(agent.angle)
            flat_list.append(agent.speed)
            _obs.extend(flat_list)
        return _obs

    def get_adjacency_matrix(self):

        x = []
        for agent in self.agents:
            x.append(agent.pos.tolist())
        x = np.asarray(x)

        if self.degree == 0:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:,2:4])
            a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())

        if self.mean_pooling:
            # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
            n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents,1))
            n_neighbors[n_neighbors == 0] = 1
            a_net = a_net / n_neighbors

        return a_net

    def _lane_following_reward(self):
        rew = []
        #for v in range(self.n_agents):
        for agent in self.agents:
            rew.append(self.compute_reward(agent.pos, agent.angle, agent.speed, agent.tiles_visited))
        return rew


    def _compute_done_reward(self, v) -> DoneRewardInfo:
        # If the agent is not in a valid pose (on drivable tiles)
        if not self._valid_pose(self.agents[v].pos, self.agents[v].angle):
            msg = "Stopping the simulator because we are at an invalid pose."
            # logger.info(msg)
            reward = REWARD_INVALID_POSE
            done_code = "invalid-pose"
            done = True
        # If the maximum time step count is reached
        elif self.step_count >= self.max_steps:
            msg = "Stopping the simulator because we reached max_steps = %s" % self.max_steps
            # logger.info(msg)
            done = True
            reward = 0
            done_code = "max-steps-reached"
        else:
            done = False
            reward = self.compute_reward(self.agents[v].pos, self.agents[v].angle, self.robot_speed)
            msg = ""
            done_code = "in-progress"
        return DoneRewardInfo(done=done, done_why=msg, reward=reward, done_code=done_code)

    def calc_reward(self):
        """
        Get step reward and if episode is done
        return: reward, done
        """
        # Reward
        # If collision
        # if self.collision == 1:
        #     print('Collision!')
        #     reward = -2000
        #     print(self.v_follow)
        # elif self.collision == 0:
        # robot_xs = agent.pos_2D[:, 0]
        # robot_ys = agent.pos_2D[:, 1]
        robot_xs = [agent.pos[0] for agent in self.agents]
        robot_ys = [agent.pos[1] for agent in self.agents]

        robot_goalxs = self.goal_xpoints
        robot_goalys = self.goal_zpoints

        # diff = ((robot_xs - robot_goalxs)**2 + (robot_ys - robot_goalys)**2)**0.5
        diff = ((robot_ys - robot_goalys) ** 2) ** 0.5
        if self.fully_cooperative:
            reward =  100/np.sum(diff)
        else:
            reward = - diff.reshape(1, self.n_agents, -1)
        # else:
        #     reward = -1
        return self.n_agents * [reward]

    def _render_img(
        self,
        width: int,
        height: int,
        multi_fbo,
        final_fbo,
        img_array,
        top_down: bool = True,
        segment: bool = False,
    ) -> np.ndarray:
        """
        Render an image of the environment into a frame buffer
        Produce a numpy RGB array image as output
        """

        if not self.graphics:
            return np.zeros((height, width, 3), np.uint8)

        # Switch to the default context
        # This is necessary on Linux nvidia drivers
        # pyglet.gl._shadow_window.switch_to()
        self.shadow_window.switch_to()

        if segment:
            gl.glDisable(gl.GL_LIGHT0)
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_COLOR_MATERIAL)
        else:
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_COLOR_MATERIAL)

        # note by default the ambient light is 0.2,0.2,0.2
        # ambient = [0.03, 0.03, 0.03, 1.0]
        ambient = [0.3, 0.3, 0.3, 1.0]

        gl.glEnable(gl.GL_POLYGON_SMOOTH)

        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, (gl.GLfloat * 4)(*ambient))
        # Bind the multisampled frame buffer
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)
        gl.glViewport(0, 0, width, height)

        # Clear the color and depth buffers

        c0, c1, c2 = self.horizon_color if not segment else [255, 0, 255]
        gl.glClearColor(c0, c1, c2, 1.0)
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(self.cam_fov_y, width / float(height), 0.04, 100.0)

        # Set modelview matrix
        # Note: we add a bit of noise to the camera position for data augmentation
        

        # TODO: change this?
        # pos = [agent.pos for agent in self.agents]
        # angle = [agent.angle for agent in self.agents]
        
        # logger.info('Pos: %s angle %s' % (self.cur_pos, self.cur_angle))
        # if self.simulated and self.domain_rand:
        #     pos = pos + self.randomization_settings["camera_noise"]

        # x, y, z = pos + self.cam_offset
        # dx, dy, dz = get_dir_vec(angle)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        if self.draw_bbox:
            y += 0.8
            gl.glRotatef(90, 1, 0, 0)
        elif not top_down:
            y += self.cam_height
            gl.glRotatef(self.cam_angle[0], 1, 0, 0)
            gl.glRotatef(self.cam_angle[1], 0, 1, 0)
            gl.glRotatef(self.cam_angle[2], 0, 0, 1)
            gl.glTranslatef(0, 0, CAMERA_FORWARD_DIST)

        if top_down:
            a = (self.grid_width * self.road_tile_size) / 2
            b = (self.grid_height * self.road_tile_size) / 2
            fov_y_deg = self.cam_fov_y
            fov_y_rad = np.deg2rad(fov_y_deg)
            H_to_fit = max(a, b) + 0.1  # borders

            H_FROM_FLOOR = H_to_fit / (np.tan(fov_y_rad / 2))

            look_from = a, H_FROM_FLOOR, b
            look_at = a, 0.0, b - 0.01
            up_vector = 0.0, 1.0, 0
            gl.gluLookAt(*look_from, *look_at, *up_vector)
        else:
            look_from = x, y, z
            look_at = x + dx, y + dy, z + dz
            up_vector = 0.0, 1.0, 0.0
            gl.gluLookAt(*look_from, *look_at, *up_vector)

        # Draw the ground quad
        gl.glDisable(gl.GL_TEXTURE_2D)
        # background is magenta when segmenting for easy isolation of main map image
        gl.glColor3f(*self.ground_color if not segment else [255, 0, 255])  # XXX
        gl.glPushMatrix()
        gl.glScalef(50, 0.01, 50)
        self.ground_vlist.draw(gl.GL_QUADS)
        gl.glPopMatrix()

        # Draw the ground/noise triangles
        if not segment:
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.1, 0.0)
            self.tri_vlist.draw(gl.GL_TRIANGLES)
            gl.glPopMatrix()

        # Draw the road quads
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        add_lights = False
        if add_lights:
            for i in range(1):
                li = gl.GL_LIGHT0 + 1 + i
                # li_pos = [i + 1, 1, i + 1, 1]

                li_pos = [0.0, 0.2, 3.0, 1.0]
                diffuse = [0.0, 0.0, 1.0, 1.0] if i % 2 == 0 else [1.0, 0.0, 0.0, 1.0]
                ambient = [0.0, 0.0, 0.0, 1.0]
                specular = [0.0, 0.0, 0.0, 1.0]
                spot_direction = [0.0, -1.0, 0.0]
                logger.debug(
                    li=li, li_pos=li_pos, ambient=ambient, diffuse=diffuse, spot_direction=spot_direction
                )
                gl.glLightfv(li, gl.GL_POSITION, (gl.GLfloat * 4)(*li_pos))
                gl.glLightfv(li, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
                gl.glLightfv(li, gl.GL_DIFFUSE, (gl.GLfloat * 4)(*diffuse))
                gl.glLightfv(li, gl.GL_SPECULAR, (gl.GLfloat * 4)(*specular))
                gl.glLightfv(li, gl.GL_SPOT_DIRECTION, (gl.GLfloat * 3)(*spot_direction))
                # gl.glLightfv(li, gl.GL_SPOT_EXPONENT, (gl.GLfloat * 1)(64.0))
                gl.glLightf(li, gl.GL_SPOT_CUTOFF, 60)

                gl.glLightfv(li, gl.GL_CONSTANT_ATTENUATION, (gl.GLfloat * 1)(1.0))
                # gl.glLightfv(li, gl.GL_LINEAR_ATTENUATION, (gl.GLfloat * 1)(0.1))
                gl.glLightfv(li, gl.GL_QUADRATIC_ATTENUATION, (gl.GLfloat * 1)(0.2))
                gl.glEnable(li)

        # For each grid tile
        for i, j in itertools.product(range(self.grid_width), range(self.grid_height)):

            # Get the tile type and angle
            tile = self._get_tile(i, j)

            if tile is None:
                continue

            # kind = tile['kind']
            angle = tile["angle"]
            color = tile["color"]
            texture = tile["texture"]

            # logger.info('drawing', tile_color=color)

            gl.glColor4f(*color)

            gl.glPushMatrix()
            TS = self.road_tile_size
            gl.glTranslatef((i + 0.5) * TS, 0, (j + 0.5) * TS)
            gl.glRotatef(angle * 90 + 180, 0, 1, 0)

            # gl.glEnable(gl.GL_BLEND)
            # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            # Bind the appropriate texture
            texture.bind(segment)

            self.road_vlist.draw(gl.GL_QUADS)
            # gl.glDisable(gl.GL_BLEND)

            gl.glPopMatrix()

            if self.draw_curve and tile["drivable"]:
                # Find curve with largest dotproduct with heading
                curves = tile["curves"]
                curve_headings = curves[:, -1, :] - curves[:, 0, :]
                curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
                dirVec = get_dir_vec(angle)
                dot_prods = np.dot(curve_headings, dirVec)

                # Current ("closest") curve drawn in Red
                pts = curves[np.argmax(dot_prods)]
                bezier_draw(pts, n=20, red=True)

                pts = self._get_curve(i, j)
                for idx, pt in enumerate(pts):
                    # Don't draw current curve in blue
                    if idx == np.argmax(dot_prods):
                        continue
                    bezier_draw(pt, n=20)

        # For each object
        for obj in self.objects:
            obj.render(draw_bbox=self.draw_bbox, segment=segment, enable_leds=self.enable_leds)

        # Draw the agent's own bounding box
        
        for v, agent in enumerate(self.agents):
            if self.draw_bbox:
                corners = get_agent_corners(agent.pos, agent.angle)
                gl.glColor3f(1, 0, 0)
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glVertex3f(corners[0, 0], 0.01, corners[0, 1])
                gl.glVertex3f(corners[1, 0], 0.01, corners[1, 1])
                gl.glVertex3f(corners[2, 0], 0.01, corners[2, 1])
                gl.glVertex3f(corners[3, 0], 0.01, corners[3, 1])
                gl.glEnd()

            if top_down:
                gl.glPushMatrix()
                gl.glTranslatef(*agent.pos)
                gl.glScalef(1, 1, 1)
                gl.glRotatef(agent.angle * 180 / np.pi, 0, 1, 0)
                # glColor3f(*self.color)
                self.mesh.render()
                gl.glPopMatrix()
            draw_xyz_axes = False
            if draw_xyz_axes:
                draw_axes()

        # Waypoints
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(255, 255, 0)
        for waypoint in self.discretized_path[0]:
        
            #print(waypoint[0], 0.02, waypoint[1])
            gl.glVertex3f(waypoint[0], 0.01, waypoint[1])
        gl.glEnd()

        ##


        # Resolve the multisampled frame buffer into the final frame buffer
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, multi_fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, final_fbo)
        gl.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)
        gl.glReadPixels(
            0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_array.ctypes.data_as(POINTER(gl.GLubyte))
        )

        # Unbind the frame buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Flip the image because OpenGL maps (0,0) to the lower-left corner
        # Note: this is necessary for gym.wrappers.Monitor to record videos
        # properly, otherwise they are vertically inverted.
        img_array = np.ascontiguousarray(np.flip(img_array, axis=0))

        return img_array

    def render_obs(self, segment: bool = False) -> np.ndarray:
        """
        Render an observation from the point of view of the agent
        """

        observation = self._render_img(
            self.camera_width,
            self.camera_height,
            self.multi_fbo,
            self.final_fbo,
            self.img_array,
            top_down=False,
            segment=segment,
        )

        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort:
            observation = self.camera_model.distort(observation)

        return observation

    def render(self, mode: str = "human", close: bool = False, segment: bool = False):
        """
        Render the environment for human viewing

        mode: "human", "top_down", "free_cam", "rgb_array"

        """
        assert mode in ["human", "top_down", "free_cam", "rgb_array"]

        if close:
            if self.window:
                self.window.close()
            self.close()

        top_down = mode == "top_down"
        # Render the image
        img = self._render_img(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            self.multi_fbo_human,
            self.final_fbo_human,
            self.img_array_human,
            top_down=True,
            segment=segment,
        )

        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort and mode != "free_cam":
            img = self.camera_model.distort(img)

        if mode == "rgb_array":
            return img

        if self.window is None:
            config = gl.Config(double_buffer=False)
            self.window = window.Window(
                width=WINDOW_WIDTH, height=WINDOW_HEIGHT, resizable=False, config=config
            )

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # Bind the default frame buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Setup orghogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)

        # Draw the image to the rendering window
        width = img.shape[1]
        height = img.shape[0]
        img = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = image.ImageData(
            width,
            height,
            "RGB",
            img.ctypes.data_as(POINTER(gl.GLubyte)),
            pitch=width * 3,
        )
        img_data.blit(0, 0, 0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

        # Display position/state information
        if mode != "free_cam":
            x, y, z = self.agents[0].pos
            self.text_label.text = (
                f"pos: ({x:.2f}, {y:.2f}, {z:.2f}), angle: "
                f"{np.rad2deg(self.agents[0].angle):.1f} deg, steps: {self.step_count}, "
                f"speed: {self.agents[0].speed:.2f} m/s"
            )
            self.text_label.draw()

        # Force execution of queued commands
        gl.glFlush()

        return img


def get_dir_vec(cur_angle: float) -> np.ndarray:
    """
    Vector pointing in the direction the agent is looking
    """

    x = math.cos(cur_angle)
    z = -math.sin(cur_angle)
    return np.array([x, 0, z])


def get_right_vec(cur_angle: float) -> np.ndarray:
    """
    Vector pointing to the right of the agent
    """

    x = math.sin(cur_angle)
    z = math.cos(cur_angle)
    return np.array([x, 0, z])


def _update_pos(self, agent):
    """
    Update the position of the robot, simulating differential drive
    returns pos, angle
    """

    action = DynamicsInfo(motor_left=agent.action[0], motor_right=agent.action[1])
    agent.state = agent.state.integrate(self.delta_time, action)
    q = agent.state.TSE2_from_state()[0]
    pos, angle = self.weird_from_cartesian(q)
    pos = np.asarray(pos)
    return pos, angle


def get_duckiebot_mesh(color: str) -> ObjMesh:
    change_materials: Dict[str, MatInfo]

    color = np.array(get_duckiebot_color_from_colorname(color))[:3]
    change_materials = {
        "gkmodel0_chassis_geom0_mat_001-material": {"Kd": color},
        "gkmodel0_chassis_geom0_mat_001-material.001": {"Kd": color},
    }
    return get_mesh("duckiebot", change_materials=change_materials)


def _actual_center(pos, angle):
    """
    Calculate the position of the geometric center of the agent
    The value of self.cur_pos is the center of rotation.
    """

    dir_vec = get_dir_vec(angle)
    return pos + (CAMERA_FORWARD_DIST - (ROBOT_LENGTH / 2)) * dir_vec


def get_agent_corners(pos, angle):
    agent_corners = agent_boundbox(
        _actual_center(pos, angle), ROBOT_WIDTH, ROBOT_LENGTH, get_dir_vec(angle), get_right_vec(angle)
    )
    return agent_corners


class FrameBufferMemory:
    multi_fbo: int
    final_fbo: int
    img_array: np.ndarray
    width: int

    height: int

    def __init__(self, *, width: int, height: int):
        """H, W"""
        self.width = width
        self.height = height

        # that's right, it's inverted
        self.multi_fbo, self.final_fbo = create_frame_buffers(width, height, 4)
        self.img_array = np.zeros(shape=(height, width, 3), dtype=np.uint8)


def draw_axes():
    gl.glPushMatrix()
    gl.glLineWidth(4.0)
    gl.glTranslatef(0.0, 0.0, 0.0)

    gl.glBegin(gl.GL_LINES)
    L = 0.3
    gl.glColor3f(1.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(L, 0.0, 0.0)

    gl.glColor3f(0.0, 1.0, 0.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, L, 0.0)

    gl.glColor3f(0.0, 0.0, 1.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 0.0, L)
    gl.glEnd()

    gl.glPopMatrix()
