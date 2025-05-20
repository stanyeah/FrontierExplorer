#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
from tf2_ros import Buffer, TransformListener

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        self.navigator = BasicNavigator()
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_cb, 1)
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)
        self.current_map = None

    def map_cb(self, msg: OccupancyGrid):
        self.current_map = msg

    def detect_frontiers(self, grid, width, height):
        data = np.array(grid).reshape((height, width))
        frontiers = []
        for y in range(1, height-1):
            for x in range(1, width-1):
                if data[y,x] != 0: continue
                neigh = data[y-1:y+2, x-1:x+2].flatten()
                if -1 in neigh:
                    frontiers.append((x,y))
        return frontiers

    def cluster_frontiers(self, frontiers):
        # naive clustering by BFS on 8-connected grid
        pts = set(frontiers)
        clusters = []
        while pts:
            stack = [pts.pop()]
            cluster = []
            while stack:
                cx, cy = stack.pop()
                cluster.append((cx, cy))
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        nb = (cx+dx, cy+dy)
                        if nb in pts:
                            pts.remove(nb)
                            stack.append(nb)
            clusters.append(cluster)
        return clusters

    def run(self):
        self.get_logger().info('Waiting for first map…')
        while rclpy.ok() and self.current_map is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Loop forever
        while rclpy.ok():
            # 1) Get robot pose in map frame
            try:
                trans = self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time())
                rx, ry = trans.transform.translation.x, trans.transform.translation.y
            except Exception as e:
                self.get_logger().warn(f'No TF yet: {e}')
                rclpy.spin_once(self, timeout_sec=0.1)
                continue

            # 2) Detect & cluster frontiers
            m = self.current_map
            F = self.detect_frontiers(m.data, m.info.width, m.info.height)
            clusters = self.cluster_frontiers(F)
            if not clusters:
                self.get_logger().info('Exploration complete!')
                break

            # 3) Score clusters by min distance to robot
            best = min(clusters, key=lambda c: min(
                ((x*m.info.resolution + m.info.origin.position.x - rx)**2 +
                 (y*m.info.resolution + m.info.origin.position.y - ry)**2)
                for x,y in c))
            # 4) Compute centroid goal
            xs, ys = zip(*best)
            gx = np.mean(xs)*m.info.resolution + m.info.origin.position.x
            gy = np.mean(ys)*m.info.resolution + m.info.origin.position.y

            # 5) Send Nav2 goal
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = float(gx)
            goal.pose.position.y = float(gy)
            goal.pose.orientation.w = 1.0
            self.get_logger().info(f'Navigating to frontier at ({gx:.2f},{gy:.2f})')
            self.navigator.goToPose(goal)
            self.navigator.waitUntilNavFinished()

        self.get_logger().info('Frontier exploration done.')
        rclpy.shutdown()

def main():
    rclpy.init()
    explorer = FrontierExplorer()
    explorer.run()

if __name__=='__main__':
    main()
