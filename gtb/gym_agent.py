import gym
from gym import spaces
import numpy as np
from PIL import Image

from .fixers import pad_rows_to_max_length


class CustomEnv(gym.Env):
    def __init__(self, walkable_tiles, str_map_without_chars, str_map, interactive_object_tiles, enemy_tiles):
        super(CustomEnv, self).__init__()
        str_map = pad_rows_to_max_length(str_map)
        str_map_without_chars = pad_rows_to_max_length(str_map_without_chars)
        
        self.map_str_without_chars = str_map_without_chars.strip().split('\n')
        self.map_str = str_map.strip().split('\n')
        
        self.tile_size = 16
        self.char_tile_size = 16
        
        self.action_space = spaces.Discrete(4)  # Up, down, left, right, pick, hit
        self.observation_space = spaces.Box(low=0, high=255, shape=(len(self.map_str) * self.tile_size, len(self.map_str[0]) * self.tile_size, 3), dtype=np.uint8)
        self.default_walkable_tile = "B"
        
        self.walkable_tiles = walkable_tiles
        self.interactive_object_tiles = interactive_object_tiles
        self.enemy_tiles = enemy_tiles
        self.picked_objects = []


        # Count the occurrences of each tile in the map
        tile_counts = {}
        for row in self.map_str:
            for tile in row:
                if tile in walkable_tiles:
                    if tile not in tile_counts:
                        tile_counts[tile] = 1
                    else:
                        tile_counts[tile] += 1

        # Determine the most common walkable tile
        if tile_counts:
            self.default_walkable_tile = max(tile_counts, key=tile_counts.get)
        else:
            raise ValueError("No walkable tiles found in the map.")
        

        self.reset()

    def reset(self):
        self.map = [list(row) for row in self.map_str]
        self.map_without_chars = [list(row) for row in self.map_str_without_chars]
        self.grid_width = max(len(row) for row in self.map)
        self.grid_height = len(self.map)
        self.player_pos = self.find_player_position()
        self.current_tile = self.default_walkable_tile  # Default current tile to 'A', change if necessary
        return self.get_state()

    def step(self, action):
        reward = 0
        if action < 4:  # Movement actions
            self.move_player(action)
        elif action == 4:  # Pick action
            reward += self.pick_object()
        elif action == 5:  # Hit action
            reward += self.hit_enemy()

        
        done = False
        info = {}
        return self.get_state(), reward, done, info

    def move_player(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Up, Down, Left, Right
        dx, dy = moves[action]
        new_row = self.player_pos[0] + dx
        new_col = self.player_pos[1] + dy

        if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]):
            new_tile = self.map[new_row][new_col]
            if new_tile in self.walkable_tiles:
                self.update_player_position(new_row, new_col, new_tile)

    def update_player_position(self, new_row, new_col, new_tile):
        self.map[self.player_pos[0]][self.player_pos[1]] = self.current_tile
        self.player_pos = (new_row, new_col)
        self.current_tile = new_tile
        self.map[new_row][new_col] = '@'

    def pick_object(self):
        reward = 0
        # Check adjacent tiles for interactive objects and pick them if present
        adjacent_positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        for dx, dy in adjacent_positions:
            x, y = self.player_pos
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                target_tile = self.map[new_y][new_x]
                if target_tile in self.interactive_object_tiles:
                    print("Picked an object!")
                    self.map[new_y][new_x] = self.default_walkable_tile 
                    reward = 1
                    break  # Exit after picking up one object
        return reward

    def hit_enemy(self):
        reward = 0
        # Check adjacent tiles for enemies and hit them if present
        adjacent_positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        for dx, dy in adjacent_positions:
            x, y = self.player_pos
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                target_tile = self.map[new_y][new_x]
                if target_tile in self.enemy_tiles:  # Assuming enemy_tiles is a list of enemy tile identifiers
                    print("Hit an enemy!")
                    self.map[new_y][new_x] = self.default_walkable_tile  # Replace with default or empty tile
                    reward = 5
                    break  # Exit after hitting one enemy
        return reward
    def get_state(self):
        #print(self.map)
        row_lengths = [len(row) for row in self.map]
        assert len(set(row_lengths)) == 1, "Not all rows in the map have the same length"
        return np.array(self.map)

    def render(self, mode='human'):
         # Draw the picked objects and their count
        if mode == 'human' or mode == 'rgb_array':
            print(self.map_str)
            
    def find_player_position(self):
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile == '@':
                    return (i, j)
        return None


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def action(self):
        return self.action_space.sample()
    
class LLMAgent:
    def __init__(self):
        pass

    def action(self, action_string):
        if action_string == 'move_up':
            return 0
        if action_string == 'move_down':
            return 1
        if action_string == 'move_left':
            return 2
        if action_string == 'move_right':
            return 3
        if action_string == 'pick_object':
            return 4
        if action_string == 'hit_enemy':
            return 5