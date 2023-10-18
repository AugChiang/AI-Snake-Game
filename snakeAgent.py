import pygame as pg
import random
import numpy as np
from enum import Enum # ref: https://haosquare.com/python-enum/
from collections import namedtuple

pg.init()
STAT_FONT = pg.font.SysFont("arial", 24)


class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

Point = namedtuple('Point', 'x,y')

# colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (200,0,0)
GREEN1 = (0,200,20)
GREEN2 = (0,180,20)
BLUE1 = (0,0,250)
BLUE2 = (0,50,200)

# window size
WIDTH, HEIGHT = 800, 600
FPS = 60
BLOCK_SIZE = 20


class AIGame:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.fps = FPS

        # display
        self.display = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("Snake Game")
        self.clock = pg.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.width//2, self.height//2)

        # list of block pos
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._gen_food()
        self.frame_iter = 0
        
    # generate a food block at random pos
    def _gen_food(self):
        x = random.randint(0, (self.width - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x,y)
        if self.food in self.snake:
            self._gen_food()
    
    def play(self, action):
        lose = False
        reward = 0
        self.frame_iter += 1
        self.clock.tick(self.fps)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        self._move(action) # update head
        self.snake.insert(0, self.head)

        # modify with reward and if snake falls into a Hamilton paths
        if self.collide() or self.frame_iter > 100*len(self.snake):
            lose = True
            reward = -10
            return lose, self.score, reward

        
        # pos head overlap with pos food
        # food is eaten, generate new food
        if self.food == self.head:
            self.score += 1
            reward = 10
            self._gen_food()
        else:
            self.snake.pop()

        self._update_display()

        return lose, self.score, reward
    
    def collide(self, pt=None):
        if pt is None:
            pt = self.head

        # collide with itself
        if pt in self.snake[1:]:
            return True

        # out of boundary
        if (pt.x > self.width - BLOCK_SIZE or
            pt.y > self.height - BLOCK_SIZE or
            pt.x < 0 or
            pt.y < 0):
            return True
        

        
        return False
    
    def _update_display(self):
        self.display.fill(BLACK)

        for pos in self.snake:
            pg.draw.rect(self.display, GREEN1, (pos.x, pos.y, BLOCK_SIZE, BLOCK_SIZE))
            pg.draw.rect(self.display, GREEN2, (pos.x+4, pos.y+4, 12, 12))
        
        pg.draw.rect(self.display, RED, (self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        score_label = STAT_FONT.render(f"Score: {self.score}", 1, WHITE)
        self.display.blit(score_label, (0,0))
        pg.display.update()
    
    def _move(self, action):
        # [left, straight, right]
        clockwise = [Direction.RIGHT, Direction.DOWN,
                     Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction) # self.direction = some enum value

        if np.array_equal(action, [0,1,0]):
            new_direction = clockwise[idx] # no change direction
        elif np.array_equal(action, [1,0,0]):
            new_direction = clockwise[(idx+1)%4] # right turn
        else: # np.array_equal(action, [0,0,1]):
            new_direction = clockwise[(idx-1)%4] # left turn

        self.direction = new_direction

        x, y = self.head.x, self.head.y
        if self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        
        self.head = Point(x, y)
