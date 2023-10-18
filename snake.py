import pygame as pg
import random
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
FPS = 10
BLOCK_SIZE = 20


class Game:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.fps = FPS

        # display
        self.display = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("Snake Game")
        self.clock = pg.time.Clock()

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
        
    # generate a food block at random pos
    def _gen_food(self):
        x = random.randint(0, (self.width - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x,y)
        if self.food in self.snake:
            self._gen_food()
    
    def play(self):
        lose = False
        self.clock.tick(self.fps)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pg.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pg.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
                elif event.key == pg.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
        self._move(self.direction) # update head
        self.snake.insert(0, self.head)

        if self._collide():
            lose = True
            return lose, self.score

        
        # pos head overlap with pos food
        # food is eaten, generate new food
        if self.food == self.head:
            self.score += 1
            self._gen_food()
        else:
            self.snake.pop()

        self._update_display()

        return lose, self.score
    
    def _collide(self):
        # out of boundary
        if (self.head.x > self.width - BLOCK_SIZE or
            self.head.y > self.height - BLOCK_SIZE or
            self.head.x < 0 or
            self.head.y < 0):
            return True
        
        # collide with itself
        if self.head in self.snake[1:]:
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
    
    def _move(self, direction):
        x, y = self.head.x, self.head.y
        if direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.RIGHT:
            x += BLOCK_SIZE
        
        self.head = Point(x, y)
        
if __name__ == '__main__':
    game = Game()
    while True:
        lose, score = game.play()

        if lose:
            break
    print("Score: ", score)
    pg.quit()
