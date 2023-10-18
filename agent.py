import torch
import random
import numpy as np
from collections import deque
from snakeAgent import Point, Direction, AIGame
from model import Linear_QNet, QTrainer
from plot_score import plot


BLOCK_SIZE = 20

MAX_MEM = int(1e5)
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.8 # discount rate must be smaller than 1, generally to be 0.8 ~ 0.9
        self.mem = deque(maxlen=MAX_MEM)
        # model and the trainer
        self.model = Linear_QNet(input_size=11,
                                 hidden_size=128,
                                 output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_game_state(self, game):
        '''
        States:
            dead ends: [right, left, straight]
            current direction: [up, down, left, right]
            food location: [up, down, left, right]
        '''
        head = game.snake[0]
        up_pos = Point(head.x, head.y - BLOCK_SIZE)
        down_pos = Point(head.x, head.y + BLOCK_SIZE)
        left_pos = Point(head.x - BLOCK_SIZE, head.y)
        right_pos = Point(head.x + BLOCK_SIZE, head.y)
        
        up_direction = game.direction == Direction.UP
        down_direction = game.direction == Direction.DOWN
        left_direction = game.direction == Direction.LEFT
        right_direction = game.direction == Direction.RIGHT

        state = [
            # dead right
            (up_direction and game.collide(right_pos)) or
            (down_direction and game.collide(left_pos)) or
            (left_direction and game.collide(up_pos)) or
            (right_direction and game.collide(down_pos)),
            # dead straight
            (up_direction and game.collide(up_pos)) or
            (down_direction and game.collide(down_pos)) or
            (left_direction and game.collide(left_pos)) or
            (right_direction and game.collide(right_pos)),
            # dead left
            (up_direction and game.collide(left_pos)) or
            (down_direction and game.collide(right_pos)) or
            (left_direction and game.collide(down_pos)) or
            (right_direction and game.collide(up_pos)),

            # move directions [up, down, left, right]
            up_direction,
            down_direction,
            left_direction,
            right_direction,

            # food position [up, down, left, right]
            game.food.y < game.head.y,
            game.food.y > game.head.y,
            game.food.x < game.head.x,
            game.food.x > game.head.x
            # game.food.y - game.head.y,
            # game.food.y - game.head.y,
            # game.food.x - game.head.x,
            # game.food.x - game.head.x
        ]

        return np.array(state, dtype=int)
        

    def memorize(self, state, action, reward, next_state, lose):
        # pop[0] if reach max len of the self.mem
        self.mem.append((state, action, reward, next_state, lose))

    def train_long_mem(self):
        if len(self.mem) > BATCH_SIZE:
            # random sample
            train_batch = random.sample(self.mem, BATCH_SIZE)
        else:
            train_batch = self.mem
        
        # use zip(*batch) to zip states, action ...
        # same with the following block:
        #   for state, action, reward, next_state, lose in train_batch:
        #         self.trainer.train_step(state, action, reward, next_state, lose)    
        states, actions, rewards, next_states, loses = zip(*train_batch)
        self.trainer.train_step(states, actions, rewards, next_states, loses)
        

    def train_short_mem(self, state, action, reward, next_state, lose):
        self.trainer.train_step(state, action, reward, next_state, lose)

    def get_action(self, state):
        # random actions: trade-off exploration / exploitation
        # randomly pick one direction (right, straight, left)
        self.epsilon = 100 - self.num_games
        final_action = [0,0,0] # left, stright, right
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0,2)
            final_action[action] = 1
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            # tensor.item() => convert to one number
            # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        
        return final_action


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = AIGame()

    while True:
        # get previous state
        prev_state = agent.get_game_state(game)

        # based on previous state, get the action
        final_action = agent.get_action(prev_state)

        # perform the final action and get the new state
        lose, score, reward = game.play(final_action)
        new_state = agent.get_game_state(game)

        # train based on the one state (short memory)
        agent.train_short_mem(state=prev_state,
                              action=final_action,
                              reward=reward,
                              next_state=new_state,
                              lose=lose)
        # memorize the single state
        agent.memorize(state=prev_state,
                       action=final_action,
                       reward=reward,
                       next_state=new_state,
                       lose=lose)
        
        if lose:
            # train replay memory (experience replay) (long memory)
            game.reset()
            agent.num_games += 1
            agent.train_long_mem()

            if score > record:
                record = score
                agent.model.save()

            print("Game: ", agent.num_games, ' Score: ', score, ' Record: ', record)

            # plot the training process
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_score.append(mean_score)
            plot(scores=plot_scores, mean_scores=plot_mean_score)

if __name__ == '__main__':
    train()
