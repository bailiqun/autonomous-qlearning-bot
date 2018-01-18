import pygame
import random
import sys
import json
import numpy as np


def event_reader():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()


class Objects:
    def __init__(self, img_file, screen):
        self.screen = screen
        self.object = pygame.image.load(img_file).convert_alpha()
        self.x = 0
        self.y = 0
        self.width = self.object.get_width()
        self.height = self.object.get_height()

    def img_set(self, path):
        self.object = pygame.image.load(path).convert_alpha()

    def pose_set(self, x, y):
        self.x = x
        self.y = y

    def pose_get(self):
        return (self.x + self.width // 2, self.y + self.height // 2)

    def paint(self):
        # pygame.draw.rect(self.screen, (255,0,0), [self.pic_x, self.pic_y, self.pic_width, self.pic_height])
        self.screen.blit(self.object, (self.x, self.y))

    def box_get(self):
        return (self.x,
                self.y,
                self.x + self.width,
                self.y + self.height)


class Car(Objects):
    def __init__(self, path, screen):
        Objects.__init__(self, path, screen)
        self.boom = pygame.image.load('img/boom.png').convert_alpha()

    def is_collision(self, barriers):
        for barrier in barriers:
            box1 = self.box_get()
            box2 = barrier.box_get()
            minx = max(box1[0], box2[0])
            miny = max(box1[1], box2[1])
            maxx = min(box1[2], box2[2])
            maxy = min(box1[3], box2[3])
            collision = not (minx > maxx or miny > maxy)
            if collision:
                self.screen.blit(self.boom, (self.x-50, self.y-50))
                return True
        return False

    def pose_set(self, x):
        if x <= 0:
            x = 0
        if x >= self.screen.get_width() - self.width:
            x = self.screen.get_width() - self.width
        self.x = x
        self.y = self.screen.get_height() - self.height


class Barrier(Objects):
    def __init__(self, path, screen):
        Objects.__init__(self, path, screen)
        self.barrier = pygame.image.load(path).convert_alpha()
        self.x = - self.height
        self.y = - self.width
        self.div = 3

    def reset(self, id):
        self.x = (self.screen.get_width() / self.div) * id
        self.y = -self.height


class Barriers:
    def __init__(self, screen):
        self.barries = []
        self.barries.append(Barrier('img/b1.png', screen))
        self.barries.append(Barrier('img/b2.png', screen))
        self.screen = screen
        self.score = 0
        self.resolution = 10
        # dx, dy 可以通过道路位置中心
        self.road = {'x': 0, 'y': 0}
        self.reset()

    def generate_diff_num(self, min, max):
        num1 = random.randint(min, max)
        num2 = num1
        while num1 == num2:
            num2 = random.randint(min, max)
        a = [0, 1, 2]
        b = [num1, num2]
        # find different with a[]
        num3 = list(set(a) ^ set(b))[0]
        return num1, num2, num3

    def reset(self):
        # set Random barrier image
        n1, n2, n3 = self.generate_diff_num(1, 4)
        p1 = "img/b%d.png" % n1
        p2 = "img/b%d.png" % n2
        self.barries[0].img_set(p1)
        self.barries[1].img_set(p2)
        # set Random position of barrier
        n1, n2, n3 = self.generate_diff_num(0, 2)
        self.barries[0].reset(n1)
        self.barries[1].reset(n2)
        self.road['x'] = (self.screen.get_width() * n3 // 3)

    def run(self, speed):
        self.barries[0].y += speed
        self.barries[1].y += speed

        self.road['y'] = self.barries[0].y + self.barries[0].height

        if self.barries[0].y > self.screen.get_height():
            self.reset()
            self.score += 1

    def paint(self):
        self.barries[0].paint()
        self.barries[1].paint()


class Lines:
    def __init__(self, path, screen):
        self.lines = []
        self.lines.append(Objects(path, screen))
        self.lines.append(Objects(path, screen))
        self.screen = screen
        self.div = 3
        for line in self.lines:
            line.pose_set(-line.width, -line.height // 2)

    def run(self, speed):
        i = 1
        for line in self.lines:
            line.x = (self.screen.get_width() // self.div) * i
            line.y = line.y + speed
            if line.y >= 0:
                line.y = -line.height // 2
            i += 1

    def paint(self):
        self.lines[0].paint()
        self.lines[1].paint()


class QLearning:
    def __init__(self):
        self.Q = {}
        self.action = ['stay', 'left', 'right']
        self.reward = {'play': 1, 'dead': -100}
        self.resolution = 1
        self.lr = 0.7
        self.gamma = 0.9
        self.S = None
        self.A = None
        self.explore_jump_rate = 0.001

    def state_map(self, dx, dy):
        if dx == None or dy == None:
            return "0,0"
        return str(dx // self.resolution) + ',' + str(dy // self.resolution)

    def update(self, state, game_state):
        # prev state
        S = self.S
        # prev action
        A = self.A
        # current state
        S_ = self.state_map(state[0], state[1])
        # current action
        A_ = 0
        # add [S][A] in Q table
        if S_ and not (S_ in self.Q):
            self.Q[S_] = [0, 0, 0]

        if game_state == 'playing':
            if S and S_ and A in [0, 1, 2] and (S in self.Q) and (S_ in self.Q):
                self.Q[S][A] = (1 - self.lr) * self.Q[S][A] + \
                               self.lr * (self.reward['play'] + self.gamma * np.max(self.Q[S_]))

            if random.random() < self.explore_jump_rate:
                A_ = random.randint(0, 2)
            elif S_ in self.Q:
                A_ = np.argmax(self.Q[S_])

            self.S = S_
            self.A = A_

        elif game_state == 'dead':
            self.Q[S][A] = (1 - self.lr) * self.Q[S][A] + \
                           self.lr * (self.reward['dead'] + self.gamma * np.max(self.Q[S_]))
            self.S = None
            self.A = 0
            # update Q table
        return self.action[self.A]

    def load_qvalues(self):
        with open('qvalues.json', 'r') as file:
            self.Q = json.loads(file)
            print('Q-values read in game.')

    def save_qvalues(self):
        with open('qvalues.json', 'w') as file:
            json.dump(self.Q, file)
            print('Q-values updated on local file.')


def game_loop(game):
    car_x = 0
    ql = game['ql']
    screen = game['screen']
    episode = game['episode']
    car = Car('img/car.png', screen)
    lines = Lines('img/line.png', screen)
    barrs = Barriers(screen)
    indicator = Objects("img/arrow.png", screen)

    while True:
        # draw background color
        screen.fill(game['bk_color'])
        lines.paint()

        # event handler
        event_reader()

        # Update Q, (dx,dy) input as state.
        dx = (car.x - barrs.road['x'])
        dy = (car.y - barrs.road['y'])
        predict = ql.update((dx, dy), 'playing')

        if predict == 'left':
            car_x = car_x - screen.get_width() // 20
        elif predict == 'right':
            car_x = car_x + screen.get_width() // 20
        car.pose_set(car_x)

        # indicate which path is access
        indicator.pose_set(barrs.road['x'] + indicator.width//2, screen.get_height()*0.3)

        # episode
        font_size = 30
        string_episode = "episode " + str(episode)
        font = pygame.font.SysFont("arial", font_size)
        text_surface = font.render(string_episode, True, (255, 255, 255))
        screen.blit(text_surface, (screen.get_width()//2 - (len(string_episode)*font_size)//6, screen.get_height() - 50))

        # Paint in Pygame
        car.paint()
        barrs.paint()
        indicator.paint()

        # score
        font_size = 80
        score = barrs.score
        string_score = str(score)
        game['score'] = barrs.score
        font = pygame.font.SysFont("arial", font_size)
        text_surface = font.render(string_score, True, (255, 255, 255))
        screen.blit(text_surface, (screen.get_width()//2 - (len(string_score) * font_size)//6, 100))

        # save Q
        if score == 100:
            ql.save_qvalues()

        # detect collision
        if car.is_collision(barrs.barries):
            barrs.run(0)
            lines.run(0)
            pygame.display.flip()
            return
        else:
            barrs.run(game['car_speed'])
            lines.run(game['car_speed'])

        # Pygame update
        pygame.display.update()
        fps.tick(game['fps'])


if __name__ == "__main__":
    game = {
        # car run speed
        'car_speed': 15,
        # pygame windows size
        'window_size': {'width': 450, 'height': 600},
        # fps
        'fps': 60,
        # background color
        'bk_color': (46, 45, 49),
        # pygame screen
        'screen': None,
        # Qlearning instance
        'ql': None,
        # epsode of game
        'episode': 0,
        # game score
        'score': 0
    }

    pygame.init()
    fps = pygame.time.Clock()
    screen = pygame.display.set_mode((game['window_size']['width'], game['window_size']['height']), 0, 32)
    pygame.display.set_caption("Q-Learning")
    q = QLearning()
    game['screen'] = screen
    game['ql'] = q
    while True:
        game_loop(game)
        q.update((None, None), 'dead')
        game['episode'] += 1
        pygame.display.update()

