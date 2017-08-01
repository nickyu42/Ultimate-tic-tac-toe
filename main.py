"""
Text based version of ultimate tic-tac-toe game
written in python 3.5 by Nick Yu

The full game board consists of 9 smaller tic tac toe boards
where each inner board indexed as

 1 | 2 | 3
-----------
 4 | 5 | 6
-----------
 7 | 8 | 9
"""
import pygame as pg
from pygame.locals import HWSURFACE, DOUBLEBUF
import numpy as np
import sys
import os

from math import ceil
from random import uniform, randint

# Constants
PLAYER_SIGN = {1: 'x', 2: 'o'}

SCREEN_WIDTH = 720
SCREEN_HEIGHT = 760
BLOCK_SIZE = SCREEN_WIDTH // 3
CELL_SIZE = BLOCK_SIZE // 3

X_COLOR     = (84 ,  84,  84)
O_COLOR     = (242, 235, 211)
BOARD_COLOR = (13 , 161, 146)
BG_COLOR    = (20 , 189, 172)
TEXT_COLOR  = (128, 128, 128)


class Particle:
    """Particle class for firework sfx"""
    def __init__(self, position, color):
        self.x, self.y = position
        self.velocity_x = uniform(-5, 5)
        self.velocity_y = uniform(-10, 10)
        self.gravity = .5
        self.lifespan = 100
        self.color = color
        self.circle_radius = randint(2, 3)

    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.velocity_y += self.gravity
        self.lifespan -= 1

    def draw(self):
        pg.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.circle_radius)

    @property
    def dead(self):
        if self.lifespan <= 0:
            return True
        elif not 0 <= self.x <= SCREEN_WIDTH:
            return True
        elif not self.y <= SCREEN_HEIGHT:
            return True
        else:
            return False


class Firework:
    def __init__(self, position, color):
        self.lifespan = 150
        self.x = position
        self.y = SCREEN_HEIGHT
        self.gravity = .1
        self.velocity = randint(8, 12)
        self.color = color

    def update(self):
        self.y -= self.velocity
        self.velocity -= self.gravity
        self.lifespan -= 1

    def draw(self):
        pg.draw.circle(screen, self.color, (int(self.x), int(self.y)), 3)

    @property
    def dead(self):
        if self.lifespan <= 0:
            return True
        else:
            return False


def _check_same(lst):
    """Check if all the items in an array are the same and unequal to . (empty)"""
    return all([x == lst[0] for x in lst]) and lst[0] != '.'


def is_captured(board):
    """
    :param board: np.ndarray of shape (3, 3)
    :return: bool
    """
    if isinstance(board, str):
        return True

    # check rows and columns
    for i in range(3):
        if _check_same(board[i, :]) or _check_same(board[:, i]):
            return True

    # check diagonals
    if _check_same([board[i, i] for i in range(3)]):
        return True

    elif _check_same([board[x, y] for x, y in ((0, 2), (1, 1), (2, 0))]):
        return True

    return False


def is_captured_all():
    """checks if the entire board has been captured"""
    board = np.full((3, 3), '.', dtype=str)

    for index, p in full_board.items():
        if not isinstance(p, np.ndarray):
            block_x = (index - 1) % 3
            block_y = ceil(index / 3) - 1
            board[block_y, block_x] = p

    return is_captured(board)


def get_index(x, y):
    """
    Get board index based on x, y coord
    :param x: Global x index
    :param y: Global y index
    :return: board_index
    """
    board_indices = np.arange(1, 10).reshape(3, 3)
    board_index = board_indices[ceil(y / 3) - 1, ceil(x / 3) - 1]
    return board_index


def get_inner(x, y):
    """
    Changes move notation to indices
    :param x: Global x index
    :param y: Global y index
    :return: x inner board, y inner board
    """
    # change to inner board
    x = (x - 1) % 3
    y = (y - 1) % 3

    return x, y


def is_valid(move, board_index, next_board):
    """
    Checks if a move is valid
    :param move: tuple(x, y)
    :param board_index: index of inner board
    :param next_board: index of next board the player is forced to play in
    """
    x, y = move
    inner_board = full_board[board_index]

    # first check if it's the next
    # and check if it's 0
    if board_index != next_board and next_board:
        return False

    # check if it's already captured
    # hack!
    elif not isinstance(inner_board, np.ndarray):
        return False

    # if it's not captured check if that spot is open
    elif inner_board[y, x] in 'xo':
        return False

    else:
        return True


def display_msg(msg):
    """Write text at the bottom of the screen"""
    text_surface = font.render(msg, True, TEXT_COLOR)
    text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_WIDTH + 20))

    screen.blit(text_surface, text_rect)


def display_text(text, center, size=None, col=(245, 245, 245)):
    """
    :param text: text to display
    :param center: (center x, center y)
    :param size: (width, height), optional parameter
    :param col: rgb tuple of color otherwise white
    """
    text_surface = font.render(text, True, col)
    if size:
        pg.transform.scale(text_surface, size)

    text_rect = text_surface.get_rect(center=center)
    screen.blit(text_surface, text_rect)


def draw_symbol(symbol, x, y, size, ghost=False):
    """
    Draws an x or o on the screen
    :param symbol: x or o
    :param x: global cell x index from 0 - 8
    :param y: global cell y index from 0 - 8
    :param size: square size in pixels
    :param ghost: if the symbol to be placed is a ghost, give transparency
    """
    if symbol == "x":
        # make the ghost slightly lighter in color
        col = [c + 20 for c in X_COLOR] if ghost else X_COLOR
        pg.draw.line(screen,
                     col,
                     (x * size + 10, y * size + 10),
                     ((x + 1) * size - 13, (y + 1) * size - 13),
                     10)
        pg.draw.line(screen,
                     col,
                     ((x + 1) * size - 13, y * size + 10),
                     (x * size + 10, (y + 1) * size - 13),
                     10)
    elif symbol == "o":
        col = [c + 10 for c in O_COLOR] if ghost else O_COLOR
        pg.draw.circle(screen,
                       col,
                       (x * size + size // 2, y * size + size // 2),
                       size // 2 - 12,
                       3)


def draw_block(start_pos, square_size, line_width):
    """
    Draws tic tac toe board
    :param start_pos: left corner of the square
    :param square_size: self-explanatory
    :param line_width: width of dividers
    """
    x, y = start_pos
    margin = line_width / 2
    for i in range(1, 3):
        start = (x, y + i * square_size - margin)
        end = (x + 3 * square_size, y + i * square_size - margin)
        pg.draw.line(screen, BOARD_COLOR, start, end, line_width)

        start = (x + i * square_size - margin, y)
        end = (x + i * square_size - margin, y + 3 * square_size)
        pg.draw.line(screen, BOARD_COLOR, start, end, line_width)


def draw_ghost(pos, symbol, next_board):
    """
    Draw a ghost piece if the block is not captured and the pos isn't on top of another symbol
    :param pos: (global x, global y) index
    :param symbol: x or o
    :param next_board: where the next player is supposed to play
    """
    board_index = get_index(*pos)
    x, y = get_inner(*pos)

    if isinstance(full_board[board_index], np.ndarray):
        if (board_index == next_board or next_board == 0) and full_board[board_index][y, x] == '.':
            draw_symbol(symbol, pos[0]-1, pos[1]-1, CELL_SIZE, ghost=True)


def draw_bg():
    screen.fill(BG_COLOR)

    for i in range(3):
        draw_block((0, 0), BLOCK_SIZE, 6)

    for y in range(3):
        for x in range(3):
            draw_block((x * BLOCK_SIZE + 10, y * BLOCK_SIZE + 10), (BLOCK_SIZE - 20) / 3, 2)


def draw_board():
    for index, board in full_board.items():
        block_x = (index - 1) % 3
        block_y = ceil(index / 3) - 1

        if not isinstance(board, np.ndarray):
            draw_symbol(board, block_x, block_y, BLOCK_SIZE)
        elif 'x' in board or 'o' in board:
            captured = np.where(np.logical_or((board == 'x'), (board == 'o')))
            for y, x in zip(*captured):
                draw_symbol(board[y, x], block_x * 3 + x, block_y * 3 + y, CELL_SIZE)


class App:
    def __init__(self):
        self.ended = False
        # where the next player has to play
        # if it's 0 it means the next player can play where he wants
        self.next_board = 0
        self.turn = 1
        self.msg = "It's x's turn"

    def fireworks(self):
        """Blits a back starry background and fireworks until a key is pressed"""
        bg = pg.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        bg.fill((0, 0, 0))

        # puts white stars in the bg
        for _ in range(25):
            pg.draw.circle(bg,
                           (240, 240, 240),
                           (randint(0, SCREEN_WIDTH), randint(0, SCREEN_HEIGHT)),
                           2)

        fireworks = []
        particles = []

        spawned_time = pg.time.get_ticks()
        self.ended = False
        while not self.ended:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.ended = True
                elif event.type == pg.KEYDOWN:
                    self.ended = True

            if pg.time.get_ticks() - spawned_time > 800:
                spawned_time = pg.time.get_ticks()
                fireworks.append(Firework(randint(0, SCREEN_WIDTH), [randint(10, 255) for _ in range(3)]))

            screen.blit(bg, (0, 0))

            for p in particles:
                p.update()
                p.draw()
                if p.dead:
                    particles.remove(p)

            for f in fireworks:
                f.update()
                f.draw()
                if f.dead:
                    fireworks.remove(f)
                    particles.extend([Particle((f.x, f.y), f.color) for _ in range(100)])

            display_text('Congratulations {} for winning the game'.format(PLAYER_SIGN[self.turn]),
                         (SCREEN_WIDTH // 2, SCREEN_WIDTH // 2))

            pg.display.flip()
            clock.tick(60)

    def set_move(self, x, y, board_index):
        """
        :param x: inner_x
        :param y: inner_y
        :param board_index: which board was clicked in
        """
        full_board[board_index][y, x] = PLAYER_SIGN[self.turn]

        if is_captured(full_board[board_index]):
            # set the inner board as captured
            full_board[board_index] = PLAYER_SIGN[self.turn]
            if is_captured_all():
                self.msg = '{} has won the game!'.format(PLAYER_SIGN[self.turn])
                self.render()
                self.ended = True

        # if the next board is captured
        # make it so the next player can set anywhere
        board_indices = np.arange(1, 10).reshape(3, 3)
        tmp_next_board = board_indices[y, x]

        if is_captured(full_board[tmp_next_board]):
            self.next_board = 0
        else:
            self.next_board = tmp_next_board

    def render(self):
        draw_bg()
        draw_board()

        pos = pg.mouse.get_pos()
        pos = (ceil(pos[0] / CELL_SIZE), ceil(pos[1] / CELL_SIZE) if pos[1] <= SCREEN_WIDTH else 9)

        draw_ghost(pos, PLAYER_SIGN[self.turn], self.next_board)

        display_msg(self.msg)

    def handle_input(self, x, y):
        x = ceil(x / CELL_SIZE)
        y = ceil(y / CELL_SIZE)

        board_index = get_index(x, y)
        x, y = get_inner(x, y)

        if is_valid((x, y), board_index, self.next_board):
            self.set_move(x, y, board_index)
            self.turn = 1 if self.turn == 2 else 2
            self.msg = "It's {}'s turn".format(PLAYER_SIGN[self.turn])
        else:
            self.msg = "You can't play there"

    def event_handler(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                # force escape if the game wasn't completed
                sys.exit()

            elif event.type == pg.MOUSEBUTTONDOWN:
                x, y = event.pos
                self.handle_input(x, y)

    def run(self):
        while not self.ended:
            self.event_handler()
            self.render()
            pg.display.flip()
            clock.tick(30)
        else:
            self.fireworks()


if __name__ == '__main__':
    os.environ['SDL_VIDEO_CENTERED'] = '1'

    pg.init()
    clock = pg.time.Clock()
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), HWSURFACE | DOUBLEBUF)
    font = pg.font.SysFont('consolas', 20)
    pg.display.set_caption("Ultimate Tic-Tac-Toe")

    full_board = {i: np.full((3, 3), '.', dtype=str) for i in range(1, 10)}

    game = App()
    game.run()

    sys.exit()










