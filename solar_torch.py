
#!python3

import random
import math
import copy
import sys
import pygame as pg
import numpy as np
import torch

dtype = torch.float
device = torch.device("mps")

G = 6.67e-11

Mb = 4.0e30                    # black hole
Ms = 2.0e30                    # sun
Me = 5.972e24                  # earth        
Mm = 6.39e23                   # mars
Mc = 6.39e16                  	# unknown comet
Mmoon = 7.348e22					# moon
AU = 1.5e11
daysec = 24.0*60*60

e_ap_v = 29290                     # earth velocity at aphelion
m_ap_v = 21970                     # mars velocity at aphelion
commet_v = 2000
moon_v = e_ap_v+1020.0

v_centro_massa = []
maxM = 0.0
minM = sys.float_info.max

X = 0
Y = 1
Z = 2

class coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class planet:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, M, color):
        self.pos = pos
        self.vel = vel
        self.f = torch.tensor([0.0,0.0,0.0]).to(device)
        self.M = M
        self.queue = []
        self.color = color

    @property    
    def M(self):
        return self._M
        
    @M.setter
    def M(self, val):
        if val != 0:
            self._M = val
            self.r = math.pow((3 * val) / (4 * math.pi), 1/3) # I still don't consider the density of the planet
        else:
            self._M = 0
            self.r = 0


GRAY = (127, 127, 127) 
WHITE = (255, 255, 255)
RED = (255, 0, 0) 
GREEN = (0, 255, 0) 
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0) 
CYAN = (0, 255, 255) 
MAGENTA = (255, 0, 255)

def aggiorna_posizioni(posizioni: torch.Tensor, masse: torch.Tensor, velocita: torch.Tensor, G, dt):
    scala_masse = 1e-15
    masse_scalate = masse * scala_masse

    # Creazione di una matrice di distanze utilizzando broadcasting
    delta_pos = posizioni.unsqueeze(1) - posizioni
    distanze = torch.norm(delta_pos, dim=2)
    
    # Calcolo delle componenti delle forze
    # Uso di torch.outer per il prodotto scalato delle masse
    F = -G * torch.outer(masse_scalate, masse_scalate) / distanze**3
    F[distanze == 0] = 0  # Evita divisione per zero

    # Calcolo della forza e descala il risultato
    forze = torch.sum(F.unsqueeze(2) * delta_pos, dim=1) / scala_masse**2
    masse_reshaped = masse.unsqueeze(1).expand_as(forze)
    velocita += forze * dt / masse_reshaped
    velocita[masse_reshaped == 0] = 0
    posizioni += velocita * dt

    centro_massa = torch.sum(posizioni * masse_scalate.unsqueeze(1), dim=0)/torch.sum(masse_scalate)
    posizioni_max = torch.max(posizioni, dim=0)
    posizioni_min = torch.min(posizioni, dim=0)
    masse_max = torch.max(masse, dim=0)
    masse_min = torch.min(masse, dim=0)

    raggi = torch.pow((masse * 3) / (4 * math.pi), 1/3)
    distanze_min = torch.min(distanze.fill_diagonal_(float('inf')))
    raggi_max = torch.max(raggi, dim=0)


    return posizioni, velocita, distanze, centro_massa, posizioni_min, posizioni_max, masse_max, masse_min, raggi, distanze_min, raggi_max


# worldSize = coordinates(1*AU, 1*AU)
worldSize = coordinates(10*4.065e8, 10*4.065e8)
screenSize = coordinates(800, 800)

# constants
WINSIZE = [screenSize.x, screenSize.y]

def w2p(x, y):
    global v_centro_massa
    correction = 1

    centerx = v_centro_massa[X]
    centery = v_centro_massa[Y]
    # centerx = pe.pos[X]
    # centery = pe.pos[Y]

    """ Convert world coordinates to screen (pixel) coordinates"""
    return (int(0.5+(x+worldSize.x/2-centerx*correction) / worldSize.x * screenSize.x),
            int(0.5+screenSize.y - (y+worldSize.y/2-centery*correction) / worldSize.y * screenSize.y))

white = 255, 240, 200
black = 20, 20, 40

def days(g):
    return 24.0*60*60*g


def draw_planet(surface: pg.surface.Surface, p: planet):
    global v_centro_massa, maxM, minM
    if p.M != 0:
        factor = int((p.M-minM)/(maxM-minM)*255)
        red = factor
        green = 255-factor
        blue = factor
        color = (red, green, blue)
        pg.draw.rect(surface, WHITE, pg.Rect(w2p(v_centro_massa[X], v_centro_massa[Y]), (5, 5)))
        pg.draw.circle(surface, p.color, w2p(p.pos[X], p.pos[Y]), math.log(p.r)/5)
        points = []
        for pos in p.queue:
            points.append(w2p(pos[X], pos[Y]))
        if len(points)>1:
            pg.draw.lines(surface, p.color, False, points)


def main():
    global v_centro_massa, maxM, minM, planets, pe

    pe = planet(pos=np.array([1.0167*AU, 0.0, 0.0]), vel=np.array([0.0, e_ap_v, 0.0]), M=Me, color=GREEN)
    pmoon = planet(pos=np.array([1.0167*AU-3.844e8, 0.0, 0.0]), vel=np.array([0.0, moon_v, 0.0]), M=Mmoon, color=WHITE)
    pm = planet(pos=np.array([1.666*AU, 0.0, 0.0]), vel=np.array([0.0, m_ap_v, 0.0]), M=Mm, color=RED)
    pc = planet(pos=np.array([6*AU, 0.3*AU, 0.0]), vel=np.array([0.0, commet_v, 0.0]), M=Mc, color=WHITE)
    ps = planet(pos=np.array([0.0, 0.0, 0.0]), vel=np.array([0.0, 0.0, 0.0]), M=Ms, color=YELLOW)
    ps2 = copy.deepcopy(ps)
    ps2.pos[Y] = 1.0167*AU*7
    ps2.vel[X] = commet_v*6
    ps3 = copy.deepcopy(ps)
    ps3.pos[Y] = -1.0167*AU*4
    ps3.vel[X] = -commet_v*10
    ps4 = copy.deepcopy(ps)
    ps4.pos[X] = -1.0167*AU*4
    ps4.vel[Y] = -commet_v*5
    ps5 = copy.deepcopy(ps)
    ps5.pos[X] = 1.0167*AU*4
    ps5.vel[Y] = commet_v*2
    pe2 = copy.deepcopy(pe)
    pe2.pos[Y] = -1.0167*AU
    pe2.vel[X] = -commet_v
    pe3 = copy.deepcopy(pe)
    pe3.pos[Y] = 1.0167*AU
    pe3.vel[X] = commet_v*7

    planets = [ps, pmoon, pe, pm, pc]

    for i in range(100):
        speed = commet_v*20
        p = copy.deepcopy(ps)
        p.pos[X] = random.uniform(-1.0167*AU*8, 1.0167*AU*8)
        p.pos[Y] = random.uniform(-1.0167*AU*8, 1.0167*AU*8)
        p.vel[X] = random.uniform(-speed, speed)
        p.vel[Y] = random.uniform(-speed, speed)
        p.M = random.uniform(Me/10, Me)
        p.color=GREEN
        planets = np.append(planets, p)

    for i in range(100):
        p = copy.deepcopy(ps)
        p.pos[X] = random.uniform(-1.0167*AU*8, 1.0167*AU*8)
        p.pos[Y] = random.uniform(-1.0167*AU*8, 1.0167*AU*8)
        p.vel[X] = random.uniform(-commet_v*10, commet_v*10)
        p.vel[Y] = random.uniform(-commet_v*10, commet_v*10)
        p.M = random.uniform(Ms/50, Ms/5)
        planets = np.append(planets, p)

    v_pos = []
    v_vel = []
    v_M = []
    for p in planets:
        v_pos.append(p.pos)
        v_vel.append(p.vel)
        v_M.append(p.M)

    posizioni = torch.tensor(v_pos, dtype=torch.float32).to(device)
    velocita = torch.tensor(v_vel, dtype=torch.float32).to(device)
    masse = torch.tensor(v_M, dtype=torch.float32).to(device)

    t = 0.0
    # dt = 1*daysec # every frame move this time
    dt = days(0.1)

    clock = pg.time.Clock()
    # initialize and prepare screen
    pg.init()
    font = pg.font.SysFont(None, 24)
    screen = pg.display.set_mode(WINSIZE)
    pg.display.set_caption("Solar system")

    # main game loop
    done = 0
    clock_tick = 300
    while not done:

        posizioni, velocita, distanze, centro_massa, posizioni_min, posizioni_max, masse_max, masse_min, raggi, distanze_min, raggi_max = aggiorna_posizioni(posizioni, masse, velocita, G, dt)
        v_dis = distanze.tolist()
        v_posizioni = posizioni.tolist()
        v_centro_massa = centro_massa.tolist()
        maxM = masse_max.values.item()
        minM = masse_min.values.item()
        v_posizioni_min = posizioni_min.values.tolist()
        v_posizioni_max = posizioni_max.values.tolist()
        minx = v_posizioni_min[X]
        miny = v_posizioni_min[Y]
        maxx = v_posizioni_max[X]
        maxy = v_posizioni_max[Y]
        v_raggi = raggi.tolist()

        i = 0
        for pl in planets:
            pl.pos[X] = v_posizioni[i][X]
            pl.pos[Y] = v_posizioni[i][Y]
            pl.queue.append(copy.copy(pl.pos))
            if len(pl.queue) > 500:
                pl.queue.pop(0)
            i += 1
            
        if distanze_min.item() < 2*raggi_max.values.item():
            print('test for collision')
            collision = False
            i = 0
            for p1 in planets:
                j = 0
                for p2 in planets:
                    if p1 != p2:
                        d = v_dis[i][j]
                        if d < (p1.r+p2.r): # collision
                            if p1.M > p2.M:
                                if p2.M != 0:
                                    p1.M += p2.M
                                    p2.M = 0
                                    collision = True
                            else:
                                if p1.M != 0:
                                    p2.M += p1.M
                                    p1.M = 0
                                    collision = True
                    j += 1
                i += 1
            if collision:
                v_M = []
                for p in planets:
                    v_M.append(p.M)
                masse = torch.tensor(v_M, dtype=torch.float32).to(device)
                print('collision')
    
        # if t > days(28) and t < days(80):
        #     worldSize.x *= 1.01
        #     worldSize.y *= 1.01
        
        # if t > days(28):
        #     clock_tick = 60
        # if t > days(100):
        #     clock_tick = 300

        # if mean[X] > worldSize.x:
        # 	worldSize.x = mean[X]
        # if mean[Y] > worldSize.y:
        # 	worldSize.y = mean[Y]

        if maxx > worldSize.x:
        	worldSize.x = maxx
        if minx < -worldSize.x and minx < 0.0:
        	worldSize.x = -minx
        if maxy > worldSize.y:
        	worldSize.y = maxy
        if miny < -worldSize.y and miny < 0.0:
        	worldSize.y = -miny	

        t += dt

        screen.fill(black)

        for pl in planets:
            draw_planet(screen, pl)
        
        img = font.render(f't={t/(24*60*60)} days', True, WHITE)
        screen.blit(img, (20, 20))

        pg.display.update()
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYUP and e.key == pg.K_ESCAPE):
                done = 1
                break
        
        clock.tick(clock_tick)
    pg.quit()


# if python says run, then we should run
if __name__ == "__main__":
    main()

    # I prefer the time of insects to the time of stars.
    #
    #                              -- Wisława Szymborska