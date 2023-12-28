#!python
""" pg.examples.stars

    We are all in the gutter,
    but some of us are looking at the stars.
                                            -- Oscar Wilde

A simple starfield example. Note you can move the 'center' of
the starfield by leftclicking in the window. This example show
the basics of creating a window, simple pixel plotting, and input
event management.
"""
import random
import math
import pygame as pg
import numpy as np
import copy
import sys

G           = 6.67e-11
Mb          = 4.0e30                    # black hole
Ms          = 2.0e30                    # sun
Me          = 5.972e24                  # earth        
Mm          = 6.39e23                   # mars
Mc          = 6.39e16                  	# unknown comet
Mmoon		= 7.348e22					# moon
AU          = 1.5e11
daysec      = 24.0*60*60

e_ap_v      = 29290                     # earth velocity at aphelion
m_ap_v      = 21970                     # mars velocity at aphelion
commet_v    = 2000
moon_v		= e_ap_v+1020.0

mass_center = np.array([0.0,0.0,0.0])
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
        self.f = np.array([0.0,0.0,0.0])
        self.M = M
        self.queue = []
        self.color = color

    @property    
    def M(self):
        return self._M
        
    @M.setter
    def M(self, val):
        self._M = val
        self.r = math.pow((3 * val) / (4 * math.pi), 1/3) # I still don't consider the density of the planet


GRAY = (127, 127, 127) 
WHITE = (255, 255, 255)
RED = (255, 0, 0) 
GREEN = (0, 255, 0) 
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0) 
CYAN = (0, 255, 255) 
MAGENTA = (255, 0, 255)


def distance(p1, p2):
    return p1.pos - p2.pos

def force_p1(p1, p2):
    rx,ry,rz = distance(p1, p2)
    modr3 = (rx**2+ry**2+rz**2)**1.5
    gravconst = G*p1.M*p2.M
    fx = -gravconst*rx/modr3
    fy = -gravconst*ry/modr3
    fz = -gravconst*rz/modr3
    return np.array([fx, fy, fz])


# worldSize = coordinates(1*AU, 1*AU)
worldSize = coordinates(10*4.065e8, 10*4.065e8)
screenSize = coordinates(800, 800)

# constants
WINSIZE = [screenSize.x, screenSize.y]

def w2p(x, y):
    global mass_center
    correction = 1

    # centerx = mass_center[X]
    # centery = mass_center[Y]
    centerx = pe.pos[X]
    centery = pe.pos[Y]

    """ Convert world coordinates to screen (pixel) coordinates"""
    return (int(0.5+(x+worldSize.x/2-centerx*correction) / worldSize.x * screenSize.x),
            int(0.5+screenSize.y - (y+worldSize.y/2-centery*correction) / worldSize.y * screenSize.y))

white = 255, 240, 200
black = 20, 20, 40

def days(g):
    return 24.0*60*60*g


def draw_planet(surface: pg.surface.Surface, p: planet):
    global mass_center, maxM, minM
    factor = int((p.M-minM)/(maxM-minM)*255)
    red = factor
    green = 255-factor
    blue = factor
    color = (red, green, blue)
    pg.draw.rect(surface, WHITE, pg.Rect(w2p(mass_center[X], mass_center[Y]), (5, 5)))
    pg.draw.circle(surface, p.color, w2p(p.pos[X], p.pos[Y]), math.log(p.r)/5)
    points = []
    for pos in p.queue:
        points.append(w2p(pos[X], pos[Y]))
    if len(points)>1:
        pg.draw.lines(surface, p.color, False, points)


def main():
    global mass_center, maxM, minM, planets, pe

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

    planets = np.array([ps, pmoon, pe, pm, pc])

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
        # compute G forces
        for p1 in planets:
            p1.f = np.array([0.0,0.0,0.0])
            for p2 in planets:
                if p1 != p2:
                    p1.f += force_p1(p1, p2)

        for pl in planets:
            # update quantities how is this calculated?  F = ma -> a = F/m
            pl.vel += pl.f*dt/pl.M

            pl.queue.append(copy.copy(pl.pos))
            if len(pl.queue) > 500:
                pl.queue.pop(0)
            # update position
            pl.pos += pl.vel*dt
        
        

        i = 0
        toberemoved = []
        for p1 in planets:
            j = 0
            for p2 in planets:
                if p1 != p2:
                    d = sum(np.abs(distance(p1, p2)))
                    if d < (p1.r+p2.r): # collision
                        if p1.M > p2.M:
                            if j not in toberemoved:
                                p1.M += p2.M
                                toberemoved.append(j)
                        else:
                            if i not in toberemoved:
                                p2.M += p1.M
                                toberemoved.append(i)
                j += 1
            i += 1
        
        j = 0
        for i in toberemoved:
            planets = np.delete(planets, i-j)
            j += 1

        num = np.array([0.0,0.0,0.0])
        den = 0.0
        minx = 0.0
        maxx = 0.0
        miny = 0.0
        maxy = 0.0

        mean = 0.0
        
        for pl in planets:
            num += pl.pos*pl.M
            den += pl.M
            if pl.M > maxM:
                maxM = pl.M
            if pl.M < minM:
                minM = pl.M
            if pl.pos[X] > maxx:
                maxx = pl.pos[X]
            if pl.pos[X] < minx:
                minx = pl.pos[X]
            if pl.pos[Y] > maxy:
                maxy = pl.pos[Y]
            if pl.pos[Y] < miny:
                miny = pl.pos[Y]
            mean += np.abs(pl.pos)

        mean /= len(planets) # to be used for view auto-centering	
        mass_center = num/den

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

        if 2*maxx > worldSize.x:
            worldSize.x = 2*maxx
        if 2*minx < -worldSize.x and minx < 0.0:
            worldSize.x = -2*minx
        if 2*maxy > worldSize.y:
            worldSize.y = 2*maxy
        if 2*miny < -worldSize.y and miny < 0.0:
            worldSize.y = -2*miny	

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