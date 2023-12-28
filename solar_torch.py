
#!python3

import random
import math
import copy
import sys
import pygame as pg
import numpy as np
import torch
from skyfield.api import load
from astroquery.jplhorizons import Horizons
from astropy import units as u
import re

object_id = "399" # Mercury
obj_id = Horizons(id=object_id, location="0") # Location 0 is the solar system barycenter

response = obj_id.ephemerides_async()

def extract_data(text):
    # Regex migliorata per gestire formati diversi di chiavi e valori
    pattern = r"([A-Za-z\s.]+)(?:\[(?:[^\]]+)\])?\s*=\s*([^\n]+)"
    matches = re.findall(pattern, text)
    
    # Creazione del dizionario migliorato
    data_dict = {}
    for key, value in matches:
        # Pulizia e normalizzazione della chiave
        key = key.strip().replace(' ', '_').replace('.', '').lower()
        # Pulizia del valore e gestione di valori multipli
        value = value.strip().split(' ')
        # Se il valore è unico, lo inserisce direttamente, altrimenti crea una lista
        data_dict[key] = value[0] if len(value) == 1 else value

    return data_dict

print(extract_data(response.text))


planets = load('de421.bsp')
earth, mars = planets['earth'], planets['mars']
sun = planets['sun']
ts = load.timescale()
t = ts.now()
position = sun.at(t).observe(earth)
ra, dec, distance = position.radec()

dtype = torch.float
device = torch.device("cuda")

G = 6.67e-11

Mb = 4.0e30                    # black hole
Ms = 1.9891e30                    # sun
Me = 5.972e24                  # earth        
Mm = 6.39e23                   # mars
Mc = 6.39e16                  	# unknown comet
Mmoon = 7.348e22					# moon
AU = 149597828677
daysec = 24.0*60*60

e_ap_v = 29290                     # earth velocity at aphelion
m_ap_v = 21970                     # mars velocity at aphelion
commet_v = 2000
moon_v = e_ap_v+1020.0

mass_center = []

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
            density = 1408 # sole kg/m^2
            self.r = math.pow((3 * val) / (4 * math.pi * density), 1/3) # I still don't consider the density of the planet
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

def update_position(position: torch.Tensor, mass: torch.Tensor, velocity: torch.Tensor, G, dt):
    mass_scale = 1e-15
    mass_rescaled = mass * mass_scale

    delta_pos = position.unsqueeze(1) - position
    distance = torch.norm(delta_pos, dim=2)
    
    F = -G * torch.outer(mass_rescaled, mass_rescaled) / distance**3
    F[distance == 0] = 0  # Avoid division by zero

    force = torch.sum(F.unsqueeze(2) * delta_pos, dim=1) / mass_scale**2
    mass_reshaped = mass.unsqueeze(1).expand_as(force)
    velocity += force * dt / mass_reshaped
    velocity[mass_reshaped == 0] = 0
    position += velocity * dt

    mass_center = torch.sum(position * mass_rescaled.unsqueeze(1), dim=0)/torch.sum(mass_rescaled)
    position_max = torch.max(position, dim=0)
    position_min = torch.min(position, dim=0)

    density = 1408 # Sun density
    radius = torch.pow((mass * 3) / (4 * math.pi * density), 1/3)
    distance_collision = radius.unsqueeze(1)+radius
    filtered_mass = torch.where(mass == 0, torch.tensor(float('inf')).to(device), torch.tensor(1.0).to(device))
    distance_filtered = distance * filtered_mass
    collision = (distance_filtered-distance_collision < 0).nonzero()
    for col in collision:
        if col[0]<col[1]:
            if mass[col[0]]>mass[col[1]]:
                mass[col[0]] += mass[col[1]]
                mass[col[1]] = 0
            else:
                mass[col[1]] += mass[col[0]]
                mass[col[0]] = 0
    to_be_drawn = mass.nonzero()
    return position.tolist(), position_min.values.tolist(), position_max.values.tolist(), mass_center.tolist(), to_be_drawn.tolist()


# worldSize = coordinates(1*AU, 1*AU)
worldSize = coordinates(10*4.065e8, 10*4.065e8)
screenSize = coordinates(800, 800)
zoom = 1.0

# constants
WINSIZE = [screenSize.x, screenSize.y]

def w2p(x, y):
    global mass_center, zoom
    correction = 1

    centerx = mass_center[X]
    centery = mass_center[Y]
    # centerx = pe.pos[X]
    # centery = pe.pos[Y]
    zws = coordinates(worldSize.x*zoom, worldSize.y*zoom)

    """ Convert world coordinates to screen (pixel) coordinates"""
    return (int(0.5+(x+zws.x/2-centerx*correction) / zws.x * screenSize.x),
            int(0.5+screenSize.y - (y+zws.y/2-centery*correction) / zws.y * screenSize.y))

white = 255, 240, 200
black = 20, 20, 40

def days(g):
    return 24.0*60*60*g


def draw_planet(surface: pg.surface.Surface, p: planet):
    global mass_center
    if p.M != 0:
        pg.draw.rect(surface, WHITE, pg.Rect(w2p(mass_center[X], mass_center[Y]), (5, 5)))
        pg.draw.circle(surface, p.color, w2p(p.pos[X], p.pos[Y]), math.log(p.r)/5)
        points = []
        for pos in p.queue:
            points.append(w2p(pos[X], pos[Y]))
        if len(points)>1:
            pg.draw.lines(surface, p.color, False, points)


def main():
    global mass_center, planets, pe, zoom

    pe = planet(pos=np.array([1.0167*AU, 0.0, 0.0]), vel=np.array([0.0, e_ap_v, 0.0]), M=Me, color=GREEN)
    pmoon = planet(pos=np.array([1.0167*AU-3.844e8, 0.0, 0.0]), vel=np.array([0.0, moon_v, 0.0]), M=Mmoon, color=WHITE)
    pm = planet(pos=np.array([1.666*AU, 0.0, 0.0]), vel=np.array([0.0, m_ap_v, 0.0]), M=Mm, color=RED)
    pc = planet(pos=np.array([6*AU, 0.3*AU, 0.0]), vel=np.array([0.0, commet_v, 0.0]), M=Mc, color=WHITE)
    ps = planet(pos=np.array([0.0, 0.0, 0.0]), vel=np.array([0.0, 0.0, 0.0]), M=Ms, color=YELLOW)
    ps2 = copy.deepcopy(ps)
    ps2.pos[Y] = 1.0167*AU*2
    ps2.vel[X] = commet_v*40
    ps3 = copy.deepcopy(ps)
    ps3.pos[Y] = -1.0167*AU*2
    ps3.vel[X] = -commet_v*40
    ps4 = copy.deepcopy(ps)
    ps4.pos[X] = -1.0167*AU*2
    ps4.vel[Y] = -commet_v*20
    ps5 = copy.deepcopy(ps)
    ps5.pos[X] = 1.0167*AU*2
    ps5.vel[Y] = commet_v*20
    pe2 = copy.deepcopy(pe)
    pe2.pos[Y] = -1.0167*AU
    pe2.vel[X] = -commet_v
    pe3 = copy.deepcopy(pe)
    pe3.pos[Y] = 1.0167*AU
    pe3.vel[X] = commet_v*7

    # planets = [ps, pmoon, pe, pm, pc]

    ps.M *= 10
    planets = [ps, ps2, ps3, ps4, ps5]

    for i in range(1000):
        speed = commet_v*20
        p = copy.deepcopy(ps)
        p.pos[X] = random.uniform(-1.0167*AU*2, 1.0167*AU*2)
        p.pos[Y] = random.uniform(-1.0167*AU*2, 1.0167*AU*2)
        p.vel[X] = random.uniform(-speed, speed)
        p.vel[Y] = random.uniform(-speed, speed)
        p.M = random.uniform(Me/10, Me)
        p.color=GREEN
        planets = np.append(planets, p)

    # for i in range(2):
    #     p = copy.deepcopy(ps)
    #     p.pos[X] = random.uniform(-1.0167*AU*2, 1.0167*AU*2)
    #     p.pos[Y] = random.uniform(-1.0167*AU*2, 1.0167*AU*2)
    #     p.vel[X] = random.uniform(-commet_v*10, commet_v*10)
    #     p.vel[Y] = random.uniform(-commet_v*10, commet_v*10)
    #     p.M = Ms #random.uniform(Ms/50, Ms/5)
    #     planets = np.append(planets, p)

    v_pos = []
    v_vel = []
    v_M = []
    for p in planets:
        v_pos.append(p.pos)
        v_vel.append(p.vel)
        v_M.append(p.M)

    t_position = torch.tensor(v_pos, dtype=torch.float32).to(device)
    t_velocity = torch.tensor(v_vel, dtype=torch.float32).to(device)
    t_mass = torch.tensor(v_M, dtype=torch.float32).to(device)

    t = 0.0
    # dt = 1*daysec # every frame move this time
    dt = days(0.02)
    # dt = days(0.1)

    clock = pg.time.Clock()
    # initialize and prepare screen
    pg.init()
    font = pg.font.SysFont(None, 24)
    screen = pg.display.set_mode(WINSIZE)
    pg.display.set_caption("Solar system")

    # main game loop
    done = 0
    clock_tick = 300
    first = False
    while not done:

        position, position_min, position_max, mass_center, to_be_drawn = update_position(t_position, t_mass, t_velocity, G, dt)
        minx = position_min[X]
        miny = position_min[Y]
        maxx = position_max[X]
        maxy = position_max[Y]

        i = 0
        for pl in planets:
            pl.pos[X] = position[i][X]
            pl.pos[Y] = position[i][Y]
            pl.M = t_mass[i].tolist()
            pl.queue.append(copy.copy(pl.pos))
            if len(pl.queue) > 1000:
                pl.queue.pop(0)
            i += 1

        if not first:
            if 2*maxx > worldSize.x:
                worldSize.x = worldSize.y = 2*maxx
            if 2*minx < -worldSize.x and minx < 0.0:
                worldSize.x = worldSize.y = -2*minx
            if 2*maxy > worldSize.y:
                worldSize.x = worldSize.y = 2*maxy
            if 2*miny < -worldSize.y and miny < 0.0:
                worldSize.x = worldSize.y = -2*miny	
            first = True

        t += dt

        screen.fill(black)

        for i in to_be_drawn:
            draw_planet(screen, planets[i[0]])
        
        img = font.render(f't={t/(24*60*60)} days', True, WHITE)
        screen.blit(img, (20, 20))

        pg.display.update()
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYUP and e.key == pg.K_ESCAPE):
                done = 1
            if (e.type == pg.KEYUP and e.key == pg.K_w):
                zoom *= 0.9
            if (e.type == pg.KEYUP and e.key == pg.K_s):
                zoom /= 0.9
                break
        
        clock.tick(clock_tick)
    pg.quit()


if __name__ == "__main__":
    main()