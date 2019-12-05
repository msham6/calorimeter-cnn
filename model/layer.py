import random
import numpy as np
from scipy.stats import multivariate_normal

class Layer:
    '''Defines an individual layer of a calorimeter. The properties of the layer are
    name, its material given as X0 per cm, the thickness, the response measuring the
    level of ionisation (in arbitrary units, zero for passive layer). The layer can
    keep track of the ionisation in it.'''

    def __init__(self, name, material, thickness, height, numcells, response=1.0):
        self._name = name
        self._material = material
        self._thickness = thickness
        self._yield = response
        self._ionisation = 0
        self._height = height
        self._width = height
        self._numcells = numcells
        self._cellsize = height/numcells
        self._cells = np.zeros((numcells, numcells))
        self._response = response
        self._missed = 0.0

    def ionise(self, particle, step):
        '''Records the ionisation in each layer from a particle going a certain length.'''
        if particle.ionise:
            # Treating it as a dot
            count = self._yield*step
            self._ionisation += count

            # Treating it as a line, total ionisation in all cells should equal
            # previous value
            if self._response > 0:

                cellsize = self._cellsize
                midcell = int(np.floor(self._numcells/2))
                # print('Midcell = ', midcell)

                y = particle.y
                y_ = y/cellsize
                if (y_ <= 0.0 or y_ >= 1.0) and abs(y_) <= midcell:
                    ycell = int(np.floor(y_ + midcell))
                elif (y_ >= 0.0) and y_ <= 1.0:
                    ycell = midcell
                else:
                    ycell = self._numcells + 1

                x = particle.x
                x_ = x/cellsize
                if (x_ <= 0.0 or x_ >= 1.0) and abs(x_) <= midcell:
                    xcell = int(np.floor(x_ + midcell))
                elif (x_ >= 0.0) and x_ <= 1.0:
                    xcell = midcell
                else:
                    xcell = self._numcells + 1

                if abs(xcell) < self._numcells and abs(ycell) < self._numcells:
                    self._cells[ycell, xcell] += count
                else:
                    self._missed += count

    def interact(self, particle, std, step):
        '''Let a particle interact (bremsstrahlung or pair production). The interaction
        length is assumed to be the same for electrons and photons.'''
        material = self._material*step
        particles = [particle]
        r = random.random()

        if r < material:
            particles = particle.interact(std)

        return particles

    def __str__(self):
        return f'{self._name:10} {self._material:.3f} {self._thickness:.2f} {self._height:.2f} cm {self._ionisation:.3f}'
