import numpy as np 
import math 
class Cat():
    time2grow = 2
    states = ["kitten", "adult"]
    personality = ["aggressive", "friendly", "lazy"]
    sex = ["male", "female", "undermined"]
    
    #we assume cats have spherical shape and we assign them a radius
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.state = self.states[0]
        self.age = 0
        self.radius = 0.5
        self.personality = "friendly"
        self.sex = "undetermined"
        
    def __str__(self):
        return self.personality + " " + self.sex + " " + self.state + " of radius " + str(self.radius) + " @(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"
   
    def stepChange(self):
        self.age += 1
        if self.state == "kitten":
            self.x += 1
            self.y += 1
            self.y += 1
            if self.age >= self.time2grow:
                self.state = "adult"
                self.radius = 1.5
        else:
            self.x += 2
            self.y += 2
            self.z += 2  
                            
    def getSize(c):
        if c.state == "kitten":
            radius = 0.5
        else:
            radius = 1.5
        return radius
       
       
    #we compute the diameter of the circle they project:
    def computeDiameter(self):
        return 2 * np.pi * self.radius
    
    # function to calculate distance btw center and given point
    def check(cx, cy, cz, x, y, z):
        x1 = math.pow((x-cx), 2)
        y1 = math.pow((y-cy), 2)
        z1 = math.pow((z-cz), 2)
        return (x1 + y1 + z1) # distance between the centre and given point
     
    
  
    
   #returns the possible Moore neighborhood positions
    def mooreNeighborhood(self):
        moore_neighborhood = []
        moore_neighborhood.append((0,-1,-1))
        moore_neighborhood.append((0,-1,0))
        moore_neighborhood.append((0,-1,1))
        moore_neighborhood.append((0,0,-1))
        moore_neighborhood.append((0,0,0))
        moore_neighborhood.append((0,0,1))
        moore_neighborhood.append((0,1,-1))
        moore_neighborhood.append((0,1,0))
        moore_neighborhood.append((0,1,1))
        
        moore_neighborhood.append((-1,0,-1))
        moore_neighborhood.append((-1,0,0))
        moore_neighborhood.append((-1,0,1))
        moore_neighborhood.append((0,0,-1))
        moore_neighborhood.append((0,0,+1))
        moore_neighborhood.append((1,0,-1))
        moore_neighborhood.append((1,0,0))
        moore_neighborhood.append((1,0,1))
        
        moore_neighborhood.append((-1,-1,0))
        moore_neighborhood.append((-1,0,0))
        moore_neighborhood.append((-1,1,0))
        moore_neighborhood.append((0,-1,0))
        moore_neighborhood.append((0,0,0))
        moore_neighborhood.append((0,1,0))
        moore_neighborhood.append((1,-1,0))
        moore_neighborhood.append((1,0,0))
        moore_neighborhood.append((1,1,0))
        
        return moore_neighborhood
        
    def vanNeumannNeighborhood(self):
        van_neumann_neighborhood = []
        van_neumann_neighborhood.append((1,0,0))
        van_neumann_neighborhood.append((-1,0,0))  
        van_neumann_neighborhood.append((0,1,0))
        van_neumann_neighborhood.append((0,-1,0))                
        van_neumann_neighborhood.append((0,0,1))
        van_neumann_neighborhood.append((0,0,1))        
        
        return van_neumann_neighborhood
        
        
        
        
        
        
        
        
        
        
        
        
        
