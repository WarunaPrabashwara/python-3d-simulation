from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import matplotlib.tri as mtri
from scipy.interpolate import griddata
import random
from matplotlib.patches import Circle, Ellipse
from matplotlib import style
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from cat import Cat
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches



def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt2(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data2(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), **kwargs)


def build_terrain(x, y, z):	
	x1 = np.linspace(x.min(), x.max(), len(np.unique(x))); 
	y1 = np.linspace(y.min(), y.max(), len(np.unique(y)));
	x2, y2 = np.meshgrid(x1, y1);

	z2 = griddata( (x, y), z, (x2, y2), method='cubic', fill_value = 1);
	z2[z2 < z.min()] = z.min();


	return x2, y2, z2

#plots terrain and obstacles
def plot_terrain(x, y, z, x2, y2, z2):
	p1 = range(np.min(x), np.max(x))
	p2 = range(np.min(y), np.max(y))
	p3 = range(np.min(z), np.max(z))

	#add boundaries to the cubic grid
	fig = plt.figure(); 
	ax = fig.add_subplot(111, projection='3d')
	#creating meshgrids to fill the faces of the cubic grid
	X, Y = np.meshgrid(p1, p2)
	ax.plot_surface(X, Y, min(z) + np.zeros_like(X), alpha=0.2, color = 'c')

	X, Z = np.meshgrid(p1, p3)
	ax.plot_surface(X, np.zeros_like(Z), Z, alpha=0.2, color = 'c')
	ax.plot_surface(X, max(y) + np.zeros_like(Z), Z, alpha=0.2, color = 'c')

	Y, Z = np.meshgrid(p2, p3)
	ax.plot_surface(np.zeros_like(Y), Y,  Z, alpha=0.2, color = 'c')
	ax.plot_surface(max(x) + np.zeros_like(Y), Y,  Z, alpha=0.2, color = 'c')
	X, Y = np.meshgrid(p1, p2)


	ax.plot_surface(x2, y2, z2, linewidth=0, rstride=1, cstride=1,
	antialiased=False, alpha = 0.5, cmap=plt.cm.viridis)
	
	positions = [(4,5,106), (1,10,115) , (8,2,110), (40,7,115) , (30,3,169), (4,50,112) ]
	sizes = [(2,1,1), (3,1,1) , (2,1,1), (3,1,1) , (1,1,1), (3,1,1)]
	colors = ["black", "black" , "black", "blue" , "blue", "blue"]
	#ax.view_init(45, 45)

	pc = plotCubeAt2(positions,sizes,colors=colors, edgecolor="k")	
	ax.add_collection3d(pc)    

	ax.set_xlabel('X') 
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	return ax

#computes the polar coordinates of the circle
def compute_polar_coordinates():
	u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
	a = np.cos(u)*np.sin(v)
	b = np.sin(u)*np.sin(v)
	c = np.cos(v)
	return a, b, c

def get_random_neighborhoods(neighborhoods, x, y, z, i, j, k):
	valid_points = []
	for points in neighborhoods:
		if np.min(x) < i + points[0] and i + points[0] < np.max(x):
			if np.min(y) < j + points[1] and j + points[1] < np.max(y):
				if np.min(z) < k + points[2] and k + points[2] < np.max(z):
					valid_points.append((i + points[0], j + points[1], k + points[2]))
	return valid_points

def is_valid(a, b, c, x, y, z):
	grid_points = zip(x,y,z)
	#print(grid_points)
	if np.min(x) <= a and b <= np.max(x):
		if np.min(y) <= b and b <= np.max(y):
			if np.min(z) <= c and c <= np.max(z):
				if (a, b, c) in grid_points:
					return True
	return False    
def add_point(ax, x, y, z, fc = 'white', ec = None, radius = 0.005):
       xy_len, z_len = ax.get_figure().get_size_inches()
       axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
       axis_rotation =  {'z': ((x, y, z), axis_length[1]/axis_length[0]),
                         'y': ((x, z, y), axis_length[2]/axis_length[0]*xy_len/z_len),
                         'x': ((y, z, x), axis_length[2]/axis_length[1]*xy_len/z_len)}
       for a, ((x0, y0, z0), ratio) in axis_rotation.items():
           p = Ellipse((x0, y0), width = radius, height = radius*ratio, fc=fc, ec=ec)
           ax.add_patch(p)
           art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)
def add_point2(ax, x, y, z, fc = 'yellow', ec = None, radius = 0.005):
       xy_len, z_len = ax.get_figure().get_size_inches()
       axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
       axis_rotation =  {'z': ((x, y, z), axis_length[1]/axis_length[0]),
                         'y': ((x, z, y), axis_length[2]/axis_length[0]*xy_len/z_len),
                         'x': ((y, z, x), axis_length[2]/axis_length[1]*xy_len/z_len)}
       for a, ((x0, y0, z0), ratio) in axis_rotation.items():
           p = Ellipse((x0, y0), width = radius, height = radius*ratio, fc=fc, ec=ec)
           ax.add_patch(p)
           art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)
def add_point3(ax, x, y, z, fc = 'orange', ec = None, radius = 0.005):
       xy_len, z_len = ax.get_figure().get_size_inches()
       axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
       axis_rotation =  {'z': ((x, y, z), axis_length[1]/axis_length[0]),
                         'y': ((x, z, y), axis_length[2]/axis_length[0]*xy_len/z_len),
                         'x': ((y, z, x), axis_length[2]/axis_length[1]*xy_len/z_len)}
       for a, ((x0, y0, z0), ratio) in axis_rotation.items():
           p = Ellipse((x0, y0), width = radius, height = radius*ratio, fc=fc, ec=ec)
           ax.add_patch(p)
           art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)

def main():
	print("\n cat1 is white in color")
	print("\n cat1 is yellow in color")
	print("\n cat1 is orange in color")
	print("\n water is blue in color")
	print("\n food is black in color")
	print("\n ### 3D Growth Simulation ###\n")
    
    	# Initialise population

	data = pd.read_csv('simple_grid.csv', header=None)

	df=data.unstack().reset_index()
	df.columns=["X","Y","Z"]
	#print(df)
	x = df['X']
	y = df['Y']
	z = df['Z']


	popgrid = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
	popgrid2 = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
	popgrid3 = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
	#print(popgrid)
	nextpopgrid = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
	nextpopgrid2 = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
	nextpopgrid3 = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
	nextgrid = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
	
	#cat can occupy the position (x_coord, y_coord, z_coord)
	potential_cats = {Cat(x_coord, y_coord, z_coord) for (x_coord, y_coord, z_coord) in zip(x, y, z)}
	potential_cats2 = {Cat(x_coord, y_coord, z_coord) for (x_coord, y_coord, z_coord) in zip(x, y, z)}
	potential_cats3 = {Cat(x_coord, y_coord, z_coord) for (x_coord, y_coord, z_coord) in zip(x, y, z)}
	
	#sample 10 cats
	cats = random.sample(potential_cats, 3)
	cats2 = random.sample(potential_cats2, 3)
	cats3 = random.sample(potential_cats3, 3)		
	#print(cats)
	#ax.view_init(30,70)

	x2, y2, z2 = build_terrain(x, y, z)
	ax = plot_terrain(x, y, z, x2, y2, z2)
	for cat in cats:
		popgrid[cat.x-1, cat.y-1, cat.z-1] = 1
		add_point(ax, cat.x-1, cat.y-1, cat.z-1 , radius=cat.radius + 0.5)
	for cat in cats2:
		popgrid2[cat.x-1, cat.y-1, cat.z-1] = 1
		add_point2(ax, cat.x-1, cat.y-1, cat.z-1 , radius=cat.radius + 0.5)
	for cat in cats3:
		popgrid3[cat.x-1, cat.y-1, cat.z-1] = 1
		add_point3(ax, cat.x-1, cat.y-1, cat.z-1 , radius=cat.radius + 0.5)		
	red_patch = mpatches.Patch(color='white', label='Cat')
	yellow_patch = mpatches.Patch(color='yellow', label='Cat2')
	blue_patch = mpatches.Patch(color='orange', label='Cat3')
	black_patch = mpatches.Patch(color='black', label='Food')
	water_patch = mpatches.Patch(color='blue', label='Water')
	plt.legend(handles=[red_patch, black_patch , yellow_patch ,blue_patch  ,water_patch ])
	plt.show()
	STEPS =  int(input("Enter how many times do you want to circulate : "))
	for i in range(STEPS):
		print("\n ### TIMESTEP ",i, "###")
		nextpopgrid = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
		nextpopgrid2 = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
		nextpopgrid3 = np.zeros((np.max(x), np.max(y), np.max(z)), dtype=int)
		x2, y2, z2 = build_terrain(x, y, z)
		ax = plot_terrain(x, y, z, x2, y2, z2)
		
		for i1 in range(np.max(x)):
			for j1 in range(np.max(y)):
				for k1 in range(np.max(z)):
					if popgrid[i1-1][j1-1][k1-1] == 1:
						cat = Cat(i1-1, j1-1, k1-1)
						#print(i1, j1, k1)
						#take step using Moore neighborhood:
						moore_neighborhoods = cat.mooreNeighborhood()
						random_neighborhood = get_random_neighborhoods(moore_neighborhoods,x,y,z,i1,j1,k1)
						mycat = Cat(i1-1, j1-1, k1-1)
						is_valid_neighborhood = False
						#choose neighborhood that is on the surface of grid:
						for idx, random_neigh in enumerate(random_neighborhood):							
							mycat.x = random_neigh[0]
							mycat.y = random_neigh[1]
							mycat.z = random_neigh[2]
							if is_valid(mycat.x-1, mycat.y-1, mycat.z-1, x, y, z):
								#print(mycat.x-1, mycat.y-1, mycat.z-1)
								#ax.plot_wireframe(mycat.radius * a + mycat.x, mycat.radius * a + mycat.y, mycat.radius * c + mycat.z , color="crimson")
								is_valid_neighborhood = True
								break
						if is_valid_neighborhood:
								add_point(ax, mycat.x, mycat.y, mycat.z, radius=mycat.radius + 0.5)
								nextpopgrid[mycat.x, mycat.y, mycat.z] = 1
						if not is_valid_neighborhood:
							#if is_valid(cat.x, cat.y, cat.z, x, y, z):
							nextpopgrid[cat.x, cat.y, cat.z] = 1
							add_point(ax, cat.x, cat.y, cat.z, radius=cat.radius + 0.5)
					if popgrid2[i1-1][j1-1][k1-1] == 1:
						cat = Cat(i1-1, j1-1, k1-1)
						#print(i1, j1, k1)
						#take step using Moore neighborhood:
						moore_neighborhoods = cat.mooreNeighborhood()
						random_neighborhood = get_random_neighborhoods(moore_neighborhoods,x,y,z,i1,j1,k1)
						mycat = Cat(i1-1, j1-1, k1-1)
						is_valid_neighborhood = False
						#choose neighborhood that is on the surface of grid:
						for idx, random_neigh in enumerate(random_neighborhood):							
							mycat.x = random_neigh[0]
							mycat.y = random_neigh[1]
							mycat.z = random_neigh[2]
							if is_valid(mycat.x-1, mycat.y-1, mycat.z-1, x, y, z):
								#print(mycat.x-1, mycat.y-1, mycat.z-1)
								#ax.plot_wireframe(mycat.radius * a + mycat.x, mycat.radius * a + mycat.y, mycat.radius * c + mycat.z , color="crimson")
								is_valid_neighborhood = True
								break
						if is_valid_neighborhood:
								add_point2(ax, mycat.x, mycat.y, mycat.z, radius=mycat.radius + 0.5)
								nextpopgrid2[mycat.x, mycat.y, mycat.z] = 1
						if not is_valid_neighborhood:
							#if is_valid(cat.x, cat.y, cat.z, x, y, z):
							nextpopgrid2[cat.x, cat.y, cat.z] = 1
							add_point2(ax, cat.x, cat.y, cat.z, radius=cat.radius + 0.5)
					if popgrid3[i1-1][j1-1][k1-1] == 1:
						cat = Cat(i1-1, j1-1, k1-1)
						#print(i1, j1, k1)
						#take step using Moore neighborhood:
						moore_neighborhoods = cat.mooreNeighborhood()
						random_neighborhood = get_random_neighborhoods(moore_neighborhoods,x,y,z,i1,j1,k1)
						mycat = Cat(i1-1, j1-1, k1-1)
						is_valid_neighborhood = False
						#choose neighborhood that is on the surface of grid:
						for idx, random_neigh in enumerate(random_neighborhood):							
							mycat.x = random_neigh[0]
							mycat.y = random_neigh[1]
							mycat.z = random_neigh[2]
							if is_valid(mycat.x-1, mycat.y-1, mycat.z-1, x, y, z):
								#print(mycat.x-1, mycat.y-1, mycat.z-1)
								#ax.plot_wireframe(mycat.radius * a + mycat.x, mycat.radius * a + mycat.y, mycat.radius * c + mycat.z , color="crimson")
								is_valid_neighborhood = True
								break
						if is_valid_neighborhood:
								add_point3(ax, mycat.x, mycat.y, mycat.z, radius=mycat.radius + 0.5)
								nextpopgrid3[mycat.x, mycat.y, mycat.z] = 1
						if not is_valid_neighborhood:
							#if is_valid(cat.x, cat.y, cat.z, x, y, z):
							nextpopgrid3[cat.x, cat.y, cat.z] = 1
							add_point3(ax, cat.x, cat.y, cat.z, radius=cat.radius + 0.5)
															
						
		popgrid = nextpopgrid
		popgrid2 = nextpopgrid2
		popgrid3 = nextpopgrid3
		
		red_patch = mpatches.Patch(color='white', label='Cat')
		yellow_patch = mpatches.Patch(color='yellow', label='Cat2')
		blue_patch = mpatches.Patch(color='orange', label='Cat3')
		black_patch = mpatches.Patch(color='black', label='Food')
		water_patch = mpatches.Patch(color='blue', label='Water')
		plt.legend(handles=[red_patch, yellow_patch , blue_patch , black_patch ,  water_patch ])
		plt.show()



    
if __name__ == "__main__":
    main()
