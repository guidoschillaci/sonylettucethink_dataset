# Authors:
# Antonio Pico Villalpando, Humboldt-Universitaet zu Berlin, Germany, pivillaa@informatik.hu-berlin.de
# Guido Schillaci, The BioRobotics Institute, Scuola Superiore Sant'Anna, Pisa, Italy, guido.schillaci@santannapisa.it


from __future__ import print_function # remove this if you use Python 3

import sys
import os
import cv2
import numpy as np
import re
import gzip 
import pickle
from util import Position, x_lims, y_lims, z_lims, speed_lim

class Cam_sim():
	def __init__(self,imagesPath):
		self.imagesPath = imagesPath
		if self.imagesPath[-1] != '/':
			self.imagesPath += '/'

	def round2mul(self,number, multiple):
		half_mult = multiple/2.
		result = np.floor( (number + half_mult) / multiple  ) * multiple
		return result.astype(np.int)
	
	def get_trajectory(self,start,end):
		trajectory =  np.array(self.get_line(start, end))
		return trajectory

	def get_trajectory_names(self,start,end):
		trajectory = self.get_trajectory(start,end)
		t_rounded = self.round2mul(trajectory,5) #there is only images every 5 mm, use closer image to real coordinate
		t_images = []
		for i in t_rounded:
			img_name = self.imagesPath + "x{:03d}_y{:03d}.jpeg".format(i[0],i[1])
			t_images.append(img_name)
		return t_images

	def get_line(self,start, end):
		"""Bresenham's Line Algorithm
		Produces a list of tuples from start and end

		>>> points1 = get_line((0, 0), (3, 4))
		>>> points2 = get_line((3, 4), (0, 0))
		>>> assert(set(points1) == set(points2))
		>>> print points1
		[(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
		>>> print points2
		[(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
		"""
		# Setup initial conditions
		x1, y1 = start
		x2, y2 = end
		dx = x2 - x1
		dy = y2 - y1

		# Determine how steep the line is
		is_steep = abs(dy) > abs(dx)

		# Rotate line
		if is_steep:
			x1, y1 = y1, x1
			x2, y2 = y2, x2

		# Swap start and end points if necessary and store swap state
		swapped = False
		if x1 > x2:
			x1, x2 = x2, x1
			y1, y2 = y2, y1
			swapped = True

		# Recalculate differentials
		dx = x2 - x1
		dy = y2 - y1

		# Calculate error
		error = int(dx / 2.0)
		ystep = 1 if y1 < y2 else -1

		# Iterate over bounding box generating points between start and end
		y = y1
		points = []
		for x in range(x1, x2 + 1):
			coord = (y, x) if is_steep else (x, y)
			points.append(coord)
			error -= abs(dy)
			if error < 0:
				y += ystep
				error += dx

		# Reverse the list if the coordinates were swapped
		if swapped:
			points.reverse()
		return points







def parse_data(file_name, pixels, reshape, step, channels=1):

	images = []
	positions = []
	commands = []
	test_pos = []
	with gzip.open(file_name, 'rb') as memory_file:
		memories = pickle.load(memory_file)
		print ('converting data...')
		count = 0
		for memory in memories:
			image = memory['image']
			if channels == 1:
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				image = cv2.resize(image, (pixels, pixels))
				images.append(np.asarray(image))

				cmd = memory['command']
	#			commands.append([float(cmd.x) / x_lims[1], float(cmd.y) / y_lims[1], float(cmd.z) / z_lims[1], float(cmd.speed) / speed_lim])
				commands.append([float(cmd.x) / x_lims[1], float(cmd.y) / y_lims[1]])
				pos = memory['position']
	#			positions.append([float(pos.x) /x_lims[1], float(pos.y) / y_lims[1], float(pos.z) / z_lims[1]])
				positions.append([float(pos.x) /x_lims[1], float(pos.y) / y_lims[1]])

				#if count%step == 0:
				#	img_name = './sample_images/img_' + str(count) + '.jpg'
				#	cv2.imwrite(img_name, image)
				count+=1

	positions = np.asarray(positions)
	commands = np.asarray(commands)
	images = np.asarray(images)
	if reshape:
		images = images.reshape((len(images), pixels, pixels, channels))
	return images.astype('float32') / 255, commands, positions

def load_data (dataset, image_size, step):
	print ('set image_size to ', image_size)
	print ('set step for test data to ', step)
	images, commands, positions = parse_data(dataset, step = step, reshape=True, pixels = image_size)
	# split train and test data
	indices = range(0, len(positions), step)
	# split images
	test_images = images[indices]
	train_images = images[~indices]
	test_cmds = commands[indices]
	train_cmds = commands[~indices]
	test_pos = positions[indices]
	train_pos = positions[~indices]
	print ("number of train images: ", len(train_images))
	print ("number of test images: ", len(test_images))

	return train_images, test_images, train_cmds, test_cmds, train_pos, test_pos

def extract_images_from_pkl(path, file_name):
	file_ = path+file_name
	with gzip.open(file_, 'rb') as memory_file:
		memories = pickle.load(memory_file)
		print ('extracting images...')
		count = 0
		for memory in memories:
			image = memory['image']
			#image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

			cmd = memory['position']
			title = path + '/x'+str(cmd.x)+'_y'+str(cmd.y)+'.jpeg'
			cv2.imwrite(title,image)
			print ('written image: ', title)
	print ('Images extracted') 


if __name__ == '__main__':

	path="./raw_images"

	compressed_dataset_filename = '/compressed_dataset.pkl'

	if os.path.isfile(path+compressed_dataset_filename):
		print ('compressed dataset already exists.')
		inp =''
		if sys.version_info[0] < 3: # python2
			inp = raw_input('Do you want to extract images and save them into ./raw_images/ folder? [y/n]:')
		else:
			inp = input('Do you want to extract images and save them into ./raw_images/ folder? [y/n]:')
		
		while inp not in ("y", "n"):
			if sys.version_info[0] < 3: 
				inp = raw_input('Type y or n: ')  
			else:
				inp = input('Type y or n: ')  

		if (inp == 'y'):
			print ('extracting jpg images from pkl file')
			extract_images_from_pkl(path, compressed_dataset_filename)
		else:
			print ('Ok. Terminating program')
	else:
		print ('creating compressed dataset')

		samples = []
		counter=0
		for file in os.listdir(path):
			filename_full = os.path.basename(file)
			filename = os.path.splitext(filename_full)[0]
			splitted = re.split('x|_|y', filename)
			p = Position()
			p.x=splitted[1]
			p.y=splitted[3]
			p.z=-90
			p.speed=1400
			#print path+'/'+os.path.relpath(file)
			cv_img = cv2.imread(path+'/'+os.path.relpath(file))
			cv_img = cv2.resize(cv_img, (64, 64))
			#image_msg= bridge.cv2_to_imgmsg(cv_img, "bgr8")
			#samples.append({'image': image_msg, 'position':p, 'command':p})
			samples.append({'image': cv_img, 'position': p, 'command': p})
			#print int(p.x), ' ', int(p.y)
			counter=counter+1
			print (counter)
		with gzip.open(path+ compressed_dataset_filename, 'wb') as file:
			pickle.dump(samples, file, protocol=2)
		print ('Compressed dataset saved to ', path+compressed_dataset_filename)
		sys.exit(1)

	print ('Testing load data from compressed dataset')
	load_data(path+compressed_dataset_filename, 64, 10)
