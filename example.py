import cam_sim
import util
import cv2

channels = 1 # grayscale or rgb image
image_size = 64 # image_size * image_size

initial_position = [[100, 100]]
command = [[200, 200]]
images = []

a = [int(initial_position[0]), int(initial_position[1])]
b = [int(command[0]), int(command[1])]

tr = cam_sim.get_trajectory(a,b)
trn = cam_sim.get_trajectory_names(a,b)
        
rounded  = cam_sim.round2mul(tr,5) # only images every 5mm

# the lists containing the generated trajectories
# cmd is kept constant until position gets to this values 
pos = [] # x,y motor positions
cmd = [] # x,y desired motor positions
img = [] # images from the current x,y motor position
for i in range(len(tr)): 
	pos.append([float(rounded[i][0]) / x_lims[1], float(rounded[i][1]) / y_lims[1]] )
	cmd([float(int(command[0])) / x_lims[1], float(int(command[1])) / y_lims[1]] )
	cv2_img = cv2.imread(trn[i],1 )
	if channels ==1:
		cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
		cv2_img = cv2.resize(cv2_img,(image_size, image_size), interpolation = cv2.INTER_LINEAR)
		cv2_img = cv2_img.astype('float32') / 255
		cv2_img.reshape(1, image_size, image_size, channels)		
		img.append(cv2_img)

print ('positions ', pos) 
