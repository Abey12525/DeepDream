from deepdreamer import model, load_image, recursive_optimize
import numpy as np 
import PIL.Image
import cv2
import os
import random as r
from math import sqrt

#layer_tensor = model.layer_tensors[3]
dream_name = 'test2'

x_size = 455
y_size = 699

created_count = 0
max_count = 200

bri_result = load_image(filename='dream/{}/img_0.jpg'.format(dream_name))
		# this impacts how quick the "zoom" is 
		
x_trim = 2 
y_trim = 1
		
bri_result = img_result[0+x_trim:y_size-y_trim, 0+y_trim:x_size-x_trim]
bri_result = cv2.resize(img_result, (x_size,y_size))
brightness = sqrt(0.241*bri_result[:,:,0]+0.691*bri_result[:,:,1]+0.068*bri_result[:,:,2])
Avg_brightness = brightness*0.6
for i in range(0,200):
	if os.path.isfile('dream/{}/img_{}.jpg'.format(dream_name,i+1)):
		print('{} already exists, continuing along ...'.format(i+1))
	else:
		img_result = load_image(filename='dream/{}/img_{}.jpg'.format(dream_name,i))
		# this impacts how quick the "zoom" is 
		
		img_result = img_result[0+x_trim:y_size-y_trim, 0+y_trim:x_size-x_trim]
		img_result = cv2.resize(img_result, (x_size,y_size))
		
		#Use these to modify the general colors and brightness of results.
		#results tend to get dimmer or brighter over time , so you want to 
		#mannually adjust this over time
		
		#+2 is slowly dimmer 
		#+3 is slowly brighter
		brightness_d = sqrt(0.241*img_result[:,:,0]+0.691*img_result[:,:,1]+0.068*img_result[:,:,2])
		if(brightness_d<Avg_brightness):
			img_result[:,:,0] += r.randint(3,4) #red
			img_result[:,:,1] += r.randint(3,4) #green
			img_result[:,:,2] += r.randint(3,4) #blue
		img_result = np.clip(img_result, 0.0 , 255.0)
		img_result = img_result.astype(np.uint8)
		
		img_result = recursive_optimize(layer_tensor=model.layer_tensors[r.randint(0,11)],
										image = img_result,
										num_iterations = 2,
										step_size = 1.0,
										rescale_factor = 0.7,
										num_repeats = 8,
										blend = 0.2)
										
		img_result = np.clip(img_result,0.0,255.0)
		img_result = img_result.astype(np.uint8)
		result = PIL.Image.fromarray(img_result,mode = 'RGB')
		result.save('dream/{}/img_{}.jpg'.format(dream_name,i+1))
		
		created_count+=1
		if created_count > max_count:
			break
											
