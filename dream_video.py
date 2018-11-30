import cv2
import os

dream_name = "test2"
dream_path = "dream/{}".format(dream_name)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('{}.avi'.format(dream_name),fourcc,13,(455,699))

for i in range(9999999999999999):
	if os.path.isfile('dream/{}/img_{}.jpg'.format(dream_name,i+1)):
		print('{} already exists, continuing along ...'.format(i+1))
	else:
		dream_length = i 
		break
		
for i in range(dream_length):
	img_path = os.path.join(dream_path, "img_{}.jpg".format(i))
	print(img_path)
	frame = cv2.imread(img_path)
	out.write(frame)
	
out.release()
