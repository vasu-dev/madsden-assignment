def approx_cordi(room_x,room_y,cam_x,cam_y,depth,startX, startY, endX, endY):

	if startX<=200:
		x = range(0,int(room_x/2+1))
	else:
		x= range(int(room_x/2+1),room_x+1)
	y = range(0,room_y+1)

	for ix in x:
		y1 = cam_y + depth**2 - (cam_x-ix)**2
		y2 = cam_y - depth**2 + (cam_x-ix)**2

		if round(y1/30) in y:
			return ix,y1/30
		elif round(y2/30) in y:
			return ix,y2/30

	return (10,10)