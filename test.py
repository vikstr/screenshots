
import numpy as np
import cv2


def canBeSame(rect1, rect2):
	x1, y1, w1, h1 = rect1
	x2, y2, w2, h2 = rect2
	xc1 = x1 + w1 / 2
	xc2 = x2 + w2 / 2
	yc1 = y1 + h1 / 2
	yc2 = y2 + h2 / 2
	if(abs(xc1 - xc2) > 3):
		return False
	if(abs(yc1 - yc2) > 3):
		return False
	if(abs(w1 - w2) > 10):
		return False
	if(abs(h1 - h2) > 10):
		return False
	return True

def kuhn(n, graph):
	answer = []
	pair= [-1] * n * 2
	used = []
	def try_kuhn(v):
		if(used[v]):
			return False
		for u in graph[v]:
			if(pair[u] == -1 or try_kuhn(u)):
				pair[u] = v
				return True
		return False
	
	for v in range(n):
		used = [0] * n * 2
		try_kuhn(v)
	for v in range(n, n + n):
		if(pair[v] == -1):
			answer.append(v)
	return answer
			

def getRects(filename):
	image = cv2.imread(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edged = cv2.Canny(gray, 10, 250)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#print(cnts[0])
	total = 0
	rects = []
	for c in cnts:
        #approximate countours
    		peri = cv2.arcLength(c, True)
    		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    		if len(approx) == 4:
    			cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
    			total += 1
    			x,y,w,h = cv2.boundingRect(c)
    			#print(x, y, w, h)
			rects += [(x,y,w,h)] 
	return rects

def getGraph(rects1, rects2):
	n = len(rects1)
	graph = []
	for i in range(2 * n):
		graph.append([])
	for i in range(n):
		for j in range(n):
			if(canBeSame(rects1[i], rects2[j])):
				graph[i].append(n + j)
				graph[n+j].append(i)
	return graph

def solve(img1, img2, result):
	rects1 = getRects(img1)
	rects2 = getRects(img2)
	#print(len(rects1))
	#print(len(rects2))
	if(len(rects1) != len(rects2)):
		print("The numbers of buttons aren`t same")
	elif(len(rects1) ==0):
		print("There is nothing here")
	else:
		graph = getGraph(rects1, rects2)
		#print(graph)
		singles = kuhn(len(rects1), graph)
		res = cv2.imread(img2)
		for rect_num in singles:
			x, y, w, h = rects2[rect_num - len(rects1)]
			cv2.rectangle(res,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.imwrite(result, res)

import sys
if len(sys.argv) == 3:
	solve(sys.argv[1], sys.argv[2], 'res.png')
elif len(sys.argv)>= 4:
	solve(sys.argv[1], sys.argv[2], sys.argv[3])
else:
	print("error")


	
	
