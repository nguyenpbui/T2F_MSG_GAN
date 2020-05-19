class Myclass(float):
	def __init__(self, real, image):
		self.y = real
		self.r = real
		self.i = image
	def foward(self, x):
		x += 1
		self.y = x
		
exam1 = Myclass(1,2)
exam1.foward(10)
print(exam1.y)
exam2 = Myclass(1,3)
#exam2(20)
print(exam2.y)