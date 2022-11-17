class people:
    name = ''
    age = 0
    __weight = 0

    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w
    
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name, self.age))

class student(people):
    grade = ''

    def __init__(self, n, a, w, g):
        people.__init__(self, n, a, w)
        self.grade = g
    
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级" %(self.name, self.age, self.grade))

s = student('ken', 10, 60, 3)
s.speak()