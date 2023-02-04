class people(): #  定义一个类，包括类关键字class,类名，及要继承的基类
    name = ''   #  定义类的属性
    age = 0     #  定义类变量
    __weight = 0    # 定义私有变量

    def __init__(self, n, a, w):  # 定义构造方法， 初始化结构， init, reverse, iter, next
        # n,a,w是 形参
        self.name = n  # self.name 是实例变量， n 是形参
        self.age = a
        self.__weight = w
    
    def speak(self):  # 定义类的方法（运行函数）
        age_sums = 0 # 定义局部变量
        age_sums += self.age
        print("%s 说: 我 %d 岁。" %(self.name, self.age))

    @staticmethod   # 静态方法的修饰词
    def print(): # 定义静态方法
        print("静态方法")

class student(people):
    grade = ''

    def __init__(self, n, a, w, g):
        people.__init__(self, n, a, w) # 调用父类的构造函数
        self.grade = g

    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级" %(self.name, self.age, self.grade))
    
    def __del__(self):
        print('析构函数')
    
    @classmethod
    def split(cls, str):
        str_list = str.split('-')
        n = str_list[0]
        a = int(str_list[1])
        w = int(str_list[2])
        g = int(str_list[3])
        data = cls(n, a, w, g)
        return data
# 实例化调用
s = student('ken', 10, 60, 3)  # 实例化类，并带有实参
s.speak()  # 调用类的方法
s.__del__()
del s  # 删除实例化的对象

# 静态化调用（调用静态方法）
people.print()

# 类方法调用
ss = student.split(('ken-10-60-3'))
ss.speak()