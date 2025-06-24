class Test:
    def __init__(self, n=1):
        self.n = n
    
    def square(self):
        return self.n**2


runner = Test(2)
print(getattr(runner, 'square')())