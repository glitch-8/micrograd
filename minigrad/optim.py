class Optim:
    
    def __init__(self, params, lr = 10e-2):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError()


class SGD(Optim):

    def __init__(self, params, lr = 10e-2, momentum=0.9, nesterovs=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterovs = nesterovs
        self.velocities = [0]*len(params)

    def step(self):
        for counter, p in enumerate(self.params):
            velocity_prev = self.velocities[counter]
            velocity_new = velocity_prev*self.momentum + self.lr*p.grad

            if self.nesterovs:
                p.data -= (self.momentum*velocity_new + self.lr*p.grad)
            else:
                p.data -= velocity_new

            self.velocities[counter] = velocity_new


class ADAGrad(Optim):

    def __init__(self, params, lr = 10e-2):
        super().__init__(params, lr)
        self.state = [0]*len(params)
    
    def step(self):
        for counter, p in enumerate(self.params):
            state_old = self.state[counter]
            state_new = state_old + p.grad**2
            p.data -= self.lr * p.grad / (state_new + 1e-6)**0.5

            self.state[counter] = state_new # update the state


class RMSProp(Optim):

    def __init__(self, params, lr = 10e-2, gamma=0.9):
        super().__init__(params, lr)
        self.gamma = gamma
        self.state = [0]*len(params)
    
    def step(self):
        for counter, p in enumerate(self.params):
            state_old = self.state[counter]
            state_new = self.gamma*state_old + (1-self.gamma)*p.grad**2
            p.data -= self.lr * p.grad / (state_new + 1e-6)**0.5

            self.state[counter] = state_new # update the state


class ADAM(Optim):

    def __init__(self, params, lr=10e-2, beta1=0.9, beta2=0.999):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0 # timestamp

        self.state = [(0, 0)]*len(params)
    
    def step(self):
        self.t += 1
        for counter, p in enumerate(self.params):
            state_old = self.state[counter]
            first_moment_old = state_old[0]
            second_moment_old = state_old[1]

            first_moment_new = self.beta1*first_moment_old + (1-self.beta1)*p.grad
            second_moment_new = self.beta2*second_moment_old + (1-self.beta2)*p.grad**2

            first_moment_corrected = first_moment_new/(1-self.beta1**self.t)
            second_moment_corrected = second_moment_new/(1-self.beta2**self.t)

            p.data -= self.lr * first_moment_corrected/(second_moment_corrected**0.5 + 1e-6)

            self.state[counter] = (first_moment_new, second_moment_new) # update state



if __name__ == '__main__':
    from minigrad.engine import Value

    params = [Value(1) if counter %2 == 0 else Value(2) for counter in range(5)]
    for counter, p in enumerate(params):
        if counter % 2 == 0:
            p.grad = 0.01
        else:
            p.grad = 0.5

    optim = SGD(params)

    for e in range(0, 2):
        optim.step()
        
        for p in params:
            p.grad = 0

        print(f'Epoch: {e}, params: {params}')


