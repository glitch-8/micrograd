from minigrad.nn import MLP
from minigrad.engine import Value


def get_training_data():
    xs = [
        [2, 3, -1],
        [3, -1, 0.5],
        [0.5, 1, 1],
        [1, -1, -1]
    ]
    ys = [-1, -1, -1, 1]
    return xs, ys


def calculate_loss(preds, actuals):
    # implement any loss function here
    # implemented rmse loss function here
    return sum(
        (ypred - yactual)**2 for ypred, yactual in zip(preds, actuals)
    )


def train_mlp():
    xs, ys = get_training_data()

    # initialize mlp
    mlp = MLP(3, [4, 4, 1], activation='sigmoid')

    # define epochs to train for
    epoch = 10000

    # define the learning rate
    learning_rate = 0.001

    for e in range(epoch):
        # forward propogation
        ypred = [mlp(x) for x in xs]
        loss: Value = calculate_loss(ypred, ys)

        # zero grad before backprop
        mlp.zero_grad()

        # backpropogation
        loss.backward()

        # recalculate the new values for all the parameters
        for p in mlp.parameters():
            p.data += -learning_rate*p.grad
        
        print(f'Epoch: {e + 1}, Loss: {loss.data}')

    ypred = [mlp(x) for x in xs]

    print(f'{ypred=}, {ys=}')


if __name__ == '__main__':
    train_mlp()