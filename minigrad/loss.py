from micrograd.engine import Value


def _apply_reduction(losses, reduction = 'mean'):
    if reduction == 'mean':
        return sum(losses)*(1/len(losses))
    elif reduction == 'sum':
        return sum(losses)
    else:
        raise ValueError(
            f'Invalid value passed for reduction: {reduction}'
            'Should be one of "mean", "sum"'
        )


def MSELoss(actuals, preds) -> Value:
    '''
    Mean Squared Error Loss
    :param actuals: Actual values
    :param preds: Predicted values
    :return: Value object containing the loss value
    '''

    return sum(
        (ypred - yactual)**2 for ypred, yactual in zip(preds, actuals)
    )


def MAELoss(actuals, preds) -> Value:
    '''
    Mean Absolute Error Loss
    :param actuals: Actual values
    :param preds: Predicted values
    :return: Value object containing the loss value
    '''

    return sum(
        abs(ypred - yactual) for ypred, yactual in zip(preds, actuals)
    )


def HingeLoss(actuals, preds) -> Value:
    '''
    Hinge Loss
    :param actuals: Actual values
    :param preds: Predicted values
    :return: Value object containing the loss value
    '''

    losses = [(1 - yactual*ypred).relu() for ypred, yactual in zip(preds, actuals)]
    loss = sum(losses)*(1/len(losses))
    return loss


def HuberLoss(actuals, preds, reduction='mean', delta=1.0) -> Value:
    '''
    Huber Loss
    :param actuals: Actual values
    :param preds: Predicted values
    :return: Value object containing the loss value
    '''

    errors = [(ypred - yactual).abs() for ypred, yactual in zip(preds, actuals)]
    losses =[
        0.5(error**2) if error <= delta else (delta*(error - 0.5*delta))
        for error in errors
    ]

    return _apply_reduction(losses, reduction)
        

def BCELoss(actuals, preds, reduction='mean') -> Value:
    '''
    Binary Cross Entropy Loss
    :param actuals: Actual values
    :param preds: Predicted values
    :return: Value object containing the loss value
    '''

    losses = [
        -yactual*ypred.log() - (1 - yactual)*(1 - ypred).log()
        for ypred, yactual in zip(preds, actuals)
    ]

    return _apply_reduction(losses, reduction)


def CategoricalCrossEntropyLoss(actuals, preds, reduction='mean') -> Value:
    '''
    Categorical Cross Entropy Loss
    :param actuals: Actual values
    :param preds: Predicted values
    :return: Value object containing the loss value
    '''

    losses = [
        sum(-yactual*ypred.log() for ypred, yactual in zip(ypreds, yactuals))
        for ypreds, yactuals in zip(preds, actuals)
    ]

    return _apply_reduction(losses, reduction)
