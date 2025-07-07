from minigrad.engine import Value


class Loss:
    
    def __init__(self, reduction='mean'):
        self.reduction = reduction
            
    def _apply_reduction(self, losses):
        if self.reduction == 'mean':
            return sum(losses)*(1/len(losses))
        elif self.reduction == 'sum':
            return sum(losses)
        else:
            raise ValueError(
                f'Invalid value passed for reduction: {self.reduction}'
                'Should be one of "mean", "sum"'
            )
    
    def _calculate_losses(self, actuals, preds):
        raise NotImplementedError

    def __call__(self, actuals, preds):
        losses = self._calculate_losses(actuals, preds)
        return self._apply_reduction(losses)


class MSELoss(Loss):

    def __init__(self, reduction='mean'):
        super().__init__(reduction)
    
    def _calculate_losses(self, actuals, preds):
        return [
            (ypred - yactual)**2 for ypred, yactual in zip(preds, actuals)
        ]
    

class MAELoss(Loss):
    
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
    
    def _calculate_losses(self, actuals, preds):
        return [
            abs(ypred - yactual) for ypred, yactual in zip(preds, actuals)
        ]


class HingeLoss(Loss):
    
    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def _calculate_losses(self, actuals, preds):
        return [
            (1 - yactual*ypred).relu() for ypred, yactual in zip(preds, actuals)
        ]


class HuberLoss(Loss):

    def __init__(self, reduction='mean', delta=1.0):
        super().__init__(reduction)
        self.delta = delta

    def _calculate_losses(self, actuals, preds):
        errors = [(ypred - yactual).abs() for ypred, yactual in zip(preds, actuals)]
        return [
            0.5*(error**2) if error <= self.delta else (self.delta*(error - 0.5*self.delta))
            for error in errors
        ]
        

class BCELoss(Loss):
    
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.epsilon = Value(10e-12) # use this to handle cases where 0/1 is passed to log function

    def _calculate_losses(self, actuals, preds):
        return [
            -yactual*ypred.min(1 - self.epsilon).max(self.epsilon).log() - (1 - yactual)*(1 - ypred).min(1 - self.epsilon).max(self.epsilon).log()
            for ypred, yactual in zip(preds, actuals)
        ]


class CategoricalCrossEntropyLoss(Loss):
    
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.epsilon = Value(10e-12) # use this to handle cases where 0/1 is passed to log function

    def _calculate_losses(self, actuals, preds):
        # apply softmax on preds
        normalizing_constants = [
            sum(p.exp() for p in ypred) for ypred in preds
        ]
        preds = [
            [p.exp()/nc for p in ypred]
            for nc, ypred in zip(normalizing_constants, preds)
        ]
        return [
            sum(-yactual*ypred.min(1 - self.epsilon).max(self.epsilon).log() for ypred, yactual in zip(ypreds, yactuals))
            for ypreds, yactuals in zip(preds, actuals)
        ]
