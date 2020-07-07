from sklearn.metrics import recall_score
from toolz import curry


# Assume auto_grad is turned off
@curry
def validate_war(batch, model, crit, env):
    inputs, targets, lens,weights = env.loadBatch(batch, weight=True)
    outputs = model(inputs,lens)
    loss = crit(outputs, targets)
    score = recall_score(
        outputs.argmax(1).cpu(),
        targets.cpu(),
        average='weighted',
        sample_weight=weights)
    return loss.item(), score


@curry
def validate_uar(batch, model, crit, env):
    inputs, targets,lens = env.loadBatch(batch)
    outputs = model(inputs,lens)
    loss = crit(outputs, targets)

    score = recall_score(
        outputs.argmax(1).cpu(),
        targets.cpu(),
        average='macro')

    return loss.item(), score


ALL = [validate_uar, validate_war]
