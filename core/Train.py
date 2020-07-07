import torch as tc
from toolz import curry
from tqdm import tqdm


@curry
def validate_loop_lazy(name, validator, loader, log):
    loss, score = 0, 0
    with tc.no_grad():
        for batch in loader:
            l, s = validator(batch)
            loss += l * len(batch[0])
            score += s * len(batch[0])
    score, loss = score / len(loader.dataset), loss / len(loader.dataset)

    log.write('[%s] score: %.3f, loss: %.3f\n' % (name, score, loss))

    return loss, score


def train(model, loader, env, validators: dict, valid_loop, crit, optim):
    best_score: {str: float} = dict.fromkeys(validators.keys(), 0)
    log = env.log_file
    for epoch in tqdm(range(env.epoch)):
        for batch in loader:
            inputs, targets, lens = env.loadBatch(batch)

            model.train()  # training

            train_loss = crit(model(inputs, lens), targets)
            train_loss.backward()

            optim.step()
            optim.zero_grad()
            model.eval()  # evaluating

        log.write('[train] %4d/%4dth epoch, loss: %.3f\n' % (epoch, env.epoch, train_loss.item()))

        for k, v in validators.items():
            loss, score = valid_loop(validator=v, name='valid-%s' % k)
            if score > best_score[k]:
                best_score[k] = score
                log.write('[valid-%s] bestscore: %.3f, loss: %.3f\n' % (k, score, loss))
                if k == env.measure:
                    env.save(model, False)
        env.EXE.submit(log.flush)

    print('Finished Training')
    model.load_state_dict(env.load())
