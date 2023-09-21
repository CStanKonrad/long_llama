from flax.training.train_state import TrainState

def get_gradient_step(train_state: TrainState):
    # Flax TrainState stores the number of calls to apply_gradients
    # the optimizer itself stores the number of times gradiends were
    # really applied to the model parameters
    try:
        return train_state.opt_state.gradient_step
    except AttributeError:
        return train_state.step