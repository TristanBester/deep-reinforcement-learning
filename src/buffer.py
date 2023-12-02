class Buffer:
    def __init__(self) -> None:
        pass

    def store(self, obs, action, reward, value, logp_action):
        pass

    def complete_trajectory(self, v_term):
        pass

    def get_data(self):
        pass
