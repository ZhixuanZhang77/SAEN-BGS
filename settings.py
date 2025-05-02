class Settings:
    def __init__(self):
        self.training = config()

class config:
    def __init__(self):
        self.device = 'cpu'#'cuda:0'

SETTINGS = Settings()
