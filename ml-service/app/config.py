import os


class Config:
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    ENV: str = os.getenv("ENV", "dev")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # MCTS settings
    MCTS_ITERATIONS: int = int(os.getenv("MCTS_ITERATIONS", "2000"))
    MCTS_TIME_LIMIT_MS: int = int(os.getenv("MCTS_TIME_LIMIT_MS", "5000"))


config = Config()
