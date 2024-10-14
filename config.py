from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    #######
    # AI configurations part
    # BASE_URI : str = "http://app.searcly.com:5004"

    # API configurations part
    OPENAI_API_KEY: Optional[str] = Field(env='OPENAI_API_KEY')
    LANGCHAIN_API_KEY: Optional[str] = Field(env='LANGCHAIN_API_KEY')


    # Paths for storing content and figures
    CONTENT_PATH: str = "data/content/"
    FIGURES_PATH: str = "figures/"
    IMAGE_SUMMARIES_PATH: str = "data/content/image_summaries.json"
    TABLE_SUMMARIES_PATH: str = "data/content/table_summaries.json"

    # Choose device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        case_sensitive = True
        env_file = '.env'


settings = Settings()
