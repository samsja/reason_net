from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
