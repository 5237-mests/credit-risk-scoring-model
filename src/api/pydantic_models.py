from pydantic import BaseModel


class PredictionRequest(BaseModel):
    TransactionId: str
    CustomerId: str
    TransactionStartTime: str
    Amount: float
    Value: float
    ProductId: str
    ProductCategory: str
    ChannelId: str
    ProviderId: str
    PricingStrategy: int


class PredictionResponse(BaseModel):
    cluster: int
    probability: float
