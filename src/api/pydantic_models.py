from pydantic import BaseModel
from typing import List

# Request model
class PredictionRequest(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    TransactionStartTime: str
    CurrencyCode: str
    Amount: float
    Value: float
    ProductCategory: str
    ChannelId: str
    ProviderId: str
    PricingStrategy: int

# Response model
class PredictionResponse(BaseModel):
    cluster: int
    probability: float
