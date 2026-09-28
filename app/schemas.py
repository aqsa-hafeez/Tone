from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RewriteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    tone: str


class RewriteResponse(BaseModel):
    rewritten_text: str
    tone: str
    cached: bool = False


class HistoryItem(BaseModel):
    id: int
    original_text: str
    tone: str
    rewritten_text: str

    class Config:
        from_attributes = True
