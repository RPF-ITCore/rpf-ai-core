from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator, model_validator
from schemas.auth import UserRole


# Request DTOs
class RegisterRequest(BaseModel):
    """Request model for user registration"""
    email: EmailStr
    password: str
    password_confirm: str
    full_name: Optional[str] = None
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets requirements"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if len(v) > 72:
            raise ValueError("Password cannot be longer than 72 characters")
        return v
    
    @field_validator('password_confirm')
    @classmethod
    def validate_password_confirm(cls, v: str) -> str:
        """Validate password confirmation length"""
        if len(v) > 72:
            raise ValueError("Password confirmation cannot be longer than 72 characters")
        return v
    
    @model_validator(mode='after')
    def passwords_match(self) -> 'RegisterRequest':
        """Validate that passwords match"""
        if self.password != self.password_confirm:
            raise ValueError("Passwords do not match")
        return self


class LoginRequest(BaseModel):
    """Request model for user login"""
    email: EmailStr
    password: str


# Response DTOs
class TokenResponse(BaseModel):
    """Response model for authentication tokens"""
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Response model for user information"""
    id: str
    email: str
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class RegisterResponse(BaseModel):
    """Response model for user registration"""
    message: str
    user: UserResponse


class UserListResponse(BaseModel):
    """Response model for listing users"""
    users: list[UserResponse]
    total: int
