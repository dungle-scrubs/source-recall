"""Domain models for the application."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


@dataclass
class User:
    """A user in the system."""

    id: str
    name: str
    email: str
    password_hash: str
    role: UserRole = UserRole.USER
    scopes: list[str] | None = None
    created_at: datetime | None = None


@dataclass
class PaymentMethod:
    """A stored payment method."""

    id: str
    user_id: str
    card_last_four: str
    is_default: bool = False


def validate_email(email: str) -> bool:
    """Check if an email address is valid.

    @param email: Email string to validate.
    @returns: True if the email looks valid.
    """
    return "@" in email and "." in email.split("@")[1]
