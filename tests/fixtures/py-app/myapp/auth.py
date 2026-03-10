"""Authentication module for the application."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class Token:
    """JWT-like token for authentication.

    Attributes:
        user_id: The authenticated user's ID.
        expires_at: When the token expires.
        scopes: Authorized scopes.
    """

    user_id: str
    expires_at: datetime
    scopes: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return datetime.now() > self.expires_at


class AuthService:
    """Handles user authentication and authorization."""

    TOKEN_TTL = timedelta(hours=24)

    def __init__(self, secret_key: str, db: Any) -> None:
        self.secret_key = secret_key
        self.db = db
        self._cache: dict[str, Token] = {}

    def authenticate(self, email: str, password: str) -> Token | None:
        """Authenticate a user by email and password.

        @param email: User's email address.
        @param password: Plain-text password to verify.
        @returns: A Token if credentials are valid, None otherwise.
        """
        user = self.db.find_user_by_email(email)
        if user is None:
            return None
        if not self._verify_password(password, user.password_hash):
            return None
        return self._issue_token(user.id, user.scopes)

    def validate_token(self, token_str: str) -> Token | None:
        """Validate and decode a token string.

        @param token_str: The raw token string.
        @returns: Decoded Token or None if invalid/expired.
        """
        if token_str in self._cache:
            cached = self._cache[token_str]
            if not cached.is_expired:
                return cached
            del self._cache[token_str]
        return self._decode_token(token_str)

    def require_scope(self, token: Token, scope: str) -> bool:
        """Check if a token has the required scope.

        @param token: The token to check.
        @param scope: Required scope string.
        @returns: True if the scope is present.
        """
        return scope in token.scopes

    def _verify_password(self, plain: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        import hashlib

        return hashlib.sha256(plain.encode()).hexdigest() == hashed

    def _issue_token(self, user_id: str, scopes: list[str]) -> Token:
        """Create a new token for a user."""
        return Token(
            user_id=user_id,
            expires_at=datetime.now() + self.TOKEN_TTL,
            scopes=scopes,
        )

    def _decode_token(self, token_str: str) -> Token | None:  # noqa: ARG002
        """Decode a token string. Returns None if invalid."""
        # Simplified for fixture purposes.
        return None
