"""
otp_service.py — OTP generation, storage, verification, and expiry.

Uses an in-memory store (dict). For multi-instance deployments,
swap _store for Redis. Single Railway instance → this is fine.
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import os

OTP_LENGTH        = int(os.getenv("OTP_LENGTH", "6"))
OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", "10"))


@dataclass
class OTPRecord:
    otp: str
    expires_at: datetime
    attempts: int = field(default=0)  # brute-force guard: max 5 attempts
    verified: bool = field(default=False)


# In-memory store: email → OTPRecord
_store: dict[str, OTPRecord] = {}

MAX_ATTEMPTS = 5


def generate_otp() -> str:
    """Return a zero-padded numeric OTP string of length OTP_LENGTH."""
    return "".join(random.choices(string.digits, k=OTP_LENGTH))


def create_otp(email: str) -> str:
    """
    Generate and store a fresh OTP for the given email.
    Overwrites any existing OTP for that address.
    """
    otp = generate_otp()
    _store[email.lower()] = OTPRecord(
        otp=otp,
        expires_at=datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES),
    )
    return otp


def verify_otp(email: str, otp: str) -> tuple[bool, str]:
    """
    Verify the OTP for the given email.

    Returns (success: bool, reason: str).
    Marks the record as verified on success so the reset step can proceed.
    """
    key = email.lower()
    record = _store.get(key)

    if record is None:
        return False, "No OTP found. Please request a new one."

    if datetime.utcnow() > record.expires_at:
        _store.pop(key, None)
        return False, "OTP expired. Please request a new one."

    if record.attempts >= MAX_ATTEMPTS:
        _store.pop(key, None)
        return False, "Too many incorrect attempts. Please request a new OTP."

    if record.otp != otp.strip():
        record.attempts += 1
        remaining = MAX_ATTEMPTS - record.attempts
        return False, f"Incorrect OTP. {remaining} attempt(s) remaining."

    # Success — mark verified so reset endpoint can trust it
    record.verified = True
    return True, "OTP verified."


def is_verified(email: str) -> bool:
    """Check whether the OTP for this email has already been verified."""
    record = _store.get(email.lower())
    return record is not None and record.verified


def consume_otp(email: str) -> None:
    """Delete the OTP record after a successful password reset."""
    _store.pop(email.lower(), None)


def time_remaining(email: str) -> Optional[int]:
    """Return seconds remaining before expiry, or None if no record."""
    record = _store.get(email.lower())
    if not record:
        return None
    delta = record.expires_at - datetime.utcnow()
    return max(0, int(delta.total_seconds()))
