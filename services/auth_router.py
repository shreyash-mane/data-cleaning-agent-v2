"""
auth_router.py — FastAPI router for forgot-password OTP flow.

Mount in main.py with:
    from services.auth_router import router as auth_router
    app.include_router(auth_router, prefix="/auth", tags=["auth"])

Endpoints:
    POST /auth/forgot/send-otp      → send OTP to email
    POST /auth/forgot/verify-otp    → verify OTP (returns verified=True/False)
    POST /auth/forgot/reset-password → verify OTP + reset password in one step
"""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr

from services.email_service import send_otp_email, send_welcome_email
from services.otp_service import (
    create_otp,
    verify_otp,
    is_verified,
    consume_otp,
    time_remaining,
    OTP_EXPIRY_MINUTES,
)

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class SendOTPRequest(BaseModel):
    email: EmailStr


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str
    confirm_password: str


class MessageResponse(BaseModel):
    success: bool
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_password(password: str) -> None:
    """Raise HTTPException if password doesn't meet requirements."""
    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters.",
        )
    if not any(c.isdigit() for c in password):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must contain at least one digit.",
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/forgot/send-otp", response_model=MessageResponse)
async def send_otp(request: SendOTPRequest):
    """
    Generate a 6-digit OTP and email it to the given address.
    Rate-limit: one OTP per 60s is handled at infra level (Railway / reverse proxy).
    """
    email = request.email.lower()

    # Check if a valid OTP was sent very recently (basic flood guard)
    secs = time_remaining(email)
    if secs is not None and secs > (OTP_EXPIRY_MINUTES * 60 - 30):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OTP already sent. Please wait 30 seconds before requesting again.",
        )

    otp = create_otp(email)

    try:
        sent = send_otp_email(to_email=email, otp=otp, expiry_minutes=OTP_EXPIRY_MINUTES)
    except Exception as exc:
        # Don't leak internal errors to the client
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to send OTP email. Please try again.",
        ) from exc

    if not sent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email service error. Please try again.",
        )

    return MessageResponse(
        success=True,
        message=f"OTP sent to {email}. Expires in {OTP_EXPIRY_MINUTES} minutes.",
    )


@router.post("/forgot/verify-otp", response_model=MessageResponse)
async def verify(request: VerifyOTPRequest):
    """
    Verify the OTP. Call this before showing the new-password field.
    The OTP is marked internally as verified; the reset endpoint checks this.
    """
    success, reason = verify_otp(request.email.lower(), request.otp)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)
    return MessageResponse(success=True, message="OTP verified. You may now set a new password.")


@router.post("/forgot/reset-password", response_model=MessageResponse)
async def reset_password(request: ResetPasswordRequest):
    """
    Verify OTP and reset password in one step.
    Requires the OTP to have been marked verified via /verify-otp first,
    OR accepts the raw OTP here directly (single-step flow).
    """
    email = request.email.lower()

    # Validate passwords match
    if request.new_password != request.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Passwords do not match.",
        )
    _validate_password(request.new_password)

    # Accept either pre-verified or verify inline
    if not is_verified(email):
        success, reason = verify_otp(email, request.otp)
        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)

    # ── Password update goes here ─────────────────────────────────────────────
    # Replace the block below with your actual DB update logic.
    # Examples:
    #   SQLAlchemy: await db.execute(update(User).where(User.email==email).values(password=hashed))
    #   MongoDB:    await users.update_one({"email": email}, {"$set": {"password": hashed}})
    #   Raw SQL:    cursor.execute("UPDATE users SET password=? WHERE email=?", (hashed, email))
    #
    # import bcrypt
    # hashed = bcrypt.hashpw(request.new_password.encode(), bcrypt.gensalt()).decode()
    # await update_user_password(email=email, hashed_password=hashed)
    # ─────────────────────────────────────────────────────────────────────────

    consume_otp(email)  # Invalidate the OTP immediately after use

    # Fire-and-forget confirmation email (don't fail the request if this errors)
    try:
        send_welcome_email(to_email=email, name=email.split("@")[0].capitalize())
    except Exception:
        pass

    return MessageResponse(success=True, message="Password reset successfully. You can now log in.")
