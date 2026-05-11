"""
email_service.py — Resend-powered email sending for DataLab.

Sender: otp@data-lab.co.uk  (verified domain on Resend)
"""

from __future__ import annotations

import os
import resend
from dotenv import load_dotenv

load_dotenv()

resend.api_key = os.environ["RESEND_API_KEY"]

FROM_EMAIL = os.getenv("FROM_EMAIL", "otp@data-lab.co.uk")
FROM_NAME  = os.getenv("FROM_NAME", "DataLab")
FROM_FULL  = f"{FROM_NAME} <{FROM_EMAIL}>"


def _otp_html(otp: str, expiry_minutes: int = 10) -> str:
    """Return the HTML body for the OTP email."""
    return f"""
    <div style="font-family:'Helvetica Neue',sans-serif;max-width:520px;margin:0 auto;
                padding:40px 32px;background:#030712;border-radius:16px;color:#e0e8ff;">

      <!-- Logo -->
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:28px;">
        <div style="width:36px;height:36px;border-radius:10px;
                    background:linear-gradient(135deg,#1d4ed8,#8b5cf6);
                    display:flex;align-items:center;justify-content:center;font-size:18px;">🧪</div>
        <span style="font-size:22px;font-weight:800;letter-spacing:-0.02em;">
          <span style="color:#e0e8ff;">Data</span>
          <span style="background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Lab</span>
        </span>
      </div>

      <h2 style="font-size:20px;font-weight:700;margin:0 0 8px;color:#e0e8ff;">
        Password Reset OTP
      </h2>
      <p style="font-size:14px;color:#6b7a9a;margin:0 0 28px;line-height:1.6;">
        Use the code below to reset your DataLab password.
        It expires in <strong style="color:#e0e8ff;">{expiry_minutes} minutes</strong>.
      </p>

      <!-- OTP box -->
      <div style="background:#0d1b3e;border:1px solid #1e3a8a;border-radius:14px;
                  padding:28px;text-align:center;margin-bottom:28px;">
        <div style="font-size:42px;font-weight:900;letter-spacing:14px;
                    color:#6fa3ef;font-family:monospace;user-select:all;">
          {otp}
        </div>
        <div style="font-size:12px;color:#4a5a7a;margin-top:10px;
                    font-family:monospace;letter-spacing:1px;">
          ONE-TIME PASSWORD
        </div>
      </div>

      <!-- Security note -->
      <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);
                  border-radius:10px;padding:14px 16px;margin-bottom:24px;">
        <p style="font-size:12px;color:#fca5a5;margin:0;line-height:1.6;">
          🔒 Never share this code with anyone. DataLab will never ask for your OTP
          via phone or email.
        </p>
      </div>

      <p style="font-size:12px;color:#2a3a5a;margin:0;text-align:center;">
        If you didn't request this, you can safely ignore this email.
      </p>

      <hr style="border:none;border-top:1px solid #0d1b3e;margin:28px 0 16px;">
      <p style="font-size:11px;color:#1e2a3a;text-align:center;margin:0;
                font-family:monospace;letter-spacing:0.5px;">
        DataLab · data-lab.co.uk · AI-Powered Data Research Suite
      </p>
    </div>
    """


def send_otp_email(to_email: str, otp: str, expiry_minutes: int = 10) -> bool:
    """
    Send an OTP email via Resend.

    Returns True on success, raises on failure (let the endpoint handle it).
    """
    params: resend.Emails.SendParams = {
        "from": FROM_FULL,
        "to": [to_email],
        "subject": f"Your DataLab OTP: {otp}",
        "html": _otp_html(otp, expiry_minutes),
    }
    response = resend.Emails.send(params)
    # Resend returns {"id": "..."} on success
    return bool(response.get("id"))


def send_welcome_email(to_email: str, name: str) -> bool:
    """Optional: send a welcome email after successful password reset."""
    params: resend.Emails.SendParams = {
        "from": FROM_FULL,
        "to": [to_email],
        "subject": "DataLab — Password Reset Successful",
        "html": f"""
        <div style="font-family:sans-serif;max-width:520px;margin:0 auto;
                    padding:40px 32px;background:#030712;border-radius:16px;color:#e0e8ff;">
          <h2 style="color:#4ade80;margin:0 0 12px;">✓ Password Reset Successful</h2>
          <p style="color:#8899bb;font-size:14px;line-height:1.6;">
            Hi {name}, your DataLab password has been successfully reset.
            You can now log in with your new password.
          </p>
          <p style="color:#2a3a5a;font-size:12px;margin-top:24px;">
            If you did not make this change, contact us immediately at otp@data-lab.co.uk.
          </p>
        </div>""",
    }
    response = resend.Emails.send(params)
    return bool(response.get("id"))
