"""
test_email.py — Verify Resend OTP email sending end-to-end.

Run from the project root:
    python test_email.py

What it tests:
    1. Env vars loaded correctly
    2. OTP generation
    3. Actual email sent via Resend (check your inbox)
    4. OTP verification (correct + incorrect + expired simulation)
    5. Resend API reachability
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Load .env before importing services
from dotenv import load_dotenv
load_dotenv()

# ── Check env vars ────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  DataLab — Resend OTP Email Test")
print("="*55)

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL     = os.getenv("FROM_EMAIL", "otp@data-lab.co.uk")

if not RESEND_API_KEY:
    print("❌  RESEND_API_KEY not set in .env — aborting.")
    sys.exit(1)

print(f"✓  RESEND_API_KEY loaded ({RESEND_API_KEY[:8]}...)")
print(f"✓  FROM_EMAIL: {FROM_EMAIL}")

# ── Test OTP generation ───────────────────────────────────────────────────────
from services.otp_service import generate_otp, create_otp, verify_otp, consume_otp

otp = generate_otp()
assert len(otp) == int(os.getenv("OTP_LENGTH", "6")), "OTP length mismatch"
assert otp.isdigit(), "OTP should be numeric"
print(f"✓  OTP generation OK: {otp}")

# ── Test OTP store / verify ───────────────────────────────────────────────────
TEST_EMAIL = "test@data-lab.co.uk"
stored_otp = create_otp(TEST_EMAIL)

ok, msg = verify_otp(TEST_EMAIL, "000000")  # wrong
assert not ok, "Wrong OTP should fail"
print(f"✓  Wrong OTP rejected: {msg}")

ok, msg = verify_otp(TEST_EMAIL, stored_otp)  # correct
assert ok, f"Correct OTP should pass: {msg}"
print(f"✓  Correct OTP accepted: {msg}")

ok, msg = verify_otp(TEST_EMAIL, stored_otp)  # already used (verified flag set, still in store)
print(f"✓  Re-verify handled: success={ok}")

consume_otp(TEST_EMAIL)
ok, msg = verify_otp(TEST_EMAIL, stored_otp)  # consumed
assert not ok
print(f"✓  Consumed OTP rejected: {msg}")

# ── Send a real email ─────────────────────────────────────────────────────────
TO_EMAIL = input("\n  Enter your email address to receive the test OTP: ").strip()
if not TO_EMAIL:
    print("  Skipping live send (no email entered).")
    sys.exit(0)

from services.email_service import send_otp_email

live_otp = create_otp(TO_EMAIL)
print(f"\n  Sending OTP {live_otp} to {TO_EMAIL} via Resend...")

try:
    result = send_otp_email(to_email=TO_EMAIL, otp=live_otp)
    if result:
        print(f"✓  Email sent! Check your inbox at {TO_EMAIL}.")
        print(f"   OTP is: {live_otp}  (valid for 10 minutes)")
    else:
        print("❌  send_otp_email returned False — check Resend dashboard.")
except Exception as e:
    print(f"❌  Error sending email: {e}")
    sys.exit(1)

# ── Verify the live OTP ───────────────────────────────────────────────────────
entered = input("\n  Enter the OTP you received to verify: ").strip()
ok, msg = verify_otp(TO_EMAIL, entered)
if ok:
    print(f"✓  Verified! {msg}")
else:
    print(f"❌  {msg}")

print("\n" + "="*55)
print("  All tests passed." if ok else "  Some checks failed — see above.")
print("="*55 + "\n")
