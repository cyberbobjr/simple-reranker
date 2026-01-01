import time
import json
import os
import logging
import threading
from typing import Dict, List
from fastapi import Request, HTTPException, status
from core.config import Config

class BruteForceProtector:
    """Protect against brute force attacks on API keys."""
    
    def __init__(self, config: Config):
        self.enabled = False
        sec = config.get("security", {}).get("brute_force", {})
        if not sec.get("enabled", False):
            return
            
        self.enabled = True
        self.max_failures = int(sec.get("max_failures", 10))
        self.window_seconds = int(sec.get("window_seconds", 60))
        self.ban_seconds = int(sec.get("ban_seconds", 300))
        self.trust_proxy = bool(sec.get("trust_proxy_headers", False))
        
        # File paths
        # Ensures ban_file is an absolute path
        raw_ban_file = sec.get("ban_file", "banned_ips.json")
        self.ban_file = os.path.abspath(raw_ban_file)
        
        # State
        self._failures: Dict[str, List[float]] = {}  # ip -> list of timestamps
        self._bans: Dict[str, float] = {}            # ip -> ban_expiry_timestamp
        
        # Thread safety
        self._lock = threading.Lock()
        
        self._load_bans()
        
    def _load_bans(self):
        """Load bans from file."""
        if not self.ban_file or not os.path.exists(self.ban_file):
            return
        try:
            with open(self.ban_file, 'r') as f:
                data = json.load(f)
                now = time.time()
                # Only keep active bans
                with self._lock:
                    self._bans = {ip: exp for ip, exp in data.items() if exp > now}
        except Exception as e:
            logging.getLogger("rerank").error(f"Failed to load bans: {e}")

    def _save_bans(self):
        """Save bans to file."""
        if not self.ban_file:
            return
        try:
            with self._lock:
                bans_copy = self._bans.copy()
            with open(self.ban_file, 'w') as f:
                json.dump(bans_copy, f)
        except Exception as e:
            logging.getLogger("rerank").error(f"Failed to save bans: {e}")

    def get_client_ip(self, request: Request) -> str:
        """Get best guess for client IP."""
        if self.trust_proxy:
            # Check standard headers
            check_headers = ["x-forwarded-for", "x-real-ip"]
            for header in check_headers:
                val = request.headers.get(header)
                if val:
                    # In case of multiple IPs in XFF, usually the first one is the client
                    if "," in val:
                        return val.split(",")[0].strip()
                    return val.strip()
        
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    def check_ip(self, request: Request):
        """Check if IP is banned. Raises HTTPException if banned."""
        if not self.enabled:
            return

        ip = self.get_client_ip(request)
        now = time.time()
        
        # Check if banned
        ban_expired = False
        with self._lock:
            if ip in self._bans:
                expiry = self._bans[ip]
                if now < expiry:
                    # Still banned
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN, 
                        detail=f"Access denied: IP banned due to too many failed attempts. Try again later."
                    )
                else:
                    # Ban expired
                    del self._bans[ip]
                    ban_expired = True
        
        # Save bans outside the lock to avoid holding lock during I/O
        if ban_expired:
            self._save_bans()

    def record_failure(self, request: Request):
        """Record a failed login attempt."""
        if not self.enabled:
            return
            
        ip = self.get_client_ip(request)
        now = time.time()
        
        should_ban = False
        with self._lock:
            # Clean old failures for this IP
            if ip not in self._failures:
                self._failures[ip] = []
            
            # Keep only failures within window
            self._failures[ip] = [t for t in self._failures[ip] if now - t < self.window_seconds]
            
            # Add new failure
            self._failures[ip].append(now)
            
            # Check if threshold reached
            if len(self._failures[ip]) >= self.max_failures:
                should_ban = True
        
        # Ban outside the lock to avoid holding lock during I/O
        if should_ban:
            self.ban_ip(ip)

    def ban_ip(self, ip: str):
        """Ban an IP."""
        expiry = time.time() + self.ban_seconds
        with self._lock:
            self._bans[ip] = expiry
            # Also clear failures so they don't immediately re-ban after expiry if they fail once
            if ip in self._failures:
                del self._failures[ip]
        
        # Save bans outside the lock to avoid holding lock during I/O
        self._save_bans()
        logging.getLogger("rerank").warning(f"Banned IP {ip} for {self.ban_seconds}s after {self.max_failures} failed attempts")

    def record_success(self, request: Request):
        """Optional: clear failures on success?"""
        pass
