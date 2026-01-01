import time
import json
import os
import logging
import threading
from typing import Dict, List
from fastapi import Request, HTTPException, status
from filelock import FileLock
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
        self.lock_file = self.ban_file + ".lock"
        
        # State
        self._failures: Dict[str, List[float]] = {}  # ip -> list of timestamps
        self._bans: Dict[str, float] = {}            # ip -> ban_expiry_timestamp
        self._last_mtime = 0.0
        
        # Thread safety for in-memory state
        # Use RLock (reentrant lock) to allow nested acquisitions by the same thread
        self._memory_lock = threading.RLock()
        
        self._reload_if_needed()
        
    def _reload_if_needed(self):
        """Reload bans if file has changed."""
        if not self.ban_file or not os.path.exists(self.ban_file):
            return
            
        try:
            mtime = os.path.getmtime(self.ban_file)
            # Thread-safe check of _last_mtime
            with self._memory_lock:
                if mtime > self._last_mtime:
                    # File changed, reload under lock to ensure we don't read partial write
                    lock = FileLock(self.lock_file)
                    with lock.acquire(timeout=5):
                        self._load_from_disk()  # RLock allows nested acquisition
                        self._last_mtime = mtime
        except Exception as e:
            # Don't crash on locking/loading error, just log
            logging.getLogger("rerank").error(f"Failed to reload bans: {e}")

    def _load_from_disk(self):
        """Internal load without checks (assumes locked or safe)."""
        if not os.path.exists(self.ban_file):
            return
        try:
            with open(self.ban_file, 'r') as f:
                data = json.load(f)
                now = time.time()
                # Thread-safe update of in-memory state
                with self._memory_lock:
                    self._bans = {ip: exp for ip, exp in data.items() if exp > now}
        except Exception as e:
            logging.getLogger("rerank").error(f"Failed to parse ban file: {e}")

    def _load_bans(self):
        """Deprecated: use _reload_if_needed."""
        self._reload_if_needed()

    def _save_bans(self):
        """Save bans to file.
        
        Acquires memory lock to read current bans, then writes them to file.
        Note: Caller is responsible for acquiring file lock before calling this method
        to prevent concurrent file access from other processes.
        """
        if not self.ban_file:
            return
        
        try:
            # Read current bans under memory lock
            with self._memory_lock:
                bans_to_save = dict(self._bans)
            
            # Write local copy to file
            with open(self.ban_file, 'w') as f:
                json.dump(bans_to_save, f)
            # Update mtime tracking so we don't reload our own write
            with self._memory_lock:
                self._last_mtime = os.path.getmtime(self.ban_file)
        except Exception as e:
            logging.getLogger("rerank").error(f"Failed to save bans: {e}")
    
    def _save_bans_unlocked(self):
        """Save bans to file (assumes file-locked AND memory-locked).
        
        Note: Caller MUST hold self._memory_lock before calling this method.
        This is only used internally where we already hold the lock to avoid
        re-acquiring it (which would work with RLock but is unnecessary overhead).
        """
        if not self.ban_file:
            return
        
        try:
            # Caller holds memory lock - direct access to self._bans is safe
            with open(self.ban_file, 'w') as f:
                json.dump(self._bans, f)
            # Update mtime tracking (caller holds lock)
            self._last_mtime = os.path.getmtime(self.ban_file)
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

        # Check for update from other workers
        self._reload_if_needed()

        ip = self.get_client_ip(request)
        now = time.time()
        
        # Check if banned (thread-safe read)
        with self._memory_lock:
            if ip not in self._bans:
                return  # Not banned, exit early
            expiry = self._bans[ip]
        
        # Process ban status outside the lock
        if now < expiry:
            # Still banned
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail=f"Access denied: IP banned due to too many failed attempts. Try again later."
            )
        else:
            # Ban expired - Remove atomically
            try:
                lock = FileLock(self.lock_file)
                with lock.acquire(timeout=5):
                    self._load_from_disk()  # Refresh state inside lock (acquires memory lock internally)
                    with self._memory_lock:
                        if ip in self._bans:  # Double check after reload
                            del self._bans[ip]
                            self._save_bans_unlocked()  # Use unlocked version since we hold memory lock
            except Exception as e:
                logging.getLogger("rerank").error(f"Failed to remove expired ban: {e}")

    def record_failure(self, request: Request):
        """Record a failed login attempt."""
        if not self.enabled:
            return
            
        ip = self.get_client_ip(request)
        now = time.time()
        
        # Thread-safe update of failures
        with self._memory_lock:
            # Clean old failures for this IP
            if ip not in self._failures:
                self._failures[ip] = []
            
            # Keep only failures within window
            self._failures[ip] = [t for t in self._failures[ip] if now - t < self.window_seconds]
            
            # Add new failure
            self._failures[ip].append(now)
            
            # Check if threshold reached
            failure_count = len(self._failures[ip])
        
        # Ban outside the lock to avoid holding it during file I/O
        if failure_count >= self.max_failures:
            self.ban_ip(ip)

    def ban_ip(self, ip: str):
        """Ban an IP."""
        expiry = time.time() + self.ban_seconds
        
        try:
            lock = FileLock(self.lock_file)
            with lock.acquire(timeout=5):
                # 1. Load latest state (critical for concurrency)
                self._load_from_disk()
                
                # 2. Update state (thread-safe)
                with self._memory_lock:
                    self._bans[ip] = expiry
                    # 3. Save (use unlocked version since we hold memory lock)
                    self._save_bans_unlocked()
                
            logging.getLogger("rerank").warning(f"Banned IP {ip} for {self.ban_seconds}s after {self.max_failures} failed attempts")
        except Exception as e:
            logging.getLogger("rerank").error(f"Failed to execute ban: {e}")
            
        # Also clear failures so they don't immediately re-ban after expiry if they fail once
        with self._memory_lock:
            if ip in self._failures:
                del self._failures[ip]
