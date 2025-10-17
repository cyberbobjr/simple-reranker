#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version information for Reranking & Embedding Service
"""

__version__ = "1.2.0"
__version_info__ = (1, 2, 0)

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get version as tuple (major, minor, patch)."""
    return __version_info__