#!/usr/bin/env python3
# Author: Kaleb Austgen
# Date: 10/21/25
# Purpose:
"""

Concrete example script (non-CLI):
- Configure COVER, SECRET, STEGO, EXTRACTED, PASSWORD at top.
- Run the script; it will embed then extract (using the provided password).
- Optional stegcracker crack example is commented out.

Requirements:
- steghide installed and in PATH.
- (optional) stegcracker installed for cracking attempts.
"""

import subprocess
import shlex
import shutil
import os
import sys
import logging
import tempfile
from pathlib import Path

# Class of Steg_Gen
# Inputs: 
#   cover: File path of item that the secret will be hidden in
#   secret: File path of the secret file insert
#   stego: Output cover that contains the secret
#   extracted: Secret from setgo
#   password: Encryption password
class Steg_Gen():
    def __init__(self):
        # module logger
        self.logger = logging.getLogger(__name__)
        return

    # Called duing Embed to ensure steghide is installed
    # Returns error if not installed
    def _check_tool(self, name):
        if shutil.which(name) is None:
            self.logger.error("Required tool '%s' not found in PATH. Please install it.", name)
            return False
        return True
    
    # Takes commmand list and runs it through subprocess
    # Takes a command list
    # Returns a runtime code, the output, and an error if produced
    def _run(self, cmd_list, capture_output=False):
        try:
            # run with no stdin to avoid interactive prompts; add a timeout so a hung child
            # doesn't block the whole script indefinitely
            proc = subprocess.run(
                cmd_list,
                capture_output=capture_output,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=30,
            )
            return proc.returncode, proc.stdout if capture_output else "", proc.stderr if capture_output else ""
        except Exception as e:
            # If it's a timeout specifically, return a distinct error string
            if isinstance(e, subprocess.TimeoutExpired):
                self.logger.error("Command timed out after 30s: %s", ' '.join(cmd_list))
                return -2, "", f"Command timed out after 30s: {' '.join(cmd_list)}"
            return -1, "", str(e)
        
    # Runs a specific seteghide command to embed the secret in the cover, output the stego and encrypt it with a password
    # And do so quietly
    def embed(self, cover, secret, stego, password):
        if not self._check_tool("steghide"):
            raise SystemExit(1)
        # Ensure cover and secret exist
        for p in (cover, secret):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Required file not found: {p}")
        # Remove existing stego target to avoid steghide asking to overwrite interactively
        try:
            if os.path.exists(stego):
                os.remove(stego)
        except Exception:
            # ignore errors removing the file and let steghide handle it
            pass
        cmd = [
            "steghide", "embed",
            "-cf", cover,
            "-ef", secret,
            "-sf", stego,
            "-p", password,
            "-q"
        ]

        # Runs the command and receives a runtimecode, the output, and an error (or "" if there was none)
        rc, out, err = self._run(cmd, capture_output=True)
        if rc == 0:
            pass
            #self.logger.info("Embedded '%s' into '%s'.", secret, stego)
        else:
            if rc == -2:
                self.logger.warning("steghide embed timed out: %s", err)
            else:
                self.logger.warning("steghide embed failed (rc=%s). Output: %s  Err: %s", rc, out, err)
            raise RuntimeError(f"steghide embed failed with return code {rc}")
        
    def _extract(self, stego: str, password: str, out_path: str):
        if not self._check_tool("steghide"):
            raise SystemExit(1)
        # Remove existing output file to avoid steghide interactive overwrite prompt
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        # steghide by default writes the original filename; -xf forces output
        cmd = [
            "steghide", "extract",
            "-sf", stego,
            "-p", password,
            "-xf", out_path,
            "-q"
        ]
        rc, out, err = self._run(cmd, capture_output=True)
        if rc == 0:
            pass
            #self.logger.info("Extracted embedded file to '%s'.", out_path)
        else:
            if rc == -2:
                self.logger.warning("steghide extract timed out: %s", err)
            else:
                self.logger.warning("steghide extract failed (rc=%s). Output: %s  Err: %s", rc, out, err)
            raise RuntimeError(f"steghide extract failed with return code {rc}")
        
    # Inputs: 
    #   cover: File path of item that the secret will be hidden in
    #   secret: File path of the secret file insert
    #   stego: Output cover that contains the secret
    #   extracted: Secret from setgo
    #   password: Encryption password
    def create(self, cover: str, secret: str, stego: str, extracted: str, password: str):
        """Using a predefined steghide command will generate a stegnography file alongside an input
        and output. Takes a file that will contain the secret, a file that is the secret, and outputs what looks
        like the original cover but now contains a secret, will extract the secret and uses a password to encrypt it"""

        # self.logger.info("Cover: %s", cover)
        # self.logger.info("Secret: %s", secret)
        # self.logger.info("Stego out: %s", stego)
        # self.logger.info("Extracted: %s", extracted)

        # Embed
        try:
            self.embed(cover, secret, stego, password)
        # Exit early if there is an error (e.g., cover file too small)
        except (SystemExit, Exception) as e:
            self.logger.warning("Embedding failed, skipping this file: %s", e)
            return 

        # Extract using same password to a temporary file so we don't persist recovered secrets
        temp_path = None
        try:
            # create a temp file path (closed immediately) — suffix can be .pdf by default
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_path = tf.name
            tf.close()

            self._extract(stego, password, temp_path)

            # 3) Quick verify: check extracted file size > 0 and matches original size
            if temp_path and os.path.isfile(temp_path):
                try:
                    orig_size = os.path.getsize(secret)
                    ext_size = os.path.getsize(temp_path)
                    #self.logger.info("original size=%d bytes, extracted size=%d bytes", orig_size, ext_size)
                    if orig_size == ext_size:
                        pass
                        #self.logger.info("sizes match — likely successful byte-for-byte extraction.")
                    else:
                        self.logger.warning("sizes differ — inspect the extracted file.")
                except Exception as e:
                    self.logger.warning("Could not compare sizes of secret and extracted file: %s", e)
            else:
                self.logger.warning("extracted file not found (temp extraction failed).")
        finally:
            # Always try to remove the temporary extracted file so no recovered secrets remain on disk
            if temp_path:
                try:
                    os.remove(temp_path)
                    self.logger.debug("Removed temporary extracted file %s", temp_path)
                except Exception:
                    # best-effort removal only
                    self.logger.debug("Failed to remove temporary extracted file %s", temp_path)

