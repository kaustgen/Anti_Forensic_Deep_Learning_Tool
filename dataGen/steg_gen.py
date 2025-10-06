#!/usr/bin/env python3
"""
simple_steg_embed_extract.py

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

# Class of Steg_Gen
# Inputs: 
#   cover: File path of item that the secret will be hidden in
#   secret: File path of the secret file insert
#   stego: Output cover that contains the secret
#   extracted: Secret from setgo
#   password: Encryption password
class Steg_Gen():
    def __init__(self):
        return 

    # Called duing Embed to ensure steghide is installed
    # Returns error if not installed
    def _check_tool(self, name):
        if shutil.which(name) is None:
            print(f"[ERROR] Required tool '{name}' not found in PATH. Please install it.")
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
                return -2, "", f"Command timed out after 30s: {' '.join(cmd_list)}"
            return -1, "", str(e)
        
    # Runs a specific seteghide command to embed the secret in the cover, output the stego and encrypt it with a password
    # And do so quietly
    def _embed(self, cover, secret, stego, password):
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

        # Runs the command and recieves a runtimecode, the output, and an error (or "" if there was none)
        rc, out, err = self._run(cmd, capture_output=True)
        if rc == 0:
            print(f"[OK] Embedded '{secret}' into '{stego}'.")
        else:
            if rc == -2:
                print(f"[ERROR] steghide embed timed out: {err}")
            else:
                print(f"[ERROR] steghide embed failed (rc={rc}).")
                print(err or out)
            raise SystemExit(1)
        
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
            print(f"[OK] Extracted embedded file to '{out_path}'.")
        else:
            if rc == -2:
                print(f"[ERROR] steghide extract timed out: {err}")
            else:
                print(f"[ERROR] steghide extract failed (rc={rc}).")
                print(err or out)
            raise SystemExit(1)
        
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

        #print("== steghide automation (concrete example) ==")
        print(f"Cover:    {cover}")
        print(f"Secret:   {secret}")
        print(f"Stego out:{stego}")
        print(f"Extracted:{extracted}")
        print("Password: (hidden in script variable)")

        # 1) Embed
        self._embed(cover, secret, stego, password)

        # 2) Extract using same password
        self._extract(stego, password, extracted)

        # 3) Quick verify: check extracted file size > 0 and matches original size
        if os.path.isfile(extracted):
            orig_size = os.path.getsize(secret)
            ext_size = os.path.getsize(extracted)
            print(f"[INFO] original size={orig_size} bytes, extracted size={ext_size} bytes")
            if orig_size == ext_size:
                print("[VERIFY] sizes match — likely successful byte-for-byte extraction.")
            else:
                print("[VERIFY] sizes differ — inspect the extracted file.")
        else:
            print("[VERIFY] extracted file not found.")

