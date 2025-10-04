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
    def check_tool(self, name):
        if shutil.which(name) is None:
            print(f"[ERROR] Required tool '{name}' not found in PATH. Please install it.")
            return False
        return True
    
    # Takes commmand list and runs it through subprocess
    # Takes a command list
    # Returns a runtime code, the output, and an error if produced
    def run(self, cmd_list, capture_output=False):
        try:
            proc = subprocess.run(cmd_list, capture_output=capture_output, text=True, check=False)
            return proc.returncode, proc.stdout if capture_output else "", proc.stderr if capture_output else ""
        except Exception as e:
            return -1, "", str(e)
        
    # Runs a specific seteghide command to embed the secret in the cover, output the stego and encrypt it with a password
    # And do so quietly
    def embed(self, cover, secret, stego, password):
        if not self.check_tool("steghide"):
            raise SystemExit(1)
        # Ensure cover and secret exist
        for p in (cover, secret):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Required file not found: {p}")
        cmd = [
            "steghide", "embed",
            "-cf", cover,
            "-ef", secret,
            "-sf", stego,
            "-p", password,
            "-q"
        ]

        # Runs the command and recieves a runtimecode, the output, and an error (or "" if there was none)
        rc, out, err = self.run(cmd, capture_output=True)
        if rc == 0:
            print(f"[OK] Embedded '{secret}' into '{stego}'.")
        else:
            print(f"[ERROR] steghide embed failed (rc={rc}).")
            print(err or out)
            raise SystemExit(1)
        
    def extract(self, stego, password, out_path):
        if not self.check_tool("steghide"):
            raise SystemExit(1)
        # steghide by default writes the original filename; -xf forces output
        cmd = [
            "steghide", "extract",
            "-sf", stego,
            "-p", password,
            "-xf", out_path,
            "-q"
        ]
        rc, out, err = self.run(cmd, capture_output=True)
        if rc == 0:
            print(f"[OK] Extracted embedded file to '{out_path}'.")
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
    def create(self, cover, secret, stego, extracted, password):
        print("== steghide automation (concrete example) ==")
        print(f"Cover:    {cover}")
        print(f"Secret:   {secret}")
        print(f"Stego out:{stego}")
        print(f"Extracted:{extracted}")
        print("Password: (hidden in script variable)")

        # 1) Embed
        self.embed(cover, secret, stego, password)

        # 2) Extract using same password
        self.extract(stego, password, extracted)

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

