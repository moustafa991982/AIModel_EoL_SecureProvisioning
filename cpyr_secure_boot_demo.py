#!/usr/bin/env python3
"""
CPYR on Jetson Orin — secure manufacturing and boot, reference simulation (v2)
==============================================================================

Mirrors the "Jetson Secure Provisioning Flow" page step for step:

  Phase 0  S1–S17   OEM secure site: root keys, FSKP keys from NVIDIA, fuse blob,
                    UEFI PK/KEK/db + signed .auth updates, PKC-signed boot chain,
                    dm-verity CPYR code partition, LUKS (dm-crypt) data partition,
                    encrypted + db-signed boot entry, EKB, golden PCRs, factory package
  Phase 1  S18–S25  factory floor: RCM, encrypted fuse blob, BootROM decrypts + burns,
                    read-back check, security-mode fuse last
  Phase 2  S26–S27  mass-flash, enrol .auth variables (setup mode → user mode)
  Phase 3  S28–S33  first boot: LUKS unlock with the generic passphrase, re-key to a
                    per-device passphrase, fTPM EK, EK certificate, registration
  Phase 4  S34–S40  attestation, end-of-line functional test, record, ship

  Boot stages B1–B11 run inside every boot, ending with CPYR attested and sending
  its first analytics over an attestation-gated session.

The cryptography is real: ECDSA P-256 (PKC chain, fTPM), RSA-2048 + PKCS#7 (UEFI .auth,
as efitools produces), RSA-OAEP (FSKP key wrapping), AES-256-GCM, AES-CTR + HMAC
(fuse blob, encrypt-then-MAC), AES-256-XTS per 4 KiB sector (LUKS data), scrypt (keyslot
KDF), NIST SP 800-108 KDF, SHA-256 Merkle tree (dm-verity). The NVIDIA components
(BootROM, PSC, MB1/MB2, OP-TEE, fTPM, UEFI) are MODELLED in Python; derivation labels
not published by NVIDIA are marked "(sim)".

Usage
  python3 cpyr_secure_boot_demo.py                     # manufacture one unit end to end
  python3 cpyr_secure_boot_demo.py --list              # list scenarios
  python3 cpyr_secure_boot_demo.py --attack all        # every scenario + summary
  python3 cpyr_secure_boot_demo.py --real-cpyr ...     # use the trained CPYR conv encoder
  python3 cpyr_secure_boot_demo.py --export DIR        # write the provisioning package
  python3 cpyr_secure_boot_demo.py --step              # pause between stages

Requires: python3 >= 3.9, cryptography >= 41, numpy (torch only for --real-cpyr)
"""
import argparse
import copy
import datetime
import hashlib
import hmac
import io
import json
import os
import struct
import sys
import tarfile
import time
import uuid

import numpy as np
from cryptography import x509
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import pkcs7
from cryptography.x509.oid import NameOID

# ============================================================================
# Output
# ============================================================================
USE_COLOR = sys.stdout.isatty()
STEP_MODE = False
QUIET = False


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def banner(title):
    print("\n" + _c("1;36", "=" * 78) + "\n" + _c("1;36", f"  {title}") + "\n" + _c("1;36", "=" * 78))


def stage(tag, title):
    if QUIET:
        return
    if STEP_MODE:
        input(_c("2", "  [enter] "))
    print("\n" + _c("1;34", f"▶ [{tag}] {title}"))


def ok(m):
    QUIET or print("   " + _c("32", "✔ ") + m)


def bad(m):
    QUIET or print("   " + _c("1;31", "✘ ") + m)


def info(m):
    QUIET or print("   " + _c("2", "· ") + m)


def warn(m):
    QUIET or print("   " + _c("33", "⚠ ") + m)


def short(b, n=8):
    return (b.hex() if isinstance(b, (bytes, bytearray)) else str(b))[: 2 * n] + "…"


class Quiet:
    def __enter__(self):
        global QUIET
        self.prev, QUIET = QUIET, True

    def __exit__(self, *a):
        global QUIET
        QUIET = self.prev


class BootHalt(Exception):
    pass


class TAError(Exception):
    pass


class EfiSecurityViolation(Exception):
    pass


class VerityError(Exception):
    pass


class FuseBlobError(Exception):
    pass


# ============================================================================
# Crypto primitives
# ============================================================================
P256_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551


def sha256(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def kdf(key, label, context, length=32):
    """NIST SP 800-108 KDF, counter mode, HMAC-SHA256 PRF."""
    out, i = b"", 1
    while len(out) < length:
        out += hmac.new(key, struct.pack(">I", i) + label + b"\x00" + context
                        + struct.pack(">I", length * 8), hashlib.sha256).digest()
        i += 1
    return out[:length]


def ec_key():
    return ec.generate_private_key(ec.SECP256R1())


def ec_from_secret(secret):
    return ec.derive_private_key(int.from_bytes(secret, "big") % (P256_N - 1) + 1, ec.SECP256R1())


def pub(k):
    return k.public_key().public_bytes(serialization.Encoding.X962,
                                       serialization.PublicFormat.UncompressedPoint)


def ec_sign(k, data):
    return k.sign(data, ec.ECDSA(hashes.SHA256()))


def ec_verify(pub_bytes, sig, data):
    try:
        ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), pub_bytes).verify(
            sig, data, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError):
        return False


def gcm_enc(key, pt, aad):
    n = os.urandom(12)
    return n + AESGCM(key).encrypt(n, pt, aad)


def gcm_dec(key, blob, aad):
    return AESGCM(key).decrypt(blob[:12], blob[12:], aad)


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def entropy(b):
    c = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    p = c[c > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


def make_cert(cn, key, issuer_cn, issuer_key, ca):
    t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    nm = lambda s: x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, s),
                              x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Demo OEM")])
    return (x509.CertificateBuilder().subject_name(nm(cn)).issuer_name(nm(issuer_cn))
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(t0).not_valid_after(t0 + datetime.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
            .sign(issuer_key, hashes.SHA256()))


# ============================================================================
# UEFI authenticated variables: EFI_SIGNATURE_LIST (.esl) and .auth
# ============================================================================
G = lambda s: uuid.UUID(s).bytes_le
EFI_GLOBAL_VARIABLE = G("8be4df61-93ca-11d2-aa0d-00e098032b8c")          # PK, KEK
EFI_IMAGE_SECURITY_DATABASE = G("d719b2cb-3d3a-4596-a3bc-dad00e67656f")  # db, dbx
EFI_CERT_X509 = G("a5c059a1-94e4-4aa7-87b5-ab155c2bf072")
EFI_CERT_SHA256 = G("c1c41626-504c-4092-aca9-41f936934328")
EFI_CERT_TYPE_PKCS7 = G("4aafd29d-68df-49ee-8aa9-347d375665a7")
OEM_OWNER = G("3f1c9a2e-5b7d-4c11-9e8a-0d2f6b4a7c55")                   # SignatureOwner (ours)
VENDOR = {"PK": EFI_GLOBAL_VARIABLE, "KEK": EFI_GLOBAL_VARIABLE,
          "db": EFI_IMAGE_SECURITY_DATABASE, "dbx": EFI_IMAGE_SECURITY_DATABASE}
AUTHORITY = {"PK": "PK", "KEK": "PK", "db": "KEK", "dbx": "KEK"}
ATTR = 0x01 | 0x02 | 0x04 | 0x20      # NV | BS | RT | TIME_BASED_AUTHENTICATED_WRITE_ACCESS
APPEND = 0x40


def esl(entries, sig_type):
    """EFI_SIGNATURE_LIST: one type, fixed-size entries of SignatureOwner GUID + data."""
    size = 16 + len(entries[0])
    body = b"".join(OEM_OWNER + e for e in entries)
    return sig_type + struct.pack("<III", 28 + len(body), 0, size) + body


def esl_parse(blob):
    out, off = [], 0
    while off < len(blob):
        typ = blob[off:off + 16]
        lsz, hsz, ssz = struct.unpack("<III", blob[off + 16:off + 28])
        p = off + 28 + hsz
        while p < off + lsz:
            out.append((typ, blob[p:p + 16], blob[p + 16:p + ssz]))
            p += ssz
        off += lsz
    return out


def efi_time(ts):
    t = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    return struct.pack("<HBBBBBBIhBB", t.year, t.month, t.day, t.hour, t.minute, t.second,
                       0, 0, 0, 0, 0)


def auth_signed_bytes(name, vendor, attrs, tstamp, data):
    return name.encode("utf-16-le") + vendor + struct.pack("<I", attrs) + tstamp + data


def build_auth(name, data, signer_key, signer_cert, ts, append=False):
    """EFI_VARIABLE_AUTHENTICATION_2 || data, signed like sign-efi-sig-list."""
    attrs = ATTR | (APPEND if append else 0)
    tstamp = efi_time(ts)
    p7 = pkcs7.PKCS7SignatureBuilder().set_data(
        auth_signed_bytes(name, VENDOR[name], attrs, tstamp, data)).add_signer(
        signer_cert, signer_key, hashes.SHA256()).sign(
        serialization.Encoding.DER, [pkcs7.PKCS7Options.DetachedSignature,
                                     pkcs7.PKCS7Options.Binary, pkcs7.PKCS7Options.NoAttributes])
    wc = struct.pack("<IHH", 24 + len(p7), 0x0200, 0x0EF1) + EFI_CERT_TYPE_PKCS7 + p7
    return {"name": name, "attrs": attrs, "blob": tstamp + wc + data}


def _der(b, o):
    t, l = b[o], b[o + 1]
    o += 2
    if l & 0x80:
        n = l & 0x7F
        l = int.from_bytes(b[o:o + n], "big")
        o += n
    return t, o, l


def _children(b, o, l):
    end, out = o + l, []
    while o < end:
        t, vo, vl = _der(b, o)
        out.append((t, vo, vl))
        o = vo + vl
    return out


def pkcs7_signature(p7):
    """Minimal DER walk: ContentInfo → SignedData → signerInfos[0].signature."""
    _, o, l = _der(p7, 0)
    content = _children(p7, o, l)[1]                       # [0] EXPLICIT SignedData
    _, so, sl = _der(p7, content[1])
    sd = _children(p7, so, sl)
    signer_set = [c for c in sd if c[0] == 0x31][-1]       # last SET = signerInfos
    si = _children(p7, signer_set[1], signer_set[2])[0]
    fields = _children(p7, si[1], si[2])
    sig = [f for f in fields if f[0] == 0x04][-1]
    return p7[sig[1]:sig[1] + sig[2]]


def parse_auth(blob):
    tstamp = blob[:16]
    dw_len = struct.unpack("<I", blob[16:20])[0]
    p7 = blob[16 + 24:16 + dw_len]
    return tstamp, p7, blob[16 + dw_len:]


class UefiVarStore:
    """Firmware side of SetVariable() for PK/KEK/db/dbx."""

    def __init__(self):
        self.vars = {n: b"" for n in ("PK", "KEK", "db", "dbx")}
        self.ts = {n: b"\0" * 16 for n in self.vars}

    @property
    def setup_mode(self):
        return not self.vars["PK"]

    def certs(self, name):
        return [x509.load_der_x509_certificate(d) for t, _, d in esl_parse(self.vars[name])
                if t == EFI_CERT_X509]

    def hashes(self, name):
        return [d for t, _, d in esl_parse(self.vars[name]) if t == EFI_CERT_SHA256]

    def set_variable(self, name, auth):
        tstamp, p7, data = parse_auth(auth["blob"])
        msg = auth_signed_bytes(name, VENDOR[name], auth["attrs"], tstamp, data)
        sig = pkcs7_signature(p7)
        signer_list = AUTHORITY[name]
        if name == "PK" and self.setup_mode:
            trusted = [c for c in pkcs7.load_der_pkcs7_certificates(p7)]   # self-signed PK
        else:
            trusted = self.certs(signer_list)
        good = False
        for c in trusted:
            try:
                c.public_key().verify(sig, msg, padding.PKCS1v15(), hashes.SHA256())
                good = True
                break
            except InvalidSignature:
                pass
        if not good:
            raise EfiSecurityViolation(f"SetVariable({name}) rejected: signature does not verify "
                                       f"against {signer_list}")
        append = bool(auth["attrs"] & APPEND)
        if not append and tstamp <= self.ts[name]:
            raise EfiSecurityViolation(f"SetVariable({name}) rejected: timestamp not newer than "
                                       "the stored one (replay)")
        self.vars[name] = self.vars[name] + data if append else data
        if not append:
            self.ts[name] = tstamp

    def snapshot(self):
        return {n: self.vars[n].hex() for n in self.vars}


# ============================================================================
# dm-verity, partitions, LUKS (dm-crypt)
# ============================================================================
BLOCK = 4096


def verity_levels(level0, salt):
    levels, level = [level0], level0
    while len(level) > 1:
        packed = b"".join(level)
        level = [sha256(salt, packed[i:i + BLOCK].ljust(BLOCK, b"\0"))
                 for i in range(0, len(packed), BLOCK)]
        levels.append(level)
    return levels


def verity_format(data, salt):
    level0 = [sha256(salt, data[i:i + BLOCK]) for i in range(0, len(data), BLOCK)]
    return level0, verity_levels(level0, salt)[-1][0]


def verity_open(data, level0, salt, trusted_root):
    if verity_levels(list(level0), salt)[-1][0] != trusted_root:
        raise VerityError("hash tree root ≠ trusted root carried in the signed initrd")
    for i in range(0, len(data), BLOCK):
        if sha256(salt, data[i:i + BLOCK]) != level0[i // BLOCK]:
            raise VerityError(f"data block {i // BLOCK} hash mismatch → I/O error")
    return data


def pack(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for n, d in files.items():
            ti = tarfile.TarInfo(n)
            ti.size, ti.mtime, ti.mode = len(d), 0, 0o444
            tar.addfile(ti, io.BytesIO(d))
    raw = buf.getvalue()
    return raw + b"\0" * (-len(raw) % BLOCK)


def unpack(raw):
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def _xts(dek, data, encrypt):
    out = bytearray()
    for s in range(0, len(data), BLOCK):
        c = Cipher(algorithms.AES(dek), modes.XTS((s // BLOCK).to_bytes(16, "little")))
        op = c.encryptor() if encrypt else c.decryptor()
        out += op.update(data[s:s + BLOCK]) + op.finalize()
    return bytes(out)


def _slot_key(passphrase, salt):
    return hashlib.scrypt(passphrase, salt=salt, n=2 ** 14, r=8, p=1, dklen=32)  # LUKS2: argon2id


def luks_format(plain, passphrase, disk_uuid):
    dek = os.urandom(64)                                   # AES-256-XTS master key (DEK)
    salt = os.urandom(16)
    hdr = {"uuid": disk_uuid, "cipher": "aes-xts-plain64", "key_bits": 512,
           "digest": sha256(b"luks-digest", dek).hex(),
           "keyslots": {"0": {"salt": salt.hex(),
                              "wrapped_dek": gcm_enc(_slot_key(passphrase, salt), dek, b"slot").hex()}}}
    return hdr, _xts(dek, plain, True)


def luks_unlock(hdr, passphrase):
    for sid, s in hdr["keyslots"].items():
        try:
            dek = gcm_dec(_slot_key(passphrase, bytes.fromhex(s["salt"])),
                          bytes.fromhex(s["wrapped_dek"]), b"slot")
            if sha256(b"luks-digest", dek).hex() == hdr["digest"]:
                return dek, sid
        except InvalidTag:
            continue
    raise BootHalt("cryptsetup: no keyslot opens with this passphrase")


def luks_add_key(hdr, dek, new_passphrase):
    sid = str(max(int(k) for k in hdr["keyslots"]) + 1)
    salt = os.urandom(16)
    hdr["keyslots"][sid] = {"salt": salt.hex(),
                            "wrapped_dek": gcm_enc(_slot_key(new_passphrase, salt), dek, b"slot").hex()}
    return sid


# ============================================================================
# CPYR detector (stand-in) and real-model runtime
# ============================================================================
DETECTOR_SRC = '''
VERSION = "{version}"
import numpy as np, json
def to_bits(frames):
    return np.unpackbits(np.asarray(frames, dtype=np.uint8), axis=1).astype(np.float32)
def score(model, frames):
    p = np.asarray(model["p"], dtype=np.float32); b = to_bits(frames)
    bce = -(b * np.log(p) + (1 - b) * np.log(1 - p)).mean(axis=1)
    w = model["window"]; n = len(bce) // w
    return bce[: n * w].reshape(n, w).mean(axis=1)
def load(model_blob, files):
    return json.loads(model_blob)
def count_windows(model, frames):
    return len(frames) // model["window"]
def detect(model, frames):
    s = score(model, frames)
    thr = model["threshold"] * {threshold_bug}
    return [(i, float(v)) for i, v in enumerate(s) if v > thr]
'''

REAL_RUNTIME_SRC = '''
VERSION = "{version}"
import io, json, torch, torch.nn as nn
def load(model_blob, files):
    ns = {{"torch": torch, "nn": nn}}
    exec(files["basemodel.py"], ns); exec(files["conv_encoder.py"], ns)
    net = ns["conv_enocder"](); net.load_state_dict(torch.load(io.BytesIO(model_blob), weights_only=True)); net.eval()
    return {{"net": net, "loss": ns["binary"](), "threshold": json.loads(files["manifest.json"])["threshold"]}}
def count_windows(model, frames):
    return len(frames)
def detect(model, frames):
    thr = model["threshold"] * {threshold_bug}
    x = torch.as_tensor(frames, dtype=torch.float32)
    with torch.no_grad():
        s = [float(model["loss"](model["net"](w.unsqueeze(0)), w.unsqueeze(0))) for w in x]
    return [(i, float(v)) for i, v in enumerate(s) if v > thr]
'''
REAL_CPYR = None


def detector_source(version):
    src = REAL_RUNTIME_SRC if REAL_CPYR else DETECTOR_SRC
    return src.format(version=version, threshold_bug="50.0" if version.startswith("1") else "1.0").encode()


def normal_frames(rng, n, t0=0):
    t = np.arange(t0, t0 + n)
    f = np.zeros((n, 8), dtype=np.int64)
    f[:, 0], f[:, 1] = 0x12, t % 16
    sp = (60 + 20 * np.sin(t / 50)).astype(int)
    f[:, 2], f[:, 3], f[:, 5], f[:, 6] = sp, (sp * 3) // 4, rng.integers(0, 4, n), 0xA5
    f[:, 7] = f[:, :7].sum(axis=1) % 256
    return f.astype(np.uint8)


def fuzz_frames(rng, n):
    return rng.integers(0, 256, (n, 8)).astype(np.uint8)


def train_standin(rng):
    ns = {}
    exec(detector_source("2.0"), ns)
    bits = ns["to_bits"](normal_frames(rng, 20000))
    m = {"p": [round(float(x), 4) for x in np.clip(bits.mean(0), 0.01, 0.99)], "window": 20, "threshold": 0.0}
    m["threshold"] = round(float(ns["score"](m, normal_frames(rng, 10000, 20000)).max()) * 1.15, 4)
    return m


def load_real_artifacts(path):
    rd = lambda n: open(os.path.join(path, n), "rb").read()
    meta = json.load(open(os.path.join(path, "train_meta.json")))
    w = rd("model.pt")
    if hashlib.sha256(w).hexdigest() != meta["weights_sha256"]:
        raise SystemExit("model.pt does not match train_meta.json — re-run train_cpyr_encoder.py")
    return {"weights": w, "meta": meta,
            "code": {"basemodel.py": rd("basemodel.py"), "conv_encoder.py": rd("conv_encoder.py")},
            "normal": np.load(os.path.join(path, "eval_normal.npy")),
            "fuzz": np.load(os.path.join(path, "eval_fuzz.npy"))}


# ============================================================================
# Measured boot helpers
# ============================================================================
PCRS = (0, 4, 7, 10)
PCR_NAMES = {0: "NVIDIA boot chain", 4: "UEFI boot entry", 7: "Secure Boot policy (PK/KEK/db/dbx)",
             10: "IMA (CPYR files)"}


def replay(log):
    p = {i: b"\0" * 32 for i in PCRS}
    for e in log:
        p[e["pcr"]] = sha256(p[e["pcr"]], bytes.fromhex(e["digest"]))
    return p


def policy_digest(p):
    return sha256(*[p[i] for i in PCRS])


# ============================================================================
# NVIDIA side (simulated): SKU ROM root key and FSKP key issuance
# ============================================================================
ROM_ROOT = sha256(b"T234 ROM root key (sim) - identical in every chip of the SKU")
OEM_CONFIG = b"OEM-DEMO-FSKP-CFG-01"


def fskp_keys(rom_root, oem_config):
    """FSKP_AK (authentication) / FSKP_EK (encryption), derived from the ROM root (sim labels)."""
    return kdf(rom_root, b"fskp-ak (sim)", oem_config), kdf(rom_root, b"fskp-ek (sim)", oem_config)


def nvidia_issue_fskp(oem_cert):
    ak, ek = fskp_keys(ROM_ROOT, OEM_CONFIG)
    oaep = padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    pk = oem_cert.public_key()
    return {"fskp_ak.bin.rsa_wrap": pk.encrypt(ak, oaep), "fskp_ek.bin.rsa_wrap": pk.encrypt(ek, oaep),
            "oem_config": OEM_CONFIG}


def fuse_blob_build(fuse_values, ak, ek):
    """Encrypt-then-MAC: AES-CTR with FSKP_EK, HMAC-SHA256 with FSKP_AK."""
    iv = os.urandom(16)
    enc = Cipher(algorithms.AES(ek), modes.CTR(iv)).encryptor()
    ct = iv + enc.update(canon(fuse_values)) + enc.finalize()
    return ct + hmac.new(ak, ct, hashlib.sha256).digest()


# ============================================================================
# Phase 0 — OEM secure site (S1–S17)
# ============================================================================
def build_release(version, k, model_blob, model_key, real, disk_uuid, generic_pass, payload_key):
    """S10, S12, S13 for one release."""
    detector = detector_source(version)
    man = {"name": "cpyr", "version": version, "model_aad": f"cpyr-model-v{version}",
           "detector_sha256": sha256(detector).hex(), "weights_sha256": sha256(model_blob).hex()}
    code = {"cpyr_detector.py": detector}
    if real:
        man.update({"threshold": real["meta"]["threshold"], "arch": "conv_enocder"})
        code.update(real["code"])
    code["manifest.json"] = canon(man)
    code_part = pack(code)
    salt = os.urandom(32)
    level0, root = verity_format(code_part, salt)
    model_enc = gcm_enc(model_key, model_blob, man["model_aad"].encode())       # layer 2
    luks_hdr, data_ct = luks_format(pack({"model.bin.enc": model_enc}), generic_pass, disk_uuid)  # layer 1
    kernel = f"Image: Linux 5.15-tegra (CPYR release {version})".encode()
    initrd = canon({"init": "nv-init + tee-supplicant + nvluks-srv-app + cryptsetup + veritysetup + cpyr-loader",
                    "cpyr_version": version, "verity_root": root.hex(), "verity_salt": salt.hex(),
                    "luks_uuid": disk_uuid})
    entry = {"name": f"Jetson Linux + CPYR v{version}",
             "kernel_enc": gcm_enc(payload_key, kernel, b"kernel").hex(),
             "initrd_enc": gcm_enc(payload_key, initrd, b"initrd").hex()}
    digest = sha256(canon(entry))
    return {"version": version, "code_files": code, "code_part": code_part, "verity_salt": salt,
            "verity_level0": level0, "verity_root": root, "luks_hdr": luks_hdr, "data_ct": data_ct,
            "model_enc": model_enc, "boot_entry": entry, "entry_digest": digest,
            "entry_sig": k["db"].sign(digest, padding.PKCS1v15(), hashes.SHA256())}


def prepare(verbose=True, seed=7):
    with (Quiet() if not verbose else _Null()):
        rng = np.random.default_rng(seed)
        stage("S1", "Generate root keys in the HSM (PKC, fTPM manufacturer CA, OEM_K1/K2)")
        k = {"pkc": ec_key(), "ca": ec_key()}
        secret = {"OEM_K1": os.urandom(32), "OEM_K2": os.urandom(32)}
        info("PKC = ECDSA P-256 boot-chain signing key; manufacturer CA = ECDSA P-256")
        ok("OEM_K1 / OEM_K2 generated — will only ever exist in the HSM and in fuses")

        stage("S2–S4", "Request FSKP keys from NVIDIA, unwrap in the HSM")
        k["oem_rsa"] = rsa.generate_private_key(65537, 3072)
        oem_cert = make_cert("Demo OEM FSKP request", k["oem_rsa"], "Demo OEM FSKP request", k["oem_rsa"], False)
        fp = oem_cert.fingerprint(hashes.SHA256()).hex()
        info(f"deliver oem_publickey.cer (RSA, SHA-256 fingerprint {fp[:16]}…, confirm out of band)")
        results = nvidia_issue_fskp(oem_cert)
        oaep = padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        fskp_ak = k["oem_rsa"].decrypt(results["fskp_ak.bin.rsa_wrap"], oaep)
        fskp_ek = k["oem_rsa"].decrypt(results["fskp_ek.bin.rsa_wrap"], oaep)
        ok("Results.zip: fskp_ak / fskp_ek .rsa_wrap unwrapped with the OEM RSA private key")

        stage("S5", "Build the fuse blob (encrypt-then-MAC with FSKP_EK / FSKP_AK)")
        pkc_hash = sha256(pub(k["pkc"]))
        fuse_values = {"OEM_K1": secret["OEM_K1"].hex(), "OEM_K2": secret["OEM_K2"].hex(),
                       "PKC_HASH": pkc_hash.hex(), "SecureBootMode": "PKC", "ODM_ID": "DEMO-01"}
        blob = fuse_blob_build(fuse_values, fskp_ak, fskp_ek)
        ok(f"fuse blob {len(blob)} B: AES-CTR(FSKP_EK) + HMAC-SHA256(FSKP_AK); only a BootROM can open it")

        stage("S6–S8", "UEFI PK / KEK / db keys, certificate chain, signed .auth updates")
        for n in ("PK", "KEK", "db"):
            k[n] = rsa.generate_private_key(65537, 2048)
        cert = {"PK": make_cert("Demo OEM UEFI PK", k["PK"], "Demo OEM UEFI PK", k["PK"], True)}
        cert["KEK"] = make_cert("Demo OEM UEFI KEK", k["KEK"], "Demo OEM UEFI PK", k["PK"], True)
        cert["db"] = make_cert("Demo OEM UEFI db signer", k["db"], "Demo OEM UEFI KEK", k["KEK"], False)
        ts0 = int(time.time())
        der = lambda c: c.public_bytes(serialization.Encoding.DER)
        auths = {"PK": build_auth("PK", esl([der(cert["PK"])], EFI_CERT_X509), k["PK"], cert["PK"], ts0),
                 "KEK": build_auth("KEK", esl([der(cert["KEK"])], EFI_CERT_X509), k["PK"], cert["PK"], ts0),
                 "db": build_auth("db", esl([der(cert["db"])], EFI_CERT_X509), k["KEK"], cert["KEK"], ts0)}
        ok("PK.auth (self-signed) · KEK.auth (signed by PK) · db.auth (signed by KEK)")
        info("each .auth = EFI_TIME + WIN_CERTIFICATE_UEFI_GUID(PKCS#7) + EFI_SIGNATURE_LIST; signature covers "
             "VariableName ‖ VendorGuid ‖ Attributes ‖ TimeStamp ‖ Data")

        stage("S9", "Sign the NVIDIA boot chain with the PKC key")
        bootchain = {}
        for name in ("BR-BCT", "MB1", "MB1-BCT", "PSC-FW", "BPMP-FW", "MB2",
                     "TOS (ATF BL31 + OP-TEE + TAs)", "UEFI (uefi_jetson.bin)"):
            img = f"{name} image".encode() + os.urandom(16)
            bootchain[name] = {"image": img, "sig": ec_sign(k["pkc"], img)}
        ok(f"{len(bootchain)} boot-chain images signed (BootROM/MB1/MB2 verify against PKC_HASH fuse)")

        stage("S10–S13", "CPYR code (dm-verity), CPYR data (LUKS), encrypted + db-signed boot entry")
        disk_key = os.urandom(32)                                     # S11
        generic_pass = kdf(disk_key, b"luks-srv-generic (sim)", b"", 16)
        payload_key, model_key, alert_seed = os.urandom(32), os.urandom(32), os.urandom(32)
        real = load_real_artifacts(REAL_CPYR) if REAL_CPYR else None
        if real:
            m = real["meta"]
            model_blob = real["weights"]
            traffic = {"normal": real["normal"][:6], "attack": real["fuzz"], "end": real["normal"][6:]}
            info(f"REAL CPYR conv_enocder: {m['params']} parameters, model.pt {m['weights_bytes']} B, "
                 f"threshold {m['threshold']:.4f}")
        else:
            model_blob = canon(train_standin(rng))
            traffic = {"normal": normal_frames(rng, 2000), "attack": fuzz_frames(rng, 600),
                       "end": normal_frames(rng, 2000, 5000)}
            info("stand-in detector trained on synthetic frames (use --real-cpyr for the CPYR conv encoder)")
        disk_uuid = str(uuid.uuid4())
        releases = {v: build_release(v, k, model_blob, model_key, real, disk_uuid, generic_pass, payload_key)
                    for v in ("1.0", "2.0")}
        rel = releases["2.0"]
        info(f"S10 CPYR code partition {len(rel['code_part'])} B → dm-verity root {short(rel['verity_root'])} "
             "(goes into the initrd)")
        info("S11 disk_enc.key generated → EKB (luks-srv derives LUKS passphrases from it)")
        info(f"S12 CPYR data partition: LUKS aes-xts-plain64 (512-bit DEK), keyslot 0 = generic passphrase; "
             f"inside it model.bin.enc ({len(rel['model_enc'])} B, AES-256-GCM with the CPYR model key)")
        ok("S13 kernel + initrd encrypted with the payload key; boot entry signed with the db key (RSA-2048)")

        stage("S14–S16", "Golden PCRs, EKB, backend")
        sv = UefiVarStore()
        for n in ("PK", "KEK", "db"):
            sv.set_variable(n, auths[n])
        golden_log = ([{"pcr": 0, "digest": sha256(b["image"]).hex()} for b in bootchain.values()]
                      + [{"pcr": 7, "digest": sha256(bytes.fromhex(sv.snapshot()[n])).hex()}
                         for n in ("PK", "KEK", "db", "dbx")]
                      + [{"pcr": 4, "digest": rel["entry_digest"].hex()}]
                      + [{"pcr": 10, "digest": sha256(d).hex()} for _, d in sorted(
                          {**rel["code_files"], "model.bin.enc": rel["model_enc"]}.items())])
        golden = replay(golden_log)
        ekb_plain = canon({"uefi_payload_key": payload_key.hex(), "disk_enc_key": disk_key.hex(),
                           "cpyr_model_key": model_key.hex(), "alert_key_seed": alert_seed.hex(),
                           "cpyr_pcr_policy": policy_digest(golden).hex()})
        for i in PCRS:
            info(f"golden PCR{i:<2} = {short(golden[i])}  {PCR_NAMES[i]}")
        ok("S14 EKB contents ready: payload key · disk key · model key · alert key · PCR policy "
           "(encrypted per device with KDF(OEM_K1, ECID) at flash time)")
        ok("S16 backend stores golden PCRs + manufacturer CA public key")

        stage("S17", "Release the factory package (no private keys, no plaintext secrets)")
        package = {"fuse_blob": blob, "bootchain": bootchain, "pkc_pub": pub(k["pkc"]),
                   "auths": auths, "release": rel, "ekb_plain_for_hsm_only": None}
        ok("package: encrypted fuse blob · signed boot chain · .auth updates · encrypted boot entry · "
           "dm-verity code partition · LUKS data partition")
        info("EKB is produced per unit by the HSM-side flashing service (needs OEM_K1 + ECID)")
    return {"keys": k, "certs": cert, "secret": secret, "fskp": (fskp_ak, fskp_ek), "fuse_values": fuse_values,
            "ekb_plain": ekb_plain, "package": package, "releases": releases, "traffic": traffic,
            "rng": rng, "ts0": ts0, "model_blob": model_blob, "real": real,
            "backend": {"ca_pub": pub(k["ca"]), "golden": golden, "devices": {}, "sessions": {}}}


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ============================================================================
# Device: chip, BootROM, secure world, fTPM
# ============================================================================
class Chip:
    def __init__(self):
        self.ecid = os.urandom(8)
        self.rom_root = ROM_ROOT
        self.fuses = {}
        self.security_mode = False

    def bootrom_burn(self, blob, oem_config):
        """S21–S22: re-derive FSKP keys, authenticate, decrypt, burn."""
        if self.security_mode:
            raise FuseBlobError("security-mode fuse already burned: fuses are locked")
        ak, ek = fskp_keys(self.rom_root, oem_config)
        ct, tag = blob[:-32], blob[-32:]
        if not hmac.compare_digest(hmac.new(ak, ct, hashlib.sha256).digest(), tag):
            raise FuseBlobError("fuse blob authentication failed (FSKP_AK) — nothing burned")
        dec = Cipher(algorithms.AES(ek), modes.CTR(ct[:16])).decryptor()
        values = json.loads(dec.update(ct[16:]) + dec.finalize())
        self.fuses = {"OEM_K1": bytes.fromhex(values["OEM_K1"]), "OEM_K2": bytes.fromhex(values["OEM_K2"]),
                      "PKC_HASH": bytes.fromhex(values["PKC_HASH"])}
        return values


class FTPM:
    def __init__(self, chip):
        self.ek = ec_from_secret(kdf(chip.fuses["OEM_K1"], b"ftpm-ek (sim)", chip.ecid))
        self.ak = ec_key()
        self.ak_cert = ec_sign(self.ek, pub(self.ak))
        self.pcrs = {i: b"\0" * 32 for i in PCRS}
        self.log = []

    def extend(self, pcr, desc, data):
        d = sha256(data)
        self.pcrs[pcr] = sha256(self.pcrs[pcr], d)
        self.log.append({"pcr": pcr, "desc": desc, "digest": d.hex()})

    def quote(self, nonce):
        body = canon({"pcrs": {str(i): self.pcrs[i].hex() for i in PCRS}, "nonce": nonce.hex()})
        return {"body": body, "sig": ec_sign(self.ak, body), "ak_pub": pub(self.ak),
                "ak_cert": self.ak_cert, "event_log": copy.deepcopy(self.log)}


class OPTEE:
    """S-EL1 OP-TEE with jetson_user_key_pta and TAs: payload, luks-srv, cpyr-ta, PKCS#11, fTPM."""

    def __init__(self, chip, ekb):
        try:
            self._k = json.loads(gcm_dec(kdf(chip.fuses["OEM_K1"], b"ekb-enc (sim)", chip.ecid), ekb, b"EKB"))
        except InvalidTag:
            raise BootHalt("EKB authentication failed (wrong device fuses or tampered EKB)")
        self.chip = chip
        self.ftpm = FTPM(chip)
        self.open = {"payload": True, "luks-srv": True, "cpyr-ta": True}
        self._alert = ec_from_secret(bytes.fromhex(self._k["alert_key_seed"]))

    def _gate(self, ta):
        if not self.open[ta]:
            raise TAError(f"{ta}: TEE_ERROR_ACCESS_DENIED — service shut down (SRV_DOWN) until reset")

    def srv_down(self, ta):
        self.open[ta] = False

    def payload_decrypt(self, blob, aad):
        self._gate("payload")
        return gcm_dec(bytes.fromhex(self._k["uefi_payload_key"]), blob, aad)

    def luks_get_pass(self, disk_uuid, generic=False):
        """luks-srv → jetson_user_key_pta: two NIST SP 800-108 steps inside the secure world."""
        self._gate("luks-srv")
        dk = bytes.fromhex(self._k["disk_enc_key"])
        if generic:
            return kdf(dk, b"luks-srv-generic (sim)", b"", 16)
        per_dev = kdf(dk, b"luks-srv-ecid", self.chip.ecid)
        return kdf(per_dev, b"luks-srv-passphrase-unique", disk_uuid.encode(), 16)

    def cpyr_get_model_key(self):
        self._gate("cpyr-ta")
        cur = policy_digest(self.ftpm.pcrs)
        if cur.hex() != self._k["cpyr_pcr_policy"]:
            raise TAError(f"cpyr-ta: PCR policy mismatch ({short(cur)} ≠ sealed "
                          f"{self._k['cpyr_pcr_policy'][:16]}…) — model key NOT released")
        return bytes.fromhex(self._k["cpyr_model_key"])

    def pkcs11_sign(self, data):
        return ec_sign(self._alert, data)


class Unit:
    def __init__(self):
        self.chip = Chip()
        self.storage = {}
        self.optee = None
        self.detector = self.model = None
        self.cpyr_running = False
        self.seq = 0
        self.credential = None


# ============================================================================
# Boot B1–B11
# ============================================================================
def boot(prog, u, srv_down=True, first_boot=False):
    st, ch = u.storage, u.chip
    u.cpyr_running, u.credential, u.detector, u.model = False, None, None, None   # fresh power-on
    early = []
    stage("B1–B3", "BootROM + PSCROM → MB1 → MB2 (PKC secure boot)")
    if ch.security_mode:
        if sha256(st["pkc_pub"]) != ch.fuses["PKC_HASH"]:
            raise BootHalt("PKC public key does not match the PKC_HASH fuse")
        for name, b in st["bootchain"].items():
            if not ec_verify(st["pkc_pub"], b["sig"], b["image"]):
                raise BootHalt(f"{name}: PKC signature invalid — boot halted")
            early.append((name, b["image"]))
        ok(f"{len(early)} images verified against the PKC_HASH fuse: " + ", ".join(st["bootchain"]))
    else:
        warn("security-mode fuse not burned: BootROM does not enforce signatures")

    stage("B4", "Secure world: ATF BL31 + OP-TEE; EKB opened with KDF(OEM_K1, ECID); fTPM up")
    u.optee = OPTEE(ch, st["ekb"])
    for name, img in early:
        u.optee.ftpm.extend(0, name, img)
    ok("EKB opened in S-EL1: payload key · disk key · model key · alert key · PCR policy")
    ok(f"fTPM TA: EK derived from fuses; PCR0 replayed = {short(u.optee.ftpm.pcrs[0])}")

    stage("B5", "UEFI: measure Secure Boot vars, verify boot entry (db/dbx), decrypt via OP-TEE")
    sv = st["uefi"]
    for n in ("PK", "KEK", "db", "dbx"):
        u.optee.ftpm.extend(7, f"EFI var {n}", sv.vars[n])
    if sv.setup_mode:
        raise BootHalt("UEFI in setup mode (no PK) — Secure Boot not enforced")
    entry = st["boot_entry"]
    digest = sha256(canon(entry))
    if digest in sv.hashes("dbx"):
        raise BootHalt("boot entry hash is in dbx (revoked) — refusing to boot")
    trusted = False
    for c in sv.certs("db"):
        try:
            c.public_key().verify(st["boot_entry_sig"], digest, padding.PKCS1v15(), hashes.SHA256())
            trusted = True
        except InvalidSignature:
            pass
    if not trusted:
        raise BootHalt("boot entry signature not trusted by any db certificate")
    u.optee.ftpm.extend(4, "boot entry", canon(entry))
    try:
        u.optee.payload_decrypt(bytes.fromhex(entry["kernel_enc"]), b"kernel")
        initrd = json.loads(u.optee.payload_decrypt(bytes.fromhex(entry["initrd_enc"]), b"initrd"))
    except InvalidTag:
        raise BootHalt("payload decryption failed (ciphertext modified)")
    ok(f"'{entry['name']}' trusted by db, not in dbx → PCR4; kernel + initrd decrypted by the OP-TEE payload TA")
    if srv_down:
        u.optee.srv_down("payload")
        ok("ExitBootServices → payload TA SRV_DOWN")
    else:
        warn("payload TA left OPEN after boot (misconfiguration)")

    stage("B6", "Linux kernel: optee driver (/dev/tee0), tpm_ftpm_tee (/dev/tpm0), IMA, dm-crypt, dm-verity")
    ok("SMC channel to OP-TEE established; fTPM exposed as /dev/tpm0")
    warn("requirement: fTPM must be up before IMA initialises, or IMA runs without extending PCR10 (verify on the BSP)")

    stage("B7", "initrd: LUKS unlock via luks-srv (+ re-key on first boot), dm-verity open, switch_root")
    hdr = st["luks_hdr"]
    if first_boot:
        gp = u.optee.luks_get_pass(hdr["uuid"], generic=True)
        dek, sid = luks_unlock(hdr, gp)
        ok(f"generic passphrase opens keyslot {sid} (factory-wide image)")
        pp = u.optee.luks_get_pass(hdr["uuid"])
        new = luks_add_key(hdr, dek, pp)
        del hdr["keyslots"][sid]
        ok(f"re-key: per-device passphrase = KDF(KDF(disk key, 'luks-srv-ecid', ECID), "
           f"'luks-srv-passphrase-unique', UUID) → keyslot {new}; generic keyslot {sid} removed")
        info("only the keyslot changed: the same DEK is re-wrapped, no data re-encrypted")
    else:
        pp = u.optee.luks_get_pass(hdr["uuid"])
        dek, sid = luks_unlock(hdr, pp)
        ok(f"per-device passphrase from luks-srv opens keyslot {sid}")
    if srv_down:
        u.optee.srv_down("luks-srv")
        ok("nvluks-srv-app → LUKS_SRV_TA_CMD_SRV_DOWN")
    else:
        warn("luks-srv left OPEN after boot (CVE-2026-24153 pattern)")
    try:
        data_files = unpack(_xts(dek, st["cpyr_data"], False))
    except (tarfile.TarError, UnicodeDecodeError, ValueError):
        data_files = {}
    try:
        code = unpack(verity_open(st["cpyr_code"], st["cpyr_verity_level0"],
                                  bytes.fromhex(initrd["verity_salt"]), bytes.fromhex(initrd["verity_root"])))
    except VerityError as e:
        bad(f"dm-verity: {e}")
        bad("CPYR code partition refused — CPYR not started")
        return u
    except tarfile.TarError:
        bad("CPYR code partition unreadable")
        return u
    ok("dm-verity: CPYR code partition verified against the root hash in the signed initrd")

    stage("B8", "rootfs services: tee-supplicant, tpm2-tss (/dev/tpmrm0), SocketCAN, attestation agent")
    ok("normal world issues requests; keys stay in the secure world")

    stage("B9", "CPYR starts: IMA → PCR10, cpyr-ta checks PCR policy, model decrypted in RAM")
    files = dict(code)
    if "model.bin.enc" not in data_files:
        bad("model.bin.enc unreadable inside the LUKS volume (data corrupted)")
        return u
    files["model.bin.enc"] = data_files["model.bin.enc"]
    for n in sorted(files):
        u.optee.ftpm.extend(10, f"IMA {n}", files[n])
        info(f"IMA sha256 {sha256(files[n]).hex()[:16]}…  {n:<17} → PCR10")
    try:
        mk = u.optee.cpyr_get_model_key()
    except TAError as e:
        bad(str(e))
        bad("CPYR cannot decrypt its model → not started")
        return u
    if srv_down:
        u.optee.srv_down("cpyr-ta")
    man = json.loads(files["manifest.json"])
    try:
        blob = gcm_dec(mk, files["model.bin.enc"], man["model_aad"].encode())
    except InvalidTag:
        bad("model.bin.enc failed AES-GCM authentication → not started")
        return u
    if sha256(blob).hex() != man["weights_sha256"]:
        bad("decrypted weights ≠ weights_sha256 in the measured manifest → not started")
        return u
    ns = {}
    exec(files["cpyr_detector.py"], ns)
    u.detector, u.model, u.cpyr_running = ns, ns["load"](blob, files), True
    ok("PCR policy matched → model key released once, cpyr-ta SRV_DOWN; weights hash matches manifest")
    ok(f"CPYR v{ns['VERSION']} running" + (" (torch.load weights_only=True, repo conv_enocder)"
                                            if "net" in u.model else ""))
    return u


# ============================================================================
# Backend: enrolment, attestation (B10), analytics (B11)
# ============================================================================
def attest(prog, u, nonce=None, quote=None):
    """B10 — returns True and installs a short-lived session credential on success."""
    be = prog["backend"]
    stage("B10", "Remote attestation: nonce → fTPM quote (AK) + event log → verify → session credential")
    rec = be["devices"].get(u.chip.ecid.hex())
    nonce = nonce or os.urandom(16)
    q = quote or u.optee.ftpm.quote(nonce)

    def fail(m):
        bad(m)
        return False
    if not rec:
        return fail("device not enrolled (unknown ECID / no EK certificate)")
    if not ec_verify(be["ca_pub"], rec["ek_cert"], rec["ek_pub"]):
        return fail("EK certificate invalid")
    if not ec_verify(rec["ek_pub"], q["ak_cert"], q["ak_pub"]):
        return fail("AK not certified by this device's EK")
    if not ec_verify(q["ak_pub"], q["sig"], q["body"]):
        return fail("quote signature invalid")
    body = json.loads(q["body"])
    if body["nonce"] != nonce.hex():
        return fail("stale quote: nonce mismatch (replay)")
    if any(replay(q["event_log"])[i].hex() != body["pcrs"][str(i)] for i in PCRS):
        return fail("event log does not replay to the quoted PCRs (log tampered)")
    ok("AK → EK → manufacturer CA · signature · nonce · event-log replay")
    diff = [i for i in PCRS if body["pcrs"][str(i)] != be["golden"][i].hex()]
    if diff:
        for i in diff:
            bad(f"PCR{i} ≠ golden — {PCR_NAMES[i]}")
        return fail("device NOT in a known-good state → quarantine, no credential")
    token = os.urandom(16).hex()
    be["sessions"][token] = {"ecid": u.chip.ecid.hex(), "ak": q["ak_pub"].hex(), "exp": time.time() + 600}
    u.credential = token
    ok("all PCRs match golden → short-lived session credential issued (bound to this AK)")
    return True


def send_analytics(prog, u, phases, verbose_label="B11"):
    """B11 — CPYR scores traffic, signs a report with the PKCS#11 alert key, sends over the attested session."""
    stage(verbose_label, "Primary analytics: CPYR report signed in OP-TEE (PKCS#11), sent over the attested session")
    if not u.cpyr_running:
        bad("CPYR not running — no analytics")
        return None
    report = {"ecid": u.chip.ecid.hex(), "release": u.detector["VERSION"],
              "pcr_digest": policy_digest(u.optee.ftpm.pcrs).hex(), "phases": {}}
    for label, frames in phases:
        hits = u.detector["detect"](u.model, frames)
        n = u.detector["count_windows"](u.model, frames)
        report["phases"][label] = {"windows": n, "flagged": len(hits)}
        (ok if bool(hits) == (label == "ATTACK") else bad)(
            f"{label:<6}: {n:>3} windows, {len(hits):>3} flagged (threshold {u.model['threshold']})")
    u.seq += 1
    report["seq"] = u.seq
    msg = canon(report)
    return {"msg": msg, "sig": u.optee.pkcs11_sign(msg), "credential": u.credential}


def backend_accept(prog, pkt):
    be = prog["backend"]
    if pkt is None:
        return False
    sess = be["sessions"].get(pkt["credential"] or "")
    if not sess or sess["exp"] < time.time():
        bad("backend: no valid attested session — analytics rejected")
        return False
    rep = json.loads(pkt["msg"])
    rec = be["devices"][sess["ecid"]]
    if rep["ecid"] != sess["ecid"] or not ec_verify(rec["alert_pub"], pkt["sig"], pkt["msg"]):
        bad("backend: report signature / device binding invalid — rejected")
        return False
    ok(f"backend: report #{rep['seq']} accepted from ECID {rep['ecid']} (session + alert-key signature valid)")
    return rep


# ============================================================================
# Phases 1–4 on one unit (S18–S40)
# ============================================================================
def manufacture(prog, u=None, verbose=True, eol=True):
    u = u or Unit()
    pk = prog["package"]
    with (Quiet() if not verbose else _Null()):
        stage("S18–S20", "Factory: receive package, RCM mode, send encrypted fuse blob")
        info(f"factory sees only ciphertext: fuse blob entropy {entropy(pk['fuse_blob']):.2f} bits/byte")
        stage("S21–S23", "BootROM re-derives FSKP keys, authenticates + decrypts, burns, read-back")
        u.chip.bootrom_burn(pk["fuse_blob"], OEM_CONFIG)
        if u.chip.fuses["PKC_HASH"] != sha256(pk["pkc_pub"]):
            raise SystemExit("S24: read-back mismatch → reject unit")
        ok("OEM_K1/K2 + PKC_HASH burned; read-back OK")
        stage("S25", "Burn the security-mode fuse LAST")
        u.chip.security_mode = True
        ok("debug locked; BootROM now enforces PKC signatures")

        stage("S26–S27", "Mass-flash + enrol UEFI variables (setup mode → user mode)")
        rel = pk["release"]
        sv = UefiVarStore()
        for n in ("PK", "KEK", "db"):
            sv.set_variable(n, pk["auths"][n])
        ok("SetVariable(PK.auth, KEK.auth, db.auth) accepted; PK enrolled → user mode")
        ekb = gcm_enc(kdf(prog["secret"]["OEM_K1"], b"ekb-enc (sim)", u.chip.ecid), prog["ekb_plain"], b"EKB")
        u.storage = {"bootchain": pk["bootchain"], "pkc_pub": pk["pkc_pub"], "ekb": ekb, "uefi": sv,
                     "boot_entry": rel["boot_entry"], "boot_entry_sig": rel["entry_sig"],
                     "cpyr_code": rel["code_part"], "cpyr_verity_level0": list(rel["verity_level0"]),
                     "luks_hdr": copy.deepcopy(rel["luks_hdr"]), "cpyr_data": rel["data_ct"]}
        ok(f"EKB for ECID {u.chip.ecid.hex()} produced by the HSM flashing service and written")

        banner_s = "S28–S30 + B1–B9"
        stage(banner_s, "First boot (generic LUKS passphrase → per-device re-key)")
        boot(prog, u, first_boot=True)

        stage("S31–S33", "Read EK public key + ECID, CA signs the EK certificate, register device")
        ek_pub = pub(u.optee.ftpm.ek)
        prog["backend"]["devices"][u.chip.ecid.hex()] = {
            "ek_pub": ek_pub, "ek_cert": ec_sign(prog["keys"]["ca"], ek_pub),
            "alert_pub": pub(ec_from_secret(bytes.fromhex(json.loads(prog["ekb_plain"])["alert_key_seed"])))}
        ok("device registered: ECID · EK certificate · alert public key")

        if eol:
            stage("S34–S37", "End-of-line attestation")
            att = attest(prog, u)
            tr = prog["traffic"]
            pkt = send_analytics(prog, u, [("NORMAL", tr["normal"]), ("ATTACK", tr["attack"]), ("END", tr["end"])],
                                 "S38–S39 + B11")
            rep = backend_accept(prog, pkt) if att else False
            passed = bool(rep) and rep["phases"]["ATTACK"]["flagged"] > 0 and \
                rep["phases"]["NORMAL"]["flagged"] == 0 and rep["phases"]["END"]["flagged"] == 0
            no_generic = not _generic_opens(prog, u)
            stage("S39–S40", "Record in manufacturing DB, ship")
            (ok if no_generic else bad)("LUKS header holds no generic keyslot")
            (ok if passed else bad)("functional test: test pattern produced alerts only in the ATTACK phase")
            u.eol_pass = att and passed and no_generic
            (ok if u.eol_pass else bad)("unit " + ("SHIPPED" if u.eol_pass else "QUARANTINED"))
    return u


def _generic_opens(prog, u):
    gp = kdf(bytes.fromhex(json.loads(prog["ekb_plain"])["disk_enc_key"]), b"luks-srv-generic (sim)", b"", 16)
    try:
        luks_unlock(u.storage["luks_hdr"], gp)
        return True
    except BootHalt:
        return False


def field_boot(prog, u, **kw):
    """Normal boot in the field: B1–B11."""
    boot(prog, u, **kw)
    a = attest(prog, u)
    tr = prog["traffic"]
    pkt = send_analytics(prog, u, [("NORMAL", tr["normal"]), ("ATTACK", tr["attack"])])
    return a, (backend_accept(prog, pkt) if a else False)


# ============================================================================
# Scenarios
# ============================================================================
def fresh(prog=None):
    prog = prog or prepare(verbose=False)
    with Quiet():
        u = manufacture(prog, verbose=False)
    return prog, u


def sc_normal():
    prog = prepare()
    u = manufacture(prog)
    banner("FIELD · second boot (per-device LUKS passphrase)")
    a, rep = field_boot(prog, u)
    return u.eol_pass and a and bool(rep), "manufactured, attested, shipped; field boot attested and reporting"


def sc_forged_fuse_blob():
    prog = prepare(verbose=False)
    stage("ATTACK", "Factory insider builds a fuse blob with their own PKC hash (no FSKP keys)")
    u = Unit()
    rogue = fuse_blob_build({"OEM_K1": os.urandom(32).hex(), "OEM_K2": os.urandom(32).hex(),
                             "PKC_HASH": sha256(pub(ec_key())).hex()}, os.urandom(32), os.urandom(32))
    try:
        u.chip.bootrom_burn(rogue, OEM_CONFIG)
        return False, "rogue blob burned"
    except FuseBlobError as e:
        ok(f"BootROM: {e}")
    stage("ATTACK", "Flip one bit of the genuine blob in transit")
    blob = bytearray(prog["package"]["fuse_blob"])
    blob[40] ^= 1
    try:
        u.chip.bootrom_burn(bytes(blob), OEM_CONFIG)
        return False, "modified blob burned"
    except FuseBlobError as e:
        ok(f"BootROM: {e}")
    return True, "only blobs authenticated with FSKP_AK are burned"


def sc_disk_theft():
    prog, u = fresh()
    stage("ATTACK", "Pull the NVMe: read the CPYR data partition offline")
    ct = u.storage["cpyr_data"]
    info(f"LUKS data partition: {len(ct)} B, entropy {entropy(ct):.2f} bits/byte (AES-XTS ciphertext)")
    gp = kdf(bytes.fromhex(json.loads(prog["ekb_plain"])["disk_enc_key"]), b"luks-srv-generic (sim)", b"", 16)
    try:
        luks_unlock(u.storage["luks_hdr"], gp)
        return False, "generic passphrase still opens a shipped unit"
    except BootHalt:
        ok("even a leaked generic passphrase fails: the unit was re-keyed on first boot")
    stage("ATTACK", "Move the NVMe to another Jetson (different ECID)")
    _, other = fresh(prog)
    other.storage["luks_hdr"], other.storage["cpyr_data"] = u.storage["luks_hdr"], u.storage["cpyr_data"]
    try:
        with Quiet():
            boot(prog, other)
        return False, "foreign disk opened"
    except BootHalt as e:
        ok(f"other unit: {e} (its ECID derives a different passphrase)")
    return True, "LUKS ciphertext; per-device passphrase bound to ECID + disk UUID"


def sc_tamper_model():
    prog, u = fresh()
    stage("ATTACK", "Flip a byte in the LUKS ciphertext of the data partition")
    d = bytearray(u.storage["cpyr_data"])
    d[700] ^= 0xFF
    u.storage["cpyr_data"] = bytes(d)
    info("AES-XTS has no integrity: the sector decrypts to garbage without an error")
    boot(prog, u)
    a = attest(prog, u)
    return (not u.cpyr_running) and not a, "XTS garbles silently; GCM / IMA / PCR policy catch it"


def sc_blind_detector():
    prog, u = fresh()
    stage("ATTACK", "Patch the detector to never alert and rebuild the dm-verity tree")
    code = unpack(u.storage["cpyr_code"])
    code["cpyr_detector.py"] = code["cpyr_detector.py"].replace(b"return [(i, float(v))", b"return []  #")
    part = pack(code)
    lvl0, root = verity_format(part, os.urandom(32))
    u.storage["cpyr_code"], u.storage["cpyr_verity_level0"] = part, lvl0
    info(f"attacker's tree is self-consistent (root {short(root)}) but the trusted root is in the signed initrd")
    boot(prog, u)
    return not u.cpyr_running, "rebuilt tree rejected: root hash anchored in the db-signed initrd"


def sc_tamper_initrd():
    prog, u = fresh()
    stage("ATTACK", "Modify the encrypted initrd in the boot entry")
    e = dict(u.storage["boot_entry"])
    b = bytearray(bytes.fromhex(e["initrd_enc"]))
    b[40] ^= 1
    e["initrd_enc"] = b.hex()
    u.storage["boot_entry"] = e
    try:
        boot(prog, u)
        return False, "accepted"
    except BootHalt as ex:
        bad(f"UEFI: {ex}")
        return True, "UEFI Secure Boot refused the modified boot entry"


def sc_db_inject():
    prog, u = fresh()
    sv = u.storage["uefi"]
    stage("ATTACK", "Root builds db.auth signed with its own key")
    k = rsa.generate_private_key(65537, 2048)
    c = make_cert("attacker", k, "attacker", k, True)
    blocked = 0
    for name in ("db", "KEK"):
        try:
            sv.set_variable(name, build_auth(name, esl([c.public_bytes(serialization.Encoding.DER)], EFI_CERT_X509),
                                             k, c, int(time.time()) + 10))
        except EfiSecurityViolation as e:
            blocked += 1
            ok(f"EFI_SECURITY_VIOLATION: {e}")
    return blocked == 2, "db needs a KEK signature, KEK needs a PK signature"


def sc_auth_replay():
    prog, u = fresh()
    sv = u.storage["uefi"]
    k, cert = prog["keys"], prog["certs"]
    stage("SETUP", "OEM rotates db: new db certificate, signed db.auth with a newer timestamp")
    k2 = rsa.generate_private_key(65537, 2048)
    c2 = make_cert("Demo OEM UEFI db signer 2", k2, "Demo OEM UEFI KEK", k["KEK"], False)
    sv.set_variable("db", build_auth("db", esl([c2.public_bytes(serialization.Encoding.DER)], EFI_CERT_X509),
                                     k["KEK"], cert["KEK"], prog["ts0"] + 3600))
    ok("db now trusts only the new signer")
    stage("ATTACK", "Replay the original, genuinely signed db.auth to restore the old certificate")
    res = []
    try:
        sv.set_variable("db", prog["package"]["auths"]["db"])
        res.append(False)
    except EfiSecurityViolation as e:
        ok(str(e))
        res.append(True)
    stage("ATTACK", "Retarget a genuine KEK-signed update to dbx (same signer authority, different variable)")
    fresh_db = build_auth("db", esl([c2.public_bytes(serialization.Encoding.DER)], EFI_CERT_X509),
                          k["KEK"], cert["KEK"], prog["ts0"] + 7200, append=True)
    try:
        sv.set_variable("dbx", fresh_db)
        res.append(False)
    except EfiSecurityViolation as e:
        ok(f"{e} — name + VendorGuid are inside the signed bytes")
        res.append(True)
    return all(res), "timestamp blocks replay; name + VendorGuid bind the signature to one variable"


def sc_downgrade():
    prog, u = fresh()
    stage("ATTACK", "Flash the old but genuinely signed release v1.0")
    v1 = prog["releases"]["1.0"]
    st = u.storage
    st["boot_entry"], st["boot_entry_sig"] = v1["boot_entry"], v1["entry_sig"]
    st["cpyr_code"], st["cpyr_verity_level0"] = v1["code_part"], list(v1["verity_level0"])
    warn("v1.0 is signed by our db key — Secure Boot alone accepts it")
    boot(prog, u)
    a = attest(prog, u)
    stopped = (not u.cpyr_running) and not a
    stage("REMEDIATION", "OEM pushes dbx.auth (append, signed by KEK) revoking the v1.0 boot entry")
    st["uefi"].set_variable("dbx", build_auth("dbx", esl([v1["entry_digest"]], EFI_CERT_SHA256),
                                              prog["keys"]["KEK"], prog["certs"]["KEK"], int(time.time()), append=True))
    ok("dbx updated (APPEND_WRITE, authorised by KEK)")
    try:
        with Quiet():
            boot(prog, u)
        return False, "v1.0 still boots"
    except BootHalt as e:
        ok(f"v1.0 no longer boots: {e}")
    return stopped, "signed ≠ current: PCR policy + attestation caught it; dbx closes it"


def _ta_probe(srv_down):
    prog, u = fresh()
    with Quiet():
        boot(prog, u, srv_down=srv_down)
    stage("ATTACK", "Root opens /dev/tee0 and calls payload TA, luks-srv and cpyr-ta after boot")
    leaks = []
    for name, fn in (("payload TA", lambda: u.optee.payload_decrypt(bytes.fromhex(u.storage["boot_entry"]["initrd_enc"]), b"initrd")),
                     ("luks-srv", lambda: u.optee.luks_get_pass(u.storage["luks_hdr"]["uuid"])),
                     ("cpyr-ta", u.optee.cpyr_get_model_key)):
        try:
            fn()
            leaks.append(name)
            bad(f"{name} answered after boot")
        except TAError as e:
            ok(str(e))
    return leaks


def sc_ta_closed():
    return not _ta_probe(True), "every key-serving TA is SRV_DOWN after use"


def sc_ta_left_open():
    leaks = _ta_probe(False)
    if leaks:
        warn("CVE-2026-24153 pattern: a key-serving TA reachable after boot")
    return bool(leaks), "demonstrates why SRV_DOWN matters (leak expected)"


def sc_replay_quote():
    prog, u = fresh()
    with Quiet():
        boot(prog, u)
    old = os.urandom(16)
    rec = u.optee.ftpm.quote(old)
    a = attest(prog, u, nonce=os.urandom(16), quote=rec)
    return not a, "fresh nonce defeats replay"


def sc_forged_log():
    prog, u = fresh()
    with Quiet():
        boot(prog, u)
    n = os.urandom(16)
    q = u.optee.ftpm.quote(n)
    q["event_log"][-1]["digest"] = sha256(b"x").hex()
    return not attest(prog, u, nonce=n, quote=q), "forged log does not replay to the signed PCRs"


def sc_unattested_analytics():
    prog, u = fresh()
    with Quiet():
        boot(prog, u)
    tr = prog["traffic"]
    stage("ATTACK", "Send analytics without attesting first")
    r1 = backend_accept(prog, send_analytics(prog, u, [("ATTACK", tr["attack"])]))
    stage("ATTACK", "Attest, then reuse the credential after it expires")
    with Quiet():
        attest(prog, u)
    prog["backend"]["sessions"][u.credential]["exp"] = time.time() - 1
    r2 = backend_accept(prog, send_analytics(prog, u, [("ATTACK", tr["attack"])]))
    return not r1 and not r2, "analytics accepted only over a live attested session"


SCENARIOS = {
    "normal": (sc_normal, "Full manufacturing (S1–S40) + field boot (B1–B11)"),
    "forged-fuse-blob": (sc_forged_fuse_blob, "Rogue or modified fuse blob at the factory"),
    "disk-theft": (sc_disk_theft, "Pull the NVMe / move it to another unit"),
    "tamper-model": (sc_tamper_model, "Flip bytes in the LUKS data partition"),
    "blind-detector": (sc_blind_detector, "Patch detector + rebuild verity tree"),
    "tamper-initrd": (sc_tamper_initrd, "Modify the encrypted, db-signed initrd"),
    "db-inject": (sc_db_inject, "Root enrols own key via a self-signed .auth"),
    "auth-replay": (sc_auth_replay, "Replay an old db.auth / retarget it to KEK"),
    "downgrade": (sc_downgrade, "Flash old signed v1.0, then revoke via dbx.auth"),
    "ta-closed": (sc_ta_closed, "Call payload TA / luks-srv / cpyr-ta after boot"),
    "ta-left-open": (sc_ta_left_open, "Same, SRV_DOWN skipped (CVE-2026-24153 pattern)"),
    "replay-quote": (sc_replay_quote, "Replay an old healthy quote"),
    "forged-log": (sc_forged_log, "Tamper the measured-boot event log"),
    "unattested-analytics": (sc_unattested_analytics, "Send analytics without / after attestation"),
}


# ============================================================================
# --export: provisioning package on disk
# ============================================================================
def export_package(prog, out):
    files = []

    def w(rel, data, desc):
        p = os.path.join(out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(data if isinstance(data, bytes) else data.encode())
        files.append((rel, len(data), desc))

    k, c, pk, rel = prog["keys"], prog["certs"], prog["package"], prog["package"]["release"]
    pem = lambda key: key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption())
    crt = lambda cc: cc.public_bytes(serialization.Encoding.PEM)
    # oem_hsm
    for n, f in (("pkc", "pkc_bootloader_signing"), ("PK", "uefi_PK"), ("KEK", "uefi_KEK"), ("db", "uefi_db"),
                 ("ca", "manufacturer_ca"), ("oem_rsa", "oem_rsa_fskp")):
        w(f"oem_hsm/{f}.key.pem", pem(k[n]), "private key")
    w("oem_hsm/fskp_ak.bin", prog["fskp"][0], "unwrapped FSKP_AK")
    w("oem_hsm/fskp_ek.bin", prog["fskp"][1], "unwrapped FSKP_EK")
    w("oem_hsm/fuse_values.json", canon(prog["fuse_values"]), "fuse values (OEM_K1/K2 secret)")
    w("oem_hsm/ekb_plaintext.json", prog["ekb_plain"], "payload, disk, model, alert keys + PCR policy")
    w("oem_hsm/cpyr_model_plaintext.bin", prog["model_blob"], "unencrypted CPYR weights")
    # nvidia exchange
    oem_cert = make_cert("Demo OEM FSKP request", k["oem_rsa"], "Demo OEM FSKP request", k["oem_rsa"], False)
    w("nvidia_exchange/oem_publickey.cer", crt(oem_cert), "sent to NVIDIA")
    # certs + uefi
    for n in ("PK", "KEK", "db"):
        w(f"certs/{n}.crt", crt(c[n]), "X.509")
        der = c[n].public_bytes(serialization.Encoding.DER)
        w(f"uefi/{n}.esl", esl([der], EFI_CERT_X509), "EFI_SIGNATURE_LIST")
        w(f"uefi/{n}.auth", pk["auths"][n]["blob"], "EFI_VARIABLE_AUTHENTICATION_2 + PKCS#7 + ESL")
    w("certs/manufacturer_ca.pub.pem", k["ca"].public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo), "public")
    # factory package
    w("factory_package/fuse_blob.bin", pk["fuse_blob"], "AES-CTR(FSKP_EK) + HMAC(FSKP_AK)")
    for name, b in pk["bootchain"].items():
        t = name.split()[0]
        w(f"factory_package/bootchain/{t}.img", b["image"], "PKC-signed")
        w(f"factory_package/bootchain/{t}.sig", b["sig"], "ECDSA P-256")
    w("factory_package/boot_entry.json", canon(rel["boot_entry"]), "kernel + initrd encrypted (payload key)")
    w("factory_package/boot_entry.sig", rel["entry_sig"], "db-key signature")
    w("factory_package/cpyr_code.img", rel["code_part"], "dm-verity data (plaintext, integrity)")
    w("factory_package/cpyr_code.hashtree", b"".join(rel["verity_level0"]), "dm-verity tree")
    w("factory_package/cpyr_data.luks", rel["data_ct"], "LUKS data area (AES-XTS)")
    w("factory_package/cpyr_data.luks_header.json", canon(rel["luks_hdr"]), "LUKS header, generic keyslot")
    # backend
    w("backend/golden_pcrs.json", canon({str(i): prog["backend"]["golden"][i].hex() for i in PCRS}), "reference")
    lines = ["# Provisioning package (generated by --export)", "", "| File | Bytes | Content |", "|---|---|---|"]
    lines += [f"| `{f}` | {n} | {d} |" for f, n, d in files]
    lines += ["", "oem_hsm/ never leaves the HSM. factory_package/ goes to the factory. uefi/*.auth are real "
              "EFI_VARIABLE_AUTHENTICATION_2 blobs (RSA-2048, PKCS#7). Per-unit EKBs are produced at flash time.",
              "Demo keys only: regenerated on every export."]
    open(os.path.join(out, "LAYOUT.md"), "w").write("\n".join(lines) + "\n")
    return files


# ============================================================================
def main():
    global USE_COLOR, STEP_MODE, REAL_CPYR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", default="normal")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--real-cpyr", nargs="?", const=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   "cpyr_artifacts"), default=None, metavar="DIR")
    ap.add_argument("--export", metavar="DIR")
    a = ap.parse_args()
    USE_COLOR = USE_COLOR and not a.no_color
    STEP_MODE, REAL_CPYR = a.step, a.real_cpyr
    if a.list:
        for n, (_, d) in SCENARIOS.items():
            print(f"  {n:<22} {d}")
        return 0
    if a.export:
        banner("EXPORT · provisioning package")
        prog = prepare(verbose=False)
        for f, n, d in export_package(prog, a.export):
            print(f"  {f:<58} {n:>7} B  {d}")
        print(f"\n  written to {a.export}/ (see LAYOUT.md)")
        return 0
    names = list(SCENARIOS) if a.attack == "all" else [a.attack]
    if any(n not in SCENARIOS for n in names):
        print("unknown scenario — use --list")
        return 2
    results = []
    for n in names:
        fn, desc = SCENARIOS[n]
        banner(f"SCENARIO: {n} — {desc}" + ("   [real CPYR]" if REAL_CPYR else ""))
        passed, why = fn()
        results.append((n, passed, why))
        print("\n   " + (_c("1;32", "RESULT: as expected") if passed else _c("1;31", "RESULT: NOT as expected"))
              + f" — {why}")
    if len(results) > 1:
        banner("SUMMARY")
        for n, p, why in results:
            print(f"  {_c('32', 'PASS') if p else _c('31', 'FAIL')}  {n:<22} {why}")
    return 0 if all(p for _, p, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
