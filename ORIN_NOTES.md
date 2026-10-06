# Reproducing parts of this on an unfused Orin dev kit

**Do not burn fuses on your only dev kit.** Fusing is permanent, and a mistake bricks the board. On an unfused board the hardware root of trust isn't enforced, so anything you build is a functional demo, not a secure one. Say that explicitly if you show it.

| Piece | On an unfused Orin | Notes |
|---|---|---|
| OP‑TEE trusted app (`cpyr-ta`) | ✅ Yes | Build with the OP‑TEE sources from the Jetson Linux BSP (`optee_src_build.sh`), rebuild `tos.img`, reflash. Start from the hello_world / sample TAs. |
| EKB with your own keys | ⚠️ Functional only | OP‑TEE's EKB tooling works, but with unburned OEM keys the EKB is protected by known/default values. |
| UEFI Secure Boot (PK/KEK/db) | ⚠️ Partial | NVIDIA documents enabling UEFI Secure Boot at flash time. Without PKC fusing, UEFI itself isn't anchored in hardware. |
| UEFI payload encryption | ⚠️ Check docs | Uses the payload key in the EKB; see *UEFI Payload Encryption* for your release. |
| dm‑verity on a CPYR partition | ✅ Yes | Standard Linux: `veritysetup format` / `veritysetup open`. Put the root hash in the initrd or the kernel cmdline. |
| LUKS for the model partition | ✅ Functional | Jetson's `luks-srv` flow works, but key derivation relies on fuse keys, so it isn't secure until fused. |
| IMA → PCR10 | ✅ If enabled in the kernel config | Needs a TPM to extend into. |
| fTPM measured boot + attestation | ❌ Not meaningfully | NVIDIA's fTPM provisioning relies on secure boot being enabled. Use the simulation for this part. |

## Disk encryption and FSKP (from NVIDIA's docs)

```bash
# Build LUKS-encrypted images with a generic passphrase for mass flashing
sudo BOARDID=3701 BOARDSKU=0004 ROOTFS_ENC=1 ./l4t_initrd_flash.sh \
    -i ./disk_enc.key -p "--generic-passphrase" --massflash 5 --no-flash \
    jetson-agx-orin-devkit internal
sudo ./l4t_initrd_flash.sh --flash-only --massflash 5 jetson-agx-orin-devkit internal

# Unlock manually (what the initrd does)
nvluks-srv-app --context-string "${DISK_UUID}" --get-unique-pass | cryptsetup luksOpen <dev> <name>

# FSKP: unwrap NVIDIA's keys, build the blob without burning, burn at the factory
openssl rsautl -decrypt -inkey oem_rsa_priv.pem -in fskp_ak.bin.rsa_wrap > fskp_ak.bin
openssl rsautl -decrypt -inkey oem_rsa_priv.pem -in fskp_ek.bin.rsa_wrap > fskp_ek.bin
fskp_fuseburn.py ... -g        # generate blob only;  ... -b / -P to burn
```

Disk encryption works on an unfused board, but its key derivation rests on fuse keys, so it isn't secure until the board is fused. Never run FSKP on your only dev kit.

## Standard commands worth knowing

```bash
# dm-verity
veritysetup format cpyr.img cpyr.hash          # prints the root hash
veritysetup open cpyr.img cpyr cpyr.hash <root_hash>
veritysetup verify cpyr.img cpyr.hash <root_hash>

# fTPM / TPM2 (on a fused device with fTPM provisioned)
tpm2_pcrread sha256:0,4,7,10
tpm2_eventlog /sys/kernel/security/tpm0/binary_bios_measurements
tpm2_quote -c ak.ctx -l sha256:0,4,7,10 -q <nonce> -m quote.msg -s quote.sig -o pcrs.bin
tpm2_checkquote -u ak.pub -m quote.msg -s quote.sig -f pcrs.bin -q <nonce>
```

Check exact script names and flags against the Jetson Linux Developer Guide for your release (r36.x for Orin, r38+/r39.x for Thor). They change between releases.
