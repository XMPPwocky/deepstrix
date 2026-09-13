# Second Strix Halo box — provisioning + interconnect (rev 1, 2026-09-13)

Facts from this box (`lumi-brain`, NixOS 26.11 "Zokor" 2026-08-31 channel, kernel 7.2.2):

| item | this box | what it means for box 2 |
|---|---|---|
| USB4 | two AMD Strix Halo USB4 host routers (`ca:00.5`, `ca:00.6`), `thunderbolt` module loaded, `boltctl` present | host-to-host USB4 networking is available on both ends; only the `thunderbolt-net` module is missing |
| Ethernet | one Realtek RTL8125 2.5 GbE (`eno1`), Wi-Fi 7 | 2.5 GbE is the fallback link and the rsync path if USB4 misbehaves |
| PCIe | one OCuLink, occupied by the 9070 XT on whichever box hosts it | no room for a 25/100 GbE NIC on the dGPU box |
| RAM | 96 GB (box 2: 128 GB) | box 2 is the natural home for the larger expert share |
| storage | LUKS `cryptroot`; `--perf-no_read_workqueue` applied | box 2 gets a **plaintext partition for the weights** (PLAN.md §3.3: dm-crypt's workqueue was the cold-read bottleneck) |
| tools missing here | `iperf3`, `thunderbolt-net` | add both to the shared module |

The system flake was provided as `/persist/lumi-brain.tgz` (extracted to `~/lumi-brain/`); the
two-host refactor is written and syntax-checked: **`docs/v41/second_box_flake.patch`** (new
`modules/strix-halo.nix` with per-box GTT/TTM options, `modules/interconnect.nix`, `hosts/lumi-brain2/`
with a plaintext ext4 `/weights` partition, `flake.nix` with `mkHost` and a host-aware install
app, README bring-up section). Box 1's effective configuration is unchanged by construction
(its values are the module defaults).

## 1. Provisioning: one flake, two `nixosConfigurations`

```
flake.nix
  nixosConfigurations.lumi-brain  = mk { host = "lumi-brain";  hw = ./hosts/lumi-brain;  }
  nixosConfigurations.lumi-brain2 = mk { host = "lumi-brain2"; hw = ./hosts/lumi-brain2; }
modules/
  common.nix        # everything both boxes share (below)
  strix-halo.nix    # kernel params, amdgpu, ROCm 7.2.3, GTT sizing
  interconnect.nix  # thunderbolt-net + static link addressing (both ends)
  deepstrix.nix     # user, dev shell deps, run scripts, persistence
hosts/lumi-brain/   hardware-configuration.nix, disks (LUKS), dGPU bits
hosts/lumi-brain2/  hardware-configuration.nix, disks (plaintext weights partition)
```

What must differ per host: `networking.hostName`, `hardware-configuration.nix` (disk UUIDs,
LUKS vs plaintext), the interconnect address (`.1` vs `.2`), the dGPU-specific bits (only
where the 9070 XT is plugged), and `amdgpu` GTT sizing for 96 vs 128 GB. Everything else —
impermanence layout (`/persist`, `/home`), users, ROCm, kernel, the `nix develop` toolchain,
`ttm.page_pool_size`, `amdgpu.no_system_mem_limit=1` — lives in the shared modules so the two
images differ by exactly one host module each.

`interconnect.nix` (both ends):

```nix
{ lib, pkgs, hostAddr, peerAddr, ... }: {
  boot.kernelModules = [ "thunderbolt" "thunderbolt-net" ];
  services.hardware.bolt.enable = true;                 # boltctl; authorise the peer once
  networking.interfaces.thunderbolt0 = {
    ipv4.addresses = [ { address = hostAddr; prefixLength = 30; } ];
    mtu = 65520;                                        # thunderbolt-net's max; measure vs 1500/9000
  };
  networking.firewall.trustedInterfaces = [ "thunderbolt0" ];
  environment.systemPackages = with pkgs; [ iperf3 bolt ];
}
```

Bring-up on box 2: install from the flake (USB installer or `nixos-anywhere` over 2.5 GbE),
`nixos-rebuild switch --flake .#lumi-brain2`, copy this box's `~/.config` bits that matter
(run scripts, `~/.cache/deepstrix` symlinks), then rsync the HF snapshot
(`/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash`, 476 GB) onto the plaintext
partition — ~5 min over a healthy USB4 link, ~30 min over 2.5 GbE — and point
`~/.cache/deepstrix/models/dsv4.1f` at it. Then `cargo test -p v4flash-core --release` and the
V4.1 kernel unit sweep (`CARGO_TARGET_DIR=target-v41 … --features v41`) as the smoke test.

## 2. Interconnect: USB4 host-to-host, and what the engine needs from it

**What the link carries (PLAN.md §4, §6).** Expert-parallel: the hub (dGPU box) runs attention,
routers and its own expert share; the remote box runs its expert share. Per decode token, per
layer: one activation push (5120 × Q8 ≈ 5–20 KB) and one expert-output return (5120 f32 ≈ 20 KB)
→ ~40 round trips × ~40 KB ≈ 1.6 MB/token. At the 32 tok/s target that is ~50 MB/s of
bandwidth — trivial — and **40 × RTT of latency on the critical path**. The plan budgets
RTT ≤ 100 µs → 4 ms/token; at 300 µs it is 12 ms/token, a third of the token budget.
Prefill moves ~1024 tokens per layer per step (5–20 MB), where bandwidth matters more.

**Day-one measurements (both directions, both MTUs):**

```
boltctl list                       # peer shows up; authorise: boltctl authorize <uuid>
ip link show thunderbolt0          # up, carrier
iperf3 -s   |   iperf3 -c <peer> -P 1  and  -P 4      # throughput, 1 and 4 streams
ping -c 200 -i 0.005 <peer>        # RTT floor (look at min/avg, not just avg)
# application RTT: a 32 KB request/reply ping-pong over TCP_NODELAY (the engine's real pattern):
#   scripts/netbench/rtt_pingpong.py  (to write: python socket, 2000 iterations, p50/p99)
```

Expected from Linux `thunderbolt-net` reports: 10–20 Gbit/s single-stream throughput
(1.2–2.5 GB/s), RTT typically 100–300 µs — right at the plan's budget, so the number decides
the design: if RTT p50 > 150 µs the engine must hide it (send the activations for layer L+1's
remote experts while the local experts of layer L run, and overlap the remote's return with the
local down-projection) rather than serialise 40 round trips. Also measure with `MTU 1500` (some
`thunderbolt-net` builds throttle at 65520), and with the second USB4 port in case the two
routers differ.

**Fallbacks.** 2.5 GbE: ~300 MB/s, RTT ~150–250 µs — similar latency, 5–8× less bandwidth
(fine for decode, poor for the weight rsync and prefill). A USB4 NIC enclosure (10/25 GbE) buys
bandwidth, not latency. No option removes the 40-RTT structure; only overlap in the engine does.

## 3. Placement once both boxes are up (from PLAN.md §3.3 / §7)

- dGPU box (this one, 96 GB): attention, indexer/compressor stores, hot experts on the dGPU,
  ~60 GB of experts on the iGPU. Box 2 (128 GB): ~110 GB of experts. Native experts total
  289 GB, so ~120 GB stay on SSD behind the LRU tier on **both** boxes — the plaintext partition
  on box 2 and the workqueue-bypassed LUKS here.
- Engram tables (2 × 101 GB fp8) stay SSD-backed on whichever box owns layers 1 and 14's
  gather; with token-only hashing they are prefetched off the critical path.

## 4. Open items

1. Path/repo of the system flake → concrete two-host diff.
2. Which box hosts the 9070 XT (OCuLink): the plan assumes the hub is the dGPU box; the 128 GB
   box as hub would give attention/KV more headroom but moves the larger expert share to 96 GB.
3. After the RTT number: pick serialised vs overlapped remote-expert scheduling for M8.

## 5. PXE netboot bring-up (2026-09-12) — install box 2 over a cable, hands-off

Box 1's uplink is Wi-Fi (`wlan3`); its 2.5 GbE port `eno1` is unused (down, no carrier), so it is
the direct link to box 2 for the install. `second_box_flake.patch` (regenerated) adds:

| file | what |
|---|---|
| `modules/installer-common.nix` | installer content shared by the ISO and the netboot image (latest kernel + firmware, networkd/iwd DHCP on `en*`/`wl*`, root autologin tty1, sshd root key-only with both keys, tailscale, bolt, tools) |
| `modules/installer.nix` | ISO: `installation-cd-minimal.nix` + common (`nix build .#installer-iso` still works) |
| `modules/installer-netboot.nix` | PXE: `installer/netboot/netboot-minimal.nix` + common; firmware forced on (netboot-minimal turns it off at priority 70); zstd level 6 |
| `flake.nix` | `nixosConfigurations.netboot`; `packages.x86_64-linux.netboot-dir` = real copies of `bzImage` (13.5 MiB), `initrd` (1.35 GiB, carries the squashfs store) and `netboot.ipxe` (nixpkgs' script: `kernel bzImage init=<toplevel>/init initrd=initrd …`, relative URLs). Copies, not symlinks: static-web-server 404s symlinks; `auto-optimise-store` hardlinks them back. |
| `modules/pxe-installer.nix` | `lumi.pxeInstaller.{enable,interface="eno1",netbootDir,ipxeBinary}`: networkd unit `05-pxe-eno1` (static `10.43.0.1/24`, `ConfigureWithoutCarrier`, sorts before `10-wan`), dnsmasq on that interface only (`bind-dynamic`, `port=0`, `dhcp-range=10.43.0.10,10.43.0.100,12h`, `dhcp-ignore-clid`, no router/DNS options, TFTP root with `ipxe.efi`+`snp.efi` from `pkgs.ipxe`, `dhcp-match=set:ipxe,175`, `dhcp-boot` → `ipxe.efi` / `http://10.43.0.1:8080/netboot.ipxe`, `log-dhcp`), static-web-server on `10.43.0.1:8080` (socket `FreeBind`), firewall UDP 67/69 + TCP 8080 on `eno1` only |
| `hosts/lumi-brain/default.nix` | imports the module and enables it (temporary, comment says so) |

Verified without root: `nix build .#netboot-dir --offline` OK; all four toplevels
(`lumi-brain`, `lumi-brain2`, `installer`, `netboot`) evaluate (`nix build --dry-run`); box 1's
`boot.kernelParams` identical to pristine; generated dnsmasq.conf passes `dnsmasq --test`; the
networkd/socket unit texts are as intended; the generated dnsmasq.conf was also run live in an
unprivileged user+network namespace on a veth pair: a plain client got a lease with
`boot_file=ipxe.efi`, `siaddr=10.43.0.1`, no router/DNS; an option-175 (iPXE) client with a
different client-id got the same address and `boot_file=http://10.43.0.1:8080/netboot.ipxe`;
`ipxe.efi`/`snp.efi` came back byte-identical over TFTP; sockets were bound to the one interface.
static-web-server served the three files byte-identical on a local port. Not verifiable without
root: the same on the real `eno1`, box 2's firmware, and the rebuild itself.

Steps (details + BIOS notes in the flake README, "PXE netboot"):

1. Box 1: `cd ~/lumi-brain/lumi-brain && sudo nixos-rebuild switch --flake .#lumi-brain`.
   Check `ip -4 addr show eno1` → `10.43.0.1/24`, `curl http://10.43.0.1:8080/netboot.ipxe`.
2. Cable box 2's RJ45 ↔ box 1's `eno1`.
3. Box 2 BIOS: Secure Boot OFF (iPXE + NixOS kernel unsigned), Network Stack / IPv4 PXE ON,
   boot `UEFI: PXE IPv4` from the one-time boot menu (F7 on GMKtec's AMI firmware; Del = setup).
   Prefer the one-time menu over a PXE-first order, which would re-netboot the installer after
   nixos-anywhere's reboot while the module is still on.
4. `journalctl -u dnsmasq -f` on box 1 → lease `10.43.0.<n>` (stable across firmware/iPXE/
   installer/installed phases thanks to `dhcp-ignore-clid`); `journalctl -u static-web-server -f`
   shows the three GETs. Fallback if iPXE stalls on the NIC: `lumi.pxeInstaller.ipxeBinary = "snp.efi"`.
5. `ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no root@10.43.0.<n>` works with
   the baked keys; `nix run /home/claude-code/lumi-brain/lumi-brain#install lumi-brain2 root@10.43.0.<n>`
   (nixos-anywhere; the target is already an installer so no kexec; LUKS passphrase prompt; the
   closure is built on box 1 and copied over the cable); box 2 reboots into lumi-brain2.
6. Installed lumi-brain2 does DHCP on `en*` → leases `10.43.0.<n>` from box 1 again while the
   cable is in: that is the way to reach it (`ssh mimir@10.43.0.<n>`) until the USB4 link is up.
   No default route is handed out, so box 2's internet is its own Wi-Fi. Then set
   `lumi.pxeInstaller.enable = false` and rebuild box 1 (or leave it on; it is harmless).

## 6. Bring-up log (2026-09-12 night)

- PXE path worked once box 2's firmware had **Network Stack enabled** (with it off the
  "network boot" entry falls straight back to setup; the lease seen before that came from the
  firmware network stack / disk OS, hostname `EVO-X3`). Box 2 = GMKtec EVO-X3, 32 cores, Strix
  Halo, one 2 TB NVMe (`/dev/nvme0n1`, Crucial CT2000E100SSD8), MT7925 Wi-Fi, USB4 ×2.
- **Firmware carve-out:** the BIOS dedicates 96 GB to the iGPU (`amdgpu: VRAM 98304M`), the OS
  sees only 31 GB of 128 → set the iGPU / UMA frame buffer to the minimum (512 MB) at the first
  post-install reboot; the lumi-brain2 config (GTT 128 GiB) assumes that.
- **USB4 host-to-host link is up** (both `thunderbolt0` carrier; IPv6 link-local reachable with
  no config). Measured with box 1 still at MTU 1500 (bug below) and box 2 at 65520:

  | metric | value |
  |---|---|
  | ping RTT (200 × 5 ms, box 1 → box 2) | avg 0.172 ms, min 0.072, max 0.786 |
  | iperf3 box 2 → box 1, 1 stream | 8.05 Gbit/s |
  | iperf3 box 1 → box 2, 1 stream | 7.20 Gbit/s |
  | iperf3 box 2 → box 1, 4 streams | 8.9 Gbit/s |

  ~1 GB/s at MTU 1500 (per-packet CPU bound); re-measure at 65520 both ends after the fix. RTT
  is ~1.7× the plan's 100 µs budget at the ICMP level; the application-level 32 KB ping-pong
  (`scripts/netbench/rtt_pingpong.py`) is the number that decides M8 scheduling.
- **Bug found:** `modules/interconnect.nix`'s `30-interconnect.network` never applied on box 1:
  thunderbolt-net interfaces carry an `enx<mac>` altname, so networking.nix's `10-wan`
  (`Name=en* eth*`) matched first and put thunderbolt0 on DHCP (state "configuring", no address,
  MTU 1500). Fixed by renaming the unit to `05-interconnect` (patch regenerated); needs a
  `nixos-rebuild switch` on box 1.
- `hosts/lumi-brain2/default.nix` now also authorises Claude's key for the `claude-code` user
  (box 1 keeps its policy), so the engine work on box 2 can be driven over the link.
- Install command (box 1, root shell; the installer accepts Claude's key, so root borrows it):
  `nix run path:/home/claude-code/lumi-brain/lumi-brain#install -- lumi-brain2 root@10.43.0.95 -i /home/claude-code/.ssh/id_ed25519`
- **Application-level RTT (scripts/netbench/rtt_pingpong.py, TCP_NODELAY request/reply over
  IPv6 link-local, 2000 iterations, box 1 at MTU 1500 / box 2 at 65520, installed box 2):**

  | payload | p50 | p90 | p99 | ×40 per token (p50) |
  |---|---|---|---|---|
  | 4 KB | 130 µs | 135 µs | 211 µs | 5.2 ms |
  | 20 KB | 325 µs | 386 µs | 2.6 ms | 13.0 ms |
  | 32 KB | 329 µs | 346 µs | 1.1 ms | 13.2 ms |

  Above the 100 µs budget of PLAN §4 at the activation-push size (5–20 KB), so the M8 design is
  the overlapped one (send layer L+1's remote activations while layer L's local experts run,
  return path overlapped with the local down-projection), not 40 serialised round trips.
  Re-measure at MTU 65520 both ends after the box-1 rebuild; p99 tails (1–2.6 ms) are
  scheduler/interrupt noise worth a `nohz_full`/IRQ-affinity look later.
- Box 2 first boot: 130 GB visible after the UMA fix, `thunderbolt0` 10.99.0.2/30 + MTU 65520
  applied (the `05-interconnect` rename works), `/weights` 1.2 TB ext4, `/persist` on LUKS.
  Firewall admits ssh only on tailscale0 + thunderbolt0, so reach it from box 1 over the link
  (IPv6 link-local until box 1 has its 10.99.0.1) or `ssh -J mimir@lumi-brain
  mimir@fe80::…%thunderbolt0` from the laptop. `claude-code`'s home dir was not created by
  activation and `/weights` is root-owned — one-time `mkdir/chown` as mimir (sudo).
- **NAPI busy polling (2026-09-13 02:49, `sysctl net.core.busy_poll=100 net.core.busy_read=100`
  on both boxes, endpoints pinned with taskset, blocking recv):**

  | payload | p50 before → after | p90 after | p99 after | ×40 per token |
  |---|---|---|---|---|
  | 4 KB | 130 → **34 µs** | 128 µs | 676 µs | **1.4 ms** |
  | 20 KB | 340 → **86 µs** | 339 µs | 420 µs | 3.4 ms |
  | 32 KB | 332 → 332 µs | 401 µs | 821 µs | 13.3 ms |

  The median was the receive interrupt path, not scheduler wakeups (userspace spinning alone
  had trimmed only the tails). 32 KB does not benefit at a 100 µs window because the reply
  arrives after the poll gives up; try 500 µs. Now in `modules/interconnect.nix`
  (`boot.kernel.sysctl`, 500 µs) for both hosts. Decode's 40 serialised round trips at the
  activation size fit the 4 ms budget with this alone; native XDP / AF_XDP in thunderbolt-net
  remains the lever for bandwidth (8 Gbit/s today) and a ~30 µs floor.
- **busy_read=500 + explicit 4 MB socket buffers + TCP_QUICKACK (2026-09-13 02:55):**

  | payload | p50 | p90 | p99 | ×40 per token |
  |---|---|---|---|---|
  | 4 KB | **32 µs** | 34 µs | 45 µs | 1.3 ms |
  | 20 KB | **67 µs** | 72 µs | 90 µs | 2.7 ms |
  | 32 KB | **97 µs** | 101 µs | 182 µs | 3.9 ms |
  | 64 KB (two segments at MTU 65520) | 540 µs | 548 µs | 2.8 ms | 21.6 ms |

  With the default 16 KB initial send buffer a 32 KB write is split and the second half waits
  on the first's ACK (p50 529 µs, p90 1.5 ms at the 500 µs window); explicit SO_SNDBUF/RCVBUF
  fix it. Anything above one segment (> 65516 B payload) still pays ~500 µs — keep messages
  under the MTU. Transport recipe for the engine: persistent TCP, TCP_NODELAY, SO_SNDBUF/RCVBUF
  ≥ 1 MB, TCP_QUICKACK, blocking recv on a pinned thread with busy_read=500 (in the flake).
  Decode's 40 round trips at the activation size are 1.3–2.7 ms/token serialised, i.e. inside
  the 4 ms budget before any overlap.
- **UDP iperf (2026-09-13 02:59, box loaded ~35):** 8.5–8.6 Gbit/s received box 2 → box 1 at 8 KB
  and 63 KB datagrams with 3.6–13 % receiver-side loss (the sender offers ~9–10 Gbit/s); 4 UDP
  streams 8.1 Gbit/s; box 1 → box 2 5.8 Gbit/s (box 1 was the loaded one). Same ceiling as TCP,
  independent of stream count → the cap is the receive path (one ring / NAPI context or the NHI
  DMA rate), not TCP. Whether it is CPU (→ XDP/AF_XDP would lift it) or DMA (→ it would not)
  needs the per-core measurement on a QUIET box (softirq accounting is hidden under busy_read).
- **QUIET-BOX per-core measurement (2026-09-13 03:08, both boxes idle, plain TCP, 1 MB writes):**
  box 2 → box 1 **8.88 Gbit/s** identically with the receiving socket's busy-poll on
  (`SO_BUSY_POLL=500`) and off (`SO_BUSY_POLL=0`). With busy-poll off, so that NAPI cost is
  visible as softirq on the RX ring's IRQ core, the receiver's busiest core is the IRQ core at
  **40 % (39 % softirq)**, the `recv` thread 3 %; the sender's busiest cores are 19 % and 18 %
  (17 % softirq TX completion). 91 K thunderbolt IRQs and 46 K NET_RX softirqs over 6 s. No core
  on either box is anywhere near saturation, so the ~8.9 Gbit/s ceiling is the host-to-host DMA
  path (NHI ring / inter-domain tunnel), not per-frame CPU. **Consequence: XDP / AF_XDP would not
  raise bandwidth on this link** (it removes skb/stack cost that is only 40 % of one core);
  its only remaining value would be latency below the 32 µs busy-poll RTT, which the engine
  does not need. Decision: stay on TCP with the busy-poll recipe; treat 1.1 GB/s as the link.
- **Frame-rate sweep (2026-09-13 03:13, quiet boxes, UDP box 2 → box 1, unlimited rate):**

  | datagram | receiver | delivered datagrams/s | loss |
  |---|---|---|---|
  | 512 B | 1.75 Gbit/s | 428 K/s | 56 % |
  | 1400 B | 4.72 Gbit/s | 421 K/s | 44 % |
  | 4000 B | 8.08 Gbit/s | 252 K/s | 0 % |
  | 8000 B (2 frames) | 8.83 Gbit/s | 276 K frames/s | 0 % |

  thunderbolt-net moves fixed 4 KB frames, one NHI ring descriptor each (the descriptor's
  length field is 12 bits). The receive side fits `t_frame ≈ 2.2 µs + bytes / 2.6 GB/s`:
  a per-descriptor turnaround of ~2 µs (serial descriptor fetch / write-back on the host
  interface's DMA engine) plus a byte rate of ~21 Gbit/s. At the 4 KB descriptor cap that is
  3.7 µs per frame = 270 K frames/s = 1.1 GB/s — the measured ceiling. The sender's engine
  pushed 965 K frames/s at 512 B, so the limit is the receiving NHI, consistent with the
  receive-side loss seen earlier. Nothing above the driver (XDP, AF_XDP, userspace stacks)
  touches this; the only levers are inside the thunderbolt driver/hardware (a second ring
  pair per direction if the per-descriptor cost is per ring, or multi-descriptor frames if it
  is per frame) — both kernel work, both unverified. Full duplex uses separate rings, so the
  two directions should add (box 1 → box 2 measured 5.8 Gbit/s while box 1 was loaded).

## OPEN OPTION: make box 2 (128 GB) primary and move the dGPU there (user, 2026-09-13)

**Assessment.** For the two-box expert-parallel design the goal requires, this is roughly
**perf-neutral**:
- Total resident experts is unchanged. Either primary carries the same non-expert overhead
  (backbone + KV + scratch), so 96+128 GB minus that overhead gives the same total residency; only
  the split changes. A lopsided split is not automatically better — PLAN §7b wants the two expert
  branches balanced by latency, so you would tune it back.
- Prefill does not move: it is dGPU-bound, the dGPU performs identically wherever it sits, and it
  hangs off the same ~7 GB/s OCuLink. The 1000 tok/s @100K target comes from CED (20 encoder layers
  over the prompt), not from which chassis hosts the card.
- The link is not the constraint for expert-parallel decode (activations are tens of KB/token
  against a measured 1.1 GB/s, 32 us RTT).

**Where it DOES win: as single-box insurance.** If two-box expert-parallel does not pan out (box 2
iGPU unusable, or the expert-executor work too costly), the fallback is single-box, where the dGPU
box's RAM *directly* sets expert residency and hence decode (residency-limited, ~10-22 tok/s).
There 96 -> 128 GB is a real gain.

**Costs.** Box 2 has no ROCm toolchain, no repo, no internet today. Box 1 holds the working
dual-GPU stack, the production V4-Flash server, the flake config and tooling. PHYSICAL prerequisite: **CONFIRMED by the user 2026-09-13** — box 2 has a suitable OCuLink slot for
the 9070 XT, so the swap is executable whenever the trigger fires.

**DECISION TRIGGER (do not re-litigate before this):** the box-2 readiness verdict (can it build
the kernels and run a HIP kernel on its gfx1151 iGPU?).
- box 2 iGPU OK  -> keep current topology; spend effort on the expert tier + two-box engine.
- box 2 iGPU bad -> seriously consider the swap, since single-box-on-the-big-box becomes the path.

### Stress-test of the "neutral" claim (2026-09-13, after the slot was confirmed)
- Both boxes are Strix Halo => IDENTICAL memory bandwidth (~256 GB/s) regardless of 96 vs 128 GB
  capacity, so the primary's bandwidth-bound iGPU MoE (measured 214 GB/s on experts) does not
  improve by moving.
- BOTH boxes already hold a local copy of the weights, so the SSD miss tail is served locally
  either way.
- Would the swap rescue SINGLE-box to the 30 tok/s goal? No. 128 GB primary gives roughly 37%
  expert residency vs ~24%; that lifts decode but still lands short of 30 even with DSpark. So
  two-box expert-parallel remains REQUIRED; the swap only raises the fallback's floor.
- Sequencing caveat: moving the dGPU takes the production V4-Flash service with it (that service
  needs the card). Do it deliberately, not mid-session.

## BOX 2 READINESS: **READY** (2026-09-13) — decision trigger RESOLVED
Verified on lumi-brain2 itself:
- `rocminfo` Agent 2 = **gfx1151** (Radeon 8060S, 40 CU, wave32), **128 GiB GPU-addressable pool**
  (`amdgpu.gttsize=131072` + `no_system_mem_limit=Y` already live on its kernel cmdline — exactly what
  expert residency needs). `/dev/kfd` + `/dev/dri/renderD128` world-RW, no group/permission issue.
- `v4flash-kernels` **builds clean on box 2** (1m18s, 69 crates, every HIP kernel via hipcc 7.2.53211 /
  ROCm 7.2.3). Real kernels EXECUTE: `rms_norm_oracle` passed ("using device 0 (gfx1151)"), all 9
  `sampler` tests passed, and **HIP graph capture works** (capture+replay 2.00 us/launch).
  (`hip_graph_smoke` fails only because it hardcodes `want_integrated=false` i.e. demands a dGPU — not a defect.)
- Weights readable at /weights/dsv4.1f (476 GB), 124 GB RAM, 703 GB free.

**How the toolchain got there — signed-closure copy over USB4, ZERO system changes on box 1.**
`claude-code` is not in box 2's `trusted-users` and there is no passwordless sudo/root-ssh on either
box, so unsigned `nix copy` AND the NAT fallback were both closed. But nix accepts paths from an
untrusted user if they carry a trusted signature: 183 of the 199 dev-shell closure paths (6.5 GiB) are
signed by `cache.nixos.org-1` and copied as an ordinary user. The 16 unsigned paths are tiny local bits
(stdenv hooks + the `rocm-merged-deepstrix` symlink farm); the farm was rsynced to `~/rocm-merged` and
the env replayed via `~/b2-dev.sh` (PATH + ROCM_PATH/HIP_PATH/LIBRARY_PATH/HIP_CLANG_PATH/
HIP_DEVICE_LIB_PATH/HSA_PATH/PKG_CONFIG_PATH). Cargo needs box 1's `~/.cargo/registry` (rsynced) + `--offline`.
Box-1 residue: a gcroot at `/tmp/b2-devprofile` pinning the dev closure from GC (tmpfs; delete to release).
Caveat: box 2 has no internet, so it is pinned to the copied closure — a NEW dependency needs another copy.

## DECISION: KEEP THE CURRENT TOPOLOGY (box 1 primary + dGPU)
The trigger above resolved "box 2 iGPU OK", so per the stated criterion the dGPU stays on box 1 and the
effort goes to the expert tier + the two-box engine. The swap option is CLOSED unless two-box
expert-parallel later proves unworkable (its OCuLink slot is confirmed, so it remains executable).
