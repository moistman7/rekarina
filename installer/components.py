"""Graphics components by hardware: which upscaler libraries this machine can use, where they come from, and the
first settings that suit its graphics card.

The game loads DLSS, DLSS Frame Generation and AMD FSR from libraries that lie beside Rekarina.exe. They are NVIDIA's
and AMD's, under their licenses, so no Rekarina release carries them: the setup's components step (oot.py,
Components) downloads the ones the detected graphics card can use, each from its vendor's own public repository at a
pinned version, and checks every file against the size and SHA-256 pinned below. Without them the game runs in its
base mode (the sharp picture with FSR 1), so a machine that is offline still gets a game; the launcher then offers
to try again.

    group    graphics card             files
    fsr      any                       amd_fidelityfx_loader_vk.dll, amd_fidelityfx_upscaler_vk.dll and
                                       amd_fidelityfx_framegeneration_vk.dll (FSR SDK 2.4: FSR Upscaling 4.1 on the
                                       Radeon cards that run it, FSR 3.1 on every other card, and frame generation)
    dlss     NVIDIA RTX 20 or newer    nvngx_dlss.dll, sl.interposer.dll, sl.common.dll and sl.dlss.dll
                                       (DLSS Super Resolution / DLAA through NVIDIA Streamline)
    dlssg    NVIDIA RTX 40 or newer    nvngx_dlssg.dll, sl.dlss_g.dll, sl.pcl.dll, sl.reflex.dll and
                                       NvLowLatencyVk.dll (DLSS Frame Generation and Reflex); needs the dlss group

    python -m installer.components [--dest DIR] [--fetch] [--gpu "NAME;VRAM_GB"]
        prints the detected card, its groups, its preset and, with --dest, the files that folder lacks; --fetch
        downloads them there

detect() names the card: Win32_VideoController through PowerShell for the adapters present, the display class
registry key for the real video memory (the WMI value is a 32-bit number and stops at 4 GiB), nvidia-smi when the
driver brought it. preset() is the table of first settings; package_oot.py writes them into a new rekarina.ini only.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import net

WIN = os.name == 'nt'
GIB = 1 << 30


# ---- the pinned files ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    title: str      # what the window and the log call it
    version: str
    license: str    # the name of the license notice
    link: str       # where the notice is


@dataclass(frozen=True)
class File:
    name: str                  # the file name beside Rekarina.exe
    group: str                 # fsr | dlss | dlssg
    source: str                # key of SOURCES
    url: str                   # the file itself, or the archive that holds it
    size: int
    sha256: str
    member: str = ''           # the path inside the archive, when url is one
    archive_size: int = 0
    archive_sha256: str = ''


SOURCES = {
    'dlss': Source('NVIDIA DLSS', '310.9.16', 'NVIDIA RTX SDKs License',
                   'https://github.com/NVIDIA/DLSS/blob/v310.9.16/LICENSE.txt'),
    'dlssg': Source('NVIDIA DLSS Frame Generation', '310.9.16', 'NVIDIA RTX SDKs License',
                    'https://github.com/NVIDIA/DLSS/blob/v310.9.16/LICENSE.txt'),
    'streamline': Source('NVIDIA Streamline', '2.14.16', 'MIT License (Streamline), NVIDIA RTX SDKs License (Reflex)',
                         'https://github.com/NVIDIA-RTX/Streamline/blob/v2.14.16/license.txt'),
    'fsr': Source('AMD FSR SDK (FSR 4.1 and 3.1)', '2.4.0', 'MIT License',
                  'https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v2.4.0/Kits/FidelityFX/docs/license.md'),
}
DLSS_RAW = 'https://raw.githubusercontent.com/NVIDIA/DLSS/v{}/lib/Windows_x86_64/rel/{}'
FSR_RAW = 'https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/signedbin/{}'
SL_ZIP = 'https://github.com/NVIDIA-RTX/Streamline/releases/download/v2.14.16/streamline-sdk-v2.14.16.zip'
SL_ZIP_SIZE = 276016170
SL_ZIP_SHA256 = '32041e51db0cab709a271b72a352281b690c2abbd669a827809401a5dd23956b'   # the digest GitHub lists for it


def streamline(name, size, sha256, group='dlssg'):
    return File(name, group, 'streamline', SL_ZIP, size, sha256, f'bin/x64/{name}', SL_ZIP_SIZE, SL_ZIP_SHA256)


# The versions the game was developed and tested with. The Streamline files are the production (signed) ones of the
# SDK archive's bin/x64, not those of bin/x64/development.
FILES = (
    # The loader and the two effect libraries it loads from beside itself (runtime/vk_ffx.cpp).
    File('amd_fidelityfx_loader_vk.dll', 'fsr', 'fsr', FSR_RAW.format('amd_fidelityfx_loader_vk.dll'), 39184,
         '38a56a01fb3565c0757d9f5c1e4e69de4b65e5f2e5769b323b200dc1f142bab9'),
    File('amd_fidelityfx_upscaler_vk.dll', 'fsr', 'fsr', FSR_RAW.format('amd_fidelityfx_upscaler_vk.dll'), 36135696,
         '4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7'),
    File('amd_fidelityfx_framegeneration_vk.dll', 'fsr', 'fsr', FSR_RAW.format('amd_fidelityfx_framegeneration_vk.dll'),
         36806928, 'd002dabff8c5e71eaa715fc99438cd4dc81230262a7897b16bc402efec3498b4'),
    File('nvngx_dlss.dll', 'dlss', 'dlss', DLSS_RAW.format(SOURCES['dlss'].version, 'nvngx_dlss.dll'), 58956912,
         'd00b5b0ab29c3ca101b70c3a5ecb3a01b7ef33766a3383a28a18623083e6123e'),
    streamline('sl.interposer.dll', 652928, 'b1d7395aebd7b9722841d64dcf7433680020b6a8e421f851c16d5ba0c2182edf', 'dlss'),
    streamline('sl.common.dll', 843904, '38eea3b4209afc455636ea372b5933d65fcc13becec6c84d101dcfc4d8648f75', 'dlss'),
    streamline('sl.dlss.dll', 422016, '18871eea87c90345da97d8109faa81ea7d5840e8d99b51b55b0e46cfa177f84e', 'dlss'),
    File('nvngx_dlssg.dll', 'dlssg', 'dlssg', DLSS_RAW.format('310.9.16', 'nvngx_dlssg.dll'), 7506544,
         '8d59d49b36410dd6b62fef2503a5871d347415bd85d943fa0848563a95dd50c7'),
    streamline('sl.dlss_g.dll', 637056, '3ed9538df724993bea8a050195a6a78b9c8c8a566d5b05ade477aab3c390e90f'),
    streamline('sl.pcl.dll', 360064, '5698c38e722318681f215c9fe68e45b9bedd51378acc699d8eee26e961e57770'),
    streamline('sl.reflex.dll', 388736, '443577e869352436ff7ec8f4d9c90418d3e95368436ce219261e0f4f07a38122'),
    streamline('NvLowLatencyVk.dll', 57840, '2a77dc3e1c724b7eea5755be0ae7423752e79a2459fae72181a9f00e3507e5d6'),
)
GROUPS = ('fsr', 'dlss', 'dlssg')


def pins_digest(files=FILES):
    """One string over every pin: the components step runs again when a release pins other files."""
    return hashlib.sha256('\n'.join(f'{f.name} {f.url} {f.member} {f.sha256}' for f in files).encode()).hexdigest()


# ---- the graphics card --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Gpu:
    name: str
    vendor: str            # nvidia | amd | intel | other
    vram: int = 0          # bytes of dedicated video memory; 0 when it is not known
    integrated: bool = False

    @property
    def vram_gb(self):
        """Whole gigabytes as the box says them: 12 for a card that reports 12,282 MiB."""
        return round(self.vram / GIB) if self.vram else 0

    @property
    def rtx(self):
        return rtx_generation(self.name) if self.vendor == 'nvidia' else None

    def to_json(self):
        return {'name': self.name, 'vendor': self.vendor, 'vram': self.vram, 'integrated': self.integrated}

    @staticmethod
    def from_json(d):
        try:
            return Gpu(str(d['name']), str(d['vendor']), int(d.get('vram') or 0), bool(d.get('integrated')))
        except (KeyError, TypeError, ValueError):
            return None


PCI_VENDORS = {'10DE': 'nvidia', '1002': 'amd', '1022': 'amd', '8086': 'intel'}


def vendor_of(name, pnp=''):
    m = re.search(r'VEN_([0-9A-F]{4})', (pnp or '').upper())
    if m and m.group(1) in PCI_VENDORS:
        return PCI_VENDORS[m.group(1)]
    n = name.upper()
    if any(w in n for w in ('NVIDIA', 'GEFORCE', 'QUADRO', 'RTX', 'GTX')):
        return 'nvidia'
    if any(w in n for w in ('AMD', 'RADEON', 'FIREPRO')):
        return 'amd'
    if any(w in n for w in ('INTEL', 'IRIS', 'ARC(TM)')):
        return 'intel'
    return 'other'


def is_integrated(name, vendor):
    """A processor's own graphics: it shares the system memory, and a discrete card beside it is the one to use."""
    n = name.upper().replace('(TM)', '').replace('(R)', '')
    if vendor == 'intel':
        return not re.search(r'\bARC\s+(PRO\s+)?[AB]\d{3}', n)   # Arc A770, B580: cards; "Arc Graphics": a processor's
    if vendor == 'amd':
        # "AMD Custom GPU 0405" and "0932" are the Steam Deck's
        return bool(re.search(r'RADEON\s+(GRAPHICS|\d{3}M\b|VEGA|R\d\s+GRAPHICS|HD\s+GRAPHICS)|VEGA\s+\d+|CUSTOM\s+GPU', n))
    return False


def rtx_generation(name):
    """20, 30, 40, 50... for an NVIDIA RTX card by its name, None for one without RTX (no DLSS).

    GeForce: the first two digits (RTX 2060, 3080 Ti, 4070 SUPER, 5070). Workstation cards carry their generation in
    words: Quadro RTX and TITAN RTX are the 20 series, RTX A2000 to A6000 the 30 series, "Ada" the 40 series,
    "Blackwell" the 50 series. An RTX name this does not recognise counts as the 20 series: DLSS, no frame generation."""
    n = name.upper().replace('(TM)', '').replace('(R)', '')
    if 'RTX' not in n:
        return None
    if 'QUADRO RTX' in n or 'TITAN RTX' in n:
        return 20
    if 'BLACKWELL' in n:
        return 50
    if re.search(r'\bADA\b', n):
        return 40
    if re.search(r'RTX\s*(PRO\s*)?A\d{3,4}\b', n):
        return 30
    m = re.search(r'RTX\s*(\d{4})\b', n)
    if m and 20 <= int(m.group(1)) // 100 <= 99:
        return int(m.group(1)) // 100
    return 20


def parse_cim(text):
    """[(name, PNP device id, AdapterRAM)] from `Get-CimInstance Win32_VideoController | ConvertTo-Json` (one object
    for one adapter, a list for several)."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    rows = data if isinstance(data, list) else [data]
    found = []
    for r in rows:
        if isinstance(r, dict) and r.get('Name'):
            ram = r.get('AdapterRAM')
            found.append((str(r['Name']).strip(), str(r.get('PNPDeviceID') or ''), int(ram) if isinstance(ram, (int, float)) else 0))
    return found


def parse_nvidia_smi(text):
    """{name: bytes} from `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` (MiB)."""
    found = {}
    for line in text.splitlines():
        name, _, mib = line.rpartition(',')
        if name.strip() and mib.strip().isdigit():
            found[name.strip()] = int(mib) << 20
    return found


def combine(cim, registry, smi):
    """The adapters as Gpu records. cim: parse_cim's rows; registry: [(driver description, matching device id, bytes
    of HardwareInformation.qwMemorySize)]; smi: parse_nvidia_smi's names. The registry also lists cards that were
    removed, so it only adds the memory to an adapter WMI names; without WMI (PowerShell blocked) its rows with a
    memory size stand in."""
    def norm(s):
        return re.sub(r'\s+', ' ', s).strip().lower()

    rows = cim or [(desc, dev, 0) for desc, dev, qw in registry if qw]
    gpus, seen = [], set()
    for name, pnp, ram in rows:
        vendor = vendor_of(name, pnp)
        vram = max([qw for desc, _, qw in registry if norm(desc) == norm(name)] + [0])
        if not vram:
            vram = ram if 0 < ram < 0xfff00000 else 0   # a full 32-bit value means "4 GiB or more": unknown
        for smi_name, smi_bytes in smi.items():
            if norm(smi_name) in norm(name) or norm(name) in norm(smi_name):
                vram = smi_bytes
        gpu = Gpu(name, vendor, vram, is_integrated(name, vendor))
        if gpu not in seen:
            seen.add(gpu)
            gpus.append(gpu)
    return gpus


def pick(gpus):
    """The card the game will use: a known vendor's discrete one with the most memory; None for an empty list."""
    return max(gpus, key=lambda g: (g.vendor != 'other', not g.integrated, g.rtx or 0, g.vram), default=None)


def run_quiet(command, timeout=20):
    """The stdout of a helper program, '' when it is missing, fails or takes too long."""
    try:
        r = subprocess.run(command, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if WIN else 0)
        return r.stdout.decode('utf-8', 'replace') if r.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def registry_adapters():
    if not WIN:
        return []
    import winreg
    found = []
    base = r'SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(root, i)
                except OSError:
                    break
                if not re.fullmatch(r'\d{4}', sub):
                    continue
                try:
                    with winreg.OpenKey(root, sub) as key:
                        def value(name):
                            try:
                                return winreg.QueryValueEx(key, name)[0]
                            except OSError:
                                return None
                        desc, qw = value('DriverDesc'), value('HardwareInformation.qwMemorySize')
                        if isinstance(qw, bytes):   # some drivers store the number as binary
                            qw = int.from_bytes(qw[:8], 'little')
                        if isinstance(desc, str):
                            found.append((desc, str(value('MatchingDeviceId') or ''), int(qw) if isinstance(qw, int) else 0))
                except OSError:
                    continue
    except OSError:
        pass
    return found


def detect_all():
    """Every display adapter of this machine (see the module text); [] when nothing could be asked."""
    cim = []
    if WIN:
        script = ('[Console]::OutputEncoding = [Text.Encoding]::UTF8; Get-CimInstance Win32_VideoController | '
                  'Select-Object Name, PNPDeviceID, AdapterRAM | ConvertTo-Json -Compress')
        system = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'WindowsPowerShell' / 'v1.0' / 'powershell.exe'
        cim = parse_cim(run_quiet([str(system) if system.is_file() else 'powershell', '-NoProfile', '-NonInteractive',
                                   '-Command', script]))
    smi = parse_nvidia_smi(run_quiet(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 10))
    gpus = combine(cim, registry_adapters(), smi)
    if not gpus and smi:   # not Windows, or neither WMI nor the registry answered
        gpus = [Gpu(name if 'NVIDIA' in name.upper() else 'NVIDIA ' + name, 'nvidia', vram) for name, vram in smi.items()]
    return gpus


def detect():
    """The graphics card the game will use, or None."""
    return pick(detect_all())


def gpu_key(gpu):
    return f'{gpu.vendor}|{gpu.name}|{gpu.vram_gb}' if gpu else 'unknown'


# ---- what a card gets ---------------------------------------------------------------------------------------------

def groups_for(gpu):
    """The groups this card can use. An unknown card gets FSR 3.1 only: it runs on any Vulkan card. Off Windows none:
    every file here is a Windows library (AMD's FSR SDK, NVIDIA's DLSS and Streamline), which the game cannot load there
    (a Linux setup downloaded 100 MB of them; DLSS on Linux goes through the driver's own NGX library, runtime/vk_dlaa.c)."""
    if os.name != 'nt':
        return []
    groups = ['fsr']
    rtx = gpu.rtx if gpu else None
    if rtx and rtx >= 20:
        groups.append('dlss')
    if rtx and rtx >= 40:
        groups.append('dlssg')
    return groups


def wanted(gpu, files=FILES):
    groups = groups_for(gpu)
    return [f for f in files if f.group in groups]


def missing(dest, gpu, files=FILES):
    """The names of the files this card can use that are not in dest."""
    return [f.name for f in wanted(gpu, files) if not (Path(dest) / f.name).is_file()]


def usable(dest, files=FILES):
    """The groups whose every file lies in dest (whatever its version: a newer one the player put there counts)."""
    dest = Path(dest)
    have = {g for g in GROUPS if [f for f in files if f.group == g]
            and all((dest / f.name).is_file() for f in files if f.group == g)}
    if 'dlss' not in have:
        have.discard('dlssg')   # frame generation runs on DLSS's motion vectors
    return have


def preset(gpu, have=GROUPS):
    """The first settings for a graphics card: {'id', 'tex', 'settings'}.

    settings are rekarina.ini lines (runtime/cfg.c: lower-case keys, the values its table allows). tools/package_oot.py
    writes them into a NEW rekarina.ini only: a settings file that exists is the player's and is never touched. `have`
    names the component groups that are in the game folder; a preset never switches on what is not there.

        id            card                                              dlaa  taa    dlss         framegen
        rtx_base      NVIDIA RTX 20 or newer with DLSS present           0     dlss   -            off
        base          AMD, Intel, NVIDIA GTX, unknown                    0     fsr3   -            off
        handheld      a processor's own graphics (handhelds,             1     fsr1   performance  off
                      laptops without a card)

      - rtx_base: DLSS is the selected backend, but temporal upscaling and frame generation stay off until enabled.
        Cards with 4 GB or less use FSR 1 at 67 % as described below.
      - base: the game's own sharp picture with FSR 1, no temporal upscaler. taa = fsr3 only chooses what the menu's
        DLSS / DLAA row would switch on (FSR 3.1, which runs on every card); it stays off.
      - handheld: the 3D scene at half of the game's 2560x1440 picture, 1280x720, scaled up by FSR 1 on the scene
        (taa = fsr1 at the performance mode's 50 %): about the panel of a Steam Deck (1280x800) and what a ROG Ally's
        graphics carry at 1080p. Until 2026-11-15 render_scale = 50 lowered the scene the same way and the game's own
        scaler, softer, brought it back. A first setting only, chosen by the scene's pixel count; not measured on those
        machines.

    Video memory, on top of any of them (unknown memory changes nothing):

        tex   memory          tex_quality   and
        auto  8 GB or more    auto
        half  under 8 GB      half
        quarter  4 GB or less quarter       base and rtx_base: FSR 1 at the quality mode (67 %)
    """
    rtx = gpu.rtx if gpu else None
    gb = gpu.vram / GIB if gpu and gpu.vram else None
    if rtx and rtx >= 20 and 'dlss' in have:
        ident, settings = 'rtx_base', {'dlaa': '0', 'taa': 'dlss', 'framegen': 'off'}
    else:
        ident, settings = 'base', {'dlaa': '0', 'taa': 'fsr3', 'framegen': 'off'}
        if gpu and gpu.integrated:
            ident, settings = 'handheld', {'dlaa': '1', 'taa': 'fsr1', 'dlss': 'performance', 'framegen': 'off'}
    tex = 'auto'
    if gb is not None and gb <= 4.5:      # a "4 GB" card reports a little under or over 4 GiB
        tex = 'quarter'
        if ident in ('base', 'rtx_base'):   # FSR 1 on a scene of 67 %: the quality mode
            settings.update({'dlaa': '1', 'taa': 'fsr1', 'dlss': 'quality'})
        # handheld: already at 50
    elif gb is not None and gb < 7.5:     # 6 GB cards; an "8 GB" card reports 7.9 to 8.2 GiB and keeps auto
        tex = 'half'
    settings['tex_quality'] = tex
    return {'id': ident, 'tex': tex, 'settings': settings}


# ---- downloading --------------------------------------------------------------------------------------------------

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def put(data, target, f):
    """A downloaded member under its name, after the checks."""
    if len(data) != f.size or hashlib.sha256(data).hexdigest() != f.sha256:
        raise net.DownloadError(f'{f.name}: {len(data)} bytes, SHA-256 {hashlib.sha256(data).hexdigest()[:16]}, not the pinned file')
    part = target.with_name(target.name + '.part')
    part.write_bytes(data)
    part.replace(target)


def fetch_members(files, dest, temp, step, cancel=None, tries=net.TRIES):
    """Files that are members of one archive: each read out of it with range requests; when the server does not do
    ranges, the whole archive once into temp. step(file) is called before each. Returns {name: reason} of failures."""
    failed = {}
    first = files[0]
    remote = net.RemoteZip(first.url, first.archive_size, tries=tries)
    try:
        for f in files:
            step(f)
            if cancel and cancel():
                raise net.Cancelled
            try:
                put(remote.read(f.member), Path(dest) / f.name, f)
            except (net.DownloadError, *net.NET_ERRORS) as e:
                failed[f.name] = str(e) or type(e).__name__
                remote.tries = 1   # as install does for whole files: the rest gets one try each
        return failed
    except net.RemoteZip.RangeUnsupported:
        pass
    archive = Path(temp) / Path(first.url).name
    try:
        net.fetch_file(first.url, archive, first.archive_size, first.archive_sha256, cancel=cancel, tries=tries)
        with zipfile.ZipFile(archive) as z:
            for f in files:
                step(f)
                try:
                    put(z.read(f.member), Path(dest) / f.name, f)
                    failed.pop(f.name, None)
                except (net.DownloadError, KeyError, OSError, zipfile.BadZipFile) as e:
                    failed[f.name] = str(e) or type(e).__name__
    except (net.DownloadError, zipfile.BadZipFile, *net.NET_ERRORS) as e:
        failed.update({f.name: str(e) or type(e).__name__ for f in files if not (Path(dest) / f.name).is_file()})
    finally:
        archive.unlink(missing_ok=True)
    return failed


def install(dest, gpu, temp, progress=None, log=None, cancel=None, files=FILES):
    """Put the files this card can use into dest. A file that is there already stays: the pinned one counts as
    present, any other as the player's own ('kept'). Nothing here raises for a failed download: the result says what
    is missing and why.

    progress(group, done bytes, total bytes); log(text). Returns {'groups', 'installed', 'present', 'kept',
    'failed': {name: reason}}."""
    dest, temp = Path(dest), Path(temp)
    log = log or (lambda text: None)
    result = {'groups': groups_for(gpu), 'installed': [], 'present': [], 'kept': [], 'failed': {}}
    need = []
    for f in wanted(gpu, files):
        target = dest / f.name
        if not target.is_file():
            need.append(f)
        elif target.stat().st_size == f.size and file_sha256(target) == f.sha256:
            result['present'].append(f.name)
        else:
            result['kept'].append(f.name)
    if not need:
        return result
    dest.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    total, done = sum(f.size for f in need), 0
    tries = net.TRIES

    def announce(f):
        src = SOURCES[f.source]
        log(f'components: {f.name} from {f.url}{" (" + f.member + ")" if f.member else ""}: {src.title} {src.version}, '
            f'{src.license}, {src.link}')
        if progress:
            progress(f.group, done, total)

    archives = {}
    for f in need:
        if f.member:
            archives.setdefault(f.url, []).append(f)
    handled = set()
    for f in need:
        if cancel and cancel():
            raise net.Cancelled
        if f.name in handled:
            continue
        if f.member:
            group = archives[f.url]
            failed = fetch_members(group, dest, temp, announce, cancel, tries)
            handled.update(g.name for g in group)
            got = [g for g in group if g.name not in failed]
        else:
            announce(f)
            base = done
            try:
                net.fetch_file(f.url, dest / f.name, f.size, f.sha256, cancel=cancel, tries=tries,
                               progress=(lambda n, _t, base=base, f=f: progress(f.group, base + n, total)) if progress else None)
                failed, got = {}, [f]
            except net.DownloadError as e:
                failed, got = {f.name: str(e)}, []
        for g in got:
            result['installed'].append(g.name)
            done += g.size
        for name, reason in failed.items():
            result['failed'][name] = reason
            log(f'components: {name} failed: {reason}')
            tries = 1   # the network is down or the source is gone: the rest gets one try each, not four
        if progress:
            progress(f.group, done, total)
    return result


def describe(gpu):
    """'NVIDIA GeForce RTX 5070, 12 GB' for the window and the log."""
    if not gpu:
        return ''
    return f'{gpu.name}, {gpu.vram_gb} GB' if gpu.vram else gpu.name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dest', type=Path, help='the game folder to look into')
    ap.add_argument('--fetch', action='store_true', help='download what --dest lacks')
    ap.add_argument('--gpu', help='pretend the card is this one: "NAME;VRAM_GB" (for tests)')
    a = ap.parse_args(argv)
    if a.gpu:
        name, _, gb = a.gpu.partition(';')
        vendor = vendor_of(name)
        gpu = Gpu(name.strip(), vendor, int(float(gb or 0) * GIB), is_integrated(name, vendor))
        adapters = [gpu]
    else:
        adapters = detect_all()
        gpu = pick(adapters)
    out = {'adapters': [g.to_json() for g in adapters], 'gpu': gpu.to_json() if gpu else None, 'rtx': gpu.rtx if gpu else None,
           'groups': groups_for(gpu)}
    if a.dest:
        if a.fetch:
            out['install'] = install(a.dest, gpu, a.dest / '.components-temp',
                                     progress=lambda g, d, t: print(f'{g}: {d} / {t}', file=sys.stderr, flush=True),
                                     log=lambda text: print(text, file=sys.stderr, flush=True))
            shutil.rmtree(a.dest / '.components-temp', ignore_errors=True)
        out['missing'] = missing(a.dest, gpu)
        out['usable'] = sorted(usable(a.dest))
    out['preset'] = preset(gpu, usable(a.dest) if a.dest else GROUPS)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
