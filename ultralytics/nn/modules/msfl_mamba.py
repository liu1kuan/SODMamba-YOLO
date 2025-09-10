import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from mamba_ssm import Mamba  # optional
	exists_mamba = True
except Exception:  # pragma: no cover
	Mamba = None  # type: ignore
	exists_mamba = False

try:
	from einops import rearrange
	exists_einops = True
except Exception:  # pragma: no cover
	def rearrange(x, pattern, **kwargs):
		return x
	exists_einops = False

try:
	from .mamba_yolo import SS2D  # repo's SS2D implementation
	exists_ss2d = True
except Exception:  # pragma: no cover
	SS2D = None  # type: ignore
	exists_ss2d = False


class DepthwiseSeparableConv(nn.Module):
	def __init__(self, channels: int):
		super().__init__()
		self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
		self.pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
		nn.init.dirac_(self.pw.weight)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.pw(self.dw(x))


class SpectralGating(nn.Module):
	"""Torch FFT-based per-channel complex gating with DW-conv fallback."""

	def __init__(self, channels: int, use_fft: bool = True):
		super().__init__()
		self.use_fft = use_fft
		if use_fft:
			self.alpha_real = nn.Parameter(torch.zeros(1, channels, 1, 1))
			self.alpha_imag = nn.Parameter(torch.zeros(1, channels, 1, 1))
		else:
			self.dw = DepthwiseSeparableConv(channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if not self.use_fft:
			return self.dw(x)
		B, C, H, W = x.shape
		X = torch.fft.rfft2(x, norm="ortho")
		alpha = self.alpha_real + 1j * self.alpha_imag
		Y = X * alpha
		return torch.fft.irfft2(Y, s=(H, W), norm="ortho")


class LocalMamba2D(nn.Module):
	"""Windowed Mamba over 2D features; falls back to DWConv+FFN when mamba-ssm missing."""

	def __init__(self, channels: int, window: int = 8, expand: int = 2):
		super().__init__()
		self.channels = channels
		self.window = window
		self.expand = expand
		self.norm = nn.LayerNorm(channels)
		self.proj_in = nn.Conv2d(channels, channels, 1, 1, 0)
		self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)
		if exists_mamba:
			self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=expand)
		else:
			self.fallback = nn.Sequential(
				nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
				nn.GELU(),
				nn.Conv2d(channels, channels, 1)
			)
			self.ffn = nn.Sequential(
				nn.Conv2d(channels, channels * expand, 1),
				nn.GELU(),
				nn.Conv2d(channels * expand, channels, 1)
			)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		res = x
		x = self.proj_in(x)
		B, C, H, W = x.shape
		w = self.window
		pad_h = (w - H % w) % w
		pad_w = (w - W % w) % w
		if pad_h or pad_w:
			x = F.pad(x, (0, pad_w, 0, pad_h))
			H2, W2 = H + pad_h, W + pad_w
		else:
			H2, W2 = H, W

		if exists_mamba and exists_einops:
			t = rearrange(x, 'b c (nh w1) (nw w2) -> b (nh nw) (w1 w2) c', w1=w, w2=w)
			t = self.norm(t)
			y = self.mamba(t)
			y = rearrange(y, 'b (nh nw) (w1 w2 c1) -> b c1 (nh w1) (nw w2)', nh=H2//w, nw=W2//w, w1=w, w2=w, c1=1)
		else:
			y = self.fallback(x)
			y = self.ffn(y)

		if pad_h or pad_w:
			y = y[:, :, :H, :W]
		y = self.proj_out(y)
		return y + res


class LocalDeformableMamba2D(nn.Module):
	"""Deformable window Mamba over 2D features.

	- Predicts per-pixel offsets then samples features via grid_sample
	- Applies windowized Mamba (or fallback) on the deformed features
	"""

	def __init__(self, channels: int, window: int = 8, expand: int = 2, max_offset_ratio: float = 0.5):
		super().__init__()
		self.channels = channels
		self.window = window
		self.expand = expand
		self.max_offset_ratio = max_offset_ratio  # relative to window size
		self.proj_in = nn.Conv2d(channels, channels, 1, 1, 0)
		self.offset_conv = nn.Conv2d(channels, 2, 3, 1, 1)  # predict dx, dy in pixels (tanh -> scaled)
		self.norm = nn.LayerNorm(channels)
		self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)
		if exists_mamba:
			self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=expand)
		else:
			self.fallback = nn.Sequential(
				nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
				nn.GELU(),
				nn.Conv2d(channels, channels, 1)
			)
			self.ffn = nn.Sequential(
				nn.Conv2d(channels, channels * expand, 1),
				nn.GELU(),
				nn.Conv2d(channels * expand, channels, 1)
			)

	def _make_base_grid(self, B: int, H: int, W: int, device, dtype):
		# grid in normalized coords [-1, 1]
		y = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
		x = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
		grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
		base = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2) with order (x, y)
		return base.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		res = x
		x = self.proj_in(x)
		B, C, H, W = x.shape
		w = self.window
		pad_h = (w - H % w) % w
		pad_w = (w - W % w) % w
		if pad_h or pad_w:
			x = F.pad(x, (0, pad_w, 0, pad_h))
			H2, W2 = H + pad_h, W + pad_w
		else:
			H2, W2 = H, W

		# predict offsets (dx, dy) in pixels and scale -> normalized offsets
		off = torch.tanh(self.offset_conv(x))  # (B, 2, H2, W2)
		max_off_px = max(1, int(self.max_offset_ratio * w))  # relative to window size
		dx = off[:, 0] * max_off_px
		dy = off[:, 1] * max_off_px
		dx_norm = (dx * 2.0) / max(W2 - 1, 1)
		dy_norm = (dy * 2.0) / max(H2 - 1, 1)
		base_grid = self._make_base_grid(B, H2, W2, x.device, x.dtype)
		grid = torch.stack((base_grid[..., 0] + dx_norm, base_grid[..., 1] + dy_norm), dim=-1)
		# sample
		y_deform = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

		if exists_mamba and exists_einops:
			t = rearrange(y_deform, 'b c (nh w1) (nw w2) -> b (nh nw) (w1 w2) c', w1=w, w2=w)
			t = self.norm(t)
			y = self.mamba(t)
			y = rearrange(y, 'b (nh nw) (w1 w2 c1) -> b c1 (nh w1) (nw w2)', nh=H2//w, nw=W2//w, w1=w, w2=w, c1=1)
		else:
			y = self.fallback(y_deform)
			y = self.ffn(y)

		if pad_h or pad_w:
			y = y[:, :, :H, :W]
		y = self.proj_out(y)
		return y + res


class SS2DWrapper2D(nn.Module):
	"""A thin wrapper that applies repo's SS2D on 2D features with in/out channel = C.

	Note: Requires selective_scan CUDA extension compiled. If unavailable, prefer LocalMamba2D.
	"""

	def __init__(self, channels: int):
		super().__init__()
		assert exists_ss2d, "SS2D is not available in this environment"
		# Use defaults similar to VSS/XSS blocks
		self.ss2d = SS2D(
			d_model=channels,
			d_state=16,
			ssm_ratio=2.0,
			ssm_rank_ratio=2.0,
			dt_rank="auto",
			act_layer=nn.SiLU,
			d_conv=3,
			conv_bias=True,
			dropout=0.0,
			forward_type="v2",
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.ss2d(x)


class MSFLMamba(nn.Module):
	"""Multi-Scale Frequency-domain Local Mamba block for YOLO neck/backbone insertion.

	Args (for YAML):
		in_channels (int): auto-filled by Ultralytics parser
		use_fft (bool): whether to enable torch FFT spectral gating
		window (int): local window size for Mamba
		expand (int): expansion ratio inside LocalMamba2D
		backend (str): 'mamba' (default) or 'ss2d' to use repo's SS2D
	"""

	def __init__(self, in_channels: int, use_fft: bool = True, window: int = 8, expand: int = 2, backend: str = "mamba"):
		super().__init__()
		self.pre_norm = nn.BatchNorm2d(in_channels)
		self.spectral = SpectralGating(in_channels, use_fft=use_fft)
		self.backend = (backend or "mamba").lower()
		if self.backend == "ss2d" and exists_ss2d:
			self.local = SS2DWrapper2D(in_channels)
		elif self.backend == "deform":
			self.local = LocalDeformableMamba2D(in_channels, window=window, expand=expand)
		else:
			self.local = LocalMamba2D(in_channels, window=window, expand=expand)
		self.post = nn.Conv2d(in_channels, in_channels, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		res = x
		x = self.pre_norm(x)
		x = self.spectral(x)
		x = self.local(x)
		x = self.post(x)
		return x + res


__all__ = ["MSFLMamba"]
