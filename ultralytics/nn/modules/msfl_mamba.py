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


class MSFLMamba(nn.Module):
	"""Multi-Scale Frequency-domain Local Mamba block for YOLO neck/backbone insertion.

	Args (for YAML):
		in_channels (int): auto-filled by Ultralytics parser
		use_fft (bool): whether to enable torch FFT spectral gating
		window (int): local window size for Mamba
		expand (int): expansion ratio inside LocalMamba2D
	"""

	def __init__(self, in_channels: int, use_fft: bool = True, window: int = 8, expand: int = 2):
		super().__init__()
		self.pre_norm = nn.BatchNorm2d(in_channels)
		self.spectral = SpectralGating(in_channels, use_fft=use_fft)
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
