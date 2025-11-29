import torch
from torch import nn
import pywt
from typing import Sequence, Tuple, Union, List
from einops import rearrange, repeat
import torch.nn.functional as F

def _as_wavelet(wavelet):
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        Wavelet: the input wavelet object or the pywt wavelet object described by the
            input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul

def construct_2d_filt(lo, hi) -> torch.Tensor:
    """Construct two dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        torch.Tensor: Stacked 2d filters of dimension
            [filt_no, 1, height, width].
            The four filters are ordered ll, lh, hl, hh.
    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    # filt = filt.unsqueeze(1)
    return filt


def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
        flip (bool): If true filters are flipped.
        device (torch.device) : PyTorch target device.
        dtype (torch.dtype): The data type sets the precision of the
               computation. Default: torch.float32.

    Returns:
        tuple: Tuple containing the four filter tensors
        dec_lo, dec_hi, rec_lo, rec_hi

    """
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    """Compute the required padding.

    Args:
        data_len (int): The length of the input vector.
        filt_len (int): The length of the used filter.

    Returns:
        tuple: The numbers to attach on the edges of the input.

    """
    # pad to ensure we see all filter positions and for pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    """Pad data for the 2d FWT.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode (str): The padding mode.
            Supported modes are "reflect", "zero", "constant" and "periodic".
            Defaults to reflect.

    Returns:
        The padded output tensor.

    """

    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))

    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad


class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi

        # # initial dec conv
        # self.conv = torch.nn.Conv2d(c1, c2 * 4, kernel_size=dec_filt.shape[-2:], groups=c1, stride=2)
        # self.conv.weight.data = dec_filt
        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]


class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        # self.convT = nn.ConvTranspose2d(c2 * 4, c1, kernel_size=weight.shape[-2:], groups=c1, stride=2)
        # self.convT.weight = torch.nn.Parameter(rec_filt)
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None:  # soft orthogonal
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:  # hard orthogonal
            idwt_kernel= torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
                # ll, lh, hl, hl, hh
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            # cat is not work for the strange transpose
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

            # remove the padding
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), "padding error, please open an issue on github "
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), "padding error, please open an issue on github "
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component





