import math
from typing import Dict, Tuple

import h5py
import hdf5plugin
from numba import jit
import numpy as np

import argparse
from pathlib import Path
import os
import torch
import cv2


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000, for ms > 0
        # (2) t[ms_to_idx[ms] - 1] < ms*1000, for ms > 0
        # (3) ms_to_idx[0] == 0
        # , where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]
    

class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            value = 2*pol-1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


def events_to_voxel_grid(dvs, num_bins, height, width):
    x, y, p, t = dvs[0], dvs[1], dvs[2], dvs[3]
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    p = p.astype('float32')
    voxel_grid = VoxelGrid(num_bins, height, width, normalize=False)
    return voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(p),
                torch.from_numpy(t))



def events_visualize(dvs):
    dvs_sum = dvs.numpy().sum(axis=0)
    dvs_positive = dvs_sum.clip(min=0)
    dvs_negative = (- dvs_sum).clip(min=0)
    max_value = max(dvs_positive.max(), dvs_negative.max())
    dvs_visualized = np.zeros((dvs_sum.shape[0], dvs_sum.shape[1], 3))
    if max_value > 0:
        dvs_visualized[:, :, 0] = (dvs_negative / max_value) * 63
        dvs_visualized[:, :, 2] = (dvs_positive / max_value) * 63
        dvs_visualized[dvs_visualized > 0] += 192
    return dvs_visualized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqpath', type=str, default='/path/to/DSEC/DSEC_train/zurich_city_00_a', help='Path to DSEC Dataset')
    parser.add_argument('--outpath', type=str, default='./data/DSEC/zurich_city_00_a', help='Path to Output Folder')

    args = parser.parse_args()

    seqpath = Path(args.seqpath)
    outpath = Path(args.outpath)
    assert seqpath.is_dir()
    print(f'start processing: {seqpath}')

    events_left_dir = seqpath / 'events' / 'left'
    outdir = outpath / 'DVS' / 'NORM' / 'BIN5'
    visdir = outpath / 'DVS' / 'NORM' / 'VIZ'
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(visdir, exist_ok=True)

    ev_data_file = events_left_dir / 'events.h5'
    ev_rect_file = events_left_dir / 'rectify_map.h5'
    timestamp_file = seqpath / 'images' / 'timestamps.txt'

    ev_data = h5py.File(str(ev_data_file), 'r')
    event_slicer = EventSlicer(ev_data)
    ev_rect = h5py.File(str(ev_rect_file), 'r')['rectify_map'][()]

    timestamps = np.loadtxt(timestamp_file, dtype='int64')
    last_timestamp = timestamps[0]

    for i in range(1, len(timestamps)):
        event_data = event_slicer.get_events(last_timestamp, timestamps[i])

        # Filter Event Number
        n = 50000
        p = event_data['p'][-n:]
        t = event_data['t'][-n:]
        x = event_data['x'][-n:]
        y = event_data['y'][-n:]

        xy_rect = ev_rect[y, x]
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        events_representation = events_to_voxel_grid((x_rect, y_rect, p, t), 5, 480, 640)
        events_visualization = events_visualize(events_representation)

        last_timestamp = timestamps[i]

        np.save(str(outdir / '{:08d}.npy'.format(i)), events_representation[:, :440])
        cv2.imwrite(str(visdir / '{:08d}.png'.format(i)), events_visualization[:440])

    print(f'done processing: {seqpath}')