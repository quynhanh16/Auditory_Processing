from typing import List

from nems.tools.signal import RasterizedSignal

from tools import load_state, save_state, load_datafile, splitting_recording


def evoked_firing_rate(signal: RasterizedSignal, cells: str | List[str]) -> float | list[float]:
    pass


if __name__ == '__main__':
    tgz_file: str = 'A1_NAT4_ozgf.fs100.ch18.tgz'

    state_file = "state.pkl"
    state = load_state(state_file)
    if state is None:
        rec = load_datafile(tgz_file, True)
        stim, resp = splitting_recording(rec, True)
        save_state(state_file, stim, resp)
    else:
        stim, resp = load_state(state_file)

    all_cellids = resp.chans
