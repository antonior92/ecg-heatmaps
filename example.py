import ecg_plot
import argparse
import matplotlib.pyplot as plt
import preprocess
import os
import read_ecg
import numpy as np
import torch
import torch.nn as nn
from generate_heatmap import GradCAM, heatmap

class ModelBaseline(nn.Module):
    def __init__(self ,):
        super(ModelBaseline, self).__init__()
        self.kernel_size = 17

        # conv layer
        downsample = self._downsample(4096, 1024)
        self.conv1 = nn.Conv1d(in_channels=8,
                               out_channels=16,
                               kernel_size=self.kernel_size,
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)
        downsample = self._downsample(1024, 256)
        self.conv2 = nn.Conv1d(in_channels=16,
                               out_channels=32,
                               kernel_size=self.kernel_size,
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)
        downsample = self._downsample(256, 32)
        self.conv3 = nn.Conv1d(in_channels=32,
                               out_channels=64,
                               kernel_size=self.kernel_size,
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)

        # linear layer
        self.lin = nn.Linear(in_features=32 * 64,
                             out_features=1)

        # ReLU
        self.relu = nn.ReLU()

    def _padding(self, downsample):
        return max(0, int(np.floor((self.kernel_size - downsample + 1) / 2)))

    def _downsample(self, seq_len_in, seq_len_out):
        return int(seq_len_in // seq_len_out)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x_flat= x.view (x.size(0), -1)
        x = self.lin(x_flat)

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ECG from wfdb')

    parser.add_argument('path_to_ecg', type=str,
                        help='Path to the file to be plot.')
    parser.add_argument('path_to_model', type=str,
                        help='Path to model weights.')
    parser.add_argument('--save', default="",
                        help='Save in the provided path. Otherwise just display image.')
    parser = preprocess.arg_parse_option(parser)
    parser = read_ecg.arg_parse_option(parser)
    args = parser.parse_args()
    print(args)

    row_height = 4
    cols = 1

    ecg, sample_rate, leads = read_ecg.read_ecg(args.path_to_ecg, format=args.fmt)
    ecg, sample_rate, leads = preprocess.preprocess_ecg(ecg, sample_rate, leads,
                                                        new_freq=400,
                                                        new_len=4096,
                                                        scale=args.scale,
                                                        powerline=60,
                                                        use_all_leads=False,
                                                        remove_baseline=True)


    model = ModelBaseline()
    ckpt = torch.load(args.path_to_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])
    model = model.eval()

    x = torch.Tensor(ecg)[None, :, :]
    y_predicted = model(x)

    probs = torch.sigmoid(y_predicted).cpu().detach().numpy().flatten()
    print('Prob(1) = {:0.3f}'.format(*probs))

    grad_cam_model = GradCAM(model)
    _ = grad_cam_model(x)  # not using the result, but need it for initialisation

    x_viz = grad_cam_model.generate('conv1')
    x_viz = x_viz.detach().cpu().data.numpy()

    ecg_plot.plot(ecg, sample_rate=sample_rate,
                  lead_index=leads, style='bw',
                  row_height=row_height, columns=cols)
    heatmap(ecg, x_viz, sample_rate=sample_rate, columns=cols, scale=2, row_height=row_height)
    # rm ticks
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off

    if args.save:
        path, ext = os.path.splitext(args.save)
        if ext == '.png':
            ecg_plot.save_as_png(path)
        elif ext == '.pdf':
            ecg_plot.save_as_pdf(path)
    else:
        ecg_plot.show()