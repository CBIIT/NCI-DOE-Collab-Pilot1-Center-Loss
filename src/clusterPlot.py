# does PCA down to three dimensions and K-means and plots stuff using colors.
# does tNSE plot too using sklearn

import argparse
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

class GeomViewData:
    def __init__(self, points, colors):
        scaler = MinMaxScaler()
        self.scaled = 100 * scaler.fit_transform(points)
        self.colors = colors

    def write(self, filename):
        with open(filename, 'w') as out:
            for i, p in enumerate(self.scaled):
                line = ' '.join(map(str,p))+' '+' '.join(map(str, self.colors[i][:3])) + '\n'
                out.write(line)

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_features', type=str, metavar='N', nargs='*', help='features, tab delimited')
    parser.add_argument('input_labels', type=str, help='labels')
    parser.add_argument('output', type=str, help='outputs a plot as a png or as a txt file from geomview')

    return parser.parse_args()

class ClusterPlot:
    def __init__(self, input_features, input_labels):
        features = np.load(input_features)
        self.labels = np.loadtxt(input_labels).astype(np.float)
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.labels)

        # if there're too many samples we'll just subsample them.
        max_num_sample = 2000
        if self.labels.shape[0] > max_num_sample:
            subsample_index = np.random.choice(self.labels.shape[0], max_num_sample)
            self.labels = self.labels[subsample_index]
            features = features[subsample_index]

        pca = PCA(n_components=3)
        self.transformed = pca.fit_transform(features)

        print('explained variance')
        print(pca.explained_variance_ratio_)

        print(input_labels)
        print('#labels', self.labels.shape, '#transformed', self.transformed.shape)

        # generate colors
        self.alpha = .6
        norm = np.linspace(0.0, 1.0, len(self.label_encoder.classes_))
        #self.color_map = mpl.cm.ScalarMappable(norm)
        #self.colors = [list(self.color_map.to_rgba(label, alpha=self.alpha)) for label in self.labels]

        self.color_map = plt.get_cmap("nipy_spectral")
        self.colors = self.color_map(norm)
        self.colors[:,3] = self.alpha

        self.plot_type = 'pca'
        self.title = os.path.basename(os.path.dirname(input_features))

    def drawPlot(self, output_png):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        self.add_plot(ax)

        fig.savefig(output_png)

    def add_plot(self, ax):
        ax.title.set_text(self.title)
        proxies = []
        cluster_names = []
        for i, label in enumerate(self.label_encoder.classes_):
            # make a fake plot for each class to generate a legend
            # https://stackoverflow.com/questions/20505105/add-a-legend-in-a-3d-scatterplot-with-scatter-in-matplotlib
            color = self.colors[self.label_encoder.transform([label])[0]]
            proxies.append(mpl.lines.Line2D([0],[0],linestyle='none',c=color, marker='o'))
            cluster_names.append("Cluster "+str(label))

            filtered = self.transformed[self.labels == label]
            ax.scatter(filtered[:,0], filtered[:,1], filtered[:,2], c=color, edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        plot_type = self.plot_type
        ax.set_xlabel(plot_type+'1')
        ax.set_ylabel(plot_type+'2')
        ax.set_zlabel(plot_type+'3')

        # shrink plots height by 10% to fit in legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                box.width, box.height * 0.9])

        #plt.legend(proxies, cluster_names, numpoints = 1,
        #    loc='upper center', bbox_to_anchor=(.5, -0.05), ncol=3, fontsize=8)

    def writeTxt(self, output_txt):
        gvd = GeomViewData(self.transformed, self.colors)
        gvd.write(output_txt)
        return

class TSNEPlot(ClusterPlot):
    def __init__(self, input_features, input_labels, perplexity):
        features = np.load(input_features)
        self.labels = np.loadtxt(input_labels).astype(np.float)
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.labels)

        # if there're too many samples we'll just subsample them.
        max_num_sample = 2000
        if self.labels.shape[0] > max_num_sample:
            subsample_index = np.random.choice(self.labels.shape[0], max_num_sample)
            self.labels = self.labels[subsample_index]
            features = features[subsample_index]

        tSNE = TSNE(n_components=3, perplexity=perplexity)
        self.transformed = tSNE.fit_transform(features)

        # generate colors
        self.alpha = .6
        norm = np.linspace(0.0, 1.0, len(self.label_encoder.classes_))
        #self.color_map = mpl.cm.ScalarMappable(norm)
        #self.colors = [list(self.color_map.to_rgba(label, alpha=self.alpha)) for label in self.labels]

        self.color_map = plt.get_cmap("nipy_spectral")
        self.colors = self.color_map(norm)
        self.colors[:,3] = self.alpha

        self.plot_type = 'tsne'
        self.title = os.path.dirname(input_features)+' perplexity:'+str(perplexity)
        self.title = os.path.basename(self.title)

def write_all_images(args):
    # this opens several output plots and joins them together
    # using PIL
    for input_feature in args.input_features:
        png_names = []
        abs_path = os.path.abspath(input_feature)
        output_name = os.path.splitext(os.path.basename(abs_path))[0]+args.output
        output_name = os.path.join(os.path.dirname(abs_path), output_name)

        cluster_plot = ClusterPlot(input_feature, args.input_labels)

        if output_name.endswith('.png'):
            cluster_plot.drawPlot(output_name)
            png_names.append(output_name)
        elif output_name.endswith('.txt'):
            cluster_plot.writeTxt(output_name)
        else:
            print("unrecognized output format. expecting .txt or .png. Instead got", output_name)

        perplexities = [30, 50, 100]

        for p in perplexities:
            tsne_name = os.path.splitext(os.path.basename(abs_path))[0]+'_tsne'+str(p)+args.output
            tsne_name = os.path.join(os.path.dirname(abs_path), tsne_name)

            tsne_model = TSNEPlot(input_feature, args.input_labels, p)
            if tsne_name.endswith('.png'):
                tsne_model.drawPlot(tsne_name)
                png_names.append(tsne_name)
            elif tsne_name.endswith('.txt'):
                tsne_model.writeTxt(tsne_name)
            else:
                print("unrecognized output format. expecting .txt or .png. Instead got", tsne_name)

        # join all pngs into one image
        if len(png_names) > 0:
            images = [Image.open(i) for i in png_names]
            total_width = sum([im.size[0] for im in images])
            max_height = max([im.size[1] for im in images])

            joined_im = Image.new('RGB', (total_width, max_height))

            offset = 0
            for im in images:
                joined_im.paste(im, (offset, 0))
                offset += im.size[0]

            joined_im.save(output_name.replace(args.output, '.joined.png'))

def projection_plots(args):
    for input_feature in args.input_features:
        abs_path = os.path.abspath(input_feature)
        output_name = os.path.splitext(os.path.basename(abs_path))[0]+'.joined.png'
        output_name = os.path.join(os.path.dirname(abs_path), output_name)

        cluster_plot = ClusterPlot(input_feature, args.input_labels)

        perplexities = [30, 50, 100]
        num_plots = len(perplexities)+1 # one plot of pca+one for each perplexity

        fig = plt.figure(figsize=(30, 6), dpi=80)
        ax = fig.add_subplot(1, num_plots, 1, projection='3d')
        cluster_plot.add_plot(ax)

        for i, p in enumerate(perplexities):
            tsne_model = TSNEPlot(input_feature, args.input_labels, p)
            tsne_model.add_plot(fig.add_subplot(1, num_plots, i+2, projection='3d'))

        fig.subplots_adjust(wspace=0)

        plt.savefig(output_name)
        plt.close(fig)

if __name__ == "__main__":
    args = parseArgs()

    projection_plots(args)
